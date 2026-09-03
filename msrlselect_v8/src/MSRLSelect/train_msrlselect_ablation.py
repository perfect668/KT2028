#!/usr/bin/env python3
"""Train corrected sequential MS-RLSelect variants for chapter 3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from tqdm import tqdm

from models.ms_model import MSRLSelectModel
from models.ms_policy import MSRegionPolicy
from models.ms_reward import compute_reward
from pipeline_common import read_config, seed_all
from thesis_selection_common import RegionSummaryDataset, ensure_summary_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("full", "no_spatial", "no_evidence", "revision"), required=True
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--policy-lr", type=float, default=2e-4)
    parser.add_argument("--budget", type=int, default=4)
    return parser.parse_args()


def choose_one(policy, base, spatial, available, deterministic: bool):
    policy_spatial = spatial if policy.training or spatial.abs().sum() else spatial
    logits, value = policy.logits_and_value(
        base["global_token"].detach(),
        base["cluster_tokens"].detach(),
        policy_spatial.detach(),
        available,
    )
    distribution = torch.distributions.Categorical(logits=logits)
    index = logits.argmax(dim=1) if deterministic else distribution.sample()
    action = F.one_hot(index, num_classes=available.shape[1]).bool()
    return action, distribution.log_prob(index), distribution.entropy(), value


def episode(model, policy, batch, variant: str, budget: int, train: bool):
    cluster_features, spatial, cluster_mask, labels, _ = batch
    device = next(model.parameters()).device
    cluster_features = cluster_features.to(device, non_blocking=True)
    spatial = spatial.to(device, non_blocking=True)
    cluster_mask = cluster_mask.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    policy_spatial = torch.zeros_like(spatial) if variant == "no_spatial" else spatial
    _, base = model.forward_aggregates(cluster_features, spatial, cluster_mask)
    selected = torch.zeros_like(cluster_mask)
    previous_confidence = torch.zeros(len(labels), device=device)
    model_losses, rewards, log_probs, entropies, values = [], [], [], [], []
    final_logits = None
    for _ in range(min(budget, cluster_mask.shape[1])):
        available = cluster_mask & ~selected
        if not bool(available.any(dim=1).all()):
            break
        action, log_prob, entropy, value = choose_one(
            policy, base, policy_spatial, available, deterministic=not train
        )
        selected |= action
        logits, auxiliary = model.forward_aggregates(
            cluster_features, spatial, cluster_mask, region_mask=selected
        )
        probabilities = logits.softmax(dim=1)
        # Labels are allowed only for the training reward.  Validation/test
        # action and stopping signals must be label-free.
        confidence = (probabilities.gather(1, labels[:, None]).squeeze(1)
                      if train else probabilities.max(dim=1).values).detach()
        reward_parts = compute_reward(confidence, previous_confidence,
                                      auxiliary["cluster_tokens"].detach(), selected,
                                      policy_spatial[..., :2].detach())
        if variant in ("revision", "no_evidence"):
            # Targeted revision: evidence gain minus redundancy and cost;
            # no diversity/coverage bonus.
            reward = (reward_parts["confidence_gain"]
                      - 0.10 * reward_parts["redundancy"]
                      - 0.01 * reward_parts["cost"])
        else:
            reward = reward_parts["reward"]
        model_losses.append(F.cross_entropy(logits, labels))
        rewards.append(reward.detach())
        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        previous_confidence = confidence
        final_logits = logits

    model_loss = torch.stack(model_losses).mean()
    if train:
        returns = []
        running = torch.zeros_like(rewards[0])
        for reward in reversed(rewards):
            running = reward + 0.1 * running
            returns.insert(0, running)
        returns_tensor = torch.stack(returns)
        values_tensor = torch.stack(values)
        advantage = returns_tensor - values_tensor
        policy_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        policy_loss += 0.5 * advantage.pow(2).mean()
        policy_loss -= 0.01 * torch.stack(entropies).mean()
    else:
        policy_loss = model_loss.new_zeros(())
    return final_logits, labels, model_loss, policy_loss, torch.stack(rewards).mean()


def run_epoch(model, policy, loader, optimizer, policy_optimizer, variant, budget, train):
    model.train(train)
    policy.train(train)
    logits_all, labels_all, losses, rewards = [], [], [], []
    iterator = tqdm(loader, desc="train" if train else "valid", leave=False)
    for batch in iterator:
        if train:
            optimizer.zero_grad(set_to_none=True)
            policy_optimizer.zero_grad(set_to_none=True)
        logits, labels, model_loss, policy_loss, reward = episode(
            model, policy, batch, variant, budget, train
        )
        if train:
            (model_loss + policy_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            policy_optimizer.step()
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        losses.append(float(model_loss.detach()))
        rewards.append(float(reward.detach()))
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    risks = logits.softmax(dim=1)[:, 1].numpy()
    return {
        "auc": float(roc_auc_score(labels, risks)),
        "auprc": float(average_precision_score(labels, risks)),
        "loss": float(np.mean(losses)),
        "reward": float(np.mean(rewards)),
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    seed_all(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = ensure_summary_cache(config, args.summary_root)
    datasets = {split: RegionSummaryDataset(path) for split, path in manifests.items()}
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=True,
        )
        for split, dataset in datasets.items()
    }
    use_spatial = args.variant != "no_spatial"
    model = MSRLSelectModel(
        token_dim=int(config["token_dim"]),
        num_layers=int(config["layers"]),
        num_heads=int(config["heads"]),
        use_spatial=use_spatial,
    ).cuda()
    policy = MSRegionPolicy(
        int(config["token_dim"]), int(config["token_dim"]), args.budget
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    policy_optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.policy_lr, weight_decay=1e-4
    )
    best_auc = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model,
            policy,
            loaders["train"],
            optimizer,
            policy_optimizer,
            args.variant,
            args.budget,
            True,
        )
        with torch.no_grad():
            valid_result = run_epoch(
                model,
                policy,
                loaders["valid"],
                optimizer,
                policy_optimizer,
                args.variant,
                args.budget,
                False,
            )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_result.items()},
            **{f"valid_{key}": value for key, value in valid_result.items()},
        }
        history.append(row)
        print(json.dumps(row))
        if valid_result["auc"] > best_auc:
            best_auc = valid_result["auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "variant": args.variant,
                    "model": model.state_dict(),
                    "policy": policy.state_dict(),
                    "config": config,
                    "budget": args.budget,
                    "valid": valid_result,
                },
                args.output_dir / "model_best.pt",
            )
    (args.output_dir / "metrics_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "variant": args.variant,
                "best_validation_auc": best_auc,
                "test_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE", "best_validation_auc": best_auc}))


if __name__ == "__main__":
    main()
