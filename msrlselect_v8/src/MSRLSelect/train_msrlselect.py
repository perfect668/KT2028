#!/usr/bin/env python3
"""Standalone MS-RLSelect training entry point.

The legacy HeMiCoRe script is intentionally left untouched.  This trainer
uses the frozen UNI features already prepared for SR386, jointly trains the
case-level predictor and context-aware region policy, and writes split-safe
metrics/checkpoints for the thesis experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.nn import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from models.ms_budget import AdaptiveBudgetController
from models.ms_model import MSRLSelectModel
from models.ms_policy import MSRegionPolicy
from models.ms_reward import compute_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MS-RLSelect on prepared UNI features")
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--valid-manifest", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--token-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--min-steps", type=int, default=2)
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--policy-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="use CUDA bfloat16 autocast")
    return p.parse_args()


class CaseDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path):
        self.frame = pd.read_csv(manifest)
        required = {"features_filepath", "label", "clusters_json_filepath"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{manifest} missing columns: {sorted(missing)}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with np.load(row.features_filepath) as z:
            features = z["img_features"].astype(np.float32, copy=True)
            coords = z["coords"].astype(np.float32, copy=True)
        clusters = json.loads(Path(row.clusters_json_filepath).read_text(encoding="utf-8"))
        return {"features": features, "coords": coords, "clusters": clusters,
                "label": int(row.label), "case_id": str(row.get("case_id", index))}


def collate_cases(items: list[dict[str, Any]]):
    max_patches = max(x["features"].shape[0] for x in items)
    dim = items[0]["features"].shape[1]
    features = np.zeros((len(items), max_patches, dim), dtype=np.float32)
    coords = np.zeros((len(items), max_patches, 2), dtype=np.float32)
    for i, item in enumerate(items):
        n = item["features"].shape[0]
        features[i, :n] = item["features"]
        coords[i, :n] = item["coords"]
    return (torch.from_numpy(features), torch.from_numpy(coords),
            [x["clusters"] for x in items],
            torch.tensor([x["label"] for x in items], dtype=torch.long),
            [x["case_id"] for x in items])


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = logits.softmax(1)[:, 1].detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    pred = (probs >= 0.5).astype(np.int64)
    out = {"f1": float(f1_score(y, pred, zero_division=0)),
           "auprc": float(average_precision_score(y, probs))}
    out["auc"] = float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else float("nan")
    out["accuracy"] = float((pred == y).mean())
    return out


def move_batch(batch, device):
    features, coords, clusters, labels, case_ids = batch
    return features.to(device, non_blocking=True), coords.to(device, non_blocking=True), clusters, labels.to(device), case_ids


def episode(model, policy, controller, features, coords, clusters, labels, train: bool, max_steps: int):
    # The encoder is evaluated once per batch; each policy step changes only
    # the region mask, making the RL trace inexpensive with frozen UNI input.
    with torch.set_grad_enabled(train):
        _, base = model(features, coords, clusters)
        valid = base["cluster_mask"]
        previous_confidence = torch.zeros(features.shape[0], device=features.device)
        gains, rewards, logps, values, entropies, losses = [], [], [], [], [], []
        final_logits = None
        for step in range(max_steps):
            action, logp, entropy, value = policy.select(
                base["global_token"].detach(), base["cluster_tokens"].detach(),
                base["spatial_features"].detach(), valid, deterministic=not train)
            logits, aux = model(features, coords, clusters, region_mask=action)
            probabilities = logits.softmax(1)
            # Labels are used for the learning reward only.  Validation/test
            # stopping must be label-free, otherwise the adaptive budget
            # would leak the ground truth into evaluation.
            confidence = (probabilities.gather(1, labels[:, None]).squeeze(1)
                          if train else probabilities.max(1).values).detach()
            reward_parts = compute_reward(
                confidence, previous_confidence, aux["cluster_tokens"].detach(), action,
                aux["spatial_features"][..., :2].detach())
            reward = reward_parts["reward"]
            gains.append(reward_parts["confidence_gain"])
            rewards.append(reward)
            logps.append(logp)
            values.append(value)
            entropies.append(entropy)
            losses.append(F.cross_entropy(logits, labels))
            previous_confidence = confidence
            final_logits = logits
            gain_stack, reward_stack = torch.stack(gains), torch.stack(rewards)
            # In a minibatch we stop the episode once every case satisfies the
            # patience rule; this keeps tensor shapes uniform while still
            # applying adaptive budget control during training and evaluation.
            if bool(controller.should_stop(step + 1, gain_stack, reward_stack).all()):
                break
        model_loss = torch.stack(losses).mean()
        reward_tensor = torch.stack(rewards)
        if train:
            # Normalize within the episode to avoid policy-gradient scale
            # depending on the number of selected regions.
            advantage = reward_tensor.detach() - torch.stack(values)
            policy_loss = -(torch.stack(logps) * reward_tensor.detach()).mean()
            policy_loss += 0.5 * advantage.pow(2).mean() - 0.01 * torch.stack(entropies).mean()
        else:
            policy_loss = model_loss.new_zeros(())
    return final_logits, model_loss, policy_loss, {"reward": reward_tensor.mean().item(), "steps": len(losses)}


def run_epoch(model, policy, loader, device, optimizer, policy_optimizer, args, train: bool):
    model.train(train); policy.train(train)
    controller = AdaptiveBudgetController(args.min_steps, args.max_steps)
    all_logits, all_labels = [], []
    total_loss = total_reward = 0.0
    iterator = tqdm(loader, desc="train" if train else "valid", leave=False)
    for batch in iterator:
        features, coords, clusters, labels, _ = move_batch(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True); policy_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            logits, model_loss, policy_loss, info = episode(model, policy, controller, features, coords, clusters, labels, train, args.max_steps)
        if train:
            (model_loss + policy_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step(); policy_optimizer.step()
        all_logits.append(logits.detach().cpu()); all_labels.append(labels.detach().cpu())
        total_loss += float(model_loss.detach()); total_reward += info["reward"]
        iterator.set_postfix(loss=f"{float(model_loss.detach()):.3f}", reward=f"{info['reward']:.3f}", steps=info["steps"])
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    result = metrics(logits, labels)
    result.update({"loss": total_loss / max(1, len(loader)), "reward": total_reward / max(1, len(loader))})
    return result


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = torch.utils.data.DataLoader(CaseDataset(args.train_manifest), args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_cases)
    valid_loader = torch.utils.data.DataLoader(CaseDataset(args.valid_manifest), args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_cases)
    test_loader = torch.utils.data.DataLoader(CaseDataset(args.test_manifest), args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_cases)
    model = MSRLSelectModel(token_dim=args.token_dim, num_layers=args.layers, num_heads=args.heads).to(device)
    policy = MSRegionPolicy(args.token_dim, args.token_dim, max_budget=args.budget).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=args.policy_lr, weight_decay=args.weight_decay)
    best_auc = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, policy, train_loader, device, optimizer, policy_optimizer, args, True)
        with torch.no_grad():
            valid_result = run_epoch(model, policy, valid_loader, device, optimizer, policy_optimizer, args, False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_result.items()},
               **{f"valid_{k}": v for k, v in valid_result.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        score = valid_result["auc"] if np.isfinite(valid_result["auc"]) else -valid_result["loss"]
        if score > best_auc:
            best_auc = score
            torch.save({"epoch": epoch, "model": model.state_dict(), "policy": policy.state_dict(),
                        "args": vars(args), "valid": valid_result}, args.output_dir / "model_best.pt")
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
    checkpoint = torch.load(args.output_dir / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]); policy.load_state_dict(checkpoint["policy"])
    with torch.no_grad():
        test_result = run_epoch(model, policy, test_loader, device, optimizer, policy_optimizer, args, False)
    (args.output_dir / "test_result.json").write_text(json.dumps(test_result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "best_valid_auc": best_auc, "test": test_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
