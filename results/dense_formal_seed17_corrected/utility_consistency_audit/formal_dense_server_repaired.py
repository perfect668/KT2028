#!/usr/bin/env python3
"""Formal single-seed Dense PPO training on all Train patients."""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from MSRLSelect.rl_core.assets import FixedE2Ensemble, load_train_assets
from MSRLSelect.rl_core.core import DenseActorCritic, SequentialAcquireEnv, collect_rollout, ppo_update, evidence_potential


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def build_histories(n, seed):
    rng = np.random.default_rng(seed)
    states = []
    for patient in range(n):
        states.append((patient, np.zeros(64, dtype=bool), np.zeros(64, dtype=np.int64), 0))
        for k in (1, 2, 4, 8):
            perm = rng.permutation(64)
            for _ in range(1 if seed == 1701 else 1):
                selected = perm[:k]
                active = np.zeros(64, dtype=bool)
                order = np.zeros(64, dtype=np.int64)
                for step, region in enumerate(selected):
                    active[region] = True
                    order[region] = step + 1
                states.append((patient, active, order, k))
    return states


@torch.no_grad()
def action_gains(env, states, device):
    ids = torch.tensor([s[0] for s in states], dtype=torch.long, device=device)
    active = torch.tensor(np.stack([s[1] for s in states]), dtype=torch.bool, device=device)
    labels = torch.tensor(env.assets.metadata.label.values[ids.cpu().numpy()], dtype=torch.float64, device=device)
    current_p, _, current_d, _ = env.predictor.predict(ids, active)
    current_phi = evidence_potential(labels, current_p, current_d)
    q = torch.full((len(states), 64), float("nan"), dtype=torch.float64, device=device)
    for start in range(0, len(states), 64):
        end = min(start + 64, len(states))
        b = end - start
        candidate = active[start:end, None, :].expand(-1, 64, -1).clone()
        candidate |= torch.eye(64, dtype=torch.bool, device=device)[None]
        valid = ~active[start:end]
        c = env.assets.coarse[ids[start:end].cpu()].to(device)[:, None].expand(-1, 64, -1, -1).reshape(-1, 64, env.assets.coarse.shape[-1])
        s = env.assets.fine_slots[ids[start:end].cpu()].to(device).flatten(-2)[:, None].expand(-1, 64, -1, -1).reshape(-1, 64, env.assets.fine_slots.shape[-1] * env.assets.fine_slots.shape[-2])
        nxt, _, nd, _ = env.predictor.predict(ids[start:end, None].expand(-1, 64).reshape(-1), candidate.reshape(-1, 64))
        yy = labels[start:end, None].expand(-1, 64).reshape(-1)
        gain = (evidence_potential(yy, nxt, nd) - current_phi[start:end, None].expand(-1, 64).reshape(-1)).reshape(b, 64)
        q[start:end] = gain.double().masked_fill(~valid, float("nan"))
    return q, active


def warmstart(policy, env, states, q, device, lambda_values, output):
    ids = torch.tensor([s[0] for s in states], dtype=torch.long, device=device)
    active = torch.tensor(np.stack([s[1] for s in states]), dtype=torch.bool, device=device)
    order = torch.tensor(np.stack([s[2] for s in states]), dtype=torch.long, device=device)
    coarse = env.assets.coarse[ids.cpu()].to(device)
    slots = env.assets.fine_slots[ids.cpu()].to(device)
    labels = env.assets.metadata.label.values[ids.cpu().numpy()]
    with torch.no_grad():
        risk, entropy, disagreement, _ = env.predictor.predict(ids, active)
    cumulative_cost = (env.costs[ids.cpu()].to(device) * active).sum(1).float()
    rows = []
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-4)
    for epoch in range(1, 21):
        permutation = torch.randperm(len(states), device=device)
        losses = []
        for start in range(0, len(states), 256):
            ix = permutation[start:start + 256]
            obs = {"patient_index": ids[ix], "coarse": coarse[ix], "fine_slots": slots[ix] * active[ix, :, None, None].to(slots.dtype),
                   "active_mask": active[ix], "acquire_order": order[ix], "risk": risk[ix], "entropy": entropy[ix],
                   "disagreement": disagreement[ix], "step": active[ix].sum(1), "cumulative_cost": cumulative_cost[ix],
                   "lambda": torch.zeros(len(ix), device=device), "remaining": (~active[ix]).sum(1)}
            logits, _ = policy(obs)
            losses_lambda = []
            for lam in lambda_values:
                net_values = q[ix] - lam * env.costs[ids[ix].cpu()].to(device)
                target = torch.nan_to_num(net_values, nan=-1e9).argmax(1)
                losses_lambda.append(F.cross_entropy(logits[:, :64], target))
            loss = torch.stack(losses_lambda).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), .5); optimizer.step(); losses.append(float(loss.detach()))
        rows.append({"epoch": epoch, "loss": float(np.mean(losses)), "states": len(states), "objective": "ACQUIRE-only warm-start"})
    with torch.no_grad():
        best_raw = torch.where(torch.isfinite(q), q, torch.full_like(q, -float("inf"))).amax(1)
    rows.append({"epoch": 0, "loss": 0.0, "states": len(states), "objective": "summary", "best_raw_gain_mean": float(best_raw.mean())})
    pd.DataFrame(rows).to_csv(output / "warmstart_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(17); np.random.seed(17)
    out = Path(args.output_dir); [((out / x).mkdir(parents=True, exist_ok=True)) for x in ("checkpoints", "logs")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assets = load_train_assets(args.data_root)
    predictor = FixedE2Ensemble(assets, device)
    env = SequentialAcquireEnv(assets, predictor, device, max_steps=8)
    states = build_histories(len(assets.metadata), 1701) + build_histories(len(assets.metadata), 1702)[len(assets.metadata):]
    if len(states) != 2295: raise AssertionError(len(states))
    q, active = action_gains(env, states, device)
    positive = q[torch.isfinite(q) & (q > 0)].detach().cpu().numpy()
    lambda_values = [0.0] + [float(np.quantile(positive, x)) for x in (.25, .5, .75)]
    (out / "lambda_values.json").write_text(json.dumps({"values": lambda_values, "positive_gain_count": int(len(positive)), "source": "Train OOF legal DeltaPhi > 0", "source_sha256": digest(args.data_root + "/outputs/thesis_evidence_interface_v7/checkpoints/E2_oof_fold0_seed17_best.pt")}, indent=2))
    policy = DenseActorCritic(assets.coarse.shape[-1]).to(device)
    warmstart(policy, env, states, q, device, lambda_values, out)
    config = {"seed": 17, "patients": len(assets.metadata), "max_acquire_steps": 8, "target_transitions": 200000, "rollout_minimum": 4096, "ppo_epochs": 4, "minibatch_size": 512, "clip": .2, "lr": 5e-5, "gamma": 1.0, "gae_lambda": .95, "device": str(device), "no_validation_during_training": True, "no_test": True}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5)
    all_logs = []; diagnostics = []; transitions = 0; round_id = 0; rng = np.random.default_rng(17)
    while transitions < 200000:
        round_id += 1; patient_order = rng.permutation(len(assets.metadata)); lam = lambda_values[(round_id - 1) % 4]; started = time.time()
        batch = collect_rollout(policy, env, patient_order, lam, minimum_transitions=4096)
        updates, initial_ratio = ppo_update(policy, optimizer, batch, device, epochs=4, minibatch_size=512, clip=.2)
        if abs(initial_ratio - 1) > 1e-5: raise RuntimeError("initial PPO ratio check failed")
        max_kl = max(abs(float(x["approx_kl"])) for x in updates)
        if max_kl > .2: raise RuntimeError(f"approx KL exceeded 0.2: {max_kl}")
        transitions += len(batch)
        for row in updates: row.update({"rollout": round_id, "transitions": transitions, "lambda": lam, "initial_ratio": initial_ratio, "elapsed_seconds": time.time() - started, "max_approx_kl_rollout": max_kl})
        all_logs.extend(updates)
        diagnostics.append({"rollout": round_id, "transitions": transitions, "lambda": lam, "rollout_transitions": len(batch), "mean_reward": float(np.mean([x["reward"] for x in batch])), "positive_reward_fraction": float(np.mean([x["reward"] > 0 for x in batch])), "mean_episode_return": float(np.mean([sum(x["reward"] for x in batch if x["episode_id"] == eid) for eid in {x["episode_id"] for x in batch}])), "max_approx_kl": max_kl, "clip_fraction": float(np.mean([x["clip_fraction"] for x in updates])), "mean_acquisitions": float(np.mean([x["step"] + (x["action"] < 64) for x in batch])), "immediate_stop_fraction": float(np.mean([x["action"] == 64 and x["step"] == 0 for x in batch])), "reached_8_fraction": float(np.mean([x["step"] >= 7 for x in batch]))})
        if transitions % 20000 < len(batch) or transitions >= 200000:
            torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "seed": 17, "transitions": transitions, "lambda_values": lambda_values, "config": config, "fixed_e2_checkpoint_sha256": [digest(p) for p in assets.checkpoint_paths], "calibration_sha256": assets.calibration_sha256, "warmstart_sha256": digest(out / "warmstart_metrics.csv")}, out / "checkpoints" / f"checkpoint_{transitions}.pt")
    pd.DataFrame(all_logs).to_csv(out / "training_log.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "train_policy_diagnostics.csv", index=False)
    final = out / "checkpoints" / "final_dense_seed17.pt"; torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "seed": 17, "transitions": transitions, "lambda_values": lambda_values, "config": config, "fixed_e2_checkpoint_sha256": [digest(p) for p in assets.checkpoint_paths], "calibration_sha256": assets.calibration_sha256, "warmstart_sha256": digest(out / "warmstart_metrics.csv")}, final)
    loaded = DenseActorCritic(assets.coarse.shape[-1]).to(device); loaded.load_state_dict(torch.load(final, map_location=device, weights_only=False)["model"]); loaded.eval(); policy.eval()
    obs = env.reset(np.arange(min(4, len(assets.metadata))), lambda_values[0]);
    with torch.no_grad(): a = policy(obs); b = loaded(obs)
    (out / "final_checkpoint_manifest.json").write_text(json.dumps({"path": str(final), "sha256": digest(final), "transitions": transitions, "reload_max_abs_output_delta": float(max((a[0] - b[0]).abs().max(), (a[1] - b[1]).abs().max())), "fixed_e2_checkpoint_sha256": [digest(p) for p in assets.checkpoint_paths], "calibration_sha256": assets.calibration_sha256}, indent=2))
    (out / "asset_manifest.json").write_text(json.dumps({"data_root": str(assets.root), "train_patients": len(assets.metadata), "fixed_e2_checkpoints": [str(p) for p in assets.checkpoint_paths], "calibration_path": str(assets.calibration_path), "calibration_sha256": assets.calibration_sha256, "validation_read": False, "test_read": False}, indent=2))
    summary = {"status": "TRAIN_COMPLETE", "transitions": transitions, "lambda_values": lambda_values, "checkpoint": str(final), "checkpoint_sha256": digest(final), "validation_run": False, "test_run": False}
    (out / "final_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
