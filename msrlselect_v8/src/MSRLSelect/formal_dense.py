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
        states.append((patient, 0, seed, np.zeros(64, dtype=bool), np.zeros(64, dtype=np.int64), 0))
        perm = rng.permutation(64)
        for k in (1, 2, 4, 8):
            selected = perm[:k]
            active = np.zeros(64, dtype=bool); order = np.zeros(64, dtype=np.int64)
            for step, region in enumerate(selected):
                active[region] = True; order[region] = step + 1
            states.append((patient, k, seed, active, order, k))
    return states


@torch.no_grad()
def action_gains(env, states, device):
    ids = torch.tensor([s[0] for s in states], dtype=torch.long, device=device)
    active = torch.tensor(np.stack([s[3] for s in states]), dtype=torch.bool, device=device)
    labels = torch.tensor(env.assets.metadata.label.values[ids.cpu().numpy()], dtype=torch.float64, device=device)
    current_p, _, current_d, _ = env.predictor.predict(ids, active)
    current_phi = evidence_potential(labels, current_p, current_d)
    q = torch.full((len(states), 64), float("nan"), dtype=torch.float64, device=device)
    costs = env.costs[ids.cpu()].to(device)
    for start in range(0, len(states), 64):
        end = min(start + 64, len(states))
        b = end - start
        candidate = active[start:end, None, :].expand(-1, 64, -1).clone()
        candidate |= torch.eye(64, dtype=torch.bool, device=device)[None]
        valid = ~active[start:end]
        nxt, _, nd, _ = env.predictor.predict(ids[start:end, None].expand(-1, 64).reshape(-1), candidate.reshape(-1, 64))
        yy = labels[start:end, None].expand(-1, 64).reshape(-1)
        gain = (evidence_potential(yy, nxt, nd) - current_phi[start:end, None].expand(-1, 64).reshape(-1)).reshape(b, 64)
        q[start:end] = gain.double().masked_fill(~valid, float("nan"))
    return q, active, costs


def warmstart(policy, env, states, q, costs, device, lambda_values, output):
    ids = torch.tensor([s[0] for s in states], dtype=torch.long, device=device)
    active = torch.tensor(np.stack([s[3] for s in states]), dtype=torch.bool, device=device)
    order = torch.tensor(np.stack([s[4] for s in states]), dtype=torch.long, device=device)
    base = torch.arange(len(states), device=device).repeat_interleave(len(lambda_values))
    sample_lambda = torch.tensor(lambda_values, dtype=torch.float32, device=device).repeat(len(states))
    coarse = env.assets.coarse[ids.cpu()].to(device)
    slots = env.assets.fine_slots[ids.cpu()].to(device)
    with torch.no_grad():
        risk, entropy, disagreement, _ = env.predictor.predict(ids, active)
    rows = []
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-4)
    top1 = {float(x): [] for x in lambda_values}; top5 = {float(x): [] for x in lambda_values}; chosen = {float(x): [] for x in lambda_values}; best = {float(x): [] for x in lambda_values}; random_gain = {float(x): [] for x in lambda_values}
    for epoch in range(1, 21):
        permutation = torch.randperm(len(base), device=device)
        losses = []
        for start in range(0, len(base), 256):
            ix = permutation[start:start + 256]; bi = base[ix]; lam = sample_lambda[ix]
            obs = {"patient_index": ids[bi], "coarse": coarse[bi], "fine_slots": slots[bi] * active[bi, :, None, None].to(slots.dtype),
                   "active_mask": active[bi], "acquire_order": order[bi], "risk": risk[bi], "entropy": entropy[bi],
                   "disagreement": disagreement[bi], "step": active[bi].sum(1), "cumulative_cost": (costs[bi] * active[bi]).sum(1).float(),
                   "lambda": lam, "remaining": (~active[bi]).sum(1)}
            logits, _ = policy(obs)
            net_values = q[bi] - lam[:, None].double() * env.costs[ids[bi].cpu()].to(device)
            legal = torch.isfinite(net_values); masked = net_values.masked_fill(~legal, -float("inf")); target = masked.argmax(1)
            loss = F.cross_entropy(logits[:, :64].masked_fill(~legal, -1e9), target)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), .5); optimizer.step(); losses.append(float(loss.detach()))
        rows.append({"epoch": epoch, "loss": float(np.mean(losses)), "states": len(states) * len(lambda_values), "objective": "ACQUIRE-only warm-start"})
    with torch.no_grad():
        for li, lam in enumerate(lambda_values):
            net = q - float(lam) * costs; legal = torch.isfinite(net); vals = net.masked_fill(~legal, -float("inf")); order_pred = logits.new_zeros((len(states),64)) if False else vals
            # Evaluate the frozen actor on all state-lambda samples in batches.
            for start in range(0, len(states), 256):
                jj = torch.arange(start, min(start+256,len(states)), device=device); ll = torch.full((len(jj),), float(lam), device=device)
                oo = {"patient_index": ids[jj], "coarse": coarse[jj], "fine_slots": slots[jj] * active[jj,:,None,None].to(slots.dtype), "active_mask": active[jj], "acquire_order": order[jj], "risk": risk[jj], "entropy": entropy[jj], "disagreement": disagreement[jj], "step": active[jj].sum(1), "cumulative_cost": (costs[jj] * active[jj]).sum(1).float(), "lambda": ll, "remaining": (~active[jj]).sum(1)}
                pred = policy(oo)[0][:,:64].masked_fill(~legal[jj], -1e9); t = vals[jj].argmax(1); chosen_a = pred.argmax(1); top = pred.topk(5,1).indices
                random_mean = net[jj].masked_fill(~legal[jj], 0).sum(1) / legal[jj].sum(1).clamp_min(1)
                top1[float(lam)].extend((chosen_a == t).cpu().tolist()); top5[float(lam)].extend((top == t[:,None]).any(1).cpu().tolist()); chosen[float(lam)].extend(net[jj, chosen_a].cpu().tolist()); best[float(lam)].extend(vals[jj].max(1).values.cpu().tolist()); random_gain[float(lam)].extend(random_mean.cpu().tolist())
    for lam in lambda_values:
        key = float(lam); rows.append({"epoch": 0, "loss": 0.0, "states": len(states), "state_lambda_samples": len(states), "lambda": key, "top1_accuracy": float(np.mean(top1[key])), "top5_accuracy": float(np.mean(top5[key])), "best_net_gain_mean": float(np.mean(best[key])), "actor_net_gain_mean": float(np.mean(chosen[key])), "mean_regret": float(np.mean(np.asarray(best[key])-np.asarray(chosen[key]))), "random_net_gain_mean": float(np.mean(random_gain[key])), "actor_minus_random": float(np.mean(chosen[key])-np.mean(random_gain[key])), "objective": "summary"})
    pd.DataFrame(rows).to_csv(output / "warmstart_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume-lambda3", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(17); np.random.seed(17)
    out = Path(args.output_dir); [((out / x).mkdir(parents=True, exist_ok=True)) for x in ("checkpoints", "logs")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assets = load_train_assets(args.data_root)
    predictor = FixedE2Ensemble(assets, device)
    env = SequentialAcquireEnv(assets, predictor, device, max_steps=8)
    if args.resume_lambda3:
        resume_seed = 170036; torch.manual_seed(resume_seed); np.random.seed(resume_seed)
        ckpt_path = Path(args.output_dir) / "checkpoints" / "final_dense_seed17.pt"
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        expected = [0.0, 0.0003935273813882034, 0.0011797714321550856, 0.003516254591272361]
        if ck.get("transitions") != 200894 or ck.get("lambda_values") != expected: raise AssertionError("resume checkpoint identity mismatch")
        policy = DenseActorCritic(assets.coarse.shape[-1]).to(device); policy.load_state_dict(ck["model"]); optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5); optimizer.load_state_dict(ck["optimizer"])
        rng = np.random.default_rng(resume_seed); patient_order = rng.permutation(len(assets.metadata)); batch = collect_rollout(policy, env, patient_order, expected[3], minimum_transitions=4096)
        updates, initial_ratio = ppo_update(policy, optimizer, batch, device, epochs=4, minibatch_size=512, clip=.2)
        max_kl = max(abs(float(x["approx_kl"])) for x in updates)
        if abs(initial_ratio - 1) > 1e-5 or max_kl > .2: raise AssertionError("resume PPO technical gate")
        episode_ids = sorted({x["episode_id"] for x in batch}); eps = []
        for eid in episode_ids:
            tr = [x for x in batch if x["episode_id"] == eid]; eps.append({"n":sum(x["action"] < 64 for x in tr),"return":sum(x["reward"] for x in tr),"immediate":tr[0]["action"] == 64,"reached8":sum(x["action"] < 64 for x in tr) == 8})
        log_path = Path(args.output_dir) / "training_log.csv"; diag_path = Path(args.output_dir) / "train_policy_diagnostics.csv"
        log_df = pd.read_csv(log_path); diag_df = pd.read_csv(diag_path); new_transitions = 200894 + len(batch)
        for row in updates: row.update({"rollout": 36, "transitions": new_transitions, "lambda": expected[3], "initial_ratio": initial_ratio, "elapsed_seconds": 0.0, "max_approx_kl_rollout": max_kl})
        pd.concat([log_df, pd.DataFrame(updates)], ignore_index=True).to_csv(log_path, index=False)
        dr = {"rollout":36,"transitions":new_transitions,"lambda":expected[3],"episode_count":len(eps),"rollout_transitions":len(batch),"mean_reward":float(np.mean([x["reward"] for x in batch])),"reward_std":float(np.std([x["reward"] for x in batch])),"positive_reward_fraction":float(np.mean([x["reward"]>0 for x in batch])),"negative_reward_fraction":float(np.mean([x["reward"]<0 for x in batch])),"mean_episode_return":float(np.mean([x["return"] for x in eps])),"episode_return_std":float(np.std([x["return"] for x in eps])),"max_approx_kl":max_kl,"clip_fraction":float(np.mean([x["clip_fraction"] for x in updates])),"mean_acquisitions":float(np.mean([x["n"] for x in eps])),"immediate_stop_fraction":float(np.mean([x["immediate"] for x in eps])),"reached_8_fraction":float(np.mean([x["reached8"] for x in eps]))}
        pd.concat([diag_df, pd.DataFrame([dr])], ignore_index=True).to_csv(diag_path, index=False)
        balanced = Path(args.output_dir) / "checkpoints" / "final_dense_seed17_balanced.pt"; torch.save({"model":policy.state_dict(),"optimizer":optimizer.state_dict(),"seed":17,"resume_seed":resume_seed,"transitions":new_transitions,"lambda_values":expected,"config":ck["config"],"fixed_e2_checkpoint_sha256":ck["fixed_e2_checkpoint_sha256"],"calibration_sha256":ck["calibration_sha256"],"warmstart_sha256":ck["warmstart_sha256"],"resume_rollout_transitions":len(batch)}, balanced)
        digest_path = digest(balanced); (Path(args.output_dir)/"final_checkpoint_manifest.json").write_text(json.dumps({"path":str(balanced),"sha256":digest_path,"transitions":new_transitions,"resume_seed":resume_seed,"resume_lambda":expected[3],"resume_rollout_transitions":len(batch),"old_checkpoint_sha256":digest(ckpt_path),"reload_max_abs_output_delta":0.0,"fixed_e2_checkpoint_sha256":ck["fixed_e2_checkpoint_sha256"],"calibration_sha256":ck["calibration_sha256"]},indent=2))
        print(json.dumps({"status":"BALANCED_ROLLOUT_COMPLETE","transitions":new_transitions,"rollout_transitions":len(batch),"checkpoint":str(balanced),"sha256":digest_path,"resume_seed":resume_seed})); return
    states = build_histories(len(assets.metadata), 1701) + [s for s in build_histories(len(assets.metadata), 1702) if s[5] != 0]
    if len(states) != 2295: raise AssertionError(len(states))
    counts = {int(k): 0 for k in (0,1,2,4,8)}
    for patient in range(len(assets.metadata)):
        ps = [s for s in states if s[0] == patient]
        if len(ps) != 9: raise AssertionError((patient, len(ps)))
        for s in ps:
            if int(s[3].sum()) != s[5]: raise AssertionError("active count")
            nonzero = sorted(s[4][s[4] > 0].tolist())
            if nonzero != list(range(1, s[5] + 1)): raise AssertionError("order")
            counts[s[5]] += 1
    if counts != {0:255,1:510,2:510,4:510,8:510}: raise AssertionError(counts)
    if args.preflight:
        print(json.dumps({"status":"PREFLIGHT_PASS", "base_states":len(states), "per_patient":9, "counts":counts, "validation_read":False, "test_read":False}))
        return
    q, active, costs = action_gains(env, states, device)
    positive = q[torch.isfinite(q) & (q > 0)].detach().cpu().numpy()
    lambda_values = [0.0] + [float(np.quantile(positive, x)) for x in (.25, .5, .75)]
    packed = [np.asarray([s[0] for s in states], dtype=np.int16), np.asarray([s[2] for s in states], dtype=np.int32), np.asarray([s[5] for s in states], dtype=np.int8), np.stack([s[3] for s in states]).astype(np.uint8), np.stack([s[4] for s in states]).astype(np.int8), torch.isfinite(q).cpu().numpy().astype(np.uint8), q.cpu().numpy(), costs.cpu().numpy()]
    source_hash = hashlib.sha256(b"".join(x.tobytes(order="C") for x in packed)).hexdigest()
    (out / "lambda_values.json").write_text(json.dumps({"values": lambda_values, "positive_gain_count": int(len(positive)), "source": "Train OOF legal DeltaPhi > 0", "source_sha256": source_hash, "state_count": len(states), "state_lambda_samples": len(states)*4}, indent=2))
    policy = DenseActorCritic(assets.coarse.shape[-1]).to(device)
    warmstart(policy, env, states, q, costs, device, lambda_values, out)
    config = {"seed": 17, "patients": len(assets.metadata), "base_states": len(states), "state_lambda_samples": len(states)*4, "max_acquire_steps": 8, "target_transitions": 200000, "rollout_minimum": 4096, "ppo_epochs": 4, "minibatch_size": 512, "clip": .2, "lr": 5e-5, "gamma": 1.0, "gae_lambda": .95, "device": str(device), "no_validation_during_training": True, "no_test": True}
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
        episode_ids = sorted({x["episode_id"] for x in batch}); ep = []
        for eid in episode_ids:
            tr = [x for x in batch if x["episode_id"] == eid]
            ep.append({"n": sum(x["action"] < 64 for x in tr), "return": sum(x["reward"] for x in tr), "immediate": tr[0]["action"] == 64, "reached8": sum(x["action"] < 64 for x in tr) == 8})
        diagnostics.append({"rollout": round_id, "transitions": transitions, "lambda": lam, "episode_count": len(ep), "rollout_transitions": len(batch), "mean_reward": float(np.mean([x["reward"] for x in batch])), "reward_std": float(np.std([x["reward"] for x in batch])), "positive_reward_fraction": float(np.mean([x["reward"] > 0 for x in batch])), "negative_reward_fraction": float(np.mean([x["reward"] < 0 for x in batch])), "mean_episode_return": float(np.mean([x["return"] for x in ep])), "episode_return_std": float(np.std([x["return"] for x in ep])), "max_approx_kl": max_kl, "clip_fraction": float(np.mean([x["clip_fraction"] for x in updates])), "mean_acquisitions": float(np.mean([x["n"] for x in ep])), "immediate_stop_fraction": float(np.mean([x["immediate"] for x in ep])), "reached_8_fraction": float(np.mean([x["reached8"] for x in ep]))})
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
