#!/usr/bin/env python3
"""Validation-only benchmark for the locked Dense PPO checkpoint.

All methods produce patient-level predictions first. Labels are read only after
the complete trajectory for a method has been generated.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from MSRLSelect.rl_core.assets import FixedE2Ensemble, load_train_assets
from MSRLSelect.rl_core.core import DenseActorCritic, SequentialAcquireEnv, evidence_potential


def sha256(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def metrics(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "s_val": float((roc_auc_score(y, p) + average_precision_score(y, p)) / 2),
        "brier": float(brier_score_loss(y, p)),
    }


def validation_assets(root, assets, val):
    evidence = root / "outputs/thesis_evidence_interface_v7"
    comp = np.load(evidence / "assets/train_pca128_components.npy")
    mean = np.load(evidence / "assets/train_pca128_mean.npy")
    ev = np.load(evidence / "assets/train_pca128_explained_variance.npy")
    pc = PCA(128)
    pc.components_, pc.mean_, pc.explained_variance_ = comp, mean, ev
    pc.explained_variance_ratio_ = ev / ev.sum()
    pc.singular_values_ = np.sqrt(np.maximum(ev * (16320 - 1), 0))
    pc.n_features_in_, pc.n_samples_, pc.n_samples_seen_, pc.noise_variance_ = 1024, 16320, 16320, 0.0
    region_root = root / "outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache/regions_k64/valid"
    coarse_rows = []
    for row in val.itertuples():
        with np.load(region_root / f"{row.case_id}.npz") as z:
            coarse_rows.append(np.concatenate([
                pc.transform(z["region_feature_mean"].astype("float32")),
                np.log1p(z["region_size"]).astype("float32")[:, None],
                z["region_centroid"], z["region_spread"]], 1))
    coarse = torch.from_numpy(np.stack(coarse_rows).astype("float32"))
    slots = torch.load(evidence / "assets/validation_moe_slots.pt", map_location="cpu", weights_only=True).float()
    class View: pass
    view = View()
    view.root, view.metadata, view.coarse, view.fine_slots = root, val, coarse, slots
    view.calibration, view.calibration_path = assets.calibration, assets.calibration_path
    view.calibration_sha256, view.checkpoint_paths, view.opened_paths = assets.calibration_sha256, assets.checkpoint_paths, []
    return view


def obs(view, predictor, ids, active, order, cumulative, lam, device):
    risk, entropy, disagreement, _ = predictor.predict(ids, active)
    return {"patient_index": ids, "coarse": view.coarse[ids.cpu()].to(device),
            "fine_slots": view.fine_slots[ids.cpu()].to(device) * active[:, :, None, None].to(view.fine_slots.dtype),
            "active_mask": active, "acquire_order": order, "risk": risk, "entropy": entropy,
            "disagreement": disagreement, "step": active.sum(1), "cumulative_cost": cumulative.float(),
            "lambda": torch.full((len(ids),), float(lam), device=device), "remaining": (~active).sum(1)}


@torch.no_grad()
def choose_fixed(view, predictor, policy, ids, budget, lam, device, method):
    n = len(ids); active = torch.zeros(n, 64, dtype=torch.bool, device=device)
    order = torch.zeros(n, 64, dtype=torch.long, device=device); cumulative = torch.zeros(n, dtype=torch.float64, device=device)
    rows = []
    for step in range(budget):
        if method == "random":
            action = torch.stack([torch.where(~active[i])[0][torch.randint((~active[i]).sum(), (1,), device=device)[0]] for i in range(n)])
        elif method == "coarse":
            score = view.coarse[ids.cpu()].to(device).norm(dim=-1).masked_fill(active, -1e9); action = score.argmax(1)
        elif method == "attention":
            # Static full-fine baseline: region informativeness from the complete fine slots.
            score = view.fine_slots[ids.cpu()].to(device).flatten(-2).norm(dim=-1).masked_fill(active, -1e9); action = score.argmax(1)
        else:
            logits, _ = policy(obs(view, predictor, ids, active, order, cumulative, lam, device))
            action = logits[:, :64].masked_fill(active, -1e9).argmax(1)
        before = active.clone(); active[torch.arange(n, device=device), action] = True
        order[torch.arange(n, device=device), action] = step + 1
        coarse_batch = view.coarse[ids.cpu()].to(device)
        costs = ((coarse_batch[:, :, 128] - coarse_batch[:, :, 128].amin(1, keepdim=True)) /
                 (coarse_batch[:, :, 128].amax(1, keepdim=True) - coarse_batch[:, :, 128].amin(1, keepdim=True)).clamp_min(1e-6))[torch.arange(n, device=device), action]
        cumulative += costs.double()
        risk, ent, dis, _ = predictor.predict(ids, active)
        for i in range(n):
            rows.append({"method": method, "budget": budget, "patient_id": str(view.metadata.iloc[i].case_id), "step": step + 1,
                         "action": int(action[i]), "active_count": int(active[i].sum()), "risk": float(risk[i]),
                         "entropy": float(ent[i]), "disagreement": float(dis[i]), "cost": float(costs[i]), "cumulative_cost": float(cumulative[i]),
                         "action_type": "ACQUIRE"})
    return active, order, cumulative, rows


@torch.no_grad()
def choose_adaptive(view, predictor, policy, ids, lam, device):
    n = len(ids); active = torch.zeros(n, 64, dtype=torch.bool, device=device); order = torch.zeros(n, 64, dtype=torch.long, device=device)
    cumulative = torch.zeros(n, dtype=torch.float64, device=device); alive = torch.ones(n, dtype=torch.bool, device=device); rows = []
    for step in range(8):
        if not alive.any(): break
        ix = torch.where(alive)[0]; ob = obs(view, predictor, ids[ix], active[ix], order[ix], cumulative[ix], lam, device)
        logits, _ = policy(ob); action = logits.argmax(1); stop = action == 64
        for j, row in enumerate(ix.tolist()):
            a = int(action[j]); rows.append({"method": "DensePPO_Adaptive", "lambda": lam, "patient_id": str(view.metadata.iloc[row].case_id), "step": step,
                "action": a, "action_type": "STOP" if a == 64 else "ACQUIRE", "stop_probability": float(torch.softmax(logits[j], 0)[64]),
                "action_probability": float(torch.softmax(logits[j], 0)[a]), "active_count": int(active[row].sum()),
                "risk": float(ob["risk"][j]), "entropy": float(ob["entropy"][j]), "disagreement": float(ob["disagreement"][j]),
                "cost": 0.0, "cumulative_cost": float(cumulative[row]), "terminal_reason": "STOP" if a == 64 else "",
                "active_mask": active[row].detach().cpu().tolist()})
        acq = ix[~stop]; aa = action[~stop]
        if acq.numel():
            active[acq, aa] = True; order[acq, aa] = step + 1
            c = view.coarse.to(device)[:, :, 128]; c = (c - c.amin(1, keepdim=True)) / (c.amax(1, keepdim=True) - c.amin(1, keepdim=True)).clamp_min(1e-6)
            cumulative[acq] += c[acq, aa].double()
        alive[ix[stop]] = False
        alive[ix[~stop] & (active[ix[~stop]].sum(1) >= 64)] = False
    return active, order, cumulative, rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-root", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--device", default="cuda")
    a = p.parse_args(); root, out = Path(a.data_root), Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu"); ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    assets = load_train_assets(root); predictor = FixedE2Ensemble(assets, device)
    val = pd.read_csv(root / "prepared/sr386_full/surgen_validate_10.csv"); view = validation_assets(root, assets, val)
    predictor = FixedE2Ensemble(view, device); ids = torch.arange(len(val), device=device)
    policy = DenseActorCritic(view.coarse.shape[-1]).to(device); policy.load_state_dict(ck["model"]); policy.eval()
    y = val.label.to_numpy(); predictions, trajectories, stop_audit = [], [], []
    def add(method, pvec, budget, costs=None, immediate=0.0, max8=0.0):
        m = metrics(y, pvec); m.update({"method": method, "mean_acquisitions": float(budget if np.isscalar(budget) else np.mean(budget)), "acquisition_std": float(0 if np.isscalar(budget) else np.std(budget)), "mean_cost": float(np.mean(costs) if costs is not None else 0), "immediate_stop_fraction": immediate, "reached_8_fraction": max8}); predictions.append(m)
    for k, name in ((0, "E2_K0"), (64, "E2_Full64")):
        active = torch.full((len(val), 64), k == 64, dtype=torch.bool, device=device); pp = predictor.predict(ids, active)[0].cpu().numpy(); add(name, pp, k)
    for budget in (4, 8):
        for method, label in (("random", "Random20"), ("coarse", "coarse-only"), ("attention", "full-fine static baseline"), ("policy", "DensePPO")):
            reps = []
            for seed in range(20 if method == "random" else 1):
                torch.manual_seed(seed); active, order, cost, rows = choose_fixed(view, predictor, policy, ids, budget, 0.0, device, method); pp = predictor.predict(ids, active)[0].cpu().numpy(); reps.append(pp); trajectories.extend(rows)
            for j, pp in enumerate(reps): add(f"{label}_K{budget}" + (f"_seed{j}" if method == "random" else ""), pp, budget)
            if method == "random": add(f"{label}_K{budget}_mean", np.mean(reps, 0), budget)
    lambda_path = out / "lambda_values.json"
    if not lambda_path.is_file():
        lambda_path = root / "outputs/thesis_rl_v8/dense_formal_seed17/lambda_values.json"
    lambda_values = json.loads(lambda_path.read_text())["values"] if lambda_path.is_file() else [0.0]
    adaptive_budgets = {}
    for lam in lambda_values:
        active, order, cost, rows = choose_adaptive(view, predictor, policy, ids, float(lam), device); pp = predictor.predict(ids, active)[0].cpu().numpy(); budgets = active.sum(1).cpu().numpy(); trajectories.extend(rows); add(f"DensePPO_Adaptive_lambda{float(lam):.8g}", pp, budgets, cost.cpu().numpy(), float(np.mean(budgets == 0)), float(np.mean(budgets >= 8)))
        adaptive_budgets[float(lam)] = budgets
        # Post-hoc one-step STOP regret. This is deliberately computed after rollout.
        labels = torch.tensor(y, dtype=torch.float64, device=device)
        for row in rows:
            if row["action_type"] != "STOP": continue
            i = int(val.index[val.case_id.astype(str) == str(row["patient_id"])][0]); mask = torch.tensor(row["active_mask"], dtype=torch.bool, device=device).unsqueeze(0); ii = torch.tensor([i], device=device)
            p0, _, d0, _ = predictor.predict(ii, mask); ph0 = evidence_potential(labels[i:i+1], p0, d0)
            candidates = mask.expand(64, -1).clone(); candidates |= torch.eye(64, dtype=torch.bool, device=device); valid = ~mask[0]
            pp1, _, dd1, _ = predictor.predict(ii.expand(64), candidates); rawc = view.coarse[i, :, 128].to(device); costs = (rawc - rawc.min()) / (rawc.max() - rawc.min()).clamp_min(1e-6); gains = (evidence_potential(labels[i:i+1].expand(64), pp1, dd1) - ph0 - float(lam) * costs.double()).masked_fill(~valid, -float("inf")); best = float(gains.max())
            stop_audit.append({"patient_id": str(row["patient_id"]), "step": row["step"], "lambda": float(lam), "best_net_gain": best, "stop_regret": max(0.0, best), "checkpoint": str(a.checkpoint)})
    # Matched-budget baselines use exactly each adaptive patient's budget.
    for lam, budgets in adaptive_budgets.items():
        for method, label in (("random", "Random20_MatchedBudget"), ("attention", "full-fine static matched-budget")):
            reps = []
            for seed in range(20 if method == "random" else 1):
                torch.manual_seed(seed); patient_predictions = []
                patient_predictions = np.zeros(len(val), dtype=np.float32)
                for budget in sorted(set(budgets.tolist())):
                    group = np.where(budgets == budget)[0]
                    if budget == 0:
                        aa = torch.zeros(len(group), 64, dtype=torch.bool, device=device)
                        patient_predictions[group] = predictor.predict(torch.tensor(group, device=device), aa)[0].cpu().numpy()
                    else:
                        aa, _, _, _ = choose_fixed(view, predictor, policy, torch.tensor(group, device=device), int(budget), lam, device, method)
                        patient_predictions[group] = predictor.predict(torch.tensor(group, device=device), aa)[0].cpu().numpy()
                reps.append(np.asarray(patient_predictions))
            for j, pp in enumerate(reps): add(f"{label}_lambda{lam:.8g}" + (f"_seed{j}" if method == "random" else ""), pp, budgets)
            if method == "random": add(f"{label}_lambda{lam:.8g}_mean", np.mean(reps, 0), budgets)
    pd.DataFrame(predictions).to_csv(out / "validation_metrics.csv", index=False); pd.DataFrame(predictions).to_csv(out / "validation_predictions.csv", index=False)
    with (out / "validation_action_sequences.jsonl").open("w") as f:
        for row in trajectories: f.write(json.dumps(row) + "\n")
    pd.DataFrame(stop_audit).to_csv(out / "validation_stop_audit.csv", index=False)
    print(json.dumps({"validation_patients": len(val), "methods": len(predictions), "checkpoint_sha256": sha256(a.checkpoint)}))


if __name__ == "__main__": main()
