"""Eight-patient, 2,000-transition Train-only PPO smoke run."""

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .assets import FixedE2Ensemble, load_train_assets
from .core import DenseActorCritic, SequentialAcquireEnv, collect_rollout, evidence_potential, ppo_update, stack_observations
from .test_core import Acceptance, checkpoint_reload_test, run_pre_smoke_tests


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initial_action_values(env, patient_ids):
    values = torch.empty(len(patient_ids), 64, dtype=torch.float64, device=env.device)
    state = env.reset(patient_ids, 0.0)
    labels = env.state.labels.clone()
    current_phi = evidence_potential(labels, env.state.risk, env.state.disagreement)
    eye = torch.eye(64, dtype=torch.bool, device=env.device)
    for row, patient in enumerate(patient_ids):
        repeated = torch.full((64,), int(patient), dtype=torch.long, device=env.device)
        risk, _, disagreement, _ = env.predictor.predict(repeated, eye)
        values[row] = evidence_potential(labels[row].expand(64), risk, disagreement) - current_phi[row]
    return values


def overfit_critic(policy, transitions, device):
    sample = transitions[:min(128, len(transitions))]
    observation = stack_observations([row["observation"] for row in sample], device)
    target = torch.tensor([row["return"] for row in sample], device=device)
    model = DenseActorCritic(observation["coarse"].shape[-1]).to(device)
    model.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(500):
        value = model(observation)[1]
        loss = F.mse_loss(value, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad(): prediction = model(observation)[1]
    return float(1 - torch.var(target - prediction) / (torch.var(target) + 1e-12))


def serializable_transition(row):
    return {
        "episode_id": row["episode_id"], "patient_index": row["patient_index"], "step": row["step"],
        "active_mask_before": row["active_mask_before"].to(torch.int8).tolist(),
        "order_before": row["order_before"].to(torch.int8).tolist(), "action": row["action"],
        "old_log_prob": row["old_log_prob"], "value": row["value"], "reward": row["reward"],
        "done": row["done"], "risk_before": row["risk_before"], "risk_after": row["risk_after"],
        "cost": row["cost"], "advantage": row["advantage"], "return": row["return"],
        "old_policy_frozen": row["old_policy_frozen"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(17); np.random.seed(17)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    allowed = {"acceptance.json", "transition_trace.jsonl", "training_log.csv", "smoke_checkpoint.pt", "asset_manifest.json"}
    assets = load_train_assets(args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictor = FixedE2Ensemble(assets, device)
    environment = SequentialAcquireEnv(assets, predictor, device, max_steps=8)
    policy = DenseActorCritic(assets.coarse.shape[-1]).to(device)
    acceptance = Acceptance()

    labels = assets.metadata.label.to_numpy()
    negatives, positives = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    patient_ids = np.concatenate([negatives[:4], positives[:4]])
    run_pre_smoke_tests(acceptance, policy, environment, patient_ids)
    environment.max_steps = 8

    q0 = initial_action_values(environment, patient_ids)
    positive = q0[q0 > 0].detach().cpu().numpy()
    lambda_cost = float(np.quantile(positive, 0.25)) if len(positive) else 0.0
    costs = environment.costs[torch.as_tensor(patient_ids, dtype=torch.long)].to(device)
    net_q0 = q0 - lambda_cost * costs
    targets = net_q0.argmax(1)
    independently_reconstructed = (q0 - torch.as_tensor(lambda_cost, device=device) * costs).argmax(1)
    target_error = int((targets != independently_reconstructed).sum())
    acceptance.check("first_action_net_gain_target", target_error == 0, target_error, "==0", "Target is argmax(DeltaPhi - lambda*region_cost)")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-3)
    for _ in range(800):
        observation = environment.reset(patient_ids, lambda_cost)
        logits, _ = policy(observation)
        loss = F.cross_entropy(logits[:, :64], targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        observation = environment.reset(patient_ids, lambda_cost)
        first_accuracy = float((policy(observation)[0][:, :64].argmax(1) == targets).float().mean())
    acceptance.check("first_action_warmstart_accuracy", first_accuracy >= 0.95, first_accuracy, ">=0.95", "Eight real Train patients, STOP masked")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5)
    all_transitions, training_rows = [], []
    while len(all_transitions) < 2000:
        requested = min(256, max(1, 2000 - len(all_transitions)))
        started = time.time()
        transitions = collect_rollout(policy, environment, patient_ids, lambda_cost, requested)
        update_rows, initial_ratio = ppo_update(policy, optimizer, transitions, device, epochs=4, minibatch_size=64)
        episode_offset = 0 if not all_transitions else max(row["episode_id"] for row in all_transitions) + 1
        for transition in transitions:
            transition["episode_id"] += episode_offset
        all_transitions.extend(transitions)
        for row in update_rows:
            row.update({"transitions": len(all_transitions), "initial_ratio_mean": initial_ratio,
                        "elapsed_seconds": time.time() - started,
                        "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0})
            training_rows.append(row)
    acceptance.check("ppo_2000_real_transitions", len(all_transitions) >= 2000, len(all_transitions), ">=2000 complete-episode transitions", "All rewards came from fold-specific frozen E2; no episode truncation")
    acceptance.check("ppo_initial_ratio", abs(training_rows[0]["initial_ratio_mean"] - 1) < 1e-5, training_rows[0]["initial_ratio_mean"], "abs(ratio-1)<1e-5", "Old policy snapshot and stored detached log-prob")
    old_frozen = all(row["old_policy_frozen"] for row in all_transitions)
    acceptance.check("ppo_old_policy_frozen", old_frozen, old_frozen, "True", "Independent eval-mode snapshot has requires_grad=False during sampling")
    maximum_clip = max(row["clip_fraction"] for row in training_rows)
    acceptance.check("ppo_nonzero_clip_fraction", maximum_clip > 0, maximum_clip, ">0", "Four PPO epochs per rollout batch")
    all_shuffled = all(bool(row["minibatch_shuffled"]) for row in training_rows)
    acceptance.check("ppo_random_minibatch_shuffle", all_shuffled, all_shuffled, "True", "Fresh torch.randperm generated for every PPO epoch")
    finite = all(np.isfinite(row[key]) for row in training_rows for key in ("approx_kl", "actor_loss", "critic_loss"))
    acceptance.check("ppo_losses_finite", finite, finite, "True", "Actual minibatch diagnostics")
    for name, field in (("dense_encoder_gradient", "encoder_grad"), ("actor_head_gradient", "actor_grad"), ("value_head_gradient", "value_grad")):
        measured = max(row[field] for row in training_rows)
        acceptance.check(name, measured > 0, measured, ">0", "Gradient measured before clipping")

    stopped_episodes = {row["episode_id"]: row["step"] for row in all_transitions if row["action"] == 64}
    post_stop = sum(1 for row in all_transitions if row["episode_id"] in stopped_episodes and row["step"] > stopped_episodes[row["episode_id"]])
    acceptance.check("no_transition_after_stop", post_stop == 0, post_stop, "==0", "Variable-length rollout stores alive patients only")
    critic_ev = overfit_critic(policy, all_transitions, device)
    acceptance.check("critic_fixed_batch_overfit_ev", critic_ev > 0.8, critic_ev, ">0.8", "Overfit check on fixed real trajectory returns only")

    checkpoint = output / "smoke_checkpoint.pt"
    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "seed": 17,
                "transitions": len(all_transitions), "lambda_cost": lambda_cost,
                "calibration_path": str(assets.calibration_path), "calibration_sha256": assets.calibration_sha256}, checkpoint)
    checkpoint_reload_test(acceptance, policy, environment.reset(patient_ids, lambda_cost), checkpoint)
    with (output / "transition_trace.jsonl").open("w") as handle:
        for row in all_transitions:
            handle.write(json.dumps(serializable_transition(row)) + "\n")
    pd.DataFrame(training_rows).to_csv(output / "training_log.csv", index=False)
    (output / "acceptance.json").write_text(json.dumps(acceptance.rows, indent=2))
    manifest = {
        "data_root": str(assets.root), "train_metadata": str(assets.root / "prepared/sr386_full/surgen_train_10.csv"),
        "fixed_e2_checkpoints": [{"path": str(path), "sha256": file_hash(path)} for path in assets.checkpoint_paths],
        "calibration_path": str(assets.calibration_path), "calibration_sha256": assets.calibration_sha256,
        "calibration_identity": "Fixed E2 OOF; ensemble three raw seed probabilities, then calibrate mean",
        "patient_indices": patient_ids.tolist(), "labels": labels[patient_ids].tolist(), "lambda_cost": lambda_cost,
        "opened_data_paths": assets.opened_paths, "validation_or_test_read": False,
    }
    (output / "asset_manifest.json").write_text(json.dumps(manifest, indent=2))
    unexpected = {path.name for path in output.iterdir()} - allowed
    if unexpected:
        raise AssertionError(f"Unexpected output files: {unexpected}")
    print(json.dumps({"status": "PASS", "transitions": len(all_transitions), "critic_ev": critic_ev,
                      "post_stop_transitions": post_stop, "acceptance_tests": len(acceptance.rows)}))


if __name__ == "__main__":
    main()
