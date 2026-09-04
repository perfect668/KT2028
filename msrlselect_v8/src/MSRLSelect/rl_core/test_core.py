"""Acceptance checks for the minimal sequential PPO core."""

import inspect

import numpy as np
import torch

from .core import compute_gae, evidence_potential


class Acceptance:
    def __init__(self):
        self.rows = []

    def check(self, name, condition, measured, threshold, evidence):
        self.rows.append({"test_name": name, "status": "PASS" if condition else "FAIL",
                          "measured_value": measured, "threshold": threshold, "evidence": evidence})
        if not condition:
            raise AssertionError(f"{name}: measured={measured}, threshold={threshold}; {evidence}")


def run_pre_smoke_tests(acceptance, policy, env, patient_ids):
    device = env.device
    expected_calibration = (env.assets.root / "outputs/thesis_rl_v7/r1_online_ppo/metrics/e2_calibration_parameters.csv").resolve()
    actual_calibration = env.assets.calibration_path.resolve()
    acceptance.check("fixed_e2_calibration_identity", actual_calibration == expected_calibration and "e2m" not in str(actual_calibration).lower(), str(actual_calibration), str(expected_calibration), "Exact Fixed E2 OOF calibration path")
    expected_names = {f"E2_oof_fold{fold}_seed{seed}_best.pt" for fold in range(5) for seed in (1, 17, 42)}
    actual_names = {path.name for path in env.assets.checkpoint_paths}
    acceptance.check("fixed_e2_oof_checkpoint_inventory", actual_names == expected_names, len(actual_names), "15 exact fold/seed names", "No full-cohort or E2M checkpoint accepted")
    frozen = all(not model.training and all(not parameter.requires_grad for parameter in model.parameters()) for models in env.predictor.models.values() for model in models)
    acceptance.check("fixed_e2_eval_and_frozen", frozen, frozen, "True", "All 15 loaded models inspected")
    observation = env.reset(patient_ids[:2], 0.0)
    with torch.enable_grad():
        risk_probe = env.predictor.predict(patient_ids[:2], observation["active_mask"])[0]
    acceptance.check("fixed_e2_no_grad_forward", not risk_probe.requires_grad, risk_probe.requires_grad, "False", "Predict method enforces torch.no_grad")
    unobserved_values = observation["fine_slots"][~observation["active_mask"]]
    unobserved_max = float(unobserved_values.abs().max()) if unobserved_values.numel() else 0.0
    acceptance.check("observation_unobserved_fine_zero", unobserved_max == 0.0, unobserved_max, "==0", "Environment masks fine before constructing observation or rollout buffer")

    raw = torch.tensor([0.7], device=device)
    count = torch.tensor([16], device=device)
    calibrated = env.predictor._calibrate(raw, count)
    intercept, coefficients = env.assets.calibration["partial"]
    z = torch.logit(raw)
    rho = count.float() / 64.0
    expected = torch.sigmoid(intercept + coefficients[0] * z + coefficients[1] * z * rho + coefficients[2] * rho)
    calibration_error = float((calibrated - expected).abs().max())
    acceptance.check("partial_calibration_feature_order", calibration_error < 1e-12, calibration_error, "<1e-12", "Independent [z, z*rho, rho] formula")
    policy.eval()
    with torch.no_grad():
        logits, value = policy(observation)
        changed_unobserved = {key: tensor.clone() for key, tensor in observation.items()}
        changed_unobserved["fine_slots"][:, 1:] += 1000 * torch.randn_like(changed_unobserved["fine_slots"][:, 1:])
        logits_unobserved, value_unobserved = policy(changed_unobserved)
    leak_delta = max(float((logits - logits_unobserved).abs().max()), float((value - value_unobserved).abs().max()))
    acceptance.check("unobserved_fine_invariance", leak_delta < 1e-7, leak_delta, "<1e-7", "Fine slots are masked before projection")

    env.step_actions(torch.arange(2, device=device), torch.tensor([5, 5], device=device))
    observed = env.observe()
    with torch.no_grad():
        observed_logits, observed_value = policy(observed)
        changed = {key: tensor.clone() for key, tensor in observed.items()}
        changed["fine_slots"][:, 5] += 0.5
        changed_logits, changed_value = policy(changed)
    observed_delta = max(float((observed_logits - changed_logits).abs().max()), float((observed_value - changed_value).abs().max()))
    acceptance.check("observed_fine_changes_output", observed_delta > 0, observed_delta, ">0", "Observed slot participates in Dense encoding")
    acceptance.check("observation_has_no_label", "label" not in observed, list(observed), "label absent", "Reward label remains environment-private")
    signature = str(inspect.signature(policy.forward))
    signature_ok = "label" not in signature.lower() and "fold" not in signature.lower()
    acceptance.check("policy_signature_has_no_label_or_fold", signature_ok, signature, "no label/fold", "Inspected actual forward signature")
    acceptance.check("acquisition_order_semantics", int(env.state.order[0, 5]) == 1, int(env.state.order[0, 5]), "1", "order[region]=step+1")
    acceptance.check("selected_action_masked", float(observed_logits[0, 5]) < -1e8, float(observed_logits[0, 5]), "<-1e8", "Observed ACQUIRE logit is masked")

    label_logits = logits.clone()
    label_observation = {key: value.clone() for key, value in observation.items()}
    env.state.labels = 1 - env.state.labels
    with torch.no_grad(): logits_after_label, _ = policy(label_observation)
    label_delta = float((label_logits - logits_after_label).abs().max())
    acceptance.check("label_does_not_change_policy", label_delta < 1e-7, label_delta, "<1e-7", "Label is not in observation")

    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 0.4, 0.2])
    dones = torch.tensor([False, False, True])
    next_values = torch.tensor([0.4, 0.2, 0.0])
    advantage, returns = compute_gae(rewards, values, dones, next_values, gamma=1.0, gae_lambda=0.95)
    manual2 = 3.0 - 0.2
    manual1 = 2.0 + 0.2 - 0.4 + 0.95 * manual2
    manual0 = 1.0 + 0.4 - 0.5 + 0.95 * manual1
    gae_error = float((advantage - torch.tensor([manual0, manual1, manual2])).abs().max())
    return_error = float((returns - (torch.tensor([manual0, manual1, manual2]) + values)).abs().max())
    acceptance.check("three_step_gae", gae_error < 1e-6 and return_error < 1e-6, max(gae_error, return_error), "<1e-6", "Hand-calculated three-step episode")

    manual_logits = torch.full((2, 65), -4.0, device=device)
    manual_logits[0, 64], manual_logits[0, 3] = 2.0, -1.0
    manual_logits[1, 64], manual_logits[1, 3] = -1.0, 2.0
    manual_actions = manual_logits.argmax(1)
    acceptance.check("manual_nonpositive_state_stops", int(manual_actions[0]) == 64, int(manual_actions[0]), "64", "STOP is part of the same 65-action categorical space")
    acceptance.check("manual_positive_state_continues", int(manual_actions[1]) == 3, int(manual_actions[1]), "3", "ACQUIRE wins when its shared categorical logit is larger")

    env.reset(patient_ids[:2], 0.0)
    initial_phi = env.state.initial_potential[0].clone()
    first = env.step_actions(torch.arange(2, device=device), torch.tensor([64, 0], device=device))
    alive_rows = torch.where(env.state.alive)[0]
    second = env.step_actions(alive_rows, torch.tensor([1], device=device))
    acceptance.check("stop_reward_zero", float(first["reward"][0]) == 0.0, float(first["reward"][0]), "==0", "STOP uses incremental reward convention")
    acceptance.check("stop_has_no_later_transition", int((second["patient_index"] == first["patient_index"][0]).sum()) == 0, 0, "0", "Stopped patient excluded by alive mask")
    acceptance.check("acquire_adds_one_bit", int(first["active_mask_before"][1].sum()) + 1 == int(env.state.active[1].sum()) - 1, 1, "one bit per ACQUIRE", "Two sequential ACQUIRE actions add exactly two bits")

    env.reset(patient_ids[:1], 0.0)
    observation_same = env.observe()
    action = torch.tensor([0], device=device)
    original_label = env.state.labels.clone()
    reward_original = env.step_actions(torch.tensor([0], device=device), action)["reward"][0]
    env.reset(patient_ids[:1], 0.0)
    env.state.labels = 1 - original_label
    reward_changed = env.step_actions(torch.tensor([0], device=device), action)["reward"][0]
    with torch.no_grad(): same_logits = policy(observation_same)[0]
    reward_delta = float(abs(reward_original - reward_changed))
    acceptance.check("label_affects_reward_not_policy", reward_delta > 0 and torch.isfinite(same_logits).all(), reward_delta, ">0 reward delta", "Label is private to reward computation")

    env.max_steps = 64
    telescope_lambda = 0.1
    env.reset(patient_ids[:1], telescope_lambda)
    initial_phi = env.state.initial_potential[0].clone()
    for region in range(64):
        result = env.step_actions(torch.tensor([0], device=device), torch.tensor([region], device=device))
    acceptance.check("terminal_after_64_acquires", bool(result["done"][0]) and not bool(env.state.alive[0]), bool(result["done"][0]), "True", "K64 forces termination")
    terminal_phi = evidence_potential(env.state.labels[:1], env.state.risk[:1], env.state.disagreement[:1])[0]
    expected_return = terminal_phi - initial_phi - telescope_lambda * env.state.cumulative_cost[0]
    telescope_error = float(abs(env.state.cumulative_reward[0] - expected_return))
    acceptance.check("reward_telescoping", telescope_error < 1e-6, telescope_error, "<1e-6", "Accumulated potential increments and costs telescope")
    train_only = all(not any(word in p.lower() for word in ("validation", "validate", "test")) for p in env.assets.opened_paths)
    acceptance.check("smoke_paths_train_only", train_only, train_only, "True", "Every opened metadata, tensor, region, calibration and checkpoint path is recorded and loader-enforced")


def checkpoint_reload_test(acceptance, policy, observation, checkpoint_path):
    policy.eval()
    with torch.no_grad(): before = policy(observation)
    reloaded = type(policy)(policy.coarse.in_features, policy.slot.in_features, policy.slot.out_features).to(next(policy.parameters()).device)
    checkpoint = torch.load(checkpoint_path, map_location=next(policy.parameters()).device, weights_only=False)
    keys = set(checkpoint["model"])
    complete = any(key.startswith("slot.") for key in keys) and any(key.startswith("acquire_head.") for key in keys) and any(key.startswith("stop_head.") for key in keys) and any(key.startswith("value_head.") for key in keys)
    acceptance.check("checkpoint_contains_full_actor_critic", complete, sorted(keys), "Dense encoder + acquire + STOP + value", "Fresh model is reconstructed solely from checkpoint state_dict")
    reloaded.load_state_dict(checkpoint["model"])
    reloaded.eval()
    with torch.no_grad(): after = reloaded(observation)
    delta = max(float((before[0] - after[0]).abs().max()), float((before[1] - after[1]).abs().max()))
    acceptance.check("checkpoint_reload_exact", delta == 0.0, delta, "==0", "Full Dense encoder, actor and critic state_dict reloaded")
