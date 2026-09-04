"""Definition-correct ACQUIRE+STOP environment and Dense PPO implementation."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def evidence_potential(label, probability, disagreement):
    label = label.to(torch.float64)
    probability = probability.to(torch.float64).clamp(1e-12, 1 - 1e-12)
    disagreement = disagreement.to(torch.float64)
    bce = -(label * probability.log() + (1 - label) * (1 - probability).log())
    return -bce - 0.25 * (label - probability).square() - 0.05 * disagreement


class DenseActorCritic(nn.Module):
    def __init__(self, coarse_dim: int, slot_dim: int = 256, hidden: int = 192):
        super().__init__()
        self.slot = nn.Linear(slot_dim, hidden)
        self.slot_block = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 384), nn.GELU(), nn.Linear(384, hidden))
        self.coarse = nn.Linear(coarse_dim, hidden)
        self.order = nn.Embedding(65, hidden)
        self.active = nn.Embedding(2, hidden)
        self.scalar = nn.Sequential(nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.region_block = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.acquire_head = nn.Linear(hidden, 1)
        self.stop_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, observation):
        coarse = observation["coarse"]
        active = observation["active_mask"].bool()
        order = observation["acquire_order"].long().clamp(0, 64)
        slots = observation["fine_slots"] * active[..., None, None].to(observation["fine_slots"].dtype)
        present = (slots.norm(dim=-1) > 0).to(slots.dtype)
        slot_mean = (slots * present[..., None]).sum(2) / present.sum(2, keepdim=True).clamp_min(1)
        fine_token = self.slot(slot_mean)
        fine_token = (fine_token + self.slot_block(fine_token)) * active[..., None].to(fine_token.dtype)
        region = self.coarse(coarse) + fine_token + self.order(order) + self.active(active.long())
        observed_summary = fine_token.sum(1) / active.sum(1, keepdim=True).clamp_min(1)
        scalars = torch.stack([
            observation["risk"], observation["entropy"], observation["disagreement"],
            observation["step"].float() / 64.0, observation["cumulative_cost"], observation["lambda"],
            observation["remaining"].float() / 64.0,
        ], -1)
        state = torch.tanh(region.mean(1) + observed_summary + self.scalar(scalars))
        candidate = self.region_block(region + state[:, None])
        acquire = self.acquire_head(candidate).squeeze(-1).masked_fill(active, -1e9)
        stop = self.stop_head(state)
        logits = torch.cat([acquire, stop], 1)
        return logits, self.value_head(state).squeeze(-1)


@dataclass
class BatchState:
    patient_indices: torch.Tensor
    labels: torch.Tensor
    active: torch.Tensor
    order: torch.Tensor
    alive: torch.Tensor
    step: torch.Tensor
    cumulative_cost: torch.Tensor
    risk: torch.Tensor
    entropy: torch.Tensor
    disagreement: torch.Tensor
    initial_potential: torch.Tensor
    cumulative_reward: torch.Tensor


class SequentialAcquireEnv:
    def __init__(self, assets, predictor, device, max_steps=64):
        self.assets, self.predictor, self.device = assets, predictor, torch.device(device)
        self.max_steps = int(max_steps)
        raw = assets.coarse[:, :, 128]
        self.costs = ((raw - raw.amin(1, keepdim=True)) /
                      (raw.amax(1, keepdim=True) - raw.amin(1, keepdim=True)).clamp_min(1e-6)).to(torch.float64)
        self.state = None
        self.lambda_cost = 0.0

    def reset(self, patient_indices, lambda_cost):
        ids = torch.as_tensor(patient_indices, dtype=torch.long, device=self.device)
        labels = torch.tensor(self.assets.metadata.label.values[ids.cpu().numpy()], dtype=torch.float64, device=self.device)
        active = torch.zeros(len(ids), 64, dtype=torch.bool, device=self.device)
        risk, entropy, disagreement, _ = self.predictor.predict(ids, active)
        initial = evidence_potential(labels, risk, disagreement)
        self.lambda_cost = float(lambda_cost)
        self.state = BatchState(ids, labels, active, torch.zeros(len(ids), 64, dtype=torch.long, device=self.device),
                                torch.ones(len(ids), dtype=torch.bool, device=self.device),
                                torch.zeros(len(ids), dtype=torch.long, device=self.device),
                                torch.zeros(len(ids), dtype=torch.float64, device=self.device), risk, entropy, disagreement,
                                initial, torch.zeros(len(ids), dtype=torch.float64, device=self.device))
        return self.observe()

    def observe(self, row_indices=None):
        state = self.state
        rows = torch.where(state.alive)[0] if row_indices is None else torch.as_tensor(row_indices, device=self.device)
        patient = state.patient_indices[rows]
        return {
            "patient_index": patient,
            "coarse": self.assets.coarse[patient.cpu()].to(self.device),
            # Keep the fixed K64 layout, but expose no unobserved fine values.
            "fine_slots": (self.assets.fine_slots[patient.cpu()].to(self.device) *
                           state.active[rows, :, None, None].to(self.assets.fine_slots.dtype)),
            "active_mask": state.active[rows].clone(),
            "acquire_order": state.order[rows].clone(),
            "risk": state.risk[rows].clone(), "entropy": state.entropy[rows].clone(),
            "disagreement": state.disagreement[rows].clone(), "step": state.step[rows].clone(),
            "cumulative_cost": state.cumulative_cost[rows].float(),
            "lambda": torch.full((len(rows),), self.lambda_cost, device=self.device),
            "remaining": (~state.active[rows]).sum(1),
        }

    def step_actions(self, row_indices, actions):
        state = self.state
        rows = torch.as_tensor(row_indices, dtype=torch.long, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        if not bool(state.alive[rows].all()):
            raise AssertionError("Dead patients cannot generate transitions")
        before_active, before_order = state.active[rows].clone(), state.order[rows].clone()
        risk_before, disagreement_before = state.risk[rows].clone(), state.disagreement[rows].clone()
        stop = actions == 64
        acquire_rows = rows[~stop]
        acquire_actions = actions[~stop]
        if acquire_rows.numel() and bool(state.active[acquire_rows, acquire_actions].any()):
            raise AssertionError("An observed region cannot be acquired again")
        if acquire_rows.numel():
            state.active[acquire_rows, acquire_actions] = True
            state.order[acquire_rows, acquire_actions] = state.step[acquire_rows] + 1
            selected_cost = self.costs[state.patient_indices[acquire_rows].cpu(), acquire_actions.cpu()].to(self.device)
            state.cumulative_cost[acquire_rows] += selected_cost
        risk_after, entropy_after, disagreement_after, _ = self.predictor.predict(state.patient_indices[rows], state.active[rows])
        before_phi = evidence_potential(state.labels[rows], risk_before, disagreement_before)
        after_phi = evidence_potential(state.labels[rows], risk_after, disagreement_after)
        costs = torch.zeros(len(rows), dtype=torch.float64, device=self.device)
        if acquire_rows.numel():
            costs[~stop] = self.costs[state.patient_indices[acquire_rows].cpu(), acquire_actions.cpu()].to(self.device)
        reward = torch.where(stop, torch.zeros_like(after_phi), after_phi - before_phi - self.lambda_cost * costs)
        state.cumulative_reward[rows] += reward
        state.risk[rows], state.entropy[rows], state.disagreement[rows] = risk_after, entropy_after, disagreement_after
        state.step[rows] += (~stop).long()
        done = stop | (state.active[rows].sum(1) == 64) | (state.step[rows] >= self.max_steps)
        state.alive[rows[done]] = False
        return {
            "rows": rows, "patient_index": state.patient_indices[rows], "active_mask_before": before_active,
            "order_before": before_order, "action": actions, "reward": reward, "done": done,
            "risk_before": risk_before, "risk_after": risk_after, "cost": costs,
        }


def compute_gae(rewards, values, dones, next_values, gamma=1.0, gae_lambda=0.95):
    rewards, values, dones, next_values = map(torch.as_tensor, (rewards, values, dones, next_values))
    advantage = torch.zeros_like(rewards, dtype=torch.float32)
    running = torch.tensor(0.0, device=rewards.device)
    for index in reversed(range(len(rewards))):
        mask = 1.0 - dones[index].float()
        delta = rewards[index] + gamma * next_values[index] * mask - values[index]
        running = delta + gamma * gae_lambda * mask * running
        advantage[index] = running
    return advantage, advantage + values


def clone_observation(observation):
    return {key: value.detach().cpu().clone() for key, value in observation.items()}


def stack_observations(observations, device):
    return {key: torch.cat([item[key] for item in observations], 0).to(device) for key in observations[0]}


def collect_rollout(policy, env, patient_indices, lambda_cost, minimum_transitions=128, deterministic=False):
    old_policy = copy.deepcopy(policy).to(env.device).eval()
    for parameter in old_policy.parameters():
        parameter.requires_grad_(False)
    if old_policy.training or any(parameter.requires_grad for parameter in old_policy.parameters()):
        raise AssertionError("Old policy snapshot must be eval-mode and frozen")
    transitions = []
    episode_counter = 0
    while len(transitions) < minimum_transitions:
        observation = env.reset(patient_indices, lambda_cost)
        episode_ids = torch.arange(episode_counter, episode_counter + len(patient_indices), device=env.device)
        episode_counter += len(patient_indices)
        while bool(env.state.alive.any()):
            rows = torch.where(env.state.alive)[0]
            observation = env.observe(rows)
            with torch.no_grad():
                logits, value = old_policy(observation)
                distribution = torch.distributions.Categorical(logits=logits)
                action = logits.argmax(1) if deterministic else distribution.sample()
                old_log_prob = distribution.log_prob(action).detach()
            result = env.step_actions(rows, action)
            next_value = torch.zeros(len(rows), device=env.device)
            continuing = ~result["done"]
            if bool(continuing.any()):
                with torch.no_grad():
                    next_value[continuing] = old_policy(env.observe(rows[continuing]))[1]
            for local in range(len(rows)):
                transitions.append({
                    "episode_id": int(episode_ids[rows[local]]), "patient_index": int(result["patient_index"][local]),
                    "step": int(observation["step"][local]), "observation": clone_observation({k: v[local:local+1] for k, v in observation.items()}),
                    "active_mask_before": result["active_mask_before"][local].cpu(), "order_before": result["order_before"][local].cpu(),
                    "action": int(action[local]), "old_log_prob": float(old_log_prob[local]), "value": float(value[local]),
                    "reward": float(result["reward"][local]), "done": bool(result["done"][local]),
                    "next_value": float(next_value[local]), "risk_before": float(result["risk_before"][local]),
                    "risk_after": float(result["risk_after"][local]), "cost": float(result["cost"][local]),
                    "old_policy_frozen": True,
                })
        if len(transitions) >= minimum_transitions:
            break
    for episode in sorted({item["episode_id"] for item in transitions}):
        indexes = [i for i, item in enumerate(transitions) if item["episode_id"] == episode]
        rewards = torch.tensor([transitions[i]["reward"] for i in indexes])
        values = torch.tensor([transitions[i]["value"] for i in indexes])
        dones = torch.tensor([transitions[i]["done"] for i in indexes])
        next_values = torch.tensor([transitions[i]["next_value"] for i in indexes])
        advantages, returns = compute_gae(rewards, values, dones, next_values)
        for j, index in enumerate(indexes):
            transitions[index]["advantage"], transitions[index]["return"] = float(advantages[j]), float(returns[j])
    return transitions


def ppo_update(policy, optimizer, transitions, device, epochs=4, minibatch_size=64, clip=0.2):
    observations = stack_observations([item["observation"] for item in transitions], device)
    actions = torch.tensor([item["action"] for item in transitions], device=device)
    old_log_prob = torch.tensor([item["old_log_prob"] for item in transitions], device=device)
    returns = torch.tensor([item["return"] for item in transitions], device=device)
    advantages = torch.tensor([item["advantage"] for item in transitions], device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    with torch.no_grad():
        initial_ratio = torch.exp(torch.distributions.Categorical(logits=policy(observations)[0]).log_prob(actions) - old_log_prob)
    rows = []
    count = len(actions)
    for epoch in range(epochs):
        permutation = torch.randperm(count, device=device)
        shuffled = not torch.equal(permutation, torch.arange(count, device=device))
        for start in range(0, count, minibatch_size):
            index = permutation[start:start + minibatch_size]
            batch = {key: value[index] for key, value in observations.items()}
            logits, value = policy(batch)
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_prob = distribution.log_prob(actions[index])
            ratio = torch.exp(new_log_prob - old_log_prob[index])
            surrogate = torch.minimum(ratio * advantages[index], ratio.clamp(1 - clip, 1 + clip) * advantages[index])
            actor_loss = -surrogate.mean()
            critic_loss = 0.5 * F.mse_loss(value, returns[index])
            entropy_loss = -distribution.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            encoder_grad = float(policy.slot.weight.grad.norm())
            actor_grad = float(policy.acquire_head.weight.grad.norm())
            value_grad = float(policy.value_head.weight.grad.norm())
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                approx_kl = (old_log_prob[index] - new_log_prob).mean()
                clip_fraction = ((ratio - 1).abs() > clip).float().mean()
            rows.append({"epoch": epoch + 1, "actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                         "entropy_loss": float(entropy_loss.detach()), "approx_kl": float(approx_kl),
                         "clip_fraction": float(clip_fraction), "ratio_mean": float(ratio.mean().detach()),
                         "minibatch_shuffled": shuffled,
                         "encoder_grad": encoder_grad, "actor_grad": actor_grad, "value_grad": value_grad})
    return rows, float(initial_ratio.mean())
