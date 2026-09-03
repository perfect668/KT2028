"""Context-aware region policy for MS-RLSelect.

Unlike the legacy policy, each region action is conditioned on its own
Transformer token and the slide token.  The module keeps the action space
binary (select/skip) and applies a hard budget after sampling.
"""
from __future__ import annotations

import torch
from torch import nn


class MSRegionPolicy(nn.Module):
    def __init__(self, token_dim: int = 256, hidden_dim: int = 256, max_budget: int = 4):
        super().__init__()
        self.max_budget = max_budget
        self.actor = nn.Sequential(
            nn.Linear(token_dim * 2 + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(token_dim + 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def _inputs(self, global_token, region_tokens, spatial_features):
        b, k, _ = region_tokens.shape
        slide = global_token[:, None, :].expand(b, k, -1)
        actor_x = torch.cat((slide, region_tokens, spatial_features), dim=-1)
        pooled = torch.cat((global_token, spatial_features.mean(dim=1)), dim=-1)
        return actor_x, pooled

    def logits_and_value(self, global_token, region_tokens, spatial_features, mask):
        actor_x, pooled = self._inputs(global_token, region_tokens, spatial_features)
        logits = self.actor(actor_x).squeeze(-1).masked_fill(~mask, -torch.inf)
        value = self.value(pooled).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def select(self, global_token, region_tokens, spatial_features, mask, deterministic=False):
        logits, value = self.logits_and_value(global_token, region_tokens, spatial_features, mask)
        dist = torch.distributions.Bernoulli(logits=logits)
        raw = (dist.probs >= 0.5) if deterministic else dist.sample().bool()
        raw &= mask
        # Ensure at least one action and enforce the adaptive budget.  For
        # over-budget samples retain the highest policy probabilities.
        probs = dist.probs.masked_fill(~mask, -torch.inf)
        for i in range(raw.shape[0]):
            if not raw[i].any():
                raw[i, probs[i].argmax()] = True
            if raw[i].sum() > self.max_budget:
                keep = probs[i].masked_fill(~raw[i], -torch.inf).topk(self.max_budget).indices
                chosen = torch.zeros_like(raw[i]); chosen[keep] = True; raw[i] = chosen
        log_prob = dist.log_prob(raw.float()).masked_fill(~mask, 0).sum(-1)
        entropy = dist.entropy().masked_fill(~mask, 0).sum(-1)
        return raw, log_prob, entropy, value

    def evaluate(self, global_token, region_tokens, spatial_features, mask, actions):
        logits, value = self.logits_and_value(global_token, region_tokens, spatial_features, mask)
        dist = torch.distributions.Bernoulli(logits=logits)
        log_prob = dist.log_prob(actions.float()).masked_fill(~mask, 0).sum(-1)
        entropy = dist.entropy().masked_fill(~mask, 0).sum(-1)
        return log_prob, entropy, value

