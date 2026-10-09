"""Partial-evidence gated ABMIL, adapted from Ilse et al. (ICML 2018).

Attention source: AMLab-Amsterdam/AttentionDeepMIL, model.py, MIT,
commit eb0434ba2795711a45d693d60120ae53532b1b93.
UNI is an external frozen patch encoder, not implemented or trained here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EvidenceState:
    maximum: torch.Tensor
    denominator: torch.Tensor
    numerator: torch.Tensor
    patch_count: int


class PartialABMIL(nn.Module):
    def __init__(self, *, input_dim: int = 1024, hidden_dim: int = 256,
                 attention_dim: int = 128, dropout: float = 0.25, prior: float = 0.5):
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError("prior must be strictly between 0 and 1")
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_v = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, 1)
        self.head = nn.Linear(hidden_dim + 1, 1)
        self.register_buffer("prior_logit", torch.logit(torch.tensor(float(prior))))

    def _encode(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.projection(patches)
        scores = self.attention_w(self.attention_v(h) * self.attention_u(h)).squeeze(-1)
        return h, scores

    def forward(self, patches: torch.Tensor, observed: torch.Tensor,
                total_patch_count: torch.Tensor) -> dict[str, torch.Tensor]:
        if patches.ndim != 3 or patches.shape[-1] != self.input_dim:
            raise ValueError("patches must have shape [batch, patches, input_dim]")
        if observed.shape != patches.shape[:2] or observed.dtype != torch.bool:
            raise ValueError("observed must be a boolean [batch, patches] mask")
        if total_patch_count.shape != (len(patches),):
            raise ValueError("one total candidate-bank patch count is required per patient")
        count = observed.sum(dim=1)
        if not torch.isfinite(total_patch_count).all() or torch.any(total_patch_count < count) or torch.any(total_patch_count < 1):
            raise ValueError("total counts must be positive and at least observed counts")
        if not torch.isfinite(patches[observed]).all():
            raise ValueError("observed features must be finite")
        if patches.shape[1] == 0:
            logits = self.prior_logit.expand(len(patches))
            return {"logits": logits, "probabilities": logits.sigmoid(),
                    "attention": patches.new_zeros(patches.shape[:2])}
        # Hidden/padded values are removed before projection, including NaNs.
        safe = torch.where(observed.unsqueeze(-1), patches, torch.zeros_like(patches))
        h, scores = self._encode(safe)
        scores = scores.masked_fill(~observed, torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=1) * observed
        pooled = (attention.unsqueeze(-1) * h).sum(dim=1)
        rho = (count / total_patch_count).to(pooled.dtype).unsqueeze(-1)
        logits = self.head(torch.cat([pooled, rho], dim=-1)).squeeze(-1)
        logits = torch.where(count > 0, logits, self.prior_logit)
        return {"logits": logits, "probabilities": logits.sigmoid(), "attention": attention}

    def empty_state(self) -> EvidenceState:
        parameter = self.head.weight
        return EvidenceState(parameter.new_tensor(-torch.inf), parameter.new_zeros(()),
                             parameter.new_zeros(self.hidden_dim), 0)

    @torch.no_grad()
    def acquire(self, state: EvidenceState, patches: torch.Tensor,
                total_patch_count: int) -> tuple[EvidenceState, torch.Tensor]:
        """Exact incremental attention in eval mode; no acquired patch is recomputed."""
        if self.training:
            raise RuntimeError("incremental acquisition requires eval mode")
        if patches.ndim != 2 or patches.shape[1] != self.input_dim or len(patches) < 1:
            raise ValueError("acquire expects a nonempty [patches, input_dim] batch")
        if not torch.isfinite(patches).all() or state.patch_count + len(patches) > total_patch_count:
            raise ValueError("invalid new evidence or candidate-bank count")
        h, scores = self._encode(patches)
        maximum = torch.maximum(state.maximum, scores.max())
        old_scale = torch.exp(state.maximum - maximum)
        weights = torch.exp(scores - maximum)
        denominator = state.denominator * old_scale + weights.sum()
        numerator = state.numerator * old_scale + (weights.unsqueeze(-1) * h).sum(dim=0)
        count = state.patch_count + len(patches)
        updated = EvidenceState(maximum, denominator, numerator, count)
        rho = numerator.new_tensor([count / total_patch_count])
        probability = self.head(torch.cat([numerator / denominator, rho])).sigmoid().squeeze()
        return updated, probability

    @torch.no_grad()
    def predict_numpy(self, patches, total_patch_count: int) -> float:
        """Adapter for AcquisitionEnv; input contains purchased patches only."""
        if self.training:
            raise RuntimeError("freeze predictor with eval() before using the environment")
        x = torch.as_tensor(patches, dtype=self.head.weight.dtype, device=self.head.weight.device).unsqueeze(0)
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        totals = torch.tensor([total_patch_count], device=x.device)
        return float(self(x, mask, totals)["probabilities"].item())
