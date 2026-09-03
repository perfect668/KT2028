"""Minimal trainable MS-RLSelect model.

This is the P1 integration point: morphology-spatial region tokens are
contextualised by a Transformer and aggregated with a learned gated attention
pool before the case-level predictor.  It is intentionally independent from
the legacy HeMiCoRe training script so the baseline remains reproducible.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.ms_token_encoder import MorphologySpatialTokenEncoder


class MSRLSelectModel(nn.Module):
    """Case-level predictor over morphology-spatial region tokens.

    ``forward`` returns logits and an auxiliary dictionary containing the
    region attention weights.  ``topk_regions`` is a deterministic selection
    primitive used by the upcoming PPO policy and by qualitative trace export.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        token_dim: int = 256,
        num_clusters: int = 10,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.encoder = MorphologySpatialTokenEncoder(
            input_dim=input_dim,
            token_dim=token_dim,
            num_clusters=num_clusters,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.gate = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.Tanh(),
            nn.Linear(token_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        cluster_indices: Sequence[Sequence[Sequence[int]]],
        region_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encoder(features, coords, cluster_indices)
        tokens = encoded["cluster_tokens"]
        mask = encoded["cluster_mask"]
        if region_mask is not None:
            if region_mask.shape != mask.shape:
                raise ValueError(f"region_mask must be {tuple(mask.shape)}, got {tuple(region_mask.shape)}")
            mask = mask & region_mask.bool()
        scores = self.gate(tokens).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        # A slide always has at least one valid morphology cluster.  The
        # denominator guard keeps the method well-defined for malformed input.
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        fused = torch.cat((encoded["global_token"], pooled), dim=-1)
        logits = self.classifier(fused)
        aux = {
            "global_token": encoded["global_token"],
            "cluster_tokens": tokens,
            "cluster_mask": encoded["cluster_mask"],
            "attention": weights,
            "spatial_features": encoded["spatial_features"],
        }
        return logits, aux

    @torch.no_grad()
    def topk_regions(self, attention: torch.Tensor, mask: torch.Tensor, budget: int) -> torch.Tensor:
        """Return a boolean `[B,K]` region selection under a fixed budget."""
        if attention.ndim != 2 or mask.shape != attention.shape:
            raise ValueError("attention and mask must both be [batch, clusters]")
        budget = max(1, min(int(budget), attention.shape[1]))
        scores = attention.masked_fill(~mask, -torch.inf)
        indices = scores.topk(k=budget, dim=1).indices
        selected = torch.zeros_like(mask)
        selected.scatter_(1, indices, True)
        return selected & mask
