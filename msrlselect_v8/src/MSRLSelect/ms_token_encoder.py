"""Morphology-spatial region token encoder for MS-RLSelect.

The module consumes frozen UNI patch embeddings, patch coordinates, and the
existing per-slide morphology clusters.  It produces one contextual token per
cluster and one slide-level global token.  It deliberately does not alter the
UNI encoder or the existing HGNN/PPO baseline.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MorphologySpatialTokenEncoder(nn.Module):
    """Build contextual morphology-spatial tokens from a batch of WSI patches.

    Args:
        input_dim: Dimension of frozen patch embeddings (1024 for UNI).
        token_dim: Internal region-token dimension.
        num_clusters: Maximum number of morphology clusters per slide.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        dropout: Transformer/projection dropout.

    Input ``cluster_indices`` is a nested Python sequence shaped
    ``batch -> cluster -> patch index``. Empty clusters are valid and masked.
    """

    spatial_dim = 5  # normalized centroid (x, y), spread (x, y), density

    def __init__(
        self,
        input_dim: int = 1024,
        token_dim: int = 256,
        num_clusters: int = 10,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if num_clusters < 1:
            raise ValueError("num_clusters must be positive")

        self.input_dim = input_dim
        self.token_dim = token_dim
        self.num_clusters = num_clusters
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_projection = nn.Sequential(
            nn.Linear(self.spatial_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.global_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.global_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(token_dim)

    @staticmethod
    def _normalise_coordinates(coords: torch.Tensor) -> torch.Tensor:
        minimum = coords.amin(dim=1, keepdim=True)
        maximum = coords.amax(dim=1, keepdim=True)
        scale = (maximum - minimum).clamp_min(1.0)
        return (coords - minimum) / scale

    def _aggregate_clusters(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        cluster_indices: Sequence[Sequence[Sequence[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, patches, _ = features.shape
        device = features.device
        dtype = features.dtype
        cluster_features = features.new_zeros(batch_size, self.num_clusters, self.input_dim)
        spatial_features = features.new_zeros(batch_size, self.num_clusters, self.spatial_dim)
        cluster_mask = torch.zeros(batch_size, self.num_clusters, device=device, dtype=torch.bool)
        normalised_coords = self._normalise_coordinates(coords.to(dtype=dtype))

        if len(cluster_indices) != batch_size:
            raise ValueError(
                f"cluster_indices batch size {len(cluster_indices)} does not match features {batch_size}"
            )

        for batch_idx, per_slide_clusters in enumerate(cluster_indices):
            if len(per_slide_clusters) > self.num_clusters:
                raise ValueError(
                    f"slide {batch_idx} has {len(per_slide_clusters)} clusters; "
                    f"expected at most {self.num_clusters}"
                )
            for cluster_idx, raw_indices in enumerate(per_slide_clusters):
                if not raw_indices:
                    continue
                indices = torch.as_tensor(raw_indices, device=device, dtype=torch.long)
                indices = indices[(indices >= 0) & (indices < patches)].unique(sorted=True)
                if indices.numel() == 0:
                    continue
                selected_features = features[batch_idx, indices]
                selected_coords = normalised_coords[batch_idx, indices]
                cluster_features[batch_idx, cluster_idx] = selected_features.mean(dim=0)
                centroid = selected_coords.mean(dim=0)
                spread = selected_coords.std(dim=0, unbiased=False)
                density = torch.tensor(
                    [indices.numel() / float(patches)], device=device, dtype=dtype
                )
                spatial_features[batch_idx, cluster_idx] = torch.cat(
                    (centroid, spread, density), dim=0
                )
                cluster_mask[batch_idx, cluster_idx] = True

        return cluster_features, spatial_features, cluster_mask

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        cluster_indices: Sequence[Sequence[Sequence[int]]],
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"features must be [batch, patches, dim], got {tuple(features.shape)}")
        if coords.ndim != 3 or coords.shape[:2] != features.shape[:2] or coords.shape[-1] != 2:
            raise ValueError(f"coords must be [batch, patches, 2], got {tuple(coords.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"expected feature dim {self.input_dim}, got {features.shape[-1]}")

        cluster_features, spatial_features, cluster_mask = self._aggregate_clusters(
            features, coords, cluster_indices
        )
        cluster_tokens = self.token_norm(
            self.feature_projection(cluster_features) + self.spatial_projection(spatial_features)
        )
        global_token = self.global_token.expand(features.shape[0], -1, -1)
        tokens = torch.cat((global_token, cluster_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(features.shape[0], 1, dtype=torch.bool, device=features.device),
                ~cluster_mask,
            ),
            dim=1,
        )
        encoded = self.output_norm(self.context_encoder(tokens, src_key_padding_mask=padding_mask))
        return {
            "global_token": encoded[:, 0],
            "cluster_tokens": encoded[:, 1:],
            "cluster_mask": cluster_mask,
            "spatial_features": spatial_features,
        }

