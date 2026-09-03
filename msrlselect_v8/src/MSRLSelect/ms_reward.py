"""Multi-objective reward used by the MS-RLSelect region policy."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _pairwise_mean_cosine(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = []
    for row, keep in zip(x, mask):
        z = row[keep]
        if z.shape[0] < 2:
            values.append(row.new_tensor(0.0))
        else:
            sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
            values.append(sim[~torch.eye(len(z), dtype=torch.bool, device=x.device)].mean())
    return torch.stack(values)


def spatial_coverage(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalized 2-D bounding-box area of selected region centroids."""
    out = []
    for row, keep in zip(coords, mask):
        z = row[keep]
        if z.shape[0] < 2:
            out.append(row.new_tensor(0.0))
        else:
            span = z.amax(0) - z.amin(0)
            out.append((span[0] * span[1]).clamp(0, 1))
    return torch.stack(out)


def compute_reward(
    confidence: torch.Tensor,
    previous_confidence: torch.Tensor,
    region_tokens: torch.Tensor,
    selected_mask: torch.Tensor,
    spatial_coords: torch.Tensor,
    *,
    diversity_weight: float = 0.10,
    coverage_weight: float = 0.05,
    redundancy_weight: float = 0.10,
    cost_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Return reward and all interpretable components for logging."""
    gain = confidence - previous_confidence
    redundancy = _pairwise_mean_cosine(region_tokens, selected_mask)
    diversity = 1.0 - redundancy
    coverage = spatial_coverage(spatial_coords, selected_mask)
    cost = selected_mask.float().mean(dim=1)
    total = gain + diversity_weight * diversity + coverage_weight * coverage \
        - redundancy_weight * redundancy - cost_weight * cost
    return {"reward": total, "confidence_gain": gain, "diversity": diversity,
            "spatial_coverage": coverage, "redundancy": redundancy, "cost": cost}
