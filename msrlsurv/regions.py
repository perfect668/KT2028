"""Spatial region construction uses coordinates only, without UNI features."""

from __future__ import annotations

import numpy as np


def spatial_regions(coords: np.ndarray, count: int) -> list[np.ndarray]:
    """Split a slide into nonempty, disjoint regions by spatial median cuts."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite [patches, 2]")
    if len(coords) == 0 or not 1 <= count <= len(coords):
        raise ValueError("region count must be between 1 and patch count")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("duplicate coordinates require source-data reconciliation")
    groups = [np.arange(len(coords))]
    while len(groups) < count:
        eligible = [i for i, group in enumerate(groups) if len(group) > 1]
        ix = max(eligible, key=lambda i: (len(groups[i]), -i))
        group = groups.pop(ix)
        positions = coords[group]
        axis = int(np.argmax(np.ptp(positions, axis=0)))
        order = np.lexsort((group, positions[:, 1 - axis], positions[:, axis]))
        ordered = group[order]
        middle = len(ordered) // 2
        groups.extend([ordered[:middle], ordered[middle:]])
    groups.sort(key=lambda g: tuple(coords[g].mean(axis=0)))
    return groups


def representative_patches(coords: np.ndarray, group: np.ndarray, limit: int = 32) -> np.ndarray:
    """Deterministic spatial coverage sampling; no label or feature access."""
    if limit < 1 or len(group) == 0:
        raise ValueError("limit and group must be positive")
    group = np.asarray(group, dtype=np.int64)
    positions = np.asarray(coords, dtype=np.float64)[group]
    order = np.lexsort((group, positions[:, 1], positions[:, 0]))
    group, positions = group[order], positions[order]
    center_dist = ((positions - positions.mean(axis=0)) ** 2).sum(axis=1)
    first = int(np.argmin(center_dist))
    selected = [first]
    distances = ((positions - positions[first]) ** 2).sum(axis=1)
    distances[first] = -1
    while len(selected) < min(limit, len(group)):
        nxt = int(np.argmax(distances))
        selected.append(nxt)
        distances = np.minimum(distances, ((positions - positions[nxt]) ** 2).sum(axis=1))
        distances[selected] = -1
    return group[selected]


def allocate_regions(patch_counts: list[int], limit: int = 64) -> list[int]:
    """Allocate a patient-wide cap across slides, retaining every slide."""
    counts = np.asarray(patch_counts, dtype=np.int64)
    if len(counts) == 0 or np.any(counts < 1) or limit < len(counts):
        raise ValueError("each nonempty slide needs at least one region")
    target = min(limit, int(counts.sum()))
    allocation = np.ones(len(counts), dtype=np.int64)
    # Repeated weighted apportionment also respects each slide's patch count.
    while allocation.sum() < target:
        priority = counts / allocation
        priority[allocation >= counts] = -1
        allocation[int(np.argmax(priority))] += 1
    return allocation.tolist()
