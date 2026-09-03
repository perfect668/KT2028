#!/usr/bin/env python3
"""Stage-1A infrastructure utilities for MS-RLSurv.

This module is audit/infrastructure only: it builds deterministic region maps,
does not access labels, and contains no optimizer or training loop.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ALGORITHM = "morphology_cluster_recursive_spatial_bisection"
VERSION = "v5-stage1a-1"


def source_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _allocate_counts(cluster_sizes: np.ndarray, total: int) -> np.ndarray:
    sizes = np.asarray(cluster_sizes, dtype=np.int64)
    nonempty = sizes > 0
    if total < int(nonempty.sum()):
        raise ValueError("total regions is smaller than the number of non-empty clusters")
    out = np.zeros_like(sizes)
    out[nonempty] = 1
    remaining = total - int(nonempty.sum())
    if remaining == 0:
        return out
    weights = sizes.astype(np.float64)
    weights[~nonempty] = 0.0
    raw = remaining * weights / weights.sum()
    floors = np.floor(raw).astype(np.int64)
    out += floors
    left = remaining - int(floors.sum())
    order = sorted(range(len(sizes)), key=lambda i: (-(raw[i] - floors[i]), -sizes[i], i))
    for i in order[:left]:
        if nonempty[i]:
            out[i] += 1
    if int(out.sum()) != total:
        raise AssertionError("largest-remainder allocation failed")
    return out


def _split_indices(indices: np.ndarray, coords: np.ndarray, parts: int) -> list[np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if parts <= 1:
        return [np.sort(indices)]
    if len(indices) < parts:
        raise ValueError("cannot create non-empty regions with fewer patches than parts")
    pts = coords[indices]
    axis = int(np.argmax(np.ptp(pts, axis=0)))
    order = np.lexsort((indices, pts[:, 1], pts[:, 0]) if axis == 0 else (indices, pts[:, 0], pts[:, 1]))
    ordered = indices[order]
    cuts = np.floor(np.arange(parts + 1) * len(ordered) / parts).astype(int)
    return [np.sort(ordered[cuts[i] : cuts[i + 1]]) for i in range(parts)]


def _recursive_partition(indices: np.ndarray, coords: np.ndarray, parts: int) -> list[np.ndarray]:
    if parts == 1:
        return [np.sort(indices)]
    left_parts = parts // 2
    right_parts = parts - left_parts
    split = _split_indices(indices, coords, 2)
    return _recursive_partition(split[0], coords, left_parts) + _recursive_partition(split[1], coords, right_parts)


def build_region_map(features: np.ndarray, coords: np.ndarray, clusters: Iterable[Iterable[int]], num_regions: int) -> dict[str, np.ndarray]:
    """Build a deterministic, exhaustive region partition for one case."""
    features = np.asarray(features)
    coords = np.asarray(coords, dtype=np.float32)
    if features.ndim != 2 or coords.shape != (len(features), 2):
        raise ValueError("features/coords shape mismatch")
    clean = [np.unique(np.asarray(c, dtype=np.int64)) for c in clusters]
    clean = [c[(c >= 0) & (c < len(features))] for c in clean]
    sizes = np.asarray([len(c) for c in clean], dtype=np.int64)
    alloc = _allocate_counts(sizes, int(num_regions))
    regions = []
    for cid, (idx, count) in enumerate(zip(clean, alloc)):
        if count:
            for sub in _recursive_partition(idx, coords, int(count)):
                center = coords[sub].mean(axis=0)
                regions.append((cid, float(center[0]), float(center[1]), sub))
    regions.sort(key=lambda x: (x[0], x[1], x[2], int(x[3][0])))
    region_patch_indices = np.concatenate([x[3] for x in regions]).astype(np.int64)
    ptr = np.zeros(len(regions) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(x[3]) for x in regions])
    patch_to_region = np.full(len(features), -1, dtype=np.int64)
    for rid, (_, _, _, idx) in enumerate(regions):
        if np.any(patch_to_region[idx] != -1):
            raise AssertionError("patch assigned to multiple regions")
        patch_to_region[idx] = rid
    if np.any(patch_to_region < 0) or len(np.unique(region_patch_indices)) != len(features):
        raise AssertionError("region partition is not exhaustive")
    return {"patch_to_region": patch_to_region, "region_ptr": ptr, "region_patch_indices": region_patch_indices,
            "region_morph_cluster": np.asarray([x[0] for x in regions], dtype=np.int64)}


def stratified_patch_indices(features: np.ndarray, coords: np.ndarray, clusters: Iterable[Iterable[int]], budget: int, seed: int = 0) -> np.ndarray:
    """Deterministic morphology-stratified spatial-coverage sampling."""
    n = len(features)
    if n <= budget:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    clean = [np.unique(np.asarray(c, dtype=np.int64)) for c in clusters]
    clean = [c[(c >= 0) & (c < n)] for c in clean]
    sizes = np.asarray([len(c) for c in clean])
    alloc = _allocate_counts(sizes, budget)
    selected = []
    for c, k in zip(clean, alloc):
        if k == 0 or len(c) == 0:
            continue
        pts = coords[c]
        order = np.lexsort((c, pts[:, 1], pts[:, 0]))
        ordered = c[order]
        # Evenly cover the coordinate-sorted cluster; tie-break with seeded rotation.
        offset = int(rng.integers(0, max(1, len(ordered))))
        positions = np.floor((np.arange(k) + 0.5) * len(ordered) / k).astype(int)
        positions = (positions + offset) % len(ordered)
        selected.extend(ordered[np.sort(positions)].tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def build_region_cache_case(features_path: str | Path, clusters_json_path: str | Path, coords_path: str | Path, out_path: str | Path, num_regions: int) -> None:
    with np.load(features_path) as z:
        features = z["img_features"].astype(np.float32)
        coords = z["coords"].astype(np.float32)
    clusters = json.loads(Path(clusters_json_path).read_text())
    region = build_region_map(features, coords, clusters, num_regions)
    ptr = region["region_ptr"]
    means, vars_, centroids, bbox, sizes, spreads, density = [], [], [], [], [], [], []
    norm = (coords - coords.min(0)) / np.maximum(coords.max(0) - coords.min(0), 1.0)
    for i in range(num_regions):
        idx = region["region_patch_indices"][ptr[i] : ptr[i + 1]]
        x, xy = features[idx], norm[idx]
        means.append(x.mean(0)); vars_.append(x.var(0)); centroids.append(xy.mean(0)); bbox.append(xy.max(0) - xy.min(0)); sizes.append(len(idx)); spreads.append(xy.std(0)); density.append(len(idx) / len(features))
    np.savez_compressed(out_path, **region, region_feature_mean=np.asarray(means, np.float32), region_feature_variance=np.asarray(vars_, np.float32), region_centroid=np.asarray(centroids, np.float32), region_bbox=np.asarray(bbox, np.float32), region_size=np.asarray(sizes, np.int32), region_spread=np.asarray(spreads, np.float32), region_density=np.asarray(density, np.float32), raw_patch_count=np.asarray([len(features)], np.int32), features_sha256=source_sha256(features_path), coords_sha256=source_sha256(coords_path), clusters_sha256=source_sha256(clusters_json_path), algorithm=ALGORITHM, version=VERSION, num_regions=np.asarray([num_regions], np.int32), seed=np.asarray([0], np.int32))
