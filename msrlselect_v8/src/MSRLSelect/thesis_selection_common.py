#!/usr/bin/env python3
"""Shared region-summary data path for the chapter 3 experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline_common import cluster_patch_indices, load_case


def summarize_case(row, num_clusters: int = 10) -> dict[str, np.ndarray]:
    features, coords, clusters = load_case(row)
    cluster_indices = cluster_patch_indices(clusters, len(features))
    cluster_features = np.zeros((num_clusters, features.shape[1]), dtype=np.float32)
    spatial_features = np.zeros((num_clusters, 5), dtype=np.float32)
    cluster_mask = np.zeros(num_clusters, dtype=bool)
    minimum = coords.min(axis=0, keepdims=True)
    scale = np.maximum(coords.max(axis=0, keepdims=True) - minimum, 1.0)
    normalized = (coords - minimum) / scale
    for cluster_id, indices in enumerate(cluster_indices[:num_clusters]):
        if not len(indices):
            continue
        cluster_features[cluster_id] = features[indices].mean(axis=0)
        selected_coords = normalized[indices]
        spatial_features[cluster_id, :2] = selected_coords.mean(axis=0)
        spatial_features[cluster_id, 2:4] = selected_coords.std(axis=0)
        spatial_features[cluster_id, 4] = len(indices) / float(len(features))
        cluster_mask[cluster_id] = True
    return {
        "cluster_features": cluster_features,
        "spatial_features": spatial_features,
        "cluster_mask": cluster_mask,
    }


def ensure_summary_cache(config: dict, output_root: Path) -> dict[str, Path]:
    output_root = Path(output_root)
    manifests = {}
    for split in ("train", "valid", "test"):
        manifest_path = Path(config[f"{split}_manifest"])
        frame = pd.read_csv(manifest_path)
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for row in frame.itertuples(index=False):
            output_file = split_dir / f"{row.case_id}.npz"
            if not output_file.exists():
                summary = summarize_case(row)
                np.savez_compressed(output_file, **summary)
            records.append(
                {
                    "case_id": str(row.case_id),
                    "label": int(row.label),
                    "summary_file": str(output_file),
                }
            )
        cached_manifest = output_root / f"{split}_manifest.csv"
        pd.DataFrame(records).to_csv(cached_manifest, index=False)
        manifests[split] = cached_manifest
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "split_counts": {
                    split: int(len(pd.read_csv(path))) for split, path in manifests.items()
                },
                "feature_definition": "mean UNI embedding per morphology cluster",
                "spatial_definition": "normalized centroid, spread, and patch density",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifests


class RegionSummaryDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path):
        self.frame = pd.read_csv(manifest)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with np.load(row.summary_file) as data:
            cluster_features = data["cluster_features"].astype(np.float32, copy=True)
            spatial_features = data["spatial_features"].astype(np.float32, copy=True)
            cluster_mask = data["cluster_mask"].astype(bool, copy=True)
        return (
            torch.from_numpy(cluster_features),
            torch.from_numpy(spatial_features),
            torch.from_numpy(cluster_mask),
            torch.tensor(int(row.label), dtype=torch.long),
            str(row.case_id),
        )
