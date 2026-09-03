#!/usr/bin/env python3
"""GPU smoke test for MS-RLSelect morphology-spatial token encoding."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ms_token_encoder import MorphologySpatialTokenEncoder
from models.rlmil import Memory, PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    return parser.parse_args()


def load_batch(manifest: Path, cases: int) -> tuple[torch.Tensor, torch.Tensor, list[list[list[int]]]]:
    frame = pd.read_csv(manifest).head(cases)
    if len(frame) < 2:
        raise ValueError("smoke test requires at least two manifest rows")
    feature_arrays, coordinate_arrays, clusters = [], [], []
    for row in frame.itertuples(index=False):
        with np.load(row.features_filepath) as payload:
            feature_arrays.append(payload["img_features"].astype(np.float32, copy=False))
            coordinate_arrays.append(payload["coords"].astype(np.float32, copy=False))
        clusters.append(json.loads(Path(row.clusters_json_filepath).read_text(encoding="utf-8")))
    feature_dim = feature_arrays[0].shape[1]
    max_patches = max(array.shape[0] for array in feature_arrays)
    features = np.zeros((len(feature_arrays), max_patches, feature_dim), dtype=np.float32)
    coordinates = np.zeros((len(feature_arrays), max_patches, 2), dtype=np.float32)
    for index, (feature_array, coordinate_array) in enumerate(zip(feature_arrays, coordinate_arrays)):
        features[index, : feature_array.shape[0]] = feature_array
        coordinates[index, : coordinate_array.shape[0]] = coordinate_array
    return torch.from_numpy(features), torch.from_numpy(coordinates), clusters


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because the existing PPO implementation uses CUDA tensors")
    torch.manual_seed(1)
    features, coordinates, clusters = load_batch(args.manifest, args.cases)
    device = torch.device("cuda")
    features, coordinates = features.to(device), coordinates.to(device)
    model = MorphologySpatialTokenEncoder(
        input_dim=features.shape[-1],
        token_dim=args.token_dim,
        num_clusters=10,
        num_layers=args.layers,
        num_heads=args.heads,
    ).to(device).eval()
    with torch.no_grad():
        encoded = model(features, coordinates, clusters)
    if not all(torch.isfinite(value).all() for value in encoded.values() if value.is_floating_point()):
        raise RuntimeError("non-finite value found in token encoder output")
    ppo = PPO(
        feature_dim=features.shape[-1],
        state_dim=args.token_dim,
        hidden_state_dim=512,
        policy_conv=False,
        action_std=0.5,
        lr=1e-5,
        gamma=0.1,
        K_epochs=3,
        action_size=10,
    )
    with torch.no_grad():
        actions = ppo.select_action(encoded["global_token"], Memory(), restart_batch=True)
    if actions.shape != (features.shape[0], 10) or not torch.isfinite(actions).all():
        raise RuntimeError(f"unexpected PPO action tensor: {tuple(actions.shape)}")
    report = {
        "status": "PASS",
        "device": str(device),
        "features": list(features.shape),
        "coordinates": list(coordinates.shape),
        "global_token": list(encoded["global_token"].shape),
        "cluster_tokens": list(encoded["cluster_tokens"].shape),
        "active_clusters": encoded["cluster_mask"].sum(dim=1).tolist(),
        "ppo_actions": list(actions.shape),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

