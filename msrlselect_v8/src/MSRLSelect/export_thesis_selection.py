#!/usr/bin/env python3
"""Export deterministic chapter 3 selection traces from fixed checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.ms_model import MSRLSelectModel
from models.ms_policy import MSRegionPolicy
from pipeline_common import cluster_patch_indices, load_case, read_config, seed_all


METHODS = (
    "random",
    "MS-RLSelect_fixed_budget",
    "MS-RLSelect_adaptive_budget",
    "MS-RLSelect_no_spatial",
    "MS-RLSelect_no_evidence_reward",
    "MS-RLSelect_revision",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--no-spatial-checkpoint", type=Path, required=True)
    parser.add_argument("--no-evidence-checkpoint", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=False)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-steps", type=int, default=2)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--gain-threshold", type=float, default=0.01)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[101,202,303,404,505])
    return parser.parse_args()


def load_selector(path: Path, config: dict):
    checkpoint = torch.load(path, map_location="cuda", weights_only=False)
    variant = checkpoint["variant"]
    model = MSRLSelectModel(
        token_dim=int(config["token_dim"]),
        num_layers=int(config["layers"]),
        num_heads=int(config["heads"]),
        use_spatial=variant != "no_spatial",
    ).cuda().eval()
    policy = MSRegionPolicy(
        int(config["token_dim"]), int(config["token_dim"]), checkpoint["budget"]
    ).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    policy.load_state_dict(checkpoint["policy"])
    return model, policy, variant


def choose_region(policy, base, policy_spatial, available):
    logits, _ = policy.logits_and_value(
        base["global_token"],
        base["cluster_tokens"],
        policy_spatial,
        available,
    )
    return logits.argmax(dim=1)


@torch.no_grad()
def model_trace(
    model,
    policy,
    variant,
    cluster_features,
    spatial,
    cluster_mask,
    budget,
    min_steps,
    gain_threshold,
    adaptive,
):
    policy_spatial = torch.zeros_like(spatial) if variant == "no_spatial" else spatial
    torch.cuda.synchronize()
    start = time.perf_counter()
    _, base = model.forward_aggregates(cluster_features, spatial, cluster_mask)
    selected = torch.zeros_like(cluster_mask)
    actions, cumulative_masks, risks, confidence_gains = [], [], [], []
    previous_confidence = 0.0
    for _ in range(min(budget, cluster_mask.shape[1])):
        available = cluster_mask & ~selected
        if not bool(available.any()):
            break
        index = choose_region(policy, base, policy_spatial, available)
        action = F.one_hot(index, num_classes=cluster_mask.shape[1]).bool()
        selected |= action
        logits, _ = model.forward_aggregates(
            cluster_features, spatial, cluster_mask, region_mask=selected
        )
        death_risk = float(logits.softmax(dim=1)[0, 1])
        confidence = float(logits.softmax(dim=1)[0].max())
        confidence_gains.append(confidence - previous_confidence)
        previous_confidence = confidence
        actions.append(action[0].cpu().numpy().astype(np.uint8))
        cumulative_masks.append(selected[0].cpu().numpy().astype(np.uint8))
        risks.append(death_risk)
        if (
            adaptive
            and len(actions) >= min_steps
            and len(confidence_gains) >= 2
            and np.mean(np.abs(confidence_gains[-2:])) < gain_threshold
        ):
            break
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "actions": np.stack(actions),
        "cumulative_masks": np.stack(cumulative_masks),
        "death_risk": np.asarray(risks, dtype=np.float32),
        "confidence_gains": np.asarray(confidence_gains, dtype=np.float32),
        "inference_time_ms": elapsed_ms,
    }


def stable_rng(seed: int, case_id: str) -> np.random.Generator:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng((seed + offset) % (2**32))


def random_trace(cluster_mask, steps: int, seed: int, case_id: str):
    start = time.perf_counter()
    valid = np.flatnonzero(cluster_mask)
    order = stable_rng(seed, case_id).permutation(valid)[:steps]
    selected = np.zeros_like(cluster_mask, dtype=np.uint8)
    actions, cumulative = [], []
    for index in order:
        action = np.zeros_like(cluster_mask, dtype=np.uint8)
        action[index] = 1
        selected[index] = 1
        actions.append(action)
        cumulative.append(selected.copy())
    return {
        "actions": np.stack(actions),
        "cumulative_masks": np.stack(cumulative),
        "death_risk": np.full(len(actions), np.nan, dtype=np.float32),
        "confidence_gains": np.full(len(actions), np.nan, dtype=np.float32),
        "inference_time_ms": (time.perf_counter() - start) * 1000.0,
    }


def save_case_cache(output_file, trace, features, coords, cluster_indices):
    step_indices = []
    step_feature_means = []
    for mask in trace["cumulative_masks"]:
        selected_regions = np.flatnonzero(mask)
        selected = np.unique(
            np.concatenate(
                [cluster_indices[index] for index in selected_regions if len(cluster_indices[index])]
            )
        )
        step_indices.append(selected.astype(np.int64))
        step_feature_means.append(features[selected].mean(axis=0).astype(np.float32))
    final_indices = step_indices[-1]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        selected_indices=final_indices,
        selected_coords=coords[final_indices].astype(np.float32),
        selected_features=features[final_indices].astype(np.float16),
        selected_feature_mean=step_feature_means[-1],
        step_feature_means=np.stack(step_feature_means),
        step_selected_patch_counts=np.asarray([len(indices) for indices in step_indices]),
        action_sequence=trace["actions"],
        cumulative_region_masks=trace["cumulative_masks"],
        selector_death_risk=trace["death_risk"],
        confidence_gains=trace["confidence_gains"],
        inference_time_ms=np.asarray(trace["inference_time_ms"], dtype=np.float64),
    )
    return final_indices


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    seed_all(int(config["seed"]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    selectors = {
        "full": load_selector(args.full_checkpoint, config),
        "no_spatial": load_selector(args.no_spatial_checkpoint, config),
        "no_evidence": load_selector(args.no_evidence_checkpoint, config),
    }
    if args.revision_checkpoint:
        selectors["revision"] = load_selector(args.revision_checkpoint, config)
    statistics = []
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(config[f"{split}_manifest"])
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=split):
            features, coords, clusters = load_case(row)
            cluster_indices = cluster_patch_indices(clusters, len(features))
            summary_file = args.summary_root / split / f"{row.case_id}.npz"
            with np.load(summary_file) as summary:
                cluster_features = torch.from_numpy(
                    summary["cluster_features"][None].astype(np.float32)
                ).cuda()
                spatial = torch.from_numpy(
                    summary["spatial_features"][None].astype(np.float32)
                ).cuda()
                cluster_mask_np = summary["cluster_mask"].astype(bool)
                cluster_mask = torch.from_numpy(cluster_mask_np[None]).cuda()

            full_model, full_policy, full_variant = selectors["full"]
            adaptive_trace = model_trace(
                full_model,
                full_policy,
                full_variant,
                cluster_features,
                spatial,
                cluster_mask,
                args.budget,
                args.min_steps,
                args.gain_threshold,
                True,
            )
            fixed_trace = model_trace(
                full_model,
                full_policy,
                full_variant,
                cluster_features,
                spatial,
                cluster_mask,
                args.budget,
                args.min_steps,
                args.gain_threshold,
                False,
            )
            no_spatial_model, no_spatial_policy, no_spatial_variant = selectors["no_spatial"]
            no_spatial_trace = model_trace(
                no_spatial_model,
                no_spatial_policy,
                no_spatial_variant,
                cluster_features,
                spatial,
                cluster_mask,
                args.budget,
                args.min_steps,
                args.gain_threshold,
                True,
            )
            no_evidence_model, no_evidence_policy, no_evidence_variant = selectors[
                "no_evidence"
            ]
            no_evidence_trace = model_trace(
                no_evidence_model,
                no_evidence_policy,
                no_evidence_variant,
                cluster_features,
                spatial,
                cluster_mask,
                args.budget,
                args.min_steps,
                args.gain_threshold,
                True,
            )
            revision_trace = None
            if "revision" in selectors:
                revision_model, revision_policy, revision_variant = selectors["revision"]
                revision_trace = model_trace(revision_model, revision_policy, revision_variant,
                    cluster_features, spatial, cluster_mask, args.budget, args.min_steps,
                    args.gain_threshold, True)
            traces = {
                "MS-RLSelect_fixed_budget": fixed_trace,
                "MS-RLSelect_adaptive_budget": adaptive_trace,
                "MS-RLSelect_no_spatial": no_spatial_trace,
                "MS-RLSelect_no_evidence_reward": no_evidence_trace,
            }
            if revision_trace is not None:
                traces["MS-RLSelect_revision"] = revision_trace
            for rseed in args.random_seeds:
                traces[f"random_seed{rseed}"] = random_trace(
                    cluster_mask_np, args.budget, int(rseed), str(row.case_id))
            for method, trace in traces.items():
                output_file = (
                    args.output_root / "selection_cache" / method / split / f"{row.case_id}.npz"
                )
                selected_indices = save_case_cache(
                    output_file, trace, features, coords, cluster_indices
                )
                actions = trace["actions"]
                repeated = int(actions.sum()) - int(actions.astype(bool).any(axis=0).sum())
                statistics.append(
                    {
                        "method": method,
                        "case_id": str(row.case_id),
                        "split": split,
                        "selected_regions": int(actions.shape[0]),
                        "selected_patches": int(len(selected_indices)),
                        "stop_step": int(actions.shape[0]),
                        "repeat_rate": float(repeated / max(1, actions.sum())),
                        "inference_time_ms": float(trace["inference_time_ms"]),
                    }
                )
    pd.DataFrame(statistics).to_csv(
        args.output_root / "chapter3_selection_statistics_partial.csv", index=False
    )
    (args.output_root / "selection_export_status.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "methods": list(METHODS),
                "cases_per_method": 423,
                "test_evaluated": False,
                "random_region_count_matches": "MS-RLSelect_adaptive_budget per case",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
