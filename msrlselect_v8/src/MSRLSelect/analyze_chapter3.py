#!/usr/bin/env python3
"""Aggregate the completed chapter-3 selection experiments without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from train_selection_predictor import CommonPredictor


def safe_name(method: str) -> str:
    return method + "_"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    methods = [
        "random", "HeMiCoRe", "MS-RLSelect_fixed_budget",
        "MS-RLSelect_adaptive_budget", "MS-RLSelect_no_spatial",
        "MS-RLSelect_no_evidence_reward",
    ]
    stats = [pd.read_csv(root / "chapter3_selection_statistics_partial.csv")]
    hemi_stats = root / "hemicore_selection_statistics.csv"
    if hemi_stats.exists():
        stats.append(pd.read_csv(hemi_stats))
    detail = pd.concat(stats, ignore_index=True)
    aggregate = detail.groupby("method", as_index=False).agg(
        selected_regions_mean=("selected_regions", "mean"),
        selected_patches_mean=("selected_patches", "mean"),
        stop_step_mean=("stop_step", "mean"),
        repeat_rate_mean=("repeat_rate", "mean"),
        inference_time_ms_mean=("inference_time_ms", "mean"),
    )
    aggregate.to_csv(root / "chapter3_selection_statistics.csv", index=False)

    result_rows, step_rows = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for method in methods:
        predictor_dirs = list((root / "predictors").glob("*"))
        predictor_dir = next(
            directory for directory in predictor_dirs
            if (directory / "run_status.json").exists()
            and json.loads((directory / "run_status.json").read_text())["method"] == method
        )
        checkpoint = torch.load(predictor_dir / "model_best.pt", map_location=device, weights_only=False)
        model = CommonPredictor().to(device).eval()
        model.load_state_dict(checkpoint["model"])
        metrics = json.loads((predictor_dir / "metrics_final.json").read_text())
        result_rows.append({
            "method": method,
            "validation_auc": metrics["valid"]["fixed_0_5"]["auc"],
            "validation_auprc": metrics["valid"]["fixed_0_5"]["auprc"],
            "test_auc": metrics["test"]["fixed_0_5"]["auc"],
            "test_auprc": metrics["test"]["fixed_0_5"]["auprc"],
            "test_auc_validation_youden": metrics["test"]["validation_youden"]["auc"],
            "test_auprc_validation_youden": metrics["test"]["validation_youden"]["auprc"],
            "prediction_dir": str(predictor_dir),
        })
        for split in ("valid", "test"):
            files = sorted((root / "selection_cache" / method / split).glob("*.npz"))
            max_steps = max(len(np.load(path)["step_feature_means"]) for path in files)
            for step in range(1, max_steps + 1):
                labels, risks = [], []
                for path in files:
                    case_id = path.stem
                    prediction = pd.read_csv(predictor_dir / f"{split}_predictions.csv")
                    label = int(prediction.loc[prediction.case_id == case_id, "label"].iloc[0])
                    with np.load(path) as data:
                        vectors = data["step_feature_means"]
                        vector = vectors[min(step - 1, len(vectors) - 1)]
                    with torch.no_grad():
                        risk = float(model(torch.from_numpy(vector[None]).to(device)).softmax(1)[0, 1])
                    labels.append(label); risks.append(risk)
                step_rows.append({
                    "method": method, "split": split, "decision_step": step,
                    "auc": float(roc_auc_score(labels, risks)),
                    "auprc": float(average_precision_score(labels, risks)),
                })
    pd.DataFrame(result_rows).to_csv(root / "chapter3_results.csv", index=False)
    pd.DataFrame(step_rows).to_csv(root / "chapter3_step_metrics.csv", index=False)

    # Compact coordinate/action/probability views for a representative test case.
    figure_dir = root / "visualizations"; figure_dir.mkdir(exist_ok=True)
    case_id = sorted((root / "selection_cache" / "MS-RLSelect_adaptive_budget" / "test").glob("*.npz"))[0].stem
    for method in methods:
        path = root / "selection_cache" / method / "test" / f"{case_id}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            coords = data["selected_coords"]
            actions = data["action_sequence"]
            risk = data.get("selector_death_risk", np.array([]))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(coords[:, 0], coords[:, 1], s=2, c=np.arange(len(coords)), cmap="viridis")
        axes[0].set_title(f"{method}: selected patch coordinates")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
        axes[1].imshow(actions.T, aspect="auto", cmap="viridis")
        axes[1].set_title("action sequence")
        axes[1].set_xlabel("decision step"); axes[1].set_ylabel("region index")
        fig.tight_layout(); fig.savefig(figure_dir / f"{case_id}_{method}_selection.png", dpi=160); plt.close(fig)
        if len(risk) and np.isfinite(risk).any():
            plt.figure(figsize=(5, 3)); plt.plot(np.arange(1, len(risk) + 1), risk, marker="o")
            plt.xlabel("decision step"); plt.ylabel("death risk"); plt.title(method); plt.tight_layout()
            plt.savefig(figure_dir / f"{case_id}_{method}_risk_by_step.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()
