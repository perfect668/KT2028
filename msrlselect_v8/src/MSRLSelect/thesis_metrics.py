#!/usr/bin/env python3
"""Shared label-1/death-risk evaluation protocol for thesis experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def youden_threshold(labels, death_risk) -> tuple[float, dict]:
    labels = np.asarray(labels, dtype=np.int64)
    death_risk = np.asarray(death_risk, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(labels, death_risk)
    youden = tpr - fpr
    index = int(np.nanargmax(youden))
    threshold = float(np.clip(thresholds[index], 0.0, 1.0))
    record = {
        "selection_split": "validation",
        "criterion": "maximum_youden_index",
        "positive_class": 1,
        "positive_class_name": "Died within 5y",
        "threshold": threshold,
        "selected_index": index,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "youden_j": youden.tolist(),
    }
    return threshold, record


def classification_metrics(labels, death_risk, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    death_risk = np.asarray(death_risk, dtype=np.float64)
    prediction = (death_risk >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, death_risk)),
        "auprc": float(average_precision_score(labels, death_risk)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "threshold": float(threshold),
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int((1 - labels).sum()),
    }


def save_prediction_artifacts(
    output_dir: Path,
    split_predictions: dict[str, pd.DataFrame],
) -> dict:
    """Save all splits using one threshold selected exclusively on Validation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = split_predictions["valid"]
    threshold, threshold_record = youden_threshold(valid["label"], valid["death_risk"])
    (output_dir / "validation_threshold.json").write_text(
        json.dumps(threshold_record, indent=2), encoding="utf-8"
    )

    final_metrics = {}
    for split in ("train", "valid", "test"):
        frame = split_predictions[split].copy()
        frame["death_risk"] = frame["death_risk"].astype(float)
        frame["prob_died_within_5y"] = frame["death_risk"]
        frame["prediction_fixed_0_5"] = (frame["death_risk"] >= 0.5).astype(int)
        frame["prediction_validation_youden"] = (
            frame["death_risk"] >= threshold
        ).astype(int)
        frame.to_csv(output_dir / f"{split}_predictions.csv", index=False)

        labels = frame["label"].to_numpy(dtype=np.int64)
        risks = frame["death_risk"].to_numpy(dtype=np.float64)
        fpr, tpr, roc_thresholds = roc_curve(labels, risks)
        precision, recall, pr_thresholds = precision_recall_curve(labels, risks)
        (output_dir / f"{split}_roc.json").write_text(
            json.dumps(
                {
                    "positive_class": 1,
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / f"{split}_pr.json").write_text(
            json.dumps(
                {
                    "positive_class": 1,
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        final_metrics[split] = {
            "fixed_0_5": classification_metrics(labels, risks, 0.5),
            "validation_youden": classification_metrics(labels, risks, threshold),
        }

    (output_dir / "metrics_final.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )
    return final_metrics
