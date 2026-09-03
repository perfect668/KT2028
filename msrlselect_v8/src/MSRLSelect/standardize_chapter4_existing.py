#!/usr/bin/env python3
"""Regenerate chapter-4 artifacts with the common death-risk protocol.

This never retrains or changes any existing checkpoint.  It reads predictions
from the completed fixed-seed runs, writes a separate v2 tree, and selects the
threshold only from Validation before applying it to Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thesis_metrics import save_prediction_artifacts


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    risk_column = next(
        column
        for column in ("death_risk", "prob_died_within_5y", "probability", "prob1")
        if column in frame.columns
    )
    case_column = "case_id" if "case_id" in frame.columns else frame.columns[0]
    return pd.DataFrame(
        {
            "case_id": frame[case_column].astype(str),
            "label": frame["label"].astype(int),
            "death_risk": frame[risk_column].astype(float),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_models = {
        "mean_mlp": args.old_root / "chapter4" / "mean_mlp",
        "global_only": args.old_root / "chapter4" / "global",
        "local_only": args.old_root / "chapter4" / "local",
        "simple_concat": args.old_root / "chapter4" / "concat",
        # The prior attention implementation is retained as a historical
        # artifact only; the true Attention-MIL run is saved independently.
        "attention_interface_legacy": args.old_root / "chapter4" / "attention",
        "gated_fusion": args.old_root.parent / "msrlselect_basic" / "sg_survnet_v2" / "fusion",
    }
    summary = {}
    for name, source in source_models.items():
        predictions = {
            split: normalize(pd.read_csv(source / f"{split}_predictions.csv"))
            for split in ("train", "valid", "test")
        }
        target = args.output_root / name
        final_metrics = save_prediction_artifacts(target, predictions)
        (target / "provenance.json").write_text(
            json.dumps(
                {
                    "source_predictions": str(source),
                    "checkpoint_retrained": False,
                    "threshold_protocol": "validation_maximum_youden_applied_to_all_splits",
                    "positive_class": "Died within 5y",
                    "legacy_attention_note": (
                        "not true patch-level Attention MIL"
                        if name == "attention_interface_legacy"
                        else None
                    ),
                },
                indent=2,
            )
        )
        summary[name] = {
            "test_auc": final_metrics["test"]["fixed_0_5"]["auc"],
            "test_auprc": final_metrics["test"]["fixed_0_5"]["auprc"],
        }
    (args.output_root / "standardization_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
