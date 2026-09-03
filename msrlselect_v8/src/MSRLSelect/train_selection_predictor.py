#!/usr/bin/env python3
"""Common downstream predictor for all chapter-3 selection methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from pipeline_common import read_config, seed_all
from thesis_metrics import save_prediction_artifacts


class SelectionDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: Path, cache_root: Path, method: str, split: str):
        frame = pd.read_csv(manifest)
        self.rows = []
        for row in frame.itertuples(index=False):
            path = cache_root / method / split / f"{row.case_id}.npz"
            with np.load(path) as data:
                vector = data["selected_feature_mean"].astype(np.float32, copy=True)
            self.rows.append((vector, int(row.label), str(row.case_id)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        vector, label, case_id = self.rows[index]
        return torch.from_numpy(vector), torch.tensor(label), case_id


class CommonPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, vectors):
        return self.net(vectors)


def prediction_frame(model, loader, device):
    model.eval()
    risks, labels, case_ids = [], [], []
    with torch.no_grad():
        for vectors, y, ids in loader:
            logits = model(vectors.to(device)).softmax(dim=1)[:, 1]
            risks.extend(logits.cpu().tolist())
            labels.extend(y.tolist())
            case_ids.extend(ids)
    return pd.DataFrame({"case_id": case_ids, "label": labels, "death_risk": risks})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config = read_config(args.config)
    seed_all(int(config["seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: SelectionDataset(
            Path(config[f"{split}_manifest"]), args.cache_root, args.method, split
        )
        for split in ("train", "valid", "test")
    }
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset, args.batch_size, shuffle=split == "train", pin_memory=True
        )
        for split, dataset in datasets.items()
    }
    device = torch.device("cuda")
    model = CommonPredictor().to(device)
    train_labels = np.asarray([label for _, label, _ in datasets["train"].rows])
    counts = np.bincount(train_labels, minlength=2)
    weights = torch.tensor(len(train_labels) / (2 * counts), dtype=torch.float32, device=device)
    loss = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for vectors, labels, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(vectors.to(device)), labels.to(device))
            value.backward()
            optimizer.step()
            losses.append(float(value.detach()))
        validation = prediction_frame(model, loaders["valid"], device)
        validation_auc = float(roc_auc_score(validation.label, validation.death_risk))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_auc": validation_auc}
        history.append(row)
        if validation_auc > best:
            best = validation_auc
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "method": args.method},
                args.output_dir / "model_best.pt",
            )
    checkpoint = torch.load(args.output_dir / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_predictions = {
        split: prediction_frame(model, loaders[split], device)
        for split in ("train", "valid", "test")
    }
    metrics = save_prediction_artifacts(args.output_dir, final_predictions)
    (args.output_dir / "metrics_history.json").write_text(json.dumps(history, indent=2))
    (args.output_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "method": args.method,
                "checkpoint_selection": "validation_auc",
                "selected_epoch": checkpoint["epoch"],
                "test_evaluated_once_after_selection": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
