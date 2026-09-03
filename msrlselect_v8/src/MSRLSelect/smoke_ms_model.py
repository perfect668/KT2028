#!/usr/bin/env python3
"""Small real-data P1 smoke train: forward, loss, backward and optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from models.ms_model import MSRLSelectModel


def argspec():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cases", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--token-dim", type=int, default=256)
    return p.parse_args()


def label_column(frame: pd.DataFrame) -> str:
    for name in ("label", "survival_label", "target", "y", "event"):
        if name in frame.columns:
            return name
    raise KeyError(f"No label column found; columns={list(frame.columns)}")


def load_rows(path: Path, n: int):
    frame = pd.read_csv(path).head(n)
    if len(frame) < 2:
        raise ValueError("at least two cases are required")
    f, c, clusters, labels = [], [], [], []
    ycol = label_column(frame)
    for row in frame.itertuples(index=False):
        with np.load(getattr(row, "features_filepath")) as z:
            f.append(z["img_features"].astype(np.float32, copy=False))
            c.append(z["coords"].astype(np.float32, copy=False))
        clusters.append(json.loads(Path(getattr(row, "clusters_json_filepath")).read_text()))
        labels.append(int(getattr(row, ycol)))
    max_p = max(x.shape[0] for x in f)
    dim = f[0].shape[1]
    fb = np.zeros((len(f), max_p, dim), dtype=np.float32)
    cb = np.zeros((len(f), max_p, 2), dtype=np.float32)
    for i, (x, xy) in enumerate(zip(f, c)):
        fb[i, : len(x)] = x
        cb[i, : len(xy)] = xy
    return torch.from_numpy(fb), torch.from_numpy(cb), clusters, torch.tensor(labels, dtype=torch.long)


def main():
    a = argspec()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(1)
    features, coords, clusters, labels = load_rows(a.manifest, a.cases)
    device = torch.device("cuda")
    model = MSRLSelectModel(input_dim=features.shape[-1], token_dim=a.token_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    model.train()
    losses = []
    for epoch in range(a.epochs):
        order = torch.arange(len(labels))
        for start in range(0, len(order), a.batch_size):
            idx = order[start:start + a.batch_size].tolist()
            x, xy, y = features[idx].to(device), coords[idx].to(device), labels[idx].to(device)
            logits, aux = model(x, xy, [clusters[i] for i in idx])
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        logits, aux = model(features[:2].to(device), coords[:2].to(device), clusters[:2])
        selected = model.topk_regions(aux["attention"], aux["cluster_mask"], budget=4)
    if not torch.isfinite(logits).all() or not torch.isfinite(aux["attention"]).all():
        raise RuntimeError("non-finite P1 output")
    print(json.dumps({"status": "PASS", "epochs": a.epochs, "cases": len(labels),
                      "last_loss": losses[-1], "logits": list(logits.shape),
                      "attention": list(aux["attention"].shape),
                      "selected_regions": selected.sum(1).tolist(),
                      "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}, indent=2))


if __name__ == "__main__":
    main()
