"""Train the first partial-evidence baseline on an explicitly isolated inner split.

This entry point is not an OOF orchestrator or a Validation/Test evaluator.
No experiments execute on import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .data import check_disjoint, patient_bank, read_manifest
from .predictor import PartialABMIL


def tensor_bag(regions, selection, device):
    features = np.concatenate([regions[int(i)] for i in selection])
    x = torch.as_tensor(features, device=device).unsqueeze(0)
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
    count = torch.tensor([sum(map(len, regions))], device=device)
    return x, mask, count


def sampled_regions(rng, size):
    candidates = sorted({min(k, size) for k in (1, 2, 4, 8, 16, 32, size)})
    return rng.choice(size, size=int(rng.choice(candidates)), replace=False)


@torch.no_grad()
def selection_loss(model, patients, seed, device):
    model.eval()
    # The same patient/subset draws are used at every checkpoint evaluation.
    rng = np.random.default_rng(seed)
    losses = []
    for patient in patients:
        regions = patient_bank(patient)
        target = torch.tensor([float(patient.label)], device=device)
        subsets = [np.arange(len(regions))] + [sampled_regions(rng, len(regions)) for _ in range(20)]
        patient_losses = [float(F.binary_cross_entropy_with_logits(
            model(*tensor_bag(regions, subset, device))["logits"], target)) for subset in subsets]
        losses.append(0.25 * patient_losses[0] + 0.75 * np.mean(patient_losses[1:]))
    return float(np.mean(losses))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-manifest", required=True, type=Path)
    parser.add_argument("--select-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.lr <= 0 or args.effective_batch < 1:
        parser.error("epochs, learning rate, and effective batch must be positive")
    fit, select = read_manifest(args.fit_manifest), read_manifest(args.select_manifest)
    check_disjoint(fit, select)
    if {p.label for p in fit} != {0, 1}:
        raise ValueError("both outcomes are required in the fitting cohort")
    # Complete input checks before creating outputs or touching model weights.
    for patient in fit + select:
        patient_bank(patient)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    prior = (sum(p.label for p in fit) + 1) / (len(fit) + 2)
    model = PartialABMIL(prior=prior).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Never overwrite another run or historical result.
    args.output.mkdir(parents=True, exist_ok=False)
    configuration = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "fit_manifest_sha256": hashlib.sha256(args.fit_manifest.read_bytes()).hexdigest(),
        "select_manifest_sha256": hashlib.sha256(args.select_manifest.read_bytes()).hexdigest(),
        "prior": prior, "checkpoint_rule": "minimum internal-selection patient-mean BCE; earlier tie",
        "checkpoint_scope": "single inner-split model, not strict OOF or final deployment model",
        "torch_version": torch.__version__,
    }
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    best = float("inf")
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(fit))
        for start in range(0, len(order), args.effective_batch):
            patient_indices = order[start:start + args.effective_batch]
            optimizer.zero_grad()
            for ix in patient_indices:
                patient = fit[int(ix)]
                regions = patient_bank(patient)
                target = torch.tensor([float(patient.label)], device=args.device)
                subsets = [np.arange(len(regions)), sampled_regions(rng, len(regions)),
                           sampled_regions(rng, len(regions))]
                loss = sum(weight * F.binary_cross_entropy_with_logits(
                    model(*tensor_bag(regions, subset, args.device))["logits"], target)
                    for weight, subset in zip((0.25, 0.375, 0.375), subsets))
                (loss / len(patient_indices)).backward()
            optimizer.step()
        score = selection_loss(model, select, args.seed + 10000, args.device)
        log.append({"epoch": epoch, "internal_selection_bce": score})
        if score < best:
            best = score
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "config": configuration, "internal_selection_bce": score}, args.output / "best.pt")
        (args.output / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"epoch={epoch} internal_selection_bce={score:.6f}", flush=True)


if __name__ == "__main__":
    main()
