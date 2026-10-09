"""Patient-level feature manifest; no automatic conversion of clinical labels."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .regions import allocate_regions, representative_patches, spatial_regions


@dataclass(frozen=True)
class Patient:
    patient_id: str
    label: int
    slides: tuple[tuple[str, Path], ...]


def read_manifest(path: str | Path) -> list[Patient]:
    path = Path(path).resolve()
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("endpoint") != "five_year_death" or content.get("partition") != "train":
        raise ValueError("predictor fitting/selection manifests must describe the verified five_year_death Train cohort")
    patients, patient_ids, slide_ids, feature_files = [], set(), set(), set()
    for row in content.get("patients", []):
        pid, label = row["patient_id"], row["label"]
        if not isinstance(pid, str) or not pid or pid in patient_ids or type(label) is not int or label not in (0, 1):
            raise ValueError("patient IDs must be unique/nonempty and verified labels must be integer 0/1")
        patient_ids.add(pid)
        slides = []
        for slide in row["slides"]:
            sid = slide["slide_id"]
            feature_path = (path.parent / slide["features_path"]).resolve()
            if not isinstance(sid, str) or not sid or sid in slide_ids or feature_path in feature_files:
                raise ValueError("slide IDs and feature files must have one patient owner")
            if not feature_path.is_file():
                raise FileNotFoundError(feature_path)
            slide_ids.add(sid)
            feature_files.add(feature_path)
            slides.append((sid, feature_path))
        if not slides:
            raise ValueError("each patient needs at least one baseline slide")
        patients.append(Patient(pid, label, tuple(sorted(slides))))
    if not patients:
        raise ValueError("patient manifest is empty")
    return patients


def check_disjoint(fit: list[Patient], select: list[Patient]) -> None:
    for field, left, right in [
        ("patient", {p.patient_id for p in fit}, {p.patient_id for p in select}),
        ("slide", {s for p in fit for s, _ in p.slides}, {s for p in select for s, _ in p.slides}),
        ("feature file", {f for p in fit for _, f in p.slides}, {f for p in select for _, f in p.slides}),
    ]:
        if left & right:
            raise ValueError(f"fitting and internal selection overlap at {field} level")


def patient_bank(patient: Patient, max_regions: int = 64, patches_per_region: int = 32) -> list[np.ndarray]:
    slides = []
    for _, path in patient.slides:
        with np.load(path, allow_pickle=False) as archive:
            features = np.asarray(archive["img_features"], dtype=np.float32)
            coords = np.asarray(archive["coords"], dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != 1024 or len(features) != len(coords) or not np.isfinite(features).all():
            raise ValueError(f"invalid UNI feature/coordinate arrays: {path}")
        slides.append((features, coords))
    allocation = allocate_regions([len(x) for x, _ in slides], max_regions)
    bank = []
    for (features, coords), number in zip(slides, allocation):
        # Region definitions and sampled IDs use coords only.
        for group in spatial_regions(coords, number):
            indices = representative_patches(coords, group, patches_per_region)
            bank.append(features[indices].copy())
    return bank
