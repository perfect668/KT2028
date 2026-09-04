"""Train-only assets and the frozen Fixed E2 OOF prediction environment."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
import torch.nn.functional as F


class EvidenceNet(nn.Module):
    """Architecture used by the frozen E2_fixed_morphology_slots checkpoints."""

    def __init__(self, coarse_dim: int, fine_dim: int = 1024):
        super().__init__()
        d = 384
        self.cg = nn.Sequential(nn.Linear(coarse_dim, d), nn.GELU(), nn.Dropout(0.3))
        self.gpool = nn.Linear(d, 1)
        self.lproj = nn.Linear(fine_dim, d)
        self.res = nn.Sequential(nn.Linear(2 * d + 1, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, d))
        self.alpha = nn.Sequential(nn.Linear(2 * d + 1, 128), nn.GELU(), nn.Linear(128, 1))
        self.gh, self.lh, self.fh, self.uh = nn.Linear(d, 1), nn.Linear(d, 1), nn.Linear(d, 1), nn.Linear(d, 1)

    def forward(self, coarse, fine, mask):
        batch, regions, _ = coarse.shape
        masks = mask.shape[1]
        coarse_tokens = self.cg(coarse)
        weights = torch.softmax(self.gpool(coarse_tokens).squeeze(-1).float(), -1)
        global_token = (coarse_tokens * weights.unsqueeze(-1).to(coarse_tokens.dtype)).sum(1)
        uncertainty = torch.sigmoid(self.uh(global_token)).squeeze(-1)
        fine_tokens = self.lproj(fine)[:, None].expand(batch, masks, regions, -1)
        count = mask.sum(-1, keepdim=True)
        local = (fine_tokens * mask.unsqueeze(-1)).sum(2) / count.clamp_min(1)
        empty = (count == 0).float()
        global_expanded = global_token[:, None].expand(batch, masks, -1)
        alpha = torch.sigmoid(self.alpha(torch.cat([global_expanded, local, empty], -1))).squeeze(-1)
        alpha = torch.where(empty.squeeze(-1) > 0, torch.zeros_like(alpha), alpha)
        ratio = mask.sum(-1, keepdim=True).float() / 64.0
        fused = F.layer_norm(
            global_expanded + alpha.unsqueeze(-1) * self.res(torch.cat([global_expanded, local, ratio], -1)),
            (384,),
        )
        return self.fh(fused).squeeze(-1), self.gh(global_token).squeeze(-1), self.lh(local).squeeze(-1), uncertainty, alpha


@dataclass
class TrainAssets:
    root: Path
    metadata: pd.DataFrame
    coarse: torch.Tensor
    fine_slots: torch.Tensor
    calibration: dict
    calibration_path: Path
    calibration_sha256: str
    checkpoint_paths: list
    opened_paths: list


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_train_only_path(path: Path):
    lowered = [part.lower() for part in path.parts]
    forbidden = ("validation", "validate", "test")
    if any(any(word in part for word in forbidden) for part in lowered):
        raise AssertionError(f"Non-Train asset access rejected: {path}")


def load_train_assets(data_root) -> TrainAssets:
    root = Path(data_root).resolve()
    evidence = root / "outputs/thesis_evidence_interface_v7"
    metadata_path = root / "prepared/sr386_full/surgen_train_10.csv"
    calibration_path = root / "outputs/thesis_rl_v7/r1_online_ppo/metrics/e2_calibration_parameters.csv"
    required = [
        metadata_path,
        calibration_path,
        evidence / "assets/train_pca128_components.npy",
        evidence / "assets/train_pca128_mean.npy",
        evidence / "assets/train_pca128_explained_variance.npy",
        evidence / "assets/train_moe_slots.pt",
    ]
    checkpoints = [
        evidence / "checkpoints" / f"E2_oof_fold{fold}_seed{seed}_best.pt"
        for fold in range(5) for seed in (1, 17, 42)
    ]
    for path in required + checkpoints:
        _assert_train_only_path(path)
    missing = [str(path) for path in required + checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Fixed E2 assets:\n" + "\n".join(missing))
    if "e2m" in str(calibration_path).lower():
        raise AssertionError("Fixed E2 cannot use an E2M calibration file")

    metadata = pd.read_csv(metadata_path)
    components = np.load(required[2])
    mean = np.load(required[3])
    explained = np.load(required[4])
    pca = PCA(n_components=128)
    pca.components_, pca.mean_, pca.explained_variance_ = components, mean, explained
    pca.explained_variance_ratio_ = explained / explained.sum()
    pca.singular_values_ = np.sqrt(np.maximum(explained * (16320 - 1), 0))
    pca.n_features_in_, pca.n_samples_, pca.n_samples_seen_, pca.noise_variance_ = 1024, 16320, 16320, 0.0

    coarse_rows = []
    region_root = root / "outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache/regions_k64/train"
    opened = [str(path) for path in required + checkpoints]
    for row in metadata.itertuples():
        region_path = region_root / f"{row.case_id}.npz"
        _assert_train_only_path(region_path)
        if not region_path.is_file():
            raise FileNotFoundError(str(region_path))
        with np.load(region_path) as region:
            projected = pca.transform(region["region_feature_mean"].astype("float32"))
            count = np.log1p(region["region_size"]).astype("float32")[:, None]
            coarse_rows.append(np.concatenate([projected, count, region["region_centroid"], region["region_spread"]], 1))
        opened.append(str(region_path))
    coarse = torch.from_numpy(np.stack(coarse_rows).astype("float32"))
    fine_slots = torch.load(required[5], map_location="cpu", weights_only=True).float()
    if coarse.shape[:2] != fine_slots.shape[:2] or coarse.shape[1] != 64:
        raise AssertionError(f"Unexpected K64 asset shapes: coarse={tuple(coarse.shape)}, fine={tuple(fine_slots.shape)}")

    table = pd.read_csv(calibration_path)
    calibration = {}
    for row in table.itertuples():
        calibration[str(row.mode)] = (float(row.intercept), [float(x) for x in str(row.coefficients).split(";")])
    if not {"k64", "partial"}.issubset(calibration):
        raise AssertionError("Fixed E2 calibration must contain k64 and partial rows")
    return TrainAssets(root, metadata, coarse, fine_slots, calibration, calibration_path,
                       _file_hash(calibration_path), checkpoints, opened)


class FixedE2Ensemble:
    """Frozen three-seed, patient-fold-specific Fixed E2 predictor."""

    def __init__(self, assets: TrainAssets, device):
        self.assets, self.device = assets, torch.device(device)
        self.models = {}
        for fold in range(5):
            fold_models = []
            for seed in (1, 17, 42):
                path = assets.root / "outputs/thesis_evidence_interface_v7/checkpoints" / f"E2_oof_fold{fold}_seed{seed}_best.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                model = EvidenceNet(assets.coarse.shape[-1]).to(self.device)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                fold_models.append(model)
            self.models[fold] = fold_models

    def _calibrate(self, probability, count):
        raw_logit = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        k64_a, k64_b = self.assets.calibration["k64"]
        part_a, part_b = self.assets.calibration["partial"]
        part_b = part_b + [0.0] * (3 - len(part_b))
        ratio = count.float() / 64.0
        # The calibration fit uses [z, z*rho, rho], in this order.
        partial = torch.sigmoid(part_a + part_b[0] * raw_logit + part_b[1] * raw_logit * ratio + part_b[2] * ratio)
        full = torch.sigmoid(k64_a + k64_b[0] * raw_logit)
        return torch.where(count == 64, full, partial)

    @torch.no_grad()
    def predict(self, patient_indices, active_mask):
        indices = torch.as_tensor(patient_indices, dtype=torch.long, device=self.device)
        active = torch.as_tensor(active_mask, dtype=torch.bool, device=self.device)
        coarse = self.assets.coarse[indices.cpu()].to(self.device)
        fine = self.assets.fine_slots[indices.cpu()].flatten(-2).to(self.device)
        seed_probabilities = torch.empty(len(indices), 3, device=self.device)
        folds = indices.remainder(5)
        for fold in range(5):
            selected = torch.where(folds == fold)[0]
            if selected.numel() == 0:
                continue
            mask = active[selected, None]
            predictions = [torch.sigmoid(model(coarse[selected], fine[selected], mask)[0][:, 0]) for model in self.models[fold]]
            seed_probabilities[selected] = torch.stack(predictions, 1)
        raw_mean = seed_probabilities.mean(1)
        probability = self._calibrate(raw_mean, active.sum(1))
        disagreement = seed_probabilities.std(1, unbiased=False)
        p = probability.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        return probability, entropy, disagreement, seed_probabilities
