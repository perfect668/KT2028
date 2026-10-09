"""Label-free acquisition state and optional training-only loss reward.

This is the cached acquisition engine, not an end-to-end WSI reader.
Gymnasium/PPO integration can wrap these reset/step/action_masks methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def log_loss(label: int, probability: float) -> float:
    if label not in (0, 1) or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("invalid binary label or probability")
    p = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return float(-label * np.log(p) - (1 - label) * np.log1p(-p))


@dataclass(frozen=True)
class Transition:
    observation: dict[str, np.ndarray]
    reward: float | None
    terminated: bool


class AcquisitionEnv:
    """Predictor receives purchased patches only; labels never enter observations."""

    def __init__(
        self,
        regions: list[np.ndarray],
        cheap: np.ndarray,
        predict: Callable[[np.ndarray, int], float],
        *,
        budget_patches: int = 512,
        cost_weight: float = 0.25,
        label: int | None = None,
        max_regions: int = 64,
        cost_unit: int = 2048,
    ):
        if not regions or len(regions) > max_regions:
            raise ValueError("invalid candidate region count")
        arrays = [np.asarray(r, dtype=np.float32).copy() for r in regions]
        dim = arrays[0].shape[1] if arrays[0].ndim == 2 else 0
        if not dim or any(r.ndim != 2 or r.shape[1] != dim or not 1 <= len(r) <= 32
                          or not np.isfinite(r).all() for r in arrays):
            raise ValueError("regions must contain 1..32 finite patch vectors with a shared dimension")
        cheap = np.asarray(cheap, dtype=np.float32)
        if cheap.ndim != 2 or len(cheap) != len(regions) or not np.isfinite(cheap).all():
            raise ValueError("cheap descriptors must match regions")
        if budget_patches < 0 or cost_unit < 1 or not np.isfinite(cost_weight) or cost_weight < 0:
            raise ValueError("invalid budget or cost unit/weight")
        if label is not None and label not in (0, 1):
            raise ValueError("training label must be binary")
        self._regions, self._predict, self._label = arrays, predict, label
        self._cheap = cheap.copy()
        self._dim, self._max_regions = dim, max_regions
        self._budget, self._cost_unit, self._weight = budget_patches, cost_unit, cost_weight
        self.stop_action = max_regions
        self.total_patches = sum(map(len, arrays))
        self.reset()

    def _probability(self) -> float:
        patches = (np.concatenate([self._regions[i] for i in self._order])
                   if self._order else np.empty((0, self._dim), dtype=np.float32))
        p = float(self._predict(patches, self.total_patches))
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("predictor returned an invalid probability")
        return p

    def reset(self) -> dict[str, np.ndarray]:
        self._order: list[int] = []
        self._spent = 0
        self._done = False
        self._p = self._probability()
        return self.observation()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self._max_regions + 1, dtype=bool)
        mask[self.stop_action] = True
        if not self._done:
            for i, region in enumerate(self._regions):
                mask[i] = i not in self._order and self._spent + len(region) <= self._budget
        return mask

    def observation(self) -> dict[str, np.ndarray]:
        cheap = np.zeros((self._max_regions, self._cheap.shape[1]), dtype=np.float32)
        cheap[:len(self._regions)] = self._cheap
        fine = np.zeros((self._max_regions, self._dim), dtype=np.float32)
        observed = np.zeros(self._max_regions, dtype=bool)
        for i in self._order:
            fine[i] = self._regions[i].mean(axis=0)
            observed[i] = True
        valid = np.arange(self._max_regions) < len(self._regions)
        costs = np.zeros(self._max_regions, dtype=np.float32)
        costs[:len(self._regions)] = [len(r) / self._cost_unit for r in self._regions]
        p = float(np.clip(self._p, 1e-6, 1 - 1e-6))
        scalars = np.asarray([
            self._p, -p * np.log(p) - (1 - p) * np.log1p(-p),
            self._spent / self.total_patches, self._spent / self._cost_unit,
            (self._budget - self._spent) / self._cost_unit,
            self._budget / self._cost_unit, self._weight,
        ], dtype=np.float32)
        return {"cheap": cheap, "fine": fine, "observed": observed,
                "valid": valid, "costs": costs, "scalars": scalars}

    def step(self, action: int) -> Transition:
        if self._done:
            raise RuntimeError("episode is finished; call reset")
        if not isinstance(action, (int, np.integer)) or not 0 <= action <= self.stop_action:
            raise ValueError("action must be a candidate index or STOP")
        if not self.action_masks()[action]:
            raise ValueError("action is repeated, padded, or exceeds the patch budget")
        if action == self.stop_action:
            self._done = True
            return Transition(self.observation(), 0.0 if self._label is not None else None, True)
        previous = self._p
        self._order.append(int(action))
        n = len(self._regions[action])
        self._spent += n
        self._p = self._probability()
        reward = (log_loss(self._label, previous) - log_loss(self._label, self._p)
                  - self._weight * n / self._cost_unit) if self._label is not None else None
        self._done = not self.action_masks()[:self._max_regions].any()
        return Transition(self.observation(), reward, self._done)
