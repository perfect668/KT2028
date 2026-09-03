"""Adaptive region-selection budget controller."""
from __future__ import annotations

import torch


class AdaptiveBudgetController:
    def __init__(self, min_steps: int = 2, max_steps: int = 6,
                 gain_threshold: float = 0.01, reward_threshold: float = 0.0):
        if not 1 <= min_steps <= max_steps:
            raise ValueError("require 1 <= min_steps <= max_steps")
        self.min_steps, self.max_steps = min_steps, max_steps
        self.gain_threshold, self.reward_threshold = gain_threshold, reward_threshold

    def should_stop(self, step: int, gains: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Per-case stop decision; the last two observations must be provided."""
        if step >= self.max_steps:
            return torch.ones_like(gains[-1], dtype=torch.bool)
        if step < self.min_steps or gains.shape[0] < 2 or rewards.shape[0] < 2:
            return torch.zeros_like(gains[-1], dtype=torch.bool)
        return (gains[-2:].abs().mean(0) < self.gain_threshold) & \
               (rewards[-2:].mean(0) < self.reward_threshold)
