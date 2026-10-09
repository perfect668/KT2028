"""Frozen-model acquisition episode. Inference does not require a label."""
import math
import numpy as np
import torch
from torch.nn import functional as F


class AcquisitionEpisode:
    def __init__(self, case, predictor, *, n_ref, budget_patches, cost_preference,
                 updater=None, training_label=None):
        if not math.isfinite(n_ref) or n_ref <= 0:
            raise ValueError('Explicit positive common N_ref required')
        if not isinstance(budget_patches, int) or budget_patches < 0:
            raise ValueError('Budget is an integer number of patches')
        if not math.isfinite(cost_preference) or cost_preference < 0:
            raise ValueError('Nonnegative finite lambda required')
        if training_label is not None and training_label not in (0, 1):
            raise ValueError('Binary training reward label required')
        if case.visible_state()['observed'].any():
            raise ValueError('Fresh zero-fine case required')
        self.case, self.predictor, self.updater = case, predictor.eval(), updater
        if updater is not None:
            updater.eval()
        self.n_ref, self.budget, self.lam = n_ref, budget_patches, cost_preference
        self._reward_label = training_label
        self.estimates = {}
        self.done = False
        self.device = next(predictor.parameters()).device

    def action_mask(self):
        state = self.case.visible_state()
        remaining = self.budget - state['spent_patch_count']
        return np.r_[state['acquirable'] & (state['cached_patch_counts'] <= remaining), True]

    @torch.no_grad()
    def visible_state(self):
        state = self.case.visible_state()
        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)
        prediction = self.predictor(tensor(state['coarse']), tensor(state['bbox_um']),
                                    {i: tensor(v) for i, v in state['observed_fine'].items()},
                                    self.estimates)
        return {'prediction': prediction, 'costs': tensor(state['cached_patch_counts']) / self.n_ref,
                'spent': state['spent_patch_count'] / self.n_ref,
                'remaining': (self.budget - state['spent_patch_count']) / self.n_ref,
                'cost_preference': self.lam,
                'legal': torch.as_tensor(self.action_mask(), device=self.device)}

    @torch.no_grad()
    def step(self, action):
        if self.done:
            raise ValueError('Episode already terminal')
        mask = self.action_mask()
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(mask) or not mask[action]:
            raise ValueError('Illegal or unaffordable action')
        reward = 0.0
        reason = 'acquire'
        if action == len(mask) - 1:
            self.done, reason = True, 'stop'
        else:
            acquired = self.case.acquire(action)
            reward = -self.lam * acquired['new_patch_cost'] / self.n_ref
            if self.updater is not None:
                state = self.visible_state()['prediction']
                self.estimates = self.updater.propagate(state['coarse_latent'], int(action),
                                  state['evidence'][action], state['observed'], self.estimates)
            if not self.action_mask()[:-1].any():
                self.done, reason = True, 'no_affordable_acquisition'
        visible = self.visible_state()
        if self.done and self._reward_label is not None:
            logit = visible['prediction']['logit']
            reward -= F.binary_cross_entropy_with_logits(logit, logit.new_tensor(self._reward_label)).item()
        # Inference has no terminal loss; returned cost is bookkeeping only.
        return visible, reward, self.done, {'reason': reason,
                'training_reward': self._reward_label is not None,
                'spent_patch_count': self.case.visible_state()['spent_patch_count']}
