import numpy as np
import pytest
import torch
from torch.nn import functional as F

from kt2028.network import EvidencePredictor, TargetedStateUpdater, SharedActorCritic
from kt2028.episode import AcquisitionEpisode


class TrackedCase:
    """Software fixture; not a simulated prognosis experiment."""
    def __init__(self):
        rng = np.random.default_rng(19)
        self.coarse = rng.normal(size=(3, 512)).astype('float32')
        self.boxes = np.array([[0, 0, 1, 1], [1, 0, 2, 1], [2, 0, 3, 1]], dtype='float32')
        self.counts = np.array([2, 3, 0])
        self.observed = np.zeros(3, dtype=bool)
        self.fine = {}
        self.reads = []
        self.spent = 0

    def visible_state(self):
        return {'coarse': self.coarse, 'bbox_um': self.boxes,
                'cached_patch_counts': self.counts, 'observed': self.observed.copy(),
                'observed_fine': dict(self.fine), 'acquirable': (self.counts > 0) & ~self.observed,
                'spent_patch_count': self.spent}

    def acquire(self, i):
        self.reads.append(i)
        self.fine[i] = np.ones((self.counts[i], 1024), dtype='float32')
        self.observed[i] = True
        self.spent += int(self.counts[i])
        return {'new_patch_cost': int(self.counts[i])}


def make_episode(**kwargs):
    torch.manual_seed(5)
    return AcquisitionEpisode(TrackedCase(), EvidencePredictor(dim=16, branches=2),
             n_ref=10, budget_patches=5, cost_preference=0.7, **kwargs)


def test_zero_fine_stop_and_label_permission():
    a = make_episode(training_label=0)
    b = make_episode(training_label=1)
    sa, sb = a.visible_state(), b.visible_state()
    assert torch.equal(sa['prediction']['logit'], sb['prediction']['logit'])
    assert a.case.reads == b.case.reads == []
    _, reward, done, _ = a.step(3)
    expected = -F.binary_cross_entropy_with_logits(sa['prediction']['logit'], torch.tensor(0.)).item()
    assert done and reward == pytest.approx(expected)
    with pytest.raises(ValueError):
        a.step(3)
    inference = make_episode()
    assert inference.step(3)[1] == 0


def test_episode_return_exactly_terminal_bce_plus_cost():
    episode = make_episode(training_label=1)
    _, r1, done, _ = episode.step(0)
    assert not done and episode.case.reads == [0]
    with pytest.raises(ValueError):
        episode.step(0)
    state, r2, done, _ = episode.step(1)
    assert done and episode.case.spent == 5
    bce = F.binary_cross_entropy_with_logits(state['prediction']['logit'], torch.tensor(1.)).item()
    assert r1 + r2 == pytest.approx(-bce - 0.7 * 5 / 10)


def test_budget_and_policy_mask():
    episode = AcquisitionEpisode(TrackedCase(), EvidencePredictor(dim=16, branches=2),
                                n_ref=10, budget_patches=2, cost_preference=0)
    assert episode.action_mask().tolist() == [True, False, False, True]
    actor = SharedActorCritic(dim=16, branches=2)
    dist, value = actor(**episode.visible_state())
    assert torch.isfinite(value) and dist.probs[1:3].sum().item() == 0
    assert dist.probs[-1] > 0
    with pytest.raises(ValueError):
        episode.step(1)
    assert episode.case.reads == []
    assert episode.step(0)[2]


def test_tsu_does_not_turn_estimates_into_acquisitions():
    episode = make_episode(updater=TargetedStateUpdater(dim=16, threshold=-1))
    episode.step(0)
    state = episode.visible_state()['prediction']
    assert state['observed'].tolist() == [True, False, False]
    assert state['estimated'].tolist() == [False, True, True]
    assert episode.action_mask()[1] and episode.case.reads == [0]
    episode.step(1)
    assert 1 not in episode.estimates and episode.case.spent == 5


def test_coarse_zero_fine_gradients_and_permutation():
    torch.manual_seed(4)
    model = EvidencePredictor(dim=16, branches=2)
    coarse = torch.randn(5, 512)
    boxes = torch.tensor([[i, 0, i+1, 1] for i in range(5)], dtype=torch.float32)
    result = model(coarse, boxes, {})
    result['logit'].backward()
    assert model.coarse_projection[0].weight.grad.abs().sum() > 0
    permutation = torch.tensor([3, 1, 4, 0, 2])
    permuted = model(coarse[permutation], boxes[permutation], {})
    assert torch.allclose(result['logit'], permuted['logit'], atol=1e-6)
