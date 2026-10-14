"""Discrete survival likelihood and metrics for the adopted common 5y censoring.

Event=1 means death; this differs from DSCA's c=1 censorship convention.
The likelihood is DSCA SurvMLE(alpha=0), expressed in stable logit space.
"""
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


def validate_edges(edges):
    a = np.asarray(edges, dtype=float)
    if a.ndim != 1 or len(a) < 3 or a[0] != 0 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
        raise ValueError('Require finite increasing interval edges beginning at0')
    return a


def encode_target(time_days, event, edges):
    edges = validate_edges(edges)
    t = float(time_days)
    if event not in (0, 1) or not np.isfinite(t) or not 0 < t <= edges[-1]:
        raise ValueError('Invalid event/time under registered survival window')
    if event:
        # (0,365],(365,730],...; an exact endpoint belongs to the ending bin.
        b = int(np.searchsorted(edges[1:], t, side='left'))
        return dict(time_days=t, event=1, event_bin=b, completed_intervals=b)
    completed = int(np.sum(edges[1:] <= t))
    return dict(time_days=t, event=0, event_bin=-1, completed_intervals=completed)


def discrete_nll(logits, target):
    """One case, no censor weighting or additional upweighting of deaths."""
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError('Expected one finite interval logit vector')
    values = [target[k] for k in ('event', 'event_bin', 'completed_intervals')]
    if any(int(v) != v for v in values):
        raise ValueError('Survival indices/event must be integers')
    event, b, completed = map(int, values)
    k = len(logits)
    if event not in (0, 1) or not 0 <= completed <= k:
        raise ValueError('Invalid survival target')
    if event and (not 0 <= b < k or completed != b):
        raise ValueError('Event bin/pre-event survival intervals disagree')
    if not event and b != -1:
        raise ValueError('Censoring is not a death event')
    survive = F.softplus(logits[:completed]).sum()  # -log(1-h)
    return survive + F.softplus(-logits[b]) if event else survive


def curve_from_logits(logits):
    if logits.ndim not in (1, 2):
        raise ValueError('Interval logits must be [K] or [B,K]')
    hazards = logits.sigmoid()
    # Log-domain product remains stable at extreme finite logits.
    survival = F.logsigmoid(-logits).cumsum(-1).exp()
    return hazards, survival


def common_censor_metrics(times, events, survival, edges):
    """No IPCW needed before a common admin cutoff: G(t-)=1.

This is NOT a generic metric implementation for earlier/variable censoring.
The integral is explicitly a trapezoid approximation on the annual grid.
"""
    edges = validate_edges(edges)
    t, e, s = np.asarray(times, float), np.asarray(events, int), np.asarray(survival, float)
    if t.ndim != 1 or len(t) == 0 or e.shape != t.shape or s.shape != (len(t), len(edges)-1):
        raise ValueError('Invalid survival metric shapes')
    if not np.isfinite(t).all() or not np.isfinite(s).all() or np.any((e != 0) & (e != 1)):
        raise ValueError('Nonfinite/invalid survival metric data')
    if np.any(t <= 0) or np.any(t > edges[-1]) or np.any(t[e == 0] != edges[-1]):
        raise ValueError('These metrics require the registered common administrative censoring')
    if np.any((s < 0) | (s > 1)) or np.any(np.diff(s, axis=1) > 1e-7):
        raise ValueError('Survival curves must be probabilities and nonincreasing')
    scores, briers = {}, []
    for j, end in enumerate(edges[1:]):
        y = ((e == 1) & (t <= end)).astype(float)
        death_prob = 1-s[:, j]
        brier = float(np.mean((death_prob-y)**2))
        scores[str(int(end))] = dict(brier=brier, auroc=float(roc_auc_score(y, death_prob)) if len(np.unique(y)) == 2 else None)
        briers.append(brier)
    # Risk ordering follows DSCA: lower sum/mean survival = higher risk.
    risk = -s.mean(axis=1)
    concordant, comparable, ties = 0., 0, 0
    for i in np.flatnonzero(e == 1):
        other = np.flatnonzero(t > t[i])
        diff = risk[i]-risk[other]
        comparable += len(other)
        ties += int(np.sum(diff == 0))
        concordant += float(np.sum(diff > 0) + .5*np.sum(diff == 0))
    integral = np.trapezoid(np.r_[0., briers], edges)/edges[-1]
    return dict(c_index=concordant/comparable if comparable else None, comparable_pairs=comparable,
                risk_tied_pairs=ties, risk_definition='minus_mean_annual_survival', by_day=scores,
                integrated_brier_annual_grid=float(integral),
                censoring_metric_contract='common admin cutoff;known alive through endpoint;no earlier censoring')
