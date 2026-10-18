"""Censor-aware annual-grid metrics; fit censoring ONLY on training patients.

Reverse Kaplan-Meier is fitted separately for each source. The administrative
endpoint is treated as known survival through that endpoint, not lost follow-up.
No clipping/epsilon substitution for unsupported inverse censoring probabilities.
"""
import numpy as np


class CensoringKM:
    def __init__(self, times, events):
        t, e = np.asarray(times, float), np.asarray(events)
        if not len(t) or t.shape != e.shape or np.any(t <= 0) or not np.isfinite(t).all() or np.any((e != 0)&(e != 1)):
            raise ValueError('Invalid training censoring observations')
        self.times = np.unique(t)
        self.after = []
        g = 1.
        for x in self.times:
            # Reverse-KM ties: deaths leave the risk set before censoring events.
            risk = np.sum(t >= x)-np.sum((t == x)&(e == 1))
            censored = np.sum((t == x)&(e == 0))
            if censored:
                g *= 1-censored/risk
            self.after.append(g)
        self.after = np.asarray(self.after)
        self.maximum_observation = float(t.max())

    def probability(self, time, *, left=False):
        if time > self.maximum_observation:
            return None
        i = int(np.searchsorted(self.times, time, side='left' if left else 'right'))-1
        value = float(self.after[i]) if i >= 0 else 1.
        return value if value > 0 else None


def fit_censoring(rows):
    result = {}
    for cohort in sorted({r['cohort'] for r in rows}):
        values = [r['survival_target'] for r in rows if r['cohort'] == cohort]
        result[cohort] = CensoringKM([r['time_days'] for r in values], [r['event'] for r in values])
    return result


def concordance(times, events, risks):
    t, e, r = np.asarray(times), np.asarray(events), np.asarray(risks)
    concordant, pairs, ties = 0., 0, 0
    for i in np.flatnonzero(e):
        other = np.flatnonzero(t > t[i])
        # A death and a censor at the same time are comparable, deaths tied are not.
        other = np.r_[other, np.flatnonzero((t == t[i]) & (e == 0))]
        delta = r[i]-r[other]
        pairs += len(other); ties += int(np.sum(delta == 0))
        concordant += float(np.sum(delta > 0)+.5*np.sum(delta == 0))
    return dict(c_index=concordant/pairs if pairs else None, comparable_pairs=pairs, risk_tied_pairs=ties)


def survival_metrics(rows, curves, edges, censoring):
    s = np.asarray(curves, float)
    edges = np.asarray(edges, float)
    if s.shape != (len(rows),len(edges)-1) or not len(rows) or not np.isfinite(s).all():
        raise ValueError('Invalid survival curves')
    if np.any((s < 0)|(s > 1)) or np.any(np.diff(s, axis=1)>1e-7):
        raise ValueError('Invalid survival probabilities')
    t = np.array([r['survival_target']['time_days'] for r in rows])
    e = np.array([r['survival_target']['event'] for r in rows])
    result = concordance(t, e, -s.mean(axis=1))
    days, briers = {}, []
    for j, day in enumerate(edges[1:]):
        cases = (e == 1)&(t <= day)
        controls = (t > day)|((e == 0)&(t == day))
        weights = np.zeros(len(rows)); unsupported = 0
        for i, row in enumerate(rows):
            if not cases[i] and not controls[i]:
                continue  # Censored before this endpoint: outcome is unknown.
            km = censoring[row['cohort']]
            # "Known alive through day" includes a censor exactly on day:
            # observability is C>=day, so the left limit is used for controls too.
            g = km.probability(t[i], left=True) if cases[i] else km.probability(day,left=True)
            if g is None:
                unsupported += 1
            else:
                weights[i] = 1/g
        brier = None if unsupported else float(np.sum(weights*(cases.astype(float)-(1-s[:,j]))**2)/len(rows))
        auc = None
        if not unsupported and np.any(cases) and np.any(controls):
            positive, negative = np.flatnonzero(cases), np.flatnonzero(controls)
            delta = (1-s[positive,j])[:,None]-(1-s[negative,j])[None,:]
            joint = weights[positive,None]*weights[negative][None,:]
            auc = float(np.sum(joint*((delta>0)+.5*(delta==0)))/joint.sum())
        days[str(int(day))] = dict(brier=brier, auroc=auc, observed_deaths=int(cases.sum()),
            known_survivors=int(controls.sum()), censored_before_endpoint=int((~cases&~controls).sum()),
            unsupported_ipcw_observations=unsupported,
            reason='training censoring distribution has no support' if unsupported else None)
        briers.append(brier)
    result.update(by_day=days, risk_definition='minus_mean_annual_survival',
        integrated_brier_annual_grid=float(np.trapezoid(np.r_[0.,briers],edges)/edges[-1])
            if all(b is not None for b in briers) else None,
        censoring_metric_contract='source-stratified training-only reverse KM; IPCW; annual grid; known-through-day left limits')
    return result
