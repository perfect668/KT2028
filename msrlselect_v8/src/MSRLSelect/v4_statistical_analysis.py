#!/usr/bin/env python3
from pathlib import Path
import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix

METHODS = ('Random', 'HeMiCoRe', 'MS-RLSurv_revision')
SEEDS = (1, 17, 42)

def metric_dict(y, p):
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'auroc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p),
            'accuracy': accuracy_score(y, pred), 'f1': f1_score(y, pred, zero_division=0),
            'sensitivity': tp / max(1, tp + fn), 'specificity': tn / max(1, tn + fp)}

def load_pred(root, seed, method):
    name = {'Random': f'random_seed{seed+1000}', 'HeMiCoRe': 'HeMiCoRe',
            'MS-RLSurv_revision': 'MS-RLSelect_revision'}[method]
    p = root / f'predictors/seed{seed}/{name}/test_predictions.csv'
    x = pd.read_csv(p)
    return x[['case_id', 'label', 'death_risk']].rename(columns={'death_risk': 'probability'})

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); ap.add_argument('--bootstrap', type=int, default=2000); a = ap.parse_args()
    out = a.root; rows = []; data = {}
    for method in METHODS:
        for seed in SEEDS:
            x = load_pred(out, seed, method); data[(method, seed)] = x
            rows.append({'method': method, 'seed': seed, **metric_dict(x.label.to_numpy(), x.probability.to_numpy())})
    pd.DataFrame(rows).to_csv(out / 'chapter3_multiseed_results.csv', index=False)
    ens_frames = {}; ens_rows = []
    for method in METHODS:
        base = data[(method, 1)][['case_id', 'label']].copy()
        ps = [data[(method, s)].set_index('case_id').loc[base.case_id, 'probability'].to_numpy() for s in SEEDS]
        base['probability'] = np.mean(ps, axis=0); ens_frames[method] = base
        ens_rows.append({'method': method, 'seed': 'ensemble', **metric_dict(base.label.to_numpy(), base.probability.to_numpy())})
    pd.concat([x.assign(method=k) for k, x in ens_frames.items()], ignore_index=True).to_csv(out / 'chapter3_ensemble_predictions.csv', index=False)
    pd.DataFrame(ens_rows).to_csv(out / 'chapter3_ensemble_results.csv', index=False)
    rng = np.random.default_rng(4242); y = ens_frames['Random'].label.to_numpy(); strata = [np.flatnonzero(y == c) for c in (0, 1)]
    boot = {m: {'auroc': [], 'auprc': []} for m in METHODS}; diff = {}
    for _ in range(a.bootstrap):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in strata])
        for m in METHODS:
            yy = ens_frames[m].label.to_numpy()[idx]; pp = ens_frames[m].probability.to_numpy()[idx]
            boot[m]['auroc'].append(roc_auc_score(yy, pp)); boot[m]['auprc'].append(average_precision_score(yy, pp))
        for m in METHODS[1:]:
            diff.setdefault((m, 'Random'), {'auroc': [], 'auprc': []})
            for metric in ('auroc', 'auprc'): diff[(m, 'Random')][metric].append(boot[m][metric][-1] - boot['Random'][metric][-1])
    ci = []
    for m in METHODS:
        for metric in ('auroc', 'auprc'):
            v = np.asarray(boot[m][metric]); ci.append({'comparison': m, 'metric': metric, 'estimate': float(v.mean()), 'ci_low': float(np.quantile(v, .025)), 'ci_high': float(np.quantile(v, .975))})
    for (m, b), dd in diff.items():
        for metric, vv in dd.items():
            v = np.asarray(vv); ci.append({'comparison': f'{m} - {b}', 'metric': metric, 'estimate': float(v.mean()), 'ci_low': float(np.quantile(v, .025)), 'ci_high': float(np.quantile(v, .975))})
    pd.DataFrame(ci).to_csv(out / 'bootstrap_confidence_intervals.csv', index=False)
    tests = []
    for m in METHODS[1:]:
        for metric in ('auroc', 'auprc'):
            v = np.asarray(diff[(m, 'Random')][metric]); p = 2 * min(np.mean(v <= 0), np.mean(v >= 0))
            tests.append({'comparison': f'{m} vs Random', 'metric': metric, 'estimate': float(v.mean()), 'p_raw': float(min(1, p))})
    t = pd.DataFrame(tests); order = np.argsort(t.p_raw.to_numpy()); adj = np.empty(len(t));
    for rank, i in enumerate(order): adj[i] = min(1, (len(t) - rank) * t.iloc[i].p_raw)
    t['p_holm'] = adj; t.to_csv(out / 'paired_significance_tests.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(ens_rows).to_string(index=False))
if __name__ == '__main__': main()
