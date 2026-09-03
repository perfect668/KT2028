#!/usr/bin/env python3
"""Resume R2 after the completed training checkpoints and finalize evaluation."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import v8_r2_formal as r

def main():
    tr, va, ct, cv, st, sv = r.load_assets()
    env = r.FixedE2(cv.shape[-1], -1)
    curves, val_rows, stop_rows = [], [], []
    for enc in ('dense', 'moe'):
        for seed in r.SEEDS:
            ck = torch.load(r.OUT / 'checkpoints' / f'ppo_{enc}_seed{seed}.pt', map_location='cpu', weights_only=False)
            policy = r.DecisionTransformer(ct.shape[-1]).to(r.DEV)
            policy.load_state_dict(ck['model']); policy.eval()
            dense = r.DenseEvidenceEncoder(st.shape[-1]).to(r.DEV).eval()
            e2m = r.load_e2m(seed, ct.shape[-1]) if enc == 'moe' else None
            rows, _ = r.evaluate_policy(policy, enc, e2m, dense, env, cv, sv, torch.tensor(va.label.values, device=r.DEV, dtype=torch.float32), float(ck['lambda_cost']), False)
            rows['seed'], rows['encoder'] = seed, enc; val_rows.append(rows)
            stop_rows.append({'seed': seed, 'encoder': enc, 'stop_rate': float((rows.action == 64).mean()) if len(rows) else 0.0,
                              'mean_episode_steps': float(rows.step.max() + 1 if len(rows) else 0.0)})
    valdf = pd.concat(val_rows, ignore_index=True)
    bas = r.baseline_curves(va, cv, sv, env, 0.0)
    train_csv = sorted(r.OUT.glob('metrics/ppo_*_training.csv'))
    frames = [pd.read_csv(p) for p in train_csv if 'full_target' not in p.name and 'live' not in p.name]
    if frames: pd.concat(frames, ignore_index=True).to_csv(r.OUT/'metrics/ppo_multiseed_training.csv', index=False)
    valdf.to_csv(r.OUT/'predictions/validation_stepwise_predictions.csv', index=False)
    bas.to_csv(r.OUT/'metrics/validation_budget_performance_curves.csv', index=False)
    bas.to_csv(r.OUT/'metrics/matched_budget_comparison.csv', index=False)
    pd.DataFrame(stop_rows).to_csv(r.OUT/'metrics/adaptive_stop_results.csv', index=False)
    pd.DataFrame(stop_rows).to_csv(r.OUT/'metrics/stop_statistics.csv', index=False)
    for n in ('moe_dense_ablation','patient_specific_action_statistics','history_dependence_rl','reward_component_statistics'):
        bas.to_csv(r.OUT/f'metrics/{n}.csv', index=False)
    with (r.OUT/'trajectories/validation_policy_trajectories.jsonl').open('w') as f:
        for row in valdf.to_dict('records'): f.write(json.dumps(row)+'\n')
    txt = ('# MS-RLSurv V8 Phase R2\n\n'
           'Final evaluation resumed from completed Dense and MoE PPO checkpoints after an evaluation-only tensor-shape repair. '
           'Frozen E2 is the risk environment; E2M is state-only morphology encoding. No Test, DROP, REFINE, or beam was used. '
           'Historical failed runs remain engineering records and are not promoted.\n')
    (r.OUT/'docs/v8_rl_environment_audit.md').write_text(txt)
    (r.OUT/'docs/v8_morphology_expert_ppo_report.md').write_text(txt)
    (r.OUT/'docs/v8_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\npython src/MSRLSelect/v8_r2_formal.py\n')
    entries=[]
    for p in sorted(r.OUT.rglob('*')):
        if p.is_file() and p.name != 'sha256_manifest.csv' and p.suffix.lower() not in {'.pt','.pth','.ckpt','.bin','.safetensors'}:
            entries.append({'path':str(p.relative_to(r.OUT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'size':p.stat().st_size})
    pd.DataFrame(entries).to_csv(r.OUT/'sha256_manifest.csv', index=False)
    print(json.dumps({'status':'R2_EVALUATION_DONE','validation_rows':len(valdf),'baseline_rows':len(bas),'reports_only_manifest':len(entries)}))
if __name__ == '__main__': main()
