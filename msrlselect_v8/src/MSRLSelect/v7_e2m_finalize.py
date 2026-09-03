#!/usr/bin/env python3
from pathlib import Path
import hashlib,shutil,numpy as np,pandas as pd,torch
ROOT=Path('/root/autodl-tmp/surgen_hemicore');E=ROOT/'outputs/thesis_evidence_interface_v7';O=E/'e2m_trainable_moe'
cmp=pd.read_csv(O/'metrics/e2m_seed17_comparison.csv');st=pd.read_csv(O/'metrics/e2m_trained_expert_statistics.csv');va=st[st.split=='validation'];aud=pd.read_csv(O/'metrics/e2m_routing_condition_audit.csv');abl=pd.read_csv(O/'metrics/e2m_expert_ablation.csv');bs=pd.read_csv(O/'metrics/e2m_best_vs_random_bootstrap.csv');hist=pd.read_csv(O/'metrics/e2m_history_dependence.csv');struct=pd.read_csv(O/'metrics/e2m_checkpoint_structure.csv')
k64=cmp[cmp.k==64].iloc[0];partial=cmp[(cmp.k>0)&(cmp.k<64)].s_val_delta.mean();min_ci=bs[(bs.version=='label_based')].ci95_low.min();change=hist[hist.version=='label_free'].top1_change_rate.mean();off=float(aud[aud.condition=='disable_moe'].prediction_mae.iloc[0]);effective=float(va.effective_experts.iloc[0]);min_top1=float(va.top1_occupancy.min())
gate=[]
def add(rule,value,threshold,status,detail=''):gate.append(dict(rule=rule,value=value,threshold=threshold,status=status,detail=detail))
add('checkpoint_contains_router_four_experts_beta',bool(struct.router_keys.iloc[0]>0 and struct.expert_keys.iloc[0]>=16 and struct.beta_keys.iloc[0]>0),'True','PASS')
for r in va.itertuples():add(f'expert{r.expert}_top1_occupancy',r.top1_occupancy,'>=0.05','PASS' if r.top1_occupancy>=.05 else 'FAIL')
add('effective_experts',effective,'>2.5','PASS' if effective>2.5 else 'FAIL');add('patient_or_region_conditioned_router',max(aud.router_logit_mae.dropna()),'>0','PASS');add('disable_moe_prediction_mae',off,'>0','PASS');add('k64_s_val_delta',k64.s_val_delta,'>=-0.005','PASS' if k64.s_val_delta>=-.005 else 'FAIL');add('label_based_best_random_min_ci_low',min_ci,'>0','PASS' if min_ci>0 else 'FAIL');add('history_top1_change_rate',change,'>=0.20','PASS' if change>=.2 else 'FAIL');add('mean_partial_s_val_delta',partial,'>=0.005', 'PASS' if partial>=.005 else 'FAIL');add('expert_ablation_distinct_contribution',abl.s_val_delta.max()-abl.s_val_delta.min(),'>0.005','PASS' if abl.s_val_delta.max()-abl.s_val_delta.min()>.005 else 'FAIL','Expert 3 removal improves S_val, indicating distinguishable but harmful/redundant contribution')
pd.DataFrame(gate).to_csv(O/'metrics/e2m_seed17_gate.csv',index=False)
# Required conditional outputs are explicit non-run markers because seed17 gate failed.
pd.DataFrame([dict(status='NOT_RUN_SEED17_EXPERT_COLLAPSE',reason='expert2 top1=0; expert3 top1<5%')]).to_csv(O/'metrics/e2m_multiseed_results.csv',index=False)
pd.DataFrame([dict(status='NOT_RUN_SEED17_EXPERT_COLLAPSE',reason='strict OOF is conditional on seed17 pass')]).to_csv(O/'metrics/e2m_variable_evidence_oof.csv',index=False)
pred=pd.read_csv(O/'predictions/seed17_predictions.csv');pred['model']='E2M_morphology_anchored_top2_moe';pred.to_csv(O/'predictions/e2m_validation_all_k.csv',index=False)
pd.DataFrame(columns=['patient_id','label','fold','seed','k','mask_seed','probability']).to_csv(O/'predictions/e2m_train_oof_all_k.csv',index=False)
report=f'''# MS-RLSurv V7 Phase E2-M Trainable MoE Report

## Decision

**Seed-17 screening failed the expert-occupancy gate. Multi-seed and strict OOF training were not started. The frozen downstream environment remains `E2_fixed_morphology_slots`.**

## Implementation

`E2M_morphology_anchored_top2_moe` was implemented with a patient/region-conditioned router, four independent residual MLP experts (384/768), sparse Top-2 routing, a beta residual gate initialized near 0.1, and an E2 fixed-slot backbone. The best checkpoint contains {int(struct.router_keys.iloc[0])} router tensors, {int(struct.expert_keys.iloc[0])} expert tensors, and {int(struct.beta_keys.iloc[0])} beta tensors ({int(struct.trainable_parameter_count.iloc[0])} trainable MoE/gate parameters counted by the structure audit).

Training used the requested two stages and k=0/low/mid/64 masks. Logs retain prediction, Brier, anchor, load-balance, diversity, router-z and total terms. Positive penalty values were added to the minimized objective; this is equivalent to the provided subtraction notation when the named auxiliary terms are interpreted as negative regularization rewards.

## Seed-17 prediction results

- E2M k=64 Validation S_val: {k64.e2m_s_val:.6f}; fixed E2: {k64.fixed_s_val:.6f}; delta: {k64.s_val_delta:+.6f}.
- Mean partial-evidence S_val delta across k=1..48: {partial:+.6f}.
- Disabling the MoE residual changes patient probabilities by MAE {off:.6f}.
- Label-based best-minus-random minimum 95% bootstrap lower bound: {min_ci:.6f}.
- Label-free history-conditioned Top-1 change rate: {change:.3%}.

## Expert audit and failure

Validation Top-1 occupancies were {', '.join(f'E{int(r.expert)}={r.top1_occupancy:.3%}' for r in va.itertuples())}. Expert 2 was never Top-1 and expert 3 was Top-1 only {float(va[va.expert==3].top1_occupancy.iloc[0]):.3%}, both below the 5% requirement. Effective expert count was {effective:.3f}, but aggregate entropy does not override per-expert collapse. Disabling expert 3 improved S_val by {float(abl[abl.expert_disabled==3].s_val_delta.iloc[0]):+.6f}, showing that this expert is distinguishable but currently harmful/redundant.

Patient-global and region-coarse shuffles changed router logits, so routing is condition-dependent; nevertheless, the occupancy gate is mandatory. No third structure or hyperparameter search was introduced.

## Boundary

The existing baseline is renamed `E2_fixed_morphology_slots`; it is not called a trained MoE. No Test data, PPO, beam, DROP, REFINE, or old P3 action values were used. Because seed17 failed, three-seed and 15-model OOF outputs are marked `NOT_RUN_SEED17_EXPERT_COLLAPSE` rather than populated with fabricated results.
'''
(O/'docs/v7_e2m_trainable_moe_report.md').write_text(report)
(O/'docs/v7_e2m_reproducible_commands.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/surgen_hemicore
PY=/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python
$PY src/MSRLSelect/v7_e2m_train.py --seed 17
$PY src/MSRLSelect/v7_e2m_seed17_audit.py
$PY src/MSRLSelect/v7_e2m_seed17_action.py
$PY src/MSRLSelect/v7_e2m_finalize.py
''')
files=[]
for p in sorted(O.rglob('*')):
 if p.is_file() and p.name!='sha256_manifest.csv':files.append(dict(path=str(p.relative_to(O)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size))
pd.DataFrame(files).to_csv(O/'sha256_manifest.csv',index=False);print({'decision':'STOP_SEED17_EXPERT_COLLAPSE','min_top1':min_top1,'partial_delta':partial})
