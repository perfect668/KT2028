#!/usr/bin/env python3
from pathlib import Path
import re, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path('/root/autodl-tmp/surgen_hemicore')
SRC=ROOT/'outputs/thesis_evidence_interface_v7'
OUT=ROOT/'outputs/thesis_rl_v7/r1_online_ppo'
OUT.joinpath('metrics').mkdir(parents=True,exist_ok=True);OUT.joinpath('docs').mkdir(exist_ok=True);OUT.joinpath('logs').mkdir(exist_ok=True)
rows=[]
def add(section,metric,value,status='INFO',source='',detail=''):
 rows.append(dict(section=section,metric=metric,value=value,status=status,source=source,detail=detail))

d=pd.read_csv(SRC/'metrics/final_candidate_multiseed.csv')
g=d.groupby(['seed','k'],as_index=False).agg(auroc=('auroc','mean'),auprc=('auprc','mean'),s_val=('s_val','mean'),brier=('brier','mean'))
for r in g.itertuples():
 for m in ['auroc','auprc','s_val','brier']: add('validation_seed_k',f'seed{r.seed}_k{r.k}_{m}',getattr(r,m),source='final_candidate_multiseed.csv')
s=g.groupby('k').agg(['mean','std'])
for k in s.index:
 for m in ['auroc','auprc','s_val','brier']:
  add('validation_summary',f'k{k}_{m}_mean',s.loc[k,(m,'mean')],source='three seed means')
  add('validation_summary',f'k{k}_{m}_sd',s.loc[k,(m,'std')],source='three seed SD')

o=pd.read_csv(SRC/'metrics/final_candidate_oof.csv')
for k in [0,1,2,4,8,16,32,48,64]:
 q=o[o.k==k]
 if q.empty:
  for m in ['auroc','auprc','brier']: add('strict_oof',f'k{k}_{m}',np.nan,'MISSING',source='final_candidate_oof.csv',detail='Strict OOF contains k=64 only')
 else:
  y=q.label.values;p=q.probability.values
  for m,v in [('auroc',roc_auc_score(y,p)),('auprc',average_precision_score(y,p)),('brier',brier_score_loss(y,p))]:add('strict_oof',f'k{k}_{m}',v,source='strict 5-fold x 3-seed ensemble')

def norm(x):return re.sub(r'T0+(\d+)',r'T\1',str(x)).replace('_01','')
p=pd.read_csv(SRC/'predictions/validation_patient_predictions.csv');p=p[p.k==64].copy();t=pd.read_csv(ROOT/'outputs/thesis_predictor_first_v6/p1r_protocol_correction/predictions/validation_platt_predictions.csv');p['id']=p.patient_id.map(norm);t['id']=t.patient_id.map(norm);m=p.merge(t,on='id');z=m.probability.values;q=m.platt_probability.values;y=m.label_x.values
mae=float(np.abs(z-q).mean()); sm=.5*(roc_auc_score(y,z)+average_precision_score(y,z)); st=.5*(roc_auc_score(y,q)+average_precision_score(y,q))
add('teacher_endpoint','k64_internal_forward_teacher_mae',mae,'FAIL' if mae>.02 else 'PASS',source='validation_patient_predictions vs P1R Platt')
add('teacher_endpoint','k64_model_s_val',sm,source='internal forward');add('teacher_endpoint','k64_teacher_s_val',st,source='P1R frozen teacher');add('teacher_endpoint','k64_s_val_drop',st-sm,'FAIL' if st-sm>.01 else 'PASS')
code=(ROOT/'src/MSRLSelect/v7_train_evidence.py').read_text();external=('torch.where' in code and 'k==64' in code)
add('teacher_endpoint','external_teacher_endpoint_replacement',bool(external),'FAIL' if external else 'PASS',source='static code inspection',detail='No k=64 branch replacing model output was found')

a=pd.read_csv(SRC/'metrics/action_signal_to_noise.csv');a=a[a.model=='E2']
add('action_identifiability','available_training_seeds',1,'FAIL',source='action_signal_to_noise.csv',detail='Rows have no seed column; generated from seed17 model only')
add('action_identifiability','reported_seed17_action_snr_mean',a[a.k>0].action_snr.mean(),source='existing CSV')
add('action_identifiability','action_snr_formula','std(delta_probability_across_patients) / std(all fine tensor elements)','INVALID',source='v7_train_evidence.py',detail='Denominator is not three-seed ensemble prediction noise; candidate actions were not enumerated within each state')
add('action_identifiability','action_snr_numerator',a[a.k>0].candidate_action_std.mean(),source='existing CSV mean')
add('action_identifiability','action_snr_denominator','global fine tensor element SD','INVALID')
add('action_identifiability','three_seed_majority_snr_gt_1','not available','FAIL',detail='No seed-specific action enumeration')
add('action_identifiability','best_vs_random_gain_mean',a[a.k>0].best_vs_random_gain.mean(),'PASS' if a[a.k>0].best_vs_random_gain.mean()>0 else 'FAIL')
add('action_identifiability','best_vs_random_bootstrap_ci95','not available','FAIL',detail='No patient/state-level action samples exist for bootstrap')
add('history','history_spearman','not available','FAIL');add('history','top1_change_rate','not available','FAIL')

r=pd.read_csv(SRC/'metrics/moe_routing_statistics.csv')
for x in r.itertuples():add('experts',f'cluster{x.cluster}_patch_fraction',x.patch_fraction,source='Train spherical KMeans initialization')
add('experts','post_training_router_occupancy','not available','FAIL');add('experts','post_training_routing_entropy','not available','FAIL');add('experts','post_training_slot_similarity','not available','FAIL')
add('experts','expert_collapse_assessment','not assessable','FAIL',detail='Saved statistics describe KMeans clusters, not learned Top-2 expert routing')

gate=pd.DataFrame(rows);gate.to_csv(OUT/'metrics/e2_freeze_gate.csv',index=False)
failed=gate[gate.status=='FAIL']
report=f'''# V7 R1 E2 Freeze Gate Report\n\n## Decision\n\n**NO-GO. Online PPO was not started.**\n\nThe mandatory teacher endpoint gate failed: k=64 internal-forward probability MAE is **{mae:.6f}**, above 0.02. The k=64 S_val drop is {st-sm:.6f}. Static inspection found no external k=64 teacher substitution.\n\n## Additional evidence gaps\n\n- Strict OOF predictions exist only for k=64, not all evidence counts.\n- Existing action SNR is seed17-only and uses fine-tensor element SD as denominator, not three-seed prediction noise.\n- No patient/state action enumeration exists for a valid best-vs-random bootstrap interval.\n- No multi-history Spearman or Top-1 change statistics exist.\n- The routing CSV contains spherical KMeans initialization occupancy, not post-training learned-expert occupancy, entropy, or slot similarity.\n\nThese are not treated as PPO results. Per the R1 stop rule, no actor warm start, environment rollout, or PPO checkpoint was produced. No Test data was read.\n\nMachine-readable details: `metrics/e2_freeze_gate.csv` ({len(gate)} rows; {len(failed)} failed checks).\n'''
(OUT/'docs/v7_r1_online_ppo_report.md').write_text(report)
(OUT/'docs/v7_r1_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python src/MSRLSelect/v7_r1_freeze_gate.py\n')
print({'decision':'NO-GO','mae':mae,'s_val_drop':st-sm,'failed_checks':len(failed)})
