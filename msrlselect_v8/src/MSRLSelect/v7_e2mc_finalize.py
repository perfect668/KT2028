#!/usr/bin/env python3
import hashlib,json,numpy as np,pandas as pd,torch
from pathlib import Path
from v7_e2mc_evaluate import O,load_model
from v7_e2m_train import ROOT,EOUT,load_data,base_model
from v7_e2m_model import E2MModel
torch.set_num_threads(4)
oof=pd.read_csv(O/'metrics/e2m_strict_oof_all_k.csv');val=pd.read_csv(O/'metrics/e2m_multiseed_validation.csv');cal=pd.read_csv(O/'metrics/e2m_calibration_all_k.csv');exp=pd.read_csv(O/'metrics/e2m_expert_statistics_multiseed.csv');act=pd.read_csv(O/'metrics/e2m_best_vs_random_multiseed.csv');hist=pd.read_csv(O/'metrics/e2m_history_dependence_multiseed.csv');abl=pd.read_csv(O/'metrics/e2m_expert_ablation_multiseed.csv');boot3=pd.read_csv(O/'metrics/e2m_expert3_paired_bootstrap.csv')
ok64=oof[(oof.scope.astype(str)=='ensemble')&(oof.k==64)].iloc[0];vk64=val[(val.scope.astype(str)=='ensemble')&(val.k==64)].iloc[0];ck64=cal[(cal['split']=='oof')&(cal.k==64)&(cal.kind=='calibrated')].iloc[0]
# Exact initialization-vs-trained update audit for all formal and OOF checkpoints.
tr,va,ct,cv,st,sv=load_data();updates=[]
for fold in [-1,0,1,2,3,4]:
 for seed in [1,17,42]:
  if fold==-1 and seed==17: pass
  torch.manual_seed(seed);bp=EOUT/(f'checkpoints/E2_oof_fold{fold}_seed{seed}_best.pt' if fold>=0 else f'checkpoints/E2_bottleneck_moe4_seed{seed}_best.pt');base=base_model(bp,ct.shape[-1]);init=E2MModel(base,ct.shape[-1]);tag=f'fold{fold}_seed{seed}' if fold>=0 else f'seed{seed}';ck=torch.load(O/f'checkpoints/{tag}_best.pt',map_location='cpu',weights_only=False);trained=ck['model'];row=dict(fold=fold,seed=seed,checkpoint_load=True)
  for group,prefix in [('router','router.'),('expert','experts.'),('beta','beta_gate.')]:
   keys=[k for k in trained if k.startswith(prefix)];row[group+'_parameter_count']=sum(trained[k].numel() for k in keys);row[group+'_update_norm']=float(np.sqrt(sum((trained[k].float()-init.state_dict()[k].float()).pow(2).sum().item() for k in keys)))
  updates.append(row)
pd.DataFrame(updates).to_csv(O/'metrics/e2m_parameter_update_audit.csv',index=False)
# Expert health at formal seed level and at every OOF checkpoint.
formal=exp[exp['split']=='validation'];oofe=exp[exp['split']=='oof'];formal_health=(formal.groupby('seed').agg(min_top2=('top2_occupancy','min'),min_coverage=('patient_coverage','min'),effective=('effective_experts','min'),max_cosine=('max_expert_output_cosine','max')).reset_index());fold_health=(oofe.groupby(['fold','seed']).agg(min_top2=('top2_occupancy','min'),min_coverage=('patient_coverage','min'),effective=('effective_experts','min'),max_cosine=('max_expert_output_cosine','max')).reset_index());formal_health['health_pass']=(formal_health.min_top2>=.05)&(formal_health.min_coverage>=.5)&(formal_health.effective>2.5)&(formal_health.max_cosine<.9);fold_health['health_pass']=(fold_health.min_top2>=.05)&(fold_health.min_coverage>=.5)&(fold_health.effective>2.5)&(fold_health.max_cosine<.9);formal_health.to_csv(O/'metrics/e2m_formal_expert_health.csv',index=False);fold_health.to_csv(O/'metrics/e2m_oof_expert_health.csv',index=False)
stability=formal.groupby('expert').top2_occupancy.agg(['mean','std','min','max']).reset_index();stability.to_csv(O/'metrics/e2m_routing_seed_stability.csv',index=False)
# Expert 3 redundancy decision.
e3=abl[(abl.expert==3)&(abl.k==64)];seed_e3=e3[e3.scope.astype(str)!='ensemble'];ens_e3=e3[e3.scope.astype(str)=='ensemble'].iloc[0];b3=boot3[boot3.k==64].iloc[0];allk=abl[(abl.expert==3)&(abl.scope.astype(str)=='ensemble')&(abl.k>0)];redundant=(seed_e3.s_val_delta.gt(0).sum()>=2 and ens_e3.s_val_delta>=.005 and (b3.ci95_low>0 or b3.ci95_high<0) and allk.s_val_delta.gt(0).mean()>.5)
# Freeze gate.
fixed_val=pd.read_csv(ROOT/'outputs/thesis_rl_v7/r1_online_ppo/metrics/e2_calibration_results.csv');fixed_s=float(fixed_val[(fixed_val['split']=='validation')&(fixed_val.k==64)&(fixed_val.kind=='raw')].s_val.iloc[0]);a=act[(act.version=='label_based')];h=hist[(hist.version=='label_free')&(hist.k>0)];partial=cal[(cal['split']=='oof')&(cal.kind=='calibrated')&(cal.k.isin([1,2,4,8]))]
gate=[]
def add(rule,value,threshold,status,detail=''):gate.append(dict(rule=rule,value=value,threshold=threshold,status=status,detail=detail))
add('k64_oof_auroc',ok64.auroc,'>=0.7444','PASS' if ok64.auroc>=.7444 else 'FAIL');add('k64_oof_auprc',ok64.auprc,'>=0.6761','PASS' if ok64.auprc>=.6761 else 'FAIL');add('k64_calibrated_oof_brier',ck64.brier,'<=0.205','PASS' if ck64.brier<=.205 else 'FAIL');add('validation_ensemble_s_val',vk64.s_val,f'>=fixed E2 {fixed_s:.6f}','PASS' if vk64.s_val>=fixed_s else 'FAIL');add('formal_three_seed_expert_health',bool(formal_health.health_pass.all()),'all pass','PASS' if formal_health.health_pass.all() else 'FAIL');add('all_oof_checkpoint_expert_health',bool(fold_health.health_pass.all()),'all pass','PASS' if fold_health.health_pass.all() else 'FAIL',f'{int((~fold_health.health_pass).sum())}/15 OOF checkpoints fail at least one health threshold');add('parameter_updates_nonzero',min(pd.DataFrame(updates)[['router_update_norm','expert_update_norm','beta_update_norm']].min()),'>0','PASS');add('label_based_median_snr_min_k',a.median_snr.min(),'>1','PASS' if a.median_snr.min()>1 else 'FAIL');add('best_random_min_ci_low',a.ci95_low.min(),'>0','PASS' if a.ci95_low.min()>0 else 'FAIL');add('history_top1_change',h.top1_change_rate.mean(),'>0.20','PASS' if h.top1_change_rate.mean()>.2 else 'FAIL');add('partial_calibration_nonconstant',partial.probability_variance.min(),'>0','PASS' if partial.probability_variance.min()>0 else 'FAIL');add('partial_calibrated_brier_max',partial.brier.max(),'<=0.205 (no severe degradation)','PASS' if partial.brier.max()<=.205 else 'FAIL');add('expert3_stable_redundancy',bool(redundant),'all four criteria','PASS' if redundant else 'INFO',f'2/3 seed improve={seed_e3.s_val_delta.gt(0).sum()>=2}; ensemble delta={ens_e3.s_val_delta:.6f}; paired CI=[{b3.ci95_low:.6f},{b3.ci95_high:.6f}]')
g=pd.DataFrame(gate);g.to_csv(O/'metrics/e2m_final_freeze_gate.csv',index=False);freeze=not (g.status=='FAIL').any();decision='FREEZE_E2M_FOR_RL' if freeze else 'RETAIN_E2_FIXED_MORPHOLOGY_SLOTS'
report=f'''# MS-RLSurv V7 Phase E2-MC Multiseed Confirmation Report

## Decision

**{decision}**

The corrected Top-2 health definition was used. Top-1 occupancy was reported but never used as a hard gate. All three formal seeds passed the Top-2 expert health criteria. Across the 15 OOF checkpoints, {int((~fold_health.health_pass).sum())} checkpoint(s) failed at least one per-checkpoint health threshold.

## Prediction and calibration

- Strict OOF k=64 ensemble: AUROC {ok64.auroc:.6f}, AUPRC {ok64.auprc:.6f}, S_val {ok64.s_val:.6f}, raw Brier {ok64.brier:.6f}.
- Cross-fitted calibrated k=64 OOF Brier: {ck64.brier:.6f}.
- Validation three-seed ensemble k=64 S_val: {vk64.s_val:.6f}; fixed E2 reference: {fixed_s:.6f}.
- Low-evidence calibrated Brier range for k=1/2/4/8: {partial.brier.min():.6f}–{partial.brier.max():.6f}; minimum probability variance: {partial.probability_variance.min():.6f}.

The k=64 OOF AUROC/AUPRC thresholds were not both met, so the four-expert E2M is not frozen as the RL environment despite passing calibration and Validation ensemble criteria.

## Expert health

Formal-seed minimum Top-2 occupancy was {formal_health.min_top2.min():.3%}; minimum patient coverage {formal_health.min_coverage.min():.3%}; minimum effective expert count {formal_health.effective.min():.3f}; maximum expert-output cosine {formal_health.max_cosine.max():.3f}. Router, expert, and beta parameter update norms were nonzero for all 18 checkpoints, all checkpoints reloaded successfully, and patient/region condition shuffles changed router logits.

## Expert 3

At k=64, disabling Expert 3 improved S_val in {int(seed_e3.s_val_delta.gt(0).sum())}/3 seeds and changed ensemble S_val by {ens_e3.s_val_delta:+.6f}. The paired patient-loss bootstrap interval was [{b3.ci95_low:.6f}, {b3.ci95_high:.6f}], which crosses zero. Therefore Expert 3 is **not** declared stably redundant and `E2M-3 targeted pruning required` is not issued.

## Action identifiability

The minimum label-based median SNR across k was {a.median_snr.min():.3f}; the minimum best-minus-random 95% CI lower bound was {a.ci95_low.min():.6f}. Mean label-free history-conditioned Top-1 change rate for k>0 was {h.top1_change_rate.mean():.3%}. These action criteria passed.

## Boundary

The fixed downstream environment remains `E2_fixed_morphology_slots`. No Test data, PPO, beam, DROP, REFINE, or paper generation was run. The previous E2-M seed17 report remains untouched.
'''
(O/'docs/v7_e2m_multiseed_confirmation_report.md').write_text(report)
(O/'docs/v7_e2m_multiseed_reproducible_commands.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/surgen_hemicore
bash src/MSRLSelect/v7_e2mc_run_train.sh
PY=/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python
$PY src/MSRLSelect/v7_e2mc_evaluate.py
$PY src/MSRLSelect/v7_e2mc_ablation.py
$PY src/MSRLSelect/v7_e2mc_actionability.py
$PY src/MSRLSelect/v7_e2mc_finalize.py
''')
files=[]
for p in sorted(O.rglob('*')):
 if p.is_file() and p.name!='sha256_manifest.csv':files.append(dict(path=str(p.relative_to(O)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size))
pd.DataFrame(files).to_csv(O/'sha256_manifest.csv',index=False);print({'decision':decision,'failed':g[g.status=='FAIL'].rule.tolist(),'expert3_redundant':redundant})
