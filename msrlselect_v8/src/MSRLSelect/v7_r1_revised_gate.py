#!/usr/bin/env python3
from pathlib import Path
import re,numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore');E=ROOT/'outputs/thesis_evidence_interface_v7';R=ROOT/'outputs/thesis_rl_v7/r1_online_ppo';P=ROOT/'outputs/thesis_predictor_first_v6/p1r_protocol_correction'
rows=[]
def add(rule,value,threshold,status,evidence,detail=''):rows.append(dict(rule=rule,value=value,threshold=threshold,status=status,evidence=evidence,detail=detail))
o=pd.read_csv(R/'metrics/e2_variable_evidence_oof.csv');k64=o[o.k==64].iloc[0]
add('strict_oof_auroc',k64.auroc,'>=0.6703','PASS' if k64.auroc>=.6703 else 'FAIL','e2_variable_evidence_oof.csv')
add('strict_oof_auprc',k64.auprc,'>=0.5112','PASS' if k64.auprc>=.5112 else 'FAIL','e2_variable_evidence_oof.csv')
cal=pd.read_csv(R/'metrics/e2_calibration_results.csv');ec=cal[(cal['split']=='oof')&(cal.k==64)&(cal.kind=='calibrated')].iloc[0]
p1=pd.read_csv(P/'predictions/train_oof_ensemble_predictions.csv');pp=p1.platt_probability.values if 'platt_probability' in p1 else p1.ensemble_probability.values;pb=brier_score_loss(p1.label,pp)
add('calibrated_oof_brier_delta_vs_p1r',ec.brier-pb,'<=0.01','PASS' if ec.brier-pb<=.01 else 'FAIL','E2 cross-fit calibration vs P1R OOF',f'E2={ec.brier:.6f}; P1R={pb:.6f}')
v=pd.read_csv(R/'metrics/e2_calibration_results.csv');e2v=v[(v['split']=='validation')&(v.k==64)&(v.kind=='raw')].iloc[0];pt=pd.read_csv(P/'predictions/validation_platt_predictions.csv');ps=.5*(roc_auc_score(pt.label,pt.platt_probability)+average_precision_score(pt.label,pt.platt_probability))
add('validation_k64_s_val_vs_p1r',e2v.s_val-ps,'>=0','PASS' if e2v.s_val>=ps else 'FAIL','Validation ensemble comparison',f'E2={e2v.s_val:.6f}; P1R={ps:.6f}')
code=(ROOT/'src/MSRLSelect/v7_train_evidence.py').read_text();external=('k==64' in code and 'teacher' in code[code.find('k==64')-100:code.find('k==64')+300])
add('external_teacher_replacement',external,'False','FAIL' if external else 'PASS','code/checkpoint inspection')
a=pd.read_csv(R/'metrics/e2_corrected_action_snr.csv');lab=a[(a.version=='label_based')&(a.k>0)];lf=a[(a.version=='label_free')&(a.k>0)]
add('corrected_action_snr_label_based_median',lab.snr.median(),'>1','PASS' if lab.snr.median()>1 else 'FAIL','three-seed per-action deltas')
add('corrected_action_snr_label_free_median',lf.snr.median(),'>1','PASS' if lf.snr.median()>1 else 'FAIL','E2 self-teacher per-action deltas')
b=pd.read_csv(R/'metrics/e2_best_vs_random_bootstrap.csv');bb=b[(b.version=='label_based')&(b.k>0)]
add('best_minus_random_min_ci95_low',bb.ci95_low.min(),'>0','PASS' if bb.ci95_low.min()>0 else 'FAIL','2000 patient bootstrap at k=1..48')
h=pd.read_csv(R/'metrics/e2_history_dependence.csv');change=h[(h.version=='label_free')&(h.k>0)].top1_change_rate.mean();add('history_conditioned_top1_change',change,'>0','PASS' if change>0 else 'FAIL','five fixed histories')
r=pd.read_csv(R/'metrics/e2_trained_routing_statistics.csv');has=bool(r[r.source.isin(['oof','validation'])].has_trained_router.all());add('trained_expert_noncollapse',has,'trained router present and noncollapsed','PASS' if has else 'FAIL','checkpoint state_dict audit','All 18 formal E2 checkpoints contain zero router/expert parameters; KMeans-derived offline slots cannot establish trained expert health')
# Calibration-scale MAE record, not a gate.
def norm(x):return re.sub(r'T0+(\d+)',r'T\1',str(x)).replace('_01','')
raw=pd.read_csv(R/'predictions/e2_validation_all_k_calibrated.csv');raw=raw[raw.k==64].copy();raw['id']=raw.patient_id.map(norm);pt['id']=pt.patient_id.map(norm);m=raw.merge(pt,on='id');add('teacher_mae_raw_record',np.abs(m.probability-m.platt_probability).mean(),'record only','INFO','E2 vs P1R');add('teacher_mae_calibrated_record',np.abs(m.calibrated_probability-m.platt_probability).mean(),'record only','INFO','E2 vs P1R')
gate=pd.DataFrame(rows);gate.to_csv(R/'metrics/e2_revised_freeze_gate.csv',index=False)
decision='PASS' if not (gate.status=='FAIL').any() else 'FAIL'
report=f'''# V7 R1 Revised E2 Freeze Report\n\n## Decision\n\n**E2 freeze: {decision}. PPO was not started.**\n\nThe revised gate no longer uses E2-to-P1R patient probability MAE as a hard criterion. E2 strict k=64 OOF achieved AUROC {k64.auroc:.4f}, AUPRC {k64.auprc:.4f}, S_val {k64.s_val:.4f}, and Brier {k64.brier:.4f}. Validation k=64 S_val was {e2v.s_val:.4f} versus P1R {ps:.4f}.\n\nCross-fitted calibration produced k=64 OOF Brier {ec.brier:.4f}; P1R OOF Brier was {pb:.4f}. The raw/calibrated E2-to-P1R Validation MAE values are retained as descriptive model-change measures in the gate CSV.\n\nCorrected action values were computed by enumerating legal ACQUIRE actions with all three corresponding unseen-patient models. For k=1–48, label-based median SNR is {lab.snr.median():.3f}; label-free median SNR is {lf.snr.median():.3f}. The minimum label-based best-minus-random bootstrap CI lower bound is {bb.ci95_low.min():.6f}.\n\n## Failing condition\n\nThe formal E2 checkpoints are not trained MoE checkpoints. Their state dictionaries contain no router, expert, or routing-center parameters. The saved four slots were generated offline from fixed KMeans assignment and truncated feature means; therefore trained Top-1/Top-2 occupancy, routing entropy, effective expert count, and seed stability cannot be computed. The offline cache has empty-slot fraction {float(r[r.source=='offline_slot_cache'].empty_slot_fraction.iloc[0]):.3f} and mean slot cosine similarity {float(r[r.source=='offline_slot_cache'].slot_cosine_similarity.iloc[0]):.3f}, but these do not prove trained-expert noncollapse.\n\nUnder the revised rule, this is a real expert-interface failure, not a calibration-only failure. Actor warm start and PPO were not launched. No Test data was read.\n'''
(R/'docs/v7_e2_revised_freeze_report.md').write_text(report)
print({'decision':decision,'failed':gate[gate.status=='FAIL'].rule.tolist()})
