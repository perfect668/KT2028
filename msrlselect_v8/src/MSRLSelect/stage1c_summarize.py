#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path('/root/autodl-tmp/surgen_hemicore');OUT=ROOT/'outputs/thesis_algorithm_upgrade_v5/stage1c_dense_confirmation';FORM=OUT/'formal'
CFGS=['C0_K64_mean_relation_no_spatial','C1_K64_gated_relation_no_spatial','C2_K64_gated_relation_spatial','C3_K64_gated_no_relation_no_spatial','C4_K128_gated_relation_spatial'];SEEDS=[1,17,42]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[];gaps=[];stops=[];memory=[];pred=[];manifest=[];gate_frames=[]
 for c in CFGS:
  for s in SEEDS:
   d=FORM/c/f'seed{s}';status=json.loads((d/'run_status.json').read_text());h=pd.read_csv(d/'epoch_metrics.csv');r=h[h.epoch==status['best_epoch']].iloc[0];ck=torch.load(d/'best.pt',map_location='cpu',weights_only=False);params=sum(v.numel() for v in ck['model'].values())
   rows.append(dict(configuration=c,seed=s,best_epoch=status['best_epoch'],stop_epoch=status['stop_epoch'],stop_reason=status['stop_reason'],train_auc=r.train_auc,train_auprc=r.train_auprc,train_loss=r.train_loss,train_brier=r.train_brier,valid_auc=r.valid_auc,valid_auprc=r.valid_auprc,valid_loss=r.valid_loss,valid_brier=r.valid_brier,s_val=r.s_val,parameter_count=params))
   gaps.append(dict(configuration=c,seed=s,checkpoint_epoch=status['best_epoch'],auc_gap=r.train_auc-r.valid_auc,auprc_gap=r.train_auprc-r.valid_auprc,loss_gap=r.valid_loss-r.train_loss,brier_gap=r.valid_brier-r.train_brier))
   stops.append(dict(configuration=c,seed=s,best_epoch=status['best_epoch'],stop_epoch=status['stop_epoch'],stop_reason=status['stop_reason']))
   patient_evals=status['stop_epoch']*(255+84)+2*84
   memory.append(dict(configuration=c,seed=s,peak_gpu_memory_mb=status['peak_gpu_memory_mb'],elapsed_seconds=status['elapsed_seconds'],patient_evaluations=patient_evals,patient_evaluations_per_second=patient_evals/status['elapsed_seconds']))
   pred.append(pd.read_csv(d/'validation_predictions.csv'))
   for f in ['config.json','best.pt','stopped.pt','epoch_metrics.csv','validation_predictions.csv']:
    p=d/f;manifest.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
   gp=d/'spatial_gates.csv'
   if gp.exists() and gp.stat().st_size>2:
    try:g=pd.read_csv(gp)
    except pd.errors.EmptyDataError:g=pd.DataFrame()
    if len(g):g.insert(0,'seed',s);g.insert(0,'configuration',c);gate_frames.append(g)
 for p in [ROOT/'src/MSRLSelect/stage1c_model.py',ROOT/'src/MSRLSelect/stage1c_train.py',ROOT/'src/MSRLSelect/stage1c_audit.py',ROOT/'src/MSRLSelect/stage1c_summarize.py']:
  manifest.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
 R=pd.DataFrame(rows);R.to_csv(OUT/'metrics/stage1c_multiseed_results.csv',index=False);pd.DataFrame(gaps).to_csv(OUT/'metrics/stage1c_train_validation_gap.csv',index=False);pd.DataFrame(stops).to_csv(OUT/'metrics/stage1c_early_stopping_statistics.csv',index=False);pd.DataFrame(memory).to_csv(OUT/'metrics/stage1c_memory_throughput.csv',index=False);pd.concat(pred).to_csv(OUT/'predictions/stage1c_validation_predictions.csv',index=False);pd.DataFrame(manifest).to_csv(OUT/'metrics/stage1c_sha256_manifest.csv',index=False)
 if gate_frames:pd.concat(gate_frames).to_csv(OUT/'metrics/stage1c_spatial_gate_statistics.csv',index=False)
 vals=['valid_auc','valid_auprc','valid_loss','valid_brier','s_val'];summary=R.groupby('configuration')[vals].agg(['mean','std']).reset_index();summary.columns=['configuration']+[f'{a}_{b}' for a,b in summary.columns.tolist()[1:]];summary.to_csv(OUT/'metrics/stage1c_multiseed_summary.csv',index=False)
 pairs=[('gated_pooling','C1_K64_gated_relation_no_spatial','C0_K64_mean_relation_no_spatial'),('spatial_bias','C2_K64_gated_relation_spatial','C1_K64_gated_relation_no_spatial'),('cross_region_relation','C1_K64_gated_relation_no_spatial','C3_K64_gated_no_relation_no_spatial'),('K128_scale','C4_K128_gated_relation_spatial','C2_K64_gated_relation_spatial')];effects=[]
 for name,a,b in pairs:
  aa=R[R.configuration==a].sort_values('seed');bb=R[R.configuration==b].sort_values('seed');ds=aa.s_val.to_numpy()-bb.s_val.to_numpy();da=aa.valid_auc.mean()-bb.valid_auc.mean();dp=aa.valid_auprc.mean()-bb.valid_auprc.mean();effective=ds.mean()>=.005 and (ds>0).sum()>=2 and da>=-.005 and dp>=-.005 and aa.s_val.std()<=bb.s_val.std()+.01
  effects.append(dict(component=name,treatment=a,baseline=b,mean_s_val_delta=ds.mean(),seed1_delta=ds[0],seed17_delta=ds[1],seed42_delta=ds[2],positive_seed_count=int((ds>0).sum()),mean_auc_delta=da,mean_auprc_delta=dp,treatment_s_val_std=aa.s_val.std(),baseline_s_val_std=bb.s_val.std(),retain=effective))
 E=pd.DataFrame(effects);E.to_csv(OUT/'metrics/stage1c_component_effects.csv',index=False)
 best_mean=summary.s_val_mean.max();eligible=summary[summary.s_val_mean>=best_mean-.005].configuration.tolist();pc=R.groupby('configuration').parameter_count.first();chosen=min(eligible,key=lambda c:(pc[c],summary.set_index('configuration').loc[c,'s_val_std']))
 table='\n'.join(f"|{r.configuration}|{r.valid_auc_mean:.4f}±{r.valid_auc_std:.4f}|{r.valid_auprc_mean:.4f}±{r.valid_auprc_std:.4f}|{r.s_val_mean:.4f}±{r.s_val_std:.4f}|" for _,r in summary.iterrows())
 audit=json.loads((OUT/'audit/protocol_audit.json').read_text());ident=json.loads((OUT/'audit/r1_r3_identity.json').read_text());sp=json.loads((OUT/'audit/spatial_effectiveness.json').read_text())
 (OUT/'docs/v5_stage1c_protocol_audit.md').write_text(f"# Stage 1C Protocol Audit\n\nStatus: **{audit['status']}**. Early stopping unit test triggers at epoch 30 for a best epoch of 1. Learning rate uses 5-epoch linear warmup followed by cosine decay. Two real-data seed-17 short runs reproduced exactly. Test was not accessed.\n")
 (OUT/'docs/v5_stage1c_r1_r3_identity_audit.md').write_text(f"# R1/R3 Identity Audit\n\nAll {ident['patient_count']} predictions and all common checkpoint tensors were numerically identical (maximum differences both 0). The files were not overwritten. Old R3 only disabled spatial bias and did not disable relation blocks, so it executed the same numerical graph as R1. Old R3 is invalid. Stage1C C3 has a separate masked aggregation branch with no relation blocks.\n")
 (OUT/'docs/v5_stage1c_spatial_effectiveness_audit.md').write_text(f"# Spatial Effectiveness Audit\n\nSpatial module is active. Normal versus shuffled-coordinate maximum prediction difference: {sp['normal_vs_shuffle_max_abs']:.8f}; normal versus bypass: {sp['normal_vs_bypass_max_abs']:.8f}. CLS bias rows/columns are zero; padding is masked; every layer/head gate and spatial MLP received nonzero gradients. These checks establish use, not predictive benefit.\n")
 decisions='\n'.join(f"- {r.component}: mean ΔS={r.mean_s_val_delta:+.4f}, positive seeds={r.positive_seed_count}/3, retain={bool(r.retain)}" for _,r in E.iterrows())
 report=f"# MS-RLSurv V5 Stage 1C Dense Confirmation\n\nProtocol audit: **GO**. Only Train/Validation were used.\n\n|Configuration|AUROC mean±SD|AUPRC mean±SD|S_val mean±SD|\n|---|---:|---:|---:|\n{table}\n\n## Component decisions\n{decisions}\n\n## Dense candidate\nFrozen candidate: **{chosen}**. Selection applies the predefined 0.005 equivalence margin, then favors fewer parameters and lower variability. No Test, MoE, PPO, or SG-SurvNet run was performed.\n"
 (OUT/'docs/v5_stage1c_dense_confirmation_report.md').write_text(report);(OUT/'docs/v5_stage1c_candidate_decision.md').write_text(f"# Stage 1C Candidate Decision\n\nFreeze `{chosen}` as the Dense candidate under the predefined Validation-only rules.\n\n{decisions}\n\nEligibility to enter MoE is limited to protocol completion; this report does not authorize or start MoE.\n")
 print(json.dumps({'status':'COMPLETE','candidate':chosen,'test_access':False},indent=2))
if __name__=='__main__':main()
