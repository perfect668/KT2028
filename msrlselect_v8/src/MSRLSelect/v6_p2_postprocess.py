from pathlib import Path
import pandas as pd, hashlib
O=Path('/root/autodl-tmp/surgen_hemicore/outputs/thesis_predictor_first_v6/p2_variable_evidence_predictor')
m=pd.read_csv(O/'metrics/variable_evidence_multiseed.csv')
c=m.groupby('k').agg(s_val_mean=('s_val','mean'),s_val_sd=('s_val','std'),auroc_mean=('auroc','mean'),auroc_sd=('auroc','std'),auprc_mean=('auprc','mean'),brier_mean=('brier','mean'),alpha_mean=('alpha_mean','mean')).reset_index()
c.to_csv(O/'metrics/evidence_count_curves.csv',index=False)
m[['k','seed','alpha_mean']].rename(columns={'alpha_mean':'alpha'}).to_csv(O/'metrics/residual_gate_statistics.csv',index=False)
p4={'full_evidence_mean_abs_diff':None,'full_consistency_pass':False,'global_baseline_s_val':float(c[c.k==0].s_val_mean.iloc[0]),'fused64_s_val':float(c[c.k==64].s_val_mean.iloc[0]),'teacher64_s_val':0.6852594503,'global_safety_pass':True,'teacher_retention_pass':False,'local_effective_pass':False,'local_only_auprc_pass':False,'residual_gate_noncollapsed':False,'auroc_sd_pass':True,'p4_pass':False,'status':'FAIL_BASELINE_CONSTANT_PREDICTIONS'}
pd.DataFrame([p4]).to_csv(O/'metrics/p4_gate_decision.csv',index=False)
pd.DataFrame([{'status':'NOT_RUN_BASELINE_GATE_FAILED','reason':'k64 fused failed teacher/global checks and no local evidence gain'}]).to_csv(O/'metrics/partial_oof_calibration.csv',index=False)
pd.DataFrame([{'status':'NOT_GENERATED_BASELINE_GATE_FAILED'}]).to_csv(O/'predictions/train_oof_variable_evidence.csv',index=False)
(O/'docs/v6_p2_variable_evidence_report.md').write_text('# V6 Phase P2 变长部分证据预测器正式训练报告\n\n使用已通过 smoke 的紧凑 K64 缓存完成 seed=1、17、42 正式训练。三个 seed 均在第30 epoch停止。Validation 各证据数量均退化为常数预测：AUROC=0.5、AUPRC=0.357143，S_val=0.428571。k=0 到 k=64 没有细粒度增量，residual gate 饱和。\n\nP4 判定：FAIL。k=64 fused 不满足冻结教师一致性和教师性能保持；k=8/16/32 没有相对 global 的增益；局部证据有效性未满足。因此不生成部分证据 OOF，不进入校准，不启动 P3。未读取 Test，未启动 PPO、MoE 或 STOP。\n')
rows=[]
for p in O.rglob('*'):
 if p.is_file() and 'sha256' not in p.name: rows.append({'path':str(p.relative_to(O)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(O/'metrics/p2_sha256_manifest.csv',index=False)
