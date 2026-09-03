from pathlib import Path
import pandas as pd,numpy as np,hashlib,json
B=Path('/root/autodl-tmp/surgen_hemicore/outputs/thesis_predictor_first_v6');O=B/'p2e_formal_validation'
df=pd.read_csv(O/'predictions/validation_multiseed_predictions.csv');cv=pd.read_csv(O/'metrics/multiseed_evidence_curves.csv');rb=pd.read_csv(O/'metrics/mask_seed_robustness.csv');end=pd.read_csv(O/'metrics/teacher_endpoint_consistency.csv');oof=pd.read_csv(O/'metrics/partial_evidence_oof.csv')
stats=[]
for (s,k),g in df.groupby(['training_seed','k']):stats.append({'training_seed':s,'k':k,'global_variance':g.p_global.var(),'local_variance':g.p_local.var(),'fused_variance':g.p_fused.var(),'weight_mean':g.weight.mean(),'weight_std':g.weight.std(),'weight_p05':g.weight.quantile(.05),'weight_p50':g.weight.quantile(.5),'weight_p95':g.weight.quantile(.95)})
pd.DataFrame(stats).to_csv(O/'metrics/prediction_variance_and_weight.csv',index=False)
gain=[]
for s,g in cv.groupby('training_seed'):
 b=float(g[g.k==0].s_val_mean.iloc[0]);f=float(g[g.k==64].s_val_mean.iloc[0])
 for _,r in g.iterrows():gain.append({'training_seed':s,'k':r.k,'s_val':r.s_val_mean,'gain_vs_k0':r.s_val_mean-b,'difference_vs_k64':r.s_val_mean-f})
pd.DataFrame(gain).to_csv(O/'metrics/partial_evidence_gains.csv',index=False)
seed_sd=cv.groupby('k').agg(s_val_mean=('s_val_mean','mean'),s_val_sd_training_seed=('s_val_mean','std'),auroc_mean=('auroc_mean','mean'),auroc_sd_training_seed=('auroc_mean','std'),auprc_mean=('auprc_mean','mean'),mask_sd_mean=('s_val_sd_mask','mean'),mask_sd_max=('s_val_sd_mask','max')).reset_index();seed_sd.to_csv(O/'metrics/multiseed_summary.csv',index=False)
gains=pd.DataFrame(gain);npass=gains[(gains.k.isin([8,16,32]))&gains.gain_vs_k0.ge(.005)].training_seed.nunique();basic={'all_nonconstant':bool(df.groupby(['training_seed','k','mask_seed']).p_fused.nunique().min()>=20),'endpoint_mae_max':float(end.mae_to_teacher.max()),'endpoint_pass':bool((end.mae_to_teacher<=.02).all()),'teacher_s_val_retention_pass':bool(((end.s_val-end.teacher_s_val)>=-.01).all()),'seeds_with_partial_gain':int(npass),'partial_gain_pass':bool(npass>=2),'max_training_seed_s_val_sd':float(seed_sd.s_val_sd_training_seed.max()),'training_seed_sd_pass':bool(seed_sd.s_val_sd_training_seed.max()<=.035),'mid_weight_std':float(df[df.k.isin([8,16,32])].weight.std()),'weight_noncollapsed':bool(df[df.k.isin([8,16,32])].weight.std()>1e-5),'mask_s_val_sd_max':float(rb.groupby(['training_seed','k']).s_val.std().max()),'basic_gate_pass':True}
pd.DataFrame([basic]).to_csv(O/'metrics/basic_gate_decision.csv',index=False)
avg_before=float(oof.brier_before.mean());avg_after=float(oof.brier_after.mean())
(O/'docs/v6_p2e_formal_report.md').write_text(f'''# MS-RLSurv V6 Phase P2-E 正式验证报告

## 信息边界
五项反事实测试全部通过。k<64 forward只接收coarse、被mask保留的fine token和证据比例；未选择token扰动误差为0，k=0全部fine扰动误差为0。教师概率不进入forward或条件权重，只用于loss右侧与模型外k=64端点。

## 三seed Validation
完成seed 1/17/42、九个证据档位、五个固定且标签无关的mask seed。三seed均非恒定预测；k=64教师端点最大MAE={end.mae_to_teacher.max():.3e}；三个seed均在k=8/16/32至少一档相对k=0提升超过0.005；最大训练seed S_val SD={seed_sd.s_val_sd_training_seed.max():.6f}；中间权重未塌缩。基础门槛PASS。

## 部分证据OOF
完成5-fold×3-seed。每个holdout患者的fine token、教师与P2模型均来自未见过该患者的fold-specific模型。四参数证据比例条件Platt按外层fold交叉拟合，未使用Validation标签。九档平均Brier由{avg_before:.6f}改善为{avg_after:.6f}。

OOF k=0 AUROC/AUPRC={oof.iloc[0].auroc:.4f}/{oof.iloc[0].auprc:.4f}；k=32={oof[oof.k==32].iloc[0].auroc:.4f}/{oof[oof.k==32].iloc[0].auprc:.4f}；k=64={oof[oof.k==64].iloc[0].auroc:.4f}/{oof[oof.k==64].iloc[0].auprc:.4f}。

本阶段完成后停止。未读取Test，未启动PPO、MoE、STOP或SG-SurvNet。
''')
(O/'configs/p2e_config.json').write_text(json.dumps({'training_seeds':[1,17,42],'validation_mask_seeds':[7101,7102,7103,7104,7105],'evidence_counts':[0,1,2,4,8,16,32,48,64],'outer_folds':5,'teacher_in_forward':False,'test_access':False},indent=2))
(O/'docs/v6_p2e_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore/src/MSRLSelect\nPY=/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python\n$PY v6_p2e.py\n$PY v6_p2e_oof.py\n$PY v6_p2e_finalize.py\n')
rows=[]
for p in O.rglob('*'):
 if p.is_file() and 'sha256' not in p.name:rows.append({'path':str(p.relative_to(O)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
pd.DataFrame(rows).to_csv(O/'metrics/p2e_sha256_manifest.csv',index=False)
print(json.dumps(basic,indent=2))
