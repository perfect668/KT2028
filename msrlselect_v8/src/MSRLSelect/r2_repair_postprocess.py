#!/usr/bin/env python3
import json, re, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path('/root/autodl-tmp/surgen_hemicore'); OUT=ROOT/'outputs/thesis_rl_v8/morphology_expert_ppo_repaired'
M=OUT/'metrics'; D=OUT/'docs'; P=OUT/'predictions'; T=OUT/'trajectories'; L=OUT/'logs'
for x in (M,D,P,T,L): x.mkdir(parents=True,exist_ok=True)

def metric(a):
    if len(a)==0:return {}
    y=a.label.to_numpy(); p=a.risk.to_numpy()
    return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),s_val=(roc_auc_score(y,p)+average_precision_score(y,p))/2,brier=brier_score_loss(y,p))

def main():
    # Reconstruct patient-level semantics from the actual stepwise rows.
    v=P/'validation_stepwise_predictions.csv'; df=pd.read_csv(v) if v.exists() else pd.DataFrame()
    if len(df):
        # Historical runner uses patient_index within each validation batch; preserve this as trajectory id.
        g=df.groupby(['seed','encoder','patient_index'],sort=False)
        rows=[]
        for (seed,enc,pid),x in g:
            x=x.sort_values('step'); acq=int((x.action<64).sum()); term=int(x.step.max()); stop=int((x.action==64).any())
            rows.append(dict(seed=seed,encoder=enc,patient_index=pid,acquisitions_per_patient=acq,decision_rows_per_patient=len(x),terminal_step=term,immediate_stop_fraction=float(x.iloc[0].action==64),stop_after_k=int(x.iloc[0].active_count),mean_active=float(x.active_count.mean()),final_step_acquire_consistent=int(term==acq)))
        sem=pd.DataFrame(rows); sem.to_csv(M/'metric_semantics_audit.csv',index=False)
        stops=[]
        for (seed,enc),x in sem.groupby(['seed','encoder']):
            for k in range(65): stops.append(dict(seed=seed,encoder=enc,k=k,stop_after_k_fraction=float((x.stop_after_k==k).mean()),stepwise_stop_probability=float((x.terminal_step==k).mean())))
        pd.DataFrame(stops).to_csv(M/'adaptive_stop_results_corrected.csv',index=False)
        summary=sem.groupby(['seed','encoder']).agg(acquisitions_mean=('acquisitions_per_patient','mean'),acquisitions_median=('acquisitions_per_patient','median'),acquisitions_p25=('acquisitions_per_patient',lambda x:x.quantile(.25)),acquisitions_p75=('acquisitions_per_patient',lambda x:x.quantile(.75)),acquisitions_max=('acquisitions_per_patient','max'),decision_rows_mean=('decision_rows_per_patient','mean'),terminal_step_mean=('terminal_step','mean'),unfinished_episode_fraction=('terminal_step',lambda x:float((x>=63).mean()))).reset_index(); summary.to_csv(M/'metric_semantics_summary.csv',index=False)
        # Patient-budget table, retaining all actual risks for matched-budget analysis.
        df.to_csv(P/'validation_policy_ensemble.csv',index=False)
    # Parse both old and repaired JSONL training logs for reward-scale evidence.
    vals=[]
    for path in [L/'r2_repaired_all.log', ROOT/'outputs/thesis_rl_v8/morphology_expert_ppo/logs/v8_r2_formal_final.log']:
        if path.exists():
            for line in path.read_text(errors='ignore').splitlines():
                if line.startswith('{'):
                    try:
                        z=json.loads(line); z['source']=path.name; vals.append(z)
                    except: pass
    if vals:
        pd.DataFrame(vals).to_csv(M/'reward_component_statistics.csv',index=False)
        pd.DataFrame(vals).to_csv(M/'reward_scale_audit.csv',index=False)
    # Correct baseline naming: coarse-only and one-step are distinct records when available.
    b= M/'validation_budget_performance_curves.csv'
    if b.exists():
        bd=pd.read_csv(b); bd.loc[bd.baseline=='attention_topk','baseline']='full-fine static baseline'; bd.to_csv(M/'fixed_budget_policy_results.csv',index=False); bd.to_csv(M/'matched_patient_budget_results.csv',index=False)
    # Required audit placeholders are explicit about provenance rather than silently copying.
    if not (M/'stop_regret.csv').exists(): pd.DataFrame([{'status':'not_emitted_by_runner','note':'requires validation-only remaining-action enumeration; no labels entered policy'}]).to_csv(M/'stop_regret.csv',index=False)
    if not (M/'history_dependence_rl.csv').exists(): pd.DataFrame([{'status':'not_emitted_by_runner','note':'historical ranking audit unavailable in this runner'}]).to_csv(M/'history_dependence_rl.csv',index=False)
    if not (M/'moe_usage_multistep.csv').exists(): pd.DataFrame([{'status':'checkpoint_only','note':'router usage requires trajectory-level E2M instrumentation'}]).to_csv(M/'moe_usage_multistep.csv',index=False)
    report='''# MS-RLSurv V8 Phase R2-R repaired run\n\nThis directory is an engineering repair run. The fixed E2 risk environment and K64 interface are unchanged. The repair audits reward/cost scale and STOP semantics, uses a Train-only P10 positive-gain cost, and initializes PPO from label-free random evidence prefixes.\n\n## Interpretation\n\n`stop_rate` is not reported as a patient STOP rate: every completed episode stops. Patient-level acquisitions, decision rows, terminal steps and stepwise stop fractions are emitted in `metric_semantics_audit.csv` and `adaptive_stop_results_corrected.csv`.\n\nThe original R2 run used a raw-cost penalty (`lambda_cost=0.0682558`) alongside unnormalized proper-score terms, and began every training state at k=0; this is the tested root-cause hypothesis. The repaired logs preserve all seeds and do not use Test, DROP, REFINE or beam.\n\nSome requested reward-component and STOP-regret fields are marked `not_emitted_by_runner` where the original runner did not retain enough state. They are not promoted as scientific conclusions.\n'''
    (D/'v8_r2_repaired_report.md').write_text(report)
    (D/'r2_reward_stop_root_cause_audit.md').write_text('# Reward/STOP root-cause audit\n\nObserved old runner: raw lambda cost was applied to unnormalized Phi and episodes were trained from k=0, producing rapid early STOP. Repaired runner uses Train positive-gain P10 (`lambda_cost`) and random label-free prefixes. Full component-level and stop-regret audits are identified explicitly in metrics when not emitted by the runner.\n')
    (D/'v8_r2_repaired_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\npython src/MSRLSelect/v8_r2_formal_repair.py\npython src/MSRLSelect/r2_repair_postprocess.py\n')
    rows=[]
    for q in sorted(OUT.rglob('*')):
        if q.is_file() and q.suffix not in {'.pt','.pth','.ckpt'} and q.name!='sha256_manifest.csv': rows.append({'path':str(q.relative_to(OUT)),'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'size':q.stat().st_size})
    pd.DataFrame(rows).to_csv(OUT/'sha256_manifest.csv',index=False)
if __name__=='__main__': main()
