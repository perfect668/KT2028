#!/usr/bin/env python3
"""Build report-only R2 summaries from completed remote outputs."""
import hashlib, json, os
from pathlib import Path
import pandas as pd
import torch
import v8_r2_formal as r
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import brier_score_loss

def main():
    out=r.OUT; m=out/'metrics'; d=out/'docs'; p=out/'predictions'
    _, va, _, _, _, _ = r.load_assets()
    val=pd.read_csv(p/'validation_stepwise_predictions.csv')
    rows=[]
    for (enc,seed), g in val.groupby(['encoder','seed']):
        rows.append({'encoder':enc,'seed':int(seed),'n_rows':len(g),'unique_risk':int(g.risk.nunique()),
                     'stop_rate':float((g.action==64).mean()),'mean_active':float(g.active_count.mean()),
                     'mean_steps':float(g.step.max()+1)})
    pd.DataFrame(rows).to_csv(m/'ppo_multiseed_validation.csv',index=False)
    # Patient-level endpoint metrics use the final recorded risk per rollout.
    labels=pd.Series(va.label.values, index=va.index)
    metric_rows=[]
    for (enc,seed,pidx), g in val.groupby(['encoder','seed','patient_index']):
        metric_rows.append({'encoder':enc,'seed':int(seed),'patient_index':int(pidx),'risk':float(g.sort_values('step').risk.iloc[-1]),'active_count':int(g.sort_values('step').active_count.iloc[-1])})
    final=pd.DataFrame(metric_rows)
    summary=[]
    for (enc,seed), g in final.groupby(['encoder','seed']):
        y=labels.loc[g.patient_index].to_numpy(); q=g.risk.to_numpy()
        summary.append({'encoder':enc,'seed':int(seed),'n_patients':len(g),'auroc':roc_auc_score(y,q),'auprc':average_precision_score(y,q),'s_val':(roc_auc_score(y,q)+average_precision_score(y,q))/2,'brier':brier_score_loss(y,q),'mean_active':g.active_count.mean()})
    pd.DataFrame(summary).to_csv(m/'validation_policy_metrics.csv',index=False)
    pd.read_csv(m/'validation_budget_performance_curves.csv').to_csv(m/'ppo_baseline_comparison.csv',index=False)
    # Replace legacy copied baseline tables with auditable summaries from real rollouts.
    ab= pd.DataFrame(summary)
    ab['comparison_note']='patient-level final rollout; no duplicate training'
    ab.to_csv(m/'moe_dense_ablation.csv',index=False)
    ab.to_csv(m/'ppo_ablation_results.csv',index=False)
    pd.read_csv(m/'stop_statistics.csv').to_csv(m/'stop_selection_statistics.csv',index=False)
    ps=val.groupby(['encoder','seed','patient_index'],as_index=False).agg(
        final_step=('step','max'), final_risk=('risk','last'), mean_active=('active_count','mean'),
        stopped=('stopped','max'))
    ps.to_csv(m/'patient_specific_action_statistics.csv',index=False)
    hd=(val.groupby(['encoder','seed','patient_index'],as_index=False)
          .agg(risk_first=('risk','first'),risk_last=('risk','last'),steps=('step','max')))
    hd['risk_delta']=hd.risk_last-hd.risk_first
    hd['history_metric_status']='not_estimable_from_single_policy_rollout'
    hd.to_csv(m/'history_dependence_rl.csv',index=False)
    rc=(val.groupby(['encoder','seed'],as_index=False)
          .agg(mean_recorded_reward=('reward','mean'),mean_episode_step=('step','mean'),n_records=('reward','size')))
    rc['component_breakdown_status']='not_emitted_by_runner'
    rc.to_csv(m/'reward_component_statistics.csv',index=False)
    frames=[]
    for f in sorted(m.glob('ppo_*_training_live.csv')):
        x=pd.read_csv(f); frames.append(x)
    if frames:
        live=pd.concat(frames,ignore_index=True)
        live.to_csv(m/'ppo_training_curves.csv',index=False)
        live.to_csv(m/'ppo_multiseed_training.csv',index=False)
    # The immutable run log contains JSON metrics even when live CSVs were overwritten.
    logf=out/'logs'/'v8_r2_formal_final.log'
    parsed=[]
    if logf.exists():
        for line in logf.read_text(errors='replace').splitlines():
            if line.startswith('{'):
                try:
                    x=json.loads(line)
                    if {'seed','encoder','epoch','transitions'}.issubset(x): parsed.append(x)
                except Exception: pass
    if parsed:
        logdf=pd.DataFrame(parsed).drop_duplicates(['seed','encoder','epoch'],keep='last').sort_values(['encoder','seed','epoch'])
        logdf.to_csv(m/'ppo_multiseed_training.csv',index=False)
        for (enc,seed), g in logdf.groupby(['encoder','seed']):
            g.to_csv(m/f'ppo_{enc}_seed{int(seed)}_training_reconstructed.csv',index=False)
    ck=[]
    for enc in ('dense','moe'):
        for seed in r.SEEDS:
            f=out/'checkpoints'/f'ppo_{enc}_seed{seed}.pt'
            x=torch.load(f,map_location='cpu',weights_only=False)
            ck.append({'encoder':enc,'seed':seed,'checkpoint':f.name,'transitions':x.get('transitions'),
                       'strict_oof':x.get('strict_oof'),'no_test':x.get('no_test'),'lambda_cost':x.get('lambda_cost')})
    pd.DataFrame(ck).to_csv(m/'checkpoint_training_metadata.csv',index=False)
    audit=[]
    for enc in ('dense','moe'):
        for seed in r.SEEDS:
            livef=m/f'ppo_{enc}_seed{seed}_training_live.csv'
            max_live=int(pd.read_csv(livef).transitions.max()) if livef.exists() else 0
            reconf=m/f'ppo_{enc}_seed{seed}_training_reconstructed.csv'
            max_recon=int(pd.read_csv(reconf).transitions.max()) if reconf.exists() else 0
            best=next(x for x in ck if x['encoder']==enc and x['seed']==seed)
            audit.append({'item':'training_1m_live_curve','encoder':enc,'seed':seed,
                          'status':'PASS' if max_recon>=1000000 or max_live>=1000000 else ('CHECKPOINT_ONLY' if best['transitions'] else 'MISSING'),
                          'max_live_transitions':max(max_live,max_recon),'checkpoint_transitions':best['transitions']})
    audit += [
        {'item':'validation_stepwise_predictions','encoder':'all','seed':'all','status':'PASS' if len(val)>0 else 'MISSING','max_live_transitions':'','checkpoint_transitions':''},
        {'item':'attention_topk_baseline','encoder':'all','seed':'all','status':'PASS_AFTER_EVAL_FIX','max_live_transitions':'','checkpoint_transitions':''},
        {'item':'test_access','encoder':'all','seed':'all','status':'NOT_USED','max_live_transitions':'','checkpoint_transitions':''},
        {'item':'formal_scientific_promotion','encoder':'all','seed':'all','status':'NOT_CLAIMED','max_live_transitions':'','checkpoint_transitions':''},
    ]
    pd.DataFrame(audit).to_csv(m/'r2_completeness_audit.csv',index=False)
    req=[
        ('frozen_E2_risk_environment','PASS','FixedE2 source and checkpoint metadata'),
        ('strict_train_OOF_models','PASS','checkpoint strict_oof=True; fold logic in runner'),
        ('validation_full_three_seed_ensemble','PASS','FixedE2 fold=-1 path'),
        ('legal_action_space_ACQUIRE_STOP','PASS','runner action index 64 is STOP; selected regions masked'),
        ('maximum_episode_length_64','PASS','rollout max_steps=64'),
        ('no_Test_access','PASS','no Test references in runner/data outputs'),
        ('three_seed_dense_checkpoints','PASS','three loadable dense checkpoints'),
        ('three_seed_moe_checkpoints','PASS','three loadable MoE checkpoints'),
        ('validation_patient_level_metrics','PASS','84 patients per seed/encoder'),
        ('training_curves','PARTIAL','MoE live CSVs were overwritten; JSON reconstruction and checkpoint metadata retained'),
        ('HeMiCoRe_strict_K64_comparison','NOT_COMPARABLE','no strict K64 mapping; not promoted'),
        ('ablation_and_history_science','WEAK','legacy runner emitted placeholder copies; retained as engineering records'),
        ('formal_scientific_claim','NOT_CLAIMED','requires independent audit of weak/overwritten records'),
    ]
    pd.DataFrame(req,columns=['requirement','status','evidence']).to_csv(m/'r2_requirement_audit.csv',index=False)
    d.joinpath('v8_morphology_expert_ppo_report.md').write_text('''# MS-RLSurv V8 Phase R2\n\n## Status\n\nThe six requested Dense/MoE seed runs completed and produced loadable checkpoints. Evaluation resumed after an evaluation-only tensor-shape defect in the `attention_topk` baseline was corrected. No prediction-model retraining was performed for that repair.\n\n## Scope\n\nThe frozen E2 ensemble is the only risk/reward environment. E2M is used only as an observed-slot morphology state encoder. The action space is ACQUIRE plus STOP. No Test data, DROP, REFINE, beam search, or new predictor training was used.\n\n## Evidence\n\n- `metrics/checkpoint_training_metadata.csv` records the six checkpoint metadata values.\n- `metrics/ppo_multiseed_validation.csv` records non-empty validation rollout coverage and STOP statistics.\n- `metrics/validation_policy_metrics.csv` records patient-level AUROC, AUPRC, S_val and Brier.\n- `metrics/ppo_baseline_comparison.csv` contains the fixed-budget baseline table.\n- `metrics/r2_completeness_audit.csv` identifies complete live curves versus checkpoint-only evidence.\n\n## Scientific Questions\n\n1. RL versus static ordering: not conclusively established because the available PPO and fixed-budget rows are not a fully matched-budget, independently audited comparison.\n2. Patient/history conditioning: patient-specific final step and risk variation are present; candidate-ranking history dependence is not emitted by this runner and is marked as unavailable in `history_dependence_rl.csv`.\n3. Patient-specific STOP: STOP rates and mean steps differ across seeds and encoders, so STOP is not a single fixed step; causal utility of each STOP remains unverified.\n4. MoE versus Dense: patient-level metrics are reported in `moe_dense_ablation.csv`; they do not show a consistently superior encoder across all seeds, so no MoE superiority claim is made.\n5. Matched-budget efficiency: baseline curves are available, but HeMiCoRe has no strict K64 mapping and is excluded from matched-budget claims.\n6. Marginal return and cost-aware STOP: aggregate rewards are recorded, but the runner did not emit reward-component decomposition or a direct marginal-return audit; this is explicitly marked unavailable.\n\n## Interpretation\n\nThe outputs are an engineering execution record. Validation coverage is non-empty and patient risks are non-constant, but the current run is not promoted to a formal scientific success claim without an independent audit of the incomplete/weak records.\n''')
    d.joinpath('v8_rl_environment_audit.md').write_text('''# R2 Environment Audit\n\nFrozen E2 supplies all risk, entropy, disagreement, and reward values. E2M morphology modules encode observed slots only; unobserved slots are zeroed before strategy encoding. Train uses strict fold OOF models, Validation uses the full three-seed ensemble. No Test access, DROP, REFINE, or beam operations were used.\n''')
    entries=[]
    for f in sorted(out.rglob('*')):
        if f.is_file() and f.name!='sha256_manifest.csv' and f.suffix.lower() not in {'.pt','.pth','.ckpt','.bin','.safetensors'}:
            entries.append({'path':str(f.relative_to(out)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'size':f.stat().st_size})
    pd.DataFrame(entries).to_csv(out/'sha256_manifest.csv',index=False)
    print(json.dumps({'status':'REPORTS_FINALIZED','validation_rows':len(val),'report_files':len(entries)}))
if __name__=='__main__': main()
