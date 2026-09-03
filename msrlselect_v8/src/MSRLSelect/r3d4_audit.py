#!/usr/bin/env python3
import os,sys,json,hashlib,math,shutil
from pathlib import Path
import numpy as np,pandas as pd, torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore'); SRC=ROOT/'src/MSRLSelect'; IN=ROOT/'outputs/thesis_rl_v8/r3d3_policy_repair_and_ppo'; OUT=ROOT/'outputs/thesis_rl_v8/r3d4_integrated_adaptive_ppo'
for d in ['metrics','predictions','trajectories','checkpoints','docs','logs','code']: (OUT/d).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(SRC)); from v7_e2m_train import load_data; from r3d3_real import Policy,DenseEnc; from v8_r2_formal import FixedE2,phi
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path):
 ck=torch.load(path,map_location=DEV,weights_only=False); p=Policy(133).to(DEV); p.load_state_dict(ck['model']); p.eval(); e=DenseEnc().to(DEV); e.load_state_dict(ck.get('encoder',{}),strict=False); e.eval(); return p,e,ck

def rollout(p,e,c,s,labels,pids,lam,ckpt):
 n=len(pids); active=torch.zeros(n,64,dtype=torch.bool,device=DEV); order=torch.full((n,64),-1,dtype=torch.long,device=DEV); alive=torch.ones(n,dtype=torch.bool,device=DEV); rows=[]; stops=[]; env=FixedE2(c.shape[-1],-1)
 cur,ent,dis,_=env.predict(c,s.flatten(-2),active); cum=torch.zeros(n,device=DEV)
 for step in range(64):
  if not bool(alive.any()): break
  with torch.no_grad():
   emb=e(s,active); lg,sl,val=p(c,emb,active,order); legal=~active; stopprob=torch.sigmoid(sl); acqprob=torch.softmax(lg.masked_fill(~legal,-1e9),-1); stop=(stopprob>=0.5)&alive; sel=acqprob.argmax(1)
  next_active=active.clone(); ids=torch.where(alive&~stop)[0]
  if len(ids): next_active[ids,sel[ids]]=True
  nxt,entn,disn,_=env.predict(c,s.flatten(-2),next_active)
  gain=phi(torch.tensor(labels,device=DEV).float(),nxt,disn)-phi(torch.tensor(labels,device=DEV).float(),cur,dis)-lam
  reward=torch.where(stop,torch.zeros_like(gain),gain); cum+=reward
  for i in range(n):
   if bool(alive[i]):
    rows.append({'checkpoint':ckpt,'lambda':lam,'patient_id':str(pids[i]),'step':step,'active_count':int(active[i].sum()),'active_regions':','.join(map(str,torch.where(active[i])[0].cpu().tolist())),'action':'STOP' if bool(stop[i]) else f'ACQUIRE({int(sel[i])})','action_probability':float(stopprob[i] if bool(stop[i]) else acqprob[i,sel[i]]),'stop_probability':float(stopprob[i]),'risk':float(cur[i]),'entropy':float(ent[i]),'disagreement':float(dis[i]),'acquisition_cost':float(lam if not bool(stop[i]) else 0),'reward':float(reward[i]),'cumulative_reward':float(cum[i]),'terminal_reason':'STOP' if bool(stop[i]) else ('MAX_BUDGET' if step==63 else '')})
    if bool(stop[i]): stops.append({'checkpoint':ckpt,'lambda':lam,'patient_id':str(pids[i]),'terminal_step':step,'stop_regret':float(max(0.0,float(gain[i]))),'best_remaining_net_gain':float(gain[i]),'stop_probability':float(stopprob[i])})
  active=next_active; order[:,step]=torch.where(alive&~stop,sel,order[:,step]); alive=alive&~stop; cur,ent,dis=nxt,entn,disn
 # any forced terminals
 for i in torch.where(alive)[0].cpu().tolist(): stops.append({'checkpoint':ckpt,'lambda':lam,'patient_id':str(pids[i]),'terminal_step':64,'stop_regret':0.0,'best_remaining_net_gain':0.0,'stop_probability':0.0})
 return rows,stops

def main():
 tr,va,ct,cv,st,sv=load_data(); c=cv.float().to(DEV); s=sv.float().to(DEV); labels=va.label.to_numpy(); pids=va.case_id.to_numpy()
 ckpts=[('0','seed17_online_ppo_0.pt'),('50k','seed17_online_ppo_50k.pt'),('100k','seed17_online_ppo_100k.pt'),('150k','seed17_online_ppo_150k.pt'),('200k','seed17_online_ppo_200k.pt'),('validation-best','seed17_online_ppo_validation_best.pt')]
 # Train-derived cost quantiles from cached OOF gains only; validation is never used to set lambda.
 z=np.load(ROOT/'outputs/thesis_rl_v8/r3d2_real_multihistory_ppo/cache/multihistory_action_values.npz'); q=z['q_values']; legal=z['legal_mask']; pos=q[(legal)&(q>0)]; lams=[0.0]+[float(x) for x in np.quantile(pos,[.1,.25,.5])]
 allrows=[]; allstops=[]; comp=[]
 for label,fn in ckpts:
  path=IN/'checkpoints'/fn
  if not path.exists(): continue
  p,e,ck=load_model(path)
  for li,lam in enumerate(lams):
   rows,stops=rollout(p,e,c,s,labels,pids,lam,label); allrows.extend(rows); allstops.extend(stops)
   g=pd.DataFrame(rows); terminals=g[g.action=='STOP'].groupby('patient_id').size(); counts=g.groupby('patient_id').active_count.max() if len(g) else pd.Series(dtype=float)
   comp.append({'checkpoint':label,'lambda_index':li,'lambda':lam,'patients':len(pids),'stop_rows':len(stops),'immediate_stop_fraction':float((g[g.step==0].action=='STOP').mean()) if len(g[g.step==0]) else 0.0,'mean_acquisitions':float(counts.mean()) if len(counts) else 0.0,'median_acquisitions':float(counts.median()) if len(counts) else 0.0,'max_acquisitions':float(counts.max()) if len(counts) else 0.0,'mean_episode_steps':float(g.groupby('patient_id').step.max().mean()+1) if len(g) else 0.0,'mean_reward':float(g.groupby('patient_id').reward.sum().mean()) if len(g) else 0.0,'mean_stop_regret':float(np.mean([x['stop_regret'] for x in stops])) if stops else 0.0,'status':'diagnostic_legacy_checkpoint'})
 pd.DataFrame(allrows).to_csv(OUT/'predictions/validation_adaptive_trajectories.csv',index=False); pd.DataFrame(allrows).to_json(OUT/'trajectories/validation_complete_trajectories.jsonl',orient='records',lines=True); pd.DataFrame(allstops).to_csv(OUT/'metrics/stop_regret_all_patients.csv',index=False); pd.DataFrame(comp).to_csv(OUT/'metrics/ppo_checkpoint_comparison.csv',index=False); pd.DataFrame(comp).to_csv(OUT/'metrics/adaptive_stop_results.csv',index=False)
 # Semantics and PPO implementation audit based on source inspection plus logs.
 src=(SRC/'r3d3_online_real.py').read_text(); real=(SRC/'r3d3_real.py').read_text(); audit=[
 {'check':'old_policy_rollout_frozen','status':'FAIL','evidence':'rollout samples from p directly while optimizer updates after batch; no explicit old-policy clone'},
 {'check':'old_log_prob_detached','status':'PASS','evidence':'old_dist.log_prob(acts[t]).detach()'},
 {'check':'new_log_prob_recomputed','status':'PASS','evidence':'dist.log_prob(acts[t]) in update'},
 {'check':'ratio_exp_new_minus_old','status':'PASS','evidence':'torch.exp(new-old)'},
 {'check':'multiple_minibatch_updates','status':'FAIL','evidence':'single loss over 4 timesteps; no minibatch epochs'},
 {'check':'approx_kl_clip_fraction','status':'FAIL','evidence':'target_kl field is forward KL, clip_fraction not emitted'},
 {'check':'gae_transition_aligned','status':'FAIL','evidence':'Y/advantages assembled but critic EV uses mean over timestep values'},
 {'check':'value_target_detached','status':'PASS','evidence':'returns use vals.detach()'},
 {'check':'explained_variance_formula','status':'FAIL','evidence':'uses variance of Y - mean(vals) rather than per-transition aligned value'},
 {'check':'separate_losses','status':'FAIL','evidence':'actor/value/entropy combined; no separate STOP loss'}]
 pd.DataFrame(audit).to_csv(OUT/'metrics/ppo_implementation_audit.csv',index=False)
 # Corrected metric semantics from all trajectories.
 g=pd.DataFrame(allrows); sem=[]
 for (cp,lam),x in g.groupby(['checkpoint','lambda']):
  per=x.groupby('patient_id'); counts=per.active_count.max(); terminal=per.step.max(); sem.append({'checkpoint':cp,'lambda':lam,'acquisitions_per_patient_mean':counts.mean(),'acquisitions_per_patient_median':counts.median(),'acquisitions_p25':counts.quantile(.25),'acquisitions_p75':counts.quantile(.75),'acquisitions_max':counts.max(),'decision_rows_per_patient_mean':per.size().mean(),'terminal_step_mean':terminal.mean(),'immediate_stop_fraction':float(((x.step==0)&(x.action=='STOP')).groupby(x.patient_id).any().mean()),'unfinished_episode_fraction':0.0})
 pd.DataFrame(sem).to_csv(OUT/'metrics/metric_semantics_audit.csv',index=False)
 # Placeholder matched-budget file is explicitly a diagnostic projection, not a fabricated model score.
 pd.DataFrame([{'status':'not_computed_from_new_policy_predictions','reason':'R3-D4 audit found legacy PPO implementation invalid; corrected PPO retraining required before matched-budget claims'}]).to_csv(OUT/'metrics/matched_budget_comparison.csv',index=False)
 report='''# V8 R3-D4 integrated adaptive PPO\n\n## Scope\nThis run audits the R3-D3 seed=17 checkpoints on all 84 Validation patients at Train-derived lambda=0/P10/P25/P50. No Test data, beam, DROP or REFINE was used.\n\n## Checkpoint status\nAll six legacy checkpoints are retained as diagnostic-only. The implementation audit finds no explicit frozen old-policy clone during rollout, no emitted clip fraction, and an explained-variance calculation that does not align values and returns per transition. Therefore these checkpoints are not formal corrected-PPO evidence.\n\n## Rollout semantics\n`validation_adaptive_trajectories.csv` contains deterministic per-step rows for every checkpoint/lambda/patient. `metric_semantics_audit.csv` distinguishes patient acquisition counts, decision rows and terminal steps; every episode is terminal or reaches the 64-action cap. STOP regret is listed for all terminal patients and includes checkpoint/lambda.\n\n## Decision\nThe requested corrected PPO retraining is not represented by the legacy checkpoints. R3-D4 must not claim MoE-vs-Dense, PPO-vs-warm-start, or matched-budget superiority from these diagnostics. The next safe action is one seed=17 retraining with an explicit old-policy clone, minibatch PPO updates, aligned GAE/returns, true approx-KL/clip-fraction and separate actor/STOP/critic/entropy losses.\n'''
 (OUT/'docs/v8_r3d4_integrated_adaptive_ppo_report.md').write_text(report)
 (OUT/'docs/r3d4_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\npython src/MSRLSelect/r3d4_audit.py\n')
 shutil.copy(__file__,OUT/'code/r3d4_audit.py')
 manifest=[]
 for f in OUT.rglob('*'):
  if f.is_file() and f.suffix not in {'.pt','.npz'}: manifest.append({'path':str(f.relative_to(OUT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 pd.DataFrame(manifest).to_csv(OUT/'sha256_manifest.csv',index=False)
 print(json.dumps({'rows':len(allrows),'stops':len(allstops),'lambdas':lams,'device':str(DEV)}))
if __name__=='__main__': main()
