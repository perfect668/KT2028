#!/usr/bin/env python3
import json, hashlib, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore'); OUT=ROOT/'outputs/thesis_rl_v8/r3_hierarchical_evidence_policy'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for n in ('metrics','predictions','checkpoints','docs','logs','configs'): (OUT/n).mkdir(parents=True,exist_ok=True)
import sys; sys.path.insert(0,str(ROOT/'src/MSRLSelect'))
from v7_e2m_train import load_data, base_model
from v8_r2_formal import FixedE2, phi, region_cost

KS=(0,1,2,4,8,16,32,48,64)
def masks(n,k,seed):
 g=torch.Generator(device=DEV).manual_seed(seed); m=torch.zeros(n,64,dtype=torch.bool,device=DEV)
 if k: m.scatter_(1,torch.rand(n,64,device=DEV,generator=g).argsort(1)[:,:k],True)
 return m
def feasibility(tr,coarse,slots):
 rows=[]; allg=[]
 for fold in range(5):
  ix=np.where(np.arange(len(tr))%5==fold)[0]; env=FixedE2(coarse.shape[-1],fold); c=coarse[ix]; s=slots[ix].flatten(-2); y=torch.tensor(tr.label.values[ix],dtype=torch.float32,device=DEV); costs=region_cost(c)
  for k in (0,1,2,4,8):
   active=masks(len(ix),k,9100+fold*101+k); p,_,d,_=env.predict(c,s,active); b=len(ix)
   cm=active[:,None].expand(-1,64,-1).clone(); cm[torch.arange(b,device=DEV)[:,None],torch.arange(64,device=DEV)[None]]=True
   pp,_,dd,_=env.predict(c[:,None].expand(-1,64,-1,-1).reshape(-1,64,c.shape[-1]),s[:,None].expand(-1,64,-1,-1).reshape(-1,64,s.shape[-1]),cm.reshape(-1,64))
   gain=(phi(y[:,None].expand(-1,64).reshape(-1),pp,dd).reshape(b,64)-phi(y,p,d)[:,None]); net=gain-costs
   legal=~active; gain=gain.masked_fill(~legal,-1e9); net=net.masked_fill(~legal,-1e9)
   for j in range(b):
    gv=gain[j][gain[j]>-1e8].detach().cpu().numpy(); nv=net[j][net[j]>-1e8].detach().cpu().numpy(); rows.append(dict(fold=fold,patient_index=int(ix[j]),k=k,positive_raw_actions=int((gv>0).sum()),positive_net_actions=int((nv>0).sum()),best_raw=float(gv.max()),random_raw=float(gv.mean()),best_net=float(nv.max()),random_net=float(nv.mean()),best_minus_random=float(gv.max()-gv.mean()),history_seed=9100+fold*101+k)); allg.extend([(int(ix[j]),k,float(x)) for x in gv])
 return pd.DataFrame(rows)

def main():
 tr,va,ct,cv,st,sv=load_data(); ct=ct.float().to(DEV); st=st.float().to(DEV)
 f=feasibility(tr,ct,st); f.to_csv(OUT/'metrics/counterfactual_continuation_value.csv',index=False)
 boot=[]; rng=np.random.default_rng(7)
 for k,g in f.groupby('k'):
  x=g.best_minus_random.to_numpy(); means=[]
  for _ in range(2000): means.append(float(x[rng.integers(0,len(x),len(x))].mean()))
  boot.append(dict(k=k,patients=len(g),patients_any_raw_positive=float((g.positive_raw_actions>0).mean()),patients_any_net_positive=float((g.positive_net_actions>0).mean()),best_minus_random_mean=float(x.mean()),ci95_low=float(np.quantile(means,.025)),ci95_high=float(np.quantile(means,.975)),max_gain=float(g.best_raw.max()),random_gain=float(g.random_raw.mean())))
 b=pd.DataFrame(boot); b.to_csv(OUT/'metrics/fixed_budget_actor_results.csv',index=False)
 gate=float(b.loc[b.k>0,'patients_any_raw_positive'].max() if len(b) else 0)
 # Implementation/audit records are explicit; subsequent stages are gated by feasibility.
 pd.DataFrame([dict(item='feasibility_gate',value=gate,threshold=.20,passed=gate>=.20),dict(item='r2_reference',value='retained',passed=True),dict(item='test_read',value=False,passed=True),dict(item='beam_drop_refine',value=False,passed=True)]).to_csv(OUT/'metrics/hierarchical_stop_results.csv',index=False)
 report=f'''# V8 Phase R3 environment feasibility\n\nR2-R is retained as the failed single-layer STOP/critic reference. This run uses fixed E2 and Train strict OOF folds.\n\nThe decisive gate is the fraction of patients with at least one positive raw (pre-cost) ACQUIRE gain after k>0. Observed maximum across k=1,2,4,8: {gate:.4f}; threshold: 0.20.\n\nNo Test data, beam, DROP, REFINE or predictor retraining was used.\n'''
 (OUT/'docs/r3_environment_feasibility.md').write_text(report)
 (OUT/'docs/r3_ppo_implementation_audit.md').write_text('# R3 PPO implementation audit\n\nR3 requires separate region and sufficiency heads, terminal masking, detached value targets, normalized returns/advantages and a continuation critic. The current artifact records the audit gate before training; no R2 runner is reused as a formal result.\n')
 (OUT/'docs/v8_r3_hierarchical_policy_report.md').write_text(report+'\nStage A/B/C are only authorized when the feasibility gate passes.\n')
 (OUT/'docs/v8_r3_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\npython src/MSRLSelect/v8_r3_hierarchical.py\n')
 rows=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.suffix not in {'.pt','.pth','.ckpt'}: rows.append(dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size))
 pd.DataFrame(rows).to_csv(OUT/'sha256_manifest.csv',index=False)
 print(json.dumps({'gate':gate,'passed':gate>=.2,'rows':len(f)}))
if __name__=='__main__': main()
