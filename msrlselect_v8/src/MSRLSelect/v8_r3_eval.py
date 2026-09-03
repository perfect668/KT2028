#!/usr/bin/env python3
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore'); OUT=ROOT/'outputs/thesis_rl_v8/r3_hierarchical_evidence_policy'; DEV=torch.device('cuda')
sys.path.insert(0,str(ROOT/'src/MSRLSelect'))
from v7_e2m_train import load_data
from v8_r2_formal import FixedE2, DenseEvidenceEncoder
from v8_r3_train import HierPolicy,evidence
def main():
 tr,va,ct,cv,st,sv=load_data(); cv,sv=cv.float().to(DEV),sv.float().to(DEV); ck=torch.load(OUT/'checkpoints/r3_dense_seed17_best.pt',map_location='cpu',weights_only=False); pol=HierPolicy(ct.shape[-1],384).to(DEV);pol.load_state_dict(ck['model']);pol.eval(); dense=DenseEvidenceEncoder(st.shape[-1]).to(DEV); env=FixedE2(cv.shape[-1],-1); y=va.label.values; rows=[]
 for budget in [1,2,4,8,16]:
  active=torch.zeros(len(va),64,dtype=torch.bool,device=DEV); order=torch.zeros_like(active,dtype=torch.long)
  for step in range(budget):
   p,en,dis,_=env.predict(cv,sv.flatten(-2),active); ev=evidence('dense',None,dense,cv,sv,active); l,stop,val=pol(cv,ev,active,order,p,en,dis,torch.zeros_like(p)); l=l.masked_fill(active,-1e9); a=l.argmax(1); rr=torch.arange(len(va),device=DEV); active[rr,a]=True;order[rr,a]=step+1
  p=env.predict(cv,sv.flatten(-2),active)[0].cpu().numpy(); au=roc_auc_score(y,p); ap=average_precision_score(y,p);rows.append(dict(encoder='dense',policy='hierarchical_fixed',budget=budget,auroc=au,auprc=ap,s_val=.5*(au+ap),brier=brier_score_loss(y,p),mean_acquire=float(active.sum(1).float().mean()),pred_std=float(p.std())))
 for name in ['random','coarse_only']:
  for budget in [1,2,4,8,16]:
   g=torch.Generator(device=DEV).manual_seed(7000+budget); active=torch.zeros(len(va),64,dtype=torch.bool,device=DEV)
   if name=='random': active.scatter_(1,torch.rand(len(va),64,device=DEV,generator=g).argsort(1)[:,:budget],True)
   else: active.scatter_(1,cv.norm(dim=-1).argsort(1,descending=True)[:,:budget],True)
   p=env.predict(cv,sv.flatten(-2),active)[0].cpu().numpy(); au=roc_auc_score(y,p);ap=average_precision_score(y,p);rows.append(dict(encoder='baseline',policy=name,budget=budget,auroc=au,auprc=ap,s_val=.5*(au+ap),brier=brier_score_loss(y,p),mean_acquire=float(active.sum(1).float().mean()),pred_std=float(p.std())))
 pd.DataFrame(rows).to_csv(OUT/'metrics/fixed_budget_actor_results_corrected.csv',index=False)
if __name__=='__main__':main()
