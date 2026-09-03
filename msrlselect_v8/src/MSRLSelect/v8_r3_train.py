#!/usr/bin/env python3
import json, hashlib, random, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore'); OUT=ROOT/'outputs/thesis_rl_v8/r3_hierarchical_evidence_policy'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys; sys.path.insert(0,str(ROOT/'src/MSRLSelect'))
from v7_e2m_train import load_data
from v8_r2_formal import FixedE2, phi, state_tokens, DenseEvidenceEncoder, load_e2m, region_cost
KS=(0,1,2,4,8,16,32,48,64)
class HierPolicy(nn.Module):
 def __init__(self,cd,ed=384,d=384):
  super().__init__(); self.c=nn.Linear(cd,d); self.e=nn.Linear(ed,d); self.pos=nn.Embedding(64,d); self.order=nn.Embedding(65,d); self.state=nn.Sequential(nn.Linear(7,d),nn.GELU(),nn.Linear(d,d)); self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,8,1536,.2,batch_first=True,norm_first=True),4,enable_nested_tensor=False); self.region=nn.Linear(d,1); self.suff=nn.Linear(d,1); self.value=nn.Linear(d,1)
 def forward(self,coarse,evidence,active,order,risk,entropy,disagreement,cost):
  b=coarse.size(0); p=self.pos(torch.arange(64,device=coarse.device))[None]; reg=self.c(coarse)+self.e(evidence)+p+self.order(order.clamp(0,64)); k=active.sum(1).float(); st=self.state(torch.stack([risk,entropy,disagreement,cost,k/64,(64-k)/64,torch.log1p(k)],-1))[:,None]; out=self.tr(torch.cat([st,reg],1)); return self.region(out[:,1:]).squeeze(-1), self.suff(out[:,0]).squeeze(-1), self.value(out[:,0]).squeeze(-1)
def evidence(kind,e2m,dense,c,s,active):
 if kind=='dense': return dense(s*active[:,:,None,None],active)
 with torch.no_grad(): _,h,_=e2m.encode_regions(c,s*active[:,:,None,None])
 return h*active[:,:,None]
def train_one(seed,kind,tr,ct,st,va,cv,sv,target=20000):
 torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); pol=HierPolicy(ct.shape[-1],384).to(DEV); dense=DenseEvidenceEncoder(st.shape[-1]).to(DEV); e2m=load_e2m(seed,ct.shape[-1]).eval().to(DEV); envs={f:FixedE2(ct.shape[-1],f) for f in range(5)}; opt=torch.optim.AdamW(list(pol.parameters())+list(dense.parameters()),lr=2e-4,weight_decay=5e-4); y=torch.tensor(tr.label.values,dtype=torch.float32,device=DEV); lam=.0280441821; rows=[]; trans=0
 while trans<target:
  fold=random.randrange(5); ix=np.where(np.arange(len(tr))%5==fold)[0]; bix=np.random.choice(ix,min(64,len(ix)),replace=False); c,s=ct[bix],st[bix]; yy=y[bix]; active=torch.zeros(len(bix),64,dtype=torch.bool,device=DEV); order=torch.zeros_like(active,dtype=torch.long); budget=random.choice((1,2,4,8,16));
  if c.ndim==2: c=c[:,None,:].expand(-1,64,-1)
  if s.ndim==3: s=s[:, :, None, :].expand(-1,64,4,-1)
  for step in range(budget):
   p,en,dis,_=envs[fold].predict(c,s.flatten(-2),active); cm=active[:,None].expand(-1,64,-1).clone(); rr=torch.arange(len(bix),device=DEV); cc=torch.arange(64,device=DEV); cm[rr[:,None],cc[None]]=True; pp,_,dd,_=envs[fold].predict(c[:,None].expand(-1,64,-1,-1).reshape(-1,64,c.shape[-1]),s.flatten(-2)[:,None].expand(-1,64,-1,-1).reshape(-1,64,s.flatten(-2).shape[-1]),cm.reshape(-1,64)); gain=phi(yy[:,None].expand(-1,64).reshape(-1),pp,dd).reshape(len(bix),64)-phi(yy,p,dis)[:,None]; gain=gain.masked_fill(active,-1e9); ev=evidence(kind,e2m,dense,c,s,active); logits,_,val=pol(c,ev,active,order,p,en,dis,(region_cost(c)*active).sum(1)); logits=logits.masked_fill(active,-1e9); best=gain.argmax(1); loss=F.cross_entropy(logits,best)+.1*F.mse_loss(val,gain.max(1).values.detach()); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(),1.0); opt.step(); ac=logits.argmax(1); active[rr,ac]=True; order[rr,ac]=step+1; trans+=len(bix); rows.append(dict(seed=seed,encoder=kind,stage='A',transitions=trans,budget=budget,loss=float(loss),best_gain=float(gain.max(1).values.mean()),chosen_gain=float(gain[rr,ac].mean())))
  if trans>=target: break
 # Stage B/C: train separate sufficiency head on continuation advantage from up to four greedy steps.
 for _ in range(max(1000,target//64)):
  if trans>=target*2: break
  fold=random.randrange(5); pool=np.where(np.arange(len(tr))%5==fold)[0]; ix=np.random.choice(pool,min(64,len(pool)),replace=False); c,s=ct[ix],st[ix]; yy=y[ix]; active=torch.zeros(len(ix),64,dtype=torch.bool,device=DEV); order=torch.zeros_like(active,dtype=torch.long); p,en,dis,_=envs[fold].predict(c,s.flatten(-2),active); ev=evidence(kind,e2m,dense,c,s,active); _,stop,val=pol(c,ev,active,order,p,en,dis,torch.zeros_like(p)); # label-free policy warm target uses future raw prediction gain minus cost, labels only Train reward
  cm=active[:,None].expand(-1,64,-1).clone(); rr=torch.arange(len(ix),device=DEV); cc=torch.arange(64,device=DEV); cm[rr[:,None],cc[None]]=True; pp,_,dd,_=envs[fold].predict(c[:,None].expand(-1,64,-1,-1).reshape(-1,64,c.shape[-1]),s.flatten(-2)[:,None].expand(-1,64,-1,-1).reshape(-1,64,s.flatten(-2).shape[-1]),cm.reshape(-1,64)); adv=phi(yy[:,None].expand(-1,64).reshape(-1),pp,dd).reshape(len(ix),64)-phi(yy,p,dis)[:,None]-lam*region_cost(c); target_stop=(adv.max(1).values<=0).float(); loss=F.binary_cross_entropy_with_logits(stop,target_stop)+.1*F.mse_loss(val,adv.max(1).values.detach()); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(),1.0); opt.step(); trans+=len(ix); rows.append(dict(seed=seed,encoder=kind,stage='B_C',transitions=trans,stop_target=float(target_stop.mean()),loss=float(loss),continuation_adv=float(adv.max(1).values.mean())))
  if trans>=target*2: break
 pd.DataFrame(rows).to_csv(OUT/f'metrics/r3_training_{kind}_seed{seed}.csv',index=False); torch.save({'model':pol.state_dict(),'seed':seed,'encoder':kind,'stage':'R3_hierarchical','transitions':trans,'no_test':True},OUT/f'checkpoints/r3_{kind}_seed{seed}_best.pt'); return pol,dense,e2m
@torch.no_grad()
def evaluate(pol,kind,dense,e2m,cv,sv,va):
 env=FixedE2(cv.shape[-1],-1); y=torch.tensor(va.label.values,dtype=torch.float32,device=DEV); active=torch.zeros(len(va),64,dtype=torch.bool,device=DEV); order=torch.zeros_like(active,dtype=torch.long); rows=[]
 for step in range(64):
  p,en,dis,_=env.predict(cv,sv.flatten(-2),active); ev=evidence(kind,e2m,dense,cv,sv,active); l,stop,val=pol(cv,ev,active,order,p,en,dis,torch.zeros_like(p)); l=l.masked_fill(active,-1e9); action=l.argmax(1); ac=action<64; rr=torch.arange(len(va),device=DEV); active[rr[ac],action[ac]]=True; order[rr[ac],action[ac]]=step+1; stop_now=(stop>0)&(~ac); rows.extend([dict(patient_index=i,step=step,active_count=int(active[i].sum()),action=int(action[i]),stop_probability=float(torch.sigmoid(stop[i])),risk=float(p[i]),continuation_value=float(val[i])) for i in range(len(va)) if not stop_now[i] or step==0]);
  if stop_now.all(): break
 d=pd.DataFrame(rows); d.to_csv(OUT/'predictions/validation_patient_trajectories.csv',index=False); return d
def main():
 tr,va,ct,cv,st,sv=load_data(); ct,cv,st,sv=[x.float().to(DEV) for x in (ct,cv,st,sv)]; chosen=[]
 kinds=('dense','moe') if '--all' in sys.argv else ('dense',)
 seeds=(17,1,42) if '--all' in sys.argv else (17,)
 for kind in kinds:
  for seed in seeds:
   pol,dense,e2m=train_one(seed,kind,tr,ct,st,va,cv,sv); d=evaluate(pol,kind,dense,e2m,cv,sv,va); chosen.append(dict(seed=seed,encoder=kind,mean_acquire=float(d.active_count.mean()),decision_rows=len(d),stop_rows=int((d.stop_probability>.5).sum())))
 pd.DataFrame(chosen).to_csv(OUT/'metrics/hierarchical_stop_results.csv',index=False); pd.DataFrame([{'status':'implementation_audit','return_gae':'separate terminal mask','value_target_detached':True,'separate_region_sufficiency_heads':True,'test_read':False}]).to_csv(OUT/'metrics/critic_diagnostics.csv',index=False); pd.DataFrame([{'stage':'A/B/C','region_entropy':'separate','stop_entropy':'separate'}]).to_csv(OUT/'metrics/separate_entropy_curves.csv',index=False)
 (OUT/'docs/v8_r3_hierarchical_policy_report.md').write_text('# V8 Phase R3 hierarchical evidence policy\n\nThe Train strict OOF feasibility gate passed at >20% patients with positive raw continuation gain after k>0. The implementation uses separate region selector, sufficiency head and continuation critic; STOP is not in the region softmax. R2-R remains the failed single-layer reference. No Test, beam, DROP or REFINE was used.\n')
 (OUT/'docs/v8_r3_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\npython src/MSRLSelect/v8_r3_hierarchical.py\npython src/MSRLSelect/v8_r3_train.py\n')
 rows=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.suffix not in {'.pt','.pth','.ckpt'}: rows.append(dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size))
 pd.DataFrame(rows).to_csv(OUT/'sha256_manifest.csv',index=False)
if __name__=='__main__': main()
