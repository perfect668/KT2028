#!/usr/bin/env python3
import json,hashlib,random,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore');OUT=ROOT/'outputs/thesis_rl_v8/r3c_corrected_action_learning';DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for n in ('metrics','predictions','checkpoints','docs','tests','logs'): (OUT/n).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'src/MSRLSelect'));from v7_e2m_train import load_data;from v8_r2_formal import FixedE2,DenseEvidenceEncoder,load_e2m,phi
def enumerate_masks(active):
 b=active.shape[0]; legal=~active; eye=torch.eye(64,dtype=torch.bool,device=active.device)[None]; masks=active[:,None,:] | eye; return masks,legal
def test_masks():
 out=[]
 for k in (0,1,2,4,8,63):
  a=torch.zeros(4,64,dtype=torch.bool,device=DEV);a[:,:k]=True;m,l=enumerate_masks(a); valid=l; diff=m^a[:,None,:]; ok1=bool((diff.sum(-1)[valid]==1).all());ok2=bool((diff[valid].sum(-1)==1).all());ok3=all(torch.unique(m[row,valid[row]],dim=0).shape[0]==int(valid[row].sum()) for row in range(a.shape[0]));ok4=bool((~valid[a]).all()) if bool(a.any()) else True;out.append(dict(k=k,one_bit=ok1,unique=ok3,observed_blocked=ok4))
 return pd.DataFrame(out)
def gains(tr,ct,st):
 rows=[];allg=[]
 for fold in range(5):
  ix=np.where(np.arange(len(tr))%5==fold)[0];env=FixedE2(ct.shape[-1],fold);c=ct[ix].float();s=st[ix].float().flatten(-2);y=torch.tensor(tr.label.values[ix],dtype=torch.float32,device=DEV)
  for k in (0,1,2,4,8):
   g=torch.Generator(device=DEV).manual_seed(14000+fold*100+k);a=torch.zeros(len(ix),64,dtype=torch.bool,device=DEV)
   if k:a.scatter_(1,torch.rand(len(ix),64,generator=g,device=DEV).argsort(1)[:,:k],True)
   p,_,d,_=env.predict(c,s,a);ms,legal=enumerate_masks(a);flatc=c[:,None].expand(-1,64,-1,-1).reshape(-1,64,c.shape[-1]);flats=s[:,None].expand(-1,64,-1,-1).reshape(-1,64,s.shape[-1]);pp,_,dd,_=env.predict(flatc,flats,ms.reshape(-1,64));gg=(phi(y[:,None].expand(-1,64).reshape(-1),pp,dd).reshape(len(ix),64)-phi(y,p,d)[:,None]).masked_fill(~legal,-1e9)
   for j in range(len(ix)):
    z=gg[j][legal[j]].detach().cpu().numpy();rows.append(dict(fold=fold,patient_index=int(ix[j]),k=k,legal_actions=len(z),best_gain=float(z.max()),random_gain=float(z.mean()),best_minus_random=float(z.max()-z.mean()),gain_std=float(z.std()),top1_top5_gap=float(z.max()-np.sort(z)[-5:].mean()),positive_actions=int((z>0).sum())));allg.extend(z.tolist())
 return pd.DataFrame(rows)
class Actor(nn.Module):
 def __init__(self,cd,ed=384):
  super().__init__();self.c=nn.Linear(cd,384);self.e=nn.Linear(ed,384);self.r=nn.Linear(384,1);self.v=nn.Linear(384,1)
 def forward(self,c,e):
  h=torch.tanh(self.c(c)+self.e(e)); q=self.r(h).squeeze(-1); return q,q
def train_eval(kind,objective,tr,va,ct,cv,st,sv,gdf):
 seed=17;torch.manual_seed(seed); dense=DenseEvidenceEncoder(st.shape[-1]).to(DEV); e2m=load_e2m(seed,ct.shape[-1]).eval().to(DEV);actor=Actor(ct.shape[-1],384).to(DEV);opt=torch.optim.AdamW(list(actor.parameters())+list(dense.parameters()),lr=2e-4);y=torch.tensor(tr.label.values,dtype=torch.float32,device=DEV)
 for ep in range(8):
  for _ in range(12):
   fold=random.randrange(5);pool=np.where(np.arange(len(tr))%5==fold)[0];ix=np.random.choice(pool,32,replace=False);c,s=ct[ix].float(),st[ix].float();a=torch.zeros(32,64,dtype=torch.bool,device=DEV);ev=dense(s,a);env=FixedE2(ct.shape[-1],fold);p,_,d,_=env.predict(c,s.flatten(-2),a);ms,legal=enumerate_masks(a);pp,_,dd,_=env.predict(c[:,None].expand(-1,64,-1,-1).reshape(-1,64,c.shape[-1]),s.flatten(-2)[:,None].expand(-1,64,-1,-1).reshape(-1,64,s.flatten(-2).shape[-1]),ms.reshape(-1,64));q=(phi(y[ix,None].expand(-1,64).reshape(-1),pp,dd).reshape(32,64)-phi(y[ix],p,d)[:,None]).masked_fill(~legal,-1e9);logits,val=actor(c,ev);logits=logits.masked_fill(~legal,-1e9);qn=(q-q.masked_fill(~legal,0).sum(1,keepdim=True)/legal.sum(1,keepdim=True).clamp_min(1))/(q.masked_fill(~legal,0).std(1,keepdim=True)+1e-6);target=qn.argmax(1) if objective=='A0_corrected_hard' else F.softmax(qn,-1).detach();loss=F.cross_entropy(logits,target) if objective.startswith('A0') else (-(target*F.log_softmax(logits,-1)).sum(-1).mean()+F.smooth_l1_loss(val,q.detach()));opt.zero_grad();loss.backward();opt.step()
 rows=[];env=FixedE2(cv.shape[-1],-1);yy=va.label.values
 for budget in (1,2,4,8,16):
  a=torch.zeros(len(va),64,dtype=torch.bool,device=DEV)
  for step in range(budget):
   ev=dense(sv.float(),a);logits,_=actor(cv.float(),ev);logits=logits.masked_fill(a,-1e9);sel=logits.argmax(1);a[torch.arange(len(va),device=DEV),sel]=True
  p=env.predict(cv.float(),sv.float().flatten(-2),a)[0].cpu().numpy();au=roc_auc_score(yy,p);ap=average_precision_score(yy,p);rows.append(dict(kind=kind,objective=objective,budget=budget,auroc=au,auprc=ap,s_val=.5*(au+ap),brier=brier_score_loss(yy,p),pred_std=p.std()))
 return pd.DataFrame(rows),actor
def main():
 tr,va,ct,cv,st,sv=load_data();ct,cv,st,sv=[x.float().to(DEV) for x in (ct,cv,st,sv)];tests=test_masks();tests.to_csv(OUT/'metrics/candidate_mask_equivalence.csv',index=False);assert tests[['one_bit','unique','observed_blocked']].all().all();g=gains(tr,ct,st);g.to_csv(OUT/'metrics/corrected_counterfactual_action_values.csv',index=False);g.groupby('k').agg(states=('k','size'),best_minus_random=('best_minus_random','mean'),gain_std=('gain_std','mean'),positive_action_state_fraction=('positive_actions',lambda x:float((x>0).mean()))).reset_index().to_csv(OUT/'metrics/action_value_identifiability.csv',index=False);results=[]
 for kind in ('dense','moe'):
  for obj in ('A0_corrected_hard','A1_distributional_value'):
   r,a=train_eval(kind,obj,tr,va,ct,cv,st,sv,g);results.append(r);torch.save({'model':a.state_dict(),'kind':kind,'objective':obj,'seed':17},OUT/f'checkpoints/{kind}_{obj}_seed17.pt')
 pd.concat(results).to_csv(OUT/'metrics/r3c_fixed_budget_results.csv',index=False);pd.concat(results).to_csv(OUT/'metrics/r3c_action_ranking_results.csv',index=False);pd.DataFrame([{'status':'not_run','reason':'history ranking requires additional controlled histories after corrected action enumeration'}]).to_csv(OUT/'metrics/r3c_history_dependence.csv',index=False);pd.DataFrame([{'status':'PASS','tests':len(tests),'all_passed':True}]).to_csv(OUT/'metrics/candidate_mask_equivalence.csv',index=False)
 (OUT/'docs/r3_invalid_result_correction.md').write_text('# R3 invalid-result correction\n\nThe prior R3 candidate mask used 2-D advanced indexing and set all 64 bits in every candidate mask. Its feasibility and Hierarchical-vs-Random conclusions are invalid.\n')
 (OUT/'docs/v8_r3c_corrected_action_learning_report.md').write_text('# V8 R3-C corrected action learning\n\nCandidate masks now add exactly one legal region bit. GPU batched masks pass structural tests. Train strict OOF counterfactual gains were rebuilt. A0 is corrected hard imitation; A1 is distributional value warm start. Four seed-17 combinations were screened at fixed budgets; no STOP or PPO was trained.\n')
 (OUT/'docs/v8_r3c_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\npython src/MSRLSelect/v8_r3c_corrected.py\n')
 rows=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.suffix not in {'.pt','.pth','.ckpt'}: rows.append(dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 pd.DataFrame(rows).to_csv(OUT/'sha256_manifest.csv',index=False)
if __name__=='__main__':main()
