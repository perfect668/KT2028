#!/usr/bin/env python3
import hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn.functional as F
from torch.distributions import Categorical,Bernoulli
ROOT=Path('/root/autodl-tmp/surgen_hemicore');OUT=ROOT/'outputs/thesis_rl_v8/r3d3_policy_repair_and_ppo';IN=ROOT/'outputs/thesis_rl_v8/r3d2_real_multihistory_ppo';SRC=ROOT/'src/MSRLSelect';sys.path.insert(0,str(SRC))
from v7_e2m_train import load_data
from v8_r2_formal import FixedE2,phi
from r3d3_real import Policy,DenseEnc
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu');EPS=1e-6
def online(seed=17):
 tr,va,ct,cv,st,sv=load_data();z=np.load(IN/'cache/multihistory_action_values.npz');q=z['q_values'];legal=z['legal_mask'];lam=float(np.quantile(q[(legal)&(q>0)],[.1])[0]);
 p=Policy(ct.shape[-1]).to(DEV);p.load_state_dict(torch.load(OUT/'checkpoints/Dense_A1_warmstart.pt',map_location=DEV,weights_only=False)['model']);enc=DenseEnc().to(DEV);opt=torch.optim.AdamW(list(p.parameters())+list(enc.parameters()),lr=5e-5);rng=np.random.default_rng(seed);logs=[];total=0;old_kl=0;envs={f:FixedE2(ct.shape[-1],f) for f in range(5)}
 while total<200000:
  fold=int(rng.integers(0,5)); pool=np.flatnonzero(np.arange(len(tr))%5==fold);pid=torch.tensor(rng.choice(pool,128),dtype=torch.long,device=DEV);active=torch.zeros(128,64,dtype=torch.bool,device=DEV);order=torch.full((128,64),-1,dtype=torch.long,device=DEV);rews=[];vals=[];logps=[];acts=[];dones=[];ent_r=[];ent_s=[];obs=[]
  env=envs[fold];pid_cpu=pid.cpu();c=ct[pid_cpu].float().to(DEV);slots=st[pid_cpu].float().to(DEV);y=torch.tensor(tr.label.values[pid_cpu.numpy()],device=DEV).float()
  for t in range(4):
   e=enc(slots,active);logits,stoplog,v=p(c,e,active,order);legal_now=~active;al=torch.cat([stoplog[:,None],logits.masked_fill(~legal_now,-1e9)],1);dist=Categorical(logits=al);a=dist.sample();lp=dist.log_prob(a);ent_r.append(Categorical(logits=logits.masked_fill(~legal_now,-1e9)).entropy());ent_s.append(Bernoulli(logits=stoplog).entropy());vals.append(v);logps.append(lp);acts.append(a.detach());obs.append((c,e,active.clone(),order.clone(),al.detach()));
   cur,_,dc,_=env.predict(c,slots.flatten(-2),active);acq=a-1;isstop=a==0;next_active=active.clone();idx=torch.where(~isstop)[0]
   if len(idx):next_active[idx,acq[idx]]=True
   nxt,_,dn,_=env.predict(c,slots.flatten(-2),next_active);r=torch.where(isstop,torch.zeros_like(cur),phi(y,nxt,dn)-phi(y,cur,dc)-lam);rews.append(r);dones.append(isstop);active=next_active;order[:,t]=torch.where(isstop,order[:,t],acq)
  with torch.no_grad():
   e=enc(slots,active);_,_,vlast=p(c,e,active,order)
  R=torch.zeros(128,device=DEV);gae=torch.zeros(128,device=DEV);adv=[];ret=[]
  for t in reversed(range(4)):
   mask=(~dones[t]).float();nextv=vlast if t==3 else vals[t+1].detach();delta=rews[t]+.99*nextv*mask-vals[t].detach();gae=delta+.99*.95*mask*gae;adv.insert(0,gae);ret.insert(0,gae+vals[t].detach())
  A=torch.stack(adv);Y=torch.stack(ret);A=(A-A.mean())/(A.std()+EPS);losses=[];kls=[]
  for t in range(4):
   c0,e0,m0,o0,oldal=obs[t];lg,sl,v=p(c0,e0,m0,o0);al=torch.cat([sl[:,None],lg.masked_fill(~(~m0),-1e9)],1);dist=Categorical(logits=al);old_dist=Categorical(logits=oldal);ratio=torch.exp(dist.log_prob(acts[t])-old_dist.log_prob(acts[t]).detach());oldv=vals[t].detach();vclip=oldv+(v-oldv).clamp(-.2,.2);kl=(F.softmax(oldal,-1)*(F.log_softmax(oldal,-1)-F.log_softmax(al,-1))).sum(-1).mean();kls.append(kl);actor_loss=-(torch.clamp(ratio,.8,1.2)*A[t].detach()).mean();value_loss=.5*torch.maximum((v-Y[t]).square(),(vclip-Y[t]).square()).mean();losses.append(actor_loss+value_loss+0.01*kl-.005*dist.entropy().mean())
  loss=torch.stack(losses).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(p.parameters())+list(enc.parameters()),.5);opt.step();total+=512;logs.append({'transitions':total,'loss':float(loss.detach()),'return_mean':float(Y.mean()),'region_entropy':float(torch.stack(ent_r).mean()),'stop_entropy':float(torch.stack(ent_s).mean()),'critic_explained_variance':float(1-((Y.detach()-torch.stack(vals).detach().mean(0))**2).var()/(Y.detach().var()+EPS)),'target_kl':float(torch.stack(kls).mean().detach())})
  if total in (51200,102400,153600,200192) or total>=200000:torch.save({'model':p.state_dict(),'encoder':enc.state_dict(),'seed':17,'transitions':total},OUT/f'checkpoints/seed17_online_ppo_{min(total,200000)//50000}k.pt')
 pd.DataFrame(logs).to_csv(OUT/'metrics/hierarchical_ppo_training_online.csv',index=False);torch.save({'model':p.state_dict(),'encoder':enc.state_dict(),'seed':17,'transitions':total},OUT/'checkpoints/seed17_online_ppo_best.pt')
 # Deterministic Validation adaptive rollout and independent fixed-budget metric recomputation.
 env=FixedE2(cv.shape[-1],-1);cc=cv.float().to(DEV);ss=sv.float().to(DEV);active=torch.zeros(len(va),64,dtype=torch.bool,device=DEV);order=torch.full((len(va),64),-1,dtype=torch.long,device=DEV);alive=torch.ones(len(va),dtype=torch.bool,device=DEV);rows=[];stoprows=[]
 for step in range(64):
  if not bool(alive.any()):break
  ee=enc(ss,active);lg,sl,v=p(cc,ee,active,order);stp=(torch.sigmoid(sl)>.5)&alive;sel=lg.masked_fill(active,-1e9).argmax(1);before=active.clone();alive=alive&~stp;active[torch.arange(len(va),device=DEV),sel]=active[torch.arange(len(va),device=DEV),sel]|alive;order[:,step]=torch.where(alive,sel,order[:,step]);risk,en,dis,_=env.predict(cc,ss.flatten(-2),active)
  for i in range(len(va)):
   if bool(stp[i]) or bool(alive[i]):rows.append({'patient_id':str(va.iloc[i].case_id),'label':int(va.label.iloc[i]),'step':step,'action':'STOP' if bool(stp[i]) else int(sel[i]),'active_count':int(active[i].sum()),'active_mask_hash':hashlib.sha1(active[i].cpu().numpy().tobytes()).hexdigest(),'risk':float(risk[i]),'entropy':float(en[i]),'disagreement':float(dis[i])})
  if bool(stp.any()):stoprows.extend([{'patient_id':str(va.iloc[i].case_id),'step':step,'stop_probability':float(torch.sigmoid(sl[i]))} for i in torch.where(stp)[0].cpu().tolist()])
 pd.DataFrame(rows).to_csv(OUT/'predictions/validation_adaptive_trajectories.csv',index=False);pd.DataFrame(stoprows).to_csv(OUT/'metrics/adaptive_stop_results.csv',index=False)
 pd.DataFrame(logs).to_csv(OUT/'metrics/critic_diagnostics.csv',index=False)
 # Independent recomputation is performed from the fixed-budget patient predictions.
 fixed=pd.read_csv(OUT/'predictions/validation_fixed_budget_predictions.csv');fixed['label']=fixed.patient_id.map(dict(zip(va.case_id,va.label)));agg=[]
 for (policy,budget),g in fixed.groupby(['policy','budget']):
  last=g.sort_values('step').groupby('patient_id').tail(1);pval=last.risk.values;y=last.label.values;au=__import__('sklearn.metrics').metrics.roc_auc_score(y,pval);ap=__import__('sklearn.metrics').metrics.average_precision_score(y,pval);agg.append({'policy':policy,'budget':budget,'auroc_independent':au,'auprc_independent':ap,'s_val_independent':(au+ap)/2,'brier_independent':__import__('sklearn.metrics').metrics.brier_score_loss(y,pval)})
 pd.DataFrame(agg).to_csv(OUT/'metrics/independent_metric_recalculation.csv',index=False)
if __name__=='__main__':online()
