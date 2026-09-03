#!/usr/bin/env python3
from pathlib import Path
import itertools, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from v7_r1_revised_inference import ROOT,EOUT,R1,DEV,KS,MASK_SEEDS,load_data,load_model

EPS=1e-8
def entropy(p):p=np.clip(p,1e-6,1-1e-6);return -(p*np.log(p)+(1-p)*np.log(1-p))
def bce(y,p):p=np.clip(p,1e-6,1-1e-6);return -(y*np.log(p)+(1-y)*np.log(1-p))
def calibrator():
 d=pd.read_csv(R1/'metrics/e2_calibration_parameters.csv');out={}
 for r in d.itertuples():out[r.mode]=(r.intercept,np.array(list(map(float,r.coefficients.split(';')))))
 return out
def cal(p,k,C):
 mode='k64' if k==64 else 'partial';b,w=C[mode];p=np.asarray(p);z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)));rho=k/64.
 X=z[:,None] if mode=='k64' else np.c_[z,z*rho,np.full(len(z),rho)];v=X@w+b;return 1/(1+np.exp(-v))
def mask_for(idx,k,hs):
 rng=np.random.default_rng(hs+100003*idx);m=np.zeros(64,bool);m[rng.choice(64,k,replace=False)]=1;return m
def evaluate():
 tr,va,ct,cv,ft,fv=load_data();fold=np.arange(len(tr))%5;C=calibrator();state_rows=[];ranking={}
 for f in range(5):
  models=[load_model(EOUT/f'checkpoints/E2_oof_fold{f}_seed{s}_best.pt',ct.shape[-1],ft.shape[-1]) for s in [1,17,42]]
  for idx in np.where(fold==f)[0]:
   c=ct[idx:idx+1].float().to(DEV);fine=ft[idx:idx+1].float().to(DEV);y=int(tr.iloc[idx].label);pid=tr.iloc[idx].case_id
   full=[]
   with torch.no_grad():
    mm=torch.ones(1,1,64,dtype=torch.bool,device=DEV)
    for n in models:full.append(float(torch.sigmoid(n(c,fine,mm)[0][0,0])))
   full=cal(np.array(full),64,C)
   for k in KS[:-1]:
    for hi,hs in enumerate(MASK_SEEDS):
     active=mask_for(idx,k,hs);cand=np.where(~active)[0];masks=np.repeat(active[None,:],len(cand)+1,0);masks[np.arange(1,len(cand)+1),cand]=True;m=torch.from_numpy(masks).to(DEV)[:,None,:].transpose(0,1)
     probs=[]
     with torch.no_grad():
      for n in models:probs.append(torch.sigmoid(n(c,fine,m)[0][0]).cpu().numpy())
     probs=np.stack(probs);base=cal(probs[:,0],k,C);after=np.stack([cal(probs[s,1:],k+1,C) for s in range(3)])
     ub=float(np.std(base));ua=np.std(after,axis=0);dunc=ub-ua
     dl=np.stack([(-bce(y,after[s])+bce(y,base[s]))+.5*dunc for s in range(3)])
     df=np.stack([(-np.abs(after[s]-full[s])+abs(base[s]-full[s]))+.5*dunc for s in range(3)])
     for version,D in [('label_based',dl),('label_free',df)]:
      mean=D.mean(0);signal=float(mean.std());noise=float(D.std(0).mean());snr=signal/(noise+EPS);best=float(mean.max());rnd=float(mean.mean());tops=np.argmax(D,1);agree=float(len(set(tops.tolist()))==1)
      state_rows.append(dict(patient_id=pid,label=y,fold=f,k=k,history_seed=hs,version=version,signal=signal,noise=noise,snr=snr,p90_p10=float(np.quantile(mean,.9)-np.quantile(mean,.1)),oracle_best=best,random_mean=rnd,best_minus_random=best-rnd,three_seed_best_action_agreement=agree,base_probability=float(base.mean()),base_disagreement=ub))
      vec=np.full(64,np.nan);vec[cand]=mean;ranking[(pid,k,version,hs)]=vec
 pd.DataFrame(state_rows).to_csv(R1/'metrics/e2_corrected_action_snr.csv',index=False)
 hist=[]
 for pid in tr.case_id:
  for k in KS[:-1]:
   for version in ['label_based','label_free']:
    vv=[ranking[(pid,k,version,h)] for h in MASK_SEEDS];sp=[];jac=[];top=[]
    for a,b in itertools.combinations(vv,2):
     q=np.isfinite(a)&np.isfinite(b);sp.append(spearmanr(a[q],b[q]).statistic if q.sum()>2 else np.nan);A=set(np.argsort(np.nan_to_num(a,nan=-1e9))[-5:]);B=set(np.argsort(np.nan_to_num(b,nan=-1e9))[-5:]);jac.append(len(A&B)/len(A|B))
    top=[int(np.nanargmax(v)) for v in vv];hist.append(dict(patient_id=pid,k=k,version=version,mean_spearman=np.nanmean(sp),mean_top5_jaccard=np.mean(jac),top1_change_rate=1-max(pd.Series(top).value_counts())/len(top)))
 pd.DataFrame(hist).to_csv(R1/'metrics/e2_history_dependence.csv',index=False)
def bootstrap():
 d=pd.read_csv(R1/'metrics/e2_corrected_action_snr.csv');g=d.groupby(['patient_id','k','version'],as_index=False).best_minus_random.mean();rng=np.random.default_rng(20260831);out=[]
 for (k,v),q in g.groupby(['k','version']):
  x=q.best_minus_random.values;bs=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(2000)]);out.append(dict(k=k,version=v,n_patients=len(x),mean=float(x.mean()),ci95_low=float(np.quantile(bs,.025)),ci95_high=float(np.quantile(bs,.975)),median_snr=float(d[(d.k==k)&(d.version==v)].snr.median())))
 pd.DataFrame(out).to_csv(R1/'metrics/e2_best_vs_random_bootstrap.csv',index=False)
def routing_audit():
 rows=[]
 for typ,pat in [('oof','E2_oof_fold*_seed*_best.pt'),('validation','E2_bottleneck_moe4_seed*_best.pt')]:
  for p in sorted((EOUT/'checkpoints').glob(pat)):
   sd=torch.load(p,map_location='cpu',weights_only=False)['model'];rk=[k for k in sd if any(z in k.lower() for z in ['router','expert','center'])];rows.append(dict(source=typ,checkpoint=p.name,has_trained_router=bool(rk),router_parameter_count=sum(sd[k].numel() for k in rk),top1_occupancy=np.nan,top2_occupancy=np.nan,patient_coverage=np.nan,routing_entropy=np.nan,effective_experts=np.nan,slot_cosine_similarity=np.nan,empty_slot_fraction=np.nan,status='UNAVAILABLE_NO_ROUTER' if not rk else 'AVAILABLE'))
 slots=torch.load(EOUT/'assets/train_moe_slots.pt',weights_only=True).float();norm=slots.norm(dim=-1);empty=(norm==0).float().mean().item();sim=[]
 for a,b in itertools.combinations(range(4),2):sim.append(torch.nn.functional.cosine_similarity(slots[:,:,a],slots[:,:,b],dim=-1)[(norm[:,:,a]>0)&(norm[:,:,b]>0)].mean().item())
 rows.append(dict(source='offline_slot_cache',checkpoint='train_moe_slots.pt',has_trained_router=False,router_parameter_count=0,top1_occupancy=np.nan,top2_occupancy=np.nan,patient_coverage=np.nan,routing_entropy=np.nan,effective_experts=np.nan,slot_cosine_similarity=np.nanmean(sim),empty_slot_fraction=empty,status='OFFLINE_KMEANS_DERIVED_NOT_TRAINED_ROUTER'))
 pd.DataFrame(rows).to_csv(R1/'metrics/e2_trained_routing_statistics.csv',index=False)
if __name__=='__main__':evaluate();bootstrap();routing_audit();print('done')
