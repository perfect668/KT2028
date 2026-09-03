#!/usr/bin/env python3
import itertools,numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from v7_e2m_seed17_audit import load,E2M,DEV
KS=[0,1,2,4,8,16,32,48];HS=[7101,7102,7103,7104,7105]
def ent(p):p=np.clip(p,1e-6,1-1e-6);return -(p*np.log(p)+(1-p)*np.log(1-p))
def bce(y,p):p=np.clip(p,1e-6,1-1e-6);return -(y*np.log(p)+(1-y)*np.log(1-p))
def mask(idx,k,seed):r=np.random.default_rng(seed+100003*idx);m=np.zeros(64,bool);m[r.choice(64,k,replace=False)]=1;return m
def main():
 tr,va,ct,cv,st,sv,model,ck=load();rows=[];ranks={}
 for idx in range(len(va)):
  c=cv[idx:idx+1].float().to(DEV);s=sv[idx:idx+1].float().to(DEV);y=int(va.iloc[idx].label);pid=va.iloc[idx].case_id
  with torch.no_grad():full=float(torch.sigmoid(model(c,s,torch.ones(1,1,64,dtype=torch.bool,device=DEV))['fused_logit'][0,0]))
  for k in KS:
   for hs in HS:
    active=mask(idx,k,hs);cand=np.where(~active)[0];mm=np.repeat(active[None],len(cand)+1,0);mm[np.arange(1,len(cand)+1),cand]=1;m=torch.from_numpy(mm).to(DEV)[None]
    with torch.no_grad():p=torch.sigmoid(model(c,s,m)['fused_logit'][0]).cpu().numpy()
    for version,d in [('label_based',-np.array([bce(y,x) for x in p[1:]])+bce(y,p[0])),('label_free',-np.abs(p[1:]-full)+abs(p[0]-full))]:
     rows.append(dict(patient_id=pid,label=y,k=k,history_seed=hs,version=version,action_signal=float(d.std()),action_noise=np.nan,corrected_snr=np.nan,oracle_best=float(d.max()),random_mean=float(d.mean()),best_minus_random=float(d.max()-d.mean()),p90_p10=float(np.quantile(d,.9)-np.quantile(d,.1)),snr_note='undefined for seed17-only screening'))
     v=np.full(64,np.nan);v[cand]=d;ranks[(pid,k,version,hs)]=v
 d=pd.DataFrame(rows);d.to_csv(E2M/'metrics/e2m_corrected_action_snr.csv',index=False);rng=np.random.default_rng(17);bs=[]
 g=d.groupby(['patient_id','k','version'],as_index=False).best_minus_random.mean()
 for (k,v),q in g.groupby(['k','version']):
  x=q.best_minus_random.values;b=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(2000)]);bs.append(dict(k=k,version=v,mean=x.mean(),ci95_low=np.quantile(b,.025),ci95_high=np.quantile(b,.975),n_patients=len(x)))
 pd.DataFrame(bs).to_csv(E2M/'metrics/e2m_best_vs_random_bootstrap.csv',index=False);hh=[]
 for pid in va.case_id:
  for k in KS:
   for version in ['label_based','label_free']:
    v=[ranks[(pid,k,version,h)] for h in HS];tops=[int(np.nanargmax(x)) for x in v];sp=[];jac=[]
    for a,b in itertools.combinations(v,2):
     q=np.isfinite(a)&np.isfinite(b);sp.append(spearmanr(a[q],b[q]).statistic);A=set(np.argsort(np.nan_to_num(a,nan=-1e9))[-5:]);B=set(np.argsort(np.nan_to_num(b,nan=-1e9))[-5:]);jac.append(len(A&B)/len(A|B))
    hh.append(dict(patient_id=pid,k=k,version=version,mean_spearman=np.nanmean(sp),top1_change_rate=1-max(pd.Series(tops).value_counts())/5,top5_jaccard=np.mean(jac)))
 pd.DataFrame(hh).to_csv(E2M/'metrics/e2m_history_dependence.csv',index=False);print(pd.DataFrame(bs).to_string(index=False))
if __name__=='__main__':main()
