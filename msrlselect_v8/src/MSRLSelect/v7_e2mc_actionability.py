#!/usr/bin/env python3
import itertools,numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from v7_e2mc_evaluate import O,load_model
from v7_e2m_train import load_data,DEV
KS=[0,1,2,4,8,16,32,48];HS=[7101,7102,7103,7104,7105];EPS=1e-8
def bce(y,p):p=np.clip(p,1e-6,1-1e-6);return -(y*np.log(p)+(1-y)*np.log(1-p))
def Cload():
 d=pd.read_csv(O/'metrics/e2m_calibration_parameters.csv');return {r.mode:(r.intercept,np.array(list(map(float,r.coefficients.split(';'))))) for r in d.itertuples()}
def cal(p,k,C):
 mode='k64' if k==64 else 'partial';b,w=C[mode];p=np.asarray(p);z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)));r=k/64.;X=z[:,None] if mode=='k64' else np.c_[z,z*r,np.full(len(z),r)];v=X@w+b;return 1/(1+np.exp(-v))
def mask(idx,k,h):r=np.random.default_rng(h+100003*idx);m=np.zeros(64,bool);m[r.choice(64,k,replace=False)]=1;return m
def main():
 tr,va,ct,cv,st,sv=load_data();fold=np.arange(len(tr))%5;C=Cload();rows=[];rank={}
 for f in range(5):
  models=[load_model(s,f,ct.shape[-1]) for s in [1,17,42]]
  for idx in np.where(fold==f)[0]:
   c=ct[idx:idx+1].float().to(DEV);slots=st[idx:idx+1].float().to(DEV);pid=tr.iloc[idx].case_id;y=int(tr.iloc[idx].label)
   full=[]
   with torch.no_grad():
    m=torch.ones(1,1,64,dtype=torch.bool,device=DEV)
    for n in models:full.append(float(torch.sigmoid(n(c,slots,m)['fused_logit'][0,0])))
   full=cal(full,64,C)
   for k in KS:
    for h in HS:
     active=mask(idx,k,h);cand=np.where(~active)[0];mm=np.repeat(active[None],len(cand)+1,0);mm[np.arange(1,len(cand)+1),cand]=1;m=torch.from_numpy(mm).to(DEV)[None];prob=[]
     with torch.no_grad():
      for n in models:prob.append(torch.sigmoid(n(c,slots,m)['fused_logit'][0]).cpu().numpy())
     prob=np.stack(prob);base=cal(prob[:,0],k,C);after=np.stack([cal(prob[s,1:],k+1,C) for s in range(3)]);ub=np.std(base);ua=np.std(after,0);du=ub-ua
     dl=np.stack([-np.array([bce(y,x) for x in after[s]])+bce(y,base[s])+.5*du for s in range(3)]);df=np.stack([-np.abs(after[s]-full[s])+abs(base[s]-full[s])+.5*du for s in range(3)])
     for version,D in [('label_based',dl),('label_free',df)]:
      mean=D.mean(0);signal=mean.std();noise=D.std(0).mean();tops=np.argmax(D,1);rows.append(dict(patient_id=pid,label=y,fold=f,k=k,history_seed=h,version=version,signal=signal,noise=noise,snr=signal/(noise+EPS),p90_p10=np.quantile(mean,.9)-np.quantile(mean,.1),oracle_best=mean.max(),random_mean=mean.mean(),best_minus_random=mean.max()-mean.mean(),three_seed_best_consistency=len(set(tops.tolist()))==1));v=np.full(64,np.nan);v[cand]=mean;rank[(pid,k,version,h)]=v
 d=pd.DataFrame(rows);d.to_csv(O/'metrics/e2m_corrected_action_snr_multiseed.csv',index=False);g=d.groupby(['patient_id','k','version'],as_index=False).best_minus_random.mean();rng=np.random.default_rng(20260831);boot=[]
 for (k,v),q in g.groupby(['k','version']):
  x=q.best_minus_random.values;b=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(2000)]);z=d[(d.k==k)&(d.version==v)];boot.append(dict(k=k,version=v,mean=x.mean(),ci95_low=np.quantile(b,.025),ci95_high=np.quantile(b,.975),median_snr=z.snr.median(),best_consistency=z.three_seed_best_consistency.mean()))
 pd.DataFrame(boot).to_csv(O/'metrics/e2m_best_vs_random_multiseed.csv',index=False);hh=[]
 for pid in tr.case_id:
  for k in KS:
   for v in ['label_based','label_free']:
    a=[rank[(pid,k,v,h)] for h in HS];sp=[];jac=[];tops=[int(np.nanargmax(x)) for x in a]
    for x,y in itertools.combinations(a,2):
     q=np.isfinite(x)&np.isfinite(y);sp.append(spearmanr(x[q],y[q]).statistic);A=set(np.argsort(np.nan_to_num(x,nan=-1e9))[-5:]);B=set(np.argsort(np.nan_to_num(y,nan=-1e9))[-5:]);jac.append(len(A&B)/len(A|B))
    hh.append(dict(patient_id=pid,k=k,version=v,mean_spearman=np.nanmean(sp),top1_change_rate=1-max(pd.Series(tops).value_counts())/5,top5_jaccard=np.mean(jac)))
 pd.DataFrame(hh).to_csv(O/'metrics/e2m_history_dependence_multiseed.csv',index=False);print('done')
if __name__=='__main__':main()
