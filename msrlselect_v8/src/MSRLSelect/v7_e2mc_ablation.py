#!/usr/bin/env python3
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from v7_e2mc_evaluate import O,load_model
from v7_e2m_train import load_data,mk,KS,MASK_SEEDS,DEV
def met(y,p):a=roc_auc_score(y,p);b=average_precision_score(y,p);return a,b,.5*(a+b),brier_score_loss(y,p)
def loss(y,p):p=np.clip(p,1e-6,1-1e-6);return -(y*np.log(p)+(1-y)*np.log(1-p))
def main():
 tr,va,ct,cv,st,sv=load_data();rows=[]
 for seed in [1,17,42]:
  m=load_model(seed,-1,ct.shape[-1])
  with torch.no_grad():
   for k in KS:
    for hs in MASK_SEEDS:
     mask=mk(len(va),[k],hs);full=torch.sigmoid(m(cv.float().to(DEV),sv.float().to(DEV),mask)['fused_logit'][:,0]).cpu().numpy()
     for e in range(4):
      ab=torch.sigmoid(m(cv.float().to(DEV),sv.float().to(DEV),mask,disabled_expert=e)['fused_logit'][:,0]).cpu().numpy()
      for i,r in enumerate(va.itertuples()):rows.append(dict(patient_id=r.case_id,label=int(r.label),seed=seed,k=k,mask_seed=hs,expert=e,full_probability=float(full[i]),ablated_probability=float(ab[i]),full_loss=float(loss(r.label,full[i])),ablated_loss=float(loss(r.label,ab[i]))))
 d=pd.DataFrame(rows);d[d.expert==3].to_csv(O/'metrics/e2m_expert3_patient_effect.csv',index=False);out=[]
 for scope,q0 in list(d.groupby('seed'))+[('ensemble',d)]:
  q=q0.groupby(['patient_id','label','k','expert'],as_index=False).agg(full_probability=('full_probability','mean'),ablated_probability=('ablated_probability','mean'))
  q['full_loss']=loss(q.label,q.full_probability);q['ablated_loss']=loss(q.label,q.ablated_probability);q['paired_loss_delta']=q.ablated_loss-q.full_loss
  for (k,e),z in q.groupby(['k','expert']):
   fa,fp,fs,fb=met(z.label,z.full_probability);aa,ap,ass,ab=met(z.label,z.ablated_probability);out.append(dict(scope=scope,k=k,expert=e,full_auroc=fa,ablated_auroc=aa,auroc_delta=aa-fa,full_auprc=fp,ablated_auprc=ap,auprc_delta=ap-fp,full_s_val=fs,ablated_s_val=ass,s_val_delta=ass-fs,full_brier=fb,ablated_brier=ab,brier_delta=ab-fb,paired_loss_delta_mean=z.paired_loss_delta.mean(),benefit_patient_fraction=(z.paired_loss_delta>0).mean(),harm_patient_fraction=(z.paired_loss_delta<0).mean(),label0_loss_delta=z[z.label==0].paired_loss_delta.mean(),label1_loss_delta=z[z.label==1].paired_loss_delta.mean()))
 pd.DataFrame(out).to_csv(O/'metrics/e2m_expert_ablation_multiseed.csv',index=False)
 # Paired patient bootstrap for ensemble expert 3 at each k.
 q=d[d.expert==3].groupby(['patient_id','label','k'],as_index=False).agg(full_probability=('full_probability','mean'),ablated_probability=('ablated_probability','mean'));q['delta']=loss(q.label,q.ablated_probability)-loss(q.label,q.full_probability);rng=np.random.default_rng(20260831);boot=[]
 for k,z in q.groupby('k'):
  x=z.delta.values;b=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(2000)]);boot.append(dict(k=k,mean_loss_delta=x.mean(),ci95_low=np.quantile(b,.025),ci95_high=np.quantile(b,.975)))
 pd.DataFrame(boot).to_csv(O/'metrics/e2m_expert3_paired_bootstrap.csv',index=False);print('done')
if __name__=='__main__':main()
