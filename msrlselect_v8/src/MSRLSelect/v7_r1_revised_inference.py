#!/usr/bin/env python3
from pathlib import Path
import re, math
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from v7_train_evidence import EvidenceNet, coarse, OUT as EOUT, ROOT, BASE, CACHE, DEV
from sklearn.decomposition import PCA

R1=ROOT/'outputs/thesis_rl_v7/r1_online_ppo'; (R1/'metrics').mkdir(parents=True,exist_ok=True);(R1/'predictions').mkdir(exist_ok=True);(R1/'docs').mkdir(exist_ok=True);(R1/'logs').mkdir(exist_ok=True)
KS=[0,1,2,4,8,16,32,48,64]; MASK_SEEDS=[7101,7102,7103,7104,7105]
def mk(n,k,seed,device=DEV):
 g=torch.Generator(device=device).manual_seed(seed);m=torch.zeros(n,1,64,dtype=torch.bool,device=device)
 if k:
  for i in range(n):m[i,0,torch.randperm(64,generator=g,device=device)[:k]]=1
 return m
def ece(y,p):
 ans=0.; bins=np.linspace(0,1,11)
 for i in range(10):
  q=(p>=bins[i])&(p<(bins[i+1] if i<9 else 1.00001))
  if q.any():ans+=q.mean()*abs(p[q].mean()-y[q].mean())
 return ans
def met(y,p):return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),s_val=.5*(roc_auc_score(y,p)+average_precision_score(y,p)),brier=brier_score_loss(y,p),nll=log_loss(y,p,labels=[0,1]),ece=ece(y,p))
def load_data():
 tr=pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv');va=pd.read_csv(ROOT/'prepared/sr386_full/surgen_validate_10.csv');comp=np.load(EOUT/'assets/train_pca128_components.npy');pm=np.load(EOUT/'assets/train_pca128_mean.npy');pev=np.load(EOUT/'assets/train_pca128_explained_variance.npy');pc=PCA(128);pc.components_=comp;pc.mean_=pm;pc.explained_variance_=pev;pc.explained_variance_ratio_=pev/pev.sum();pc.singular_values_=np.sqrt(np.maximum(pev*(16320-1),0));pc.n_features_in_=1024;pc.n_samples_=16320;pc.n_samples_seen_=16320;pc.noise_variance_=0
 ct=coarse('train',tr,pc);cv=coarse('valid',va,pc);ft=torch.load(EOUT/'assets/train_moe_slots.pt',weights_only=True).reshape(len(tr),64,-1);fv=torch.load(EOUT/'assets/validation_moe_slots.pt',weights_only=True).reshape(len(va),64,-1);return tr,va,ct,cv,ft,fv
def load_model(path,cd,fd):
 x=torch.load(path,map_location='cpu',weights_only=False);n=EvidenceNet(cd,fd).to(DEV);n.load_state_dict(x['model']);n.eval();return n
def infer():
 tr,va,ct,cv,ft,fv=load_data();fold=np.arange(len(tr))%5;rows=[]
 with torch.no_grad():
  for f in range(5):
   ii=np.where(fold==f)[0]
   for seed in [1,17,42]:
    n=load_model(EOUT/f'checkpoints/E2_oof_fold{f}_seed{seed}_best.pt',ct.shape[-1],ft.shape[-1])
    for k in KS:
     seeds=MASK_SEEDS if k not in [0,64] else [MASK_SEEDS[0]]
     for ms in seeds:
      z=n(ct[ii].float().to(DEV),ft[ii].float().to(DEV),mk(len(ii),k,ms));p=torch.sigmoid(z[0][:,0]).cpu().numpy()
      for j,idx in enumerate(ii):rows.append(dict(patient_id=tr.iloc[idx].case_id,label=int(tr.iloc[idx].label),fold=f,seed=seed,k=k,mask_seed=ms,probability=float(p[j])))
 pd.DataFrame(rows).to_csv(R1/'predictions/e2_train_oof_all_k.csv',index=False)
 rows=[]
 with torch.no_grad():
  for seed in [1,17,42]:
   n=load_model(EOUT/f'checkpoints/E2_bottleneck_moe4_seed{seed}_best.pt',cv.shape[-1],fv.shape[-1])
   for k in KS:
    seeds=MASK_SEEDS if k not in [0,64] else [MASK_SEEDS[0]]
    for ms in seeds:
     z=n(cv.float().to(DEV),fv.float().to(DEV),mk(len(va),k,ms));p=torch.sigmoid(z[0][:,0]).cpu().numpy()
     for j in range(len(va)):rows.append(dict(patient_id=va.iloc[j].case_id,label=int(va.iloc[j].label),seed=seed,k=k,mask_seed=ms,probability=float(p[j])))
 pd.DataFrame(rows).to_csv(R1/'predictions/e2_validation_all_k.csv',index=False)
 return tr,va
def summarize(path,out):
 d=pd.read_csv(path); pat=d.groupby(['patient_id','label','k'],as_index=False).probability.mean();stats=[]
 for k,q in pat.groupby('k'):
  x=met(q.label.values,q.probability.values); raw=d[d.k==k]; seed=raw.groupby(['patient_id','mask_seed','seed']).probability.mean().unstack().var(1).mean();mask=raw.groupby(['patient_id','seed','mask_seed']).probability.mean().groupby(['patient_id','seed']).var().mean()
  stats.append(dict(k=k,**x,seed_prediction_variance=seed,mask_history_variance=mask,n_patients=len(q)))
 pd.DataFrame(stats).to_csv(out,index=False);return pat
def feats(p,k,conditional):
 p=np.asarray(p);z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)));rho=np.asarray(k)/64.
 return np.c_[z,z*rho,rho] if conditional else z[:,None]
def calibration():
 od=pd.read_csv(R1/'predictions/e2_train_oof_all_k.csv');vd=pd.read_csv(R1/'predictions/e2_validation_all_k.csv');op=od.groupby(['patient_id','label','fold','k'],as_index=False).probability.mean();vp=vd.groupby(['patient_id','label','k'],as_index=False).probability.mean(); op['calibrated_probability']=np.nan;vp['calibrated_probability']=np.nan;pars=[]
 for mode in ['k64','partial']:
  sel=op.k.eq(64) if mode=='k64' else op.k.lt(64);sub=op[sel]
  for f in range(5):
   a=sub.fold.ne(f);b=sub.fold.eq(f);lr=LogisticRegression(C=1e6,max_iter=2000).fit(feats(sub.loc[a,'probability'],sub.loc[a,'k'],mode=='partial'),sub.loc[a,'label']);op.loc[sub.index[b],'calibrated_probability']=lr.predict_proba(feats(sub.loc[b,'probability'],sub.loc[b,'k'],mode=='partial'))[:,1]
  lr=LogisticRegression(C=1e6,max_iter=2000).fit(feats(sub.probability,sub.k,mode=='partial'),sub.label);vv=vp.k.eq(64) if mode=='k64' else vp.k.lt(64);vp.loc[vv,'calibrated_probability']=lr.predict_proba(feats(vp.loc[vv,'probability'],vp.loc[vv,'k'],mode=='partial'))[:,1];pars.append(dict(mode=mode,intercept=float(lr.intercept_[0]),coefficients=';'.join(map(str,lr.coef_[0]))))
 op.to_csv(R1/'predictions/e2_train_oof_all_k_calibrated.csv',index=False);vp.to_csv(R1/'predictions/e2_validation_all_k_calibrated.csv',index=False)
 out=[]
 for split,d in [('oof',op),('validation',vp)]:
  for k,q in d.groupby('k'):
   for kind,col in [('raw','probability'),('calibrated','calibrated_probability')]:out.append(dict(split=split,k=k,kind=kind,**met(q.label.values,q[col].values)))
 pd.DataFrame(out).to_csv(R1/'metrics/e2_calibration_results.csv',index=False);pd.DataFrame(pars).to_csv(R1/'metrics/e2_calibration_parameters.csv',index=False)
if __name__=='__main__':
 infer();summarize(R1/'predictions/e2_train_oof_all_k.csv',R1/'metrics/e2_variable_evidence_oof.csv');calibration();print('done')
