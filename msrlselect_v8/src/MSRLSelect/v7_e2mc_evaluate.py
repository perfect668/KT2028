#!/usr/bin/env python3
import itertools,math,re
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from sklearn.linear_model import LogisticRegression
from v7_e2m_train import ROOT,EOUT,DEV,load_data,base_model,KS,MASK_SEEDS
from v7_e2m_model import E2MModel
O=EOUT/'e2m_multiseed_confirmation';torch.set_num_threads(4)
def ece(y,p):
 r=0.;e=np.linspace(0,1,11)
 for i in range(10):
  q=(p>=e[i])&(p<(e[i+1] if i<9 else 1.00001))
  if q.any():r+=q.mean()*abs(p[q].mean()-y[q].mean())
 return r
def met(y,p):a=roc_auc_score(y,p);b=average_precision_score(y,p);return dict(auroc=a,auprc=b,s_val=.5*(a+b),brier=brier_score_loss(y,p),nll=log_loss(y,p,labels=[0,1]),ece=ece(y,p),probability_variance=np.var(p))
def load_model(seed,fold,cd):
 bp=EOUT/(f'checkpoints/E2_oof_fold{fold}_seed{seed}_best.pt' if fold>=0 else f'checkpoints/E2_bottleneck_moe4_seed{seed}_best.pt');base=base_model(bp,cd);m=E2MModel(base,cd).to(DEV);tag=f'fold{fold}_seed{seed}' if fold>=0 else f'seed{seed}';m.load_state_dict(torch.load(O/f'checkpoints/{tag}_best.pt',map_location='cpu',weights_only=False)['model']);m.eval();return m
def concat_predictions():
 o=[]
 for f in range(5):
  for s in [1,17,42]:
   d=pd.read_csv(O/f'predictions/fold{f}_seed{s}_predictions.csv');d['fold']=f;o.append(d)
 od=pd.concat(o,ignore_index=True);od.to_csv(O/'predictions/e2m_train_oof_all_k.csv',index=False)
 vd=pd.concat([pd.read_csv(O/f'predictions/seed{s}_predictions.csv') for s in [1,17,42]],ignore_index=True);vd.to_csv(O/'predictions/e2m_validation_all_k.csv',index=False);return od,vd
def summarize(od,vd):
 out=[]
 for split,d in [('oof',od),('validation',vd)]:
  for scope,q0 in list(d.groupby('seed'))+[('ensemble',d)]:
   q=q0.groupby(['patient_id','label','k'],as_index=False).probability.mean()
   for k,z in q.groupby('k'):
    raw=q0[q0.k==k];seedvar=raw.groupby(['patient_id','mask_seed','seed']).probability.mean().unstack().var(1).mean() if scope=='ensemble' else np.nan;maskvar=raw.groupby(['patient_id','seed','mask_seed']).probability.mean().groupby(['patient_id','seed']).var().mean();out.append(dict(split=split,scope=scope,k=k,**met(z.label,z.probability),seed_variance=seedvar,mask_history_variance=maskvar))
 allm=pd.DataFrame(out);allm[allm['split']=='validation'].to_csv(O/'metrics/e2m_multiseed_validation.csv',index=False);allm[allm['split']=='oof'].to_csv(O/'metrics/e2m_strict_oof_all_k.csv',index=False)
def X(p,k,cond):
 p=np.asarray(p);z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)));r=np.asarray(k)/64.;return np.c_[z,z*r,r] if cond else z[:,None]
def calibrate(od,vd):
 op=od.groupby(['patient_id','label','fold','k'],as_index=False).probability.mean();vp=vd.groupby(['patient_id','label','k'],as_index=False).probability.mean();op['calibrated_probability']=np.nan;vp['calibrated_probability']=np.nan;pars=[]
 for mode in ['k64','partial']:
  sel=op.k.eq(64) if mode=='k64' else op.k.lt(64);sub=op[sel]
  for f in range(5):
   a=sub.fold.ne(f);b=sub.fold.eq(f);lr=LogisticRegression(C=1e6,max_iter=2000).fit(X(sub.loc[a,'probability'],sub.loc[a,'k'],mode=='partial'),sub.loc[a,'label']);op.loc[sub.index[b],'calibrated_probability']=lr.predict_proba(X(sub.loc[b,'probability'],sub.loc[b,'k'],mode=='partial'))[:,1]
  lr=LogisticRegression(C=1e6,max_iter=2000).fit(X(sub.probability,sub.k,mode=='partial'),sub.label);q=vp.k.eq(64) if mode=='k64' else vp.k.lt(64);vp.loc[q,'calibrated_probability']=lr.predict_proba(X(vp.loc[q,'probability'],vp.loc[q,'k'],mode=='partial'))[:,1];pars.append(dict(mode=mode,intercept=lr.intercept_[0],coefficients=';'.join(map(str,lr.coef_[0]))))
 op.to_csv(O/'predictions/e2m_train_oof_all_k_calibrated.csv',index=False);vp.to_csv(O/'predictions/e2m_validation_all_k_calibrated.csv',index=False);out=[]
 for split,d in [('oof',op),('validation',vp)]:
  for k,q in d.groupby('k'):
   for typ,col in [('raw','probability'),('calibrated','calibrated_probability')]:out.append(dict(split=split,k=k,kind=typ,**met(q.label,q[col])))
 pd.DataFrame(out).to_csv(O/'metrics/e2m_calibration_all_k.csv',index=False);pd.DataFrame(pars).to_csv(O/'metrics/e2m_calibration_parameters.csv',index=False)
def route_stats_and_tensors():
 tr,va,ct,cv,st,sv=load_data();fold=np.arange(len(tr))%5;records=[];conditions=[];store={k:[] for k in ['patient_index','seed','router_probability','top1','top2','beta','expert_outputs']}
 for split in ['oof','validation']:
  for seed in [1,17,42]:
   loops=range(5) if split=='oof' else [-1]
   for f in loops:
    idx=np.where(fold==f)[0] if f>=0 else np.arange(len(va));c=(ct[idx] if f>=0 else cv).float().to(DEV);ss=(st[idx] if f>=0 else sv).float().to(DEV);m=load_model(seed,f,ct.shape[-1])
    with torch.no_grad():
     o=m(c,ss,torch.ones(len(idx),1,64,dtype=torch.bool,device=DEV));w=o['weights'];present=o['present'];valid=present.any(-1);top1=w.argmax(-1);top2=w.topk(2,-1).indices;agg=w[valid].mean(0);eff=float(torch.exp(-(agg*agg.clamp_min(1e-8).log()).sum()));ent=float((-(w.clamp_min(1e-8)*w.clamp_min(1e-8).log()).sum(-1))[valid].mean());eo=o['expert_outputs'];cos=[]
     for a,b in itertools.combinations(range(4),2):
      q=present[...,a]&present[...,b];cos.append(float(torch.nn.functional.cosine_similarity(eo[...,a,:],eo[...,b,:],dim=-1)[q].mean()))
     for e in range(4):records.append(dict(split=split,fold=f,seed=seed,expert=e,top1_occupancy=float(((top1==e)&valid).sum()/valid.sum()),top2_occupancy=float(((w[...,e]>0)&valid).sum()/(2*valid.sum())),patient_coverage=float((((w[...,e]>0)&valid).any(1)).float().mean()),routing_entropy=ent,effective_experts=eff,max_expert_output_cosine=max(cos),empty_slot_fraction=float((~present).float().mean())))
     sg=m(c,ss,torch.ones(len(idx),1,64,dtype=torch.bool,device=DEV),shuffle_global=True);sr=m(c,ss,torch.ones(len(idx),1,64,dtype=torch.bool,device=DEV),shuffle_region=True);conditions.append(dict(split=split,fold=f,seed=seed,global_shuffle_logit_mae=float((o['router_logits']-sg['router_logits']).abs()[present].mean()),global_shuffle_top1_change=float((top1!=sg['weights'].argmax(-1))[valid].float().mean()),region_shuffle_logit_mae=float((o['router_logits']-sr['router_logits']).abs()[present].mean()),region_shuffle_top1_change=float((top1!=sr['weights'].argmax(-1))[valid].float().mean())))
     if split=='oof':
      store['patient_index'].append(torch.tensor(idx,dtype=torch.int16));store['seed'].append(torch.full((len(idx),),seed,dtype=torch.int16));store['router_probability'].append(w.cpu().half());store['top1'].append(top1.cpu().to(torch.int8));store['top2'].append(top2.cpu().to(torch.int8));store['beta'].append(o['beta'].cpu().half());store['expert_outputs'].append(eo.cpu().half())
 pd.DataFrame(records).to_csv(O/'metrics/e2m_expert_statistics_multiseed.csv',index=False);pd.DataFrame(conditions).to_csv(O/'metrics/e2m_routing_condition_audit.csv',index=False);torch.save({k:torch.cat(v) for k,v in store.items()},O/'predictions/e2m_oof_router_expert_outputs.pt')
if __name__=='__main__':
 od,vd=concat_predictions();summarize(od,vd);calibrate(od,vd);route_stats_and_tensors();print('done')
