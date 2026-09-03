#!/usr/bin/env python3
import argparse,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
torch.set_num_threads(4)
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.decomposition import PCA
from v7_train_evidence import EvidenceNet,coarse,ROOT,OUT as EOUT,DEV
from v7_e2m_model import E2MModel

E2M=EOUT/'e2m_trainable_moe';KS=[0,1,2,4,8,16,32,48,64];MASK_SEEDS=[7101,7102,7103,7104,7105]
for x in ['metrics','predictions','checkpoints','logs','docs','configs']:(E2M/x).mkdir(parents=True,exist_ok=True)
def metric(y,p):
 a=roc_auc_score(y,p);b=average_precision_score(y,p);return a,b,.5*(a+b),brier_score_loss(y,p)
def load_data():
 tr=pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv');va=pd.read_csv(ROOT/'prepared/sr386_full/surgen_validate_10.csv');comp=np.load(EOUT/'assets/train_pca128_components.npy');pm=np.load(EOUT/'assets/train_pca128_mean.npy');pev=np.load(EOUT/'assets/train_pca128_explained_variance.npy');pc=PCA(128);pc.components_=comp;pc.mean_=pm;pc.explained_variance_=pev;pc.explained_variance_ratio_=pev/pev.sum();pc.singular_values_=np.sqrt(np.maximum(pev*(16320-1),0));pc.n_features_in_=1024;pc.n_samples_=16320;pc.n_samples_seen_=16320;pc.noise_variance_=0
 ct=coarse('train',tr,pc);cv=coarse('valid',va,pc);st=torch.load(EOUT/'assets/train_moe_slots.pt',weights_only=True);sv=torch.load(EOUT/'assets/validation_moe_slots.pt',weights_only=True);return tr,va,ct,cv,st,sv
def mk(n,ks,seed):
 g=torch.Generator(device=DEV).manual_seed(seed);m=torch.zeros(n,len(ks),64,dtype=torch.bool,device=DEV)
 for b in range(n):
  for j,k in enumerate(ks):
   if k:m[b,j,torch.randperm(64,generator=g,device=DEV)[:k]]=1
 return m
def base_model(path,cd):
 q=torch.load(path,map_location='cpu',weights_only=False);b=EvidenceNet(cd,1024).to(DEV);b.load_state_dict(q['model']);b.eval();return b
def losses(model,base,c,s,m,y,posw):
 out=model(c,s,m);flat=s.flatten(-2)
 with torch.no_grad():blogit=base(c,flat,m)[0]
 yy=y[:,None].expand_as(out['fused_logit']);pred=F.binary_cross_entropy_with_logits(out['fused_logit'],yy,pos_weight=posw);brier=((torch.sigmoid(out['fused_logit'])-yy)**2).mean();anchor=F.mse_loss(torch.sigmoid(out['fused_logit']),torch.sigmoid(blogit))
 w=out['weights'];imp=w.sum((0,1));imp=imp/imp.sum().clamp_min(1);balance=((imp-.25)**2).sum();eo=F.normalize(out['expert_outputs'],dim=-1);div=[]
 for a in range(4):
  for b in range(a+1,4):div.append((eo[:,:,a]*eo[:,:,b]).sum(-1).pow(2).mean())
 diversity=torch.stack(div).mean();valid=out['present'];z=torch.logsumexp(out['router_logits'].masked_fill(~valid,-1e9),-1);zloss=z[valid.any(-1)].pow(2).mean();total=pred+.1*brier+.1*anchor+.01*balance+.005*diversity+.001*zloss
 return total,dict(prediction=pred,brier=brier,anchor=anchor,load_balance=balance,expert_diversity=diversity,router_z=zloss,total=total),out
def evaluate(model,c,s,meta,seed,patient_rows=False):
 model.eval();rows=[]
 with torch.no_grad():
  for k in KS:
   for ms in MASK_SEEDS:
    m=mk(len(meta),[k],ms);o=model(c.float().to(DEV),s.float().to(DEV),m);p=torch.sigmoid(o['fused_logit'][:,0]).cpu().numpy();au,ap,sv,br=metric(meta.label,p)
    rows.append(dict(seed=seed,k=k,mask_seed=ms,auroc=au,auprc=ap,s_val=sv,brier=br,probability_std=p.std(),beta_mean=float(o['beta'].mean()),router_entropy=float((-(o['weights'].clamp_min(1e-8)*o['weights'].clamp_min(1e-8).log()).sum(-1))[o['present'].any(-1)].mean())))
    if patient_rows:
     for i,r in enumerate(meta.itertuples()):yield_rows.append(dict(patient_id=r.case_id,label=int(r.label),seed=seed,k=k,mask_seed=ms,probability=float(p[i])))
 return pd.DataFrame(rows)
def train(seed=17,fold=-1):
 global yield_rows;yield_rows=[];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);tr,va,ct,cv,st,sv=load_data();allidx=np.arange(len(tr))
 if fold>=0:ti=allidx[allidx%5!=fold];vi=allidx[allidx%5==fold];train_meta=tr.iloc[ti].reset_index(drop=True);val_meta=tr.iloc[vi].reset_index(drop=True);tc,vc,ts,vs=ct[ti],ct[vi],st[ti],st[vi];bp=EOUT/f'checkpoints/E2_oof_fold{fold}_seed{seed}_best.pt';tag=f'fold{fold}_seed{seed}'
 else:train_meta,val_meta=tr,va;tc,vc,ts,vs=ct,cv,st,sv;bp=EOUT/f'checkpoints/E2_bottleneck_moe4_seed{seed}_best.pt';tag=f'seed{seed}'
 base=base_model(bp,tc.shape[-1]);model=E2MModel(base,tc.shape[-1]).to(DEV);posw=torch.tensor([(train_meta.label==0).sum()/(train_meta.label==1).sum()],device=DEV);history=[];best=-1;best_state=None;best_epoch=0;global_ep=0
 for stage,maxep in [('A',15),('B',30)]:
  model.set_stage(stage);moe_names=('slot_projection','experts','router','beta_gate','region_norm','slot_identity');slow=[];fast=[]
  for n,p in model.named_parameters():
   if p.requires_grad:(fast if n.startswith(moe_names) else slow).append(p)
  groups=[{'params':fast,'lr':1e-4}]+([{'params':slow,'lr':1e-5}] if slow else []);opt=torch.optim.AdamW(groups,weight_decay=5e-4);patience=0
  for ep in range(1,maxep+1):
   global_ep+=1;model.train();order=np.random.default_rng(seed+global_ep).permutation(len(train_meta));acc=[]
   for a in range(0,len(order),24):
    ix=order[a:a+24];low=random.choice([1,2,4,8]);mid=random.choice([16,32,48]);m=mk(len(ix),[0,low,mid,64],seed+global_ep*1000+a);c=tc[ix].float().to(DEV);ss=ts[ix].float().to(DEV);y=torch.tensor(train_meta.label.values[ix],device=DEV).float();opt.zero_grad();L,parts,_=losses(model,base,c,ss,m,y,posw);L.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();acc.append({k:float(v.detach()) for k,v in parts.items()})
   ev=evaluate(model,vc,vs,val_meta,seed);q=ev[ev.k==64].iloc[0];rec=dict(stage=stage,epoch_stage=ep,epoch=global_ep,val_auroc=q.auroc,val_auprc=q.auprc,val_s_val=q.s_val,val_brier=q.brier,**{f'train_{k}':np.mean([x[k] for x in acc]) for k in acc[0]});history.append(rec);print(json.dumps(rec),flush=True)
   if q.s_val>best+1e-8:best=q.s_val;best_epoch=global_ep;best_state={k:v.detach().cpu() for k,v in model.state_dict().items()};patience=0;torch.save({'model':best_state,'seed':seed,'fold':fold,'best_epoch':best_epoch,'architecture':'E2M_morphology_anchored_top2_moe'},E2M/f'checkpoints/{tag}_best.pt')
   else:patience+=1
   if patience>=5:break
 torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'seed':seed,'fold':fold,'stop_epoch':global_ep},E2M/f'checkpoints/{tag}_stopped.pt');model.load_state_dict(best_state);pd.DataFrame(history).to_csv(E2M/f'metrics/{tag}_training_curve.csv',index=False);res=evaluate(model,vc,vs,val_meta,seed,patient_rows=True);res['fold']=fold;res.to_csv(E2M/f'metrics/{tag}_results.csv',index=False);pd.DataFrame(yield_rows).to_csv(E2M/f'predictions/{tag}_predictions.csv',index=False);return model,res
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=17);p.add_argument('--fold',type=int,default=-1);p.add_argument('--out-dir',default=str(E2M));a=p.parse_args();E2M=Path(a.out_dir)
 for x in ['metrics','predictions','checkpoints','logs','docs','configs']:(E2M/x).mkdir(parents=True,exist_ok=True)
 train(a.seed,a.fold)
