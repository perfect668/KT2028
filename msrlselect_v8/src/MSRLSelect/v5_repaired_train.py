#!/usr/bin/env python3
import argparse,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from v5_repaired_model import RepairedDenseModel

CFG={'R0_K64_mean_no_spatial':('k64','mean_pool',False,True,16,1),'R1_K64_gated_no_spatial':('k64','gated_attention_pool',False,True,16,1),'R2_K64_gated_spatial':('k64','gated_attention_pool',True,True,16,1),'R3_K64_gated_spatial_no_relation':('k64','gated_attention_pool',True,False,16,1),'R4_K128_gated_spatial':('k128','gated_attention_pool',True,True,16,1),'R5_K64_local_transformer_spatial':('k64','local_transformer_pool',True,True,4,4),'R6_P1024_direct_spatial':('p1024','gated_attention_pool',True,True,2,8)}
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def load(root,split,r,kind):
 with np.load(r.features_filepath) as z:f=z['img_features'].astype(np.float32);c=z['coords'].astype(np.float32)
 cache=root/'outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache'
 if kind=='p1024':
  ix=np.load(cache/'p1024_stratified'/split/f'{r.case_id}.npz')['patch_indices']; x=f[ix][None,None]; pm=np.ones((1,1,len(ix)),bool); cent=((c[ix]-c.min(0))/np.maximum(c.max(0)-c.min(0),1)).mean(0)[None,None]; size=np.ones((1,1),np.float32)
 else:
  z=np.load(cache/f'regions_{kind}'/split/f'{r.case_id}.npz');q=z['region_patch_indices'];ptr=z['region_ptr'];parts=[f[q[ptr[i]:ptr[i+1]]] for i in range(int(kind[1:]))];p=max(map(len,parts));x=np.zeros((1,len(parts),p,1024),np.float32);pm=np.zeros((1,len(parts),p),bool)
  for i,a in enumerate(parts):x[0,i,:len(a)]=a;pm[0,i,:len(a)]=1
  cent=z['region_centroid'][None].astype(np.float32);size=(z['region_size']/len(f))[None].astype(np.float32)
 return x,pm,cent,size
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',required=True);p.add_argument('--epochs',type=int,default=30);a=p.parse_args();seed(17);kind,pool,sp,rel,bs,acc=CFG[a.config];data=a.root/'prepared/sr386_full';d=a.out/a.config/'seed17';d.mkdir(parents=True,exist_ok=True);dev=torch.device('cuda');train=pd.read_csv(data/'surgen_train_10.csv').itertuples(index=False);valid=pd.read_csv(data/'surgen_validate_10.csv').itertuples(index=False);train=list(train);valid=list(valid);m=RepairedDenseModel(pool=pool,spatial=sp,no_relation=not rel).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);best=-1;hist=[]
 for ep in range(1,a.epochs+1):
  m.train();losses=[];opt.zero_grad(set_to_none=True)
  for st in range(0,len(train),bs):
   batch=train[st:st+bs]; arr=[load(a.root,'train',r,kind) for r in batch]; pmax=max(x[0].shape[2] for x in arr); rmax=max(x[0].shape[1] for x in arr); X=np.zeros((len(arr),rmax,pmax,1024),np.float32);PM=np.zeros((len(arr),rmax,pmax),bool);C=np.zeros((len(arr),rmax,2),np.float32);S=np.zeros((len(arr),rmax),np.float32);RM=np.zeros((len(arr),rmax),bool);Y=[]
   for i,(x,pm,c,s) in enumerate(arr):X[i,:x.shape[1],:x.shape[2]]=x;PM[i,:x.shape[1],:x.shape[2]]=pm;C[i,:x.shape[1]]=c;S[i,:x.shape[1]]=s;RM[i,:x.shape[1]]=1;Y.append(batch[i].label)
   X=torch.from_numpy(X).to(dev).bfloat16();PM=torch.from_numpy(PM).to(dev);C=torch.from_numpy(C).to(dev).bfloat16();S=torch.from_numpy(S).to(dev).bfloat16();RM=torch.from_numpy(RM).to(dev);Y=torch.tensor(Y,device=dev)
   with torch.autocast('cuda',dtype=torch.bfloat16):o=m(X,PM,None,C,S,RM);l=nn.functional.cross_entropy(o,Y)/acc
   l.backward();
   if ((st//bs)+1)%acc==0:nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
   losses.append(float(l.detach())*acc)
  m.eval();ys=[];ps=[]
  with torch.no_grad():
   for r in valid:
    x,pm,c,s=load(a.root,'valid',r,kind);X=torch.from_numpy(x).to(dev).bfloat16();PM=torch.from_numpy(pm).to(dev);C=torch.from_numpy(c).to(dev).bfloat16();S=torch.from_numpy(s).to(dev).bfloat16();RM=torch.ones(1,X.shape[1],dtype=torch.bool,device=dev)
    with torch.autocast('cuda',dtype=torch.bfloat16):o=m(X,PM,None,C,S,RM)
    ys.append(int(r.label));ps.append(float(o.softmax(1)[0,1]))
  if not np.isfinite(ps).all(): raise FloatingPointError('Validation NaN')
  auc=roc_auc_score(ys,ps);ap=average_precision_score(ys,ps);sv=.5*(auc+ap);hist.append({'epoch':ep,'train_loss':np.mean(losses),'valid_auc':auc,'valid_auprc':ap,'s_val':sv,'valid_brier':brier_score_loss(ys,ps)})
  if sv>best:best=sv;torch.save({'model':m.state_dict(),'epoch':ep,'configuration':a.config,'seed':17,'s_val':sv},d/'seed17_best.pt');pd.DataFrame({'patient_id':[r.patient_id for r in valid],'label':ys,'death_risk':ps,'configuration':a.config,'seed':17,'checkpoint_epoch':ep}).to_csv(d/'validation_predictions.csv',index=False)
  pd.DataFrame(hist).to_csv(d/'epoch_metrics.csv',index=False)
 (d/'run_status.json').write_text(json.dumps({'status':'COMPLETE','configuration':a.config,'seed':17,'best_s_val':best,'test_evaluated':False},indent=2))
if __name__=='__main__':main()
