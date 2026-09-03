#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,f1_score,confusion_matrix,brier_score_loss
from tqdm import tqdm
from models.v5_dense_region_transformer import DenseRegionTransformer

SEED=17
CFGS={
'D0_K10':('k10','gated_attention_pool',True,True,16,1),
'D1_K64':('k64','gated_attention_pool',True,True,16,1),
'D2_K128':('k128','gated_attention_pool',True,True,16,1),
'D3_P1024':('p1024','gated_attention_pool',True,True,2,8),
'D4_K128_no_spatial':('k128','gated_attention_pool',False,True,16,1),
'D5_K128_no_relation':('k128','gated_attention_pool',False,False,16,1),
'D6_K128_local_transformer':('k128','local_transformer_pool',True,True,4,4),}
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def metrics(y,p):
 pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); return {'auc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'accuracy':float(accuracy_score(y,pred)),'f1':float(f1_score(y,pred,zero_division=0)),'sensitivity':float(tp/(tp+fn)) if tp+fn else 0.,'specificity':float(tn/(tn+fp)) if tn+fp else 0.,'brier':float(brier_score_loss(y,p)),'loss':float(np.mean(-(y*np.log(p+1e-7)+(1-y)*np.log(1-p+1e-7))))}
class Cases:
 def __init__(self,manifest,root,kind,split):
  self.df=pd.read_csv(manifest); self.root=Path(root); self.kind=kind; self.split=split; self.items=[]
  for r in self.df.itertuples(index=False):
   with np.load(self._path(r),allow_pickle=False) as z:
    if kind=='p1024':
     with np.load(r.features_filepath) as fz: f=fz['img_features'].astype(np.float32); c=fz['coords'].astype(np.float32)
     ix=z['patch_indices'].astype(np.int64); feat=f[ix,None,:]; cen=((c[ix]-c.min(0))/(np.maximum(c.max(0)-c.min(0),1))).mean(0)[None]
    else:
     feat=z['region_feature_mean'].astype(np.float32)[:,None,:]; cen=z['region_centroid'].astype(np.float32)
    self.items.append((feat,cen,int(r.label),str(r.patient_id),str(r.case_id),int(len(f) if kind=='p1024' else z['raw_patch_count'][0])))
 def _path(self,r):
  if self.kind=='k10': return self.root/'cache/k10'/self.split/f'{r.case_id}.npz'
  return self.root/f'cache/{"p1024_stratified" if self.kind=="p1024" else "regions_"+self.kind}'/self.split/f'{r.case_id}.npz'
 def __len__(self): return len(self.items)
 def __getitem__(self,i): return self.items[i]
def collate(items):
 b=len(items); r=max(x[0].shape[0] for x in items); p=max(x[0].shape[1] for x in items); f=np.zeros((b,r,p,1024),np.float32); pm=np.zeros((b,r,p),bool); rm=np.zeros((b,r),bool); c=np.zeros((b,r,2),np.float32); ys=[]; ids=[]; raws=[]
 for i,(x,cc,y,pi,ci,raw) in enumerate(items): f[i,:x.shape[0],:x.shape[1]]=x; pm[i,:x.shape[0],:x.shape[1]]=True; rm[i,:x.shape[0]]=True; c[i,:x.shape[0]]=cc; ys.append(y); ids.append(pi); raws.append(raw)
 return torch.from_numpy(f),torch.from_numpy(rm),torch.from_numpy(pm),torch.from_numpy(c),torch.tensor(ys),ids,raws
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--config',required=True); ap.add_argument('--epochs',type=int,default=150); a=ap.parse_args(); seed_all(SEED); torch.backends.cuda.matmul.allow_tf32=True; dev=torch.device('cuda'); root=a.root; out=a.out/a.config/'seed17'; out.mkdir(parents=True,exist_ok=True); kind,pool,spatial,relation,bs,acc=CFGS[a.config]
 droot=root/'outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure'; data=root/'prepared/sr386_full';
 if kind=='k10':
  # Build lightweight K10 cache from existing stage-0 summaries copied into a private stage1b cache.
  sdir=droot/'cache/k10'; sdir.mkdir(parents=True,exist_ok=True)
  for split,name in [('train','surgen_train_10.csv'),('valid','surgen_validate_10.csv')]: (sdir/split).mkdir(parents=True,exist_ok=True)
  # derive K10 from raw clusters directly
  from v5_stage1a_infrastructure import build_region_cache_case
  for split,name in [('train','surgen_train_10.csv'),('valid','surgen_validate_10.csv')]:
   for r in pd.read_csv(data/name).itertuples(index=False): build_region_cache_case(r.features_filepath,r.clusters_json_filepath,r.coordinates_filepath,sdir/split/f'{r.case_id}.npz',10)
 ds={s:Cases(data/('surgen_train_10.csv' if s=='train' else 'surgen_validate_10.csv'),droot,kind,s) for s in ('train','valid')}; loaders={s:torch.utils.data.DataLoader(ds[s],bs,shuffle=s=='train',num_workers=0,collate_fn=collate,pin_memory=True) for s in ds}
 model=DenseRegionTransformer(pool=pool).to(dev); model._no_relation=not relation; model.train(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); counts=np.bincount([x[2] for x in ds['train'].items],minlength=2); crit=nn.CrossEntropyLoss(weight=torch.tensor(len(ds['train'])/(2*counts),device=dev,dtype=torch.float32)); best=-1.; best_epoch=0; hist=[]; since=0
 for ep in range(1,a.epochs+1):
  model.train(); tl=[]; t0=time.time(); opt.zero_grad(set_to_none=True)
  for step,b in enumerate(tqdm(loaders['train'],desc=a.config,leave=False)):
   x,rm,pm,c,y,_,_=b; x=x.to(dev).bfloat16(); rm=rm.to(dev); pm=pm.to(dev); c=c.to(dev).bfloat16(); y=y.to(dev)
   if not spatial: c=torch.zeros_like(c)
   with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(x,rm,c,pm); loss=crit(logits,y)/acc
   loss.backward();
   if (step+1)%acc==0: nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
   tl.append(float(loss.detach())*acc)
  model.eval(); ys=[]; ps=[]; vl=[]; pred_rows=[]
  with torch.no_grad():
   for b in loaders['valid']:
    x,rm,pm,c,y,ids,raw=b; x=x.to(dev).bfloat16(); rm=rm.to(dev); pm=pm.to(dev); c=c.to(dev).bfloat16(); y=y.to(dev)
    if not spatial: c=torch.zeros_like(c)
    with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(x,rm,c,pm); l=crit(logits,y)
    p=logits.softmax(1)[:,1].float().cpu().numpy(); ys.extend(y.cpu().numpy()); ps.extend(p); vl.append(float(l)); pred_rows.extend(zip(ids,y.cpu().numpy(),p,raw))
  m=metrics(np.asarray(ys),np.asarray(ps)); score=.5*m['auc']+.5*m['auprc']; hist.append({'epoch':ep,'train_loss':float(np.mean(tl)),'valid_loss':m['loss'],'valid_auc':m['auc'],'valid_auprc':m['auprc'],'valid_accuracy':m['accuracy'],'valid_f1':m['f1'],'valid_sensitivity':m['sensitivity'],'valid_specificity':m['specificity'],'valid_brier':m['brier'],'s_val':score,'lr':opt.param_groups[0]['lr'],'epoch_seconds':time.time()-t0})
  if score>best+1e-4 or (abs(score-best)<=1e-4 and (m['loss']<hist[best_epoch-1]['valid_loss'] if best_epoch else True)):
   best=score; best_epoch=ep; since=0; torch.save({'model':model.state_dict(),'epoch':ep,'configuration':a.config,'seed':SEED,'valid':m,'s_val':score,'config':{'d_model':384,'layers':4,'heads':8,'ffn':1536,'dropout':.2,'pool':pool,'spatial':spatial,'relation':relation}},out/'seed17_best.pt'); pd.DataFrame([{'patient_id':z[0],'label':int(z[1]),'death_risk':float(z[2]),'predicted_class':int(z[2]>=.5),'checkpoint_epoch':ep,'configuration':a.config,'seed':SEED,'raw_patch_count':z[3]} for z in pred_rows]).to_csv(out/'validation_predictions.csv',index=False)
  else: since+=1
  pd.DataFrame(hist).to_csv(out/'epoch_metrics.csv',index=False)
  if ep>=30 and since>=20: break
 pd.DataFrame(hist).to_csv(out/'epoch_metrics.csv',index=False); (out/'run_status.json').write_text(json.dumps({'status':'COMPLETE','configuration':a.config,'seed':SEED,'best_epoch':best_epoch,'best_s_val':best,'test_evaluated':False},indent=2))
if __name__=='__main__': main()
