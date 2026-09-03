#!/usr/bin/env python3
"""Train the frozen-selection SG-SurvNet basic ablations."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import roc_curve,precision_recall_curve
from tqdm import tqdm
from pipeline_common import read_config,load_case,metric_dict,seed_all

class SGDataset(torch.utils.data.Dataset):
 def __init__(self,manifest,cache_root,mode):
  self.df=pd.read_csv(manifest); self.cache=Path(cache_root); self.mode=mode; self.items=[]
  # Aggregate once at dataset construction.  The source patch features remain
  # unchanged; this avoids repeating hundreds of NPZ reads every epoch.
  for _,r in self.df.iterrows():
   f,_,_=load_case(r); global_x=f.mean(0).astype(np.float32)
   if mode=='global': local_x=np.zeros_like(global_x)
   else:
    z=np.load(self.cache/self._split_for_path(manifest)/f'{r.case_id}.npz'); local_x=z['selected_features'].astype(np.float32).mean(0)
   self.items.append((global_x,local_x,int(r.label),str(r.case_id)))
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  g,l,y,case=self.items[i]; return torch.from_numpy(g),torch.from_numpy(l),torch.tensor(y),case
 def _split_for_path(self,manifest):
  name=Path(manifest).name
  return 'train' if 'train' in name else ('valid' if 'validate' in name else 'test')

class SGSurvNet(nn.Module):
 def __init__(self,dim=1024,mode='fusion'):
  super().__init__(); self.mode=mode
  self.global_proj=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Dropout(.2))
  self.local_proj=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Dropout(.2))
  self.gate=nn.Sequential(nn.Linear(dim*2,dim//4),nn.GELU(),nn.Linear(dim//4,1),nn.Sigmoid())
  self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim//2),nn.GELU(),nn.Dropout(.2),nn.Linear(dim//2,2))
 def forward(self,g,l,return_gate=False):
  if self.mode=='global': z=self.global_proj(g)
  elif self.mode=='local': z=self.local_proj(l)
  else:
   gp,lp=self.global_proj(g),self.local_proj(l); gate=self.gate(torch.cat([gp,lp],1)); z=gate*gp+(1-gate)*lp
  return (self.head(z), gate.squeeze(1)) if return_gate and self.mode=='fusion' else self.head(z)

def evaluate(model,loader,device,criterion,outdir,split,save=True):
 model.eval(); ys=[];ps=[];ids=[];losses=[]
 with torch.no_grad():
  for g,l,y,case in loader:
   logits=model(g.to(device),l.to(device)); losses.append(float(criterion(logits,y.to(device)))); ps.extend(logits.softmax(1)[:,1].cpu().tolist()); ys.extend(y.tolist());ids.extend(case)
 result=metric_dict(ys,ps); result['loss']=float(np.mean(losses));
 if save:
  pd.DataFrame({'case_id':ids,'label':ys,'probability':ps,'prediction':(np.asarray(ps)>=.5).astype(int)}).to_csv(outdir/f'{split}_predictions.csv',index=False)
  fpr,tpr,thr=roc_curve(ys,ps); prec,rec,prthr=precision_recall_curve(ys,ps)
  (outdir/f'{split}_roc.json').write_text(json.dumps({'fpr':fpr.tolist(),'tpr':tpr.tolist(),'thresholds':thr.tolist()})); (outdir/f'{split}_pr.json').write_text(json.dumps({'precision':prec.tolist(),'recall':rec.tolist(),'thresholds':prthr.tolist()}))
 return result

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--cache-root',type=Path,required=True); p.add_argument('--mode',choices=['global','local','fusion'],required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=1e-3); a=p.parse_args(); c=read_config(a.config); seed_all(int(c['seed'])); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); a.output_dir.mkdir(parents=True,exist_ok=True)
 manifests={'train':c['train_manifest'],'valid':c['valid_manifest'],'test':c['test_manifest']}; datasets={}
 for split,m in manifests.items():
  ds=SGDataset(m,a.cache_root,a.mode); ds.split=split; datasets[split]=ds
 loaders={s:torch.utils.data.DataLoader(ds,a.batch_size,shuffle=s=='train',num_workers=int(c.get('num_workers',0)),pin_memory=device.type=='cuda') for s,ds in datasets.items()}
 counts=pd.read_csv(manifests['train']).label.value_counts().reindex([0,1]).fillna(0).to_numpy(); weights=torch.tensor(len(pd.read_csv(manifests['train']))/(2*counts),dtype=torch.float32,device=device); criterion=nn.CrossEntropyLoss(weight=weights); model=SGSurvNet(mode=a.mode).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.0001); best=-float('inf'); history=[]
 for epoch in range(1,a.epochs+1):
  model.train(); train_loss=[]
  for g,l,y,_ in tqdm(loaders['train'],desc=f'{a.mode} train',leave=False):
   opt.zero_grad(set_to_none=True); loss=criterion(model(g.to(device),l.to(device)),y.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); train_loss.append(float(loss.detach()))
  valid=evaluate(model,loaders['valid'],device,criterion,a.output_dir,'valid',save=False); row={'epoch':epoch,'train_loss':float(np.mean(train_loss)),**{'valid_'+k:v for k,v in valid.items()}}; history.append(row); print(json.dumps(row))
  score=valid['auc']
  if np.isfinite(score) and score>best: best=score; torch.save({'model':model.state_dict(),'mode':a.mode,'epoch':epoch,'class_weights':weights.cpu().tolist(),'config':c},a.output_dir/'model_best.pt')
 (a.output_dir/'metrics.json').write_text(json.dumps(history,indent=2)); ck=torch.load(a.output_dir/'model_best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model']); results={}
 for split in ('train','valid','test'): results[split]=evaluate(model,loaders[split],device,criterion,a.output_dir,split,save=True)
 (a.output_dir/'metrics_final.json').write_text(json.dumps(results,indent=2)); (a.output_dir/'config.json').write_text(json.dumps({'mode':a.mode,'seed':c['seed'],'class_weights':weights.cpu().tolist(),'epochs':a.epochs},indent=2)); print(json.dumps({'status':'COMPLETE','mode':a.mode,'best_valid_auc':best,'results':results},indent=2))
if __name__=='__main__': main()
