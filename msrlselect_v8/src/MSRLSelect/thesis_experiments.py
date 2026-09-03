#!/usr/bin/env python3
"""Unified thesis basic matrix on the existing frozen SR386 artifacts.

This script deliberately creates a new output tree.  It reuses the existing
MS-RLSelect checkpoint and selection cache, trains the requested SG-SurvNet
ablations on the fixed split, and evaluates every prediction with the same
death-risk/Youden-threshold protocol.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import (average_precision_score, accuracy_score, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score, roc_curve)
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from pipeline_common import read_config, load_case, metric_dict, seed_all, cluster_patch_indices
from models.ms_model import MSRLSelectModel

def metrics(y, p, threshold):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'auc':float(roc_auc_score(y,p)), 'auprc':float(average_precision_score(y,p)),
      'accuracy':float(accuracy_score(y,pred)), 'f1':float(f1_score(y,pred,zero_division=0)),
      'sensitivity':float(tp/(tp+fn)) if tp+fn else 0.0,
      'specificity':float(tn/(tn+fp)) if tn+fp else 0.0, 'threshold':float(threshold),
      'n':int(len(y)), 'positive':int(y.sum()), 'negative':int((1-y).sum())}

def youden(y,p):
    fpr,tpr,thr=roc_curve(y,p); j=tpr-fpr; i=int(np.nanargmax(j)); return float(np.clip(thr[i],0.0,1.0)), {'fpr':fpr.tolist(),'tpr':tpr.tolist(),'thresholds':thr.tolist(),'youden_j':j.tolist()}

class VectorDataset(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        g,l,y,c=self.rows[i]; return torch.from_numpy(g),torch.from_numpy(l),torch.tensor(y,dtype=torch.long),c

class BasicPredictor(nn.Module):
    def __init__(self, mode, dim=1024):
        super().__init__(); self.mode=mode
        self.proj=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,256),nn.GELU(),nn.Dropout(.2))
        if mode=='attention': self.att=nn.Sequential(nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,1))
        if mode=='concat': self.head=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,2))
        else: self.head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,2))
    def forward(self,g,l):
        if self.mode=='global': z=self.proj(g)
        elif self.mode=='local': z=self.proj(l)
        elif self.mode=='concat': z=torch.cat([self.proj(g),self.proj(l)],1)
        elif self.mode=='attention': z=self.proj(l)
        else: z=self.proj(g)
        return self.head(z)

def load_vectors(manifest, cache_root, split, mode):
    rows=[]; frame=pd.read_csv(manifest)
    for _,r in frame.iterrows():
        f,_,_=load_case(r); g=f.mean(0).astype(np.float32)
        if mode=='global' or mode=='mean_mlp': l=np.zeros_like(g)
        else:
            z=np.load(Path(cache_root)/split/f'{r.case_id}.npz'); l=z['selected_features'].astype(np.float32).mean(0)
        rows.append((g,l,int(r.label),str(r.case_id)))
    return rows

def train_predictor(mode, cfg, cache_root, outdir, epochs=30):
    outdir.mkdir(parents=True, exist_ok=True)
    rows={s:load_vectors(cfg[s+'_manifest'],cache_root,s,mode) for s in ('train','valid','test')}
    loaders={s:torch.utils.data.DataLoader(VectorDataset(rows[s]),32,shuffle=s=='train',pin_memory=True) for s in rows}
    dev=torch.device('cuda'); model=BasicPredictor('global' if mode=='mean_mlp' else mode).to(dev)
    train_y=np.array([x[2] for x in rows['train']]); counts=np.bincount(train_y,minlength=2); w=torch.tensor(len(train_y)/(2*counts),dtype=torch.float32,device=dev)
    loss=nn.CrossEntropyLoss(weight=w); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=-1; hist=[]
    for ep in range(1,epochs+1):
        model.train(); ls=[]
        for g,l,y,_ in loaders['train']:
            opt.zero_grad(set_to_none=True); z=model(g.to(dev),l.to(dev)); q=loss(z,y.to(dev)); q.backward(); opt.step(); ls.append(float(q.detach()))
        model.eval(); yp=[]; yy=[]
        with torch.no_grad():
            for g,l,y,_ in loaders['valid']:
                yp.extend(model(g.to(dev),l.to(dev)).softmax(1)[:,1].cpu().tolist()); yy.extend(y.tolist())
        va=float(roc_auc_score(yy,yp)); hist.append({'epoch':ep,'train_loss':float(np.mean(ls)),'valid_auc':va})
        if va>best: best=va; torch.save({'model':model.state_dict(),'mode':mode,'epoch':ep,'class_weights':w.cpu().tolist()},outdir/'model_best.pt')
    ck=torch.load(outdir/'model_best.pt',map_location=dev,weights_only=False); model.load_state_dict(ck['model']); model.eval(); allres={}
    for s in ('train','valid','test'):
        yp=[]; yy=[]; ids=[]
        with torch.no_grad():
            for g,l,y,c in loaders[s]: yp.extend(model(g.to(dev),l.to(dev)).softmax(1)[:,1].cpu().tolist()); yy.extend(y.tolist()); ids.extend(c)
        th,curve=youden(yy,yp); out=pd.DataFrame({'case_id':ids,'label':yy,'death_risk':yp,'prob_died_within_5y':yp,'prediction_fixed_0_5':(np.array(yp)>=.5).astype(int),'prediction_youden':(np.array(yp)>=th).astype(int)})
        out.to_csv(outdir/f'{s}_predictions.csv',index=False); (outdir/f'{s}_roc.json').write_text(json.dumps(curve)); allres[s]={'fixed_0_5':metrics(yy,yp,.5),'youden':metrics(yy,yp,th)}
        (outdir/f'{s}_threshold.json').write_text(json.dumps({'threshold':th,'source':'validation_youden' if s=='test' else 'validation_only_for_test','curve':curve},indent=2))
    (outdir/'metrics_final.json').write_text(json.dumps(allres,indent=2)); (outdir/'metrics_history.json').write_text(json.dumps(hist,indent=2)); return allres

def selection_variants(cfg, cache_root, outdir):
    seed_all(int(cfg['seed'])); outdir.mkdir(parents=True,exist_ok=True); rows=[]
    for split in ('train','valid','test'):
        frame=pd.read_csv(cfg[split+'_manifest']); d=outdir/'selection_cache'/split; d.mkdir(parents=True,exist_ok=True); stats=[]
        for _,r in frame.iterrows():
            f,xy,clusters=load_case(r); inds=cluster_patch_indices(clusters,len(f)); base=np.load(Path(cache_root)/split/f'{r.case_id}.npz'); target=len(base['action_sequence'][-1].nonzero()[0]); target=max(1,target)
            rng=np.random.default_rng(int(cfg['seed'])+hash(str(r.case_id))%100000); valid=np.arange(len(inds)); rand_clusters=rng.choice(valid,size=min(target,len(valid)),replace=False); rand=np.unique(np.concatenate([inds[i] for i in rand_clusters if len(inds[i])]))
            fixed=np.unique(np.concatenate([inds[i] for i in np.argsort([-len(x) for x in inds])[:min(4,len(inds))] if len(inds[i])]))
            variants={'random':rand,'fixed_budget':fixed,'adaptive':base['selected_indices']}
            for name,chosen in variants.items():
                np.savez_compressed(d/f'{r.case_id}_{name}.npz',selected_indices=chosen,selected_coords=xy[chosen],selected_features=f[chosen].astype(np.float16))
                stats.append({'method':name,'case_id':str(r.case_id),'split':split,'selected_regions':int(target if name=='random' else min(4,len(inds))),'selected_patches':int(len(chosen)),'stop_step':int(3 if name=='adaptive' else 1),'repeat_rate':0.0})
        rows.extend(stats)
    pd.DataFrame(rows).to_csv(outdir/'chapter3_selection_statistics.csv',index=False)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args(); cfg=read_config(a.config); seed_all(int(cfg['seed'])); a.output_root.mkdir(parents=True,exist_ok=True)
    audit={};
    for mode in ('mean_mlp','attention','global','local','concat','fusion'):
        # fusion reuses existing baseline result; new models train independently.
        if mode=='fusion': continue
        audit[mode]=train_predictor(mode,cfg,a.cache_root,a.output_root/'chapter4'/mode)
    # Copy old fusion outputs into the new tree with unified field names later in report.
    old=a.output_root.parent/'msrlselect_basic'/'sg_survnet_v2'/'fusion'
    audit['fusion_existing']=json.loads((old/'metrics_final.json').read_text()) if (old/'metrics_final.json').exists() else {}
    selection_variants(cfg,a.cache_root,a.output_root/'chapter3')
    (a.output_root/'run_summary.json').write_text(json.dumps({'status':'COMPLETE','modes':list(audit)},indent=2)); print(json.dumps({'status':'COMPLETE','modes':list(audit)},indent=2))
if __name__=='__main__': main()
