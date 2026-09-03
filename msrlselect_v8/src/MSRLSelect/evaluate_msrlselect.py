#!/usr/bin/env python3
"""Evaluate deterministic MS-RLSelect final selected masks and save patient predictions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from pipeline_common import read_config,load_case,cluster_patch_indices,metric_dict,seed_all
ROOT=Path(__file__).resolve().parents[1]
import sys;sys.path.insert(0,str(ROOT))
from models.ms_model import MSRLSelectModel

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();c=read_config(a.config);seed_all(int(c['seed']));dev=torch.device('cuda');ck=torch.load(c['msrlselect_checkpoint'],map_location=dev,weights_only=False);ka=ck.get('args',{});model=MSRLSelectModel(token_dim=int(ka.get('token_dim',c['token_dim'])),num_layers=int(ka.get('layers',c['layers'])),num_heads=int(ka.get('heads',c['heads']))).to(dev).eval();model.load_state_dict(ck['model']);a.output_root.mkdir(parents=True,exist_ok=True)
 for split in ('train','valid','test'):
  frame=pd.read_csv(c[split+'_manifest']); rows=[]
  for row in frame.itertuples(index=False):
   f,xy,clusters=load_case(row);z=np.load(a.cache_root/split/f'{row.case_id}.npz');chosen=set(z['selected_indices'].tolist()); mask=[bool(set(q.tolist())&chosen) for q in cluster_patch_indices(clusters,len(f))];x=torch.from_numpy(f[None]).to(dev);co=torch.from_numpy(xy[None]).to(dev)
   with torch.no_grad(): logits,_=model(x,co,[clusters],region_mask=torch.tensor([mask],device=dev));prob=float(logits.softmax(1)[0,1].item())
   rows.append({'case_id':str(row.case_id),'split':split,'label':int(row.label),'probability':prob,'prediction':int(prob>=.5)})
  out=pd.DataFrame(rows);out.to_csv(a.output_root/f'{split}_predictions.csv',index=False);(a.output_root/f'{split}_metrics.json').write_text(json.dumps(metric_dict(out.label,out.probability),indent=2))
 print(json.dumps({'status':'COMPLETE','output_root':str(a.output_root)},indent=2))
if __name__=='__main__':main()
