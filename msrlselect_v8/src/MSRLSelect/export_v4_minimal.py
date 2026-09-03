#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from pipeline_common import read_config, seed_all, load_case, cluster_patch_indices
from export_thesis_selection import load_selector, model_trace, random_trace, save_case_cache

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--summary-root',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--random-seed',type=int,required=True); a=p.parse_args(); c=read_config(a.config); seed_all(int(c['seed'])); a.output_root.mkdir(parents=True,exist_ok=True); model,policy,var=load_selector(a.checkpoint,c); stats=[]
 for split in ('train','valid','test'):
  frame=pd.read_csv(c[f'{split}_manifest'])
  for row in tqdm(frame.itertuples(index=False),total=len(frame),desc=f'{split}-seed{c["seed"]}'):
   features,coords,clusters=load_case(row); ci=cluster_patch_indices(clusters,len(features)); sf=a.summary_root/split/f'{row.case_id}.npz'
   with np.load(sf) as z:
    cf=torch.from_numpy(z['cluster_features'][None].astype(np.float32)).cuda(); sp=torch.from_numpy(z['spatial_features'][None].astype(np.float32)).cuda(); cm=torch.from_numpy(z['cluster_mask'][None].astype(bool)).cuda()
   tr=model_trace(model,policy,'revision',cf,sp,cm,4,2,.01,False); rr=random_trace(cm[0].cpu().numpy(),4,a.random_seed,str(row.case_id))
   for name,trace in (('MS-RLSelect_revision',tr),(f'random_seed{a.random_seed}',rr)):
    f=a.output_root/'selection_cache'/name/split/f'{row.case_id}.npz'; idx=save_case_cache(f,trace,features,coords,ci); acts=trace['actions']; stats.append({'method':name,'case_id':str(row.case_id),'split':split,'selected_regions':int(acts.shape[0]),'selected_patches':int(len(idx)),'stop_step':int(acts.shape[0]),'repeat_rate':0.0,'inference_time_ms':float(trace['inference_time_ms'])})
 pd.DataFrame(stats).to_csv(a.output_root/'chapter3_selection_statistics_partial.csv',index=False); (a.output_root/'selection_export_status.json').write_text(json.dumps({'status':'COMPLETE','seed':int(c['seed']),'random_seed':a.random_seed,'methods':['MS-RLSelect_revision',f'random_seed{a.random_seed}'],'test_evaluated':False},indent=2))
if __name__=='__main__': main()
