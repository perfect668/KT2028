#!/usr/bin/env python3
"""Deterministic MS-RLSelect inference and reusable selection-cache export."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch
from pipeline_common import read_config, load_case, cluster_patch_indices, seed_all
ROOT=Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT))
from models.ms_model import MSRLSelectModel
from models.ms_policy import MSRegionPolicy

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True); p.add_argument('--device',default='cuda'); a=p.parse_args(); c=read_config(a.config)
 if not torch.cuda.is_available() and a.device=='cuda': raise RuntimeError('CUDA is required for MS-RLSelect export')
 seed_all(int(c['seed'])); device=torch.device(a.device)
 ck=torch.load(c['msrlselect_checkpoint'],map_location=device,weights_only=False); ckargs=ck.get('args',{})
 token_dim=int(ckargs.get('token_dim',c['token_dim'])); layers=int(ckargs.get('layers',c['layers'])); heads=int(ckargs.get('heads',c['heads']))
 model=MSRLSelectModel(token_dim=token_dim,num_layers=layers,num_heads=heads).to(device).eval()
 policy=MSRegionPolicy(token_dim,token_dim,int(c['budget'])).to(device).eval(); model.load_state_dict(ck['model']); policy.load_state_dict(ck['policy'])
 manifests={'train':c['train_manifest'],'valid':c['valid_manifest'],'test':c['test_manifest']}; total=0; rows=[]
 for split,manifest in manifests.items():
  outdir=a.output_root/split; outdir.mkdir(parents=True,exist_ok=True); mf=[]
  frame=pd.read_csv(manifest)
  for row in frame.itertuples(index=False):
   features,coords,clusters=load_case(row); n=len(features); cluster_indices=cluster_patch_indices(clusters,n)
   x=torch.from_numpy(features[None]).to(device); xy=torch.from_numpy(coords[None]).to(device)
   with torch.no_grad():
    _,base=model(x,xy,[clusters]); valid=base['cluster_mask']; prev=0.; actions=[]; step_patch=[]; gains=[]; final=None
    for step in range(int(c['max_steps'])):
     action,_,_,_=policy.select(base['global_token'],base['cluster_tokens'],base['spatial_features'],valid,deterministic=True)
     action=action[0].bool(); selected=np.unique(np.concatenate([cluster_indices[i] for i in torch.where(action)[0].tolist() if len(cluster_indices[i])],axis=0))
     # The frozen basic policy is conditioned on the slide tokens and has no
     # recurrent state.  Deterministic inference therefore returns the same
     # action on repeated calls; evaluate the selected mask once and reuse its
     # confidence for the stop rule instead of redundantly re-encoding a WSI.
     if step == 0:
      logits,_=model(x,xy,[clusters],region_mask=action[None]); selected_confidence=float(logits.softmax(1).max(1).values.item())
     confidence=selected_confidence; gain=confidence-prev; prev=confidence
     actions.append(action.cpu().numpy().astype(np.uint8)); step_patch.append(selected.tolist()); gains.append(gain); final=selected
     if step+1>=int(c['min_steps']) and len(gains)>=2 and float(np.mean(np.abs(gains[-2:])))<0.01: break
   if final is None or len(final)==0: raise RuntimeError(f'no selected patches for {row.case_id}')
   rel=f'{row.case_id}.npz'; np.savez_compressed(outdir/rel, selected_indices=final.astype(np.int64), selected_coords=coords[final].astype(np.float32), selected_features=features[final].astype(np.float16), action_sequence=np.stack(actions), confidence_gains=np.asarray(gains,dtype=np.float32))
   metadata={'case_id':str(row.case_id),'patient_id':str(getattr(row,'patient_id',row.case_id)),'split':split,'label':int(row.label),'cache_file':rel,'selected_patch_count':int(len(final)),'selected_region_count':int(actions[-1].sum()),'stop_step':int(len(actions)),'step_selected_patch_indices':step_patch}
   with (outdir/'manifest.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(metadata)+'\n')
   mf.append(metadata); rows.append(metadata); total+=1
  (outdir/'summary.json').write_text(json.dumps({'split':split,'count':len(mf),'selected_patch_count_mean':float(np.mean([m['selected_patch_count'] for m in mf])),'stop_step_mean':float(np.mean([m['stop_step'] for m in mf]))},indent=2))
 expected=423; got=len(rows); ids=[r['case_id'] for r in rows]; result={'status':'PASS' if got==expected and len(set(ids))==expected else 'FAIL','total':got,'expected':expected,'unique_case_ids':len(set(ids)),'split_counts':{s:sum(r['split']==s for r in rows) for s in manifests}}
 (a.output_root/'export_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
