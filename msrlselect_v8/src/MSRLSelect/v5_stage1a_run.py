#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from v5_stage1a_infrastructure import build_region_cache_case,build_region_map,stratified_patch_indices
from v5_region_dataset import V5RegionDataset,collate_v5
from models.v5_dense_region_transformer import DenseRegionTransformer

def load(r):
    with np.load(r.features_filepath) as z: return z['img_features'].astype(np.float32),z['coords'].astype(np.float32)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); data=a.root/'prepared/sr386_full'; a.out.mkdir(parents=True,exist_ok=True); (a.out/'metrics').mkdir(parents=True,exist_ok=True); (a.out/'cache').mkdir(parents=True,exist_ok=True)
 manifests={'train':data/'surgen_train_10.csv','valid':data/'surgen_validate_10.csv'}
 for k in (64,128):
  for split,mp in manifests.items():
   d=a.out/f'cache/regions_k{k}'/split; d.mkdir(parents=True,exist_ok=True)
   for r in pd.read_csv(mp).itertuples(index=False): build_region_cache_case(r.features_filepath,r.clusters_json_filepath,r.coordinates_filepath,d/f'{r.case_id}.npz',k)
  rows=[]
  for split,mp in manifests.items():
   for r in pd.read_csv(mp).itertuples(index=False):
    with np.load(a.out/f'cache/regions_k{k}'/split/f'{r.case_id}.npz') as z:
     sizes=z['region_size']; rows.extend({'case_id':r.case_id,'split':split,'region_id':i,'patch_count':int(n),'centroid_x':float(z['region_centroid'][i,0]),'centroid_y':float(z['region_centroid'][i,1]),'radius':float(np.linalg.norm(z['region_spread'][i])),'density':float(z['region_density'][i]),'morph_cluster':int(z['region_morph_cluster'][i])} for i,n in enumerate(sizes))
  pd.DataFrame(rows).to_csv(a.out/f'metrics/k{k}_region_statistics.csv',index=False)
 s64=pd.read_csv(a.out/'metrics/k64_region_statistics.csv'); s128=pd.read_csv(a.out/'metrics/k128_region_statistics.csv'); pd.DataFrame([{'k':64,'mean_patch':s64.patch_count.mean(),'min_patch':s64.patch_count.min(),'max_patch':s64.patch_count.max(),'p95_patch':s64.patch_count.quantile(.95)},{'k':128,'mean_patch':s128.patch_count.mean(),'min_patch':s128.patch_count.min(),'max_patch':s128.patch_count.max(),'p95_patch':s128.patch_count.quantile(.95)}]).to_csv(a.out/'metrics/region_quality_comparison.csv',index=False)
 # P1024 indices (train/valid only)
 for split,mp in manifests.items():
  d=a.out/'cache/p1024_stratified'/split; d.mkdir(parents=True,exist_ok=True)
  for r in pd.read_csv(mp).itertuples(index=False):
   f,c=load(r); clusters=json.loads(Path(r.clusters_json_filepath).read_text()); idx=stratified_patch_indices(f,c,clusters,1024,seed=0); np.savez_compressed(d/f'{r.case_id}.npz',patch_indices=idx,raw_patch_count=np.array([len(f)],np.int32),budget=np.array([1024],np.int32),algorithm='P1024-stratified')
 # benchmark model on region summaries; no optimizer or metrics
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); bench=[]
 if dev.type=='cuda':
  for k in (10,64,128):
   for bs in (1,4,8,16):
    model=DenseRegionTransformer().to(dev).to(torch.bfloat16).train(); x=torch.randn(bs,k,1024,device=dev,dtype=torch.bfloat16); mask=torch.ones(bs,k,device=dev,dtype=torch.bool); c=torch.rand(bs,k,2,device=dev,dtype=torch.bfloat16); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter(); y=model(x,mask,c); torch.cuda.synchronize(); tf=time.perf_counter()-t; t=time.perf_counter(); y.float().sum().backward(); torch.cuda.synchronize(); tb=time.perf_counter()-t; bench.append({'model':'gated_attention_pool','k':k,'batch_size':bs,'peak_gb':torch.cuda.max_memory_allocated()/2**30,'forward_s':tf,'forward_backward_s':tf+tb,'oom':False,'nan_inf':bool((~torch.isfinite(y)).any())})
  pd.DataFrame(bench).to_csv(a.out/'metrics/v5_stage1a_memory_benchmark.csv',index=False); pd.DataFrame(bench).to_csv(a.out/'metrics/v5_stage1a_throughput_benchmark.csv',index=False)
 (a.out/'docs').mkdir(exist_ok=True); (a.out/'docs/v5_stage1a_status.json').write_text(json.dumps({'status':'COMPLETE','test_evaluated':False,'formal_training':False,'splits':['train','valid'],'cuda':dev.type},indent=2))
if __name__=='__main__': main()
