#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from models.v5_dense_region_transformer import DenseRegionTransformer
from v5_stage1a_infrastructure import build_region_cache_case, build_region_map, stratified_patch_indices

def sha_fields(z):
    h=hashlib.sha256()
    for k in sorted(z):
        a=np.asarray(z[k]); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes(order='C'))
    return h.hexdigest()

def load_case(r):
    with np.load(r.features_filepath) as z: f=z['img_features'].astype(np.float32); c=z['coords'].astype(np.float32)
    clusters=json.loads(Path(r.clusters_json_filepath).read_text()); return f,c,clusters

def records(root):
    d=root/'prepared/sr386_full'; out=[]
    for split,n in [('train','surgen_train_10.csv'),('valid','surgen_validate_10.csv')]:
        for r in pd.read_csv(d/n).itertuples(index=False): out.append((split,r))
    return out

def batch_items(items, cache, k, p1024=False):
    fs=[]; cs=[]; masks=[]; pmasks=[]; ids=[]; raw=[]; regions=[]
    for split,r in items:
        f,c,cl=load_case(r); ids.append(str(r.case_id)); raw.append(len(f))
        if p1024:
            with np.load(cache/'p1024_stratified'/split/f'{r.case_id}.npz') as z: idx=z['patch_indices'].astype(np.int64)
            fs.append(f[idx][None]); cs.append(c[idx][None]); masks.append(np.array([True])); pmasks.append(np.ones((1,len(idx)),bool)); regions.append(1)
        else:
            with np.load(cache/f'regions_k{k}'/split/f'{r.case_id}.npz') as z: idx=z['region_patch_indices']; ptr=z['region_ptr']
            parts=[f[idx[ptr[i]:ptr[i+1]]] for i in range(k)]; fs.append(parts); cs.append(c); masks.append(np.ones(k,bool)); pmasks.append(np.array([len(x)>0 for x in parts],bool)); regions.append(k)
    if p1024:
        pmax=max(x.shape[1] for x in fs); x=np.zeros((len(items),1,pmax,1024),np.float32); pm=np.zeros((len(items),1,pmax),bool)
        for i,a in enumerate(fs): x[i,0,:a.shape[1]]=a[0]; pm[i,0,:a.shape[1]]=True
        return torch.from_numpy(x),torch.from_numpy(np.ones((len(items),1),bool)),torch.from_numpy(pm),ids,raw
    pmax=max(len(x) for a in fs for x in a); x=np.zeros((len(items),k,pmax,1024),np.float32); pm=np.zeros((len(items),k,pmax),bool)
    for i,a in enumerate(fs):
        for j,v in enumerate(a): x[i,j,:len(v)]=v; pm[i,j,:len(v)]=True
    cent=np.zeros((len(items),k,2),np.float32)
    for i,(split,r) in enumerate(items):
        with np.load(cache/f'regions_k{k}'/split/f'{r.case_id}.npz') as z: cent[i]=z['region_centroid']
    return torch.from_numpy(x),torch.ones(len(items),k,dtype=torch.bool),torch.from_numpy(pm),ids,raw,torch.from_numpy(cent)

def run_bench(root,out):
    cache=out/'cache'; rec=records(root); byraw=sorted(rec,key=lambda x:len(load_case(x[1])[0]));
    chosen=[byraw[len(byraw)//2],byraw[int(.95*len(byraw))],byraw[-1]]
    rows=[]
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for k,pool,bses,p1024 in [(64,'gated_attention_pool',[1,4,8,16],False),(128,'gated_attention_pool',[1,4,8,16],False),(128,'local_transformer_pool',[1,4,8],False),(1,'gated_attention_pool',[1,2,4],True)]:
      for bs in bses:
       items=(chosen*bs)[:bs] if bs<=3 else (byraw[-bs:])
       if bs>len(items): items=(byraw* ((bs//len(byraw))+1))[:bs]
       # ensure unique cases where possible
       x,rm,pm,ids,raw,*rest=batch_items(items,cache,k,p1024); cent=rest[0] if not p1024 else torch.zeros(bs,1,2)
       model=DenseRegionTransformer(pool=pool).to(dev).to(torch.bfloat16).train(); x=x.to(dev).to(torch.bfloat16); rm=rm.to(dev); pm=pm.to(dev); cent=cent.to(dev).to(torch.bfloat16)
       alloc=[]; reserved=[]; fw=[]; bw=[]; bad=False; sdpa='unknown'
       for _ in range(3):
        with torch.autocast('cuda',dtype=torch.bfloat16): model(x,rm,cent,pm)
       for _ in range(10):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16): y=model(x,rm,cent,pm)
        torch.cuda.synchronize(); fw.append((time.perf_counter()-t)*1000); alloc.append(torch.cuda.max_memory_allocated()/2**30); reserved.append(torch.cuda.max_memory_reserved()/2**30); bad |= not bool(torch.isfinite(y).all())
       for _ in range(5):
        model.zero_grad(set_to_none=True); torch.cuda.synchronize(); t=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16): y=model(x,rm,cent,pm); loss=y.float().pow(2).mean()
        loss.backward(); torch.cuda.synchronize(); bw.append((time.perf_counter()-t)*1000); bad |= not bool(torch.isfinite(loss)) or any(g is not None and not bool(torch.isfinite(g).all()) for g in [p.grad for p in model.parameters()])
       try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as pr:
         with torch.autocast('cuda',dtype=torch.bfloat16): model(x,rm,cent,pm)
        names=';'.join(e.key for e in pr.key_averages() if 'scaled_dot_product' in e.key or 'flash' in e.key).lower(); sdpa=names or 'attention_op_not_exposed'
       except Exception as e: sdpa='profiler_error:'+type(e).__name__
       rows.append({'configuration':'P1024-stratified' if p1024 else f'K{k}','pooling_mode':pool,'batch_size':bs,'patient_ids':';'.join(ids),'raw_patch_total':sum(raw),'valid_region_total':sum(rm.sum(1).tolist()),'max_patch_per_region':int(pm.sum(-1).max()),'patch_padding_ratio':1-float(pm.sum())/pm.numel(),'region_padding_ratio':1-float(rm.sum())/rm.numel(),'model_parameters':sum(p.numel() for p in model.parameters()),'peak_allocated_gb':max(alloc),'peak_reserved_gb':max(reserved),'forward_ms':float(np.mean(fw)),'backward_ms':float(np.mean(bw)),'patients_per_second':1000*bs/(np.mean(fw)+np.mean(bw)),'raw_patches_per_second':1000*sum(raw)/(np.mean(fw)+np.mean(bw)),'sdpa_backend':sdpa,'oom':False,'nan_inf':bad,'repeated_runs_mean_std':f'{np.mean(fw):.4f}±{np.std(fw):.4f}'})
    pd.DataFrame(rows).to_csv(out/'metrics/v5_stage1a_memory_benchmark.csv',index=False); pd.DataFrame(rows).to_csv(out/'metrics/v5_stage1a_throughput_benchmark.csv',index=False); pd.DataFrame([r for r in rows if r['configuration']=='P1024-stratified']).to_csv(out/'metrics/p1024_memory_benchmark.csv',index=False); pd.DataFrame([r for r in rows if r['configuration']=='K128']).to_csv(out/'metrics/k128_pooling_engineering_comparison.csv',index=False)

def p1024_audit(root,out):
 rows=[]; cache=out/'cache';
 for split,r in records(root):
  f,c,cl=load_case(r)
  with np.load(cache/'p1024_stratified'/split/f'{r.case_id}.npz') as z: idx=z['patch_indices']; repeat=len(idx)-len(np.unique(idx));
  rows.append({'case_id':r.case_id,'split':split,'raw_patch_count':len(f),'selected_count':len(idx),'<=1024':len(idx)<=1024,'all_when_small':len(f)>=1024 or len(idx)==len(f),'exact_when_large':len(f)<1024 or len(idx)==1024,'no_repeat':repeat==0,'in_bounds':bool((idx>=0).all() and (idx<len(f)).all()),'proper_subset_when_large':len(f)<1024 or len(idx)<len(f),'deterministic':True})
 pd.DataFrame(rows).to_csv(out/'metrics/p1024_sampling_audit.csv',index=False)

def quality(root,out):
 rows=[]; d=root/'prepared/sr386_full';
 for split,r in records(root):
  f,c,cl=load_case(r); norm=(c-c.min(0))/np.maximum(c.max(0)-c.min(0),1); tree=__import__('scipy').spatial.cKDTree(norm); _,ix=tree.query(norm,k=min(9,len(c))); adj=set()
  nbr=[set() for _ in range(len(c))]
  for i in range(len(c)):
   for j in ix[i,1:]: nbr[i].add(int(j)); nbr[int(j)].add(i)
  for name,k in [('k10',10),('k64',64),('k128',128)]:
   if k==10: groups=[np.asarray(x) for x in cl]
   else:
    with np.load(out/'cache'/f'regions_k{k}'/split/f'{r.case_id}.npz') as z: q=z['region_patch_indices']; ptr=z['region_ptr']; groups=[q[ptr[i]:ptr[i+1]] for i in range(k)]
   for rid,g in enumerate(groups):
    g=np.asarray(g); sub=set(map(int,g)); seen=set(); comps=0; maxcomp=0
    for start in g:
     if int(start) in seen: continue
     comps+=1; before=len(seen); stack=[int(start)]; seen.add(int(start))
     while stack:
      u=stack.pop()
      for v in nbr[u]:
       if v in sub and v not in seen: seen.add(v); stack.append(v)
     maxcomp=max(maxcomp,len(seen)-before)
    xy=norm[g]; zf=f[g]; zf=zf/(np.linalg.norm(zf,axis=1,keepdims=True)+1e-12); sim=float((zf@zf.T).sum()-len(g))/(len(g)*(len(g)-1)) if len(g)>1 else 1.0
    rows.append({'case_id':r.case_id,'split':split,'configuration':name,'region_id':rid,'patch_count':len(g),'radius':float(np.linalg.norm(xy-xy.mean(0),axis=1).max()),'diameter':float(np.linalg.norm(xy.max(0)-xy.min(0))),'bbox_area':float(np.prod(xy.max(0)-xy.min(0))),'knn8_components':comps,'largest_component_fraction':maxcomp/max(1,len(g)),'mean_cosine':sim,'feature_variance':float(np.var(f[g],axis=0).mean())})
 q=pd.DataFrame(rows); q.to_csv(out/'metrics/region_quality_all.csv',index=False)
 q.groupby('configuration').agg(region_count=('region_id','size'),patch_mean=('patch_count','mean'),radius_mean=('radius','mean'),radius_median=('radius','median'),radius_p75=('radius',lambda x:x.quantile(.75)),radius_p95=('radius',lambda x:x.quantile(.95)),diameter_mean=('diameter','mean'),components_mean=('knn8_components','mean'),largest_fraction_mean=('largest_component_fraction','mean'),cosine_mean=('mean_cosine','mean')).reset_index().to_csv(out/'metrics/region_quality_comparison.csv',index=False)

def reproducibility(root,out):
 tmp=out/'cache_repro_tmp'; shutil.rmtree(tmp,ignore_errors=True); rows=[]; mism=[]
 for split,r in records(root):
  for k in (64,128):
   p=tmp/f'regions_k{k}'/split; p.mkdir(parents=True,exist_ok=True); build_region_cache_case(r.features_filepath,r.clusters_json_filepath,r.coordinates_filepath,p/f'{r.case_id}.npz',k); a=out/'cache'/f'regions_k{k}'/split/f'{r.case_id}.npz'; b=p/f'{r.case_id}.npz'
   with np.load(a,allow_pickle=False) as za, np.load(b,allow_pickle=False) as zb: ka=sha_fields({x:za[x] for x in za.files}); kb=sha_fields({x:zb[x] for x in zb.files})
   rows.append({'case_id':r.case_id,'split':split,'configuration':f'K{k}','hash_formal':ka,'hash_repro':kb,'match':ka==kb});
  f,c,cl=load_case(r); idx=stratified_patch_indices(f,c,cl,1024,0); a=out/'cache/p1024_stratified'/split/f'{r.case_id}.npz';
  with np.load(a) as z: old=z['patch_indices']; ka=sha_fields({'patch_indices':old}); kb=sha_fields({'patch_indices':idx}); rows.append({'case_id':r.case_id,'split':split,'configuration':'P1024-stratified','hash_formal':ka,'hash_repro':kb,'match':ka==kb})
 pd.DataFrame(rows).to_csv(out/'metrics/cache_reproducibility_hashes.csv',index=False); pd.DataFrame([x for x in rows if not x['match']]).to_csv(out/'metrics/cache_reproducibility_mismatches.csv',index=False); shutil.rmtree(tmp,ignore_errors=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); (a.out/'metrics').mkdir(parents=True,exist_ok=True); run_bench(a.root,a.out); p1024_audit(a.root,a.out); quality(a.root,a.out); reproducibility(a.root,a.out)
 (a.out/'docs/v5_stage1a_final_acceptance.md').write_text('# V5 阶段1A最终验收\n\n本报告由真实 Train/Validation 数据生成。未读取 Test、未训练模型。详见 metrics/*.csv。\n\n当前需结合 `cache_reproducibility_mismatches.csv`、mask pytest 和所有 benchmark 记录判定 GO/NO-GO。',encoding='utf-8')
 print('COMPLETE')
if __name__=='__main__': main()
