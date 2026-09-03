#!/usr/bin/env python3
"""Read-only Stage-0 audit for MS-RLSurv V5.

This script never calls an optimizer or metric evaluator. Test files are read
only for integrity/count checks; model forward checks use Train/Validation.
"""
from __future__ import annotations
import argparse, json, math, re, subprocess
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

def q(v):
    v=np.asarray(v,float); return {"min":float(np.min(v)) if len(v) else None,"max":float(np.max(v)) if len(v) else None,"mean":float(np.mean(v)) if len(v) else None,"median":float(np.median(v)) if len(v) else None,"p25":float(np.quantile(v,.25)) if len(v) else None,"p75":float(np.quantile(v,.75)) if len(v) else None,"p95":float(np.quantile(v,.95)) if len(v) else None}
def norm_coords(c):
 c=np.asarray(c,float); lo=c.min(0,keepdims=True); scale=np.maximum(c.max(0,keepdims=True)-lo,1.0); return (c-lo)/scale
def cluster_components(x,k=8):
 if len(x)<=1:return 1
 kk=min(k,len(x)); d,ix=cKDTree(x).query(x,k=kk+1)
 rows=np.repeat(np.arange(len(x)),kk); cols=ix[:,1:].reshape(-1)
 a=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(x),len(x))); a=a.maximum(a.T)
 return int(connected_components(a,directed=False,return_labels=False))
def mean_cosine(x,limit=256):
 if len(x)<2:return 1.0 if len(x)==1 else float('nan')
 if len(x)>limit:
  rng=np.random.default_rng(20260827); x=x[rng.choice(len(x),limit,replace=False)]
 z=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12); s=z@z.T; n=len(z); return float((s.sum()-np.trace(s))/(n*(n-1)))
def load_row(row):
 with np.load(row.features_filepath,allow_pickle=False) as z: f=z['img_features'].astype(np.float32); c=z['coords'].astype(np.float32)
 with np.load(row.clusters_filepath,allow_pickle=False) as z: lab=z['features_cluster_indices'].reshape(-1).astype(np.int64)
 clusters=json.loads(Path(row.clusters_json_filepath).read_text())
 return f,c,lab,clusters
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); root=a.root; out=a.out; out.mkdir(parents=True,exist_ok=True); (out/'stage0').mkdir(exist_ok=True)
 manifests={s:root/'prepared/sr386_full'/f'surgen_{"validate" if s=="valid" else s}_10.csv' for s in ('train','valid','test')}
 frames={s:pd.read_csv(p) for s,p in manifests.items()}; allf=pd.concat([x.assign(split=s) for s,x in frames.items()],ignore_index=True)
 # Integrity
 rows=[]; patch_counts=[]; empty_cases=[]; zero_cases=[]; duplicate_cases=[]; label_rows=[]
 for r in allf.itertuples(index=False):
  rec={'case_id':r.case_id,'patient_id':r.patient_id,'split':r.split,'label':int(r.label),'features_exists':Path(r.features_filepath).exists(),'coords_exists':Path(r.coordinates_filepath).exists(),'clusters_exists':Path(r.clusters_filepath).exists(),'clusters_json_exists':Path(r.clusters_json_filepath).exists()}
  try:
   f,c,lab,clusters=load_row(r); rec.update({'feature_rows':len(f),'coord_rows':len(c),'cluster_label_rows':len(lab),'feature_dim':f.shape[1] if f.ndim==2 else -1,'feature_nan':int(np.isnan(f).sum()),'feature_inf':int(np.isinf(f).sum()),'coord_nan':int(np.isnan(c).sum()),'coord_inf':int(np.isinf(c).sum()),'all_zero_feature':bool(np.all(f==0)),'duplicate_coord_rows':int(len(c)-len(np.unique(c,axis=0))), 'cluster_json_count':len(clusters),'cluster_npz_unique':int(len(np.unique(lab))),'cluster_empty_count':int(sum(len(x)==0 for x in clusters))})
   patch_counts.append(len(f));
   if rec['cluster_empty_count'] or rec['cluster_npz_unique']<10: empty_cases.append(r.case_id)
   if rec['all_zero_feature']: zero_cases.append(r.case_id)
   if rec['duplicate_coord_rows']: duplicate_cases.append(r.case_id)
  except Exception as e: rec['error']=repr(e)
  rows.append(rec)
 integ=pd.DataFrame(rows); integ.to_csv(out/'stage0/patient_data_integrity.csv',index=False)
 split_summary={}
 for s,f in frames.items(): split_summary[s]={'n':int(len(f)),'label_0':int((f.label==0).sum()),'label_1':int((f.label==1).sum()),'patient_ids':int(f.patient_id.nunique()),'case_ids':int(f.case_id.nunique())}
 overlap={a+'__'+b:sorted(set(frames[a].patient_id)&set(frames[b].patient_id)) for a,b in combinations(frames,2)}
 split_summary.update({'expected_counts':{'train':255,'valid':84,'test':84},'patient_overlap':overlap,'total_cases':int(len(allf)),'patch_count_summary':q(patch_counts),'empty_cluster_cases':empty_cases,'all_zero_feature_cases':zero_cases,'duplicate_coordinate_cases':duplicate_cases,'label_definition':'0=survived 5y; 1=died within 5y','output_definition':'probability/death_risk is class-1 five-year death risk; AUROC/AUPRC use higher risk for label 1','feature_dim_unique':sorted(integ.feature_dim.dropna().unique().tolist()),'row_mismatch_cases':integ.loc[(integ.feature_rows!=integ.coord_rows)|(integ.feature_rows!=integ.cluster_label_rows),'case_id'].tolist()})
 (out/'stage0/split_and_label_summary.json').write_text(json.dumps(split_summary,indent=2,ensure_ascii=False))
 # Cluster statistics
 spatial=[]; morph=[]; centers={}
 for r in allf.itertuples(index=False):
  f,c,lab,clusters=load_row(r); nc=norm_coords(c); centers[r.case_id]=[]
  for cid,idx0 in enumerate(clusters):
   idx=np.asarray(idx0,dtype=int); idx=idx[(idx>=0)&(idx<len(f))]; valid=idx
   if len(valid):
    xy=nc[valid]; cen=xy.mean(0); rad=np.linalg.norm(xy-cen,axis=1); bbox=float(np.linalg.norm(xy.max(0)-xy.min(0))); comps=cluster_components(xy); max_comp=0
    # component labels for max component fraction
    if len(xy)>1:
     kk=min(8,len(xy)); d,ix=cKDTree(xy).query(xy,k=kk+1); rr=np.repeat(np.arange(len(xy)),kk); cc=ix[:,1:].reshape(-1); aa=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(xy),len(xy))).maximum(coo_matrix((np.ones(len(rr)),(cc,rr)),shape=(len(xy),len(xy)))); _,labs=connected_components(aa,directed=False); max_comp=int(np.bincount(labs).max())
    spatial.append({'case_id':r.case_id,'split':r.split,'cluster_id':cid,'patch_count':len(valid),'centroid_x':float(cen[0]),'centroid_y':float(cen[1]),'radius_max':float(rad.max()),'diameter_bbox':bbox,'knn8_components':comps,'largest_component_fraction':max_comp/max(1,len(valid))})
    morph.append({'case_id':r.case_id,'split':r.split,'cluster_id':cid,'patch_count':len(valid),'feature_variance_mean':float(np.var(f[valid],axis=0).mean()),'mean_cosine':mean_cosine(f[valid])})
    centers[r.case_id].append((cid,cen,f[valid].mean(0)))
   else:
    spatial.append({'case_id':r.case_id,'split':r.split,'cluster_id':cid,'patch_count':0,'centroid_x':None,'centroid_y':None,'radius_max':None,'diameter_bbox':None,'knn8_components':0,'largest_component_fraction':0.0}); morph.append({'case_id':r.case_id,'split':r.split,'cluster_id':cid,'patch_count':0,'feature_variance_mean':None,'mean_cosine':None}); centers[r.case_id].append((cid,None,None))
 pd.DataFrame(spatial).to_csv(out/'stage0/cluster_spatial_statistics.csv',index=False); pd.DataFrame(morph).to_csv(out/'stage0/cluster_morphology_statistics.csv',index=False)
 align=[]
 ids=list(centers)
 for ca,cb in combinations(ids,2):
  aa=[x for x in centers[ca] if x[1] is not None]; bb=[x for x in centers[cb] if x[1] is not None]
  if not aa or not bb: continue
  same=[]
  for cid,xy,fa in aa:
   hit=[z for z in bb if z[0]==cid]
   if hit: same.append(float(np.dot(fa,hit[0][2])/(np.linalg.norm(fa)*np.linalg.norm(hit[0][2])+1e-12)))
  cost=np.ones((len(aa),len(bb)))
  for i,(_,_,fa) in enumerate(aa):
   for j,(_,_,fb) in enumerate(bb): cost[i,j]=1-float(np.dot(fa,fb)/(np.linalg.norm(fa)*np.linalg.norm(fb)+1e-12))
  ri,ci=linear_sum_assignment(cost); best=float(np.mean(1-cost[ri,ci]))
  align.append({'case_a':ca,'case_b':cb,'same_id_mean_cosine':float(np.mean(same)) if same else None,'hungarian_best_mean_cosine':best,'alignment_gain':best-float(np.mean(same)) if same else None,'same_id_pairs':len(same),'matched_pairs':len(ri)})
 pd.DataFrame(align).to_csv(out/'stage0/cluster_cross_patient_alignment.csv',index=False)
 # Token/mask dry run on at least 10 Train/Valid cases only.
 dry=[]
 for _,r in pd.concat([frames['train'].assign(split='train'),frames['valid'].assign(split='valid')]).sort_values('features_filepath').drop_duplicates('case_id').head(12).iterrows():
  f,c,lab,clusters=load_row(r); counts=[len(x) for x in clusters]; valid=np.array([n>0 for n in counts],bool); dry.append({'case_id':r.case_id,'split':r.split,'feature_shape':str((1,len(f),f.shape[1])),'coord_shape':str((1,len(c),2)),'cluster_token_shape':'(1,10,256)','global_token_shape':'(1,256)','padding_mask_shape':'(1,11)','valid_cluster_count':int(valid.sum()),'padding_cluster_count':int((~valid).sum()),'action_mask_shape':'(1,10)','action_valid_count':int(valid.sum()),'feature_value_min':float(f.min()),'feature_value_max':float(f.max()),'coord_norm_min':float(norm_coords(c).min()),'coord_norm_max':float(norm_coords(c).max()),'mask_direction':'True=valid/actionable; Transformer padding=True for ~cluster_mask'})
 pd.DataFrame(dry).to_csv(out/'stage0/token_and_mask_dryrun.csv',index=False); (out/'stage0/token_pipeline_shapes.json').write_text(json.dumps({'input_features':'[B,P,1024]','input_coords':'[B,P,2]','cluster_features':'[B,10,1024]','spatial_features':'[B,10,5]','cluster_tokens':'[B,10,256]','global_token':'[B,256]','transformer_tokens':'[B,11,256]','padding_mask':'[B,11]','policy_action_mask':'[B,10]','dry_run_cases':len(dry)},indent=2))
 # Architecture/checkpoint audit, using Train/Valid only for forward.
 arch={}; ck_audit={}
 try:
  import torch, sys
  sys.path.insert(0,str(root/'src/MSRLSelect')); from models.ms_model import MSRLSelectModel; from models.ms_policy import MSRegionPolicy
  ckp=root/'outputs/thesis_statistical_validation_v4/predictors/seed1/MS-RLSelect_revision/model_best.pt'
  # selector checkpoint is authoritative for Transformer+policy
  ckp=root/'outputs/thesis_statistical_validation_v4/selectors/seed1/model_best.pt' if (root/'outputs/thesis_statistical_validation_v4/selectors/seed1/model_best.pt').exists() else root/'outputs/thesis_targeted_revision_v3/revision/model_best.pt'
  ck=torch.load(ckp,map_location='cpu',weights_only=False); cfg=ck.get('config',ck.get('args',{})); model=MSRLSelectModel(token_dim=int(cfg.get('token_dim',256)),num_layers=int(cfg.get('layers',2)),num_heads=int(cfg.get('heads',4)),use_spatial=True); policy=MSRegionPolicy(int(cfg.get('token_dim',256)),int(cfg.get('token_dim',256)),int(ck.get('budget',4)))
  miss,unexp=model.load_state_dict(ck['model'],strict=False); pmiss,punexp=policy.load_state_dict(ck['policy'],strict=False)
  arch={'checkpoint':str(ckp),'checkpoint_config':cfg,'input_dim':1024,'token_dim':model.encoder.token_dim,'num_clusters':model.encoder.num_clusters,'transformer_layers':len(model.encoder.context_encoder.layers),'num_heads':model.encoder.context_encoder.layers[0].self_attn.num_heads,'has_global_token':True,'attention_type':'standard nn.TransformerEncoder full self-attention','spatial_injection':'projected 5D statistics plus weak gated residual before Transformer','coordinate_stats':'per-case min-max normalization','model_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters()),'module_parameters':{n:sum(p.numel() for p in m.parameters()) for n,m in model.named_children()}}
  ck_audit={'checkpoint':str(ckp),'model_missing_keys':list(miss),'model_unexpected_keys':list(unexp),'policy_missing_keys':list(pmiss),'policy_unexpected_keys':list(punexp),'strict_equivalent':not(miss or unexp or pmiss or punexp)}
  # Forward only on first Train and Valid cases.
  with torch.no_grad():
   for s in ('train','valid'):
    r=frames[s].iloc[0]; f,c,lab,clusters=load_row(r); ft=torch.from_numpy(f[None]); ct=torch.from_numpy(c[None]); logits,aux=model(ft,ct,[clusters]); arch.setdefault('forward_checks',[]).append({'split':s,'case_id':r.case_id,'logits_shape':list(logits.shape),'global_shape':list(aux['global_token'].shape),'cluster_shape':list(aux['cluster_tokens'].shape),'mask_shape':list(aux['cluster_mask'].shape),'mask_valid':int(aux['cluster_mask'].sum())})
 except Exception as e: ck_audit={'error':repr(e)}; arch={'error':repr(e)}
 (out/'stage0/current_model_architecture.json').write_text(json.dumps(arch,indent=2,ensure_ascii=False)); (out/'stage0/checkpoint_loading_audit.json').write_text(json.dumps(ck_audit,indent=2,ensure_ascii=False))
 # Resource feasibility from observed distribution and theoretical matrices.
 n=np.asarray(patch_counts); att={str(k):{'tokens':k,'matrix_elements':k*k,'per_head_fp32_bytes':k*k*4,'per_head_bf16_bytes':k*k*2} for k in (64,128,1024)}
 feas={'observed_patch_counts':q(n),'cases_below_64':int((n<64).sum()),'cases_below_128':int((n<128).sum()),'region_construction':'existing UNI embeddings + coordinates suffice for deterministic within-cluster spatial subdivision; original WSI is not required for K64/K128 cache construction','attention_matrices':att,'estimated_R384_4layer_8head':'approximately 20M parameters, exact activation memory depends on batch and FFN implementation','P1024_recommendation':'BF16 SDPA/Flash Attention, batch 2-4 with gradient accumulation; benchmark only in Stage 1','cache_format':'per-case compressed NPZ with patch_to_region, region_indices, region_features, centroid, spread, area, morphology statistics','blocking_interfaces':['new region assignment/cache schema','variable region count and padding mask','region-level dataset/collate','RL policy STOP action','SG-SurvNet residual local input']}
 (out/'stage0/region_scale_feasibility.json').write_text(json.dumps(feas,indent=2,ensure_ascii=False))
 print(json.dumps({'status':'COMPLETE','cases':len(allf),'out':str(out)},ensure_ascii=False))
if __name__=='__main__': main()
