#!/usr/bin/env python3
import csv,hashlib,json,math,os,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
sys.path.insert(0,str(Path(__file__).parent))
from stage1c_model import Stage1CDenseModel
from stage1c_train import Store,batchify

ROOT=Path('/root/autodl-tmp/surgen_hemicore');OUT=ROOT/'outputs/thesis_predictor_first_v6';S1=ROOT/'outputs/thesis_algorithm_upgrade_v5/stage1c_dense_confirmation';DATA=ROOT/'prepared/sr386_full';PY='/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python'
SEEDS=[1,17,42]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def met(y,p):return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'nll':log_loss(y,p,labels=[0,1])}
def ece(y,p,bins=10):
 y=np.asarray(y);p=np.asarray(p);edges=np.linspace(0,1,bins+1);v=0.
 for i in range(bins):
  m=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else edges[i+1]+1e-12))
  if m.any():v+=m.mean()*abs(p[m].mean()-y[m].mean())
 return float(v)
def calibrated(p,t):
 p=np.clip(np.asarray(p),1e-6,1-1e-6);z=np.log(p/(1-p))/t;return 1/(1+np.exp(-z))

@torch.no_grad()
def infer_and_tokens(model,store,rows,dev):
 model.eval();ys=[];ps=[];toks=[]
 for st in range(0,len(rows),16):
  batch=rows[st:st+16];X,PM,C,S,RM,Y=batchify(store,'train' if len(rows)==255 else 'valid',batch,dev)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   b,r,p,d=X.shape;z=model.proj(X.reshape(b*r,p,d));pooled,_=model.pool(z,PM.reshape(b*r,p));pooled=pooled.reshape(b,r,-1);o=model(X,PM,C,S,RM)
  ys.extend(Y.cpu().tolist());ps.extend(o.softmax(1)[:,1].float().cpu().tolist());toks.append(pooled.float().cpu().numpy().astype(np.float16))
 return np.asarray(ys),np.asarray(ps),np.concatenate(toks)

def audit_metadata():
 files=[]
 for p in sorted((ROOT/'metadata/dataset_csv').glob('SR386*.csv')):
  # Phase P0 intentionally reads only Train/Validation metadata records.
  split='test' if '_test' in p.name else ('validation' if '_validate' in p.name else ('train' if '_train' in p.name else 'other'))
  with p.open(encoding='utf-8-sig',newline='') as f:fields=next(csv.reader(f))
  n=None;missing=None
  if split!='test':
   d=pd.read_csv(p);n=len(d);missing={c:float(d[c].isna().mean()) for c in d.columns}
  files.append({'file':str(p.relative_to(ROOT)),'split':split,'fields':fields,'rows_read':n,'missing_rate':missing})
 return files

def main():
 for d in ['docs','metrics','predictions','checkpoints/final_c3','tokens','audit','configs']: (OUT/d).mkdir(parents=True,exist_ok=True)
 metadata=audit_metadata();(OUT/'audit/metadata_field_inventory.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
 # Label alignment on development splits only.
 aligns=[]
 for split,src in [('train','SR386_5y_sur_train.csv'),('validation','SR386_5y_sur_validate.csv')]:
  a=pd.read_csv(DATA/f'surgen_{"validate" if split=="validation" else split}_10.csv');b=pd.read_csv(ROOT/'metadata/dataset_csv'/src);left=a.assign(join_id=a.patient_id.str.replace(r'_T0*', '_T',regex=True));m=left.merge(b[['case_id','label']],left_on='join_id',right_on='case_id',suffixes=('_prepared','_metadata'));aligns.append({'split':split,'prepared_n':len(a),'metadata_n':len(b),'matched_n':len(m),'label_mismatch_n':int((m.label_prepared!=m.label_metadata).sum())})
 (OUT/'audit/label_alignment.json').write_text(json.dumps(aligns,indent=2))
 # Reuse three frozen C3 teachers; hard links avoid checkpoint duplication.
 reuse=[];dev=torch.device('cuda');store=Store(ROOT,'k64');train=list(pd.read_csv(DATA/'surgen_train_10.csv').itertuples(index=False));valid=list(pd.read_csv(DATA/'surgen_validate_10.csv').itertuples(index=False));all_val=[];perseed=[]
 for seed in SEEDS:
  src=S1/f'formal/C3_K64_gated_no_relation_no_spatial/seed{seed}/best.pt';dst=OUT/f'checkpoints/final_c3/seed{seed}_best.pt'
  if dst.exists():dst.unlink()
  os.link(src,dst);ck=torch.load(src,map_location=dev,weights_only=False);m=Stage1CDenseModel(pool='gated_attention_pool',spatial=False,relation=False).to(dev);missing=m.load_state_dict(ck['model'],strict=True);yt,pt,tt=infer_and_tokens(m,store,train,dev);yv,pv,tv=infer_and_tokens(m,store,valid,dev)
  np.savez_compressed(OUT/f'tokens/full_teacher_seed{seed}.npz',train_patient_id=np.array([r.patient_id for r in train]),train_region_tokens=tt,validation_patient_id=np.array([r.patient_id for r in valid]),validation_region_tokens=tv)
  mt=met(yt,pt);mv=met(yv,pv);row={'seed':seed,**{f'train_{k}':v for k,v in mt.items()},**{f'validation_{k}':v for k,v in mv.items()},'auroc_gap':mt['auroc']-mv['auroc'],'s_val':.5*(mv['auroc']+mv['auprc']),'checkpoint_epoch':ck['epoch']};perseed.append(row);all_val.append(pv)
  pd.DataFrame({'patient_id':[r.patient_id for r in valid],'label':yv,'death_risk':pv,'seed':seed,'checkpoint_epoch':ck['epoch']}).to_csv(OUT/f'predictions/validation_seed{seed}.csv',index=False)
  reuse.append({'seed':seed,'source':str(src.relative_to(ROOT)),'reused':str(dst.relative_to(ROOT)),'sha256':sha(src),'hardlink_same_inode':src.stat().st_ino==dst.stat().st_ino,'strict_load':True,'checkpoint_epoch':ck['epoch']})
 pd.DataFrame(perseed).to_csv(OUT/'metrics/full_teacher_multiseed.csv',index=False);(OUT/'audit/c3_reuse_manifest.json').write_text(json.dumps(reuse,indent=2))
 # OOF fold predictions and calibration learned only from Train OOF.
 folds=[]
 for f in range(5):
  d=OUT/f'checkpoints/folds/fold{f}';st=json.loads((d/'run_status.json').read_text());assert st['status']=='COMPLETE' and not st['test_evaluated'];q=pd.read_csv(d/'oof_predictions.csv');folds.append(q)
 oof=pd.concat(folds).sort_values('patient_id').reset_index(drop=True);assert len(oof)==255 and oof.patient_id.nunique()==255
 y=oof.label.to_numpy();p=oof.death_risk.to_numpy();res=minimize_scalar(lambda t:log_loss(y,calibrated(p,t),labels=[0,1]),bounds=(.05,10),method='bounded');T=float(res.x);pc=calibrated(p,T);oof['calibrated_death_risk']=pc;oof.to_csv(OUT/'predictions/train_oof_full_predictions.csv',index=False)
 # Label shuffle sanity uses OOF predictions and fixed RNG.
 rng=np.random.default_rng(6600);sh=np.array([roc_auc_score(rng.permutation(y),p) for _ in range(2000)]);shuffle={'repetitions':2000,'mean_auroc':float(sh.mean()),'sd':float(sh.std(ddof=1)),'ci95':[float(np.quantile(sh,.025)),float(np.quantile(sh,.975))]};(OUT/'audit/label_shuffle.json').write_text(json.dumps(shuffle,indent=2))
 vmean=np.mean(all_val,axis=0);vc=calibrated(vmean,T);yv=np.array([r.label for r in valid]);pd.DataFrame({'patient_id':[r.patient_id for r in valid],'label':yv,'ensemble_death_risk':vmean,'calibrated_death_risk':vc,'temperature_from_train_oof':T}).to_csv(OUT/'predictions/validation_full_predictions.csv',index=False)
 prev=float(np.mean(y));base_o=np.full(len(y),prev);base_v=np.full(len(yv),prev);calrows=[]
 for scope,yy,raw,cal,base in [('train_oof',y,p,pc,base_o),('validation',yv,vmean,vc,base_v)]:
  for name,prob in [('constant_train_prevalence',base),('uncalibrated',raw),('temperature_scaled',cal)]:calrows.append({'scope':scope,'method':name,'temperature':T if name=='temperature_scaled' else np.nan,**met(yy,prob),'ece_10bin':ece(yy,prob)})
 pd.DataFrame(calrows).to_csv(OUT/'metrics/oof_calibration_metrics.csv',index=False)
 # P0-P3 gates.
 mean_s=float(np.mean([r['s_val'] for r in perseed]));auc_sd=float(np.std([r['validation_auroc'] for r in perseed],ddof=1));ap_mean=float(np.mean([r['validation_auprc'] for r in perseed]));val_cal=met(yv,vc);val_base=brier_score_loss(yv,base_v);p0=all(x['label_mismatch_n']==0 and x['matched_n']==x['prepared_n'] for x in aligns) and abs(shuffle['mean_auroc']-.5)<.03
 p1=mean_s>=.6751 and auc_sd<=.035 and ap_mean>yv.mean()+.05
 max_gap=max(r['auroc_gap'] for r in perseed);p2=max_gap<=.10
 brier_improvement=(val_base-val_cal['brier'])/val_base;p3=brier_improvement>=.05 and ece(yv,vc)<=.10
 gates={'P0':{'pass':bool(p0),'label_alignment':aligns,'label_shuffle':shuffle},'P1':{'pass':bool(p1),'mean_s_val':mean_s,'threshold':.6751,'validation_auroc_sd':auc_sd,'mean_auprc':ap_mean,'validation_positive_rate':float(yv.mean())},'P2':{'pass':bool(p2),'maximum_seed_train_validation_auroc_gap':max_gap,'threshold':.10},'P3':{'pass':bool(p3),'validation_calibrated_brier':val_cal['brier'],'constant_brier':val_base,'relative_brier_improvement':brier_improvement,'validation_calibrated_ece':ece(yv,vc),'temperature':T},'overall_enter_P2':bool(p0 and p1 and p2 and p3)};(OUT/'metrics/p0_p3_gate_decision.json').write_text(json.dumps(gates,indent=2))
 # Hash manifest after all primary artifacts.
 manifest=[]
 for pth in sorted([p for p in OUT.rglob('*') if p.is_file()] + [ROOT/'src/MSRLSelect/v6_oof_train.py',ROOT/'src/MSRLSelect/v6_finalize.py']):manifest.append({'path':str(pth.relative_to(ROOT)),'sha256':sha(pth),'bytes':pth.stat().st_size})
 pd.DataFrame(manifest).to_csv(OUT/'metrics/v6_sha256_manifest.csv',index=False)
 print(json.dumps(gates,indent=2))
if __name__=='__main__':main()
