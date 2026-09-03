#!/usr/bin/env python3
import json, time, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path('/root/autodl-tmp/surgen_hemicore'); BASE=ROOT/'outputs/thesis_predictor_first_v6'; OUT=BASE/'p2_variable_evidence_predictor'; CACHE=OUT/'cache_compact'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COUNTS=[0,1,2,4,8,16,32,48,64]
def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def make_cache(split, csvname):
 rows=pd.read_csv(ROOT/'prepared/sr386_full'/csvname); ids=rows.patient_id.astype(str).tolist(); coarse=[]; labels=rows.label.to_numpy(np.int64)
 toks={}
 for s in (1,17,42):
  z=np.load(BASE/f'tokens/full_teacher_seed{s}.npz'); prefix='train' if split=='train' else 'validation'; toks[s]=z[f'{prefix}_region_tokens']
  assert list(z[f'{prefix}_patient_id'].astype(str))==ids
 for row in rows.itertuples():
  with np.load(row.features_filepath) as f: feat=f['img_features'].astype(np.float32); coords=f['coords'].astype(np.float32)
  cache_split='valid' if split=='validation' else 'train'
  rc=np.load(ROOT/'outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache/regions_k64'/cache_split/(row.case_id+'.npz'))
  q,ptr=rc['region_patch_indices'],rc['region_ptr']; arr=[]
  for i in range(64):
   x=feat[q[ptr[i]:ptr[i+1]]]; c=coords[q[ptr[i]:ptr[i+1]]]
   mu=x.mean(0) if len(x) else np.zeros(1024,np.float32); sd=x.std(0) if len(x)>1 else np.zeros(1024,np.float32); cen=rc['region_centroid'][i].astype(np.float32); spr=rc['region_spread'][i].astype(np.float32)
   arr.append(np.concatenate([mu,sd,np.log1p(len(x)).reshape(1),cen,spr]))
  coarse.append(np.asarray(arr,np.float32))
 tmap=pd.read_csv(BASE/'p1r_protocol_correction/predictions/train_oof_ensemble_predictions.csv' if split=='train' else BASE/'p1r_protocol_correction/predictions/validation_platt_predictions.csv'); col='ensemble_probability' if split=='train' else 'platt_probability'; tm=dict(zip(tmap.patient_id.astype(str),tmap[col].astype(float)))
 d=CACHE/split; d.mkdir(parents=True,exist_ok=True); torch.save(torch.from_numpy(np.asarray(coarse)).half(),d/'coarse.pt'); torch.save(torch.ones(len(ids),64,dtype=torch.bool),d/'region_valid_mask.pt')
 for s in (1,17,42): torch.save(torch.from_numpy(toks[s]).half(),d/f'fine_seed{s}.pt')
 pd.DataFrame({'patient_id':ids,'label':labels,'teacher_risk':[tm[x] for x in ids],'teacher_source':['train_oof_calibrated' if split=='train' else 'validation_final_calibrated']*len(ids)}).to_csv(d/'metadata.csv',index=False)
 return len(ids)
class Pool(nn.Module):
 def __init__(self,d): super().__init__(); self.a=nn.Sequential(nn.Linear(d,d//2),nn.Tanh(),nn.Linear(d//2,1))
 def forward(self,x,m):
  score=self.a(x).squeeze(-1).float().masked_fill(~m,-1e4); w=torch.softmax(score,-1)*m.float(); w=w/w.sum(-1,keepdim=True).clamp_min(1e-8); return (x*w.unsqueeze(-1).to(x.dtype)).sum(1)
class Net(nn.Module):
 def __init__(self,coarse_dim=2053,d=384):
  super().__init__(); self.d=d; self.coarse=nn.Sequential(nn.Linear(coarse_dim,d),nn.GELU(),nn.Dropout(.3)); self.gpool=Pool(d); self.lpool=Pool(d); self.res=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.3)); self.alpha=nn.Linear(2*d+1,1); self.hf=nn.Linear(d,1); self.hg=nn.Linear(d,1); self.hl=nn.Linear(d,1)
 def forward(self,c,f,m):
  assert c.shape[-1]==2053 and f.shape[-1]==self.d and m.dtype==torch.bool
  g=self.gpool(self.coarse(c),torch.ones(c.shape[:2],dtype=torch.bool,device=c.device)); b,M,r=m.shape; gg=g[:,None,:].expand(b,M,-1); l=self.lpool(f[:,None,:,:].expand(b,M,r,-1).reshape(b*M,r,-1),m.reshape(b*M,r)).reshape(b,M,-1); empty=(~m.any(-1,keepdim=True)).float(); a=torch.sigmoid(self.alpha(torch.cat([gg,l,empty],-1))); z=gg+a*self.res(torch.cat([gg,l],-1)); return self.hf(z).squeeze(-1),self.hg(g).squeeze(-1),self.hl(l).squeeze(-1),a.squeeze(-1)
def load(split,seed):
 d=CACHE/split; meta=pd.read_csv(d/'metadata.csv'); return torch.load(d/'coarse.pt',weights_only=True).to(DEV),torch.load(d/f'fine_seed{seed}.pt',weights_only=True).to(DEV),meta
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=2);args=ap.parse_args();
 for x in ['docs','metrics','figures','checkpoints','logs']: (OUT/x).mkdir(parents=True,exist_ok=True)
 t0=time.perf_counter(); ntr=make_cache('train','surgen_train_10.csv'); nva=make_cache('validation','surgen_validate_10.csv'); load_s=time.perf_counter()-t0
 torch.save({'schema':{'coarse_dim':2053,'fine_dim':384,'regions':64,'dtype':'float16'}},OUT/'cache_schema.pt')
 trc,trf,trm=load('train',17); vac,vaf,vam=load('validation',17); y=torch.tensor(trm.label.to_numpy(),device=DEV).float(); teacher=torch.tensor(trm.teacher_risk.to_numpy(),device=DEV).float();
 torch.manual_seed(17); net=Net().to(DEV); opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=5e-4); bs=16; rows=[]; peak=0.; total=time.perf_counter()
 for ep in range(1,args.epochs+1):
  net.train(); st=time.perf_counter(); loss_sum=0.; batches=0
  for a in range(0,ntr,bs):
   ix=slice(a,min(a+bs,ntr)); B=min(bs,ntr-a); masks=[]
   for k in COUNTS:
    m=torch.zeros(B,4,64,dtype=torch.bool,device=DEV)
    for j in range(B):
     if k: m[j,:,torch.randperm(64,device=DEV)[:k]]=True
    masks.append(m)
   m=torch.cat(masks,1)
   with torch.autocast('cuda',dtype=torch.bfloat16): pf,pg,pl,alpha=net(trc[ix],trf[ix],m); yy=y[ix,None].expand_as(pf); tt=teacher[ix,None].expand_as(pf); loss=nn.functional.binary_cross_entropy_with_logits(pf,yy)-.30*nn.functional.binary_cross_entropy_with_logits(pg[:,None].expand_as(pf),yy)-.20*nn.functional.binary_cross_entropy_with_logits(pl,yy)-.30*nn.functional.kl_div(F.log_softmax(torch.stack([-pf,pf],-1),-1),torch.stack([1-tt,tt],-1),reduction='batchmean')
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();loss_sum+=float(loss);batches+=1; peak=max(peak,torch.cuda.max_memory_allocated()/2**20 if DEV.type=='cuda' else 0)
  net.eval();
  with torch.no_grad():
   m=torch.zeros(nva,len(COUNTS),64,dtype=torch.bool,device=DEV)
   for j,k in enumerate(COUNTS):
    if k:m[:,j,torch.arange(k,device=DEV)]=True
   with torch.autocast('cuda',dtype=torch.bfloat16): pf,pg,pl,al=net(vac,vaf,m)
   p=torch.sigmoid(pf).float().cpu().numpy(); yy=vam.label.to_numpy();
  for j,k in enumerate(COUNTS): rows.append({'epoch':ep,'k':k,'loss':loss_sum/batches,'auroc':roc_auc_score(yy,p[:,j]),'auprc':average_precision_score(yy,p[:,j]),'brier':brier_score_loss(yy,p[:,j]),'alpha_mean':float(al[:,j].mean())})
  pd.DataFrame(rows).to_csv(OUT/'metrics/p2_smoke_metrics.csv',index=False); print(json.dumps({'epoch':ep,'seconds':time.perf_counter()-st,'batches':batches,'peak_mib':peak}))
 torch.save(net.state_dict(),OUT/'checkpoints/smoke_seed17.pt'); pd.DataFrame({'cache_load_seconds':[load_s],'smoke_total_seconds':[time.perf_counter()-total],'epoch_seconds':[(time.perf_counter()-total)/args.epochs],'peak_gpu_mib':[peak],'train_patients':[ntr],'validation_patients':[nva]}).to_csv(OUT/'metrics/p2_smoke_throughput.csv',index=False)
 (OUT/'docs/v6_p2_pipeline_repair_report.md').write_text('# V6 P2 数据管线修复与Smoke验收\n\n本阶段生成固定K64紧凑缓存，训练阶段不读取patch文件，不构造patch级padding。仅运行seed=17、2个epoch；未启动三seed、OOF、P4、Test、PPO或MoE。详见metrics。\n')
if __name__=='__main__':main()
