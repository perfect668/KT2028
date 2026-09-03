#!/usr/bin/env python3
import argparse,json,math,random,hashlib,sys,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).parent)); from stage1c_train import Store,seed_all

ROOT=Path('/root/autodl-tmp/surgen_hemicore'); BASE=ROOT/'outputs/thesis_predictor_first_v6'; OUT=BASE/'p2_variable_evidence_predictor'; DEV=torch.device('cuda'); COUNTS=[0,1,2,4,8,16,32,48,64]; STRATS=['uniform_random','morphology_diverse','redundant','teacher_ranked']; SEEDS=[1,17,42]
def logit(p):p=np.clip(np.asarray(p),1e-6,1-1e-6);return np.log(p/(1-p))
def cal(p,a=.2629993315,b=-.2729875987):return 1/(1+np.exp(-(a*logit(p)+b)))
def met(y,p):return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),brier=brier_score_loss(y,p),nll=log_loss(y,p,labels=[0,1]),s_val=.5*(roc_auc_score(y,p)+average_precision_score(y,p)))
def ece(y,p):
 y=np.asarray(y);p=np.asarray(p);e=np.linspace(0,1,11);z=0
 for i in range(10):
  q=(p>=e[i])&(p<(e[i+1] if i<9 else 1.01))
  if q.any():z+=q.mean()*abs(p[q].mean()-y[q].mean())
 return z
class Pool(nn.Module):
 def __init__(self,d):super().__init__();self.v=nn.Linear(d,d);self.u=nn.Linear(d,d);self.a=nn.Linear(d,1)
 def forward(self,x,m):
  s=self.a(torch.tanh(self.v(x))*torch.sigmoid(self.u(x))).squeeze(-1).float().masked_fill(~m,-1e4);w=torch.softmax(s,-1)*m.float();w=w/w.sum(-1,keepdim=True).clamp_min(1e-8);return (x*w.to(x.dtype).unsqueeze(-1)).sum(1),w
class P2Model(nn.Module):
 def __init__(self):
  super().__init__(); self.patch_proj=nn.Sequential(nn.Linear(1024,384),nn.GELU(),nn.Dropout(.3));self.pool=Pool(384);self.coarse=nn.Sequential(nn.Linear(2053,384),nn.GELU(),nn.Dropout(.3));self.global_pool=Pool(384);self.local_pool=Pool(384);self.res=nn.Sequential(nn.Linear(768,384),nn.GELU(),nn.Dropout(.3));self.alpha=nn.Sequential(nn.Linear(769,1));self.head=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,2));self.global_head=nn.Linear(384,2);self.local_head=nn.Linear(384,2);self.suff=nn.Linear(384,1)
 def forward(self,patches,pm,coarse,rm,pf):
  b,r,p,d=patches.shape;z=self.patch_proj(patches.reshape(b*r,p,d));fine,w=self.pool(z,pm.reshape(b*r,p));fine=fine.reshape(b,r,-1);g,_=self.global_pool(self.coarse(coarse),rm);lm=pf.any(-1);l,_=self.local_pool(fine,pf);l=torch.where(lm[:,None],l,torch.zeros_like(l));a=torch.sigmoid(self.alpha(torch.cat([g,l,1-lm.float().unsqueeze(1)],1))).squeeze(1);f=torch.layer_norm(g+a[:,None]*self.res(torch.cat([g,l],1)),(384,));return self.head(f),self.global_head(g),self.local_head(l),self.suff(f).squeeze(1),a,w
def load_patient(root,store,row,split='train'):
 with np.load(row.features_filepath) as z:f=z['img_features'].astype(np.float32);coords=z['coords'].astype(np.float32)
 z=np.load(root/'outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache/regions_k64'/split/f'{row.case_id}.npz');q=z['region_patch_indices'];ptr=z['region_ptr'];parts=[f[q[ptr[i]:ptr[i+1]]] for i in range(64)];
 n=np.array([len(x) for x in parts],np.float32);cent=z['region_centroid'].astype(np.float32);std=np.array([x.std(0).mean() if len(x)>1 else 0 for x in parts],np.float32);spread=np.zeros((64,2),np.float32)
 for i in range(64):
  if len(parts[i]):
   cc=coords[q[ptr[i]:ptr[i+1]]];spread[i]=cc.max(0)-cc.min(0)
 coarse=np.concatenate([np.array([x.mean(0) if len(x) else np.zeros(1024) for x in parts]),np.array([x.std(0) if len(x)>1 else np.zeros(1024) for x in parts]),np.log1p(n)[:,None],cent,spread],1).astype(np.float32)
 return parts,coarse
def masks(parts,k,seed,teacher=None):
 n=len(parts);rng=np.random.default_rng(seed);valid=np.arange(n);out=[]
 out.append(rng.choice(valid,k,replace=False) if k else np.array([],int))
 if k:
  # farthest-point diversity on deterministic region descriptors
  desc=np.array([x.mean(0) if len(x) else np.zeros(1024) for x in parts]);chosen=[int(rng.integers(n))]
  while len(chosen)<k:
   dist=((desc-desc[chosen[-1]])**2).mean(1);dist[chosen]=-1;chosen.append(int(dist.argmax()))
  out.append(np.array(chosen));
  sim=desc@desc.T/(np.linalg.norm(desc,axis=1)[:,None]*np.linalg.norm(desc,axis=1)[None,:]+1e-8);chosen=[int(rng.integers(n))]
  while len(chosen)<k:
   score=sim[:,chosen].mean(1);score[chosen]=2;chosen.append(int(score.argmin()))
  out.append(np.array(chosen))
  rank=np.argsort(-(np.linalg.norm(desc,axis=1)))
  out.append(rank[:k])
 else: out += [np.array([],int)]*3
 return out
def batch(rows,store,model,split,k,seed,teachers=None):
 rec=[]
 for j,row in enumerate(rows):
  parts,coarse=load_patient(ROOT,store,row,split);ix=masks(parts,k,seed+j,teachers);rec.append((parts,coarse,ix))
 b=len(rec);X=np.zeros((b,64,max(max(map(len,x[0])) for x in rec),1024),np.float32);PM=np.zeros(X.shape[:3],bool);C=np.stack([x[1] for x in rec]);PF=np.zeros((b,64),bool)
 for i,(parts,_,sets) in enumerate(rec):
  for j,p in enumerate(parts):X[i,j,:len(p)]=p;PM[i,j,:len(p)]=1
  for z in sets[0]:PF[i,z]=1
 return torch.from_numpy(X).to(DEV).bfloat16(),torch.from_numpy(PM).to(DEV),torch.from_numpy(C).to(DEV).bfloat16(),torch.ones(b,64,dtype=torch.bool,device=DEV),torch.from_numpy(PF).to(DEV),torch.tensor([r.label for r in rows],device=DEV)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=int(os.environ.get('P2_EPOCHS','120'))); ap.add_argument('--batch-size',type=int,default=int(os.environ.get('P2_BATCH','2'))); args=ap.parse_args()
 for d in ['docs','metrics','predictions','figures','checkpoints','configs','logs']: (OUT/d).mkdir(parents=True,exist_ok=True)
 train=list(pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv').itertuples(index=False));valid=list(pd.read_csv(ROOT/'prepared/sr386_full/surgen_validate_10.csv').itertuples(index=False));teacher_tr=pd.read_csv(BASE/'p1r_protocol_correction/predictions/train_oof_ensemble_predictions.csv').sort_values('patient_id');teacher_v=pd.read_csv(BASE/'p1r_protocol_correction/predictions/validation_platt_predictions.csv').sort_values('patient_id');store=Store(ROOT,'k64')
 val_masks={}
 for i,r in enumerate(valid):
  parts,_=load_patient(ROOT,store,r,'valid');val_masks[r.patient_id]={k:[m.tolist() for m in masks(parts,k,9000+i*100+k)] for k in COUNTS}
 (OUT/'audit_validation_masks.json').write_text(json.dumps(val_masks))
 allrows=[];predval=[]
 for seed in SEEDS:
  seed_all(seed);m=P2Model().to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=5e-4);best=-1;best_ep=0;hist=[]
  pt={r.patient_id:float(p) for r,p in zip(teacher_tr.itertuples(),teacher_tr.ensemble_probability)};pv={r.patient_id:float(p) for r,p in zip(teacher_v.itertuples(),teacher_v.platt_probability)}
  for ep in range(1,args.epochs+1):
   m.train();losses=[]
   for k in COUNTS:
    for st in range(0,len(train),args.batch_size):
     rs=train[st:st+args.batch_size];X,PM,C,RM,PF,Y=batch(rs,store,m,'train',k,seed+ep+k);opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.bfloat16):fo,go,lo,qo,aa,w=m(X,PM,C,RM,PF);y=Y.float();tf=torch.tensor([pt[r.patient_id] for r in rs],device=DEV);lf=nn.functional.cross_entropy(fo,Y);lg=nn.functional.cross_entropy(go,Y);ll=nn.functional.cross_entropy(lo,Y);kl=nn.functional.kl_div(torch.log_softmax(fo,1),torch.stack([1-tf,tf],1),reduction='batchmean');lb=((fo.softmax(1)[:,1]-y)**2).mean();qs=((torch.abs(fo.softmax(1)[:,1]-tf)<.05).float());lq=nn.functional.binary_cross_entropy_with_logits(qo,qs);loss=lf-.30*lg-.20*ll-.30*kl-.10*lb-.10*lq
     loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach()))
   vm=[];vy=[];va=[]
   m.eval()
   with torch.no_grad():
    for k in COUNTS:
     pp=[];yy=[]
     for st in range(0,len(valid),args.batch_size):
      rs=valid[st:st+args.batch_size];X,PM,C,RM,PF,Y=batch(rs,store,m,'valid',k,9000+seed+k)
      with torch.autocast('cuda',dtype=torch.bfloat16):fo,go,lo,qo,aa,w=m(X,PM,C,RM,PF)
      pp.extend(fo.softmax(1)[:,1].float().cpu().tolist());yy.extend(Y.cpu().tolist())
     mm=met(yy,pp);vm.append(dict(seed=seed,epoch=ep,k=k,**mm,alpha_mean=float(aa.mean().cpu())))
   sv=np.mean([x['s_val'] for x in vm if x['k']==64]);hist.extend(vm)
   if sv>best+1e-12:best=sv;best_ep=ep;torch.save({'model':m.state_dict(),'seed':seed,'epoch':ep},OUT/f'checkpoints/seed{seed}_best.pt')
   if ep>=30 and ep-best_ep>=20: break
  pd.DataFrame(hist).to_csv(OUT/f'metrics/seed{seed}_epoch_metrics.csv',index=False)
  # Best checkpoint predictions on fixed validation masks and all counts.
  m.load_state_dict(torch.load(OUT/f'checkpoints/seed{seed}_best.pt',map_location=DEV,weights_only=False)['model']);m.eval()
  for k in COUNTS:
   pp=[];yy=[];aa=[]
   with torch.no_grad():
    for st in range(0,len(valid),args.batch_size):
     rs=valid[st:st+args.batch_size];X,PM,C,RM,PF,Y=batch(rs,store,m,'valid',k,9000+seed+k)
     with torch.autocast('cuda',dtype=torch.bfloat16):fo,go,lo,qo,a,w=m(X,PM,C,RM,PF)
     pp.extend(fo.softmax(1)[:,1].float().cpu().tolist());yy.extend(Y.cpu().tolist());aa.extend(a.cpu().tolist())
   for r,p,a0 in zip(valid,pp,aa):predval.append({'patient_id':r.patient_id,'label':r.label,'seed':seed,'k':k,'probability':p,'alpha':a0})
 allv=pd.DataFrame(predval);allv.to_csv(OUT/'predictions/validation_variable_evidence.csv',index=False)
 # aggregate same-patient masks by k (the current fixed generator yields one representative per strategy in audit; model evaluation uses equal patient weighting)
 teacher_vmap={r.patient_id:float(p) for r,p in zip(teacher_v.itertuples(),teacher_v.platt_probability)};rows=[]
 for k in COUNTS:
  g=allv[allv.k==k].groupby('patient_id');y=np.array([int(g.get_group(pid).label.iloc[0]) for pid in g.groups]);p=np.array([g.get_group(pid).probability.mean() for pid in g.groups]);rows.append({'k':k,**met(y,p),'ece':ece(y,p),'alpha_mean':float(allv[allv.k==k].alpha.mean())})
 pd.DataFrame(rows).to_csv(OUT/'metrics/evidence_count_curves.csv',index=False)
 # teacher consistency / p4 gates
 g64=allv[allv.k==64].groupby('patient_id').probability.mean();ids=list(g64.index);tf=np.array([teacher_vmap[i] for i in ids]);pf=np.array([g64[i] for i in ids]);m64=met(np.array([int(valid[[r.patient_id for r in valid].index(i)].label) for i in ids]),pf);tm=met(np.array([valid[[r.patient_id for r in valid].index(i)].label for i in ids]),tf);curve=pd.read_csv(OUT/'metrics/evidence_count_curves.csv');p4={'full_evidence_mean_abs_diff':float(np.mean(np.abs(pf-tf))),'full_consistency_pass':bool(np.mean(np.abs(pf-tf))<=.02),'global_baseline_s_val':float(curve[curve.k==0].s_val.iloc[0]),'fused64_s_val':float(m64['s_val']),'teacher64_s_val':float(tm['s_val']),'global_safety_pass':bool(m64['s_val']>=curve[curve.k==0].s_val.iloc[0]-.005),'teacher_retention_pass':bool(m64['s_val']>=tm['s_val']-.01),'local_effective_pass':False,'calibration_ece64':float(curve[curve.k==64].ece.iloc[0])}
 p4['local_effective_pass']=bool(any(curve[curve.k.isin([8,16,32])].s_val.to_numpy()>curve[curve.k==0].s_val.iloc[0]+.005));p4['p4_pass']=bool(all([p4['full_consistency_pass'],p4['global_safety_pass'],p4['teacher_retention_pass'],p4['local_effective_pass'],p4['calibration_ece64']<=.10]));pd.DataFrame([p4]).to_csv(OUT/'metrics/p4_gate_decision.csv',index=False)
 plt.figure();plt.plot(curve.k,curve.s_val,'o-');plt.xlabel('Evidence regions k');plt.ylabel('S_val');plt.grid();plt.savefig(OUT/'figures/evidence_count_performance_curve.png',dpi=180);plt.close();plt.figure();plt.plot(curve.k,curve.brier,'o-',label='Brier');plt.plot(curve.k,curve.ece,'o-',label='ECE');plt.legend();plt.xlabel('Evidence regions k');plt.savefig(OUT/'figures/evidence_count_calibration_curve.png',dpi=180);plt.close();plt.figure();plt.hist(allv.alpha,bins=20);plt.xlabel('alpha');plt.savefig(OUT/'figures/residual_gate_distribution.png',dpi=180);plt.close()
 (OUT/'docs/v6_p2_variable_evidence_report.md').write_text('# V6 Phase P2 变长证据预测器\n\n本阶段仅训练global粗粒度、local细粒度和全局恒等残差融合预测器。P1R C3三seed教师冻结；Train使用P1R OOF ensemble，Validation使用P1R最终Platt ensemble。结果见`metrics/p4_gate_decision.csv`。本阶段未启动RL、MoE、PPO、Test或论文写作。\n')
 (OUT/'docs/v6_p2_architecture_and_interface.md').write_text('# P2结构与接口\n\n粗粒度分支读取全部64区域的均值、标准差、log patch数、中心和离散度；细粒度分支读取当前mask区域的Stage1 gated token。融合为`LN(g+alpha*R(g,l))`，global为恒等主路径。\n')
 (OUT/'configs/p2_config.json').write_text(json.dumps({'counts':COUNTS,'strategies':STRATS,'seeds':SEEDS,'loss':'BCE_fused-.30BCE_global-.20BCE_local-.30KL-.10Brier-.10BCE_sufficiency','teacher':'P1R calibrated','test_access':False},indent=2));print(json.dumps(p4,indent=2))
if __name__=='__main__':main()
