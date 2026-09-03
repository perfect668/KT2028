#!/usr/bin/env python3
import argparse,csv,hashlib,json,math,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from torch import nn
from stage1c_model import Stage1CDenseModel

CONFIGS={
'C0_K64_mean_relation_no_spatial':dict(kind='k64',pool='mean_pool',relation=True,spatial=False),
'C1_K64_gated_relation_no_spatial':dict(kind='k64',pool='gated_attention_pool',relation=True,spatial=False),
'C2_K64_gated_relation_spatial':dict(kind='k64',pool='gated_attention_pool',relation=True,spatial=True),
'C3_K64_gated_no_relation_no_spatial':dict(kind='k64',pool='gated_attention_pool',relation=False,spatial=False),
'C4_K128_gated_relation_spatial':dict(kind='k128',pool='gated_attention_pool',relation=True,spatial=True)}

def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def lr_for_epoch(ep,max_ep=120,base=1e-4,warmup=5):
 if ep<=warmup:return base*ep/warmup
 return base*.5*(1+math.cos(math.pi*(ep-warmup)/(max_ep-warmup)))

class EarlyStop:
 def __init__(self,min_epoch=30,patience=20):self.min_epoch=min_epoch;self.patience=patience;self.best=-float('inf');self.best_epoch=0
 def update(self,epoch,score):
  improved=score>self.best+1e-12
  if improved:self.best=score;self.best_epoch=epoch
  stop=epoch>=self.min_epoch and epoch-self.best_epoch>=self.patience
  return improved,stop,('patience_exhausted' if stop else '')

class Store:
 def __init__(self,root,kind):self.root=root;self.kind=kind;self.cache={}
 def get(self,split,r):
  key=(split,r.case_id)
  if key in self.cache:return self.cache[key]
  with np.load(r.features_filepath) as z:f=z['img_features'].astype(np.float32)
  z=np.load(self.root/'outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache'/f'regions_{self.kind}'/split/f'{r.case_id}.npz')
  q=z['region_patch_indices'];ptr=z['region_ptr'];parts=[f[q[ptr[i]:ptr[i+1]]] for i in range(int(self.kind[1:]))]
  cent=z['region_centroid'].astype(np.float32);size=(z['region_size']/len(f)).astype(np.float32)
  self.cache[key]=(parts,cent,size);return self.cache[key]

def batchify(store,split,rows,dev):
 arr=[store.get(split,r) for r in rows]; rmax=max(len(a[0]) for a in arr);pmax=max(max(map(len,a[0])) for a in arr)
 X=np.zeros((len(arr),rmax,pmax,1024),np.float32);PM=np.zeros((len(arr),rmax,pmax),bool);C=np.zeros((len(arr),rmax,2),np.float32);S=np.zeros((len(arr),rmax),np.float32);RM=np.zeros((len(arr),rmax),bool)
 for i,(parts,c,s) in enumerate(arr):
  for j,a in enumerate(parts):X[i,j,:len(a)]=a;PM[i,j,:len(a)]=1
  C[i,:len(parts)]=c;S[i,:len(parts)]=s;RM[i,:len(parts)]=1
 return (torch.from_numpy(X).to(dev).bfloat16(),torch.from_numpy(PM).to(dev),torch.from_numpy(C).to(dev).bfloat16(),torch.from_numpy(S).to(dev).bfloat16(),torch.from_numpy(RM).to(dev),torch.tensor([r.label for r in rows],device=dev))

def scores(y,p):
 return dict(auc=roc_auc_score(y,p),auprc=average_precision_score(y,p),brier=brier_score_loss(y,p))

@torch.no_grad()
def evaluate(model,store,split,rows,dev,bs=16,mode='normal'):
 model.eval();ys=[];ps=[];loss=[]
 for st in range(0,len(rows),bs):
  batch=rows[st:st+bs];X,PM,C,S,RM,Y=batchify(store,split,batch,dev)
  if mode=='shuffle_coords':
   for i in range(C.shape[0]):C[i]=C[i,torch.randperm(C.shape[1],device=dev)]
  with torch.autocast('cuda',dtype=torch.bfloat16):o=model(X,PM,C,S,RM,bypass_spatial=(mode=='bypass_spatial'));l=nn.functional.cross_entropy(o,Y)
  prob=o.softmax(1)[:,1].float().cpu().tolist();ys.extend(Y.cpu().tolist());ps.extend(prob);loss.append(float(l))
 s=scores(ys,ps);s['loss']=float(np.mean(loss));return s,ys,ps

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',choices=CONFIGS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=120);p.add_argument('--short',action='store_true');a=p.parse_args()
 cfg=CONFIGS[a.config]|dict(seed=a.seed,max_epochs=a.epochs,min_stop_epoch=30,patience=20,base_lr=1e-4,weight_decay=1e-4,dropout=.2,d_model=384,layers=4,heads=8,warmup_epochs=5,gradient_clip=1.0,precision='BF16',batch_size=16,effective_batch_size=16,checkpoint_metric='0.5*AUROC+0.5*AUPRC',test_access=False)
 seed_all(a.seed);dev=torch.device('cuda');data=a.root/'prepared/sr386_full';out=a.out/a.config/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(cfg,indent=2))
 train=list(pd.read_csv(data/'surgen_train_10.csv').itertuples(index=False));valid=list(pd.read_csv(data/'surgen_validate_10.csv').itertuples(index=False));assert len(train)==255 and len(valid)==84
 store=Store(a.root,cfg['kind']);model=Stage1CDenseModel(pool=cfg['pool'],spatial=cfg['spatial'],relation=cfg['relation']).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=cfg['base_lr'],weight_decay=cfg['weight_decay']);stopper=EarlyStop();hist=[];gate_rows=[];t0=time.time();torch.cuda.reset_peak_memory_stats()
 s0,_,_=evaluate(model,store,'valid',valid,dev);hist.append(dict(epoch=0,lr=0,train_loss=np.nan,train_auc=np.nan,train_auprc=np.nan,train_brier=np.nan,valid_loss=s0['loss'],valid_auc=s0['auc'],valid_auprc=s0['auprc'],valid_brier=s0['brier'],s_val=.5*(s0['auc']+s0['auprc'])))
 best_epoch=0;best_score=-1.;reason='max_epochs';stop_epoch=0
 max_epochs=min(a.epochs,2) if a.short else a.epochs
 for ep in range(1,max_epochs+1):
  lr=lr_for_epoch(ep,a.epochs);[g.update(lr=lr) for g in opt.param_groups];model.train();order=np.random.default_rng(a.seed+ep).permutation(len(train));ys=[];ps=[];losses=[];gate_grad=[];mlp_grad=[]
  for st in range(0,len(train),16):
   batch=[train[i] for i in order[st:st+16]];X,PM,C,S,RM,Y=batchify(store,'train',batch,dev);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):o=model(X,PM,C,S,RM);loss=nn.functional.cross_entropy(o,Y)
   loss.backward();
   for b in model.blocks:
    if b.gate is not None:
     gate_grad.append(float(b.gate.grad.abs().mean()) if b.gate.grad is not None else 0.);mlp_grad.append(float(b.bias_mlp[-1].weight.grad.abs().mean()) if b.bias_mlp[-1].weight.grad is not None else 0.)
   nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();losses.append(float(loss.detach()));ys.extend(Y.cpu().tolist());ps.extend(o.softmax(1)[:,1].detach().float().cpu().tolist())
  tr=scores(ys,ps);va,vy,vp=evaluate(model,store,'valid',valid,dev);sv=.5*(va['auc']+va['auprc']);improved,should_stop,why=stopper.update(ep,sv)
  hist.append(dict(epoch=ep,lr=lr,train_loss=np.mean(losses),train_auc=tr['auc'],train_auprc=tr['auprc'],train_brier=tr['brier'],valid_loss=va['loss'],valid_auc=va['auc'],valid_auprc=va['auprc'],valid_brier=va['brier'],s_val=sv))
  gates=[torch.sigmoid(b.gate).detach().cpu().tolist() for b in model.blocks if b.gate is not None]
  for li,g in enumerate(gates):
   for hi,v in enumerate(g):gate_rows.append(dict(epoch=ep,layer=li,head=hi,gate=v,gate_grad_mean=np.mean(gate_grad) if gate_grad else 0,spatial_mlp_grad_mean=np.mean(mlp_grad) if mlp_grad else 0))
  pd.DataFrame(hist).to_csv(out/'epoch_metrics.csv',index=False);pd.DataFrame(gate_rows).to_csv(out/'spatial_gates.csv',index=False)
  if improved:
   best_epoch=ep;best_score=sv;torch.save(dict(model=model.state_dict(),epoch=ep,configuration=a.config,seed=a.seed,s_val=sv,config=cfg),out/'best.pt')
  stop_epoch=ep
  if should_stop and not a.short:reason=why;break
 torch.save(dict(model=model.state_dict(),epoch=stop_epoch,configuration=a.config,seed=a.seed,config=cfg),out/'stopped.pt')
 ck=torch.load(out/'best.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model'],strict=True);va,vy,vp=evaluate(model,store,'valid',valid,dev)
 pd.DataFrame(dict(patient_id=[r.patient_id for r in valid],label=vy,death_risk=vp,configuration=a.config,seed=a.seed,checkpoint_epoch=best_epoch)).to_csv(out/'validation_predictions.csv',index=False)
 status=dict(status='COMPLETE',best_epoch=best_epoch,stop_epoch=stop_epoch,stop_reason=reason,best_s_val=best_score,peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20,elapsed_seconds=time.time()-t0,test_evaluated=False)
 (out/'run_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
if __name__=='__main__':main()
