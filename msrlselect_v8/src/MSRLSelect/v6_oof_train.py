#!/usr/bin/env python3
import argparse,json,math,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from torch import nn
from stage1c_model import Stage1CDenseModel
from stage1c_train import Store,batchify,EarlyStop,lr_for_epoch,seed_all

def metrics(y,p,loss=None):
 return {'auc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'brier':brier_score_loss(y,p),'loss':loss}

@torch.no_grad()
def evaluate(model,store,rows,dev,split='train'):
 model.eval();ys=[];ps=[];loss=[]
 for st in range(0,len(rows),16):
  batch=rows[st:st+16];X,PM,C,S,RM,Y=batchify(store,split,batch,dev)
  with torch.autocast('cuda',dtype=torch.bfloat16):o=model(X,PM,C,S,RM);l=nn.functional.cross_entropy(o,Y)
  ys.extend(Y.cpu().tolist());ps.extend(o.softmax(1)[:,1].float().cpu().tolist());loss.append(float(l))
 return metrics(ys,ps,float(np.mean(loss))),ys,ps

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--epochs',type=int,default=120);a=ap.parse_args()
 seed=6100+a.fold;seed_all(seed);data=a.root/'prepared/sr386_full';all_rows=list(pd.read_csv(data/'surgen_train_10.csv').itertuples(index=False));y=np.array([r.label for r in all_rows])
 outer=list(StratifiedKFold(5,shuffle=True,random_state=6100).split(np.arange(len(y)),y));fit_idx,hold_idx=outer[a.fold]
 inner=StratifiedShuffleSplit(1,test_size=.15,random_state=6200+a.fold);sub,sel=next(inner.split(fit_idx,y[fit_idx]));train_idx=fit_idx[sub];select_idx=fit_idx[sel]
 train=[all_rows[i] for i in train_idx];select=[all_rows[i] for i in select_idx];hold=[all_rows[i] for i in hold_idx]
 assert set(r.patient_id for r in train).isdisjoint(r.patient_id for r in select);assert set(r.patient_id for r in fit_idx if False)==set()
 assert set(r.patient_id for r in hold).isdisjoint([r.patient_id for r in train+select])
 out=a.out/f'fold{a.fold}';out.mkdir(parents=True,exist_ok=True);cfg={'phase':'P1','fold':a.fold,'outer_seed':6100,'training_seed':seed,'train_n':len(train),'checkpoint_selection_n':len(select),'oof_n':len(hold),'architecture':'C3_K64_gated_no_relation_no_spatial','test_access':False,'max_epochs':a.epochs,'min_stop_epoch':30,'patience':20,'warmup_epochs':5,'base_lr':1e-4,'weight_decay':1e-4,'dropout':.2,'gradient_clip':1.0,'precision':'BF16','batch_size':16}
 (out/'config.json').write_text(json.dumps(cfg,indent=2));pd.DataFrame({'patient_id':[r.patient_id for r in hold],'label':[r.label for r in hold]}).to_csv(out/'oof_members.csv',index=False)
 dev=torch.device('cuda');store=Store(a.root,'k64');m=Stage1CDenseModel(pool='gated_attention_pool',spatial=False,relation=False).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);es=EarlyStop(30,20);hist=[];best=-1.;best_ep=0;reason='max_epochs';torch.cuda.reset_peak_memory_stats();t0=time.time()
 s0,_,_=evaluate(m,store,select,dev);hist.append({'epoch':0,'lr':0,'train_loss':np.nan,'train_auc':np.nan,'train_auprc':np.nan,'train_brier':np.nan,'selection_loss':s0['loss'],'selection_auc':s0['auc'],'selection_auprc':s0['auprc'],'selection_brier':s0['brier'],'s_val':.5*(s0['auc']+s0['auprc'])})
 for ep in range(1,a.epochs+1):
  lr=lr_for_epoch(ep,a.epochs);[g.update(lr=lr) for g in opt.param_groups];m.train();rng=np.random.default_rng(seed+ep);order=rng.permutation(len(train));ys=[];ps=[];losses=[]
  for st in range(0,len(train),16):
   batch=[train[i] for i in order[st:st+16]];X,PM,C,S,RM,Y=batchify(store,'train',batch,dev);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):o=m(X,PM,C,S,RM);loss=nn.functional.cross_entropy(o,Y)
   loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();ys.extend(Y.cpu().tolist());ps.extend(o.softmax(1)[:,1].detach().float().cpu().tolist());losses.append(float(loss.detach()))
  tr=metrics(ys,ps,float(np.mean(losses)));va,_,_=evaluate(m,store,select,dev);sv=.5*(va['auc']+va['auprc']);improved,stop,why=es.update(ep,sv)
  hist.append({'epoch':ep,'lr':lr,'train_loss':tr['loss'],'train_auc':tr['auc'],'train_auprc':tr['auprc'],'train_brier':tr['brier'],'selection_loss':va['loss'],'selection_auc':va['auc'],'selection_auprc':va['auprc'],'selection_brier':va['brier'],'s_val':sv});pd.DataFrame(hist).to_csv(out/'epoch_metrics.csv',index=False)
  if improved:best=sv;best_ep=ep;torch.save({'model':m.state_dict(),'epoch':ep,'fold':a.fold,'config':cfg},out/'best.pt')
  if stop:reason=why;break
 stop_ep=ep;torch.save({'model':m.state_dict(),'epoch':ep,'fold':a.fold,'config':cfg},out/'stopped.pt');ck=torch.load(out/'best.pt',map_location=dev,weights_only=False);m.load_state_dict(ck['model'],strict=True);met,yy,pp=evaluate(m,store,hold,dev)
 pd.DataFrame({'patient_id':[r.patient_id for r in hold],'label':yy,'death_risk':pp,'fold':a.fold,'checkpoint_epoch':best_ep}).to_csv(out/'oof_predictions.csv',index=False)
 status={'status':'COMPLETE','fold':a.fold,'best_epoch':best_ep,'stop_epoch':stop_ep,'stop_reason':reason,'oof_metrics':met,'elapsed_seconds':time.time()-t0,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/2**20,'test_evaluated':False};(out/'run_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
if __name__=='__main__':main()
