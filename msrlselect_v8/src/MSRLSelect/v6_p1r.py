#!/usr/bin/env python3
import argparse, json, math, random, hashlib, csv, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from scipy.optimize import minimize
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from stage1c_model import Stage1CDenseModel
from stage1c_train import Store, batchify, seed_all

ROOT=Path('/root/autodl-tmp/surgen_hemicore'); OUT=ROOT/'outputs/thesis_predictor_first_v6/p1r_protocol_correction'; DATA=ROOT/'prepared/sr386_full'; DEV=torch.device('cuda')
EPOCHS={1:19,17:13,42:22}; SEEDS=[1,17,42]
def met(y,p): return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'nll':float(log_loss(y,p,labels=[0,1]))}
def ece(y,p,bins=10):
 y=np.asarray(y);p=np.asarray(p);e=np.linspace(0,1,bins+1);z=0.
 for i in range(bins):
  q=(p>=e[i])&(p<(e[i+1] if i<bins-1 else e[i+1]+1e-12))
  if q.any():z+=q.mean()*abs(p[q].mean()-y[q].mean())
 return float(z)
def adaptive_ece(y,p):
 order=np.argsort(p); chunks=np.array_split(order, min(10,len(p))); return float(sum(len(ix)/len(p)*abs(np.mean(p[ix])-np.mean(np.asarray(y)[ix])) for ix in chunks if len(ix)))
def logits(p): return np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6))
def platt_fit(y,p):
 z=logits(p); y=np.asarray(y)
 def fun(ab):
  q=1/(1+np.exp(-(ab[0]*z+ab[1]))); return log_loss(y,q,labels=[0,1])
 r=minimize(fun,[1.,0.],method='BFGS'); return float(r.x[0]),float(r.x[1])
def apply(p,a,b): return 1/(1+np.exp(-(a*logits(p)+b)))
@torch.no_grad()
def infer(model,store,rows,split):
 model.eval();ys=[];ps=[]
 for st in range(0,len(rows),16):
  batch=rows[st:st+16];X,PM,C,S,RM,Y=batchify(store,split,batch,DEV)
  with torch.autocast('cuda',dtype=torch.bfloat16):o=model(X,PM,C,S,RM)
  ys += Y.cpu().tolist(); ps += o.softmax(1)[:,1].float().cpu().tolist()
 return np.array(ys),np.array(ps)
def train_fold(train,fold,seed,epochs,out):
 seed_all(seed+fold*10000);m=Stage1CDenseModel(pool='gated_attention_pool',spatial=False,relation=False).to(DEV);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);store=Store(ROOT,'k64');hist=[]
 for ep in range(1,epochs+1):
  m.train();order=np.random.default_rng(seed+ep+fold).permutation(len(train));ls=[]
  for st in range(0,len(train),16):
   batch=[train[i] for i in order[st:st+16]];X,PM,C,S,RM,Y=batchify(store,'train',batch,DEV);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):o=m(X,PM,C,S,RM);loss=nn.functional.cross_entropy(o,Y)
   loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();ls.append(float(loss.detach()))
  hist.append({'epoch':ep,'train_loss':np.mean(ls)})
 d=out/f'fold{fold}_seed{seed}';d.mkdir(parents=True,exist_ok=True);torch.save({'model':m.state_dict(),'fold':fold,'seed':seed,'epochs':epochs},d/'best.pt');pd.DataFrame(hist).to_csv(d/'epoch_metrics.csv',index=False);return d
def bootstrap_metrics(y,p,y2,p2,n=2000,seed=777):
 rng=np.random.default_rng(seed);y=np.asarray(y);p=np.asarray(p);y2=np.asarray(y2);p2=np.asarray(p2);idx0=np.where(y==0)[0];idx1=np.where(y==1)[0];vals=[]
 for _ in range(n):
  ix=np.r_[rng.choice(idx0,len(idx0),True),rng.choice(idx1,len(idx1),True)];vals.append([roc_auc_score(y[ix],p[ix]),average_precision_score(y[ix],p[ix]),roc_auc_score(y[ix],p[ix])-roc_auc_score(y2[ix],p2[ix]),average_precision_score(y[ix],p[ix])-average_precision_score(y2[ix],p2[ix])])
 return np.asarray(vals)
def main():
 for x in ['docs','metrics','predictions','figures','checkpoints','configs','logs']: (OUT/x).mkdir(parents=True,exist_ok=True)
 train=list(pd.read_csv(DATA/'surgen_train_10.csv').itertuples(index=False));valid=list(pd.read_csv(DATA/'surgen_validate_10.csv').itertuples(index=False));y=np.array([r.label for r in train]); skf=StratifiedKFold(5,shuffle=True,random_state=6100); splits=list(skf.split(np.arange(len(y)),y));store=Store(ROOT,'k64');oof=np.zeros((len(train),3));vpred=np.zeros((len(valid),3));rows=[]
 for fi,(fit,hold_idx) in enumerate(splits):
  holdrows=[train[i] for i in hold_idx];
  for si,seed in enumerate(SEEDS):
   d=train_fold([train[i] for i in fit],fi,seed,EPOCHS[seed],OUT/'checkpoints');m=Stage1CDenseModel(pool='gated_attention_pool',spatial=False,relation=False).to(DEV);ck=torch.load(d/'best.pt',map_location=DEV,weights_only=False);m.load_state_dict(ck['model'],strict=True);yh,ph=infer(m,store,holdrows,'train');oof[hold_idx,si]=ph;yv,pv=infer(m,store,valid,'valid');vpred[:,si]=pv;pd.DataFrame({'patient_id':[r.patient_id for r in holdrows],'label':yh,'probability':ph,'fold':fi,'seed':seed}).to_csv(d/'holdout_predictions.csv',index=False);rows.append({'fold':fi,'seed':seed,'epochs':EPOCHS[seed],'holdout_n':len(hold_idx),'holdout_auroc':roc_auc_score(yh,ph),'holdout_auprc':average_precision_score(yh,ph)})
 # Save ensemble patient predictions.
 oe=oof.mean(1);ve=vpred.mean(1);op=pd.DataFrame({'patient_id':[r.patient_id for r in train],'label':y,'seed1_probability':oof[:,0],'seed17_probability':oof[:,1],'seed42_probability':oof[:,2],'ensemble_probability':oe});vp=pd.DataFrame({'patient_id':[r.patient_id for r in valid],'label':[r.label for r in valid],'seed1_probability':vpred[:,0],'seed17_probability':vpred[:,1],'seed42_probability':vpred[:,2],'ensemble_probability':ve});op.to_csv(OUT/'predictions/train_oof_ensemble_predictions.csv',index=False);vp.to_csv(OUT/'predictions/validation_ensemble_predictions.csv',index=False)
 # Platt cross-fitting on OOF ensemble; final fit on all OOF.
 cal=np.zeros(len(op)); folds=np.zeros(len(op),int)
 for fi,(_,hold) in enumerate(splits):
  fit=np.setdiff1d(np.arange(len(op)),hold);a,b=platt_fit(y[fit],oe[fit]);cal[hold]=apply(oe[hold],a,b);folds[hold]=fi
 a,b=platt_fit(y,oe);vc=apply(ve,a,b);op['platt_oof_probability']=cal;op['platt_fold']=folds;op.to_csv(OUT/'predictions/train_oof_ensemble_predictions.csv',index=False);vp['platt_probability']=vc;vp.to_csv(OUT/'predictions/validation_platt_predictions.csv',index=False)
 const=float(y.mean());metric_rows=[]
 for scope,yy,raw,pl in [('train_oof',y,oe,cal),('validation',np.array([r.label for r in valid]),ve,vc)]:
  base=np.full(len(yy),const); 
  for name,p in [('constant_train_prevalence',base),('ensemble_uncalibrated',raw),('platt_crossfit' if scope=='train_oof' else 'platt_final',pl)]: metric_rows.append({'scope':scope,'method':name,'auroc':roc_auc_score(yy,p),'auprc':average_precision_score(yy,p),'brier':brier_score_loss(yy,p),'nll':log_loss(yy,p,labels=[0,1]),'ece_10bin':ece(yy,p),'ece_adaptive':adaptive_ece(yy,p),'platt_a':a if 'platt' in name else np.nan,'platt_b':b if 'platt' in name else np.nan})
 pd.DataFrame(metric_rows).to_csv(OUT/'metrics/platt_calibration_metrics.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'metrics/oof_ensemble_multiseed.csv',index=False)
 vals=bootstrap_metrics(y,cal,y,np.full(len(y),const)); vb=bootstrap_metrics(np.array([r.label for r in valid]),vc,np.array([r.label for r in valid]),np.full(len(valid),const));
 # Bootstrap P3 on Validation against constant baseline.
 rng=np.random.default_rng(778);yv=np.array([r.label for r in valid]);brows=[];idx0=np.where(yv==0)[0];idx1=np.where(yv==1)[0]
 for i in range(2000):
  ix=np.r_[rng.choice(idx0,len(idx0),True),rng.choice(idx1,len(idx1),True)];bc=brier_score_loss(yv[ix],np.full(len(ix),const));bcal=brier_score_loss(yv[ix],vc[ix]);brows.append({'replicate':i,'brier_improvement':(bc-bcal)/bc,'ece_10bin':ece(yv[ix],vc[ix]),'ece_adaptive':adaptive_ece(yv[ix],vc[ix])})
 boot=pd.DataFrame(brows);boot.to_csv(OUT/'metrics/calibration_bootstrap.csv',index=False)
 gen={'oof_ensemble':met(y,oe),'validation_ensemble':met(yv,ve),'oof_ensemble_bootstrap_auc_ci':[float(np.quantile(vals[:,0],.025)),float(np.quantile(vals[:,0],.975))],'oof_ensemble_bootstrap_auprc_ci':[float(np.quantile(vals[:,1],.025)),float(np.quantile(vals[:,1],.975))],'oof_validation_auc_abs_diff':abs(roc_auc_score(y,oe)-roc_auc_score(yv,ve)),'oof_validation_auprc_diff':average_precision_score(y,oe)-average_precision_score(yv,ve),'five_fold_auroc_mean':float(pd.DataFrame(rows).holdout_auroc.mean()),'five_fold_auroc_std':float(pd.DataFrame(rows).holdout_auroc.std()),'p2_pass':bool(vals[:,0].min()>.5 and average_precision_score(y,oe)>=const+.05 and abs(roc_auc_score(y,oe)-roc_auc_score(yv,ve))<=.1)}
 pd.DataFrame([gen]).to_csv(OUT/'metrics/corrected_generalization_gate.csv',index=False)
 p3={'validation_brier_improvement':float((brier_score_loss(yv,np.full(len(yv),const))-brier_score_loss(yv,vc))/brier_score_loss(yv,np.full(len(yv),const))),'validation_ece':ece(yv,vc),'p3_pass':bool(gen['p2_pass'] and ((brier_score_loss(yv,np.full(len(yv),const))-brier_score_loss(yv,vc))/brier_score_loss(yv,np.full(len(yv),const))>=.05) and ece(yv,vc)<=.1),'platt_a':a,'platt_b':b,'bootstrap_brier_improvement_ci':[float(boot.brier_improvement.quantile(.025)),float(boot.brier_improvement.quantile(.975))],'bootstrap_ece_ci':[float(boot.ece_10bin.quantile(.025)),float(boot.ece_10bin.quantile(.975))]};pd.DataFrame([p3]).to_csv(OUT/'metrics/p3_decision.csv',index=False)
 plt.figure(figsize=(5,5));
 for yy,pp,label,color in [(y,oe,'OOF uncalibrated','C0'),(y,cal,'OOF Platt cross-fit','C1'),(yv,vc,'Validation Platt','C2')]:
  bins=np.linspace(0,1,11);xs=[];ys=[]
  for j in range(10):
   q=(pp>=bins[j])&(pp<(bins[j+1] if j<9 else 1.01));
   if q.any():xs.append(pp[q].mean());ys.append(yy[q].mean())
  plt.plot(xs,ys,'o-',label=label,color=color)
 plt.plot([0,1],[0,1],'k--');plt.xlabel('Mean predicted risk');plt.ylabel('Observed mortality');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(OUT/'figures/reliability_diagram.png',dpi=180);plt.close()
 config={'phase':'P1R','architecture':'C3_K64_gated_no_relation_no_spatial','outer_folds':5,'seeds':SEEDS,'fixed_epochs':EPOCHS,'checkpoint_selection':'Stage1C seed-specific best epoch; no outer holdout selection','platt':'5-fold cross-fitted on OOF ensemble then refit all OOF','bootstrap_replicates':2000,'test_access':False,'next_phase_started':False};(OUT/'configs/p1r_config.json').write_text(json.dumps(config,indent=2));(OUT/'docs/v6_p1r_protocol_correction_report.md').write_text('# MS-RLSurv V6 Phase P1R\n\nP1R执行完成。见 metrics/corrected_generalization_gate.csv 与 metrics/p3_decision.csv。仅使用Train/Validation；Test、P2、MoE、PPO和SG-SurvNet均未启动。\n\nOOF ensemble由5折×3 seed、每个seed固定Stage1C最佳epoch生成。Platt scaling在OOF内部五折cross-fitting，最终参数只在全部OOF拟合后应用Validation。原temperature scaling仅作为历史对照保留。\n\n本报告的PASS/FAIL以原始点估计门槛为准，Bootstrap仅提供不确定性，不替代门槛。\n');print(json.dumps({'p2':gen,'p3':p3},indent=2))
if __name__=='__main__':main()
