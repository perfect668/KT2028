#!/usr/bin/env python3
import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).parent))
from stage1c_model import Stage1CDenseModel
from stage1c_train import Store,batchify
from v6_p2_pipeline_repair import load
from v6_p2e import LegalObservedModel,masks,endpoint,met,ROOT,BASE,OUT,DEV,KS,SEEDS,MASK_SEEDS,A,B
def fold_assets(fold,rows):
 store=Store(ROOT,'k64');heads=[];tokens=[];probs=[]
 for s in SEEDS:
  ck=torch.load(BASE/f'p1r_protocol_correction/checkpoints/fold{fold}_seed{s}/best.pt',map_location=DEV,weights_only=False);m=Stage1CDenseModel(pool='gated_attention_pool',relation=False,spatial=False).to(DEV);m.load_state_dict(ck['model']);m.eval();tt=[];pp=[]
  with torch.no_grad():
   for a in range(0,len(rows),16):
    X,PM,C,S,RM,Y=batchify(store,'train',rows[a:a+16],DEV)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     b,r,p,d=X.shape;z=m.proj(X.reshape(b*r,p,d));pooled,_=m.pool(z,PM.reshape(b*r,p));pooled=pooled.reshape(b,r,-1);o=m(X,PM,C,S,RM)
    tt.append(pooled.float());pp.extend(o.softmax(1)[:,1].float().cpu().tolist())
  heads.append(m.head);tokens.append(torch.cat(tt));probs.append(np.asarray(pp))
 raw=np.mean(probs,0).clip(1e-6,1-1e-6);teacher=1/(1+np.exp(-(A*np.log(raw/(1-raw))+B)));return heads,tokens,torch.tensor(teacher,device=DEV).float()
def fit_fold(fold,train_idx,hold_idx,coarse,meta,rows):
 hs,fs,teacher=fold_assets(fold,rows);outputs=[]
 for seed in SEEDS:
  torch.manual_seed(seed+fold*1000);m=LegalObservedModel(hs).to(DEV);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=5e-4);best=1e9;state=None
  ti=torch.tensor(train_idx,device=DEV);hi=torch.tensor(hold_idx,device=DEV)
  for ep in range(1,31):
   m.train();order=np.random.default_rng(seed+fold*1000+ep).permutation(len(train_idx))
   for a in range(0,len(order),16):
    ids=ti[torch.tensor(order[a:a+16],device=DEV)];M=torch.cat([masks(len(ids),k,seed+ep*100+k,M=4) for k in KS],1);zf,zg,zl,w=m(coarse[ids].float(),[x[ids] for x in fs],M);y=torch.tensor(meta.label.values,device=DEV).float()[ids,None].expand_as(zf);rho=M.sum(-1).float()/64.;target=teacher[ids,None].expand_as(zf);loss=torch.nn.functional.binary_cross_entropy_with_logits(zf,y)+.30*(rho*torch.nn.functional.binary_cross_entropy_with_logits(zf,target,reduction='none')).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  torch.save({'model':m.state_dict(),'fold':fold,'seed':seed,'epoch':30},OUT/f'checkpoints/oof_fold{fold}_seed{seed}.pt');m.eval()
  with torch.no_grad():
   for k in KS:
    for ms in MASK_SEEDS:
     M=masks(len(hold_idx),k,ms);zf,zg,zl,w=m(coarse[hi].float(),[x[hi] for x in fs],M);zf=endpoint(zf,M,teacher[hi]);p=torch.sigmoid(zf[:,0]).cpu().numpy()
     for idx,pr in zip(hold_idx,p):outputs.append({'patient_id':meta.patient_id.iloc[idx],'label':meta.label.iloc[idx],'outer_fold':fold,'training_seed':seed,'mask_seed':ms,'k':k,'probability':pr})
 return outputs
def ece(y,p):
 z=0.;edges=np.linspace(0,1,11)
 for i in range(10):
  q=(p>=edges[i])&(p<(edges[i+1] if i<9 else 1.01))
  if q.any():z+=q.mean()*abs(p[q].mean()-y[q].mean())
 return z
def apply(p,k,x):
 rho=k/64.;z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)));return 1/(1+np.exp(-((x[0]+x[1]*rho)*z+x[2]+x[3]*rho)))
def main():
 cv=pd.read_csv(OUT/'metrics/multiseed_evidence_curves.csv');pred=pd.read_csv(OUT/'predictions/validation_multiseed_predictions.csv');teacher=pd.read_csv(BASE/'p1r_protocol_correction/predictions/validation_platt_predictions.csv')[['patient_id','platt_probability']];p64=pred[pred.k==64].merge(teacher,on='patient_id');end=[]
 for s,g in p64.groupby('training_seed'):
  q=g.groupby('patient_id').first();end.append({'training_seed':s,'mae_to_teacher':float(np.mean(np.abs(q.p_fused-q.platt_probability))),'s_val':met(q.label,q.p_fused)[2],'teacher_s_val':met(q.label,q.platt_probability)[2]})
 pd.DataFrame(end).to_csv(OUT/'metrics/teacher_endpoint_consistency.csv',index=False);base=cv[cv.k==0].set_index('training_seed').s_val_mean;g=[]
 for s in SEEDS:g.append(float(cv[(cv.training_seed==s)&cv.k.isin([8,16,32])].s_val_mean.max()-base[s]))
 basic=(pd.DataFrame(end).mae_to_teacher<=.02).all() and sum(x>=.005 for x in g)>=2 and cv[cv.k==64].s_val_mean.std()<=.035
 if not basic:print('BASIC FAIL');return
 coarse,_,meta=load('train',17);rows=list(pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv').itertuples(index=False));y=meta.label.values;splits=list(StratifiedKFold(5,shuffle=True,random_state=6100).split(np.arange(len(y)),y));out=[]
 for f,(tr,ho) in enumerate(splits):out+=fit_fold(f,tr,ho,coarse,meta,rows)
 raw=pd.DataFrame(out);raw.to_csv(OUT/'predictions/train_partial_oof_predictions.csv',index=False);ens=raw.groupby(['patient_id','label','outer_fold','k']).probability.mean().reset_index();cal=[];pars=[]
 for f in range(5):
  fit=ens.outer_fold!=f;hold=~fit
  def obj(x):return log_loss(ens.loc[fit,'label'],np.array([apply(p,k,x) for p,k in zip(ens.loc[fit,'probability'],ens.loc[fit,'k'])]),labels=[0,1])
  z=minimize(obj,[1,0,0,0],method='BFGS').x;pars.append(z)
  q=ens[hold].copy();q['calibrated']=[apply(p,k,z) for p,k in zip(q.probability,q.k)];cal.append(q)
 cf=pd.concat(cal);metrics=[]
 for k,q in cf.groupby('k'):
  au,ap,sv,br=met(q.label,q.probability);acu,apu,svu,bru=met(q.label,q.calibrated);metrics.append({'k':k,'auroc':au,'auprc':ap,'brier_before':br,'ece_before':ece(q.label.values,q.probability.values),'brier_after':bru,'ece_after':ece(q.label.values,q.calibrated.values),'a0_mean':np.mean(pars,0)[0],'a1_mean':np.mean(pars,0)[1],'b0_mean':np.mean(pars,0)[2],'b1_mean':np.mean(pars,0)[3]})
 pd.DataFrame(metrics).to_csv(OUT/'metrics/partial_evidence_oof.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'metrics/evidence_conditional_calibration.csv',index=False)
 (OUT/'docs/v6_p2e_formal_report.md').write_text('# P2-E正式验证\n\n信息边界与三seed基础门槛通过。完成5-fold×3-seed未见患者部分证据OOF及按患者分组的四参数证据比例条件校准。未读取Test，未启动PPO、MoE、STOP或SG-SurvNet。\n');print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__':main()
