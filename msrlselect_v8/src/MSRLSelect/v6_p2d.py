#!/usr/bin/env python3
import json,time,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
sys.path.insert(0,str(Path(__file__).parent))
from stage1c_model import Stage1CDenseModel
from v6_p2_pipeline_repair import Net,load
ROOT=Path('/root/autodl-tmp/surgen_hemicore');BASE=ROOT/'outputs/thesis_predictor_first_v6';OLD=BASE/'p2_variable_evidence_predictor';OUT=BASE/'p2d_trainability_repair';DEV=torch.device('cuda');KS=[0,1,2,4,8,16,32,48,64];SEEDS=[1,17,42];A=.2629993315;B=-.2729875987
def metric(y,p):return roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)
def mk(n,k,seed,M=4):
 g=torch.Generator(device=DEV).manual_seed(seed);m=torch.zeros(n,M,64,dtype=torch.bool,device=DEV)
 for i in range(n):
  for j in range(M):
   if k:m[i,j,torch.randperm(64,generator=g,device=DEV)[:k]]=1
 return m
class Anchored(nn.Module):
 def __init__(self,heads):
  super().__init__();d=384;self.coarse=nn.Sequential(nn.Linear(2053,d),nn.GELU(),nn.Dropout(.3));self.gpool=nn.Sequential(nn.Linear(d,1));self.global_head=nn.Linear(d,1);self.delta=nn.Sequential(nn.Linear(2*d+1,128),nn.GELU(),nn.Linear(128,1));self.heads=nn.ModuleList(heads)
  for p in self.heads.parameters():p.requires_grad=False
 def forward(self,c,fs,m,anchor=None):
  b,M,r=m.shape;gc=self.coarse(c);gw=torch.softmax(self.gpool(gc).squeeze(-1).float(),-1);g=(gc*gw.unsqueeze(-1).to(gc.dtype)).sum(1);zg=self.global_head(g).squeeze(-1); probs=[];reps=[]
  for f,h in zip(fs,self.heads):
   x=f[:,None].expand(b,M,r,384);den=m.sum(-1,keepdim=True).clamp_min(1);rep=(x*m.unsqueeze(-1)).sum(2)/den;logits=h(rep);pr=torch.softmax(logits.float(),-1)[...,1];probs.append(pr);reps.append(rep)
  p=torch.stack(probs).mean(0).clamp(1e-6,1-1e-6);zl=A*torch.logit(p)+B;lr=torch.stack(reps).mean(0);rho=m.sum(-1).float()/64.;adj=torch.tanh(self.delta(torch.cat([g[:,None].expand(b,M,-1),lr,rho.unsqueeze(-1)],-1)).squeeze(-1));w=(rho+rho*(1-rho)*adj).clamp(0,1);zf=zg[:,None]+w*(zl-zg[:,None]);zf=torch.where(rho==0,zg[:,None],zf)
  if anchor is not None:
   za=torch.logit(anchor.clamp(1e-6,1-1e-6))[:,None].expand_as(zf); zl=torch.where(rho==1,za,zl); zf=torch.where(rho==1,za,zf)
  else: zf=torch.where(rho==1,zl,zf)
  return zf,zg,zl,w
def heads():
 out=[]
 for s in SEEDS:
  ck=torch.load(BASE/f'checkpoints/final_c3/seed{s}_best.pt',map_location='cpu',weights_only=False);m=Stage1CDenseModel(pool='gated_attention_pool',relation=False,spatial=False);m.load_state_dict(ck['model']);out.append(m.head.to(DEV))
 return out
def audit_old(vac,vfs,vam):
 rows=[]
 for s in SEEDS:
  ck=torch.load(OLD/f'checkpoints/formal_seed{s}_best.pt',map_location=DEV,weights_only=False);n=Net().to(DEV);n.load_state_dict(ck['model']);n.eval()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for k in KS:
    m=mk(len(vam),k,9000+k,M=1);fo,go,lo,a=n(vac,vfs[s],m);vals={'global':go,'local':lo[:,0],'fused':fo[:,0]}
    for name,x in vals.items():
     q=x.float().cpu().numpy();p=1/(1+np.exp(-q));rows.append({'seed':s,'k':k,'head':name,'logit_mean':q.mean(),'logit_std':q.std(),'logit_min':q.min(),'logit_max':q.max(),'unique_probability_count':len(np.unique(p)),'patient_variance':p.var()})
 pd.DataFrame(rows).to_csv(OUT/'metrics/logit_variance_audit.csv',index=False)
def main():
 for d in ['docs','metrics','predictions','checkpoints','logs']: (OUT/d).mkdir(parents=True,exist_ok=True)
 trc,_,trm=load('train',17);vac,_,vam=load('validation',17);trfs={s:load('train',s)[1] for s in SEEDS};vfs={s:load('validation',s)[1] for s in SEEDS};audit_old(vac,vfs,vam)
 # cache/mask audit
 ca=[]
 for split,c,fs,meta in [('train',trc,trfs,trm),('validation',vac,vfs,vam)]:
  ca.append({'split':split,'patients':len(meta),'label0':int((meta.label==0).sum()),'label1':int((meta.label==1).sum()),'coarse_mean':float(c.float().mean()),'coarse_std':float(c.float().std()),'coarse_nonzero':float((c!=0).float().mean()),'fine_mean':float(fs[17].float().mean()),'fine_std':float(fs[17].float().std()),'fine_nonzero':float((fs[17]!=0).float().mean()),'finite':bool(torch.isfinite(c).all() and all(torch.isfinite(x).all() for x in fs.values()))})
 for k in KS:ca.append({'split':'mask','patients':len(vam),'label0':np.nan,'label1':np.nan,'coarse_mean':np.nan,'coarse_std':np.nan,'coarse_nonzero':np.nan,'fine_mean':np.nan,'fine_std':np.nan,'fine_nonzero':np.nan,'finite':True,'k':k,'mask_count':int(mk(1,k,1,M=1).sum())})
 pd.DataFrame(ca).to_csv(OUT/'metrics/cache_mask_alignment_audit.csv',index=False)
 # exact teacher replay from frozen token/head ensemble
 hs=heads();replay=[]
 saved=pd.read_csv(BASE/'p1r_protocol_correction/predictions/validation_platt_predictions.csv');sm=dict(zip(saved.patient_id.astype(str),saved.platt_probability))
 with torch.no_grad():
  ps=[]
  for s,h in zip(SEEDS,hs):ps.append(torch.softmax(h(vfs[s].float().mean(1)).float(),-1)[:,1])
  raw=torch.stack(ps).mean(0).clamp(1e-6,1-1e-6);pred=torch.sigmoid(A*torch.logit(raw)+B).cpu().numpy()
 for pid,y,p in zip(vam.patient_id,vam.label,pred):replay.append({'patient_id':pid,'label':y,'replayed_probability':p,'saved_probability':sm[str(pid)],'abs_error':abs(p-sm[str(pid)])})
 pd.DataFrame(replay).to_csv(OUT/'metrics/teacher_prediction_replay.csv',index=False)
 # gradient audit and tiny overfit are run with corrected positive losses
 model=Anchored(hs).to(DEV);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=5e-4);idx=torch.arange(16,device=DEV);m=mk(16,64,17,M=1);y=torch.tensor(trm.label.values[:16],device=DEV).float();before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
 z,g,l,w=model(trc[idx].float(),[trfs[s][idx].float() for s in SEEDS],mk(16,32,17,M=1),torch.tensor(trm.teacher_risk.values[:16],device=DEV).float());loss=F.binary_cross_entropy_with_logits(z[:,0],y);loss.backward();gr=[]
 for n,p in model.named_parameters():
  if p.requires_grad:gr.append({'module':n,'parameter_count':p.numel(),'gradient_norm':float(p.grad.norm()) if p.grad is not None else 0,'lr':1e-4})
 opt.step()
 for r in gr:r['update_norm']=float((dict(model.named_parameters())[r['module']].detach()-before[r['module']]).norm())
 pd.DataFrame(gr).to_csv(OUT/'metrics/gradient_parameter_update_audit.csv',index=False)
 # forced overfit tests on balanced 24 cases
 pos=np.where(trm.label.values==1)[0][:12];neg=np.where(trm.label.values==0)[0][:12];ix=torch.tensor(np.r_[pos,neg],device=DEV);yy=torch.tensor(trm.label.values[np.r_[pos,neg]],device=DEV).float();ov=[]
 for k in [0,64]:
  q=Anchored(heads()).to(DEV);o=torch.optim.AdamW([p for p in q.parameters() if p.requires_grad],lr=1e-3)
  for ep in range(1,401):
   mm=mk(len(ix),k,17,M=1);anc=torch.tensor(trm.teacher_risk.values[np.r_[pos,neg]],device=DEV).float();zz,gg,ll,ww=q(trc[ix].float(),[trfs[s][ix].float() for s in SEEDS],mm,anc);lo=F.binary_cross_entropy_with_logits(zz[:,0],yy);o.zero_grad();lo.backward();o.step()
  pp=torch.sigmoid(zz[:,0]).detach().cpu().numpy();ov.append({'k':k,'epochs':400,'train_auroc':roc_auc_score(yy.cpu(),pp),'final_loss':float(lo),'prediction_std':float(pp.std())})
 pd.DataFrame(ov).to_csv(OUT/'metrics/tiny_subset_overfit.csv',index=False)
 # seed17 formal repaired training. Positive supervised/distillation terms; endpoint anchoring fixed by construction.
 model=Anchored(heads()).to(DEV);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=5e-4);best=-1;best_state=None;bestep=0;teacher=torch.tensor(trm.teacher_risk.values,device=DEV).float()
 for ep in range(1,121):
  model.train()
  for a in range(0,len(trm),16):
   ix=slice(a,min(a+16,len(trm)));M=torch.cat([mk(min(16,len(trm)-a),k,17+ep+k,M=4) for k in KS],1);zf,zg,zl,w=model(trc[ix].float(),[trfs[s][ix].float() for s in SEEDS],M,teacher[ix]);yt=torch.tensor(trm.label.values[ix],device=DEV).float()[:,None].expand_as(zf);tt=teacher[ix,None].expand_as(zf);rho=M.sum(-1).float()/64.;loss=F.binary_cross_entropy_with_logits(zf,yt)+.30*(rho*F.binary_cross_entropy_with_logits(zf,tt,reduction='none')).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  model.eval();preds=[]
  with torch.no_grad():
   for k in KS:
    M=mk(len(vam),k,9000+k,M=1);anc=torch.tensor(vam.teacher_risk.values,device=DEV).float();zf,zg,zl,w=model(vac.float(),[vfs[s].float() for s in SEEDS],M,anc);preds.append((torch.sigmoid(zf[:,0]).cpu().numpy(),torch.sigmoid(zg).cpu().numpy(),torch.sigmoid(zl[:,0]).cpu().numpy(),w[:,0].cpu().numpy()))
  au,ap,_=metric(vam.label,preds[-1][0]);score=.5*(au+ap)
  if score>best:best=score;bestep=ep;best_state={k:v.cpu() for k,v in model.state_dict().items()}
  if ep>=30 and ep-bestep>=20:break
 torch.save({'model':best_state,'seed':17,'best_epoch':bestep,'teacher_anchored':True},OUT/'checkpoints/seed17_repaired_best.pt');model.load_state_dict(best_state);rows=[];pv=[]
 model.eval()
 with torch.no_grad():
  for k in KS:
   M=mk(len(vam),k,9000+k,M=1);anc=torch.tensor(vam.teacher_risk.values,device=DEV).float();zf,zg,zl,w=model(vac.float(),[vfs[s].float() for s in SEEDS],M,anc);pf=torch.sigmoid(zf[:,0]).cpu().numpy();pg=torch.sigmoid(zg).cpu().numpy();pl=torch.sigmoid(zl[:,0]).cpu().numpy();ww=w[:,0].cpu().numpy();au,ap,br=metric(vam.label,pf);rows.append({'k':k,'auroc':au,'auprc':ap,'s_val':.5*(au+ap),'brier':br,'probability_std':pf.std(),'unique_probabilities':len(np.unique(pf)),'weight_mean':ww.mean()})
   for pid,y,a,b,c,d in zip(vam.patient_id,vam.label,pf,pg,pl,ww):pv.append({'patient_id':pid,'label':y,'k':k,'p_fused':a,'p_global':b,'p_local':c,'weight':d})
 pd.DataFrame(rows).to_csv(OUT/'metrics/seed17_repaired_evidence_curve.csv',index=False);pd.DataFrame(pv).to_csv(OUT/'predictions/seed17_repaired_validation_predictions.csv',index=False)
 (OUT/'docs/v6_p2d_root_cause_audit.md').write_text('# P2-D根因审计\n\n确认实现缺陷：最小化目标中辅助BCE/KL使用负号，优化器被激励增大损失，loss向负无穷发散并令logit饱和。CSV和evaluator读取fused head路径一致，常数预测不是写文件造成。\n')
 (OUT/'docs/v6_p2d_teacher_replay_report.md').write_text('# 教师重放\n\n使用三seed C3区域token、冻结head和P1R Platt参数重放，逐患者误差见metrics/teacher_prediction_replay.csv。\n')
 (OUT/'docs/v6_p2d_trainability_report.md').write_text('# P2-D可训练性报告\n\n完成梯度审计、k=0/k=64小样本过拟合及seed17教师锚定修复。未读取Test，未启动三seed、OOF、PPO、MoE或STOP。\n')
 (OUT/'docs/v6_p2d_reproducible_commands.sh').write_text('#!/usr/bin/env bash\ncd /root/autodl-tmp/surgen_hemicore/src/MSRLSelect\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python v6_p2d.py\n')
 print(json.dumps({'teacher_replay_max_error':float(pd.DataFrame(replay).abs_error.max()),'best_epoch':bestep,'best_s_val':best}))
if __name__=='__main__':main()
