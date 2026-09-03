#!/usr/bin/env python3
import json,sys,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
sys.path.insert(0,str(Path(__file__).parent))
from stage1c_model import Stage1CDenseModel
from v6_p2_pipeline_repair import load
ROOT=Path('/root/autodl-tmp/surgen_hemicore');BASE=ROOT/'outputs/thesis_predictor_first_v6';OUT=BASE/'p2e_formal_validation';DEV=torch.device('cuda');KS=[0,1,2,4,8,16,32,48,64];SEEDS=[1,17,42];MASK_SEEDS=[7101,7102,7103,7104,7105];A=.2629993315;B=-.2729875987
def met(y,p):
 au=roc_auc_score(y,p);ap=average_precision_score(y,p);return au,ap,.5*(au+ap),brier_score_loss(y,p)
def masks(n,k,seed,M=1):
 g=torch.Generator(device=DEV).manual_seed(seed);m=torch.zeros(n,M,64,dtype=torch.bool,device=DEV)
 for i in range(n):
  for j in range(M):
   if k:m[i,j,torch.randperm(64,generator=g,device=DEV)[:k]]=True
 return m
def frozen_heads():
 hs=[]
 for s in SEEDS:
  ck=torch.load(BASE/f'checkpoints/final_c3/seed{s}_best.pt',map_location='cpu',weights_only=False);q=Stage1CDenseModel(pool='gated_attention_pool',relation=False,spatial=False);q.load_state_dict(ck['model']);hs.append(q.head.to(DEV).eval())
 return hs
class LegalObservedModel(nn.Module):
 def __init__(self,heads):
  super().__init__();d=384;self.coarse=nn.Sequential(nn.Linear(2053,d),nn.GELU(),nn.Dropout(.3));self.gscore=nn.Linear(d,1);self.gh=nn.Linear(d,1);self.cond=nn.Sequential(nn.Linear(2*d+1,128),nn.GELU(),nn.Linear(128,1));self.heads=nn.ModuleList(heads)
  for p in self.heads.parameters():p.requires_grad=False
 def forward(self,coarse,fine,mask):
  # The signature contains no teacher output, label, ranking, or full-evidence representation.
  b,M,r=mask.shape;cg=self.coarse(coarse);gw=torch.softmax(self.gscore(cg).squeeze(-1).float(),-1);g=(cg*gw.unsqueeze(-1).to(cg.dtype)).sum(1);zg=self.gh(g).squeeze(-1);reps=[];ps=[]
  for x,h in zip(fine,self.heads):
   selected=x[:,None].expand(b,M,r,384)*mask.unsqueeze(-1);rep=selected.sum(2)/mask.sum(-1,keepdim=True).clamp_min(1);reps.append(rep);ps.append(torch.softmax(h(rep).float(),-1)[...,1])
  p=torch.stack(ps).mean(0).clamp(1e-6,1-1e-6);zl=A*torch.logit(p)+B;lr=torch.stack(reps).mean(0);rho=mask.sum(-1).float()/64.;adj=torch.tanh(self.cond(torch.cat([g[:,None].expand(b,M,-1),lr,rho.unsqueeze(-1)],-1)).squeeze(-1));w=(rho+rho*(1-rho)*adj).clamp(0,1);zf=zg[:,None]+w*(zl-zg[:,None]);zf=torch.where(rho==0,zg[:,None],zf);return zf,zg,zl,w
def endpoint(zf,mask,teacher):
 # Endpoint policy is outside forward. Teacher is never visible to the model or w(x,k).
 full=mask.all(-1);return torch.where(full,torch.logit(teacher.clamp(1e-6,1-1e-6))[:,None].expand_as(zf),zf)
def leakage(model,c,fs,teacher):
 model.eval();rows=[];k=8;m=masks(4,k,8111);zf,_,_,w=model(c[:4], [x[:4] for x in fs],m);base=zf.detach();changed=[]
 for x in fs:
  q=x[:4].clone();un=(~m[:,0]).unsqueeze(-1).expand_as(q);q[un]=q.flip(1)[un];changed.append(q)
 z2,_,_,w2=model(c[:4],changed,m);e1=float((base-z2).abs().max());rows.append({'test':'unselected_token_permutation','max_abs_error':e1,'changed':e1>1e-7,'pass':e1<1e-7})
 selected=[]
 for x in fs:
  q=x[:4].clone();sel=m[:,0].unsqueeze(-1).expand_as(q);q[sel]=(q[sel]+.25);selected.append(q)
 z3,_,_,_=model(c[:4],selected,m);d=float((base-z3).abs().max());rows.append({'test':'selected_token_perturbation','max_abs_error':d,'changed':d>1e-7,'pass':d>1e-7})
 m0=masks(4,0,1);a=model(c[:4],[x[:4] for x in fs],m0)[0];b=model(c[:4],[torch.randn_like(x[:4]) for x in fs],m0)[0];e0=float((a-b).abs().max());rows.append({'test':'k0_all_fine_permutation','max_abs_error':e0,'changed':e0>1e-7,'pass':e0<1e-7})
 m64=masks(4,64,1);raw=model(c[:4],[x[:4] for x in fs],m64)[0];anch=endpoint(raw,m64,teacher[:4]);e64=float((torch.sigmoid(anch[:,0])-teacher[:4]).abs().max());rows.append({'test':'k64_teacher_endpoint','max_abs_error':e64,'changed':False,'pass':e64<1e-7})
 ew=float((w-w2).abs().max());rows.append({'test':'conditional_weight_legal_observation','max_abs_error':ew,'changed':ew>1e-7,'pass':ew<1e-7})
 return rows
def evaluate(model,c,fs,meta,teacher,seed):
 model.eval();pred=[];curves=[];rob=[]
 with torch.no_grad():
  for k in KS:
   for ms in MASK_SEEDS:
    m=masks(len(meta),k,ms);zf,zg,zl,w=model(c,fs,m);zf=endpoint(zf,m,teacher);pf=torch.sigmoid(zf[:,0]).cpu().numpy();pg=torch.sigmoid(zg).cpu().numpy();pl=torch.sigmoid(zl[:,0]).cpu().numpy();ww=w[:,0].cpu().numpy();au,ap,sv,br=met(meta.label,pf);rob.append({'training_seed':seed,'mask_seed':ms,'k':k,'auroc':au,'auprc':ap,'s_val':sv,'brier':br})
    for pid,y,a,b,d,e in zip(meta.patient_id,meta.label,pf,pg,pl,ww):pred.append({'patient_id':pid,'label':y,'training_seed':seed,'mask_seed':ms,'k':k,'p_fused':a,'p_global':b,'p_local':d,'weight':e})
 return pred,rob
def main():
 for d in ['docs','metrics','predictions','checkpoints','logs','configs']: (OUT/d).mkdir(parents=True,exist_ok=True)
 trc,_,trm=load('train',17);vac,_,vam=load('validation',17);trfs=[load('train',s)[1].float() for s in SEEDS];vfs=[load('validation',s)[1].float() for s in SEEDS];tt=torch.tensor(trm.teacher_risk.values,device=DEV).float();vt=torch.tensor(vam.teacher_risk.values,device=DEV).float()
 torch.manual_seed(17);audit=leakage(LegalObservedModel(frozen_heads()).to(DEV),vac.float(),vfs,vt);pd.DataFrame(audit).to_csv(OUT/'metrics/information_leakage_tests.csv',index=False)
 if not all(x['pass'] for x in audit):
  (OUT/'docs/v6_p2e_information_boundary_audit.md').write_text('# P2-E信息边界审计\n\nNO-GO，至少一项反事实泄漏测试失败；未运行三seed。\n');print(json.dumps(audit,indent=2));return
 (OUT/'docs/v6_p2e_information_boundary_audit.md').write_text('# P2-E信息边界审计\n\nGO。模型forward仅接收coarse、当前fine张量与选择mask；未选择token先乘bool mask后才聚合。教师概率只用于loss右侧及模型外k=64端点替换，不进入forward或条件权重。五项反事实测试全部通过，详见metrics/information_leakage_tests.csv。\n')
 allpred=[];allrob=[]
 for seed in SEEDS:
  torch.manual_seed(seed);model=LegalObservedModel(frozen_heads()).to(DEV);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=5e-4);best=-1;bestep=0;beststate=None;hist=[]
  for ep in range(1,121):
   model.train();ls=[]
   for a in range(0,len(trm),16):
    ix=slice(a,min(a+16,len(trm)));m=torch.cat([masks(min(16,len(trm)-a),k,seed+ep*100+k,M=4) for k in KS],1);zf,zg,zl,w=model(trc[ix].float(),[x[ix] for x in trfs],m);y=torch.tensor(trm.label.values[ix],device=DEV).float()[:,None].expand_as(zf);rho=m.sum(-1).float()/64.;target=tt[ix,None].expand_as(zf);sup=F.binary_cross_entropy_with_logits(zf,y);dist=(rho*F.binary_cross_entropy_with_logits(zf,target,reduction='none')).mean();loss=sup+.30*dist;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();ls.append(float(loss.detach()))
   model.eval();
   with torch.no_grad():
    sv=[]
    for ms in MASK_SEEDS:
     m=masks(len(vam),32,ms);z=model(vac.float(),vfs,m)[0];p=torch.sigmoid(z[:,0]).cpu().numpy();sv.append(met(vam.label,p)[2])
   score=float(np.mean(sv));hist.append({'epoch':ep,'loss':np.mean(ls),'validation_k32_s_val':score})
   if score>best+1e-12:best=score;bestep=ep;beststate={k:v.cpu() for k,v in model.state_dict().items()};torch.save({'model':beststate,'seed':seed,'best_epoch':ep,'information_boundary':'legal'},OUT/f'checkpoints/seed{seed}_best.pt')
   if ep>=30 and ep-bestep>=20:break
  torch.save({'model':model.state_dict(),'seed':seed,'stop_epoch':ep},OUT/f'checkpoints/seed{seed}_stopped.pt');pd.DataFrame(hist).to_csv(OUT/f'metrics/seed{seed}_training_curve.csv',index=False);model.load_state_dict(beststate);p,r=evaluate(model,vac.float(),vfs,vam,vt,seed);allpred+=p;allrob+=r
 df=pd.DataFrame(allpred);rb=pd.DataFrame(allrob);df.to_csv(OUT/'predictions/validation_multiseed_predictions.csv',index=False);rb.to_csv(OUT/'metrics/mask_seed_robustness.csv',index=False)
 curves=[]
 for (s,k),g in rb.groupby(['training_seed','k']):curves.append({'training_seed':s,'k':k,'auroc_mean':g.auroc.mean(),'auroc_sd_mask':g.auroc.std(),'auprc_mean':g.auprc.mean(),'auprc_sd_mask':g.auprc.std(),'s_val_mean':g.s_val.mean(),'s_val_sd_mask':g.s_val.std(),'brier_mean':g.brier.mean()})
 cv=pd.DataFrame(curves);cv.to_csv(OUT/'metrics/multiseed_evidence_curves.csv',index=False)
 end=[]
 for s,g in df[df.k==64].groupby('training_seed'):
  q=g.groupby('patient_id').first().reset_index();mae=float(np.mean(np.abs(q.p_fused-vam.teacher_risk.values)));au,ap,sv,br=met(q.label,q.p_fused);end.append({'training_seed':s,'mae_to_teacher':mae,'s_val':sv,'teacher_s_val':met(vam.label,vam.teacher_risk.values)[2]})
 pd.DataFrame(end).to_csv(OUT/'metrics/teacher_endpoint_consistency.csv',index=False)
 # Basic gates before OOF.
 gates=[]
 for s in SEEDS:
  q=cv[cv.training_seed==s];base=float(q[q.k==0].s_val_mean.iloc[0]);gain=float(q[q.k.isin([8,16,32])].s_val_mean.max()-base);gates.append({'seed':s,'partial_gain':gain,'nonconstant':bool(df[df.training_seed==s].groupby(['k','mask_seed']).p_fused.nunique().min()>=20)})
 mid=cv[cv.k.isin([8,16,32])];perseed=pd.DataFrame(gates);sd64=float(cv[cv.k==64].s_val_mean.std());endpoint=pd.DataFrame(end);weight_mid=df[df.k.isin([8,16,32])].weight
 basic=all(perseed.nonconstant) and (endpoint.mae_to_teacher<=.02).all() and ((endpoint.s_val-endpoint.teacher_s_val)>=-.01).all() and int((perseed.partial_gain>=.005).sum())>=2 and sd64<=.035 and weight_mid.std()>1e-5
 pd.DataFrame([{'basic_gate_pass':basic,'seeds_with_partial_gain':int((perseed.partial_gain>=.005).sum()),'k64_s_val_sd':sd64,'mid_weight_std':weight_mid.std()}]).to_csv(OUT/'metrics/basic_gate_decision.csv',index=False)
 if basic:
  # OOF is deliberately not fabricated: current compact fine tokens are from full-Train C3 encoders and violate unseen-patient OOF.
  pd.DataFrame([{'status':'BLOCKED','reason':'compact fine tokens were produced by full-Train C3 checkpoints; genuine unseen-patient OOF requires fold-specific fine tokens and heads'}]).to_csv(OUT/'metrics/partial_evidence_oof.csv',index=False);pd.DataFrame([{'status':'NOT_RUN_OOF_INFORMATION_BOUNDARY'}]).to_csv(OUT/'metrics/evidence_conditional_calibration.csv',index=False);pd.DataFrame([{'status':'NOT_GENERATED'}]).to_csv(OUT/'predictions/train_partial_oof_predictions.csv',index=False)
 else:
  pd.DataFrame([{'status':'NOT_RUN_BASIC_GATE_FAILED'}]).to_csv(OUT/'metrics/partial_evidence_oof.csv',index=False);pd.DataFrame([{'status':'NOT_RUN_BASIC_GATE_FAILED'}]).to_csv(OUT/'metrics/evidence_conditional_calibration.csv',index=False);pd.DataFrame([{'status':'NOT_GENERATED'}]).to_csv(OUT/'predictions/train_partial_oof_predictions.csv',index=False)
 (OUT/'docs/v6_p2e_formal_report.md').write_text(f'# P2-E正式三seed验证\n\n信息边界审计通过，完成seed 1/17/42与五个固定Validation mask seed。基础门槛={basic}。部分证据OOF状态见metrics/partial_evidence_oof.csv。未读取Test，未启动PPO、MoE、STOP或SG-SurvNet。\n')
 (OUT/'docs/v6_p2e_reproducible_commands.sh').write_text('#!/usr/bin/env bash\ncd /root/autodl-tmp/surgen_hemicore/src/MSRLSelect\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python v6_p2e.py\n')
 (OUT/'configs/p2e_config.json').write_text(json.dumps({'seeds':SEEDS,'mask_seeds':MASK_SEEDS,'k':KS,'test_access':False,'teacher_in_forward':False},indent=2));print(json.dumps({'basic_gate':basic,'seeds_with_gain':int((perseed.partial_gain>=.005).sum()),'sd64':sd64}))
if __name__=='__main__':main()
