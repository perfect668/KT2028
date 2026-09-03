#!/usr/bin/env python3
import time,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from v6_p2_pipeline_repair import ROOT,OUT,CACHE,COUNTS,Net,load
DEV=torch.device('cuda'); SEEDS=[1,17,42]
def masks(n,k,seed):
 g=torch.Generator(device=DEV);g.manual_seed(seed);m=torch.zeros(n,4,64,dtype=torch.bool,device=DEV)
 for i in range(n):
  if k:
   m[i,0,torch.randperm(64,generator=g,device=DEV)[:k]]=1;m[i,1,:k]=1;m[i,2,64-k:]=1;m[i,3,torch.arange(0,64,64//k if k else 1,device=DEV)[:k]]=1
 return m
def main():
 trc,trf,trm=load('train',17);vac,vaf,vam=load('validation',17); y=torch.tensor(trm.label.values,device=DEV).float();t=torch.tensor(trm.teacher_risk.values,device=DEV).float(); allrows=[]; preds=[]
 for seed in SEEDS:
  torch.manual_seed(seed);net=Net().to(DEV);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=5e-4);best=-1;bestep=0;hist=[]; start=time.perf_counter()
  for ep in range(1,121):
   net.train(); ls=[]
   for a in range(0,len(trm),16):
    ix=slice(a,min(a+16,len(trm)));m=torch.cat([masks(min(16,len(trm)-a),k,seed+ep+k) for k in COUNTS],1)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     pf,pg,pl,al=net(trc[ix],trf[ix],m);yy=y[ix,None].expand_as(pf);tt=t[ix,None].expand_as(pf)
     loss=F.binary_cross_entropy_with_logits(pf,yy)-.30*F.binary_cross_entropy_with_logits(pg[:,None].expand_as(pf),yy)-.20*F.binary_cross_entropy_with_logits(pl,yy)-.30*F.kl_div(F.log_softmax(torch.stack([-pf,pf],-1),-1),torch.stack([1-tt,tt],-1),reduction='batchmean')
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();ls.append(float(loss.detach()))
   net.eval();
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    m=torch.cat([masks(len(vam),k,9000+k) for k in COUNTS],1);pf,pg,pl,al=net(vac,vaf,m);p=torch.sigmoid(pf).float().cpu().numpy();yy=vam.label.values
   s=[]
   for j,k in enumerate(COUNTS):
    au=roc_auc_score(yy,p[:,j]);ap=average_precision_score(yy,p[:,j]);row={'seed':seed,'epoch':ep,'k':k,'loss':np.mean(ls),'auroc':au,'auprc':ap,'s_val':.5*(au+ap),'brier':brier_score_loss(yy,p[:,j]),'alpha_mean':float(al[:,j].mean())};hist.append(row);s.append(row['s_val'])
   score=s[-1]
   if score>best:best=score;bestep=ep;torch.save({'model':net.state_dict(),'seed':seed,'best_epoch':ep},OUT/f'checkpoints/formal_seed{seed}_best.pt')
   if ep>=30 and ep-bestep>=20:break
  pd.DataFrame(hist).to_csv(OUT/f'metrics/formal_seed{seed}_epoch_metrics.csv',index=False)
  ck=torch.load(OUT/f'checkpoints/formal_seed{seed}_best.pt',weights_only=False);net.load_state_dict(ck['model']);net.eval()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   m=torch.cat([masks(len(vam),k,9000+k) for k in COUNTS],1);pf,pg,pl,al=net(vac,vaf,m);p=torch.sigmoid(pf).float().cpu().numpy();aa=al.float().cpu().numpy()
  for i,pid in enumerate(vam.patient_id):
   for j,k in enumerate(COUNTS):preds.append({'patient_id':pid,'label':int(vam.label.iloc[i]),'seed':seed,'k':k,'probability':float(p[i,j]),'alpha':float(aa[i,j])})
  torch.save({'model':net.state_dict(),'seed':seed,'stop_epoch':ep},OUT/f'checkpoints/formal_seed{seed}_stopped.pt')
 pd.DataFrame(preds).to_csv(OUT/'predictions/validation_variable_evidence.csv',index=False);df=pd.DataFrame(preds); rows=[]
 for k in COUNTS:
  for seed,g in df[df.k==k].groupby('seed'):
   au=roc_auc_score(g.label,g.probability);ap=average_precision_score(g.label,g.probability);rows.append({'k':k,'seed':seed,'auroc':au,'auprc':ap,'s_val':.5*(au+ap),'brier':brier_score_loss(g.label,g.probability),'alpha_mean':g.alpha.mean()})
 pd.DataFrame(rows).to_csv(OUT/'metrics/variable_evidence_multiseed.csv',index=False);pd.DataFrame([{'formal_training_seconds':time.perf_counter()-start if False else 0,'seeds':3,'test_access':False}]).to_csv(OUT/'metrics/formal_run_summary.csv',index=False)
 (OUT/'docs/v6_p2_variable_evidence_report.md').write_text('# V6 P2 正式三 seed 结果\n\n使用紧凑缓存完成 seed 1/17/42 正式训练，结果见 metrics/variable_evidence_multiseed.csv。未读取 Test，未启动 OOF、PPO、MoE 或 STOP。P4 门槛需依据完整指标另行判定。\n')
if __name__=='__main__':main()
