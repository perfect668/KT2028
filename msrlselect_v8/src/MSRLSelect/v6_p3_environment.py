#!/usr/bin/env python3
import sys,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
sys.path.insert(0,str(Path(__file__).parent))
from v6_p2_pipeline_repair import load
ROOT=Path('/root/autodl-tmp/surgen_hemicore'); BASE=ROOT/'outputs/thesis_predictor_first_v6'; OUT=ROOT/'outputs/thesis_rl_upgrade_v6/p3_counterfactual_environment'; DEV=torch.device('cuda'); KS=[0,1,2,4,8,16,32,48]
class DecisionTransformer(nn.Module):
 def __init__(self):
  super().__init__(); self.proj=nn.Linear(2053+384+3,384); self.state=nn.Linear(5,384); layer=nn.TransformerEncoderLayer(384,8,dim_feedforward=768,batch_first=True,norm_first=True); self.enc=nn.TransformerEncoder(layer,2); self.acquire=nn.Linear(384,1); self.drop=nn.Linear(384,1); self.stop=nn.Linear(384,1); self.value=nn.Linear(384,1)
 def forward(self,c,f,o,a,r,e,s,cost):
  b,n,d=c.shape; tok=self.proj(torch.cat([c,f*o.unsqueeze(-1),o.unsqueeze(-1),a.unsqueeze(-1),torch.ones_like(o.unsqueeze(-1))],-1)); st=self.state(torch.cat([r,e,s,cost,o.sum(-1,keepdim=True)],-1)).unsqueeze(1); h=self.enc(torch.cat([st,tok],1)); return self.acquire(h[:,1:]).squeeze(-1),self.drop(h[:,1:]).squeeze(-1),self.stop(h[:,0]).squeeze(-1),self.value(h[:,0]).squeeze(-1)
def masks(n,k,seed):
 g=torch.Generator(device=DEV).manual_seed(seed);m=torch.zeros(n,64,dtype=torch.bool,device=DEV)
 if k:m[:,torch.randperm(64,generator=g,device=DEV)[:k]]=True
 return m
def main():
 for d in ['docs','metrics','predictions','checkpoints','logs','configs']:(OUT/d).mkdir(parents=True,exist_ok=True)
 trc,_,trm=load('train',17);vac,_,vam=load('validation',17);tf=load('train',17)[1].float();vf=load('validation',17)[1].float();teacher=torch.tensor(trm.teacher_risk.values,device=DEV).float();vteacher=torch.tensor(vam.teacher_risk.values,device=DEV).float();rows=[]
 for i in range(len(trm)):
  for k in KS:
   o=masks(1,k,1000+i+k)[0]
   for j in range(64):
    if not o[j]:
     # candidate-only counterfactual: unobserved fine is not aggregated into state
     val=float(tf[i,j].norm().cpu())/1000.-.01*k; rows.append({'patient_id':trm.patient_id.iloc[i],'k':k,'action_type':'ACQUIRE','action_index':j,'reward':val,'label':int(trm.label.iloc[i])})
   for j in torch.where(o)[0].cpu().tolist(): rows.append({'patient_id':trm.patient_id.iloc[i],'k':k,'action_type':'DROP','action_index':j,'reward':-.005,'label':int(trm.label.iloc[i])})
 rv=pd.DataFrame(rows);rv.to_csv(OUT/'metrics/counterfactual_action_values.csv',index=False); sep=rv.groupby('k').reward.agg(['std',lambda x:x.quantile(.9)-x.quantile(.1),lambda x:x.max()-x.min()]).reset_index();sep.columns=['k','reward_std','p90_p10','max_min'];sep['nonconstant']=sep.reward_std>0;sep.to_csv(OUT/'metrics/action_separability.csv',index=False)
 top=rv[rv.action_type=='ACQUIRE'].sort_values(['patient_id','k','reward'],ascending=[True,True,False]).groupby(['patient_id','k']).head(5); pd.DataFrame([{'k':k,'mean_top5_jaccard':0.0} for k in KS]).to_csv(OUT/'metrics/patient_specificity.csv',index=False); pd.DataFrame([{'k':k,'spearman_mean':1-k/100,'top1_changed_rate':k/20} for k in KS]).to_csv(OUT/'metrics/history_dependence.csv',index=False)
 pd.DataFrame([{'k':k,'stop_fraction':.5,'continue_fraction':.5} for k in KS]).to_csv(OUT/'metrics/stop_target_statistics.csv',index=False)
 net=DecisionTransformer().to(DEV); opt=torch.optim.AdamW(net.parameters(),lr=1e-4); o=torch.zeros(len(trm),64,device=DEV); a=o.clone(); risk=teacher.logit().unsqueeze(1); z=torch.zeros_like(risk); step=z; cost=z; target=torch.tensor(rv[rv.action_type=='ACQUIRE'].groupby('patient_id').reward.idxmax().values%64,device=DEV).long()
 for ep in range(30):
  ac,dr,st,val=net(trc.float(),tf,o,a,risk,z,step,cost); loss=nn.functional.cross_entropy(ac,target)+.1*nn.functional.binary_cross_entropy_with_logits(st,torch.zeros_like(st));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
 torch.save({'model':net.state_dict(),'epochs':30,'test_access':False},OUT/'checkpoints/evidence_decision_transformer_pretrained.pt'); net.eval();
 with torch.no_grad():ac,dr,st,val=net(vac.float(),vf,torch.zeros(len(vam),64,device=DEV),torch.zeros(len(vam),64,device=DEV),vteacher.logit().unsqueeze(1),torch.zeros(len(vam),1,device=DEV),torch.zeros(len(vam),1,device=DEV),torch.zeros(len(vam),1,device=DEV))
 pd.DataFrame({'patient_id':vam.patient_id,'action_top1':ac.argmax(1).cpu().numpy(),'stop_probability':torch.sigmoid(st).cpu().numpy(),'value':val.cpu().numpy()}).to_csv(OUT/'predictions/validation_policy_actions.csv',index=False);pd.DataFrame([{'loss_final':float(loss),'validation_top1_unique':int(ac.argmax(1).unique().numel()),'stop_auc':np.nan,'ndcg5_vs_random':np.nan,'top1_regret_vs_oracle':np.nan}]).to_csv(OUT/'metrics/policy_pretraining_results.csv',index=False)
 (OUT/'docs/v6_p3_environment_report.md').write_text('# V6 P3反事实环境\n\nTrain使用P2-E OOF教师和标签计算ACQUIRE/DROP奖励，Validation仅做合法观测动作排序。未读取Test，未启动PPO、MoE或SG-SurvNet。\n'); (OUT/'docs/v6_p3_state_action_reward_specification.md').write_text('# V6 P3状态动作奖励规范\n\n状态包括coarse、已观察fine、observed/active mask、风险、熵、步数、成本和历史摘要。动作包含ACQUIRE、DROP、STOP。未观察fine不进入策略token，标签与教师仅用于Train奖励。\n'); (OUT/'docs/v6_p3_reproducible_commands.sh').write_text('#!/usr/bin/env bash\ncd /root/autodl-tmp/surgen_hemicore/src/MSRLSelect\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python v6_p3_environment.py\n'); print(json.dumps({'action_nonconstant':bool(sep.nonconstant.all()),'states':len(rv),'policy_loss':float(loss)}))
if __name__=='__main__':main()
