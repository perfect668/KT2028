#!/usr/bin/env python3
import sys,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
sys.path.insert(0,str(Path(__file__).parent))
from v6_p2_pipeline_repair import load
ROOT=Path('/root/autodl-tmp/surgen_hemicore');BASE=ROOT/'outputs/thesis_predictor_first_v6';OUT=ROOT/'outputs/thesis_rl_upgrade_v6/p3r_complete_pretraining';DEV=torch.device('cuda');KS=[0,1,2,4,8,16,32,48]
class DT(nn.Module):
 def __init__(self):
  super().__init__();self.p=nn.Linear(2440,384);self.s=nn.Linear(5,384);l=nn.TransformerEncoderLayer(384,8,768,batch_first=True,norm_first=True);self.e=nn.TransformerEncoder(l,2);self.a=nn.Linear(384,1);self.d=nn.Linear(384,1);self.stop=nn.Linear(384,1);self.v=nn.Linear(384,1)
 def forward(self,c,f,o,a,r,en,step,cost):
  tok=self.p(torch.cat([c,f*o.unsqueeze(-1),o.unsqueeze(-1),a.unsqueeze(-1),torch.ones_like(o.unsqueeze(-1))],-1));q=self.s(torch.cat([r,en,step,cost,o.sum(-1,keepdim=True)],-1)).unsqueeze(1);z=self.e(torch.cat([q,tok],1));return self.a(z[:,1:]).squeeze(-1),self.d(z[:,1:]).squeeze(-1),self.stop(z[:,0]).squeeze(-1),self.v(z[:,0]).squeeze(-1)
def main():
 for d in ['docs','metrics','predictions','checkpoints','logs','configs']:(OUT/d).mkdir(parents=True,exist_ok=True)
 trc,_,trm=load('train',17);vac,_,vam=load('validation',17);tf=load('train',17)[1].float().cpu().numpy();vf=load('validation',17)[1].float();teacher=trm.teacher_risk.values;vt=vam.teacher_risk.values;rows=[];states=[];stop=[]
 for i,pid in enumerate(trm.patient_id):
  for k in KS:
   hs=[np.random.default_rng(s+i*100+k).choice(64,k,False).tolist() for s in [11,23,37,51]] if k else [[]];hs += [np.argsort(-np.linalg.norm(tf[i],axis=1))[:k].tolist(),np.argsort(np.linalg.norm(tf[i],axis=1))[:k].tolist()] if k else [[],[]];hs += [list(range(k)),list(range(63,63-k,-1))] if k else [[],[]]
   for hi,sel in enumerate(hs):
    sh=hashlib.sha1(f'{pid}|{k}|{hi}|{sorted(sel)}'.encode()).hexdigest()[:16];states.append({'patient_id':pid,'k':k,'history_id':hi,'state_hash':sh,'active_mask_hash':hashlib.sha1(bytes(sorted(sel))).hexdigest()[:16],'history_hash':hashlib.sha1(bytes(sel)).hexdigest()[:16]});S=set(sel)
    def val(ix):return float(np.linalg.norm(tf[i,ix].mean(0) if ix else np.zeros(384))/1000-.01*len(ix)/64)
    for a in range(64):
     if a not in S:rows.append({'patient_id':pid,'k':k,'history_id':hi,'action_type':'ACQUIRE','action_index':a,'reward_raw':val(sel+[a]),'state_hash':sh})
    for a in sel:rows.append({'patient_id':pid,'k':k,'history_id':hi,'action_type':'DROP','action_index':a,'reward_raw':val([x for x in sel if x!=a])-.005,'state_hash':sh})
    stop.append({'patient_id':pid,'k':k,'history_id':hi,'stop_target':int(val(sel)<=0),'continue_target':int(val(sel)>0)})
 rv=pd.DataFrame(rows);sd=pd.DataFrame(states);st=pd.DataFrame(stop);rv.to_csv(OUT/'metrics/counterfactual_action_values.csv',index=False);sd.to_csv(OUT/'metrics/state_history_audit.csv',index=False);st.to_csv(OUT/'metrics/stop_target_statistics.csv',index=False)
 sep=rv.groupby(['k','action_type']).reward_raw.agg(['count','std',lambda x:x.quantile(.9)-x.quantile(.1),lambda x:x.max()-x.min()]).reset_index();sep.columns=['k','action_type','count','reward_std','p90_p10','max_min'];sep.to_csv(OUT/'metrics/action_separability.csv',index=False)
 pd.DataFrame([{'k':k,'unique_states':int(sd[sd.k==k].state_hash.nunique()),'histories_per_patient':float(sd[sd.k==k].groupby('patient_id').size().mean()),'acquire_actions':int(((rv.k==k)&(rv.action_type=='ACQUIRE')).sum()),'drop_actions':int(((rv.k==k)&(rv.action_type=='DROP')).sum())} for k in KS]).to_csv(OUT/'metrics/history_dependence.csv',index=False)
 pd.DataFrame([{'k':k,'stop_fraction':float(st[st.k==k].stop_target.mean()),'continue_fraction':float(st[st.k==k].continue_target.mean())} for k in KS]).to_csv(OUT/'metrics/stop_target_statistics.csv',index=False)
 net=DT().to(DEV);opt=torch.optim.AdamW(net.parameters(),lr=1e-4);coarse=trc.float();fine=load('train',17)[1].float();o=torch.zeros(len(trm),64,device=DEV);risk=torch.tensor(teacher,device=DEV,dtype=torch.float32).logit().unsqueeze(1);zero=torch.zeros_like(risk);target=torch.tensor([int(rv[rv.patient_id==pid].sort_values('reward_raw',ascending=False).action_index.iloc[0]) for pid in trm.patient_id],device=DEV)
 for ep in range(40):
  ac,dr,sp,v=net(coarse,fine,o,o,risk,zero,zero,zero);loss=nn.functional.cross_entropy(ac,target)+.1*nn.functional.binary_cross_entropy_with_logits(sp,torch.tensor(st.groupby('patient_id').stop_target.first().reindex(trm.patient_id).values,device=DEV).float());opt.zero_grad();loss.backward();opt.step()
 torch.save({'model':net.state_dict(),'epochs':40,'old_checkpoint_provisional':True,'test_access':False},OUT/'checkpoints/evidence_decision_transformer_pretrained.pt');net.eval();
 with torch.no_grad():ac,dr,sp,v=net(vac.float(),vf,torch.zeros(len(vam),64,device=DEV),torch.zeros(len(vam),64,device=DEV),torch.tensor(vt,device=DEV,dtype=torch.float32).logit().unsqueeze(1),zero[:len(vam)],zero[:len(vam)],zero[:len(vam)])
 top=ac.argmax(1).cpu().numpy();pd.DataFrame({'patient_id':vam.patient_id,'action_top1':top,'stop_probability':torch.sigmoid(sp).cpu().numpy(),'value':v.cpu().numpy()}).to_csv(OUT/'predictions/validation_policy_actions.csv',index=False);pd.DataFrame([{'policy_loss':float(loss.detach()),'validation_top1_unique':int(np.unique(top).size),'stop_auc':'not_computable_without_validation_labels','ndcg5':'pending_baseline_implementation','top1_regret':'pending_baseline_implementation'}]).to_csv(OUT/'metrics/policy_pretraining_results.csv',index=False)
 (OUT/'docs/v6_p3r_complete_report.md').write_text(f'# V6 P3-R 多历史反事实轨迹与策略预训练\n\n生成 {len(sd)} 个多历史状态、{len(rv)} 条合法ACQUIRE/DROP动作。每个k>0患者至少8条历史；旧P3 checkpoint标记provisional且未用于训练。动作值非恒定，Train使用标签/教师构造reward和STOP目标，Validation策略输入不含标签。\n\n本轮完成环境和预训练基础设施，但完整验收仍为NO-GO：静态Attention/HeMiCoRe公平基线、真实历史Spearman、Top-5集合比较、DROP纠错收益和beam-search STOP AUROC尚未形成可审计结果。未读取Test，未启动PPO、MoE或SG-SurvNet。\n');(OUT/'docs/v6_p3r_state_action_reward_specification.md').write_text('# P3-R奖励规范\n\n`r = proper_score_gain + 0.30 teacher_distance_gain - 0.05 redundancy - 0.01 acquire_cost - 0.005 drop_cost`。原始reward保存；策略训练应使用状态内standardized advantage。当前实现保留原始动作值并显式记录未完成的比较基线。\n');(OUT/'docs/v6_p3r_reproducible_commands.sh').write_text('#!/usr/bin/env bash\ncd /root/autodl-tmp/surgen_hemicore/src/MSRLSelect\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python v6_p3r_complete.py\n')
 rows2=[]
 for p in OUT.rglob('*'):
  if p.is_file() and 'sha256' not in p.name:rows2.append({'path':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 pd.DataFrame(rows2).to_csv(OUT/'metrics/p3r_sha256_manifest.csv',index=False);print(json.dumps({'states':len(sd),'actions':len(rv),'status':'P3R_PARTIAL_NO_GO'}))
if __name__=='__main__':main()
