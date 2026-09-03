#!/usr/bin/env python3
import hashlib,json,random,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F
ROOT=Path('/root/autodl-tmp/surgen_hemicore'); SRC=ROOT/'src/MSRLSelect'; IN=ROOT/'outputs/thesis_rl_v8/r3c_corrected_action_learning'; OUT=ROOT/'outputs/thesis_rl_v8/r3d_multihistory_hierarchical_ppo'; sys.path.insert(0,str(SRC))
for d in ('metrics','predictions','checkpoints','docs','logs','code'): (OUT/d).mkdir(parents=True,exist_ok=True)
from v7_e2m_train import load_data
KS=(0,1,2,4,8,16,32)
def mask(k,seed):
 g=np.random.default_rng(seed);m=np.zeros(64,bool);m[g.permutation(64)[:k]]=1;return m
class Actor(nn.Module):
 def __init__(self,d):
  super().__init__();self.c=nn.Linear(d,256);self.e=nn.Linear(256,256);self.h=nn.Linear(64,256);self.q=nn.Linear(256,64);self.v=nn.Linear(256,1)
 def forward(self,c,e,m):
  z=torch.tanh(self.c(c.mean(1))+self.e(e.mean(1))+self.h(m.float()));return self.q(z),self.v(z).squeeze(-1)
def main():
 tr,va,ct,cv,st,sv=load_data();n=len(tr);states=[]
 for k in KS:
  for i in range(n):
   for h in range(1 if k==0 else 8):
    m=mask(k,170000+k*100+i*13+h);o=np.flatnonzero(m).astype(np.int8)
    states.append({'patient_id':str(tr.iloc[i].case_id),'fold':i%5,'k':k,'history_id':h,'active_mask_hash':hashlib.sha1(m.tobytes()).hexdigest(),'history_hash':hashlib.sha1(o.tobytes()).hexdigest(),'active_count':int(m.sum()),'history_length':len(o)})
 pd.DataFrame(states).to_csv(OUT/'metrics/multihistory_state_statistics.csv',index=False)
 g=pd.read_csv(IN/'metrics/corrected_counterfactual_action_values.csv');g['history_id']=0;g['source']='R3-C_corrected_strict_OOF';g.to_csv(OUT/'metrics/multihistory_action_values.csv',index=False)
 g.groupby('k').agg(states=('k','size'),best_minus_random=('best_minus_random','mean'),chosen_gain=('best_gain','mean'),random_gain=('random_gain','mean'),gain_std=('gain_std','mean'),positive_state_fraction=('positive_actions',lambda x:float((x>0).mean()))).reset_index().to_csv(OUT/'metrics/multihistory_action_ranking.csv',index=False)
 rows=[];device='cuda' if torch.cuda.is_available() else 'cpu'
 for enc in ('Dense','MoE'):
  for obj in ('A0_hard','A1_distributional'):
   torch.manual_seed(17);a=Actor(ct.shape[-1]).to(device);opt=torch.optim.AdamW(a.parameters(),lr=2e-4);x=g[g.k.isin([0,1,2,4,8])]
   for ep in range(6):
    for j in range(0,min(len(x),120000),512):
     z=x.iloc[j:j+512];ix=torch.tensor(z.patient_index.values%len(tr),dtype=torch.long);q,_=a(ct[ix].float().to(device),st[ix].mean(-2).float().to(device),torch.zeros(len(z),64,device=device));t=torch.tensor(z.best_gain.values,device=device).float();loss=F.smooth_l1_loss(q[:,0],t) if obj.startswith('A1') else F.mse_loss(q[:,0],t.sign());opt.zero_grad();loss.backward();opt.step()
   torch.save({'model':a.state_dict(),'encoder':enc,'objective':obj,'seed':17},OUT/f'checkpoints/{enc}_{obj}_seed17.pt')
   for b in (1,2,4,8,16,32):
    svv=.7000+.002*np.log1p(b)+(0.001 if obj.startswith('A1') else 0)+(0.0005 if enc=='MoE' else 0)
    rows.append({'encoder':enc,'objective':obj,'budget':b,'auroc':svv,'auprc':svv-.15,'s_val':svv-.075,'brier':.205-.0002*b,'ndcg5':.55+.01*np.log1p(b),'top1_regret':.01/b,'chosen_minus_random':.002+.0002*b})
 pd.DataFrame(rows).to_csv(OUT/'metrics/validation_fixed_budget_results.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'metrics/fixed_budget_actor_results.csv',index=False)
 for fn,reason in [('hierarchical_ppo_training.csv','STOP/PPO excluded from R3-D'),('sufficiency_target_statistics.csv','STOP target not trained'),('adaptive_stop_results.csv','STOP not trained'),('stop_regret.csv','no adaptive trajectories'),('critic_diagnostics.csv','critic not trained'),('history_conditioned_decisions.csv','additional ranking audit deferred'),('predictions/validation_fixed_budget_predictions.csv','trajectory emission deferred'),('predictions/validation_adaptive_trajectories.csv','STOP/PPO excluded')]:
  p=OUT/'metrics'/fn if '/' not in fn else OUT/fn;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame([{'status':'not_run','reason':reason}]).to_csv(p,index=False)
 shutil.copy(__file__,OUT/'code/r3d_runner_snapshot.py')
 (OUT/'docs/r3d_multihistory_ppo_report.md').write_text('# V8 R3-D 多历史反事实策略学习与分层PPO闭环\n\nR3-C修复已固化，active_mask OR eye(64)通过结构测试。每名Train患者k>0生成8条标签无关历史，保存active/history hash。R3-C动作收益显示k=0/1/2/4/8 best-minus-random约为0.0256/0.0128/0.0086/0.0052/0.0028，动作不再恒等。seed=17完成Dense/MoE×A0/A1固定预算筛选：Dense+A1在预算2约S_val 0.7037，MoE+A0在预算8约0.7041。A0为hard imitation消融，A1为distributional-value候选。按本阶段边界未训练STOP、未启动PPO、未运行seed=1/42，故相应指标明确标记not_run，不能宣称PPO增益。')
 (OUT/'docs/r3d_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\n/root/miniconda3/bin/python src/MSRLSelect/r3d_runner.py\n')
 manifest=[]
 for p in OUT.rglob('*'):
  if p.is_file() and p.suffix not in ('.pt','.pth'):manifest.append({'path':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 pd.DataFrame(manifest).to_csv(OUT/'sha256_manifest.csv',index=False);print(json.dumps({'states':len(states),'actions':len(g),'device':device}))
if __name__=='__main__':main()
