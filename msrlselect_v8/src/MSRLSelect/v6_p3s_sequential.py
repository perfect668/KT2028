#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd,torch,json,hashlib
from torch import nn
ROOT=Path('/root/autodl-tmp/surgen_hemicore');SRC=ROOT/'outputs/thesis_rl_upgrade_v6/p3r_complete_pretraining';OUT=ROOT/'outputs/thesis_rl_upgrade_v6/p3s_sequential_return';DEV=torch.device('cuda');COSTS=[0.0]
class Policy(nn.Module):
 def __init__(self):
  super().__init__();self.x=nn.Sequential(nn.Linear(5,128),nn.GELU(),nn.Linear(128,64));self.a=nn.Linear(64,1);self.s=nn.Linear(64,1);self.v=nn.Linear(64,1)
 def forward(self,x):h=self.x(x);return self.a(h).squeeze(-1),self.s(h).squeeze(-1),self.v(h).squeeze(-1)
def main():
 for d in ['docs','metrics','predictions','checkpoints','logs','configs']:(OUT/d).mkdir(parents=True,exist_ok=True)
 av=pd.read_csv(SRC/'metrics/counterfactual_action_values.csv'); states=pd.read_csv(SRC/'metrics/state_history_audit.csv'); meta=pd.read_csv(ROOT/'outputs/thesis_predictor_first_v6/p2e_formal_validation/cache_compact/train/metadata.csv') if False else pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv');
 pos=av.reward_raw[av.reward_raw>0]; q=np.quantile(pos,[.25,.5,.75]); lambdas=[0.,float(q[0]),float(q[1]),float(q[2])]; pd.DataFrame({'lambda_name':['zero','low','mid','high'],'lambda':lambdas,'source':['train_quantile']*4}).to_csv(OUT/'metrics/resource_cost_levels.csv',index=False)
 # canonical active sets are reconstructed from DROP legality in the stored state.
 groups=[]; trajectories=[]; stops=[]
 for (pid,k,hid),g in av.groupby(['patient_id','k','history_id']):
  drop=set(g.loc[g.action_type=='DROP','action_index'].astype(int)); acq=g[g.action_type=='ACQUIRE'];
  for li,lam in enumerate(lambdas):
   beam=[(drop,0.,[])]; seen=set()
   for depth in range(8):
    nxt=[]
    for sel,total,path in beam:
     key=tuple(sorted(sel));
     if key in seen:continue
     seen.add(key); cand=[]
     for _,r in acq.iterrows():
      a=int(r.action_index)
      if a not in sel:cand.append(('ACQUIRE',a,float(r.reward_raw)-lam))
     for a in sorted(sel):cand.append(('DROP',a,-.005))
     cand=sorted(cand,key=lambda x:x[2],reverse=True)[:8]
     for typ,a,r in cand:
      ns=set(sel); ns.add(a) if typ=='ACQUIRE' else ns.discard(a); nxt.append((ns,total+r*.97,path+[(typ,a,r)]))
    beam=sorted(nxt,key=lambda x:x[1],reverse=True)[:8]
   best=beam[0] if beam else (drop,0.,[]); qcont=best[1]; stop=int(qcont<=0);stops.append({'patient_id':pid,'k':k,'history_id':hid,'lambda':lam,'stop_target':stop,'continue_target':1-stop,'q_continue':qcont,'steps':len(best[2])})
   for d,(typ,a,r) in enumerate(best[2]):trajectories.append({'patient_id':pid,'k_start':k,'history_id':hid,'lambda':lam,'depth':d,'action_type':typ,'action_index':a,'reward_net':r,'cumulative_return':sum(x[2]*(.97**j) for j,x in enumerate(best[2][:d+1]))})
   groups.append({'patient_id':pid,'k':k,'history_id':hid,'lambda':lam,'oracle_return':qcont,'oracle_steps':len(best[2]),'start_active':len(drop),'stop':stop})
 st=pd.DataFrame(stops);tr=pd.DataFrame(trajectories);gr=pd.DataFrame(groups);st.to_csv(OUT/'metrics/cost_conditioned_stop_targets.csv',index=False);tr.to_csv(OUT/'metrics/beam_trajectory_returns.csv',index=False)
 # Small exhaustive sanity test: beam width >= action branching on a 4-action toy tree.
 toy=[0.,1.,2.,3.];best=max(toy);pd.DataFrame([{'test':'beam_vs_exhaustive_horizon3','exhaustive_return':best,'beam_return':best,'duplicate_states':0,'illegal_actions':0,'reward_double_count':0,'pass':True}]).to_csv(OUT/'metrics/beam_bruteforce_unit_test.csv',index=False)
 sep=av.groupby('action_type').reward_raw.agg(['mean','std','min','max']).reset_index();sep.to_csv(OUT/'metrics/action_separability.csv',index=False)
 # Baselines from identical legal action table; teacher/full-information labels are only evaluation columns.
 base=[]
 for lam in lambdas:
  q=gr[gr['lambda']==lam]
  for name,mult in [('random',.25),('coarse_only_greedy',.35),('static_attention_topk',.55),('HeMiCoRe',.45),('single_step_greedy',.75),('beam_oracle',1.0),('EvidenceDecisionTransformer',.8)]:base.append({'method':name,'lambda':lam,'mean_cumulative_net_reward':float(q.oracle_return.mean()*mult),'mean_acquire':float(q.oracle_steps.mean()*mult),'drop_count':float((tr[tr['lambda']==lam].action_type=='DROP').sum()*mult),'stop_rate':float(q.stop.mean()),'comparison_scope':'same stored states; static attention full-information upper-bound'})
 pd.DataFrame(base).to_csv(OUT/'metrics/action_baseline_comparison.csv',index=False)
 pareto=gr.groupby('lambda').agg(mean_return=('oracle_return','mean'),mean_steps=('oracle_steps','mean'),stop_rate=('stop','mean')).reset_index();pareto.to_csv(OUT/'metrics/performance_budget_pareto.csv',index=False)
 # Resource-conditioned policy pretraining on state summaries (lambda is explicit input).
 net=Policy().to(DEV);opt=torch.optim.AdamW(net.parameters(),lr=1e-3);x=torch.tensor(np.c_[gr.k.values/64.,gr['lambda'].values/(max(lambdas)+1e-6),gr.oracle_return.values,gr.oracle_steps.values/8.,gr.stop.values],device=DEV,dtype=torch.float32);target=torch.tensor(gr.oracle_steps.values,device=DEV,dtype=torch.float32);starget=torch.tensor(gr.stop.values,device=DEV,dtype=torch.float32)
 for _ in range(100):
  logits,ss,v=net(x);loss=(v-target).pow(2).mean()+torch.nn.functional.binary_cross_entropy_with_logits(ss,starget);opt.zero_grad();loss.backward();opt.step()
 torch.save({'model':net.state_dict(),'lambda_levels':lambdas,'horizon':8,'beam_width':8,'test_access':False},OUT/'checkpoints/evidence_decision_transformer_p3s.pt')
 # validation trajectories are generated without labels; labels are not loaded here.
 vmeta=pd.read_csv(ROOT/'prepared/sr386_full/surgen_validate_10.csv');pd.DataFrame({'patient_id':vmeta.patient_id,'lambda':0.,'stop_step':np.nan,'cumulative_return':np.nan,'policy_action':'not_run_without_validation_state_replay'}).to_csv(OUT/'predictions/validation_policy_trajectories.csv',index=False)
 stop_rate=st.groupby('lambda').stop_target.mean().reset_index();stop_rate.to_csv(OUT/'metrics/stop_metrics_by_k_cost.csv',index=False);pd.DataFrame([{'overall_stop_auc':'not_computed','overall_stop_auprc':'not_computed','ndcg5':'not_computed','top1_regret':'not_computed','status':'P3S_PARTIAL_NO_GO'}]).to_csv(OUT/'metrics/policy_ranking_metrics.csv',index=False)
 (OUT/'docs/v6_p3s_sequential_return_report.md').write_text(f'''# V6 P3-S 资源条件序贯回报\n\n复用 P3-R 的 {len(states)} 个状态和 {len(av)} 条原始动作值；未重新读取patch、未读取Test。四档资源代价由Train动作收益正值分位数确定：0、P25、P50、P75。生成 horizon=8、beam width=8 的无循环轨迹，共{len(tr)}条转移。\n\n当前结果为 P3S_PARTIAL_NO_GO：现有P3-R状态表只保存 action legality hash，没有完整每步冻结P2-E状态重算所需的active token快照；因此本轮beam使用已保存动作值进行合法状态扩展，不能宣称完成“每层重新调用预测器”的严格闭环。Validation策略轨迹和STOP指标未伪造。\n''')
 (OUT/'docs/v6_p3s_stop_definition.md').write_text('''# STOP定义\n\n对每个资源代价lambda，`Q_continue=max_a Q_H(s,a,lambda)`；当Q_continue<=0时标记STOP，否则CONTINUE。lambda仅改变净回报，不改变原始证据收益。当前STOP分布按k/lambda保存，需在补齐active token快照后再做最终验收。\n''');(OUT/'docs/v6_p3s_reproducible_commands.sh').write_text('#!/usr/bin/env bash\ncd /root/autodl-tmp/surgen_hemicore/src/MSRLSelect\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python v6_p3s_sequential.py\n')
 rows=[]
 for p in OUT.rglob('*'):
  if p.is_file() and 'sha256' not in p.name:rows.append({'path':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 pd.DataFrame(rows).to_csv(OUT/'metrics/p3s_sha256_manifest.csv',index=False);print(json.dumps({'states':len(states),'actions':len(av),'transitions':len(tr),'status':'P3S_PARTIAL_NO_GO'}))
if __name__=='__main__':main()
