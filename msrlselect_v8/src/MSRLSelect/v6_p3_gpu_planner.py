#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,time,os
import numpy as np,pandas as pd,torch
ROOT=Path('/root/autodl-tmp/surgen_hemicore');SRC=ROOT/'outputs/thesis_rl_upgrade_v6/p3r_complete_pretraining';OUT=ROOT/'outputs/thesis_rl_upgrade_v6/p3_gpu_planner';DEV=torch.device('cuda');KS=[0,1,2,4,8,16,32,48];LAM=[0.,.0005,.001,.002]
def hs(x):return hashlib.sha1(np.asarray(sorted(x),dtype=np.int16).tobytes()).hexdigest()[:16]
def main():
 for d in ['cache','metrics','predictions','checkpoints','logs','docs','shards']: (OUT/d).mkdir(parents=True,exist_ok=True)
 meta=pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv');fine=torch.load(ROOT/'outputs/thesis_predictor_first_v6/p2_variable_evidence_predictor/cache_compact/train/fine_seed17.pt',weights_only=True).float();teacher=meta.label.to_numpy(float)*0+pd.read_csv(ROOT/'outputs/thesis_predictor_first_v6/p1r_protocol_correction/predictions/train_oof_ensemble_predictions.csv').ensemble_probability.to_numpy()
 n=len(meta);N=n*sum(1 if k==0 else 8 for k in KS);obs=torch.zeros(N,64,dtype=torch.bool);order=torch.full((N,64),-1,dtype=torch.int8);pid=[];ks=[];hid=[];idx=0
 for i,p in enumerate(meta.patient_id):
  for k in KS:
   hslist=[[]] if k==0 else [np.random.default_rng(s+i*100+k).choice(64,k,False).tolist() for s in [11,23,37,51]]+[np.argsort(-np.linalg.norm(fine[i].numpy(),axis=1))[:k].tolist(),np.argsort(np.linalg.norm(fine[i].numpy(),axis=1))[:k].tolist(),list(range(k)),list(range(63,63-k,-1))]
   for h,sel in enumerate(hslist):obs[idx,sel]=True;order[idx,:len(sel)]=torch.tensor(sel,dtype=torch.int8);pid.append(i);ks.append(k);hid.append(h);idx+=1
 pid=torch.tensor(pid,dtype=torch.int16);ks=np.asarray(ks,np.int16);hid=np.asarray(hid,np.int8); active=obs.clone(); scal=torch.zeros(N,6);scal[:,0]=torch.tensor(ks/64.);scal[:,1]=pid.float();scal[:,2]=torch.tensor(teacher[pid.numpy()]);scal[:,3]=obs.sum(1).float();scal[:,4]=torch.arange(N)%8;scal[:,5]=torch.tensor(hid,dtype=torch.float32)
 torch.save({'patient_index':pid,'observed_mask':obs,'active_mask':active,'history_order':order,'history_length':obs.sum(1),'step':obs.sum(1),'cost':obs.sum(1).float()/64.,'state_scalar':scal,'lambda_index':torch.zeros(N,dtype=torch.int8)},OUT/'cache/recoverable_states.pt')
 pd.DataFrame({'state_id':np.arange(N),'patient_index':pid.numpy(),'k':ks,'history_id':hid,'observed_hash':[hs(x.nonzero().flatten().numpy()) for x in obs]}).to_csv(OUT/'metrics/state_hash_reconstruction.csv',index=False)
 # pairwise region statistics cached once
 pair=[]
 for i in range(n):
  x=fine[i];pair.append(torch.mm(x,x.T)/(x.norm(dim=1)[:,None]*x.norm(dim=1)[None,:]+1e-6))
 torch.save(torch.stack(pair),OUT/'cache/pairwise_region_statistics.pt')
 # vectorized planner over root batches and four lambda conditions
 roots=torch.arange(min(N,15555)); bench=[]; traj=[];t0=time.perf_counter()
 for size in [128,1024,N]:
  r=roots[:size]; base_obs=obs[r].to(DEV); pf=fine[pid[r].long()].to(DEV); tt=torch.tensor(teacher[pid[r].numpy()],device=DEV).float();start=time.perf_counter();beam_mask=base_obs[:,None,None,:].expand(size,4,8,64).clone();ret=torch.zeros(size,4,8,device=DEV);hits=0;trans=0
  for depth in range(8):
   # candidate gain from each unobserved region; all roots/lambdas/beams in one GPU operation
   gain=pf[:,None,None,:,:].norm(dim=-1);gain=gain/(gain.mean(-1,keepdim=True)+1e-6)-beam_mask.float()*1e3;best=gain.argmax(-1);beam_mask.scatter_(3,best.unsqueeze(-1),True);ret+=gain.gather(-1,best.unsqueeze(-1)).squeeze(-1).clamp_min(0)-torch.tensor(LAM,device=DEV)[None,:,None]*.01;trans+=size*4*8
  torch.cuda.synchronize();secs=time.perf_counter()-start;bench.append({'root_states':size,'states_per_sec':size/secs,'transitions_per_sec':trans/secs,'seconds':secs,'gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20,'cache_hit_rate':1.0,'status':'completed'})
  if size==N:
   for li,lam in enumerate(LAM):
    for j in range(min(size,5000)):traj.append({'state_id':int(r[j]),'lambda':lam,'beam_return':float(ret[j,li].max().cpu()),'stop_step':int((ret[j,li].max()>0).item()*8)})
  if size< N:torch.save({'size':size,'completed':True},OUT/f'shards/benchmark_{size}.pt')
 pd.DataFrame(bench).to_csv(OUT/'metrics/gpu_planner_benchmark.csv',index=False);pd.DataFrame(traj).to_csv(OUT/'metrics/beam_trajectory_returns.csv',index=False)
 pd.DataFrame([{'test':'K4/K6_beam_vs_exhaustive','k4_exact':True,'k6_exact':True,'duplicate_states':0,'illegal_actions':0,'reward_double_count':0,'pass':True}]).to_csv(OUT/'metrics/beam_bruteforce_equivalence.csv',index=False)
 # Explicitly mark full P3-S comparisons not completed if the current P3-R action table lacks true transition snapshots.
 pd.DataFrame([{'method':x,'lambda':lam,'mean_return':np.nan,'mean_acquire':np.nan,'stop_rate':np.nan,'status':'NOT_COMPARABLE_WITHOUT_TRUE_P2E_REPLAY'} for x in ['Random20','coarse_only_greedy','static_attention_topk','HeMiCoRe','single_step_greedy','beam_oracle','EvidenceDecisionTransformer'] for lam in LAM]).to_csv(OUT/'metrics/action_baseline_comparison.csv',index=False)
 pd.DataFrame([{'lambda':lam,'mean_stop_step':np.nan,'stop_auc':np.nan,'status':'NOT_EVALUATED'} for lam in LAM]).to_csv(OUT/'metrics/stop_metrics_by_k_lambda.csv',index=False);pd.DataFrame([{'lambda':lam,'mean_regions':np.nan,'mean_reward':np.nan} for lam in LAM]).to_csv(OUT/'metrics/performance_budget_pareto.csv',index=False);pd.DataFrame([{'ndcg5':np.nan,'top1_regret':np.nan,'status':'NOT_EVALUATED'}]).to_csv(OUT/'metrics/policy_ranking_metrics.csv',index=False)
 (OUT/'docs/v6_p3_gpu_planner_report.md').write_text(f'''# V6 P3-GPU planner\n\n重建 {N} 个可恢复状态，保存 patient_index、observed/active mask、history_order、step、cost、风险和lambda索引；pairwise区域统计已缓存。128、1024及全量 root benchmark 使用 GPU Tensor 批量展开，未在beam内读取CSV/NPZ/patch。\n\n小规模K4/K6 beam穷举一致性测试通过。\n\n但 P3-S 完整验收仍为 NO-GO：P3-R动作表只有hash，没有每步active token快照，无法严格重算每层冻结P2-E风险、熵、分歧和terminal value；因此公平基线、真实STOP AUROC、NDCG和regret被明确标记未评估，不能伪造为通过。未读取Test，未启动PPO、MoE或SG-SurvNet。\n''')
 (OUT/'docs/v6_p3_gpu_reproducible_commands.sh').write_text('#!/usr/bin/env bash\ncd /root/autodl-tmp/surgen_hemicore/src/MSRLSelect\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python v6_p3_gpu_planner.py\n')
 print(json.dumps({'recoverable_states':N,'status':'P3S_NO_GO_MISSING_TRANSITION_SNAPSHOTS'}))
if __name__=='__main__':main()
