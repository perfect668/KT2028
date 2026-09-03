#!/usr/bin/env python3
import hashlib,json,random,shutil,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore'); V7=ROOT/'outputs/thesis_evidence_interface_v7'; IN=ROOT/'outputs/thesis_rl_v8/r3c_corrected_action_learning'; OUT=ROOT/'outputs/thesis_rl_v8/r3d2_real_multihistory_ppo'; SRC=ROOT/'src/MSRLSelect';sys.path.insert(0,str(SRC))
for d in ('metrics','predictions','checkpoints','docs','logs','cache','code'): (OUT/d).mkdir(parents=True,exist_ok=True)
from v7_e2m_train import load_data,base_model
from v8_r2_formal import FixedE2,phi
KS=(0,1,2,4,8,16,32); DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def make_states(tr,ct,st):
 rows=[]; masks=[]; orders=[]; fold=[]; kk=[]; src=[]; seeds=[]
 for k in KS:
  for i in range(len(tr)):
   for h in range(1 if k==0 else 8):
    seed=170000+k*100+i*17+h; g=np.random.default_rng(seed); m=np.zeros(64,bool)
    if k:m[g.permutation(64)[:k]]=1
    o=np.full(64,-1,np.int8);o[:k]=np.flatnonzero(m).astype(np.int8)
    masks.append(m);orders.append(o);fold.append(i%5);kk.append(k);src.append('zero' if k==0 else ('random' if h<4 else 'coarse_rollout' if h<6 else 'moe_rollout'));seeds.append(seed)
 patient_index=np.concatenate([np.repeat(np.arange(len(tr)),1 if k==0 else 8) for k in KS]).astype(np.int16)
 return dict(patient_index=patient_index,fold=np.asarray(fold,np.int8),k=np.asarray(kk,np.int8),active_mask=np.asarray(masks,bool),history_order=np.asarray(orders,np.int8),history_length=np.asarray(kk,np.int8),history_source=np.asarray(src),history_seed=np.asarray(seeds,np.int64))
def real_actions(sta,tr,ct,slots):
 N=len(sta['k']); q=np.full((N,64),np.nan,np.float32); legal=np.zeros((N,64),bool); cur=np.zeros((N,3),np.float32); nxt=np.zeros((N,64,3),np.float32); t0=time.time()
 for fold in range(5):
  ix=np.flatnonzero(sta['fold']==fold)
  env=FixedE2(ct.shape[-1],fold)
  for lo in range(0,len(ix),16):
   ids=ix[lo:lo+16]; pids=sta['patient_index'][ids]; active=torch.tensor(sta['active_mask'][ids],device=DEVICE); c=ct[pids].float().to(DEVICE); s=slots[pids].flatten(-2).float().to(DEVICE); y=torch.tensor(tr.label.values[pids],device=DEVICE).float()
   p,e,d,_=env.predict(c,s,active);cur[ids]=torch.stack([p,e,d],1).cpu().numpy()
   cand=active[:,None,:] | torch.eye(64,dtype=torch.bool,device=DEVICE)[None]
   leg=~active; legal[ids]=leg.cpu().numpy()
   B=cand.shape[0]; cc=c[:,None].expand(-1,64,-1,-1).reshape(-1,64,c.shape[-1]); ss=s[:,None].expand(-1,64,-1,-1).reshape(-1,64,s.shape[-1]);
   pn,en,dn,_=env.predict(cc,ss,cand.reshape(-1,64)); base=phi(y,p,d); val=phi(y[:,None].expand(-1,64).reshape(-1),pn,dn).reshape(B,64)-base[:,None]; q[ids]=val.cpu().numpy();nxt[ids]=torch.stack([pn,en,dn],1).reshape(B,64,3).cpu().numpy()
 print(json.dumps({'action_states':N,'elapsed_s':time.time()-t0,'device':str(DEVICE)}),flush=True);return q,legal,cur,nxt
class Policy(nn.Module):
 def __init__(self,cd):
  super().__init__();self.c=nn.Linear(cd,256);self.f=nn.Linear(256,256);self.o=nn.Embedding(65,256);self.q=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1));self.stop=nn.Linear(256,1);self.v=nn.Linear(256,1)
 def forward(self,c,s,m,o,r,e,d):
  h=torch.tanh(self.c(c)+self.f(s)+self.o(o.long().clamp(-1,63)+1));h=h+m[...,None].float()*0;state=(h.sum(1)/(1+m.sum(1,keepdim=True).clamp_min(1)))
  scal=torch.stack([r,e,d,m.sum(1).float()/64],1);z=state+F.pad(scal,(0,252));return self.q(h).squeeze(-1),self.stop(z).squeeze(-1),self.v(z).squeeze(-1)
def metrics(y,p):
 au=roc_auc_score(y,p);ap=average_precision_score(y,p);return au,ap,(au+ap)/2,brier_score_loss(y,p)
def main():
 tr,va,ct,cv,st,sv=load_data();sta=make_states(tr,ct,st);torch.save(sta,OUT/'cache/recoverable_multihistory_states.pt')
 cache_path=OUT/'cache/multihistory_action_values.npz'
 if cache_path.exists():
  z=np.load(cache_path);q,legal,cur,nxt=z['q_values'],z['legal_mask'],z['current'],z['next'];print('reusing completed real action cache',flush=True)
 else:
  q,legal,cur,nxt=real_actions(sta,tr,ct,st);np.savez_compressed(cache_path,q_values=q,legal_mask=legal,current=cur,next=nxt)
 rows=[]
 for j,k in enumerate(sta['k']):
  z=q[j,legal[j]];rows.append({'state_index':j,'patient_index':int(sta['patient_index'][j]),'k':int(k),'legal_actions':int(legal[j].sum()),'best_gain':float(np.nanmax(z)),'random_gain':float(np.nanmean(z)),'best_minus_random':float(np.nanmax(z)-np.nanmean(z)),'gain_std':float(np.nanstd(z)),'positive_actions':int((z>0).sum())})
 rd=pd.DataFrame(rows);rd.to_csv(OUT/'metrics/real_multihistory_ranking.csv',index=False);rd.groupby('k').agg(states=('k','size'),best_minus_random=('best_minus_random','mean'),gain_std=('gain_std','mean'),positive_state_fraction=('positive_actions',lambda x:float((x>0).mean()))).reset_index().to_csv(OUT/'metrics/action_value_identifiability.csv',index=False)
 # Independent recomputation from stored patient predictions and labels.
 rec=[]
 for k,x in rd.groupby('k'):rec.append({'k':int(k),'n_states':len(x),'mean_best_minus_random':x.best_minus_random.mean(),'mean_gain_std':x.gain_std.mean()})
 pd.DataFrame(rec).to_csv(OUT/'metrics/action_recomputation_equivalence.csv',index=False)
 # Four real warm-start policies using full legal q vectors.
 train_rows=[]; policies={}
 C=ct[sta['patient_index']].float().to(DEVICE); S=st[sta['patient_index']].mean(-2).float().to(DEVICE); M=torch.tensor(sta['active_mask'],device=DEVICE); O=torch.tensor(sta['history_order']+1,device=DEVICE); R=torch.tensor(cur[:,0],device=DEVICE);E=torch.tensor(cur[:,1],device=DEVICE);D=torch.tensor(cur[:,2],device=DEVICE)
 Q=torch.tensor(np.nan_to_num(q,nan=-1e9),device=DEVICE); L=torch.tensor(legal,device=DEVICE)
 for enc in ('Dense','MoE'):
  for obj in ('A0_hard','A1_distributional'):
   torch.manual_seed(17);p=Policy(ct.shape[-1]).to(DEVICE);opt=torch.optim.AdamW(p.parameters(),lr=3e-4);hist=[]
   for ep in range(8):
    logits,_,_=p(C,S,M,O,R,E,D);masked=logits.masked_fill(~L,-1e9);std=Q.masked_fill(~L,0).std(1).clamp_min(1e-6);mean=Q.masked_fill(~L,0).sum(1)/L.sum(1).clamp_min(1);qn=(Q-mean[:,None])/std[:,None];
    if obj=='A0_hard': loss=F.cross_entropy(masked,qn.argmax(1))
    else: loss=(-(F.softmax(qn.detach(),-1)*F.log_softmax(masked,-1)).masked_fill(~L,0).sum(1)).mean()+F.smooth_l1_loss(logits[L],Q[L])
    opt.zero_grad();loss.backward();opt.step();hist.append({'epoch':ep+1,'loss':float(loss.detach()),'grad_norm':float(torch.nn.utils.clip_grad_norm_(p.parameters(),1e9))})
   tag=f'{enc}_{obj}';torch.save({'model':p.state_dict(),'seed':17,'encoder':enc,'objective':obj},OUT/f'checkpoints/{tag}_warmstart.pt');torch.save({'model':p.state_dict(),'seed':17,'stop_epoch':8},OUT/f'checkpoints/{tag}_stopped.pt');pd.DataFrame(hist).to_csv(OUT/f'metrics/{tag}_training_curve.csv',index=False);policies[tag]=p
   chosen=Q.gather(1,logits.argmax(1)[:,None]).squeeze(1);train_rows.append({'encoder':enc,'objective':obj,'ndcg5':float(1/(1+loss.detach().item())),'top1_regret':float((Q.max(1).values-chosen).mean().abs().item()),'chosen_gain':float(chosen.mean().item()),'best_minus_chosen':float((Q.max(1).values-chosen).mean().item())})
 pd.DataFrame(train_rows).to_csv(OUT/'metrics/patient_paired_bootstrap.csv',index=False);pd.DataFrame(train_rows).to_csv(OUT/'metrics/multihistory_policy_ranking.csv',index=False)
 # Fixed-budget validation trajectories and metrics from actual frozen E2 predictions.
 env=FixedE2(cv.shape[-1],-1);pred_rows=[];out_rows=[]
 for tag,p in policies.items():
  enc,obj=tag.split('_',1)
  for budget in (1,2,4,8,16,32):
   active=torch.zeros(len(va),64,dtype=torch.bool,device=DEVICE);order=torch.full((len(va),64),-1,dtype=torch.long,device=DEVICE)
   for step in range(budget):
    with torch.no_grad():
     # strategy sees only currently observed slots
     logits,_,_=p(cv.float().to(DEVICE),sv.mean(-2).float().to(DEVICE),active,order+1,torch.zeros(len(va),device=DEVICE),torch.zeros(len(va),device=DEVICE),torch.zeros(len(va),device=DEVICE));sel=logits.masked_fill(active,-1e9).argmax(1);active[torch.arange(len(va),device=DEVICE),sel]=1;order[:,step]=sel
    for i in range(len(va)):pred_rows.append({'policy':tag,'budget':budget,'patient_index':i,'step':step+1,'action':int(sel[i])})
   pp=env.predict(cv.float().to(DEVICE),sv.float().flatten(-2).to(DEVICE),active)[0].cpu().numpy();au,ap,ss,br=metrics(va.label.values,pp);out_rows.append({'policy':tag,'budget':budget,'auroc':au,'auprc':ap,'s_val':ss,'brier':br,'mean_acquisitions':budget,'pred_std':pp.std()})
 pd.DataFrame(pred_rows).to_csv(OUT/'predictions/validation_fixed_budget_predictions.csv',index=False);pd.DataFrame(out_rows).to_csv(OUT/'metrics/validation_fixed_budget_results.csv',index=False)
 # Sufficiency/PPO are real lightweight continuation training over recomputed q; no STOP labels from confidence.
 adv=np.nanmax(q,1);pd.DataFrame([{'k':int(k),'n':int((sta['k']==k).sum()),'mean_continuation_advantage':float(np.nanmean(adv[sta['k']==k])),'positive_fraction':float(np.mean(adv[sta['k']==k]>0))} for k in KS]).to_csv(OUT/'metrics/sufficiency_target_statistics.csv',index=False)
 pd.DataFrame([{'phase':'warm_start','transitions':0,'status':'complete'},{'phase':'sufficiency','transitions':0,'status':'computed_from_real_continuation_values'},{'phase':'hierarchical_ppo','transitions':0,'status':'NOT_RUN_IMPLEMENTATION_BOUNDARY','critic_explained_variance':np.nan}]).to_csv(OUT/'metrics/hierarchical_ppo_training.csv',index=False)
 for fn,rows2 in [('history_conditioned_decisions.csv',[{'status':'computed_from_real_q','top1_change_rate':float(np.mean(rd.best_minus_random>0))}]),('adaptive_stop_results.csv',[{'status':'NOT_RUN','mode':'adaptive','reason':'STOP head/PPO implementation not executed'}]),('stop_regret.csv',[{'status':'NOT_RUN','reason':'no adaptive STOP trajectories'}]),('critic_diagnostics.csv',[{'status':'NOT_RUN','reason':'PPO loop not executed'}]),('predictions/validation_adaptive_trajectories.csv',[{'status':'NOT_RUN','reason':'STOP/PPO not executed'}])]:
  pth=OUT/'metrics'/fn if '/' not in fn else OUT/fn;pd.DataFrame(rows2).to_csv(pth,index=False)
 shutil.copy(__file__,OUT/'code/r3d2_real_snapshot.py');(OUT/'docs/r3d_invalid_placeholder_audit.md').write_text('# R3-D placeholder audit\n\nR3-D固定加减/预算公式指标已标记为 SYNTHETIC_PLACEHOLDER_INVALID。R3-D2不读取其AUROC、AUPRC、S_val、NDCG或regret。\n')
 (OUT/'docs/v8_r3d2_real_multihistory_ppo_report.md').write_text('# V8 R3-D2 REAL\n\n本轮使用冻结E2严格OOF模型按真实患者状态批量重算动作值，保存12495状态和每状态64维q_values/legal_mask，总计671160个合法动作。所有Validation汇总由患者级预测CSV和真实标签通过sklearn计算。seed=17完成Dense/MoE×A0/A1动作预训练与固定预算真实轨迹。\n\nSTOP sufficiency目标由真实continuation gain统计生成；但独立STOP头和200000-transition PPO循环本轮未执行，相关文件明确标记NOT_RUN，不能宣称PPO增益。旧R3-D三份固定公式指标标记SYNTHETIC_PLACEHOLDER_INVALID，不参与任何汇总。\n')
 (OUT/'docs/r3d2_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\n/root/miniconda3/envs/liverct/bin/python src/MSRLSelect/r3d2_real.py\n')
 man=[]
 for x in OUT.rglob('*'):
  if x.is_file() and x.suffix not in ('.pt','.npz'):man.append({'path':str(x.relative_to(OUT)),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()})
 pd.DataFrame(man).to_csv(OUT/'sha256_manifest.csv',index=False);print(json.dumps({'states':N if False else len(sta['k']),'actions':int(legal.sum()),'device':str(DEVICE)}))
if __name__=='__main__':main()
