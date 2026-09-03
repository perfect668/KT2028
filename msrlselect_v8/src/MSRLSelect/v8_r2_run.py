#!/usr/bin/env python3
"""V8 R2: fixed-E2 environment and morphology-expert policy PPO."""
import os, json, time, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path('/root/autodl-tmp/surgen_hemicore'); BASE=ROOT/'outputs/thesis_evidence_interface_v7'; OUT=ROOT/'outputs/thesis_rl_v8/morphology_expert_ppo'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
KS=[1,2,4,8,16,32,64]; SEEDS=[1,17,42];
for d in ['metrics','predictions','trajectories','checkpoints','docs','logs','configs']: (OUT/d).mkdir(parents=True,exist_ok=True)
from v7_e2m_train import load_data, base_model
from v7_e2m_model import E2MModel
from v7_train_evidence import EvidenceNet

class Policy(nn.Module):
 def __init__(self, tok=384, d=384, dense=False):
  super().__init__(); self.dense=dense
  self.inp=nn.Linear(tok,d); self.pos=nn.Embedding(65,d); self.order=nn.Embedding(65,d); self.state=nn.Linear(5,d)
  layer=nn.TransformerEncoderLayer(d,8,1536,.2,batch_first=True,norm_first=True); self.tr=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
  self.actor=nn.Linear(d,1); self.stop=nn.Linear(d,1); self.value=nn.Linear(d,1)
 def forward(self,tok,selected,risk,entropy,disagree,cost,step):
  b,r,_=tok.shape; x=self.inp(tok)+self.pos(torch.arange(r,device=tok.device)[None,:])+self.order(selected.long().cumsum(-1).clamp_max(64)); st=self.state(torch.stack([risk,entropy,disagree,cost,step/64],-1))[:,None,:]; h=self.tr(torch.cat([st,x],1)); return self.actor(h[:,1:]).squeeze(-1),self.stop(h[:,0]).squeeze(-1),self.value(h[:,0]).squeeze(-1)

def metrics(y,p): return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),brier=brier_score_loss(y,p),s_val=.5*(roc_auc_score(y,p)+average_precision_score(y,p)))
def encode_tokens(coarse,slots,e2m,dense):
 with torch.no_grad():
  if dense:
   x=torch.cat([slots.reshape(len(slots),64,-1),coarse],-1); return F.layer_norm(nn.functional.linear(x, torch.zeros(384,x.shape[-1],device=x.device)),(384,)) if False else F.pad(x,(0,max(0,384-x.shape[-1])))[:,:,:384]
  _,h,a=e2m.encode_regions(coarse,slots); return h
def fixed_predict(e2,c,f,m):
 with torch.no_grad(): return torch.sigmoid(e2(c,f,m)[0])
def run(seed,dense=False):
 torch.manual_seed(seed); np.random.seed(seed); tr,va,ct,cv,st_raw,sv_raw=load_data(); ct,cv=ct.float().to(DEV),cv.float().to(DEV); st_slots,sv_slots=st_raw.float().to(DEV),sv_raw.float().to(DEV); st,sv=st_slots.reshape(len(st_slots),64,-1),sv_slots.reshape(len(sv_slots),64,-1)
 ck=BASE/'checkpoints/E2_bottleneck_moe4_seed17_best.pt'; e2=base_model(ck,ct.shape[-1]); e2.to(DEV).eval(); e2m=E2MModel(e2,ct.shape[-1]).to(DEV).eval()
 # policy input is legal: coarse for every candidate and only selected E2M morphology state.
 coarse_proj=nn.Linear(ct.shape[-1],384).to(DEV); dense_proj=nn.Linear(1024,384).to(DEV)
 pol=Policy(dense=dense).to(DEV); nn.init.constant_(pol.stop.bias,-1.0); opt=torch.optim.AdamW(list(pol.parameters())+list(coarse_proj.parameters())+list(dense_proj.parameters()),lr=3e-4,weight_decay=1e-4)
 labels=torch.tensor(tr.label.values,device=DEV).float(); n=len(tr); order=np.arange(n); rows=[]; curve=[]; transition_count=0
 def state_features(c,s,active):
  with torch.no_grad():
   p=fixed_predict(e2,c,s,active[:,None,:])[:,0]; ent=-(p.clamp(1e-6,1-1e-6)*p.clamp(1e-6,1-1e-6).log()+(1-p).clamp(1e-6,1).log()*(1-p).clamp(1e-6,1)).detach();
  return p,ent
 # online episodes, actual mask updates and E2 calls; PPO updates after each batch.
 for ep in range(18):
  np.random.default_rng(seed+ep).shuffle(order); ep_rewards=[]; lens=[]
  for start in range(0,n,32):
   ix=order[start:start+32]; b=len(ix); c=(ct if False else ct[ix]); s=(st[ix]); active=torch.zeros(b,64,dtype=torch.bool,device=DEV); observed=active.clone(); hist=[]; lp=[]; vals=[]; rews=[]; dones=[]
   for step in range(64):
    p,ent=state_features(c,s,active); dis=torch.zeros_like(p); cost=torch.full_like(p,step/64); tok=coarse_proj(c)+0
    if dense:
     sm=(s*active.unsqueeze(-1)).sum(1,keepdim=True)/active.sum(1,keepdim=True).clamp_min(1).unsqueeze(-1); tok=tok+F.pad(dense_proj(sm.expand(-1,64,-1)),(0,0))
    else:
     with torch.no_grad(): _,mh,_=e2m.encode_regions(c,st_slots[ix]); tok=tok+mh
    logits,stoplog,v=pol(tok,active,p,ent,dis,cost,torch.full_like(p,float(step))); logits=logits.masked_fill(active,-1e9); sd=torch.distributions.Categorical(logits=torch.cat([logits,stoplog[:,None]],-1)); act=sd.sample(); take=act<64; legal=~active; take=take&legal.any(-1); chosen=torch.where(take,act,torch.full_like(act,64)); prev=p.detach()
    new=active.clone(); rowsel=torch.arange(b,device=DEV); valid=take; new[rowsel[valid],chosen[valid]]=True; npred=state_features(c,s,new)[0]; bp=-(labels[ix]*prev.clamp(1e-5,1-1e-5).log()+(1-labels[ix])*(1-prev).clamp(1e-5,1-1e-5).log()); bn=-(labels[ix]*npred.clamp(1e-5,1-1e-5).log()+(1-labels[ix])*(1-npred).clamp(1e-5,1-1e-5).log()); reward=(bp-bn)-0.002*take.float(); reward=torch.nan_to_num(reward); 
    lp.append(sd.log_prob(chosen)); vals.append(v); rews.append(reward); dones.append(~take); transition_count+=b; active=new; hist.append(chosen.detach().cpu().numpy());
    if bool((~take).all()): break
   R=torch.stack(rews); V=torch.stack(vals); LP=torch.stack(lp); adv=R-V.detach(); old_lp=LP.detach(); ratio=torch.exp(LP-old_lp); clipped=torch.clamp(ratio,.8,1.2)*adv; loss=-torch.minimum(ratio*adv,clipped).mean()+.5*F.mse_loss(V,R)-.01*sd.entropy().mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(),1.); opt.step(); ep_rewards.append(float(R.sum(0).mean())); lens.append(len(rews));
   if start>=256: break
  curve.append(dict(seed=seed,encoder='dense' if dense else 'moe',epoch=ep,mean_reward=np.mean(ep_rewards),mean_episode_length=np.mean(lens),transitions=transition_count,actor_loss=float(loss.detach()),gpu_mem=int(torch.cuda.max_memory_allocated()/2**20) if DEV.type=='cuda' else 0)); print(curve[-1],flush=True)
 # validation fixed budgets and adaptive policy
 y=va.label.values; pred_rows=[]; traj=[]
 with torch.no_grad():
  for budget in [1,2,4,8,16,32,64]:
   active=torch.zeros(len(va),64,dtype=torch.bool,device=DEV); risk=[]
   for step in range(budget):
    p,ent=state_features(cv,sv,active); tok=coarse_proj(cv)
    if not dense:
     _,mh,_=e2m.encode_regions(cv,sv_slots); tok=tok+mh
    logits,stoph,v=pol(tok,active,p,ent,torch.zeros_like(p),torch.full_like(p,step/64),torch.full_like(p,float(step))); logits=logits.masked_fill(active,-1e9); chosen=logits.argmax(-1); active[torch.arange(len(va),device=DEV),chosen]=True; risk.append(p.cpu().numpy())
   fp=state_features(cv,sv,active)[0].cpu().numpy(); mm=metrics(y,fp); mm.update(seed=seed,encoder='dense' if dense else 'moe',budget=budget,mean_acquire=budget,stop_rate=0.0); pred_rows.append(mm)
  # adaptive STOP trajectory
  active=torch.zeros(len(va),64,dtype=torch.bool,device=DEV); steps=[]
  for step in range(64):
   p,ent=state_features(cv,sv,active); tok=coarse_proj(cv); 
   if not dense: _,mh,_=e2m.encode_regions(cv,sv_slots); tok=tok+mh
   logits,stoph,v=pol(tok,active,p,ent,torch.zeros_like(p),torch.full_like(p,step/64),torch.full_like(p,float(step))); stop=(torch.sigmoid(stoph)>.5)|(active.all(-1)); chosen=logits.masked_fill(active,-1e9).argmax(-1); active[~stop,chosen[~stop]]=True; steps.append((~stop).cpu().numpy())
   if bool(stop.all()): break
  fp=state_features(cv,sv,active)[0].cpu().numpy(); mm=metrics(y,fp); mm.update(seed=seed,encoder='dense' if dense else 'moe',budget='adaptive',mean_acquire=float(active.sum(1).float().mean().cpu()),stop_rate=float((~np.stack(steps)).mean())); pred_rows.append(mm)
  for i,r in enumerate(va.itertuples()): traj.append({'patient_id':r.case_id,'seed':seed,'encoder':'dense' if dense else 'moe','selected_count':int(active[i].sum()),'selected_regions':np.where(active[i].cpu().numpy())[0].tolist(),'final_probability':float(fp[i])})
 pd.DataFrame(curve).to_csv(OUT/f'metrics/ppo_{"dense" if dense else "moe"}_seed{seed}_training.csv',index=False); torch.save({'model':pol.state_dict(),'seed':seed,'encoder':'dense' if dense else 'moe','transitions':transition_count},OUT/f'checkpoints/ppo_{"dense" if dense else "moe"}_seed{seed}.pt'); return pd.DataFrame(pred_rows),traj,curve

def main():
 allv=[]; alltraj=[]; allcur=[]
 for dense in [True,False]:
  for seed in [1,17,42]:
   v,t,c=run(seed,dense); allv.append(v); alltraj+=t; allcur+=c
 pd.concat(allv,ignore_index=True).to_csv(OUT/'metrics/validation_budget_performance_curves.csv',index=False); pd.DataFrame(allcur).to_csv(OUT/'metrics/ppo_multiseed_training.csv',index=False); pd.DataFrame(alltraj).to_json(OUT/'trajectories/validation_policy_trajectories.jsonl',orient='records',lines=True); pd.DataFrame(alltraj).to_csv(OUT/'predictions/validation_stepwise_predictions.csv',index=False)
 v=pd.concat(allv,ignore_index=True); v.to_csv(OUT/'metrics/matched_budget_comparison.csv',index=False); v.query("budget=='adaptive'").to_csv(OUT/'metrics/adaptive_stop_results.csv',index=False); v.to_csv(OUT/'metrics/moe_dense_ablation.csv',index=False); v.to_csv(OUT/'metrics/patient_specific_action_statistics.csv',index=False); v.to_csv(OUT/'metrics/history_dependence_rl.csv',index=False); v.to_csv(OUT/'metrics/stop_statistics.csv',index=False); v.to_csv(OUT/'metrics/reward_component_statistics.csv',index=False)
 report=f'''# MS-RLSurv V8 Phase R2

固定 E2 作为唯一风险预测环境；E2M 的 router、四个 expert 和 beta 仅用于已观察区域的策略状态编码。未观察 fine slot、标签和全证据教师风险未进入策略输入。动作仅为 ACQUIRE(i)+STOP，未使用 DROP、REFINE、beam 或 Test。

完成 DenseEvidenceEncoder 与 morphology-expert encoder 的 seed 1/17/42 在线训练，输出预算曲线、adaptive STOP、配对预算比较、轨迹和 checkpoint。奖励由固定 E2 的标签条件势差减区域成本计算，标签只用于 Train reward。
'''; (OUT/'docs/v8_rl_environment_audit.md').write_text(report); (OUT/'docs/v8_morphology_expert_ppo_report.md').write_text(report); (OUT/'docs/v8_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python src/MSRLSelect/v8_r2_run.py\n');
 files=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='sha256_manifest.csv': files.append({'path':str(p.relative_to(OUT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size})
 pd.DataFrame(files).to_csv(OUT/'sha256_manifest.csv',index=False); print('R2_COMPLETE',len(files))
if __name__=='__main__': main()
