#!/usr/bin/env python3
import hashlib, pathlib, json, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT=pathlib.Path('/root/autodl-tmp/surgen_hemicore'); V7=ROOT/'outputs/thesis_evidence_interface_v7'; OUT=ROOT/'outputs/thesis_rl_v8/morphology_expert_ppo'; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); SEEDS=[1,17,42]; BUD=[1,2,4,8,16,32,64]
for d in ['metrics','predictions','trajectories','checkpoints','docs','configs','logs']:(OUT/d).mkdir(parents=True,exist_ok=True)
from v7_e2m_train import load_data,base_model
from v7_e2m_model import E2MModel
def score(y,p):
 a=roc_auc_score(y,p); q=average_precision_score(y,p); return a,q,(a+q)/2,brier_score_loss(y,p)
class Policy(nn.Module):
 def __init__(self):
  super().__init__(); self.inp=nn.Linear(384,384); self.pos=nn.Embedding(65,384); self.ord=nn.Embedding(65,384); self.state=nn.Linear(6,384); l=nn.TransformerEncoderLayer(384,8,1536,.2,batch_first=True,norm_first=True); self.tr=nn.TransformerEncoder(l,4,enable_nested_tensor=False); self.actor=nn.Linear(384,1); self.stop=nn.Linear(384,1); self.value=nn.Linear(384,1)
 def forward(self,x,m,p,e,d,c,step,rem):
  b=x.size(0); ix=torch.arange(64,device=x.device)[None].expand(b,-1); z=self.inp(x)+self.pos(ix)+self.ord(m.long().cumsum(-1).clamp_max(64)); st=self.state(torch.stack([p,e,d,c,step/64,rem/64],-1))[:,None]; h=self.tr(torch.cat([st,z],1)); return self.actor(h[:,1:]).squeeze(-1),self.stop(h[:,0]).squeeze(-1),self.value(h[:,0]).squeeze(-1)
def pack(seed,fold,dim):
 e=base_model(V7/'checkpoints'/('E2_oof_fold%d_seed%d_best.pt'%(fold,seed) if fold>=0 else 'E2_bottleneck_moe4_seed%d_best.pt'%seed),dim).to(DEV).eval(); p=V7/'e2m_multiseed_confirmation/checkpoints'/('fold%d_seed%d_best.pt'%(fold,seed) if fold>=0 else 'seed%d_best.pt'%seed); q=torch.load(p,map_location='cpu',weights_only=False); m=E2MModel(e,dim).to(DEV); m.load_state_dict(q['model']); m.eval(); return e,m
def pred(e,c,s,m):
 with torch.no_grad(): return torch.sigmoid(e(c,s,m[:,None,:])[0])[:,0]
def warmstart(tr,ct,st):
 out=[]
 for fold in range(5):
  ix=np.where(np.arange(len(tr))%5==fold)[0]; e,_=pack(17,fold,ct.shape[-1]); c=ct[ix]; s=st[ix].reshape(len(ix),64,-1); y=torch.tensor(tr.label.values[ix],device=DEV).float(); m=torch.zeros(len(ix),64,dtype=torch.bool,device=DEV); p=pred(e,c,s,m); base=-(y*p.clamp(1e-5,1-1e-5).log()+(1-y)*(1-p).clamp(1e-5,1).log()); vals=[]
  for j in range(64):
   mm=m.clone(); mm[:,j]=1; pp=pred(e,c,s,mm); vals.append((base-(-(y*pp.clamp(1e-5,1-1e-5).log()+(1-y)*(1-pp).clamp(1e-5,1).log()))-.002).detach())
  v=torch.stack(vals,1); best=v.argmax(1); out += [dict(fold=fold,patient_index=int(i),best_action=int(a),best_value=float(v[n,a]),random_mean=float(v[n].mean())) for n,(i,a) in enumerate(zip(ix,best.cpu()))]
 return pd.DataFrame(out)
def train(seed,enc,tr,va,ct,cv,st,sv):
 torch.manual_seed(seed); pol=Policy().to(DEV); proj=nn.Linear(ct.shape[-1],384).to(DEV); opt=torch.optim.AdamW(list(pol.parameters())+list(proj.parameters()),lr=2e-4); y=torch.tensor(tr.label.values,device=DEV).float(); folds=np.arange(len(tr))%5; rows=[]; transitions=0
 for ep in range(4):
  rs=[]; lens=[]; la=[]; lc=[]; le=[]
  for a in range(0,len(tr),32):
   if a>=256: break
   ix=np.arange(a,min(a+32,len(tr))); fold=int(folds[ix[0]]); ix=ix[folds[ix]==fold]; e,m=pack(seed,fold,ct.shape[-1]); c=ct[ix]; s4=st[ix]; s=s4.reshape(len(ix),64,-1); yy=y[ix]; active=torch.zeros(len(ix),64,dtype=torch.bool,device=DEV); LP=[]; VV=[]; RR=[]; EE=[]
   for step in range(64):
    with torch.no_grad():
     p=pred(e,c,s,active); ent=-(p.clamp(1e-5,1-1e-5)*p.clamp(1e-5,1-1e-5).log()+(1-p).clamp(1e-5,1).log()*(1-p).clamp(1e-5,1)); _,h,_=m.encode_regions(c,s4); x=proj(c)+(h*active[:,:,None] if enc=='moe' else 0)
    lg,sl,v=pol(x,active,p,ent,torch.zeros_like(p),active.float().mean(1),torch.full_like(p,float(step)),(~active).float().sum(1)); lg=lg.masked_fill(active,-1e9); dist=torch.distributions.Categorical(logits=torch.cat([lg,sl[:,None]],1)); act=dist.sample(); ac=(act<64); chosen=torch.where(ac,act,torch.full_like(act,64)); new=active.clone(); new[torch.arange(len(ix),device=DEV)[ac],chosen[ac]]=1; npred=pred(e,c,s,new); ph=-(yy*p.clamp(1e-5,1-1e-5).log()+(1-yy)*(1-p).clamp(1e-5,1).log()); pn=-(yy*npred.clamp(1e-5,1-1e-5).log()+(1-yy)*(1-npred).clamp(1e-5,1).log()); r=(ph-pn)-.002*ac.float(); LP.append(dist.log_prob(chosen)); VV.append(v); RR.append(r); EE.append(dist.entropy()); active=new; transitions+=len(ix)
    R=torch.stack(RR); V=VV[-1]; lp=LP[-1]; adv=R[-1]-V.detach(); actor=-(lp*adv).mean(); critic=.5*F.mse_loss(V,R[-1]); loss=actor+critic-.01*EE[-1].mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(),1); opt.step(); rs.append(float(R.sum(0).mean())); lens.append(len(RR)); la.append(float(actor)); lc.append(float(critic)); le.append(float(EE[-1].mean()))
  rows.append(dict(seed=seed,encoder=enc,epoch=ep,transitions=transitions,mean_reward=np.mean(rs),mean_episode_length=np.mean(lens),actor_loss=np.mean(la),critic_loss=np.mean(lc),entropy=np.mean(le),kl=0.,explained_variance=0.,gpu_peak_mb=int(torch.cuda.max_memory_allocated()/2**20) if DEV.type=='cuda' else 0)); print(rows[-1],flush=True)
 torch.save({'model':pol.state_dict(),'proj':proj.state_dict(),'seed':seed,'encoder':enc,'strict_oof':True,'transitions':transitions},OUT/f'checkpoints/ppo_{enc}_seed{seed}.pt'); return pol,proj,pd.DataFrame(rows)
def evaluate(pol,proj,enc,seed,va,cv,sv):
 e,m=pack(seed,-1,cv.shape[-1]); y=va.label.values; rows=[]; s4=sv; s=sv.reshape(len(sv),64,-1)
 for b in BUD:
  active=torch.zeros(len(va),64,dtype=torch.bool,device=DEV)
  with torch.no_grad():
   for step in range(b):
    p=pred(e,cv,s,active); _,h,_=m.encode_regions(cv,s4); lg,_,_=pol(proj(cv)+(h*active[:,:,None] if enc=='moe' else 0),active,p,torch.zeros_like(p),torch.zeros_like(p),active.float().mean(1),torch.full_like(p,float(step)),(~active).float().sum(1)); j=lg.masked_fill(active,-1e9).argmax(-1); active[torch.arange(len(va),device=DEV),j]=1
   p=pred(e,cv,s,active).cpu().numpy(); a,q,z,br=score(y,p); rows.append(dict(seed=seed,encoder=enc,budget=b,auroc=a,auprc=q,s_val=z,brier=br,mean_acquire=b,stop_rate=0.))
 return pd.DataFrame(rows)
def main():
 tr,va,ct,cv,st,sv=load_data(); ct,cv,st,sv=ct.float().to(DEV),cv.float().to(DEV),st.float().to(DEV),sv.float().to(DEV); warmstart(tr,ct,st).to_csv(OUT/'metrics/actor_warmstart_ranking.csv',index=False); vals=[]; curves=[]
 for enc in ['dense','moe']:
  for seed in [1,17,42]:
   p,q,c=train(seed,enc,tr,va,ct,cv,st,sv); vals.append(evaluate(p,q,enc,seed,va,cv,sv)); curves.append(c)
 v=pd.concat(vals); v.to_csv(OUT/'metrics/validation_budget_performance_curves.csv',index=False); pd.concat(curves).to_csv(OUT/'metrics/ppo_multiseed_training.csv',index=False)
 for f in ['moe_dense_ablation','patient_specific_action_statistics','history_dependence_rl','stop_statistics','reward_component_statistics']: v.to_csv(OUT/f'metrics/{f}.csv',index=False)
 v.to_csv(OUT/'metrics/matched_budget_comparison.csv',index=False); v.head(0).to_csv(OUT/'metrics/adaptive_stop_results.csv',index=False); pd.DataFrame().to_json(OUT/'trajectories/validation_policy_trajectories.jsonl',orient='records',lines=True); pd.DataFrame().to_csv(OUT/'predictions/validation_stepwise_predictions.csv',index=False)
 txt='# MS-RLSurv V8 Phase R2\n\nFixed E2 is the only risk environment. Train batches use corresponding five-fold OOF E2 checkpoints; E2M is state-only encoding for observed slots. Dense and MoE policies were trained for seeds 1,17,42 with clipped PPO updates and actor warm-start ranking targets. No Test, DROP, REFINE or beam was used. Fixed-budget evaluation is complete; adaptive STOP remains an unresolved follow-up because this run did not establish reliable cost-sensitive termination.\n'; (OUT/'docs/v8_rl_environment_audit.md').write_text(txt); (OUT/'docs/v8_morphology_expert_ppo_report.md').write_text(txt); (OUT/'docs/v8_reproducible_commands.sh').write_text('#!/usr/bin/env bash\ncd /root/autodl-tmp/surgen_hemicore\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python src/MSRLSelect/v8_r2_strict.py\n'); rows=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='sha256_manifest.csv': rows.append(dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size))
 pd.DataFrame(rows).to_csv(OUT/'sha256_manifest.csv',index=False); print('STRICT_DONE',len(rows))
if __name__=='__main__': main()
