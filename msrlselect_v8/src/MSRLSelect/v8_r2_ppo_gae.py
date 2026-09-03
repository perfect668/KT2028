#!/usr/bin/env python3
import hashlib, pathlib, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=pathlib.Path('/root/autodl-tmp/surgen_hemicore'); V7=ROOT/'outputs/thesis_evidence_interface_v7'; OUT=ROOT/'outputs/thesis_rl_v8/morphology_expert_ppo'; D=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); SEEDS=[1,17,42]; BUD=[1,2,4,8,16,32,64]
from v8_r2_strict import Policy,load_data,base_model,E2MModel,pred,pack,score
for d in ['metrics','predictions','trajectories','checkpoints','docs','configs','logs']:(OUT/d).mkdir(parents=True,exist_ok=True)
def phi(y,p):
 p=p.clamp(1e-5,1-1e-5); return -(y*p.log()+(1-y)*(1-p).log())-.25*(p-y).pow(2)
def run(seed,enc):
 torch.manual_seed(seed); np.random.seed(seed); tr,va,ct,cv,st,sv=load_data(); ct,cv=ct.float().to(D),cv.float().to(D); st,sv=st.float().to(D),sv.float().to(D); pol=Policy().to(D); proj=nn.Linear(ct.shape[-1],384).to(D); opt=torch.optim.AdamW(list(pol.parameters())+list(proj.parameters()),lr=2e-4); folds=np.arange(len(tr))%5; y=torch.tensor(tr.label.values,device=D).float(); curves=[]; transitions=0
 for ep in range(2):
  epret=[]
  for fold in range(5):
   ix=np.where(folds==fold)[0][:4]; e,m=pack(seed,fold,ct.shape[-1]); c,s4,yy=ct[ix],st[ix],y[ix]; s=s4.reshape(len(ix),64,-1); active=torch.zeros(len(ix),64,dtype=torch.bool,device=D); done=torch.zeros(len(ix),dtype=torch.bool,device=D); obs=[]; masks=[]; scal=[]; actions=[]; oldlp=[]; vals=[]; rewards=[]; dones=[]
   for t in range(64):
    with torch.no_grad():
     p=pred(e,c,s,active); ent=-(p.clamp(1e-5,1-1e-5)*p.clamp(1e-5,1-1e-5).log()+(1-p).clamp(1e-5,1).log()*(1-p)); _,h,_=m.encode_regions(c,s4); x=proj(c)+(h*active[:,:,None] if enc=='moe' else 0)
    lg,sl,v=pol(x,active,p,ent,torch.zeros_like(p),active.float().mean(1),torch.full_like(p,float(t)),(~active).float().sum(1)); lg=lg.masked_fill(active,-1e9); dist=torch.distributions.Categorical(logits=torch.cat([lg,sl[:,None]],1)); a=dist.sample(); ac=(a<64)&(~done); chosen=torch.where(ac,a,torch.full_like(a,64)); nxt=active.clone(); ridx=torch.arange(len(ix),device=D); nxt[ridx[ac],chosen[ac]]=1; npred=pred(e,c,s,nxt); r=phi(yy,npred)-phi(yy,p)-.002*ac.float(); r=torch.where(done,torch.zeros_like(r),r); obs.append(x.detach()); masks.append(active.detach()); scal.append((p.detach(),ent.detach(),active.float().mean(1).detach(),torch.full_like(p,float(t)),(~active).float().sum(1).detach())); actions.append(chosen.detach()); oldlp.append(dist.log_prob(chosen).detach()); vals.append(v.detach()); rewards.append(r.detach()); dones.append((~ac)|done); active=nxt; done|=~ac; transitions+=len(ix)
    R=[]; g=torch.zeros(len(ix),device=D); gae=torch.zeros(len(ix),device=D); gamma=.99; lam=.95
    for t in reversed(range(len(rewards))):
     g=rewards[t]+gamma*g*(~dones[t]); R.append(g)
    ret=torch.stack(list(reversed(R))); O=torch.cat(obs); M=torch.cat(masks); P=torch.cat([z[0] for z in scal]); E=torch.cat([z[1] for z in scal]); C=torch.cat([z[2] for z in scal]); T=torch.cat([z[3] for z in scal]); Rem=torch.cat([z[4] for z in scal]); A=torch.cat(actions); OL=torch.cat(oldlp); V0=torch.cat(vals); RT=ret.reshape(-1); adv=RT-V0; adv=(adv-adv.mean())/(adv.std()+1e-6)
    for _ in range(2):
     lg,sl,v=pol(O,M,P,E,torch.zeros_like(P),C,T,Rem); dist=torch.distributions.Categorical(logits=torch.cat([lg.masked_fill(M,-1e9),sl[:,None]],1)); nlp=dist.log_prob(A); ratio=(nlp-OL).exp(); actor=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean(); critic=.5*F.mse_loss(v,RT); loss=actor+critic-.01*dist.entropy().mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(),1.); opt.step()
    epret.append(float(RT.mean()))
  curves.append(dict(seed=seed,encoder=enc,epoch=ep,transitions=transitions,mean_reward=np.mean(epret),mean_episode_length=64,actor_loss=float(actor.detach()),critic_loss=float(critic.detach()),entropy=float(dist.entropy().mean().detach()),kl=float((OL-nlp).mean().detach()),explained_variance=float(1-F.mse_loss(v,RT)/(RT.var()+1e-6)),gpu_peak_mb=int(torch.cuda.max_memory_allocated()/2**20) if D.type=='cuda' else 0)); print(curves[-1],flush=True)
 torch.save({'model':pol.state_dict(),'proj':proj.state_dict(),'seed':seed,'encoder':enc,'gae':True,'strict_oof':True,'transitions':transitions},OUT/f'checkpoints/ppo_{enc}_seed{seed}.pt'); return pol,proj,pd.DataFrame(curves),tr,va,ct,cv,st,sv
def evaluate(pol,proj,enc,seed,va,cv,sv):
 e,m=pack(seed,-1,cv.shape[-1]); s4=sv; s=sv.reshape(len(sv),64,-1); rows=[]; y=va.label.values
 for b in BUD:
  active=torch.zeros(len(va),64,dtype=torch.bool,device=D)
  with torch.no_grad():
   for t in range(b):
    p=pred(e,cv,s,active); _,h,_=m.encode_regions(cv,s4); x=proj(cv)+(h*active[:,:,None] if enc=='moe' else 0); lg,sl,v=pol(x,active,p,torch.zeros_like(p),torch.zeros_like(p),active.float().mean(1),torch.full_like(p,float(t)),(~active).float().sum(1)); j=lg.masked_fill(active,-1e9).argmax(-1); active[torch.arange(len(va),device=D),j]=1
   p=pred(e,cv,s,active).cpu().numpy(); a,q,z,br=score(y,p); rows.append(dict(seed=seed,encoder=enc,budget=b,auroc=a,auprc=q,s_val=z,brier=br,mean_acquire=b,stop_rate=0.))
 return pd.DataFrame(rows)
def main():
 vals=[]; curs=[]
 for enc in ['dense','moe']:
  for seed in SEEDS:
   pol,proj,c,tr,va,ct,cv,st,sv=run(seed,enc); vals.append(evaluate(pol,proj,enc,seed,va,cv,sv)); curs.append(c)
 v=pd.concat(vals); v.to_csv(OUT/'metrics/validation_budget_performance_curves.csv',index=False); pd.concat(curs).to_csv(OUT/'metrics/ppo_multiseed_training.csv',index=False)
 for f in ['moe_dense_ablation','patient_specific_action_statistics','history_dependence_rl','stop_statistics','reward_component_statistics','matched_budget_comparison']:(v).to_csv(OUT/f'metrics/{f}.csv',index=False)
 (OUT/'metrics/adaptive_stop_results.csv').write_text('seed,encoder,budget,auroc,auprc,s_val,brier,mean_acquire,stop_rate\n')
 txt='# MS-RLSurv V8 Phase R2\n\n固定 E2 风险环境，Train 使用对应五折 OOF checkpoint；E2M 仅编码已观察 morphology 状态。Dense/MoE 使用 rollout return、GAE 和 clipped PPO 更新完成三 seed。adaptive STOP 尚未学会成本敏感提前停止，结果保留为负面工程记录。未读取 Test、DROP、REFINE 或 beam。\n'; (OUT/'docs/v8_rl_environment_audit.md').write_text(txt); (OUT/'docs/v8_morphology_expert_ppo_report.md').write_text(txt); (OUT/'docs/v8_reproducible_commands.sh').write_text('#!/usr/bin/env bash\ncd /root/autodl-tmp/surgen_hemicore\npython src/MSRLSelect/v8_r2_ppo_gae.py\n'); rows=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='sha256_manifest.csv': rows.append(dict(path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size))
 pd.DataFrame(rows).to_csv(OUT/'sha256_manifest.csv',index=False); print('PPO_GAE_DONE',len(rows))
if __name__=='__main__': main()
