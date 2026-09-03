#!/usr/bin/env python3
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn.functional as F
ROOT=Path('/root/autodl-tmp/surgen_hemicore'); OUT=ROOT/'outputs/thesis_rl_v8/r3d4_integrated_adaptive_ppo'; IN=ROOT/'outputs/thesis_rl_v8/r3d2_real_multihistory_ppo'; sys.path.insert(0,str(ROOT/'src/MSRLSelect'))
from v7_e2m_train import load_data
from r3d3_real import Policy,DenseEnc
D=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
 tr,va,ct,cv,st,sv=load_data(); sta=torch.load(IN/'cache/recoverable_multihistory_states.pt',weights_only=False); z=np.load(IN/'cache/multihistory_action_values.npz'); q=z['q_values'].astype('float32'); legal=z['legal_mask']; pid=torch.tensor(sta['patient_index'],dtype=torch.long); C=ct[pid].float().to(D); S=st[pid].float().to(D); M=torch.tensor(sta['active_mask'],device=D); O=torch.tensor(sta['history_order'],device=D); Q=torch.tensor(np.nan_to_num(q,nan=0),device=D); L=torch.tensor(legal,device=D)
 e=DenseEnc().to(D); e.eval(); E=e(S,M).detach(); p=Policy(ct.shape[-1]).to(D); opt=torch.optim.AdamW(p.parameters(),lr=5e-5); rng=np.random.default_rng(17); logs=[]; total=0; batch=1024; epochs=4; mb=256
 while total<200000:
  ix=torch.tensor(rng.integers(0,len(q),batch),device=D); Cb,Eb,Mb,Ob=C[ix],E[ix],M[ix],O[ix]; Lb=L[ix]; old=copy.deepcopy(p).eval();
  with torch.no_grad():
   lg,sl,v=old(Cb,Eb,Mb,Ob); all_old=torch.cat([sl[:,None],lg.masked_fill(~Lb,-1e9)],1); dist_old=torch.distributions.Categorical(logits=all_old); act=dist_old.sample(); oldlp=dist_old.log_prob(act).detach(); acq=(act-1).clamp_min(0); rew=torch.where(act==0,torch.zeros(batch,device=D),Q[ix].gather(1,acq[:,None]).squeeze(1)); ret=rew; adv=(ret-ret.mean())/(ret.std()+1e-6)
  agg=[]
  for ep in range(epochs):
   perm=torch.randperm(batch,device=D)
   for stx in range(0,batch,mb):
    j=perm[stx:stx+mb]; lg,sl,v=p(Cb[j],Eb[j],Mb[j],Ob[j]); all_new=torch.cat([sl[:,None],lg.masked_fill(~Lb[j],-1e9)],1); dist=torch.distributions.Categorical(logits=all_new); nlp=dist.log_prob(act[j]); ratio=torch.exp(nlp-oldlp[j]); clipfrac=((ratio-1).abs()>0.2).float().mean(); actor=-(torch.minimum(ratio*adv[j],ratio.clamp(.8,1.2)*adv[j])).mean(); critic=.5*F.mse_loss(v,ret[j]); stop_loss=F.binary_cross_entropy_with_logits(sl,(act[j]==0).float()); ent_r=torch.distributions.Categorical(logits=lg.masked_fill(~Lb[j],-1e9)).entropy().mean(); ent_s=torch.distributions.Bernoulli(logits=sl).entropy().mean(); kl=(oldlp[j]-nlp).mean(); loss=actor+critic+0.1*stop_loss-0.005*ent_r-0.005*ent_s
    opt.zero_grad(); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(p.parameters(),.5); opt.step(); agg.append((actor.item(),stop_loss.item(),critic.item(),ent_r.item(),ent_s.item(),kl.item(),clipfrac.item(),grad.item()))
  total+=batch; arr=np.asarray(agg)
  with torch.no_grad(): vp=p(Cb,Eb,Mb,Ob)[2]
  ev=1-float(torch.var(ret-vp)/(torch.var(ret)+1e-6)); logs.append({'transitions':total,'actor_loss':arr[:,0].mean(),'stop_loss':arr[:,1].mean(),'critic_loss':arr[:,2].mean(),'entropy_region':arr[:,3].mean(),'entropy_stop':arr[:,4].mean(),'approx_kl':arr[:,5].mean(),'clip_fraction':arr[:,6].mean(),'grad_norm':arr[:,7].mean(),'explained_variance':ev,'gpu_memory_mb':torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0,'elapsed_sec':0.0})
  if total%50000<batch or total>=200000: torch.save({'model':p.state_dict(),'encoder':e.state_dict(),'seed':17,'transitions':total,'implementation':'corrected_old_policy_minibatch'},OUT/'checkpoints'/f'seed17_corrected_ppo_{min(total,200000)//50000}k.pt')
 pd.DataFrame(logs).to_csv(OUT/'metrics/corrected_ppo_training.csv',index=False); torch.save({'model':p.state_dict(),'encoder':e.state_dict(),'seed':17,'transitions':total,'implementation':'corrected_old_policy_minibatch'},OUT/'checkpoints/seed17_corrected_ppo_best.pt'); print(json.dumps({'transitions':total,'rows':len(logs)}))
if __name__=='__main__':main()
