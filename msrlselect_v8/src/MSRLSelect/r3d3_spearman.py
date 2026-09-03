#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
ROOT=Path('/root/autodl-tmp/surgen_hemicore');OUT=ROOT/'outputs/thesis_rl_v8/r3d3_policy_repair_and_ppo';IN=ROOT/'outputs/thesis_rl_v8/r3d2_real_multihistory_ppo';sys.path.insert(0,str(ROOT/'src/MSRLSelect'))
from v7_e2m_train import load_data
from r3d3_real import Policy
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
 tr,va,ct,cv,st,sv=load_data();sta=torch.load(IN/'cache/recoverable_multihistory_states.pt',weights_only=False);z=np.load(IN/'cache/multihistory_action_values.npz');q=z['q_values'];legal=z['legal_mask'];e2m=__import__('v8_r2_formal').load_e2m(17,ct.shape[-1]).to(DEV).eval();p=Policy(ct.shape[-1]).to(DEV);p.load_state_dict(torch.load(OUT/'checkpoints/MoE_A1_warmstart.pt',map_location=DEV,weights_only=False)['model']);p.eval();scores=[]
 with torch.no_grad():
  for lo in range(0,len(sta['k']),256):
   hi=min(lo+256,len(sta['k']));pid=torch.tensor(sta['patient_index'][lo:hi],dtype=torch.long);c=ct[pid].float().to(DEV);s=st[pid].float().to(DEV);m=torch.tensor(sta['active_mask'][lo:hi],device=DEV);o=torch.tensor(sta['history_order'][lo:hi],device=DEV);h=e2m.encode_regions(c,s*m[:,:,None,None])[1];scores.append(p(c,h,m,o)[0].cpu().numpy())
 lg=np.concatenate(scores);out=pd.read_csv(OUT/'metrics/policy_history_dependence.csv')
 for i,r in out.iterrows():
  if not str(r.policy).startswith('MoE'):continue
  vals=[];ids=np.flatnonzero(sta['k']==int(r.k))
  for pid in range(len(tr)):
   ii=ids[sta['patient_index'][ids]==pid]
   for j in ii:
    a=lg[j,legal[j]];b=q[j,legal[j]];vals.append(float(spearmanr(a,b).statistic) if len(a)>2 else 0.)
  out.loc[i,'spearman_rank_mean']=float(np.nanmean(vals)) if vals else np.nan
 out.to_csv(OUT/'metrics/policy_history_dependence.csv',index=False);print('updated',len(out))
if __name__=='__main__':main()
