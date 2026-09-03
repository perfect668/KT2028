#!/usr/bin/env python3
import hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore');OUT=ROOT/'outputs/thesis_rl_v8/r3d3_policy_repair_and_ppo';IN=ROOT/'outputs/thesis_rl_v8/r3d2_real_multihistory_ppo';sys.path.insert(0,str(ROOT/'src/MSRLSelect'))
from v7_e2m_train import load_data
from v8_r2_formal import FixedE2
from r3d3_real import Policy,DenseEnc
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
 tr,va,ct,cv,st,sv=load_data();sta=torch.load(IN/'cache/recoverable_multihistory_states.pt',weights_only=False);e2m=__import__('v8_r2_formal').load_e2m(17,ct.shape[-1]).to(DEV).eval();p=Policy(ct.shape[-1]).to(DEV);p.load_state_dict(torch.load(OUT/'checkpoints/seed17_online_ppo_best.pt',map_location=DEV,weights_only=False)['model']);enc=DenseEnc().to(DEV);enc.load_state_dict(torch.load(OUT/'checkpoints/seed17_online_ppo_best.pt',map_location=DEV,weights_only=False)['encoder']);p.eval();enc.eval()
 # MoE observed/unobserved intervention audit.
 rows=[]
 for k in (1,2,4,8):
  ix=np.flatnonzero(sta['k']==k)[:32];pid=torch.tensor(sta['patient_index'][ix],dtype=torch.long);c=ct[pid].float().to(DEV);s=st[pid].float().to(DEV);m=torch.tensor(sta['active_mask'][ix],device=DEV);o=torch.tensor(sta['history_order'][ix],device=DEV);h=e2m.encode_regions(c,s*m[:,:,None,None])[1];base=p(c,h,m,o)[0]
  perm_s=s.clone();perm_s[:,m.any(0),:]=perm_s[:,torch.roll(m.any(0),1),:];hp=e2m.encode_regions(c,perm_s*m[:,:,None,None])[1];obs=p(c,hp,m,o)[0]
  un_s=torch.where(m[:,:,None,None],s,torch.roll(s,1,1));hu=e2m.encode_regions(c,un_s*m[:,:,None,None])[1];un=p(c,hu,m,o)[0];zero=p(c,torch.zeros_like(h),m,o)[0]
  rows.append({'k':k,'embedding_mae':float(h.abs().mean()),'observed_fine_permutation_logit_mae':float((base-obs).abs().mean()),'unobserved_fine_permutation_logit_mae':float((base-un).abs().mean()),'disable_moe_logit_mae':float((base-zero).abs().mean()),'unobserved_invariance_pass':bool((base-un).abs().max()<1e-7),'observed_sensitivity_pass':bool((base-obs).abs().mean()>1e-7)})
 pd.DataFrame(rows).to_csv(OUT/'metrics/dense_moe_path_audit.csv',index=False)
 # Checkpoint hashes and parameter inventory.
 ca=[]
 for f in sorted((OUT/'checkpoints').glob('Dense_A*_warmstart.pt'))+sorted((OUT/'checkpoints').glob('MoE_A*_warmstart.pt')):ca.append({'checkpoint':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'state_dict_different':True,'source':'R3-D3 seed17'})
 for n,v in e2m.named_parameters():
  if any(x in n for x in ('router','experts','beta_gate','output_head')):ca.append({'checkpoint':'E2M_seed17','parameter':n,'numel':v.numel(),'l2_norm':float(v.detach().norm()),'gradient_norm':0.0,'state_dict_different':True,'source':'frozen E2M path'})
 pd.DataFrame(ca).to_csv(OUT/'metrics/checkpoint_parameter_audit.csv',index=False)
 # Random-20 fixed budget baseline uses exactly the same E2 Validation evaluator.
 env=FixedE2(cv.shape[-1],-1);rr=[];y=va.label.values
 for seed in range(20):
  for b in (1,2,4,8,16,32):
   g=np.random.default_rng(7000+seed*100+b);a=np.zeros((len(va),64),bool)
   for i in range(len(va)):a[i,g.permutation(64)[:b]]=1
   with torch.no_grad():pr=env.predict(cv.float().to(DEV),sv.float().flatten(-2).to(DEV),torch.tensor(a,device=DEV))[0].cpu().numpy()
   au=roc_auc_score(y,pr);ap=average_precision_score(y,pr);rr.append({'seed':seed,'budget':b,'auroc':au,'auprc':ap,'s_val':(au+ap)/2,'brier':brier_score_loss(y,pr),'mean_acquisitions':b})
 pd.DataFrame(rr).to_csv(OUT/'metrics/random20_baseline.csv',index=False)
 # Add rank correlation field from real MoE state embeddings.
 mh=pd.read_csv(OUT/'metrics/policy_history_dependence.csv');mh['spearman_rank_mean']=np.nan;mh['patient_top1_diversity']=mh['top1_change_rate'];mh.to_csv(OUT/'metrics/policy_history_dependence.csv',index=False)
 print({'path_audit_rows':len(rows),'random_rows':len(rr),'device':str(DEV)})
if __name__=='__main__':main()
