#!/usr/bin/env python3
import itertools,numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from v7_e2m_train import ROOT,EOUT,E2M,DEV,load_data,base_model,mk,KS,MASK_SEEDS
from v7_e2m_model import E2MModel
def metric(y,p):a=roc_auc_score(y,p);b=average_precision_score(y,p);return a,b,.5*(a+b),brier_score_loss(y,p)
def load():
 tr,va,ct,cv,st,sv=load_data();base=base_model(EOUT/'checkpoints/E2_bottleneck_moe4_seed17_best.pt',ct.shape[-1]);m=E2MModel(base,ct.shape[-1]).to(DEV);q=torch.load(E2M/'checkpoints/seed17_best.pt',map_location='cpu',weights_only=False);m.load_state_dict(q['model']);m.eval();return tr,va,ct,cv,st,sv,m,q
def audit_split(name,c,s,meta,m):
 rows=[]
 with torch.no_grad():
  o=m(c.float().to(DEV),s.float().to(DEV),mk(len(meta),[64],7101));w=o['weights'].cpu();present=o['present'].cpu();eo=o['expert_outputs'].cpu();valid=present.any(-1);top=w.argmax(-1)
  top1=torch.stack([((top==e)&valid).sum() for e in range(4)]).float();top1/=top1.sum();top2=torch.stack([((w[...,e]>0)&valid).sum() for e in range(4)]).float();top2/=top2.sum();agg=w[valid].mean(0);effective=float(torch.exp(-(agg*agg.clamp_min(1e-8).log()).sum()));ent=float((-(w.clamp_min(1e-8)*w.clamp_min(1e-8).log()).sum(-1))[valid].mean())
  for e in range(4):
   coverage=float((((w[...,e]>0)&valid).any(1)).float().mean());rows.append(dict(split=name,seed=17,expert=e,top1_occupancy=float(top1[e]),top2_occupancy=float(top2[e]),patient_coverage=coverage,routing_entropy=ent,effective_experts=effective,empty_slot_fraction=float((~present).float().mean())))
  sims=[]
  for a,b in itertools.combinations(range(4),2):
   q=present[...,a]&present[...,b];sims.append(dict(split=name,seed=17,expert_a=a,expert_b=b,output_cosine=float(torch.nn.functional.cosine_similarity(eo[...,a,:],eo[...,b,:],dim=-1)[q].mean())))
 return rows,sims,o
def main():
 tr,va,ct,cv,st,sv,m,ck=load();r1,s1,_=audit_split('train',ct,st,tr,m);r2,s2,o=audit_split('validation',cv,sv,va,m);stats=pd.DataFrame(r1+r2);stats.to_csv(E2M/'metrics/e2m_trained_expert_statistics.csv',index=False);pd.DataFrame(s1+s2).to_csv(E2M/'metrics/e2m_expert_output_similarity.csv',index=False)
 # Prediction and routing condition audits.
 y=va.label.values;mask=mk(len(va),[64],7101)
 with torch.no_grad():
  normal=m(cv.float().to(DEV),sv.float().to(DEV),mask);p=torch.sigmoid(normal['fused_logit'][:,0]).cpu().numpy();off=torch.sigmoid(m(cv.float().to(DEV),sv.float().to(DEV),mask,disable_moe=True)['fused_logit'][:,0]).cpu().numpy();sg=m(cv.float().to(DEV),sv.float().to(DEV),mask,shuffle_global=True);sr=m(cv.float().to(DEV),sv.float().to(DEV),mask,shuffle_region=True)
 aud=[dict(condition='disable_moe',prediction_mae=float(np.abs(p-off).mean()),router_logit_mae=np.nan,top1_change=np.nan),dict(condition='shuffle_patient_global',prediction_mae=float(np.abs(p-torch.sigmoid(sg['fused_logit'][:,0]).cpu().numpy()).mean()),router_logit_mae=float((normal['router_logits']-sg['router_logits']).abs()[normal['present']].mean()),top1_change=float((normal['weights'].argmax(-1)!=sg['weights'].argmax(-1))[normal['present'].any(-1)].float().mean())),dict(condition='shuffle_region_coarse',prediction_mae=float(np.abs(p-torch.sigmoid(sr['fused_logit'][:,0]).cpu().numpy()).mean()),router_logit_mae=float((normal['router_logits']-sr['router_logits']).abs()[normal['present']].mean()),top1_change=float((normal['weights'].argmax(-1)!=sr['weights'].argmax(-1))[normal['present'].any(-1)].float().mean()))]
 pd.DataFrame(aud).to_csv(E2M/'metrics/e2m_routing_condition_audit.csv',index=False)
 abl=[]
 with torch.no_grad():
  for e in range(4):
   pp=torch.sigmoid(m(cv.float().to(DEV),sv.float().to(DEV),mask,disabled_expert=e)['fused_logit'][:,0]).cpu().numpy();a,b,sval,br=metric(y,pp);abl.append(dict(expert_disabled=e,auroc=a,auprc=b,s_val=sval,brier=br,prediction_mae=float(np.abs(pp-p).mean()),s_val_delta=sval-metric(y,p)[2]))
 pd.DataFrame(abl).to_csv(E2M/'metrics/e2m_expert_ablation.csv',index=False)
 # Seed17 comparison uses identical five masks and all k.
 e2m=pd.read_csv(E2M/'metrics/seed17_results.csv');fixed=pd.read_csv(EOUT/'metrics/final_candidate_multiseed.csv');fixed=fixed[fixed.seed==17];cmp=[]
 for k in KS:
  a=e2m[e2m.k==k].mean(numeric_only=True);b=fixed[fixed.k==k].mean(numeric_only=True);cmp.append(dict(k=k,e2m_auroc=a.auroc,e2m_auprc=a.auprc,e2m_s_val=a.s_val,e2m_brier=a.brier,fixed_auroc=b.auroc,fixed_auprc=b.auprc,fixed_s_val=b.s_val,fixed_brier=b.brier,s_val_delta=a.s_val-b.s_val))
 pd.DataFrame(cmp).to_csv(E2M/'metrics/e2m_seed17_comparison.csv',index=False)
 sd=ck['model'];keys=[k for k in sd if k.startswith('router.') or k.startswith('experts.') or k.startswith('beta_gate.')];pd.DataFrame([dict(checkpoint='seed17_best.pt',router_keys=sum(k.startswith('router.') for k in keys),expert_keys=sum(k.startswith('experts.') for k in keys),beta_keys=sum(k.startswith('beta_gate.') for k in keys),trainable_parameter_count=sum(sd[k].numel() for k in keys))]).to_csv(E2M/'metrics/e2m_checkpoint_structure.csv',index=False)
 print({'k64_s':metric(y,p)[2],'min_top1':stats[stats.split=='validation'].top1_occupancy.min(),'effective':stats[stats.split=='validation'].effective_experts.iloc[0],'partial_delta':pd.DataFrame(cmp).query('k<64 and k>0').s_val_delta.mean()})
if __name__=='__main__':main()
