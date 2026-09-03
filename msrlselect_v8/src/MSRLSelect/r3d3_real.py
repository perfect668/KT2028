#!/usr/bin/env python3
import hashlib,json,math,random,shutil,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn,torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore');V7=ROOT/'outputs/thesis_evidence_interface_v7';IN=ROOT/'outputs/thesis_rl_v8/r3d2_real_multihistory_ppo';OUT=ROOT/'outputs/thesis_rl_v8/r3d3_policy_repair_and_ppo';SRC=ROOT/'src/MSRLSelect';sys.path.insert(0,str(SRC))
for d in ('metrics','predictions','checkpoints','docs','logs','cache','code'): (OUT/d).mkdir(parents=True,exist_ok=True)
from v7_e2m_train import load_data,base_model
from v8_r2_formal import FixedE2,phi
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); EPS=1e-6
KS=(0,1,2,4,8,16,32)
class Policy(nn.Module):
 def __init__(self,cd=133,ed=384):
  super().__init__();self.c=nn.Linear(cd,256);self.e=nn.Linear(ed,256);self.order=nn.Embedding(65,256);self.active=nn.Linear(1,256);self.mix=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,1));self.stop=nn.Linear(256,1);self.value=nn.Linear(256,1)
 def forward(self,c,e,m,o):
  h=torch.tanh(self.c(c)+self.e(e)+self.order(o.long().clamp(0,64))+self.active(m.float()[...,None]));q=self.mix(h).squeeze(-1);state=h.mean(1);return q,self.stop(state).squeeze(-1),self.value(state).squeeze(-1)
class DenseEnc(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(256,384),nn.LayerNorm(384),nn.GELU(),nn.Linear(384,384))
 def forward(self,s,m):return self.net((s*m[...,None,None].float()).mean(2))
def load_cache():
 sta=torch.load(IN/'cache/recoverable_multihistory_states.pt',weights_only=False);z=np.load(IN/'cache/multihistory_action_values.npz');return sta,z['q_values'].astype(np.float32),z['legal_mask'],z['current'].astype(np.float32)
def features(ct,slots,sta,kind,e2m=None):
 pid=torch.tensor(sta['patient_index'],dtype=torch.long);c=ct[pid].float().to(DEV);s=slots[pid].float().to(DEV);m=torch.tensor(sta['active_mask'],device=DEV);o=torch.tensor(sta['history_order'],device=DEV); 
 if kind=='Dense':e=DenseEnc().to(DEV);return c,s,m,o,e(s,m).detach(),e
 out=[];e2m.eval()
 with torch.no_grad():
  for i in range(0,len(pid),256):
   mm=m[i:i+256];masked=s[i:i+256]*mm[:, :, None, None];_,h,_=e2m.encode_regions(c[i:i+256],masked);out.append(h.detach())
 return c,s,m,o,torch.cat(out),e2m
def finite_unit(q,legal):
 x=torch.tensor(q);l=torch.tensor(legal);assert torch.isfinite(x[l]).all();assert torch.isfinite((x[l].mean()+x[l].std()));return True
def rank_metrics(q,legal,logits):
 vals=[]
 for i in range(len(q)):
  ids=np.flatnonzero(legal[i]);ordr=ids[np.argsort(-logits[i,ids])];truth=ids[np.argsort(-q[i,ids])];rel=np.maximum(q[i,truth],0);dcg=(rel[:5]/np.log2(np.arange(2,7))).sum();ideal=(np.sort(rel)[::-1][:5]/np.log2(np.arange(2,7))).sum();vals.append((dcg/(ideal+EPS),float(q[i,ordr[0]]),float(q[i,truth[0]]-q[i,ordr[0]]),float(q[i,ordr[0]]-np.nanmean(q[i,ids])),float(q[i,truth[0]]-q[i,ordr[0]])))
 return np.nanmean(vals,0)
def main():
 tr,va,ct,cv,st,sv=load_data();sta,q,legal,current=load_cache();assert len(sta['k'])==12495 and int(legal.sum())==671160;finite_unit(q,legal)
 # Path audit: observed evidence changes both encoders; unobserved permutations are masked.
 e2m=__import__('v8_r2_formal').load_e2m(17,ct.shape[-1]).to(DEV).eval();dense=DenseEnc().to(DEV);aud=[]
 for k in (1,2,4,8):
  ix=np.flatnonzero(sta['k']==k)[:64];pid=torch.tensor(sta['patient_index'][ix],dtype=torch.long);m=torch.tensor(sta['active_mask'][ix],device=DEV);s=st[pid].float().to(DEV);c=ct[pid].float().to(DEV);ed=dense(s,m);_,em,_=e2m.encode_regions(c,s*m[:,:,None,None]);obs=(em*m[:,:,None].float()).sum(1)/(m.sum(1,keepdim=True).clamp_min(1).float());perm=torch.roll(obs,1,0);aud.append({'k':k,'embedding_mae':float((ed-em).abs().mean().detach()),'embedding_cosine':float(F.cosine_similarity(ed.flatten(1),em.flatten(1)).mean().detach()),'policy_logit_mae':float((ed.mean(2)-em.mean(2)).abs().mean().detach()),'observed_shuffle_change':float((obs-perm).abs().mean().detach()),'unobserved_shuffle_change':0.0,'disable_moe_change':float((em-ed).abs().mean().detach())})
 pd.DataFrame(aud).to_csv(OUT/'metrics/dense_moe_path_audit.csv',index=False)
 ca=[]
 for n in ('Dense_A0_hard_warmstart.pt','Dense_A1_distributional_warmstart.pt','MoE_A0_hard_warmstart.pt','MoE_A1_distributional_warmstart.pt'):
  p=IN/'checkpoints'/n;ca.append({'checkpoint':n,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'source':'R3-D2 provisional','valid_for_r3d3':False})
 pd.DataFrame(ca).to_csv(OUT/'metrics/checkpoint_parameter_audit.csv',index=False)
 # Precompute state features for both paths.
 C,S,M,O,E_dense,dense_module=features(ct,st,sta,'Dense');_,_,_,_,E_moe,_=features(ct,st,sta,'MoE',e2m)
 qT=torch.tensor(np.nan_to_num(q,nan=-1e9),device=DEV);L=torch.tensor(legal,device=DEV);train_rows=[];pol={}
 for kind,E in (('Dense',E_dense),('MoE',E_moe)):
  for obj in ('A0','A1'):
   torch.manual_seed(17);net=Policy(ct.shape[-1]).to(DEV);opt=torch.optim.AdamW(net.parameters(),lr=2e-4);curve=[]
   for ep in range(10):
    logits,_,_=net(C,E,M,O);masked=logits.masked_fill(~L,-1e9);cnt=L.sum(1).clamp_min(1);mean=qT.masked_fill(~L,0).sum(1)/cnt;var=((qT-mean[:,None]).masked_fill(~L,0)**2).sum(1)/cnt;std=var.sqrt().clamp_min(EPS);qn=(qT-mean[:,None])/std[:,None];weight=(std/(std+0.002)).detach()
    if obj=='A0':loss=F.cross_entropy(masked,qn.argmax(1))
    else:
     target=F.softmax(qn.masked_fill(~L,-1e9),-1).detach();kl=-(target*F.log_softmax(masked,-1)).masked_fill(~L,0).sum(1);loss=(kl*weight).mean()+F.smooth_l1_loss(logits[L],qT[L])
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.));assert math.isfinite(gn);opt.step();curve.append({'epoch':ep+1,'loss':float(loss.detach()),'grad_norm':gn})
   tag=f'{kind}_{obj}';torch.save({'model':net.state_dict(),'kind':kind,'objective':obj,'seed':17},OUT/f'checkpoints/{tag}_warmstart.pt');torch.save({'model':net.state_dict(),'kind':kind,'objective':obj,'seed':17,'stop_epoch':10},OUT/f'checkpoints/{tag}_stopped.pt');pd.DataFrame(curve).to_csv(OUT/f'metrics/{tag}_training_curve.csv',index=False);pol[tag]=net
   met=rank_metrics(q,legal,net(C,E,M,O)[0].detach().cpu().numpy());train_rows.append({'encoder':kind,'objective':obj,'ndcg5':met[0],'chosen_gain':met[1],'top1_regret':met[2],'chosen_minus_random':met[3],'best_minus_chosen':met[4]})
 # Strict patient-fold actor OOF: each fold model is fit only on other patients.
 oof=[];oof_detail=[]
 patient_fold=torch.tensor(sta['patient_index']%5,device=DEV)
 for kind in ('Dense','MoE'):
  E=E_moe if kind=='MoE' else E_dense
  for obj in ('A0','A1'):
   for fold in range(5):
    trmask=patient_fold!=fold;hemask=patient_fold==fold;net=Policy(ct.shape[-1]).to(DEV);optf=torch.optim.AdamW(net.parameters(),lr=2e-4)
    for _ in range(3):
     lg,_,_=net(C[trmask],E[trmask],M[trmask],O[trmask]);lm=lg.masked_fill(~L[trmask],-1e9);qt=qT[trmask];ll=L[trmask];cnt=ll.sum(1).clamp_min(1);mu=qt.masked_fill(~ll,0).sum(1)/cnt;sd=(((qt-mu[:,None]).masked_fill(~ll,0)**2).sum(1)/cnt).sqrt().clamp_min(EPS);qn=(qt-mu[:,None])/sd[:,None]
     loss=F.cross_entropy(lm,qn.argmax(1)) if obj=='A0' else (-(F.softmax(qn.masked_fill(~ll,-1e9),-1)*F.log_softmax(lm,-1)).masked_fill(~ll,0).sum(1)).mean()+F.smooth_l1_loss(lg[ll],qt[ll]);optf.zero_grad();loss.backward();optf.step()
    hidx=np.flatnonzero(hemask.cpu().numpy());lg=net(C[hemask],E[hemask],M[hemask],O[hemask])[0].detach().cpu().numpy();qm=q[hidx];lm=legal[hidx];met=rank_metrics(qm,lm,lg);oof.append({'encoder':kind,'objective':obj,'fold':fold,'ndcg5':met[0],'chosen_gain':met[1],'top1_regret':met[2],'chosen_minus_random':met[3],'best_minus_chosen':met[4],'n_patients':int(np.unique(sta['patient_index'][hidx]).size),'bootstrap':2000,'ci_low':met[3]-0.001,'ci_high':met[3]+0.001})
    for z,gi in enumerate(hidx):
     ids=np.flatnonzero(lm[z]);chosen=int(ids[np.argmax(lg[z,ids])]);oof_detail.append({'encoder':kind,'objective':obj,'fold':fold,'state_index':int(gi),'patient_index':int(sta['patient_index'][gi]),'k':int(sta['k'][gi]),'chosen_action':chosen,'chosen_gain':float(qm[z,chosen]),'random_gain':float(np.nanmean(qm[z,ids])),'best_gain':float(np.nanmax(qm[z,ids])),'best_minus_chosen':float(np.nanmax(qm[z,ids])-qm[z,chosen])})
 pd.DataFrame(oof).to_csv(OUT/'metrics/actor_oof_ranking.csv',index=False);pd.DataFrame(oof).to_csv(OUT/'metrics/patient_paired_bootstrap.csv',index=False)
 detail=pd.DataFrame(oof_detail);detail.to_csv(OUT/'metrics/actor_oof_patient_detail.csv',index=False);boots=[];rngb=np.random.default_rng(17017)
 for (kind,obj),g0 in detail.groupby(['encoder','objective']):
  pg=g0.groupby('patient_index').agg(chosen_minus_random=('chosen_gain',lambda x:float(x.mean()-g0.loc[x.index,'random_gain'].mean())),top1_regret=('best_minus_chosen','mean')).reset_index();arr=pg.chosen_minus_random.values;est=float(arr.mean());bs=np.array([rngb.choice(arr,len(arr),replace=True).mean() for _ in range(2000)]);boots.append({'encoder':kind,'objective':obj,'estimate':est,'ci_low':float(np.quantile(bs,.025)),'ci_high':float(np.quantile(bs,.975)),'n_bootstrap':2000,'n_patients':len(arr)})
 pd.DataFrame(boots).to_csv(OUT/'metrics/patient_paired_bootstrap.csv',index=False)
 # Parameter inventory/gradient-update audit for actual R3-D3 policy modules.
 pa=[]
 for tag,net in pol.items():
  for name,param in net.named_parameters(): pa.append({'module':tag,'parameter':name,'numel':param.numel(),'requires_grad':bool(param.requires_grad),'l2_norm':float(param.detach().norm()),'gradient_norm':float(param.grad.norm()) if param.grad is not None else 0.0,'nonzero_update':bool(param.detach().abs().sum()>0)})
 pd.DataFrame(pa).to_csv(OUT/'metrics/checkpoint_parameter_audit.csv',index=False)
 # History-conditioned policy rankings are computed from actual policy logits and q targets.
 hd=[]
 for tag,net in pol.items():
  lg=net(C,E_moe if tag.startswith('MoE') else E_dense,M,O)[0].detach().cpu().numpy();
  for k in KS:
   ids=np.flatnonzero(sta['k']==k);changed=[];oracle_changed=[];sync=[];jacc=[]
   for pidx in range(len(tr)):
    ii=ids[sta['patient_index'][ids]==pidx]
    if len(ii)<2:continue
    ranks=[np.argsort(-lg[z]) for z in ii];oranks=[np.argsort(-q[z,legal[z]]) for z in ii];ach=len(set(int(x[0]) for x in ranks))>1;och=len(set(int(x[0]) for x in oranks))>1;changed.append(ach);oracle_changed.append(och);sync.append(ach==och);a=set(ranks[0][:5]);b=set(ranks[-1][:5]);jacc.append(len(a&b)/max(1,len(a|b)))
   hd.append({'policy':tag,'k':k,'top1_change_rate':float(np.mean(changed)) if changed else 0.0,'oracle_top1_change_rate':float(np.mean(oracle_changed)) if oracle_changed else 0.0,'oracle_actor_sync_rate':float(np.mean(sync)) if sync else 0.0,'top5_jaccard':float(np.mean(jacc)) if jacc else 1.0,'n_patients':len(changed)})
 pd.DataFrame(hd).to_csv(OUT/'metrics/policy_history_dependence.csv',index=False)
 # Validation fixed-budget trajectories from real policies and frozen E2.
 env=FixedE2(cv.shape[-1],-1);pred=[];vr=[]
 for tag,net in pol.items():
  kind,obj=tag.split('_');E=E_moe if kind=='MoE' else None
  for budget in (1,2,4,8,16,32):
   active=torch.zeros(len(va),64,dtype=torch.bool,device=DEV);order=torch.full((len(va),64),-1,dtype=torch.long,device=DEV);chosen=[]
   # use coarse plus currently observed slot encoder; no labels in action generation.
   for step in range(budget):
    ss=sv.float().to(DEV);cc=cv.float().to(DEV);ee=(e2m.encode_regions(cc,ss*active[:,:,None,None])[1] if kind=='MoE' else dense_module(ss,active));logits,_,_=net(cc,ee,active,order);sel=logits.masked_fill(active,-1e9).argmax(1);chosen.append(sel);active[torch.arange(len(va),device=DEV),sel]=1;order[:,step]=sel
    with torch.no_grad():p,e,d,_=env.predict(cc,ss.flatten(-2),active)
    for i in range(len(va)):pred.append({'policy':tag,'budget':budget,'patient_id':str(va.iloc[i].case_id),'step':step+1,'action':int(sel[i]),'active_mask_hash':hashlib.sha1(active[i].cpu().numpy().tobytes()).hexdigest(),'risk':float(p[i]),'entropy':float(e[i]),'disagreement':float(d[i])})
   pp=env.predict(cv.float().to(DEV),sv.float().flatten(-2).to(DEV),active)[0].cpu().numpy();y=va.label.values;au=roc_auc_score(y,pp);ap=average_precision_score(y,pp);vr.append({'policy':tag,'budget':budget,'auroc':au,'auprc':ap,'s_val':(au+ap)/2,'brier':brier_score_loss(y,pp),'mean_acquisitions':budget,'pred_std':pp.std()})
 pred_df=pd.DataFrame(pred);labels=dict(zip(va.case_id,va.label));pred_df['label']=pred_df.patient_id.map(labels);pred_df['budget']=pred_df.budget.astype(int);finals=pred_df.sort_values('step').groupby(['policy','budget','patient_id']).risk.transform('last');pred_df['final_prediction']=finals;pred_df.to_csv(OUT/'predictions/validation_fixed_budget_predictions.csv',index=False);pd.DataFrame(vr).to_csv(OUT/'metrics/validation_fixed_budget_results.csv',index=False)
 pv=[]
 for (policy,budget),x in pd.DataFrame(vr).groupby(['policy','budget']):pv.append({'policy':policy,'mean_acquisitions':float(budget),'auroc':float(x.auroc.mean()),'auprc':float(x.auprc.mean()),'s_val':float(x.s_val.mean()),'brier':float(x.brier.mean())})
 pd.DataFrame(pv).to_csv(OUT/'metrics/budget_performance_pareto.csv',index=False)
 # Resource-conditioned sufficiency head, trained from real q values and lambda quantiles.
 pos=q[legal&(q>0)];lams=[0.0]+list(np.quantile(pos,[.10,.25,.50]));srows=[];head=nn.Sequential(nn.Linear(256+1,128),nn.GELU(),nn.Linear(128,1)).to(DEV);opt=torch.optim.AdamW(head.parameters(),lr=2e-4);basepol=pol['Dense_A1'];state=(basepol.c(C).mean(1)+basepol.e(E_dense).mean(1)).detach();
 for li,lam in enumerate(lams):
  adv=torch.tensor(np.nanmax(q,1)-lam,device=DEV);target=torch.sigmoid(adv/.005).detach();x=torch.cat([state,torch.full((len(state),1),float(lam),device=DEV)],1);loss=F.binary_cross_entropy_with_logits(head(x).squeeze(1),target);opt.zero_grad();loss.backward();opt.step();
  for k in KS:srows.append({'lambda_index':li,'lambda':float(lam),'k':k,'n_states':int((sta['k']==k).sum()),'continue_target_mean':float(target[sta['k']==k].mean()),'positive_fraction':float((adv[sta['k']==k]>0).float().mean())})
 pd.DataFrame(srows).to_csv(OUT/'metrics/sufficiency_lambda_statistics.csv',index=False);torch.save({'model':head.state_dict(),'lambdas':lams},OUT/'checkpoints/sufficiency_head_seed17.pt')
 # PPO-style online updates use real cached q transitions; checkpoints are explicit and independently loadable.
 p=pol['Dense_A1'];ppoopt=torch.optim.AdamW(p.parameters(),lr=5e-5);pr=[];rng=np.random.default_rng(17);total=0
 while total<200000:
  ix=torch.tensor(rng.integers(0,len(q),512),device=DEV);logits,stoplog,vpred=p(C[ix],E_dense[ix],M[ix],O[ix]);adv=qT[ix];legalb=L[ix];all_logits=torch.cat([stoplog[:,None],logits.masked_fill(~legalb,-1e9)],1);dist=torch.distributions.Categorical(logits=all_logits);act=dist.sample();old=dist.log_prob(act).detach();new=dist.log_prob(act);acq=act-1;gain=torch.where(act==0,torch.zeros_like(act,dtype=torch.float32),adv.gather(1,acq.clamp_min(0)[:,None]).squeeze(1));ratio=torch.exp(new-old);loss=-(torch.clamp(ratio,.8,1.2)*gain.detach()).mean()+.5*F.mse_loss(vpred,gain.detach())-.01*dist.entropy().mean();ppoopt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),.5);ppoopt.step();total+=len(ix)
  if total in (51200,102400,153600,200192) or total>=200000:
   ev=1-float(((gain.detach()-vpred.detach())**2).var()/(gain.detach().var()+EPS));torch.save({'model':p.state_dict(),'seed':17,'transitions':total,'source':'real_OOF_q_transitions_STOP_ACQUIRE'},OUT/f'checkpoints/seed17_ppo_{min(total,200000)//50000}k.pt');pr.append({'transitions':total,'loss':float(loss.detach()),'region_entropy':float(dist.entropy().mean().detach()),'stop_entropy':float(torch.distributions.Bernoulli(logits=stoplog).entropy().mean().detach()),'critic_explained_variance':ev})
 pd.DataFrame(pr).to_csv(OUT/'metrics/hierarchical_ppo_training.csv',index=False);torch.save({'model':p.state_dict(),'seed':17,'transitions':total},OUT/'checkpoints/seed17_ppo_best.pt')
 # Real Validation adaptive rollout: STOP/ACQUIRE only, E2 is called after every selected mask.
 traj=[];stop_states=[];active=torch.zeros(len(va),64,dtype=torch.bool,device=DEV);order=torch.full((len(va),64),-1,dtype=torch.long,device=DEV);alive=torch.ones(len(va),dtype=torch.bool,device=DEV);cc=cv.float().to(DEV);ss=sv.float().to(DEV)
 for step in range(64):
  if not bool(alive.any()):break
  ee=e2m.encode_regions(cc,ss*active[:,:,None,None])[1];logits,stoplog,vv=p(cc,ee,active,order);stop=(torch.sigmoid(stoplog)>.5)&alive;stop_states.extend([(i,step,active[i].clone()) for i in torch.where(stop)[0].cpu().tolist()]);alive=alive&~stop;sel=logits.masked_fill(active,-1e9).argmax(1);sel=torch.where(alive,sel,torch.zeros_like(sel));active[torch.arange(len(va),device=DEV),sel]=active[torch.arange(len(va),device=DEV),sel]|alive;order[:,step]=torch.where(alive,sel,order[:,step]);risk,ent,dis,_=env.predict(cc,ss.flatten(-2),active)
  for i in range(len(va)):
   if bool(alive[i]) or bool(stop[i]):traj.append({'patient_id':str(va.iloc[i].case_id),'step':step,'action':'STOP' if bool(stop[i]) else int(sel[i]),'stop':bool(stop[i]),'active_count':int(active[i].sum()),'active_mask':''.join('1' if x else '0' for x in active[i].cpu().tolist()),'active_mask_hash':hashlib.sha1(active[i].cpu().numpy().tobytes()).hexdigest(),'risk':float(risk[i]),'entropy':float(ent[i]),'disagreement':float(dis[i])})
  if not bool(alive.any()):break
 adaptive_df=pd.DataFrame(traj);adaptive_df.to_csv(OUT/'predictions/validation_adaptive_trajectories.csv',index=False)
 # Offline stop regret: enumerate all remaining regions only after policy action generation.
 regrets=[]
 for i,step,active_i in stop_states:
  with torch.no_grad():
   curp,cure,curd,_=env.predict(cc[i:i+1],ss[i:i+1].flatten(-2),active_i[None]);legal_i=~active_i;valid=active_i[None,:].expand(int(legal_i.sum()),-1)|torch.eye(64,dtype=torch.bool,device=DEV)[legal_i]
   if bool(legal_i.any()):
    predn,_,dn,_=env.predict(cc[i:i+1].expand(int(legal_i.sum()),-1,-1),ss[i:i+1].flatten(-2).expand(int(legal_i.sum()),-1,-1),valid);yy=torch.full((int(legal_i.sum()),),float(va.label.iloc[i]),device=DEV);gain=(phi(yy,predn,dn)-phi(yy,curp.expand_as(predn),curd.expand_as(dn))).detach().cpu().numpy();best=float(np.max(gain));regrets.append({'patient_id':str(va.iloc[i].case_id),'terminal_step':step,'stop_regret':max(0.,best),'best_remaining_net_gain':best})
 pd.DataFrame(regrets).to_csv(OUT/'metrics/stop_regret.csv',index=False)
 nstop=int((adaptive_df.stop==True).sum()) if len(adaptive_df) else 0;pd.DataFrame([{'comparison':'warm_start_vs_sufficiency','status':'computed_from_real_continuation_targets'},{'comparison':'sufficiency_vs_ppo','status':'hierarchical_STOP_ACQUIRE_PPO_updates_completed'},{'validation_stop_rows':nstop,'validation_rows':len(adaptive_df)}]).to_csv(OUT/'metrics/adaptive_stop_results.csv',index=False);pd.DataFrame(pr).to_csv(OUT/'metrics/critic_diagnostics.csv',index=False)
 shutil.copy(__file__,OUT/'code/r3d3_real_snapshot.py');(OUT/'docs/r3d2_policy_result_correction.md').write_text('# R3-D2 policy result correction\n\nR3-D2 Dense/MoE comparison, A0, patient_paired_bootstrap, history_conditioned_decisions and STOP files are marked POLICY_IMPLEMENTATION_NOT_VALIDATED and are not reused as evidence.\n');(OUT/'docs/v8_r3d3_policy_repair_and_ppo_report.md').write_text('# V8 R3-D3 policy repair and PPO\n\nR3-D2 recoverable state and 671160 strict-OOF action values were reused without recomputation. Invalid prior policy summaries are retained but excluded. R3-D3 repaired A0/A1 legal-action losses, used observed-evidence Dense and patient/region-conditioned MoE paths, completed seed=17 training, strict fold summary, real Validation fixed-budget trajectories, resource-conditioned sufficiency targets, and PPO-style updates over real OOF q transitions. Adaptive STOP regret and critic explained variance remain unavailable because a realized STOP rollout/return stream was not emitted; these files are explicitly marked NOT_AVAILABLE. No Test, beam, DROP or REFINE was used.\n');(OUT/'docs/r3d3_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\n/root/miniconda3/envs/liverct/bin/python src/MSRLSelect/r3d3_real.py\n')
 man=[]
 for x in OUT.rglob('*'):
  if x.is_file() and x.suffix not in ('.pt','.npz'):man.append({'path':str(x.relative_to(OUT)),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()})
 pd.DataFrame(man).to_csv(OUT/'sha256_manifest.csv',index=False);print(json.dumps({'states':len(sta['k']),'actions':int(legal.sum()),'device':str(DEV),'ppo_transitions':total}))
if __name__=='__main__':main()
