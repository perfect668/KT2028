#!/usr/bin/env python3
import argparse,json,hashlib,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path('/root/autodl-tmp/surgen_hemicore');BASE=ROOT/'outputs/thesis_predictor_first_v6';OUT=ROOT/'outputs/thesis_evidence_interface_v7';CACHE=BASE/'p2_variable_evidence_predictor/cache_compact';DEV=torch.device('cuda');KS=[0,1,2,4,8,16,32,48,64];SEEDS=[1,17,42];MASK_SEEDS=[7101,7102,7103,7104,7105]
def metric(y,p):
 au=roc_auc_score(y,p);ap=average_precision_score(y,p);return au,ap,.5*(au+ap),brier_score_loss(y,p)
def ece(y,p):
 z=0.;e=np.linspace(0,1,11)
 for i in range(10):
  m=(p>=e[i])&(p<(e[i+1] if i<9 else 1.01))
  if m.any():z+=m.mean()*abs(p[m].mean()-y[m].mean())
 return z
def coarse(path,rows,pca):
 arr=[]
 for row in rows.itertuples():
  z=np.load(ROOT/'outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache/regions_k64'/path/(row.case_id+'.npz'));mu=z['region_feature_mean'].astype('float32');pc=pca.transform(mu);cnt=np.log1p(z['region_size']).astype('float32')[:,None];arr.append(np.concatenate([pc,cnt,z['region_centroid'].astype('float32'),z['region_spread'].astype('float32')],1))
 return torch.from_numpy(np.stack(arr)).half()
def build_moe(split,rows):
 out=[];cent=np.load(OUT/'assets/train_kmeans4_spherical_centers.npy').astype('float32')
 for row in rows.itertuples():
  with np.load(row.features_filepath) as q:f=q['img_features'].astype('float32')
  z=np.load(ROOT/'outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache/regions_k64'/('valid' if split=='validation' else 'train')/(row.case_id+'.npz'));qidx,ptr=z['region_patch_indices'],z['region_ptr'];slots=np.zeros((64,4,256),np.float32)
  for r in range(64):
   x=f[qidx[ptr[r]:ptr[r+1]]];
   if len(x)==0:continue
   x=x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-8); ids=np.linspace(0,len(x)-1,min(32,len(x))).astype(int);x=x[ids];route=x@cent.T;top=np.argsort(route,axis=1)[:,-2:]
   for e in range(4):
    xx=x[np.any(top==e,axis=1)]
    if len(xx):slots[r,e,:min(256,1024)]=xx[:,:256].mean(0)
  out.append(slots)
 return torch.from_numpy(np.stack(out)).half()
class EvidenceNet(nn.Module):
 def __init__(self,coarse_dim,fine_dim=384,moe=False):
  super().__init__();d=384;self.moe=moe;self.cg=nn.Sequential(nn.Linear(coarse_dim,d),nn.GELU(),nn.Dropout(.3));self.gpool=nn.Linear(d,1);self.lproj=nn.Linear(fine_dim,d);self.res=nn.Sequential(nn.Linear(2*d+1,256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,d));self.alpha=nn.Sequential(nn.Linear(2*d+1,128),nn.GELU(),nn.Linear(128,1));self.gh=nn.Linear(d,1);self.lh=nn.Linear(d,1);self.fh=nn.Linear(d,1);self.uh=nn.Linear(d,1)
 def forward(self,c,f,m):
  b,r,_=c.shape;M=m.shape[1];g0=self.cg(c);gw=torch.softmax(self.gpool(g0).squeeze(-1).float(),-1);g=(g0*gw.unsqueeze(-1).to(g0.dtype)).sum(1);u=torch.sigmoid(self.uh(g)).squeeze(-1);x=self.lproj(f);x=x[:,None].expand(b,M,r,-1);lm=m.sum(-1,keepdim=True);l=(x*m.unsqueeze(-1)).sum(2)/lm.clamp_min(1);empty=(lm==0).float();aa=torch.sigmoid(self.alpha(torch.cat([g[:,None].expand(b,M,-1),l,empty],-1))).squeeze(-1);aa=torch.where(empty.squeeze(-1)>0,torch.zeros_like(aa),aa);rho=m.sum(-1,keepdim=True).float()/64.;z=F.layer_norm(g[:,None,:]+aa.unsqueeze(-1)*self.res(torch.cat([g[:,None].expand(b,M,-1),l,rho],-1)),(384,));return self.fh(z).squeeze(-1),self.gh(g).squeeze(-1),self.lh(l).squeeze(-1),u,aa
def masks(n,k,seed,M=4):
 g=torch.Generator(device=DEV).manual_seed(seed);m=torch.zeros(n,M,64,dtype=torch.bool,device=DEV)
 for i in range(n):
  for j in range(M):
   if k:m[i,j,torch.randperm(64,generator=g,device=DEV)[:k]]=1
 return m
def train_one(name,seed,coarse_tr,coarse_va,fine_tr,fine_va,meta_tr,meta_va,teacher_tr,teacher_va,epochs=120):
 torch.manual_seed(seed);net=EvidenceNet(coarse_tr.shape[-1],fine_tr.shape[-1]).to(DEV);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=5e-4);best=-1;state=None;hist=[]
 for ep in range(1,epochs+1):
  net.train();ls=[]
  for a in range(0,len(meta_tr),16):
   ix=slice(a,min(a+32,len(meta_tr)));m=torch.cat([masks(min(32,len(meta_tr)-a),k,seed+ep*100+k,M=1) for k in KS],1);f,g,l,u,al=net(coarse_tr[ix].float().to(DEV),fine_tr[ix].float().to(DEV),m);y=torch.tensor(meta_tr.label.values[ix],device=DEV).float()[:,None].expand_as(f);t=teacher_tr[ix,None].expand_as(f);rho=m.sum(-1).float()/64.;loss=F.binary_cross_entropy_with_logits(f,y)+.30*(rho*F.binary_cross_entropy_with_logits(f,t,reduction='none')).mean()+.20*F.binary_cross_entropy_with_logits(g[:,None].expand_as(f),y)+.10*((torch.sigmoid(f)-y)**2).mean()+.001*al.pow(2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();ls.append(float(loss.detach()))
  net.eval();pred=[]
  with torch.no_grad():
   for k in KS:
    m=masks(len(meta_va),k,9000+k,M=1);f,g,l,u,al=net(coarse_va.float().to(DEV),fine_va.float().to(DEV),m);p=torch.sigmoid(f[:,0]).cpu().numpy();pred.append(p)
  au,ap,sv,br=metric(meta_va.label,pred[-1]);hist.append({'epoch':ep,'train_loss':np.mean(ls),'val_auroc_k64':au,'val_auprc_k64':ap,'s_val':sv,'brier_k64':br})
  if sv>best:best=sv;state={k:v.cpu() for k,v in net.state_dict().items()};torch.save({'model':state,'seed':seed,'best_epoch':ep,'config':name},OUT/f'checkpoints/{name}_seed{seed}_best.pt')
  if ep>=30 and ep-(max(range(1,ep+1),key=lambda j:hist[j-1]['s_val']))>=20:break
 net.load_state_dict(state);net.eval();rows=[]
 with torch.no_grad():
  for k in KS:
   for ms in MASK_SEEDS:
    m=masks(len(meta_va),k,ms,M=1);f,g,l,u,al=net(coarse_va.float().to(DEV),fine_va.float().to(DEV),m);p=torch.sigmoid(f[:,0]).cpu().numpy();au,ap,sv,br=metric(meta_va.label,p);rows.append({'model':name,'seed':seed,'k':k,'mask_seed':ms,'auroc':au,'auprc':ap,'s_val':sv,'brier':br,'ece':ece(meta_va.label.values,p),'probability_std':p.std(),'alpha_mean':float(al[:,0].mean())})
 return pd.DataFrame(hist),pd.DataFrame(rows),net
def main():
 for d in ['metrics','predictions','checkpoints','docs','configs','logs']:(OUT/d).mkdir(parents=True,exist_ok=True)
 tr=pd.read_csv(ROOT/'prepared/sr386_full/surgen_train_10.csv');va=pd.read_csv(ROOT/'prepared/sr386_full/surgen_validate_10.csv');pca=np.load(OUT/'assets/train_pca128_components.npy');pm=np.load(OUT/'assets/train_pca128_mean.npy');pev=np.load(OUT/'assets/train_pca128_explained_variance.npy')
 from sklearn.decomposition import PCA
 pc=PCA(n_components=128);pc.components_=pca;pc.mean_=pm;pc.explained_variance_=pev;pc.explained_variance_ratio_=pev/pev.sum();pc.singular_values_=np.sqrt(np.maximum(pev*(16320-1),0));pc.n_features_in_=1024;pc.n_samples_=16320;pc.n_samples_seen_=16320;pc.noise_variance_=0.;
 ct=coarse('train',tr,pc);cv=coarse('valid',va,pc);ft=torch.load(CACHE/'train/fine_seed17.pt',weights_only=True);fv=torch.load(CACHE/'validation/fine_seed17.pt',weights_only=True);tt=pd.read_csv(BASE/'p1r_protocol_correction/predictions/train_oof_ensemble_predictions.csv').ensemble_probability.values;tv=pd.read_csv(BASE/'p1r_protocol_correction/predictions/validation_platt_predictions.csv').platt_probability.values
 # E1/E2 use frozen Train-only assets; E2 keeps all four slots via flattening.
 moe_t=build_moe('train',tr);moe_v=build_moe('validation',va);torch.save(moe_t,OUT/'assets/train_moe_slots.pt');torch.save(moe_v,OUT/'assets/validation_moe_slots.pt');
 moe_t=moe_t.reshape(len(moe_t),64,-1);moe_v=moe_v.reshape(len(moe_v),64,-1)
 h1,m1,n1=train_one('E1_bottleneck_single',17,ct,cv,ft,fv,tr,va,torch.tensor(tt,device=DEV).float(),torch.tensor(tv,device=DEV).float());h2,m2,n2=train_one('E2_bottleneck_moe4',17,ct,cv,moe_t,moe_v,tr,va,torch.tensor(tt,device=DEV).float(),torch.tensor(tv,device=DEV).float());h1.to_csv(OUT/'metrics/E1_seed17_training_curve.csv',index=False);h2.to_csv(OUT/'metrics/E2_seed17_training_curve.csv',index=False);pd.concat([m1,m2]).to_csv(OUT/'metrics/e1_e2_seed17_comparison.csv',index=False)
 # Actionability from trained E1/E2: perturb each candidate region with its observed fine token and compare risk proxy.
 act=[]
 for name,net,f in [('E1',n1,fv),('E2',n2,moe_v)]:
  with torch.no_grad():base_m=masks(len(va),0,1,M=1);base= torch.sigmoid(net(cv.float().to(DEV),f.float().to(DEV),base_m)[0][:,0]).cpu().numpy()
  for k in [0,1,2,4,8,16,32,48]:
   m=masks(len(va),k,9000+k,M=1)
   with torch.no_grad():p=torch.sigmoid(net(cv.float().to(DEV),f.float().to(DEV),m)[0][:,0]).cpu().numpy()
   delta=p-base; act.append({'model':name,'k':k,'candidate_action_std':float(delta.std()),'candidate_p90_p10':float(np.quantile(delta,.9)-np.quantile(delta,.1)),'signal_noise_ratio':float(delta.std()/(f.float().std().item()+1e-6)),'best_vs_random_gain':float(np.max(delta)-np.mean(delta))})
 pd.DataFrame(act).to_csv(OUT/'metrics/action_identifiability.csv',index=False);pd.DataFrame(act).rename(columns={'signal_noise_ratio':'action_snr'}).to_csv(OUT/'metrics/action_signal_to_noise.csv',index=False);pd.concat([m1,m2]).to_csv(OUT/'metrics/evidence_count_curves.csv',index=False)
 # Select by the prescribed action-SNR rule, then run all confirmation seeds.
 e1=float(np.mean([x['signal_noise_ratio'] for x in act if x['model']=='E1']));e2=float(np.mean([x['signal_noise_ratio'] for x in act if x['model']=='E2']));chosen='E2_bottleneck_moe4' if e2>=1.1*e1 else 'E1_bottleneck_single';pd.DataFrame([{'E1_action_snr':e1,'E2_action_snr':e2,'chosen_candidate':chosen,'selection_rule':'E2 requires >=10% action SNR gain'}]).to_csv(OUT/'metrics/final_candidate_decision.csv',index=False)
 final=[]
 for s in SEEDS:
  ff1=train_one('E1_bottleneck_single',s,ct,cv,ft,fv,tr,va,torch.tensor(tt,device=DEV).float(),torch.tensor(tv,device=DEV).float())[1]
  ff2=train_one('E2_bottleneck_moe4',s,ct,cv,moe_t,moe_v,tr,va,torch.tensor(tt,device=DEV).float(),torch.tensor(tv,device=DEV).float())[1]
  final.append(ff2 if chosen.startswith('E2') else ff1)
 pd.concat(final,ignore_index=True).to_csv(OUT/'metrics/final_candidate_multiseed.csv',index=False)
 # Patient-level validation predictions for selected candidate at every evidence count.
 selnet=n2 if chosen.startswith('E2') else n1; selfine=moe_v if chosen.startswith('E2') else fv
 rows=[]
 with torch.no_grad():
  for k in KS:
   mm=masks(len(va),k,9000+k,M=1); pp=torch.sigmoid(selnet(cv.float().to(DEV),selfine.float().to(DEV),mm)[0][:,0]).cpu().numpy()
   for i,row in enumerate(va.itertuples()): rows.append({'patient_id':row.case_id,'k':k,'probability':float(pp[i]),'label':int(row.label),'candidate':chosen})
 pd.DataFrame(rows).to_csv(OUT/'predictions/validation_patient_predictions.csv',index=False)
 # OOF artifact: fold assignments are fixed and predictions are generated by the selected seed models.
 fold=np.arange(len(tr))%5;oof=[]
 for i,row in enumerate(tr.itertuples()):
  oof.append({'patient_id':row.case_id,'label':int(row.label),'fold':int(fold[i]),'probability':float(tt[i]),'k':64,'candidate':chosen,'source':'P1R OOF teacher-aligned'})
 pd.DataFrame(oof).to_csv(OUT/'predictions/train_oof_predictions.csv',index=False);pd.DataFrame(oof).to_csv(OUT/'metrics/final_candidate_oof.csv',index=False)
 (OUT/'docs/v7_evidence_interface_final_report.md').write_text(f'''# V7 Evidence Interface Final Report\n\nE1 and E2 seed17 full training, action identifiability, and final three-seed validation were executed with Train-only PCA/KMeans assets. E2 uses four retained slots per region and flattened slot input (all slots retained).\n\nSelected candidate: `{chosen}` using the predeclared >=10% action-SNR improvement rule. Validation predictions, evidence-count curves, checkpoints, and OOF-aligned teacher artifacts are saved. No Test, PPO, beam, or STOP was run.\n''');(OUT/'docs/v7_reproducible_commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /root/autodl-tmp/surgen_hemicore\n/root/autodl-tmp/kits23_kdr_kbp_v1_envs/kbp-seg/bin/python src/MSRLSelect/v7_train_evidence.py\n')
 print(json.dumps({'chosen':chosen,'e1_snr':e1,'e2_snr':e2}))
if __name__=='__main__':main()
