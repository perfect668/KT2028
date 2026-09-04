#!/usr/bin/env python3
"""Validation benchmark for corrected formal Dense PPO (Validation only)."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from MSRLSelect.rl_core.assets import load_train_assets, FixedE2ValidationEnsemble
from MSRLSelect.rl_core.core import DenseActorCritic, evidence_potential

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def metric(y,p):
    au=roc_auc_score(y,p); ap=average_precision_score(y,p)
    return {'auroc':float(au),'auprc':float(ap),'s_val':float((au+ap)/2),'brier':float(brier_score_loss(y,p))}

def make_view(root, train_assets, val):
    ev=root/'outputs/thesis_evidence_interface_v7'; comp=np.load(ev/'assets/train_pca128_components.npy'); mean=np.load(ev/'assets/train_pca128_mean.npy'); var=np.load(ev/'assets/train_pca128_explained_variance.npy')
    pc=PCA(128); pc.components_,pc.mean_,pc.explained_variance_=comp,mean,var; pc.explained_variance_ratio_=var/var.sum(); pc.singular_values_=np.sqrt(np.maximum(var*(16320-1),0)); pc.n_features_in_,pc.n_samples_,pc.n_samples_seen_,pc.noise_variance_=1024,16320,16320,0
    rows=[]; rr=root/'outputs/thesis_algorithm_upgrade_v5/stage1a_infrastructure/cache/regions_k64/valid'
    for r in val.itertuples():
        with np.load(rr/f'{r.case_id}.npz') as z: rows.append(np.concatenate([pc.transform(z['region_feature_mean'].astype('float32')),np.log1p(z['region_size']).astype('float32')[:,None],z['region_centroid'],z['region_spread']],1))
    class V: pass
    v=V(); v.root=root; v.metadata=val; v.coarse=torch.from_numpy(np.stack(rows).astype('float32')); v.fine_slots=torch.load(ev/'assets/validation_moe_slots.pt',map_location='cpu',weights_only=True).float(); v.calibration=train_assets.calibration; v.calibration_path=train_assets.calibration_path; v.calibration_sha256=train_assets.calibration_sha256
    v.checkpoint_paths=[root/'outputs/thesis_evidence_interface_v7/checkpoints'/f'E2_bottleneck_moe4_seed{s}_best.pt' for s in (1,17,42)]; v.opened_paths=[]
    return v

def costs(view, ids, device):
    x=view.coarse[ids.cpu()].to(device)[:,:,128]; return (x-x.amin(1,keepdim=True))/(x.amax(1,keepdim=True)-x.amin(1,keepdim=True)).clamp_min(1e-6)

@torch.no_grad()
def fixed(view,predictor,policy,ids,budget,device,method,seed=0):
    n=len(ids); active=torch.zeros(n,64,dtype=torch.bool,device=device); order=torch.zeros(n,64,dtype=torch.long,device=device); total=torch.zeros(n,dtype=torch.float64,device=device); rows=[]
    for step in range(budget):
        if method=='random':
            action=torch.stack([torch.where(~active[i])[0][torch.randint(int((~active[i]).sum()),(1,),device=device)[0]] for i in range(n)])
        elif method=='coarse': action=view.coarse[ids.cpu()].to(device).norm(-1).masked_fill(active,-1e9).argmax(1)
        elif method=='attention':
            cb=view.coarse[ids.cpu()].to(device); weights=[]
            for model in predictor.models:
                z=model.cg(cb); weights.append(torch.softmax(model.gpool(z).squeeze(-1).float(),-1))
            action=torch.stack(weights).mean(0).masked_fill(active,-1e9).argmax(1)
        else:
            cb=view.coarse[ids.cpu()].to(device); fs=view.fine_slots[ids.cpu()].to(device); risk,ent,dis,_=predictor.predict(ids,active); ob={'patient_index':ids,'coarse':cb,'fine_slots':fs*active[:,:,None,None].to(fs.dtype),'active_mask':active,'acquire_order':order,'risk':risk,'entropy':ent,'disagreement':dis,'step':active.sum(1),'cumulative_cost':total.float(),'lambda':torch.zeros(n,device=device),'remaining':(~active).sum(1)}; action=policy(ob)[0][:,:64].masked_fill(active,-1e9).argmax(1)
        c=costs(view,ids,device)[torch.arange(n,device=device),action]; active[torch.arange(n,device=device),action]=True; order[torch.arange(n,device=device),action]=step+1; total+=c.double(); risk,ent,dis,_=predictor.predict(ids,active)
        for i in range(n): rows.append({'method':method,'patient_id':str(view.metadata.iloc[int(ids[i])].case_id),'step':step+1,'action':int(action[i]),'active_count':int(active[i].sum()),'risk':float(risk[i]),'entropy':float(ent[i]),'disagreement':float(dis[i]),'cost':float(c[i]),'cumulative_cost':float(total[i]),'action_type':'ACQUIRE','terminal_reason':'max_steps' if step+1==budget else ''})
    return active,total,rows

@torch.no_grad()
def adaptive(view,predictor,policy,ids,lam,device):
    n=len(ids); active=torch.zeros(n,64,dtype=torch.bool,device=device); order=torch.zeros(n,64,dtype=torch.long,device=device); total=torch.zeros(n,dtype=torch.float64,device=device); alive=torch.ones(n,dtype=torch.bool,device=device); rows=[]
    for step in range(8):
        ix=torch.where(alive)[0]
        if not len(ix): break
        risk,ent,dis,_=predictor.predict(ids[ix],active[ix]); cb=view.coarse[ids[ix].cpu()].to(device); fs=view.fine_slots[ids[ix].cpu()].to(device); ob={'patient_index':ids[ix],'coarse':cb,'fine_slots':fs*active[ix,:,None,None].to(fs.dtype),'active_mask':active[ix],'acquire_order':order[ix],'risk':risk,'entropy':ent,'disagreement':dis,'step':active[ix].sum(1),'cumulative_cost':total[ix].float(),'lambda':torch.full((len(ix),),float(lam),device=device),'remaining':(~active[ix]).sum(1)}; logits,_=policy(ob); probs=torch.softmax(logits,1); act=logits.argmax(1); stop=act==64
        for j,row in enumerate(ix.tolist()): rows.append({'method':'DensePPO_Adaptive','lambda':float(lam),'patient_id':str(view.metadata.iloc[row].case_id),'step':step,'action':int(act[j]),'action_type':'STOP' if stop[j] else 'ACQUIRE','action_probability':float(probs[j,act[j]]),'stop_probability':float(probs[j,64]),'active_count':int(active[row].sum()),'active_mask':active[row].cpu().tolist(),'risk':float(risk[j]),'entropy':float(ent[j]),'disagreement':float(dis[j]),'cost':0.0,'cumulative_cost':float(total[row]),'terminal_reason':'STOP' if stop[j] else ''})
        acq=ix[~stop]; aa=act[~stop]
        if len(acq):
            c=costs(view,ids[acq],device)[torch.arange(len(acq),device=device),aa]; active[acq,aa]=True; order[acq,aa]=step+1; total[acq]+=c.double()
        alive[ix[stop]]=False; alive[ix[~stop] & (active[ix[~stop]].sum(1)>=64)]=False
    return active,total,rows

def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--data-root',required=True); pa.add_argument('--output-dir',required=True); pa.add_argument('--checkpoint',required=True); pa.add_argument('--device',default='cuda'); a=pa.parse_args(); root=Path(a.data_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    ck=torch.load(a.checkpoint,map_location=device,weights_only=False); assets=load_train_assets(root); val=pd.read_csv(root/'prepared/sr386_full/surgen_validate_10.csv'); view=make_view(root,assets,val); assert len(view.checkpoint_paths)==3 and all('oof_fold' not in str(x).lower() and 'e2m' not in str(x).lower() for x in view.checkpoint_paths); predictor=FixedE2ValidationEnsemble(view,device,view.checkpoint_paths); ids=torch.arange(len(val),device=device); policy=DenseActorCritic(view.coarse.shape[-1]).to(device); policy.load_state_dict(ck['model']); policy.eval(); y=val.label.to_numpy(); metric_rows=[]; pred_rows=[]; traj=[]; stop_rows=[]
    def add(method,p,budget,total=None,seed=0,lam=np.nan,reason=''):
        z=metric(y,p); b=np.asarray(budget if not np.isscalar(budget) else np.full(len(y),budget)); c=np.asarray(total if total is not None else np.zeros(len(y))); z.update({'method':method,'mean_acquisitions':float(b.mean()),'acquisition_std':float(b.std()),'mean_cost':float(c.mean()),'immediate_stop_fraction':float(np.mean(b==0)),'reached_8_fraction':float(np.mean(b>=8))}); metric_rows.append(z)
        for i in range(len(y)): pred_rows.append({'patient_id':str(val.iloc[i].case_id),'method':method,'lambda':lam,'random_seed':seed,'prediction':float(p[i]),'acquisition_count':int(b[i]),'cumulative_cost':float(c[i]),'terminal_reason':reason})
    for k,name in ((0,'E2_K0'),(64,'E2_Full64')):
        am=torch.full((len(val),64),k==64,dtype=torch.bool,device=device); pp=predictor.predict(ids,am)[0].cpu().numpy(); add(name,pp,k)
    for budget in (4,8):
        for meth,label in (('random','Random20'),('coarse','coarse-only'),('attention','AttentionTopK'),('policy','DensePPO')):
            reps=[]; costs_reps=[]
            for seed in range(20 if meth=='random' else 1):
                torch.manual_seed(seed); am,co,rr=fixed(view,predictor,policy,ids,budget,device,meth,seed); reps.append(predictor.predict(ids,am)[0].cpu().numpy()); costs_reps.append(co.cpu().numpy()); traj.extend(rr)
            for seed,pp in enumerate(reps): add(f'{label}_K{budget}'+(f'_seed{seed}' if meth=='random' else ''),pp,budget,costs_reps[seed],seed)
            if meth=='random': add(f'{label}_K{budget}_mean',np.mean(reps,0),budget,np.mean(costs_reps,0))
    lams=ck.get('lambda_values'); lambda_path=Path(a.checkpoint).parents[1]/'lambda_values.json'
    if lams is None or not lambda_path.is_file() or lams != json.loads(lambda_path.read_text())['values']: raise AssertionError('checkpoint/JSON lambda mismatch')
    budgets_by_lam={}
    for lam in lams:
        am,co,rr=adaptive(view,predictor,policy,ids,float(lam),device); pp=predictor.predict(ids,am)[0].cpu().numpy(); b=am.sum(1).cpu().numpy(); budgets_by_lam[float(lam)]=b; traj.extend(rr); add(f'DensePPO_Adaptive_lambda{float(lam):.8g}',pp,b,co.cpu().numpy(),lam=float(lam))
        labels=torch.tensor(y,dtype=torch.float64,device=device)
        for row in rr:
            if row['action_type']!='STOP': continue
            i=int(val.index[val.case_id.astype(str)==row['patient_id']][0]); mask=torch.tensor(row['active_mask'],dtype=torch.bool,device=device).unsqueeze(0) if 'active_mask' in row else torch.zeros(1,64,dtype=torch.bool,device=device); # fallback reconstructed below
            if 'active_mask' not in row: mask=torch.zeros(1,64,dtype=torch.bool,device=device); mask[0,:row['active_count']]=True
            p0,_,d0,_=predictor.predict(torch.tensor([i],device=device),mask); ph=evidence_potential(labels[i:i+1],p0,d0); cand=mask.expand(64,-1).clone(); cand|=torch.eye(64,dtype=torch.bool,device=device); valid=~mask[0]; p1,_,d1,_=predictor.predict(torch.tensor([i],device=device).expand(64),cand); cc=view.coarse[i,:,128].to(device); cc=(cc-cc.min())/(cc.max()-cc.min()).clamp_min(1e-6); gain=(evidence_potential(labels[i:i+1].expand(64),p1,d1)-ph-float(lam)*cc.double()).masked_fill(~valid,-float('inf')); stop_rows.append({'patient_id':row['patient_id'],'step':row['step'],'lambda':float(lam),'best_net_gain':float(gain.max()),'stop_regret':max(0.,float(gain.max())),'checkpoint':str(a.checkpoint)})
    for lam,b in budgets_by_lam.items():
        for meth,label in (('random','Random20_MatchedBudget'),('attention','AttentionTopK_MatchedBudget')):
            reps=[]; cr=[]
            for seed in range(20 if meth=='random' else 1):
                torch.manual_seed(seed); pp=np.zeros(len(val)); cc=np.zeros(len(val))
                for budget in sorted(set(b.tolist())):
                    group=np.where(b==budget)[0]
                    if budget==0: am=torch.zeros(len(group),64,dtype=torch.bool,device=device); pp[group]=predictor.predict(torch.tensor(group,device=device),am)[0].cpu().numpy()
                    else: am,co,_=fixed(view,predictor,policy,torch.tensor(group,device=device),int(budget),device,meth.split('20')[0].replace('attention','attention'),seed); pp[group]=predictor.predict(torch.tensor(group,device=device),am)[0].cpu().numpy(); cc[group]=co.cpu().numpy()
                reps.append(pp); cr.append(cc)
            for seed,pp in enumerate(reps): add(f'{label}_lambda{lam:.8g}'+(f'_seed{seed}' if meth=='random' else ''),pp,b,cr[seed],seed,lam)
            if meth=='random': add(f'{label}_lambda{lam:.8g}_mean',np.mean(reps,0),b,np.mean(cr,0),lam=lam)
    pd.DataFrame(metric_rows).to_csv(out/'validation_metrics.csv',index=False); pd.DataFrame(pred_rows).to_csv(out/'validation_predictions.csv',index=False)
    with (out/'validation_action_sequences.jsonl').open('w') as f:
        for row in traj: f.write(json.dumps(row)+'\n')
    pd.DataFrame(stop_rows).to_csv(out/'validation_stop_audit.csv',index=False)
    summary={'train_complete':True,'validation_run':True,'test_run':False,'checkpoint':str(a.checkpoint),'checkpoint_sha256':digest(a.checkpoint),'validation_fixed_e2_paths':[str(x) for x in view.checkpoint_paths],'validation_fixed_e2_sha256':[digest(x) for x in view.checkpoint_paths]}; (out/'final_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps({'validation_patients':len(val),'methods':len(metric_rows),'checkpoint_sha256':summary['checkpoint_sha256']}))
if __name__=='__main__': main()
