"""W149: three internal folds, fixed morphology and four predeclared linear Cox heads."""
import argparse
import collections
import hashlib
import json
import math
import re
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .run_multicancer import split_rows, write
from .run_open_survival import Bags
from .open_survival_adapter import make_panther, cox_breslow, labels, risk_metrics, baseline_curves
from .censored_metrics import fit_censoring, survival_metrics
from .uni_encoder import sha256_file


ARMS = dict(fine_weak=(.001, 'full'), fine_ridge=(.1, 'full'),
            fine_pca64=(.1, 'fine64'), dual_pca64=(.1, 'fine32_coarse32'))


def cancer_pools(rows):
    result={c:[r for r in rows if r['cancer']==c] for c in ['CRC','LUAD']}
    assert all(result[c] and all(r['cancer']==c for r in result[c]) for c in result)
    assert not ({r['canonical_case_key'] for r in result['CRC']}&{r['canonical_case_key'] for r in result['LUAD']})
    return result


def emit(root, **values):
    values['at'] = time.time();write(root/'status.json', values);print(json.dumps(values), flush=True)


def inner_folds(rows, seed=41):
    buckets = collections.defaultdict(list)
    for i,r in enumerate(rows):buckets[int(r['survival_target']['event'])].append(i)
    held = [[] for _ in range(3)]
    for indices in buckets.values():
        indices.sort(key=lambda i: hashlib.sha256(f"{seed}:W149:{rows[i]['canonical_case_key']}".encode()).hexdigest())
        for j,i in enumerate(indices):held[j%3].append(i)
    all_indices=set(range(len(rows)));result=[]
    for indices in held:
        train=sorted(all_indices-set(indices));valid=sorted(indices)
        assert not set(train)&set(valid);result.append((train,valid))
    assert sorted(i for _,v in result for i in v)==list(range(len(rows)))
    return result


def describe(rows):
    t=np.array([r['survival_target']['time_days'] for r in rows]);e=np.array([r['survival_target']['event'] for r in rows])
    sites=[]
    for r in rows:
        m=re.search(r'TCGA-([A-Za-z0-9]{2})-',r['original_patient_id']);sites.append(m.group(1) if m else 'unknown')
    counts=collections.Counter(sites)
    return dict(patients=len(rows),slides=sum(len(r['slides']) for r in rows),events=int(e.sum()),
          censoring_before_year1=int(((e==0)&(t<365)).sum()),observed_at_year5=int((t>=1825).sum()),
          observed_time_median_days=float(np.median(t)),observed_time_quartiles=np.quantile(t,[.25,.75]).tolist(),
          tissue_source_sites=len(counts),largest_site_fraction=max(counts.values())/len(rows))


class ScaleBags(Bags):
    def get_scale(self, index, scale, count=None, rng=None):
        if scale=='fine':return self.get(index,count,rng)
        import zarr
        key=('coarse',index)
        if key in self.cache:
            values=self.cache.pop(key);self.cache[key]=values
        else:
            values=np.concatenate([np.asarray(zarr.open_group(str(self.root/'features'/s['asset_id']/'coarse.zarr'),mode='r')['features'][:],dtype=np.float32) for s in self.rows[index]['slides']])
            assert np.isfinite(values).all() and len(values)
            while self.cache and self.bytes+values.nbytes>self.cfg['ram_cache_gib']*1024**3:
                _,old=self.cache.popitem(last=False);self.bytes-=old.nbytes
            self.cache[key]=values;self.bytes+=values.nbytes
        if count and len(values)>count:return torch.from_numpy(values[rng.choice(len(values),count,replace=False)].copy())
        return torch.from_numpy(values)


def representations(data, train, scale, root, folder, cfg):
    from sklearn.cluster import KMeans
    expected=folder.name.split('_inner')[0]
    assert all(r['cancer']==expected for r in data.rows),'Run cancer does not match feature pool'
    prefix=folder/scale;prefix.mkdir(exist_ok=True);proto=prefix/'prototypes.npy';cache=prefix/'representations_private.pt'
    if cache.exists():
        saved=torch.load(cache,weights_only=True);assert saved.shape==(len(data.rows),32784) and torch.isfinite(saved).all();return saved.numpy().astype(np.float64)
    if not proto.exists():
        rng=np.random.default_rng(cfg['seed']);per=math.ceil(cfg['prototype_patch_budget']/len(train))
        sample=np.concatenate([data.get_scale(i,scale,per,rng).numpy().copy() for i in train])[:cfg['prototype_patch_budget']]
        emit(root,state='train_only_kmeans',fold=folder.name,scale=scale,sampled_patches=len(sample))
        km=KMeans(n_clusters=16,n_init=5,max_iter=100,random_state=cfg['seed']).fit(sample)
        np.save(proto,km.cluster_centers_.astype(np.float32));write(prefix/'prototype_provenance.json',dict(fit_partition='this_inner_training_only',patients=len(train),sampled_patches=len(sample),
            training_ids_sha256=hashlib.sha256(json.dumps([data.rows[i]['canonical_case_key'] for i in train]).encode()).hexdigest(),sha256=sha256_file(proto)))
        del sample,km
    encoder=make_panther(root/'official',proto).cuda().eval();out=[]
    with torch.no_grad():
        for i in range(len(data.rows)):
            value,_=encoder(data.get_scale(i,scale).cuda().unsqueeze(0));assert torch.isfinite(value).all();out.append(value.cpu().squeeze(0))
            if (i+1)%50==0:emit(root,state='fixed_gmm_representation',fold=folder.name,scale=scale,patients=i+1,total=len(data.rows))
    saved=torch.stack(out);torch.save(saved,cache);del encoder;torch.cuda.empty_cache()
    return saved.numpy().astype(np.float64)


def transform_fit(x, train, pca_dim=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scaler=StandardScaler().fit(x[train]);y=scaler.transform(x)
    pca=None
    if pca_dim:
        pca=PCA(n_components=pca_dim,svd_solver='randomized',iterated_power=3,random_state=41).fit(y[train]);y=pca.transform(y)
    y=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
    assert np.isfinite(y).all()
    return y,dict(scaler=scaler,pca=pca)


def fit_ridge(x, rows, alpha):
    x=torch.as_tensor(x,dtype=torch.float64);t,e=labels(rows);t=t.double();e=e.double()
    beta=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([beta],lr=1,max_iter=500,history_size=20,tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    history=[]
    def closure():
        opt.zero_grad();likelihood=cox_breslow(x@beta,t,e);loss=likelihood+.5*alpha*beta.square().sum()
        assert torch.isfinite(loss);loss.backward();assert torch.isfinite(beta.grad).all()
        history.append(dict(objective=float(loss.detach()),cox=float(likelihood.detach()),weight_norm=float(beta.detach().norm())))
        return loss
    opt.step(closure);closure();grad=float(beta.grad.abs().max())
    assert grad<1e-5, f'Convex solver did not converge: {grad}'
    return beta.detach().numpy(),dict(alpha=alpha,gradient_inf=grad,iterations=int(opt.state[beta]['n_iter']),history=history)


def selected_inner_uncertainty(root, old_root):
    from sksurv.metrics import concordance_index_censored
    output={};rng=np.random.default_rng(149)
    for cancer in ['CRC','LUAD']:
        for method in ['nnMIL','PANTHER','OTSurv']:
            folder=old_root/'runs'/f'fold0_{cancer}_{method}'
            rows=json.loads((folder/'split_private.json').read_text())['inner_validation']
            risk=np.array(json.loads((folder/'predictions_private.json').read_text())['inner_risk'])
            t=np.array([r['survival_target']['time_days'] for r in rows]);e=np.array([r['survival_target']['event'] for r in rows],bool)
            def score(ix):return float(concordance_index_censored(e[ix],t[ix],risk[ix])[0])
            draws=[]
            for _ in range(1000):
                ix=rng.integers(0,len(rows),len(rows))
                try:v=score(ix)
                except ValueError:continue
                if np.isfinite(v):draws.append(v)
            leave=[score(np.delete(np.arange(len(rows)),i)) for i in range(len(rows))]
            output[f'{cancer}_{method}']=dict(patients=len(rows),events=int(e.sum()),c_index=score(np.arange(len(rows))),
               conditional_patient_bootstrap_percentiles=np.quantile(draws,[.025,.975]).tolist(),valid_bootstraps=len(draws),
               leave_one_patient_out_range=[min(leave),max(leave)])
    write(root/'selected_inner_uncertainty.json',dict(results=output,scope='Fixed already-selected inner predictions: conditional sensitivity only, excludes training/model-selection uncertainty; not external generalization CI'))


def main(root, audit_only=False):
    import joblib
    cfg=json.loads((root/'config.json').read_text());prov=json.loads((root/'provenance.json').read_text())
    assert sha256_file(root/'config.json')==prov['config_sha256']
    assert sha256_file(root/'patients_private.json')==prov['patients_sha256']
    for name,sha in prov['code'].items():assert sha256_file(root/name)==sha
    for rec in json.loads((root/'source_manifest.json').read_text()):
        assert sha256_file(root/'official'/rec['repo'].split('/')[-1]/rec['path'])==rec['sha256']
    torch.set_num_threads(cfg['cpu_threads']);torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    rows=json.loads((root/'patients_private.json').read_text());assert len({r['canonical_case_key'] for r in rows})==len(rows)
    assets=[s['asset_id'] for r in rows for s in r['slides']];assert len(set(assets))==len(assets)
    old_train,old_inner,outer=split_rows(rows,0,cfg['seed']);dev=old_train+old_inner;pools=cancer_pools(dev)
    audit={};splits={}
    for c in ['CRC','LUAD']:
        cr=pools[c];folds=inner_folds(cr,cfg['seed'])
        audit[c]=dict(development=describe(cr),original_train=describe([r for r in old_train if r['cancer']==c]),original_inner=describe([r for r in old_inner if r['cancer']==c]),outer_count=sum(r['cancer']==c for r in outer),new_inner_folds=[dict(train=describe([cr[i] for i in tr]),validation=describe([cr[i] for i in va])) for tr,va in folds])
        splits[c]=[dict(train=[cr[i]['canonical_case_key'] for i in tr],validation=[cr[i]['canonical_case_key'] for i in va]) for tr,va in folds]
    write(root/'data_audit.json',dict(cancers=audit,canonical_patients_unique=True,slide_asset_ids_unique=True,original_slides_read=0,new_outer_predictions=0))
    write(root/'splits_private.json',splits)
    if not (root/'selected_inner_uncertainty.json').exists():selected_inner_uncertainty(root,Path(cfg['w147_root']))
    emit(root,state='data_audit_completed',fits=24)
    if audit_only:return
    results=[]
    for cancer in ['CRC','LUAD']:
        cr=pools[cancer];assert all(r['cancer']==cancer for r in cr);data=ScaleBags(cr,root,cfg)
        for fold,(train,valid) in enumerate(inner_folds(cr,cfg['seed'])):
            folder=root/'runs'/f'{cancer}_inner{fold}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'completion.json').exists():
                receipt=json.loads((folder/'completion.json').read_text())
                for name,sha in receipt['head_sha256'].items():assert sha256_file(folder/name)==sha
                results.extend(json.loads((folder/'results.json').read_text()));continue
            write(folder/'split_private.json',dict(train=[cr[i] for i in train],validation=[cr[i] for i in valid]))
            fine=representations(data,train,'fine',root,folder,cfg)
            coarse=representations(data,train,'coarse',root,folder,cfg)
            xf,tf=transform_fit(fine,train);xp,tp=transform_fit(fine,train,64)
            # Use the same leading 32 fine PCs; replacement by coarse PCs preserves a 64-D readout budget.
            fine32=tp['pca'].transform(tp['scaler'].transform(fine))[:,:32]
            fine32/=np.maximum(np.linalg.norm(fine32,axis=1,keepdims=True),1e-12)
            xc,tc=transform_fit(coarse,train,32);dual=np.concatenate([fine32,xc],axis=1)/math.sqrt(2)
            joblib.dump(dict(fine=tf,fine_pca64=tp,coarse_pca32=tc),folder/'transforms_private.joblib')
            train_rows=[cr[i] for i in train];valid_rows=[cr[i] for i in valid];fold_results=[];hashes={}
            for arm,(alpha,kind) in ARMS.items():
                x={'full':xf,'fine64':xp,'fine32_coarse32':dual}[kind]
                emit(root,state='training_convex_head',cancer=cancer,inner_fold=fold,arm=arm,train_events=sum(r['survival_target']['event'] for r in train_rows))
                beta,fit=fit_ridge(x[train],train_rows,alpha);train_risk=x[train]@beta;valid_risk=x[valid]@beta
                curves=baseline_curves(train_risk,train_rows,valid_risk,cfg['data_contract']['interval_edges_days'])
                metric=risk_metrics(valid_risk,valid_rows);prob=survival_metrics(valid_rows,curves,cfg['data_contract']['interval_edges_days'],fit_censoring(train_rows))
                rec=dict(cancer=cancer,inner_fold=fold,arm=arm,parameters=len(beta),alpha=alpha,train=risk_metrics(train_risk,train_rows),validation=metric,annual_ibs=prob['integrated_brier_annual_grid'],gradient_inf=fit['gradient_inf'])
                np.save(folder/(arm+'_head.npy'),beta);hashes[arm+'_head.npy']=sha256_file(folder/(arm+'_head.npy'))
                write(folder/(arm+'_solver_history.json'),fit)
                write(folder/(arm+'_predictions_private.json'),dict(train_risk=train_risk.tolist(),validation_risk=valid_risk.tolist(),validation_survival=curves.tolist()))
                fold_results.append(rec)
            write(folder/'results.json',fold_results);results.extend(fold_results);write(root/'results.json',results)
            write(folder/'completion.json',dict(state='completed',heads=4,head_sha256=hashes,new_outer_predictions=0,at=time.time()))
            emit(root,state='inner_fold_completed',cancer=cancer,inner_fold=fold,completed_heads=len(results))
        del data
    summary={}
    for cancer in ['CRC','LUAD']:
        summary[cancer]={}
        for arm in ARMS:
            rr=[r for r in results if r['cancer']==cancer and r['arm']==arm];cs=[r['validation']['c_index'] for r in rr]
            summary[cancer][arm]=dict(mean_fold_c=float(np.mean(cs)),fold_c=cs,fold_c_sd_descriptive=float(np.std(cs,ddof=1)),mean_ibs=float(np.mean([r['annual_ibs'] for r in rr])),
                mean_train_c=float(np.mean([r['train']['c_index'] for r in rr])),validation_events=[r['validation']['events'] for r in rr])
    write(root/'summary.json',dict(results=summary,scope='Development screening on three internal folds; no new outer evaluation; do not concatenate unaligned raw Cox risks across folds'))
    write(root/'suite_completed.json',dict(state='completed',heads=len(results),inner_folds=6,new_outer_predictions=0,at=time.time()))
    emit(root,state='completed',heads=len(results))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--audit-only',action='store_true');args=p.parse_args()
    try:main(args.root,args.audit_only)
    except Exception:
        write(args.root/'failure.json',dict(state='failed',at=time.time(),traceback=traceback.format_exc()));raise
