"""W151: three R3 arms x CRC/LUAD x three existing internal folds; no outer inference."""
import argparse
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .component_network import make_predictor
from .run_multicancer import Dataset, split_rows, write
from .run_controlled_survival import cancer_pools, inner_folds, describe
from .train_survival import objective
from .fusion_data import move
from .censored_metrics import fit_censoring, survival_metrics
from .survival import discrete_nll
from .uni_encoder import sha256_file


ARMS = ('baseline','mase','subspace')
CANCERS = ('CRC','LUAD')


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)


def emit(root, **values):
    values['updated_unix']=time.time()
    write(root/'status.json', values)
    print(json.dumps(values,allow_nan=False),flush=True)


def atomic_save(path, data):
    temporary=path.with_suffix('.partial.pt')
    torch.save(data,temporary);temporary.replace(path)


def verify(root):
    provenance=json.loads((root/'provenance.json').read_text())
    for relative,sha in provenance['files'].items():
        if sha256_file(root/relative)!=sha:raise ValueError('Frozen source/configuration changed: '+relative)
    return provenance


def data_splits(root, cfg):
    rows=json.loads((root/'patients_private.json').read_text())
    assert len({r['canonical_case_key'] for r in rows})==len(rows)
    slide_ids=[s['asset_id'] for r in rows for s in r['slides']]
    assert len(slide_ids)==len(set(slide_ids))
    train,inner,outer=split_rows(rows,0,cfg['seed'])
    pools=cancer_pools(train+inner)
    outer_ids={r['canonical_case_key'] for r in outer}
    splits={}; audit={};jobs=[]
    for cancer in CANCERS:
        cr=pools[cancer];splits[cancer]=[];audit[cancer]=[]
        for fold,(tr,va) in enumerate(inner_folds(cr,cfg['seed'])):
            training=[cr[i] for i in tr];valid=[cr[i] for i in va]
            assert not outer_ids & {r['canonical_case_key'] for r in training+valid}
            assert all(r['cancer']==cancer for r in training+valid)
            splits[cancer].append(dict(train=[r['canonical_case_key'] for r in training],validation=[r['canonical_case_key'] for r in valid]))
            audit[cancer].append(dict(train=describe(training),validation=describe(valid)))
            jobs.append((cancer,fold,training,valid))
    expected=json.loads((root/'reference_splits_private.json').read_text())
    assert splits==expected,'New folds differ from already-audited W149 patient folds'
    if (root/'splits_private.json').exists():assert json.loads((root/'splits_private.json').read_text())==splits
    else:write(root/'splits_private.json',splits)
    write(root/'data_audit.json',dict(cancers=audit,patients_unique=True,slide_assets_unique=True,
        matches_w149=True,outer_patients_excluded=len(outer_ids),outer_predictions=0))
    return jobs


def evaluate(model, data, cfg, censoring, modes, root=None, context=None):
    model.eval();metrics={};predictions={};usage={}
    with torch.no_grad():
        for mode in modes:
            curves=[];losses=[];budgets=[]
            for i in range(len(data)):
                inputs,target=data.sample(i,mode,0)
                out=model(**move(inputs,'cuda'))
                curves.append(out['survival'].cpu().numpy());losses.append(float(discrete_nll(out['logits'],target)))
                budgets.append(dict(regions=len(inputs['observed_fine']),fine_patches=sum(len(x) for x in inputs['observed_fine'].values())))
                if root is not None and (i+1)%32==0:
                    emit(root,state='evaluating',context=context,mode=mode,patients_done=i+1,patients_total=len(data))
            value=survival_metrics(data.rows,np.stack(curves),cfg['data_contract']['interval_edges_days'],censoring)
            value.update(nll=float(np.mean(losses)),mean_regions=float(np.mean([u['regions'] for u in budgets])),
                         mean_fine_patches=float(np.mean([u['fine_patches'] for u in budgets])))
            metrics[mode]=value;predictions[mode]=np.stack(curves).tolist();usage[mode]=budgets
    return metrics,predictions,usage


def plot_learning(folder, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not history:return
    fig,axes=plt.subplots(1,3,figsize=(14,3.7),constrained_layout=True)
    epochs=[h['epoch'] for h in history]
    axes[0].plot(epochs,[h['monitor_full']['nll'] for h in history],label='Fixed training monitor (64 max)')
    axes[0].plot(epochs,[h['validation']['full']['nll'] for h in history],label='Inner validation')
    axes[0].set(title='Full-observation NLL',xlabel='Epoch',ylabel='Lower is better');axes[0].legend(fontsize=7)
    for role,label in [('monitor_full','Training monitor'),('validation','Inner validation')]:
        vals=[h['monitor_full']['c_index'] if role=='monitor_full' else h['validation']['full']['c_index'] for h in history]
        axes[1].plot(epochs,vals,label=label)
    axes[1].set(title='Full-observation C-index',xlabel='Epoch',ylabel='Higher is better',ylim=(0,1));axes[1].legend(fontsize=7)
    axes[2].plot(epochs,[h['selection_nll'] for h in history],label='Mean NLL across five observation modes')
    axes[2].set(title='Checkpoint selection',xlabel='Epoch',ylabel='Lower is better');axes[2].legend(fontsize=7)
    fig.suptitle(folder.name+' | development only')
    for ext in ('png','svg','pdf'):fig.savefig(folder/('learning_curves.'+ext),dpi=160)
    plt.close(fig)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=[]
    for p in sorted((root/'runs').glob('*/completion.json')):
        done=json.loads(p.read_text());metrics=json.loads((p.parent/'inner_metrics.json').read_text())
        results.append(dict(cancer=done['cancer'],fold=done['fold'],arm=done['arm'],
                            best_epoch=done['best_epoch'],epochs=done['epochs'],metrics=metrics,
                            parameters=done['parameters'],run=p.parent.name))
    write(root/'results.json',results)
    summary={}
    for cancer in CANCERS:
        summary[cancer]={}
        for arm in ARMS:
            values=[r for r in results if r['cancer']==cancer and r['arm']==arm]
            entry=dict(completed_folds=len(values),expected_folds=3)
            for key in ['c_index','nll','integrated_brier_annual_grid']:
                v=[r['metrics']['full'][key] for r in values if r['metrics']['full'][key] is not None]
                entry[key]=dict(mean=float(np.mean(v)) if v else None,sd=float(np.std(v,ddof=1)) if len(v)>1 else None)
            summary[cancer][arm]=entry
    write(root/'summary.json',dict(completed=len(results),expected=18,outer_predictions=0,results=summary))
    colors=['#3679a4','#cc7742','#458e77']
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,cancer in enumerate(CANCERS):
        for col,key in enumerate(['c_index','nll','integrated_brier_annual_grid']):
            ax=axes[row,col]
            for j,arm in enumerate(ARMS):
                values=[r['metrics']['full'][key] for r in results if r['cancer']==cancer and r['arm']==arm and r['metrics']['full'][key] is not None]
                if values:
                    ax.bar(j,np.mean(values),color=colors[j],alpha=.65)
                    if len(values)>1:ax.errorbar(j,np.mean(values),yerr=np.std(values,ddof=1),color='black',capsize=3)
                    ax.scatter(j+np.linspace(-.07,.07,len(values)),values,c='black',s=15,zorder=3)
                else:ax.text(j,.05,'Pending',ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
            ax.set(xticks=range(3),xticklabels=['R3','R3 + MASE','R3 + subspace'],title=cancer+' | '+key)
            if key=='c_index':ax.set_ylim(0,1)
    fig.suptitle(f'W151 | {len(results)}/18 fits complete | internal folds only; dots = folds, bars = available mean, error = SD')
    for ext in ('png','svg','pdf'):fig.savefig(root/('comparison.'+ext),dpi=170)
    plt.close(fig)
    cards=''.join(f'<section><h2>{p.name}</h2><img src="runs/{p.name}/learning_curves.png"><p><a href="runs/{p.name}/budget_response.png">Observation budget</a> | <a href="runs/{p.name}/attention_example.png">Spatial attention</a></p></section>' for p in sorted((root/'runs').iterdir()) if p.is_dir())
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>R3 controlled components</title><style>body{font:16px system-ui;max-width:1250px;margin:35px auto;background:#fafbfc;color:#253544}img{max-width:100%}section{padding:15px;margin-top:24px;background:white}</style><h1>R3 component experiments / W151</h1><p>Three matched arms; CRC and LUAD; three internal patient folds. Fixed UNI 5x/20x, five-year OS. No new outer-fold inference. Incomplete fits are not results. Full-observation metrics; all five observation modes select checkpoints.</p><img src="comparison.png">'+cards,encoding='utf-8')


def plot_best(folder, model, valid, cfg, metrics, predictions):
    import matplotlib.pyplot as plt
    modes=cfg['validation_modes'];fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for ax,key in zip(axes,['c_index','nll','integrated_brier_annual_grid']):
        ax.plot(range(len(modes)),[metrics[m][key] for m in modes],marker='o')
        ax.set(xticks=range(len(modes)),xticklabels=modes,title=key,xlabel='Fine observation budget')
    for ext in ('png','svg','pdf'):fig.savefig(folder/('budget_response.'+ext),dpi=160)
    plt.close(fig)
    order=sorted(range(len(valid)),key=lambda i:hashlib.sha256(valid.rows[i]['canonical_case_key'].encode()).hexdigest())
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    for k,i in enumerate(order[:6]):ax.step(range(6),[1]+predictions['full'][i],where='post',label=f'Example {k+1}')
    ax.set(xlabel='Years',ylabel='Predicted survival',ylim=(0,1.02),title='Fixed best / inner validation examples');ax.legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(folder/('survival_examples.'+ext),dpi=160)
    plt.close(fig)
    inputs,_=valid.sample(order[0],'full');case=move(inputs,'cuda')
    with torch.no_grad():
        out=model(**case)
        fused=out['coarse_tokens'].clone()
        fused[out['observed_ids']]*=out['coarse_gates']
        coarse_weights=model.coarse_pool(fused)[1][0].cpu().numpy()
        model.fine_pool(out['observed_tokens'])
        if hasattr(model.fine_pool,'last_weights'):fine_weights=model.fine_pool.last_weights.mean(0).cpu().numpy()
        else:fine_weights=model.fine_pool(out['observed_tokens'])[1][0].cpu().numpy()
    slides=inputs['slide_indices'].numpy();boxes=inputs['boxes'].numpy();xy=(boxes[:,:2]+boxes[:,2:])/2
    fweights=np.full(len(boxes),np.nan);fweights[out['observed_ids']]=fine_weights
    fig,axes=plt.subplots(2,len(np.unique(slides)),figsize=(5*len(np.unique(slides)),7),squeeze=False,constrained_layout=True)
    for j,sid in enumerate(np.unique(slides)):
        ids=np.where(slides==sid)[0]
        for row,weights,title in [(0,coarse_weights,'Coarse attention'),(1,fweights,'Fine regional attention')]:
            ax=axes[row,j];im=ax.scatter(xy[ids,0],xy[ids,1],c=weights[ids],s=9,cmap='magma')
            ax.set_aspect('equal');ax.invert_yaxis();ax.set_title(f'{title} / slide {j+1}');fig.colorbar(im,ax=ax)
    fig.suptitle('Outcome-independent example | model attention, not pathology labels')
    for ext in ('png','svg'):fig.savefig(folder/('attention_example.'+ext),dpi=160)
    plt.close(fig)


def preflight(root,cfg,jobs):
    """Geometry audit plus actual training-mode maximum-case checks, with zero updates."""
    entries={};all_rows={}
    for cancer,fold,tr,_ in jobs:
        for r in tr:
            key=r['canonical_case_key']
            if key in entries:continue
            n=canvas=0
            for slide in r['slides']:
                folder=root/'features'/slide['asset_id']
                if slide.get('receipt_sha256'):assert sha256_file(folder/'receipt.json')==slide['receipt_sha256']
                with np.load(folder/'geometry.npz',allow_pickle=False) as g:
                    boxes=g['coarse_bbox_um'];n+=len(boxes)
                    xy=(boxes[:,:2]-boxes[:,:2].min(0))/448.
                    assert np.allclose(xy,np.rint(xy),atol=1e-4,rtol=0)
                    assert len(np.unique(np.rint(xy),axis=0))==len(xy)
                    canvas+=int(np.prod(np.rint(xy).max(0)+1))
            entries[key]=dict(cancer=cancer,fold=fold,coarse_regions=n,canvas_cells=canvas)
            all_rows[key]=r
    selected=[]
    for cancer in CANCERS:
        pool=[key for key,e in entries.items() if e['cancer']==cancer]
        selected.extend(max(pool,key=lambda k:entries[k][criterion]) for criterion in ['coarse_regions','canvas_cells'])
    selected=list(dict.fromkeys(selected));checks=[]
    write(root/'geometry_audit_private.json',dict(entries=entries,preflight_cases=selected))
    for arm in ARMS:
        for key in selected:
            e=entries[key];job=next(j for j in jobs if j[0]==e['cancer'] and j[1]==e['fold'])
            data=Dataset(job[2],root,cfg,'independent');i=next(i for i,r in enumerate(data.rows) if r['canonical_case_key']==key)
            seed(cfg['seed']+e['fold']);model=make_predictor(arm,cfg,root/'official/EAGLE-Net').cuda().train()
            before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.cuda.reset_peak_memory_stats();began=time.time()
            emit(root,state='preflight',arm=arm,cancer=e['cancer'],fold=e['fold'],coarse_regions=e['coarse_regions'],canvas_cells=e['canvas_cells'])
            loss=objective(model,data,i,0,'cuda',cfg['loss_weights']);loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
            peak=torch.cuda.max_memory_reserved();model.zero_grad(set_to_none=True);model.eval()
            inputs,_=data.sample(i,'full')
            with torch.no_grad():expected=model(**move(inputs,'cuda'))['survival'].cpu()
            clone=make_predictor(arm,cfg,root/'official/EAGLE-Net').cuda().eval();clone.load_state_dict(model.state_dict(),strict=True)
            with torch.no_grad():torch.testing.assert_close(expected,clone(**move(inputs,'cuda'))['survival'].cpu(),atol=1e-6,rtol=1e-5)
            checks.append(dict(arm=arm,**e,finite_training_gradient=True,strict_reload=True,unchanged_parameters=True,
                optimizer_steps=0,peak_reserved_bytes=peak,seconds=time.time()-began))
            write(root/'preflight_progress.json',checks)
            del loss,model,clone,data,before,inputs;torch.cuda.empty_cache()
    write(root/'preflight_completed.json',dict(checks=checks,geometry_patients=len(entries),optimizer_steps=0,
        config_sha256=sha256_file(root/'config.json'),provenance_sha256=sha256_file(root/'provenance.json'),outer_predictions=0))
    emit(root,state='preflight_completed',checks=len(checks),geometry_patients=len(entries),optimizer_steps=0)


def fit(root,cfg,cancer,fold,tr,va,arm):
    folder=root/'runs'/f'{cancer}_inner{fold}_{arm}';folder.mkdir(exist_ok=True)
    signature=sha256_file(root/'provenance.json')
    if (folder/'completion.json').exists():
        done=json.loads((folder/'completion.json').read_text())
        assert done['provenance_sha256']==signature and sha256_file(folder/'best.pt')==done['best_sha256']
        return
    write(folder/'split_private.json',dict(train=tr,validation=va))
    seed(cfg['seed']+fold);model=make_predictor(arm,cfg,root/'official/EAGLE-Net').cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    data=Dataset(tr,root,cfg,'independent');valid=Dataset(va,root,cfg,'independent');censoring=fit_censoring(tr)
    monitor_rows=sorted(tr,key=lambda r:hashlib.sha256(r['canonical_case_key'].encode()).hexdigest())[:64]
    monitor=Dataset(monitor_rows,root,cfg,'independent')
    history=[];best=float('inf');best_epoch=0;start=0;steps=0
    last=folder/'last.pt';seed(cfg['seed']+fold+151000)
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        assert saved['provenance_sha256']==signature
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];start=saved['epoch'];steps=saved['steps']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        emit(root,state='resume_epoch_checkpoint',run=folder.name,next_epoch=start+1)
    write(folder/'run_metadata.json',dict(arm=arm,cancer=cancer,fold=fold,parameters=sum(p.numel() for p in model.parameters()),
        train_patients=len(tr),validation_patients=len(va),train_events=sum(r['survival_target']['event'] for r in tr),
        validation_events=sum(r['survival_target']['event'] for r in va),monitor_patients=len(monitor_rows),provenance_sha256=signature))
    for epoch in range(start,cfg['max_epochs']):
        if history and len(history)-best_epoch>=cfg['patience']:break
        began=time.time();torch.cuda.reset_peak_memory_stats();model.train();opt.zero_grad(set_to_none=True);total=0.
        order=np.random.default_rng(cfg['seed']+fold*1000+epoch).permutation(len(tr))
        for j,i in enumerate(order):
            loss=objective(model,data,int(i),epoch,'cuda',cfg['loss_weights'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            group=min(cfg['accumulate_cases'],len(order)-(j//cfg['accumulate_cases'])*cfg['accumulate_cases'])
            (loss/group).backward();total+=float(loss.detach())
            if (j+1)%cfg['accumulate_cases']==0 or j+1==len(order):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True);steps+=1
                emit(root,state='training',run=folder.name,arm=arm,cancer=cancer,fold=fold,epoch=epoch+1,
                    patients_done=j+1,patients_total=len(tr),optimizer_steps=steps,mean_objective=total/(j+1),gradient_norm=float(norm))
        scores,_,_=evaluate(model,valid,cfg,censoring,cfg['validation_modes'],root,folder.name+'/validation')
        monitor_scores,_,_=evaluate(model,monitor,cfg,censoring,['full'],root,folder.name+'/training_monitor')
        selection=float(np.mean([s['nll'] for s in scores.values()]))
        history.append(dict(epoch=epoch+1,train_objective=total/len(tr),selection_nll=selection,validation=scores,
            monitor_full=monitor_scores['full'],seconds=time.time()-began,peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        if selection<best:
            best=selection;best_epoch=epoch+1
            atomic_save(folder/'best.pt',dict(model=model.state_dict(),epoch=best_epoch,provenance_sha256=signature))
        atomic_save(last,dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,steps=steps,
            history=history,best=best,best_epoch=best_epoch,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),provenance_sha256=signature))
        write(folder/'history.json',history);plot_learning(folder,history);report(root)
        emit(root,state='epoch_complete',run=folder.name,epoch=epoch+1,best_epoch=best_epoch,selection_nll=selection,
             full_c_index=scores['full']['c_index'],full_ibs=scores['full']['integrated_brier_annual_grid'])
    saved=torch.load(folder/'best.pt',map_location='cuda',weights_only=False)
    assert saved['provenance_sha256']==signature;model.load_state_dict(saved['model'],strict=True)
    scores,predictions,usage=evaluate(model,valid,cfg,censoring,cfg['validation_modes'],root,folder.name+'/fixed_best_validation')
    train_scores,train_predictions,_=evaluate(model,data,cfg,censoring,['zero','0.2','full'],root,folder.name+'/fixed_best_train')
    write(folder/'inner_metrics.json',scores);write(folder/'train_metrics.json',train_scores)
    write(folder/'predictions_private.json',dict(validation=predictions,train=train_predictions))
    write(folder/'usage_private.json',usage)
    plot_best(folder,model,valid,cfg,scores,predictions)
    write(folder/'completion.json',dict(state='completed',arm=arm,cancer=cancer,fold=fold,best_epoch=best_epoch,
        epochs=len(history),optimizer_steps=steps,best_sha256=sha256_file(folder/'best.pt'),strict_reload=True,
        parameters=sum(p.numel() for p in model.parameters()),provenance_sha256=signature,outer_predictions=0,at=time.time()))
    report(root);emit(root,state='fit_completed',run=folder.name,best_epoch=best_epoch)
    del model,opt,data,valid,monitor;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--report-only',action='store_true');args=p.parse_args();root=args.root
    cfg=json.loads((root/'config.json').read_text());verify(root)
    torch.set_num_threads(cfg['cpu_threads']);torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    jobs=data_splits(root,cfg)
    if args.report_only:report(root);return
    if not args.execute:preflight(root,cfg,jobs);return
    checked=json.loads((root/'preflight_completed.json').read_text())
    assert checked['config_sha256']==sha256_file(root/'config.json') and checked['provenance_sha256']==sha256_file(root/'provenance.json')
    for cancer,fold,tr,va in jobs:
        for arm in ARMS:fit(root,cfg,cancer,fold,tr,va,arm)
    report(root)
    write(root/'suite_completed.json',dict(fits=18,outer_predictions=0,provenance_sha256=sha256_file(root/'provenance.json'),at=time.time()))
    emit(root,state='completed',fits=18,outer_predictions=0)


if __name__=='__main__':
    try:main()
    except Exception:
        import sys
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1]);root.mkdir(parents=True,exist_ok=True)
            failure=dict(state='failed',traceback=traceback.format_exc(),at=time.time())
            write(root/'failure.json',failure);emit(root,**failure)
        raise
