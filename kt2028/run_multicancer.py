"""Registered patient-level nested five-fold suite, persistent status and anonymous plots."""
import argparse
import copy
import hashlib
import json
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .censored_metrics import fit_censoring, survival_metrics
from .fusion_data import move
from .mixed_survival_data import MixedSurvivalDataset, inspect_asset
from .multicancer_network import CANCERS, CancerPredictor
from .survival_network import SurvivalPredictor
from .survival import discrete_nll
from .train_survival import objective
from .uni_encoder import sha256_file


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def log(**value):
    print(json.dumps(dict(at=time.time(), **value)), flush=True)


def split_rows(rows, fold, seed):
    buckets = {}
    for r in rows:
        buckets.setdefault((r['cancer'], r['survival_target']['event']), []).append(r)
    train, val, outer = [], [], []
    for key, values in sorted(buckets.items()):
        ordered = sorted(values, key=lambda r: hashlib.sha256(f"{seed}:outer:{r['canonical_case_key']}".encode()).hexdigest())
        outer.extend(r for i, r in enumerate(ordered) if i % 5 == fold)
        inner = [r for i, r in enumerate(ordered) if i % 5 != fold]
        inner.sort(key=lambda r: hashlib.sha256(f"{seed}:{fold}:inner:{r['canonical_case_key']}".encode()).hexdigest())
        n = max(1, int(len(inner) * .1 + .5))
        val.extend(inner[:n]); train.extend(inner[n:])
    sets = [{r['canonical_case_key'] for r in x} for x in (train, val, outer)]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    assert sum(map(len, sets)) == len(rows)
    return train, val, outer


class Dataset(MixedSurvivalDataset):
    def __init__(self, rows, root, cfg, kind):
        # All schema/target/asset checks run before this adapter is constructed.
        self.rows, self.roots = rows, {c: str(root/'features') for c in CANCERS}
        self.contract, self.seed, self._cache, self.kind = cfg['data_contract'], cfg['seed'], {}, kind

    def sample(self, i, mode, epoch=0):
        inputs, target = super().sample(i, mode, epoch)
        if self.kind != 'independent':
            inputs['cancer_id'] = CANCERS.index(self.rows[i]['cancer'])
        return inputs, target


def evaluate(model, dataset, cfg, censoring, modes):
    model.eval(); scores = {}; predictions = {}
    with torch.no_grad():
        for mode in modes:
            curves, losses = [], []
            for i in range(len(dataset)):
                inputs, target = dataset.sample(i, mode)
                out = model(**move(inputs, 'cuda'))
                curves.append(out['survival'].cpu().numpy())
                losses.append(float(discrete_nll(out['logits'], target)))
            curves = np.stack(curves)
            by = {}
            for cancer in sorted({r['cancer'] for r in dataset.rows}):
                ix = [i for i, r in enumerate(dataset.rows) if r['cancer'] == cancer]
                by[cancer] = survival_metrics([dataset.rows[i] for i in ix], curves[ix],
                    cfg['data_contract']['interval_edges_days'], censoring)
                by[cancer]['nll'] = float(np.mean(np.array(losses)[ix]))
            scores[mode] = {'by_cancer': by, 'macro_nll': float(np.mean([v['nll'] for v in by.values()]))}
            predictions[mode] = curves.tolist()
    return scores, predictions


def figures(folder, history, metrics=None, predictions=None, rows=None, routing=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    epochs = [h['epoch'] for h in history]
    axes[0].plot(epochs, [h['train_objective'] for h in history], label='Weighted training objective')
    axes[0].set(xlabel='Epoch', ylabel='Objective', title='Training objective'); axes[0].legend(fontsize=8)
    axes[1].plot(epochs, [h['selection_nll'] for h in history], label='Inner validation NLL')
    axes[1].set(xlabel='Epoch', ylabel='NLL (lower is better)', title='Checkpoint selection'); axes[1].legend(fontsize=8)
    for ext in ('png', 'svg'): fig.savefig(folder/('learning_curves.'+ext), dpi=180)
    plt.close(fig)
    if metrics:
        by=metrics['full']['by_cancer']; cancers=list(by)
        fig, axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
        for ax, key, title in zip(axes, ['c_index','integrated_brier_annual_grid'], ['C-index (higher is better)','IBS (lower is better)']):
            values=[by[c][key] for c in cancers]
            ax.bar(cancers,[v if v is not None else 0 for v in values],color='#3a7ca5')
            ax.set_title(title)
            for i,v in enumerate(values): ax.text(i,(v or 0), 'NA' if v is None else f'{v:.3f}',ha='center',va='bottom')
        for ext in ('png','svg'): fig.savefig(folder/('outer_metrics.'+ext),dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(1,len(cancers),figsize=(4*len(cancers),3.5),squeeze=False,constrained_layout=True)
        for ax,c in zip(axes[0],cancers):
            ix=[i for i,r in enumerate(rows) if r['cancer']==c]
            # Outcome-independent SHA ordering; include first six patients, no cherry-picking.
            ix=sorted(ix,key=lambda i:hashlib.sha256(rows[i]['canonical_case_key'].encode()).hexdigest())[:6]
            for j,i in enumerate(ix): ax.step(np.arange(6),[1]+predictions['full'][i],where='post',label=f'Case {j+1}')
            ax.set(xlabel='Years since diagnosis',ylabel='Predicted survival',ylim=(0,1.02),title=c+' | outer fold');ax.legend(fontsize=7)
        for ext in ('png','svg'):fig.savefig(folder/('survival_examples.'+ext),dpi=180)
        plt.close(fig)
    if routing is not None:
        fig,ax=plt.subplots(figsize=(5,4),constrained_layout=True)
        mat=np.array(routing);im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=['E1','E2','E3','E4'],yticklabels=CANCERS,title='Cancer-conditioned routing')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center')
        fig.colorbar(im,ax=ax)
        for ext in ('png','svg'):fig.savefig(folder/('routing.'+ext),dpi=180)
        plt.close(fig)


def overview(root):
    """Aggregate independent models as one comparison arm; keep unavailable runs blank."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=[]
    for p in sorted((root/'runs').glob('*/outer_metrics.json')):
        if not (p.parent/'completion.json').exists():continue
        fold,job=p.parent.name.split('_',1)
        for cancer,v in json.loads(p.read_text())['full']['by_cancer'].items():
            results.append(dict(fold=fold,arm=job if job in ('shared','moe') else 'independent',cancer=cancer,**v))
    write(root/'comparison.json',results)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    colors=['#5685a6','#e3a34b','#56a889']
    for ax,key,title in zip(axes,['c_index','integrated_brier_annual_grid'],['Outer-fold C-index','Outer-fold IBS']):
        for j,arm in enumerate(('independent','shared','moe')):
            means=[];errors=[]
            for cancer in CANCERS:
                vals=[r[key] for r in results if r['arm']==arm and r['cancer']==cancer and r[key] is not None]
                means.append(np.mean(vals) if vals else np.nan);errors.append(np.std(vals,ddof=1) if len(vals)>1 else 0)
            ax.bar(np.arange(4)+(j-1)*.25,means,width=.25,yerr=errors,color=colors[j],label=arm)
        ax.set(xticks=range(4),xticklabels=CANCERS,title=title);ax.legend(fontsize=8)
    fig.suptitle('Nested CV | available completed folds only; error bars = SD')
    for ext in ('png','svg'):fig.savefig(root/('comparison.'+ext),dpi=180)
    plt.close(fig)
    links=''.join(f'<li>{p.name}: <a href="runs/{p.name}/learning_curves.png">learning curves</a> '
                  f'<a href="runs/{p.name}/outer_metrics.png">outer metrics</a> '
                  f'<a href="runs/{p.name}/survival_examples.png">survival examples</a></li>'
                  for p in sorted((root/'runs').iterdir()) if p.is_dir())
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Four-cancer experiments</title>'
        '<h1>Four-cancer survival experiments</h1><p>Frozen UNI 5x/20x; OS 0–1825 days. '
        'Outer folds do not select checkpoints. Incomplete runs are not results.</p>'
        '<img width="1000" src="comparison.png"><ul>'+links+'</ul>',encoding='utf-8')


def fit(job, rows, root, cfg, fold, manifests):
    name, kind, cancer = job
    folder=root/'runs'/f'fold{fold}_{name}'
    if (folder/'completion.json').exists():return
    if folder.exists():raise RuntimeError(f'Incomplete run preserved: {folder}')
    folder.mkdir()
    train, val, outer=([r for r in group if cancer is None or r['cancer']==cancer] for group in manifests)
    for part, data in zip(('train','validate','outer'),(train,val,outer)):
        write(folder/(part+'_private.json'),data)
    random.seed(cfg['seed']+fold);np.random.seed(cfg['seed']+fold);torch.manual_seed(cfg['seed']+fold)
    model=(SurvivalPredictor(**cfg['architecture']) if kind=='independent' else CancerPredictor(kind,**cfg['architecture'])).cuda()
    data=Dataset(train,root,cfg,kind);valid=Dataset(val,root,cfg,kind);test=Dataset(outer,root,cfg,kind)
    censoring=fit_censoring(train)
    log(event='preflight',job=name,fold=fold,train=len(train),validation=len(val),outer=len(outer))
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    i=next(i for i,r in enumerate(train) if r['survival_target']['event'])
    model.eval();loss=objective(model,data,i,0,'cuda',cfg['loss_weights']);loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
    assert any(p.grad is not None and torch.any(p.grad!=0) for p in model.parameters())
    clone=copy.deepcopy(model);clone.load_state_dict(model.state_dict(),strict=True)
    inputs,_=data.sample(i,'full')
    with torch.no_grad():assert torch.allclose(model(**move(inputs,'cuda'))['logits'],clone(**move(inputs,'cuda'))['logits'],atol=1e-6)
    del before,clone;model.zero_grad(set_to_none=True)
    write(folder/'preflight.json',dict(finite_gradients=True,unchanged_parameters=True,strict_reload=True,optimizer_steps=0))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    counts={c:sum(r['cancer']==c for r in train) for c in {r['cancer'] for r in train}}
    history=[];best=float('inf');stale=0;best_epoch=0
    for epoch in range(cfg['max_epochs']):
        model.train();opt.zero_grad(set_to_none=True);total=0.
        order=np.random.default_rng(cfg['seed']+fold*1000+epoch).permutation(len(train))
        for j,i in enumerate(order):
            i=int(i);loss=objective(model,data,i,epoch,'cuda',cfg['loss_weights'])
            loss=loss*len(train)/(len(counts)*counts[train[i]['cancer']])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            group=min(cfg['accumulate_cases'],len(order)-(j//cfg['accumulate_cases'])*cfg['accumulate_cases'])
            (loss/group).backward();total+=float(loss.detach())
            if (j+1)%cfg['accumulate_cases']==0 or j+1==len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True)
                write(root/'status.json',dict(state='training',fold=fold,job=name,epoch=epoch+1,patients_done=j+1,patients_total=len(train),updated_unix=time.time()))
                log(event='optimizer_step',fold=fold,job=name,epoch=epoch+1,patients=j+1,total=len(train),objective=total/(j+1))
        scores,_=evaluate(model,valid,cfg,censoring,cfg['validation_modes'])
        selection=float(np.mean([s['macro_nll'] for s in scores.values()]))
        history.append(dict(epoch=epoch+1,train_objective=total/len(train),selection_nll=selection,validation=scores))
        write(folder/'history.json',history)
        if selection<best:
            best=selection;best_epoch=epoch+1;stale=0
            torch.save(dict(model=model.state_dict(),epoch=best_epoch),folder/'best.partial.pt')
            (folder/'best.partial.pt').replace(folder/'best.pt')
        else:stale+=1
        figures(folder,history)
        overview(root)
        log(event='epoch_complete',fold=fold,job=name,epoch=epoch+1,selection_nll=selection,best_epoch=best_epoch)
        if stale>=cfg['patience']:break
    saved=torch.load(folder/'best.pt',map_location='cuda',weights_only=True);model.load_state_dict(saved['model'],strict=True)
    metrics,predictions=evaluate(model,test,cfg,censoring,['full'])
    write(folder/'outer_metrics.json',metrics)
    write(folder/'outer_predictions_private.json',dict(rows=outer,predictions=predictions))
    routing=model.routing('cuda') if kind=='moe' else None
    figures(folder,history,metrics,predictions,outer,routing)
    write(folder/'completion.json',dict(selected_epoch=best_epoch,epochs=len(history),checkpoint_sha256=sha256_file(folder/'best.pt'),strict_reload=True,outer_used_for_selection=False,routing=routing))
    overview(root)
    del model,opt,data,valid,test;torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    torch.set_num_threads(2)
    cfg=json.loads((root/'config.json').read_text());rows=json.loads((root/'patients_private.json').read_text())
    assert cfg['outer_folds']==5 and cfg['old_checkpoint_reuse'] is False
    assert len({r['canonical_case_key'] for r in rows})==len(rows)
    slides=[s['slide_id'] for r in rows for s in r['slides']];assert len(set(slides))==len(slides)
    sources={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    provenance=dict(config_sha256=sha256_file(root/'config.json'),patients_sha256=sha256_file(root/'patients_private.json'),source_sha256=sources)
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==provenance
    else:write(root/'provenance.json',provenance)
    log(event='asset_preflight',patients=len(rows),slides=len(slides))
    for j,r in enumerate(rows):
        for s in r['slides']:inspect_asset(root/'features'/s['asset_id'],s,cfg['data_contract']['feature_protocol'])
        if j%50==0:log(event='asset_checked',patients=j+1)
    jobs=[(c,'independent',c) for c in CANCERS]+[('shared','shared',None),('moe','moe',None)]
    for fold in range(5):
        manifests=split_rows(rows,fold,cfg['seed'])
        for job in jobs:fit(job,rows,root,cfg,fold,manifests)
    write(root/'status.json',dict(state='completed',runs=30,updated_unix=time.time()))
    log(event='suite_completed',runs=30)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        import sys
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1])
            write(root/'status.json',dict(state='failed',updated_unix=time.time(),error=traceback.format_exc()))
        raise
