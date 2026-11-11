"""W140: actual official DSCA modules, fixed cohort/endpoint, causal diagnostic arms."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .dsca_source_adapter import SourceDSCA,source_likelihood,source_optimizer,save_optimizer,load_optimizer
from .run_multicancer import Dataset,split_rows,write
from .fusion_data import move
from .survival import discrete_nll,curve_from_logits
from .survival_network import SurvivalPredictor
from .censored_metrics import fit_censoring,survival_metrics
from .uni_encoder import sha256_file


def emit(root,**state):
    state['at']=time.time();write(root/'status.json',state);print(json.dumps(state),flush=True)


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def choose_device(cfg):
    # Preserve both existing GPU queues. CPU training is real training, not a stub.
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=p.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        if b'kt2028.run_selection_ablation' in args:return 'cpu'
    if torch.cuda.is_available():
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free>=cfg['gpu_minimum_free_mib']:
            torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction']);return 'cuda'
    return 'cpu'


def recipe(cfg,arm):
    return cfg['official_recipe'] if arm=='dsca_official_recipe' else cfg['matched_recipe']


def model_for(root,arm,r):
    if arm=='r3_full_only':return SurvivalPredictor(dim=256,heads=4,dropout=r['dropout'],intervals=5,fusion='cross_gate')
    return SourceDSCA(root/'official',dropout=r['dropout'],dim=384)


def evaluate(model,data,cfg,censoring,device):
    model.eval();curves=[];losses=[]
    with torch.no_grad():
        for i in range(len(data)):
            x,t=data.sample(i,'full');out=model(**move(x,device))
            curves.append(out['survival'].cpu().numpy());losses.append(float(discrete_nll(out['logits'],t)))
    curves=np.stack(curves)
    metrics=survival_metrics(data.rows,curves,cfg['data_contract']['interval_edges_days'],censoring)
    metrics['nll']=float(np.mean(losses));return metrics,curves.tolist()


def constant_reference(train,val,cfg,censoring):
    targets=[r['survival_target'] for r in train];h=[]
    for k in range(5):
        d=sum(t['event'] and t['event_bin']==k for t in targets)
        n=sum((t['event'] and t['event_bin']>=k) or (not t['event'] and t['completed_intervals']>k) for t in targets)
        h.append(np.clip(d/n if n else 0.,1e-7,1-1e-7))
    logits=torch.tensor(np.log(np.array(h)/(1-np.array(h))),dtype=torch.float64)
    curve=curve_from_logits(logits)[1].numpy()
    metrics=survival_metrics(val,np.tile(curve,(len(val),1)),cfg['data_contract']['interval_edges_days'],censoring)
    metrics['nll']=float(np.mean([float(discrete_nll(logits,r['survival_target'])) for r in val]))
    return dict(metrics=metrics,hazards=list(map(float,h)),fit_partition='train_only')


def plot(folder,history,reference,metrics=None,predictions=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    e=[x['epoch'] for x in history]
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    axes[0].plot(e,[x['train_objective'] for x in history]);axes[0].set(title='Training objective (includes registered regularization)',xlabel='Epoch',ylabel='Objective')
    axes[1].plot(e,[x['validation']['nll'] for x in history],label='Inner validation')
    axes[1].axhline(reference['metrics']['nll'],color='#888',linestyle=':',label='Train-only constant reference')
    axes[1].set(title='Probability error / checkpoint selection',xlabel='Epoch',ylabel='NLL');axes[1].legend(fontsize=7)
    axes[2].plot(e,[x['validation']['c_index'] for x in history]);axes[2].axhline(.5,color='#888',linestyle=':')
    axes[2].set(title='Inner validation risk ranking',xlabel='Epoch',ylabel='C-index',ylim=(0,1))
    for ext in ['png','svg']:fig.savefig(folder/('learning_curves.'+ext),dpi=160)
    plt.close(fig)
    if predictions is not None:
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        # Caller supplies outcome-independent ID-hash-selected examples.
        for i,s in enumerate(predictions):ax.step(range(6),[1]+s,where='post',label=f'Case {i+1}')
        ax.set(xlabel='Years since diagnosis',ylabel='Predicted survival',ylim=(0,1.02),title='Fixed cases / '+metrics);ax.legend(fontsize=8)
        for ext in ['png','svg']:fig.savefig(folder/('survival_examples.'+ext),dpi=160)
        plt.close(fig)


def overview(root,cfg):
    results=[]
    for p in sorted((root/'runs').glob('*/completion.json')):
        c=json.loads(p.read_text());results.append(c)
    write(root/'comparison.json',results)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,c in enumerate(cfg['cancers']):
        p=root/'runs'/('fold0_'+c+'_dsca_official_recipe')/'completion.json'
        old=root/'baseline'/'runs'/('fold0_'+c)/'outer_metrics.json'
        if old.exists():
            m=json.loads(old.read_text())['full']['by_cancer'][c]
            axes[0].bar(i-.18,m['c_index'],width=.35,color='#869eac',label='Existing R3' if i==0 else None)
            axes[1].bar(i-.18,m['nll'],width=.35,color='#869eac')
        if p.exists():
            m=json.loads((p.parent/'outer_metrics.json').read_text())
            axes[0].bar(i+.18,m['c_index'],width=.35,color='#bd7c4a',label='Official-source DSCA' if i==0 else None)
            axes[1].bar(i+.18,m['nll'],width=.35,color='#bd7c4a')
    axes[0].set(title='Outer fold 0 C-index',ylim=(0,1));axes[0].legend(fontsize=8)
    axes[1].set_title('Outer fold 0 NLL')
    for ax in axes:ax.set_xticks(range(4),cfg['cancers'])
    fig.suptitle('Same patients, UNI and five-year endpoint / missing DSCA bars = not completed')
    for ext in ['png','svg']:fig.savefig(root/('comparison.'+ext),dpi=160)
    plt.close(fig)


def fit(root,cfg,job,partitions):
    cancer,arm,outer_enabled=job['cancer'],job['arm'],job['outer_evaluation']
    name='fold0_'+cancer+'_'+arm;folder=root/'runs'/name;r=recipe(cfg,arm)
    if (folder/'completion.json').exists():
        receipt=json.loads((folder/'completion.json').read_text())
        assert sha256_file(folder/'best.pt')==receipt['checkpoint_sha256'];return
    folder.mkdir(exist_ok=True)
    train,val,outer=([x for x in p if x['cancer']==cancer] for p in partitions)
    write(folder/'split_private.json',dict(train=train,inner_validation=val,outer=outer if outer_enabled else 'not_loaded'))
    write(folder/'registered_recipe.json',dict(recipe=r,job=job))
    data=Dataset(train,root,cfg,'independent');valid=Dataset(val,root,cfg,'independent');censoring=fit_censoring(train)
    reference=constant_reference(train,val,cfg,censoring);write(folder/'constant_reference.json',reference)
    seed(cfg['seed']);device=choose_device(cfg);model=model_for(root,arm,r).to(device)
    emit(root,state='preflight',job=name,device=device,train=len(train),inner_validation=len(val),parameters=sum(p.numel() for p in model.parameters()))
    source_loss=model.official.SurvMLE(alpha=r['alpha']) if arm=='dsca_official_recipe' else None
    def objective(out,t):
        if source_loss is not None:return source_likelihood(source_loss,out['hazards'][None],[t],r['alpha'])
        return discrete_nll(out['logits'],t)
    if arm=='dsca_official_recipe':opt=source_optimizer(model,model.official,lr=r['lr'],weight_decay=r['weight_decay'])
    else:opt=torch.optim.AdamW(model.parameters(),lr=r['lr'],weight_decay=r['weight_decay'])
    base_opt=getattr(opt,'base_optimizer',opt)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(base_opt,mode='min',factor=.5,patience=10) if r['scheduler'] else None
    last=folder/'last.pt';history=[];best=float('inf');stale=0;first_epoch=0;best_epoch=0
    if last.exists():
        state=torch.load(last,map_location=device,weights_only=False)
        model.load_state_dict(state['model'],strict=True);load_optimizer(opt,model,state['optimizer'],device)
        if scheduler:scheduler.load_state_dict(state['scheduler'])
        history=state['history'];best=state['best'];stale=state['stale'];first_epoch=state['epoch'];best_epoch=state['best_epoch']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu())
        if device=='cuda' and state.get('cuda_rng') is not None:torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        emit(root,state='resumed',job=name,completed_epochs=first_epoch,device=device)
    else:
        # Max coarse bag and max fine count are checked in actual training mode.
        sizes=[]
        for row in train:
            receipts=[json.loads((root/'features'/s['asset_id']/'receipt.json').read_text()) for s in row['slides']]
            sizes.append((sum(x['coarse']['rows'] for x in receipts),sum(x['fine']['rows'] for x in receipts)))
        indices=sorted({max(range(len(train)),key=lambda i:sizes[i][axis]) for axis in [0,1]})
        state_before={n:p.detach().clone() for n,p in model.named_parameters()};started=time.time()
        with torch.random.fork_rng(devices=[0] if device=='cuda' else []):
            for i in indices:
                model.train();x,t=data.sample(i,'full');out=model(**move(x,device));loss=objective(out,t);loss.backward()
                assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                base_opt.zero_grad(set_to_none=True)
        assert all(torch.equal(state_before[n],p) for n,p in model.named_parameters());del state_before
        write(folder/'preflight.json',dict(training_mode=True,largest_train_coarse_and_fine=True,finite_gradient=True,unchanged_parameters=True,optimizer_steps=0,device=device,seconds=time.time()-started))
        # Same preflight parameters, strict reload and identical eval outputs.
        clone=model_for(root,arm,r).to(device).eval();clone.load_state_dict(model.state_dict(),strict=True);model.eval()
        with torch.no_grad():torch.testing.assert_close(model(**move(x,device))['logits'],clone(**move(x,device))['logits'])
        del clone,x,out,loss
        # Reset so preflight construction and checks never shift the training seed.
        seed(cfg['seed'])
    optimizer_steps=0
    for epoch in range(first_epoch,r['max_epochs']):
        model.train();base_opt.zero_grad(set_to_none=True);total=0.;regularization=[];started=time.time()
        order=np.random.default_rng(cfg['seed']+epoch).permutation(len(train))
        for j,i in enumerate(order):
            x,t=data.sample(int(i),'full');out=model(**move(x,device));loss=objective(out,t)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DSCA/R3 full objective')
            group=min(r['accumulate'],len(order)-(j//r['accumulate'])*r['accumulate'])
            (loss/group).backward();total+=float(loss.detach())
            if (j+1)%r['accumulate']==0 or j+1==len(order):
                reg=r['l1']*sum(p.abs().sum() for p in model.parameters()) if r['l1'] else None
                if reg is not None:reg.backward();regularization.append(float(reg.detach()))
                if r['clip'] is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),r['clip'],error_if_nonfinite=True)
                elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
                opt.step();base_opt.zero_grad(set_to_none=True);optimizer_steps+=1
                emit(root,state='training',job=name,device=device,epoch=epoch+1,patients=j+1,total=len(train),train_nll=total/(j+1),optimizer_steps_this_session=optimizer_steps)
            del x,out,loss
        emit(root,state='inner_validation',job=name,device=device,epoch=epoch+1)
        metrics,_=evaluate(model,valid,cfg,censoring,device);nll=metrics['nll']
        record=dict(epoch=epoch+1,train_objective=total/len(train)+(float(np.mean(regularization)) if regularization else 0),train_likelihood=total/len(train),validation=metrics,selection_nll=nll,learning_rate=base_opt.param_groups[0]['lr'],seconds=time.time()-started)
        history.append(record)
        if nll<=best:
            best=nll;stale=0;best_epoch=epoch+1
            torch.save(dict(model=model.state_dict(),epoch=best_epoch),folder/'best.partial.pt');(folder/'best.partial.pt').replace(folder/'best.pt')
        else:stale+=1
        if scheduler:scheduler.step(nll)
        state=dict(model=model.state_dict(),optimizer=save_optimizer(opt,model),scheduler=scheduler.state_dict() if scheduler else None,epoch=epoch+1,best_epoch=best_epoch,best=best,stale=stale,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None)
        torch.save(state,folder/'last.partial.pt');(folder/'last.partial.pt').replace(last)
        write(folder/'history.json',history);plot(folder,history,reference);overview(root,cfg)
        emit(root,state='epoch_complete',job=name,device=device,epoch=epoch+1,validation_nll=nll,validation_c=metrics['c_index'],best_epoch=best_epoch,seconds=record['seconds'])
        if stale>=r['patience']:break
    saved=torch.load(folder/'best.pt',map_location=device,weights_only=True);model.load_state_dict(saved['model'],strict=True)
    metrics,preds=evaluate(model,valid,cfg,censoring,device);write(folder/'inner_metrics.json',metrics)
    write(folder/'inner_predictions_private.json',dict(rows=val,predictions=preds))
    examples=sorted(range(len(val)),key=lambda i:hashlib.sha256(val[i]['canonical_case_key'].encode()).hexdigest())[:6]
    plot(folder,history,reference,'inner validation',[preds[i] for i in examples])
    if outer_enabled:
        emit(root,state='fixed_best_outer_evaluation',job=name,epoch=best_epoch)
        test=Dataset(outer,root,cfg,'independent');m,p=evaluate(model,test,cfg,censoring,device)
        write(folder/'outer_metrics.json',m);write(folder/'outer_predictions_private.json',dict(rows=outer,predictions=p))
    write(folder/'completion.json',dict(cancer=cancer,arm=arm,fold=0,epochs=len(history),best_epoch=best_epoch,checkpoint_sha256=sha256_file(folder/'best.pt'),parameters=sum(p.numel() for p in model.parameters()),inner_metrics=metrics,outer_evaluation=outer_enabled,outer_used_for_selection=False,device=device))
    overview(root,cfg)
    del model,opt,base_opt,data,valid,state
    if device=='cuda':torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    root=args.root;cfg=json.loads((root/'config.json').read_text())
    if not args.execute:raise RuntimeError('Explicit --execute required for the registered suite')
    sys.path.insert(0,str(root/'pydeps'));torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(1)
    for name,h in cfg['official_source_sha256'].items():
        if sha256_file(root/'official'/name)!=h:raise ValueError('Official source changed: '+name)
    sources={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    provenance=dict(config_sha256=sha256_file(root/'config.json'),patients_sha256=sha256_file(root/'patients_private.json'),source_sha256=sources)
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==provenance
    else:write(root/'provenance.json',provenance)
    rows=json.loads((root/'patients_private.json').read_text());partitions=split_rows(rows,0,cfg['seed'])
    (root/'runs').mkdir(exist_ok=True)
    overview(root,cfg)
    for job in cfg['jobs']:fit(root,cfg,job,partitions)
    emit(root,state='completed',runs=len(cfg['jobs']))


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        if '--root' in sys.argv:emit(Path(sys.argv[sys.argv.index('--root')+1]),state='failed',error=traceback.format_exc())
        raise
