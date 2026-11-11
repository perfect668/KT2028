"""Parallel W136 development: capacity, fixed-budget selection, frozen/alternating training.

Uses W135 fold0 TRAIN and INNER validation only. Outer patients are never evaluated.
No edits to the running W135 source/configuration or stored feature arrays.
"""
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
from .fusion_data import move
from .run_multicancer import Dataset,write,split_rows,figures
from .selection_network import RegionSelector,choose,predictor
from .survival import discrete_nll
from .censored_metrics import fit_censoring,survival_metrics
from .train_survival import objective
from .uni_encoder import sha256_file


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def emit(root,**state):
    state.update(at=time.time())
    write(root/'status.json',state);print(json.dumps(state),flush=True)


def freeze(model):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)


def enable(model):
    for p in model.parameters():p.requires_grad_(True)


def selected_inputs(data,i,reference,policy,cfg,stochastic=False):
    case=data.case(i)
    with torch.no_grad():
        coarse=move(case.inputs([]),'cuda');out=reference(**coarse)
    counts=torch.as_tensor(case.counts,device='cuda')
    scores=policy(out['coarse_tokens'],out['coarse_patient'],coarse['boxes'],counts,coarse['slide_indices'])
    ids,logp,entropy=choose(scores,counts,cfg['selection_fraction'],stochastic)
    return move(case.inputs(ids),'cuda'),dict(data.rows[i]['survival_target']),logp,entropy,dict(
        regions=len(ids),eligible_regions=int((case.counts>0).sum()),fine_patches=int(case.counts[ids].sum()),
        available_fine_patches=int(case.counts.sum()))


def evaluate(model,data,cfg,censoring,reference=None,policy=None,modes=None):
    model.eval()
    if policy is not None:policy.eval()
    modes=modes or ['zero','0.1','0.2','0.4','full']
    scores={};predictions={};budgets={}
    with torch.no_grad():
        for mode in modes:
            curves=[];losses=[];usage=[]
            for i in range(len(data)):
                if mode=='selected':
                    inputs,target,_,_,u=selected_inputs(data,i,reference,policy,cfg);usage.append(u)
                else:
                    x,target=data.sample(i,mode,0);inputs=move(x,'cuda')
                    usage.append(dict(regions=len(x['observed_fine']),fine_patches=sum(len(v) for v in x['observed_fine'].values())))
                out=model(**inputs);curves.append(out['survival'].cpu().numpy());losses.append(float(discrete_nll(out['logits'],target)))
            curves=np.stack(curves);c=data.rows[0]['cancer']
            m=survival_metrics(data.rows,curves,cfg['data_contract']['interval_edges_days'],censoring)
            m['nll']=float(np.mean(losses));scores[mode]=dict(by_cancer={c:m},macro_nll=m['nll'])
            predictions[mode]=curves.tolist();budgets[mode]=usage
    return scores,predictions,budgets


def plot(root,folder,history,scores=None,predictions=None,rows=None):
    figures(folder,history)
    if folder.name.endswith('frozen_predictor'):
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
        values=[h for h in history if h.get('mean_reward_advantage') is not None]
        axes[0].plot([h['epoch'] for h in values],[h['mean_reward_advantage'] for h in values])
        axes[0].set(xlabel='Selector epoch',ylabel='Random NLL - selected NLL',title='Training reward advantage (higher is better)')
        axes[1].plot([h['epoch'] for h in history],[h['selection_nll'] for h in history])
        axes[1].set(xlabel='Selector epoch',ylabel='Inner validation NLL',title='Frozen predictor / learned selection')
        for ext in ('png','svg'):fig.savefig(folder/('learning_curves.'+ext),dpi=180)
        plt.close(fig)
    if scores:
        import matplotlib.pyplot as plt
        modes=list(scores);c=rows[0]['cancer']
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
        for ax,key,title in zip(axes,['nll','c_index'],['Inner validation NLL (lower is better)','Inner validation C-index (higher is better)']):
            values=[scores[m]['by_cancer'][c][key] for m in modes]
            ax.bar(modes,[v if v is not None else 0 for v in values]);ax.set_title(title)
            for j,v in enumerate(values):ax.text(j,v or 0,'NA' if v is None else f'{v:.3f}',ha='center',va='bottom')
        for ext in ('png','svg'):fig.savefig(folder/('inner_metrics.'+ext),dpi=180)
        plt.close(fig)
        mode='selected' if 'selected' in scores else 'full'
        ix=sorted(range(len(rows)),key=lambda i:hashlib.sha256(rows[i]['canonical_case_key'].encode()).hexdigest())[:6]
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        for j,i in enumerate(ix):ax.step(range(6),[1]+predictions[mode][i],where='post',label=f'Case {j+1}')
        ax.set(xlabel='Years since diagnosis',ylabel='Predicted survival',ylim=(0,1.02),title=c+' | inner validation / '+mode);ax.legend()
        for ext in ('png','svg'):fig.savefig(folder/('survival_examples.'+ext),dpi=180)
        plt.close(fig)
    cards=[]
    for run in sorted((root/'runs').iterdir()):
        if run.is_dir():cards.append(f'<h2>{run.name}</h2><img width="1000" src="runs/{run.name}/learning_curves.png">')
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
        '<h1>W136 Lightweight predictor and region selection</h1><p>Development / inner validation only. '
        '20% eligible parent-region budget; actual fine-patch counts are recorded. No outer-fold results.</p>'+''.join(cards),encoding='utf-8')


def save_pair(folder,model,policy=None):
    torch.save(dict(model=model.state_dict(),policy=policy.state_dict() if policy is not None else None),folder/'best.partial.pt')
    (folder/'best.partial.pt').replace(folder/'best.pt')


def load_pair(folder,model,policy=None):
    checkpoint=torch.load(folder/'best.pt',map_location='cuda',weights_only=True)
    model.load_state_dict(checkpoint['model'],strict=True)
    if policy is not None:policy.load_state_dict(checkpoint['policy'],strict=True)


def complete(root,folder,history,model,data,cfg,censoring,policy=None,reference=None):
    load_pair(folder,model,policy)
    modes=['zero','0.2','full']+(['selected'] if policy is not None else [])
    scores,predictions,budgets=evaluate(model,data,cfg,censoring,reference,policy,modes)
    write(folder/'inner_metrics.json',scores);write(folder/'usage_private.json',budgets)
    # Anonymized examples only in figures; patient-level rows are not exported locally.
    plot(root,folder,history,scores,predictions,data.rows)
    if policy is not None:
        import matplotlib.pyplot as plt
        i=min(range(len(data)),key=lambda i:hashlib.sha256(data.rows[i]['canonical_case_key'].encode()).hexdigest())
        case=data.case(i)
        with torch.no_grad():inputs,_,_,_,_=selected_inputs(data,i,reference,policy,cfg)
        selected=set(inputs['observed_fine'])
        slides=np.unique(case.slide_indices)
        fig,axes=plt.subplots(1,len(slides),figsize=(5*len(slides),4),squeeze=False,constrained_layout=True)
        for ax,sid in zip(axes[0],slides):
            ix=np.where(case.slide_indices==sid)[0];xy=(case.boxes[ix,:2]+case.boxes[ix,2:])/2
            chosen=np.array([int(j) in selected for j in ix])
            ax.scatter(xy[~chosen,0],xy[~chosen,1],s=3,c='#bdc9cf',label='Not read at fine scale')
            ax.scatter(xy[chosen,0],xy[chosen,1],s=8,c='#b95047',label='Selected fine region')
            ax.invert_yaxis();ax.set_aspect('equal');ax.set_title(f'Inner example / slide {int(sid)+1}');ax.legend(fontsize=7)
        for ext in ('png','svg'):fig.savefig(folder/('selected_regions.'+ext),dpi=180)
        plt.close(fig)
    write(folder/'completion.json',dict(steps=len(history),best_inner_nll=min(h['selection_nll'] for h in history),
        checkpoint_sha256=sha256_file(folder/'best.pt'),predictor_parameters=sum(p.numel() for p in model.parameters()),
        selector_parameters=sum(p.numel() for p in policy.parameters()) if policy is not None else 0,
        outer_used=False,development_only=True))


def new_folder(root,name):
    folder=root/'runs'/name
    if folder.exists():raise FileExistsError('Preserve existing run: '+str(folder))
    folder.mkdir();return folder


def completed_folder(root,name):
    folder=root/'runs'/name
    receipt=folder/'completion.json'
    if not receipt.exists():return None
    result=json.loads(receipt.read_text())
    if result.get('checkpoint_sha256')!=sha256_file(folder/'best.pt'):
        raise ValueError('Completed checkpoint SHA mismatch: '+name)
    if result.get('outer_used') is not False or result.get('development_only') is not True:
        raise ValueError('Unexpected completion protocol: '+name)
    emit(root,state='skipping_completed',stage=name,checkpoint_verified=True)
    return folder


def warmup(root,cancer,data,val,cfg,censoring,gated=False):
    name=cancer+('_gated128' if gated else '_r3_128')
    done=completed_folder(root,name)
    if done is not None:return done
    folder=new_folder(root,name);seed(cfg['seed'])
    model=predictor(gated,**cfg['architecture']).cuda()
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    i=next(i for i,r in enumerate(data.rows) if r['survival_target']['event'])
    model.eval();loss=objective(model,data,i,0,'cuda',cfg['loss_weights']);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters())
    assert any(p.grad is not None and torch.any(p.grad!=0) for p in model.parameters())
    del before;model.zero_grad(set_to_none=True)
    # Stress check the largest training coarse bag before any optimizer update.
    max_i=max(range(len(data)),key=lambda j:sum(json.loads((root/'features'/s['asset_id']/'receipt.json').read_text())['coarse']['rows'] for s in data.rows[j]['slides']))
    # Training dropout can select a much larger attention/backward workspace.
    # Preserve RNG so this resource check does not change the registered training seed.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model.train();stress=objective(model,data,max_i,0,'cuda',cfg['loss_weights']);stress.backward()
        assert torch.isfinite(stress) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    write(folder/'preflight.json',dict(finite_gradient=True,unchanged_parameters=True,largest_train_case_checked=True,
        stress_training_mode=True,max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_steps=0))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    history=[];best=float('inf');stale=0
    for epoch in range(cfg['max_epochs']):
        model.train();opt.zero_grad(set_to_none=True);total=0.
        order=np.random.default_rng(cfg['seed']+epoch).permutation(len(data))
        for j,i in enumerate(order):
            loss=objective(model,data,int(i),epoch,'cuda',cfg['loss_weights'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite predictor objective')
            group=min(cfg['accumulate_cases'],len(order)-(j//cfg['accumulate_cases'])*cfg['accumulate_cases'])
            (loss/group).backward();total+=float(loss.detach())
            if (j+1)%cfg['accumulate_cases']==0 or j+1==len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True)
                emit(root,stage=name,state='training',epoch=epoch+1,patients=j+1,total=len(order),objective=total/(j+1))
        emit(root,stage=name,state='inner_validation',epoch=epoch+1)
        scores,_,_=evaluate(model,val,cfg,censoring)
        nll=float(np.mean([v['macro_nll'] for v in scores.values()]))
        history.append(dict(epoch=epoch+1,train_objective=total/len(data),selection_nll=nll,validation=scores))
        write(folder/'history.json',history);plot(root,folder,history)
        if nll<best:best=nll;stale=0;save_pair(folder,model)
        else:stale+=1
        if stale>=cfg['patience']:break
    complete(root,folder,history,model,val,cfg,censoring)
    del opt,model;torch.cuda.empty_cache()
    return folder


def policy_epoch(root,stage,model,reference,policy,data,cfg,opt,round_id):
    freeze(model);freeze(reference);enable(policy);policy.train();total=0.;used=0
    for i in np.random.default_rng(cfg['seed']+round_id).permutation(len(data)):
        i=int(i);target=data.rows[i]['survival_target']
        if not target['event'] and not target['completed_intervals']:continue
        inputs,target,logp,entropy,_=selected_inputs(data,i,reference,policy,cfg,True)
        random_x,_=data.sample(i,str(cfg['selection_fraction']),round_id)
        with torch.no_grad():
            chosen=discrete_nll(model(**inputs)['logits'],target)
            baseline=discrete_nll(model(**move(random_x,'cuda'))['logits'],target)
            advantage=(baseline-chosen).clamp(-cfg['advantage_clip'],cfg['advantage_clip'])
        loss=-advantage*logp-cfg['policy_entropy']*entropy
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite policy objective')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True);opt.step()
        total+=float(advantage);used+=1
        if used%8==0:emit(root,state='training',stage=stage,phase='selector',round=round_id,patients=used,mean_nll_advantage=total/used)
    assert all(p.grad is None for p in model.parameters())
    return total/max(used,1)


def predictor_epoch(root,stage,model,reference,policy,data,cfg,opt,round_id):
    freeze(policy);freeze(reference);enable(model);model.train();total=0.;opt.zero_grad(set_to_none=True)
    seed(cfg['seed']+10000+round_id)
    order=np.random.default_rng(cfg['seed']+round_id).permutation(len(data))
    for j,i in enumerate(order):
        i=int(i);inputs,target,_,_,_=selected_inputs(data,i,reference,policy,cfg,False)
        a=discrete_nll(model(**inputs)['logits'],target)
        random_x,_=data.sample(i,str(cfg['selection_fraction']),round_id)
        b=discrete_nll(model(**move(random_x,'cuda'))['logits'],target)
        zero_x,_=data.sample(i,'zero',round_id)
        c=discrete_nll(model(**move(zero_x,'cuda'))['logits'],target)
        loss=.5*a+.25*b+.25*c
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adapted predictor objective')
        group=min(cfg['accumulate_cases'],len(order)-(j//cfg['accumulate_cases'])*cfg['accumulate_cases'])
        (loss/group).backward();total+=float(loss.detach())
        if (j+1)%cfg['accumulate_cases']==0 or j+1==len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            opt.step();opt.zero_grad(set_to_none=True)
            emit(root,state='training',stage=stage,phase='predictor',round=round_id,patients=j+1,total=len(order),objective=total/(j+1))
    assert all(p.grad is None for p in policy.parameters())
    return total/len(data)


def selection_arm(root,cancer,mode,base_folder,policy_folder,data,val,cfg,censoring):
    done=completed_folder(root,cancer+'_'+mode)
    if done is not None:return done
    folder=new_folder(root,cancer+'_'+mode);seed(cfg['seed'])
    model=predictor(**cfg['architecture']).cuda();load_pair(base_folder,model)
    reference=copy.deepcopy(model);freeze(reference)
    policy=RegionSelector(cfg['architecture']['dim']).cuda()
    if policy_folder is not None:load_pair(policy_folder,model,policy)
    model.zero_grad(set_to_none=True);policy.zero_grad(set_to_none=True)
    freeze(model)
    policy_opt=torch.optim.AdamW(policy.parameters(),lr=cfg['policy_lr'],weight_decay=cfg['weight_decay'])
    pred_opt=torch.optim.AdamW(model.parameters(),lr=cfg['adaptation_lr'],weight_decay=cfg['weight_decay'])
    initial,_,_=evaluate(model,val,cfg,censoring,reference,policy,['selected'])
    best=initial['selected']['macro_nll'];save_pair(folder,model,policy)
    history=[dict(epoch=0,train_objective=None,selection_nll=best,phase='initial_pair')]
    rounds=cfg['policy_epochs'] if mode=='frozen_predictor' else cfg['alternating_rounds']
    for r in range(1,rounds+1):
        advantage=None
        if mode in ['frozen_predictor','alternating']:
            model.zero_grad(set_to_none=True)
            advantage=policy_epoch(root,folder.name,model,reference,policy,data,cfg,policy_opt,r)
        policy.zero_grad(set_to_none=True)
        loss=0.
        if mode in ['fixed_selector_adapt','alternating']:
            loss=predictor_epoch(root,folder.name,model,reference,policy,data,cfg,pred_opt,r)
        scores,_,_=evaluate(model,val,cfg,censoring,reference,policy,['selected'])
        nll=scores['selected']['macro_nll']
        history.append(dict(epoch=r,train_objective=loss,selection_nll=nll,mean_reward_advantage=advantage))
        write(folder/'history.json',history);plot(root,folder,history)
        if nll<best:best=nll;save_pair(folder,model,policy)
    complete(root,folder,history,model,val,cfg,censoring,policy,reference)
    del model,reference,policy,policy_opt,pred_opt;torch.cuda.empty_cache()
    return folder


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    cfg=json.loads((root/'config.json').read_text());torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    rows=json.loads((root/'patients_private.json').read_text())
    train,val,outer=split_rows(rows,0,cfg['seed'])
    write(root/'provenance.json',dict(config_sha256=sha256_file(root/'config.json'),patients_sha256=sha256_file(root/'patients_private.json'),
        source_sha256={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},outer_evaluations=0))
    (root/'runs').mkdir(exist_ok=True)
    for cancer in cfg['development_cancers']:
        tr=[r for r in train if r['cancer']==cancer];va=[r for r in val if r['cancer']==cancer]
        assert not ({r['canonical_case_key'] for r in tr+va}&{r['canonical_case_key'] for r in outer})
        data=Dataset(tr,root,cfg,'independent');valid=Dataset(va,root,cfg,'independent');censoring=fit_censoring(tr)
        write(root/(cancer+'_split_private.json'),dict(train=tr,inner_validation=va,outer_not_loaded=True))
        emit(root,stage=cancer,state='preflight',train=len(tr),inner_validation=len(va))
        base=warmup(root,cancer,data,valid,cfg,censoring,False)
        frozen=selection_arm(root,cancer,'frozen_predictor',base,None,data,valid,cfg,censoring)
        selection_arm(root,cancer,'fixed_selector_adapt',base,frozen,data,valid,cfg,censoring)
        selection_arm(root,cancer,'alternating',base,frozen,data,valid,cfg,censoring)
        warmup(root,cancer,data,valid,cfg,censoring,True)
    emit(root,state='completed',runs=10,outer_evaluations=0)


if __name__=='__main__':
    try:main()
    except Exception:
        import sys
        traceback.print_exc()
        root=Path(sys.argv[sys.argv.index('--root')+1])
        emit(root,state='failed',error=traceback.format_exc())
        raise
