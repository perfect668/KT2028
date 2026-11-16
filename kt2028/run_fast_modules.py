"""W154: exactly three new fits, one fixed CRC split, four epochs each."""
import argparse
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .fast_module_network import FastModulePredictor,check_prototype_source
from .observed_cache import CachedDataset,DatasetSubset
from .run_component_ablation import evaluate,plot_learning,atomic_save,seed
from .run_multicancer import write
from .censored_metrics import fit_censoring
from .train_survival import objective
from .fusion_data import move
from .uni_encoder import sha256_file

ARMS=['joint','morphology','transport']
NAMES={'baseline':'Original R3 (reused 4-epoch prefix)','joint':'Paired-region fusion',
       'morphology':'Morphology-group pooling','transport':'OT prototype pooling'}


def emit(root,**value):
    value['updated_unix']=time.time();write(root/'status.json',value)
    print(json.dumps(value,allow_nan=False),flush=True)


def verify(root):
    doc=json.loads((root/'provenance.json').read_text())
    for name,sha in doc['files'].items():assert sha256_file(root/name)==sha,name
    baseline=Path(doc['baseline_root'])
    assert sha256_file(baseline/'best.pt')==doc['baseline_best_sha256']
    return doc


def model_for(arm,root,cfg,prototypes):
    seed(cfg['seed'])
    return FastModulePredictor(arm,prototypes=prototypes,ot_source=root/'official/OTSurv',**cfg['architecture'])


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reference=json.loads((root/'baseline_reference.json').read_text())
    rows=[dict(arm='baseline',metrics=reference['metrics'],best_epoch=reference['best_epoch'],reused=True)]
    for arm in ARMS:
        p=root/'runs'/arm/'completion.json'
        if p.exists():
            done=json.loads(p.read_text());metrics=json.loads((p.parent/'inner_metrics.json').read_text())
            rows.append(dict(arm=arm,metrics=metrics,best_epoch=done['best_epoch'],reused=False))
    count=len(rows)-1
    write(root/'summary.json',dict(completed_new_fits=count,expected_new_fits=3,baseline_reused=True,
        split='CRC fixed inner0',max_epochs=4,outer_predictions=0,results=rows))
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    colors=['#3679a4','#cc7742','#458e77','#8a69a7']
    for ax,key,title in zip(axes,['c_index','nll','integrated_brier_annual_grid'],['C-index (higher is better)','NLL (lower is better)','IBS (lower is better)']):
        for j,arm in enumerate(['baseline']+ARMS):
            item=next((r for r in rows if r['arm']==arm),None)
            if item:
                value=item['metrics']['full'][key]
                if value is not None:
                    ax.bar(j,value,color=colors[j],width=.65);ax.text(j,value,f'{value:.4f}',ha='center',va='bottom',fontsize=9)
            else:ax.text(j,.04,'Pending',ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
        ax.set(xticks=range(4),xticklabels=['R3','Paired','Groups','OT'],title=title,xlim=(-.6,3.6))
        if key=='c_index':ax.set_ylim(0,1)
    fig.suptitle(f'Fast screen | {count}/3 new fits complete | CRC fixed split | maximum four epochs')
    for ext in ['png','svg','pdf']:fig.savefig(root/('comparison.'+ext),dpi=170)
    plt.close(fig)
    cards=''.join(f'<section><h2>{NAMES[a]}</h2><img src="runs/{a}/learning_curves.png"><p><a href="runs/{a}/budget_response.png">Fine observation budget</a></p></section>' for a in ARMS if (root/'runs'/a/'history.json').exists())
    table=''.join('<tr><td>'+NAMES[r['arm']]+'</td>'+''.join('<td>'+f"{r['metrics']['full'][k]:.4f}"+'</td>' for k in ['c_index','nll','integrated_brier_annual_grid'])+'</tr>' for r in rows)
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>R3 remaining modules: fast screen</title><style>body{font:17px system-ui;max-width:1250px;margin:32px auto;color:#253544;background:#fafbfc}img{max-width:100%}td,th{padding:10px 20px;text-align:right}td:first-child{text-align:left}section{background:white;padding:15px;margin-top:25px}</style><h1>剩余模块快速验证</h1><p>仅CRC固定第一内部划分；只新增三个实验，每项最多4轮。原R3复用同划分前4轮的匹配最佳权重，不重跑。验证集参与选模，无新外层评价。已停止的W151多折队列不恢复。</p><p>已完成 '+str(count)+'/3 个新实验。</p><table><tr><th>Condition</th><th>C-index ↑</th><th>NLL ↓</th><th>IBS ↓</th></tr>'+table+'</table><img src="comparison.png">'+cards,encoding='utf-8')


def final_plots(folder,metrics,predictions,rows):
    import matplotlib.pyplot as plt
    modes=['zero','0.2','full'];fig,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    for ax,key in zip(axes,['c_index','nll','integrated_brier_annual_grid']):
        ax.plot(range(3),[metrics[m][key] for m in modes],marker='o')
        ax.set(xticks=range(3),xticklabels=['0%','20%','100%'],xlabel='Eligible fine parent regions',title=key)
    for ext in ['png','svg','pdf']:fig.savefig(folder/('budget_response.'+ext),dpi=160)
    plt.close(fig)
    order=sorted(range(len(rows)),key=lambda i:hashlib.sha256(rows[i]['canonical_case_key'].encode()).hexdigest())[:6]
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    for j,i in enumerate(order):ax.step(range(6),[1]+predictions['full'][i],where='post',label=f'Example {j+1}')
    ax.set(xlabel='Years',ylabel='Predicted survival',ylim=(0,1.02),title='Inner validation / outcome-independent examples');ax.legend(fontsize=8)
    for ext in ['png','svg']:fig.savefig(folder/('survival_examples.'+ext),dpi=160)
    plt.close(fig)


def preflight(root,cfg,data,prototypes):
    largest=max(range(len(data)),key=lambda i:sum(json.loads((root/'features'/s['asset_id']/'receipt.json').read_text())['coarse']['rows'] for s in data.rows[i]['slides']))
    proofs=[]
    for arm in ARMS:
        model=model_for(arm,root,cfg,prototypes).cuda().train()
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.cuda.reset_peak_memory_stats();began=time.time();emit(root,state='preflight',arm=arm)
        loss=objective(model,data,largest,0,'cuda',cfg['loss_weights']);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
        model.zero_grad(set_to_none=True);model.eval();inputs,_=data.sample(largest,'full')
        clone=model_for(arm,root,cfg,prototypes).cuda().eval();clone.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():torch.testing.assert_close(model(**move(inputs,'cuda'))['survival'],clone(**move(inputs,'cuda'))['survival'],atol=1e-6,rtol=1e-5)
        proofs.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),finite_training_gradient=True,
            unchanged_parameters=True,strict_reload=True,optimizer_steps=0,peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.time()-began))
        del loss,model,clone,before,inputs;torch.cuda.empty_cache()
    write(root/'preflight_completed.json',dict(checks=proofs,optimizer_steps=0,provenance_sha256=sha256_file(root/'provenance.json')))
    emit(root,state='preflight_completed',checks=3,optimizer_steps=0)


def fit(arm,root,cfg,data,valid,monitor,prototypes):
    folder=root/'runs'/arm;folder.mkdir(exist_ok=True)
    signature=sha256_file(root/'provenance.json')
    if (folder/'completion.json').exists():
        done=json.loads((folder/'completion.json').read_text())
        assert done['provenance_sha256']==signature and sha256_file(folder/'best.pt')==done['best_sha256'];return
    model=model_for(arm,root,cfg,prototypes).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    censoring=fit_censoring(data.rows);history=[];best=float('inf');best_epoch=0;steps=0;start=0
    seed(cfg['seed']+151000);last=folder/'last.pt'
    if last.exists():
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        assert checkpoint['provenance_sha256']==signature
        model.load_state_dict(checkpoint['model'],strict=True);opt.load_state_dict(checkpoint['optimizer'])
        history=checkpoint['history'];best=checkpoint['best'];best_epoch=checkpoint['best_epoch'];steps=checkpoint['steps'];start=checkpoint['epoch']
        random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng']);torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    write(folder/'metadata.json',dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),train_patients=len(data),validation_patients=len(valid),max_epochs=4,provenance_sha256=signature))
    for epoch in range(start,4):
        began=time.time();torch.cuda.reset_peak_memory_stats();model.train();opt.zero_grad(set_to_none=True);total=0.
        order=np.random.default_rng(cfg['seed']+epoch).permutation(len(data))
        for j,i in enumerate(order):
            loss=objective(model,data,int(i),epoch,'cuda',cfg['loss_weights'])
            assert torch.isfinite(loss)
            group=min(cfg['accumulate_cases'],len(order)-(j//cfg['accumulate_cases'])*cfg['accumulate_cases'])
            (loss/group).backward();total+=float(loss.detach())
            if (j+1)%cfg['accumulate_cases']==0 or j+1==len(order):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True);steps+=1
                emit(root,state='training',arm=arm,epoch=epoch+1,patients_done=j+1,patients_total=len(data),optimizer_steps=steps,
                    objective=total/(j+1),gradient_norm=float(norm),cache_gib=data.region_cache.bytes/1024**3)
        scores,_,_=evaluate(model,valid,cfg,censoring,['full'],root,arm+'/validation')
        training,_,_=evaluate(model,monitor,cfg,censoring,['full'],root,arm+'/training_monitor')
        selection=scores['full']['nll']
        history.append(dict(epoch=epoch+1,train_objective=total/len(data),selection_nll=selection,validation=scores,
            monitor_full=training['full'],seconds=time.time()-began,peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        if selection<best:
            best=selection;best_epoch=epoch+1;atomic_save(folder/'best.pt',dict(model=model.state_dict(),epoch=best_epoch,provenance_sha256=signature))
        atomic_save(last,dict(model=model.state_dict(),optimizer=opt.state_dict(),history=history,best=best,best_epoch=best_epoch,
            epoch=epoch+1,steps=steps,provenance_sha256=signature,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        write(folder/'history.json',history);plot_learning(folder,history);report(root)
        emit(root,state='epoch_completed',arm=arm,epoch=epoch+1,best_epoch=best_epoch,full_nll=selection,full_c_index=scores['full']['c_index'])
    best_state=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(best_state['model'],strict=True)
    scores,predictions,usage=evaluate(model,valid,cfg,censoring,['zero','0.2','full'],root,arm+'/fixed_best')
    train_scores,_,_=evaluate(model,monitor,cfg,censoring,['full'],root,arm+'/fixed_best_training_monitor')
    write(folder/'inner_metrics.json',scores);write(folder/'monitor_metrics.json',train_scores)
    write(folder/'predictions_private.json',dict(validation=predictions,usage=usage))
    final_plots(folder,scores,predictions,valid.rows)
    write(folder/'completion.json',dict(arm=arm,state='completed',epochs=4,best_epoch=best_epoch,optimizer_steps=steps,
        best_sha256=sha256_file(folder/'best.pt'),provenance_sha256=signature,strict_reload=True,outer_predictions=0,at=time.time()))
    report(root);emit(root,state='fit_completed',arm=arm,best_epoch=best_epoch,full_c_index=scores['full']['c_index'])
    del model,opt;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();root=args.root;cfg=json.loads((root/'config.json').read_text());verify(root)
    assert cfg['max_epochs']==4 and cfg['cancers']==['CRC'] and cfg['fold']==0 and cfg['arms']==ARMS
    split=json.loads((root/'split_private.json').read_text());tr,va=split['train'],split['validation']
    assert len(tr)==292 and len(va)==147 and all(r['cancer']=='CRC' for r in tr+va)
    assert not {r['canonical_case_key'] for r in tr}&{r['canonical_case_key'] for r in va}
    prototypes=check_prototype_source(root/'prototypes.npy',root/'prototype_provenance.json',tr)
    torch.set_num_threads(cfg['cpu_threads']);torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data=CachedDataset(tr,root,cfg,cfg['train_cache_gib']);valid=CachedDataset(va,root,cfg,cfg['validation_cache_gib'])
    indices=sorted(range(len(tr)),key=lambda i:hashlib.sha256(tr[i]['canonical_case_key'].encode()).hexdigest())[:64]
    monitor=DatasetSubset(data,indices)
    report(root)
    if not args.execute:preflight(root,cfg,data,prototypes);return
    proof=json.loads((root/'preflight_completed.json').read_text());assert proof['provenance_sha256']==sha256_file(root/'provenance.json')
    for arm in ARMS:fit(arm,root,cfg,data,valid,monitor,prototypes)
    write(root/'suite_completed.json',dict(new_fits=3,baseline_reused=True,epochs_per_fit=4,outer_predictions=0,
        provenance_sha256=sha256_file(root/'provenance.json'),at=time.time()))
    emit(root,state='completed',new_fits=3,baseline_reused=True,outer_predictions=0)


if __name__=='__main__':
    try:main()
    except Exception:
        import sys
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1]);root.mkdir(exist_ok=True,parents=True)
            error=traceback.format_exc();write(root/'failure.json',dict(at=time.time(),traceback=error));emit(root,state='failed',error=error)
        raise
