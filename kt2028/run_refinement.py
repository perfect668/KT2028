"""W159: four bounded, sequential single-change R3 development fits."""
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

from .refined_survival_network import RefinedSurvivalPredictor
from .exposure_survival import exposure_nll
from .survival import discrete_nll
from .observed_cache import CachedDataset,DatasetSubset
from .run_component_ablation import evaluate,atomic_save,seed
from .run_multicancer import write
from .censored_metrics import fit_censoring
from .fusion_data import move
from .uni_encoder import sha256_file

STAGES=['gate','residual','kv_norm','exposure']
DEFAULT=dict(identity_coarse_gate=False,fine_attention_scale=None,normalize_fine_kv=False,exposure_likelihood=False)
TITLES={'baseline':'Original R3 (reused)','gate':'Identity-neutral coarse gate',
        'residual':'Fine attention residual scale','kv_norm':'Fine key/value LayerNorm','exposure':'Within-interval exposure likelihood'}


def emit(root,**values):
    values['updated_unix']=time.time();write(root/'status.json',values)
    print(json.dumps(values,allow_nan=False),flush=True)


def verify(root):
    doc=json.loads((root/'provenance.json').read_text())
    for name,digest in doc['files'].items():assert sha256_file(root/name)==digest,name
    assert sha256_file(Path(doc['baseline_checkpoint']))==doc['baseline_best_sha256']
    return doc


def model_for(spec,cfg):
    seed(cfg['seed'])
    options={k:v for k,v in spec.items() if k!='exposure_likelihood'}
    return RefinedSurvivalPredictor(**options,**cfg['architecture'])


def propose(stage,retained):
    result=copy.deepcopy(retained)
    key,value={'gate':('identity_coarse_gate',True),'residual':('fine_attention_scale',.1),
        'kv_norm':('normalize_fine_kv',True),'exposure':('exposure_likelihood',True)}[stage]
    assert result[key]==DEFAULT[key]
    result[key]=value
    return result


def training_loss(model,data,i,epoch,cfg,spec):
    partial,target=data.sample(i,'random',epoch)
    out=model(**move(partial,'cuda'))
    complete,_=data.sample(i,'full',epoch)
    full=model(**move(complete,'cuda'),return_aux=True)
    assert full['fine_aux_logits'] is not None
    nll=(lambda logits:exposure_nll(logits,target,cfg['data_contract']['interval_edges_days'])) if spec['exposure_likelihood'] else (lambda logits:discrete_nll(logits,target))
    values=dict(partial=nll(out['logits']),full=nll(full['logits']),
        coarse_aux=nll(full['coarse_aux_logits']),fine_aux=nll(full['fine_aux_logits']))
    return sum(cfg['loss_weights'][key]*value for key,value in values.items())


def decision(metrics,reference):
    # Freeze before training. No tradeoff hidden by selecting on C alone.
    current,previous=metrics['full'],reference['full']
    checks=dict(nll_improves=current['nll']<previous['nll']-1e-6,
        c_index_noninferior=current['c_index']>=previous['c_index']-1e-6,
        ibs_noninferior=current['integrated_brier_annual_grid']<=previous['integrated_brier_annual_grid']+1e-6)
    return dict(retain=all(checks.values()),checks=checks,
        delta_full={key:current[key]-previous[key] for key in ['nll','c_index','integrated_brier_annual_grid']})


def plot_learning(folder,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not history:return
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),constrained_layout=True)
    for ax,key,title in zip(axes,['nll','c_index','integrated_brier_annual_grid'],['Annual NLL (lower)','C-index (higher)','IPCW IBS (lower)']):
        ax.plot([r['epoch'] for r in history],[r['monitor_full'][key] for r in history],'o-',label='Fixed Train monitor, 64')
        ax.plot([r['epoch'] for r in history],[r['validation']['full'][key] for r in history],'o-',label='Development validation, 147')
        ax.set(title=title,xlabel='Epoch');ax.legend(fontsize=7)
    fig.suptitle(folder.name+' | four epochs, fixed CRC split')
    for ext in ['png','svg','pdf']:fig.savefig(folder/('learning_curves.'+ext),dpi=170)
    plt.close(fig)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=json.loads((root/'baseline_reference.json').read_text())
    results=[dict(stage='baseline',metrics=base['metrics'],spec=DEFAULT,best_epoch=base['best_epoch'],retained=True)]
    retained='baseline'
    for stage in STAGES:
        path=root/'runs'/stage/'completion.json'
        if path.exists():
            done=json.loads(path.read_text())
            item=dict(stage=stage,metrics=json.loads((path.parent/'inner_metrics.json').read_text()),
                spec=done['spec'],best_epoch=done['best_epoch'],retained=done['decision']['retain'],
                compared_to=done['compared_to'],decision=done['decision'])
            results.append(item)
            if item['retained']:retained=stage
    summary=dict(completed_new_fits=len(results)-1,expected_new_fits=4,split='CRC fixed inner0',
        train_patients=292,validation_patients=147,max_epochs=4,retained_stage=retained,
        baseline_reused=True,outer_predictions=0,results=results)
    write(root/'summary.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),constrained_layout=True)
    labels=['R3','Gate','Residual','KV norm','Exposure']
    for row,mode in enumerate(['full','0.2']):
        for ax,key,title in zip(axes[row],['c_index','nll','integrated_brier_annual_grid'],['C-index (higher)','Annual NLL (lower)','IPCW IBS (lower)']):
            for j,stage in enumerate(['baseline']+STAGES):
                item=next((r for r in results if r['stage']==stage),None)
                if item:
                    value=item['metrics'][mode][key];color='#216a83' if stage==retained else '#8995a3'
                    ax.bar(j,value,color=color);ax.text(j,value,f'{value:.4f}',ha='center',va='bottom',fontsize=8)
                else:ax.text(j,.03,'pending',ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
            ax.set(title=('100% fine | ' if mode=='full' else '20% fine | ')+title,xticks=range(5),xticklabels=labels)
            if key=='c_index':ax.set_ylim(0,1)
    fig.suptitle(f'Sequential R3 repair | {len(results)-1}/4 fits | retained: {retained} | development only')
    for ext in ['png','svg','pdf']:fig.savefig(root/('comparison.'+ext),dpi=180)
    plt.close(fig)
    rows=''.join('<tr><td>'+TITLES[r['stage']]+'</td><td>'+str(r.get('compared_to','—'))+'</td><td>'+str(r['retained'])+'</td>'+''.join('<td>'+f"{r['metrics']['full'][k]:.5f}"+'</td>' for k in ['c_index','nll','integrated_brier_annual_grid'])+'</tr>' for r in results)
    cards=''.join(f'<h2>{TITLES[s]}</h2><img src="runs/{s}/learning_curves.png">' for s in STAGES if (root/'runs'/s/'history.json').exists())
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>R3逐项修复验证</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:32px auto;color:#213747;padding:0 18px}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse;width:100%}</style><h1>R3逐项修复与效果验证</h1><p>仅CRC固定开发划分：Train292/验证147，原R3对照复用，4项各最多4轮。每项从相同初始种子重新拟合，基于当前保留结构只新增一个因素。仅当full年度NLL降低且C-index/IBS均不退步时保留；不按20%结果追加选择。年度OS输出和评价保持，区间暴露似然单独作为第4项。</p><p>此验证集已参与开发及反复选择，结果是短程开发证据，不代表独立测试泛化。</p><table><tr><th>候选</th><th>对照</th><th>保留</th><th>C-index ↑</th><th>NLL ↓</th><th>IBS ↓</th></tr>'+rows+'</table><img src="comparison.png">'+cards,encoding='utf-8')
    return summary


@torch.no_grad()
def mechanism(model,data):
    keys=sorted(range(len(data)),key=lambda i:hashlib.sha256(data.rows[i]['canonical_case_key'].encode()).hexdigest())[:8]
    values=[]
    for i in keys:
        for mode in ['zero','0.2','full']:
            pools=[];fine=[];calls=[]
            hs=[model.coarse_pool.register_forward_hook(lambda m,a,o:pools.append(o[0].detach())),
                model.fine_encoder.register_forward_hook(lambda m,a,o:fine.append(o.detach().reshape(-1,model.dim))),
                model.local_attention.register_forward_pre_hook(lambda m,a,k:calls.append((a[0].detach(),a[1].detach(),k['key_padding_mask'].detach())),with_kwargs=True)]
            inputs,_=data.sample(i,mode,0)
            with torch.no_grad():out=model(**move(inputs,'cuda'))
            for handle in hs:handle.remove()
            v=dict(mode=mode,coarse_attenuation=float(out['coarse_patient'].norm()/pools[0][0].norm()))
            if fine:
                x=torch.cat(fine);v['fine_difference_energy_fraction']=float((x-x.mean(0)).square().mean()/x.square().mean())
                q,k,mask=calls[0];layer=model.local_attention;dim=model.dim;h=layer.num_heads;hd=dim//h
                pq=torch.nn.functional.linear(q,layer.in_proj_weight[:dim],layer.in_proj_bias[:dim]).reshape(len(q),1,h,hd).transpose(1,2)
                pk=torch.nn.functional.linear(k,layer.in_proj_weight[dim:2*dim],layer.in_proj_bias[dim:2*dim]).reshape(len(k),k.shape[1],h,hd).transpose(1,2)
                score=(pq@pk.transpose(-1,-2)).squeeze(2)/hd**.5
                weights=score.masked_fill(mask[:,None,:],-torch.inf).softmax(-1);n=(~mask).sum(1);valid=n>1
                v['local_attention_entropy']=float((-(weights*weights.clamp_min(1e-30).log()).sum(-1)[valid]/n[valid,None].log()).mean())
            values.append(v)
    result={}
    for mode in ['zero','0.2','full']:
        selected=[v for v in values if v['mode']==mode]
        result[mode]={k:dict(median=float(np.median([v[k] for v in selected])),mean=float(np.mean([v[k] for v in selected]))) for k in selected[0] if k!='mode'}
    if hasattr(model.fine_encoder.layers[0],'attention_scale'):
        scale=model.fine_encoder.layers[0].attention_scale.detach()
        result['learned_attention_scale']=dict(mean=float(scale.mean()),min=float(scale.min()),max=float(scale.max()))
    return dict(train_samples=8,statistics=result)


def preflight(root,cfg,data):
    largest=max(range(len(data)),key=lambda i:len(data.case(i).coarse))
    target_i=next(i for i,r in enumerate(data.rows) if r['survival_target']['event'])
    proofs=[];spec=copy.deepcopy(DEFAULT)
    for stage in STAGES:
        spec=propose(stage,spec);m=model_for(spec,cfg).cuda().train()
        before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        emit(root,state='preflight',stage=stage);torch.cuda.reset_peak_memory_stats()
        loss=training_loss(m,data,largest,0,cfg,spec);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in before.items())
        m.zero_grad(set_to_none=True);m.eval();inputs,_=data.sample(target_i,'full')
        clone=model_for(spec,cfg).cuda().eval();clone.load_state_dict(m.state_dict(),strict=True)
        with torch.no_grad():torch.testing.assert_close(m(**move(inputs,'cuda'))['survival'],clone(**move(inputs,'cuda'))['survival'],atol=1e-6,rtol=1e-5)
        proofs.append(dict(stage=stage,spec=copy.deepcopy(spec),finite_training_gradient=True,strict_reload=True,
            unchanged_parameters=True,optimizer_steps=0,peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del m,clone,loss,before;torch.cuda.empty_cache()
    edges=cfg['data_contract']['interval_edges_days'];early=[r['survival_target'] for r in data.rows if not r['survival_target']['event'] and r['survival_target']['completed_intervals']==0]
    for target in early:
        l=torch.zeros(5,requires_grad=True);loss=exposure_nll(l,target,edges);g,=torch.autograd.grad(loss,l)
        assert float(loss.detach())>0 and float(g[0])>0
    write(root/'preflight_completed.json',dict(checks=proofs,early_censor_positive_gradients=len(early),
        optimizer_steps=0,provenance_sha256=sha256_file(root/'provenance.json')))


def fit(stage,spec,compared_to,root,cfg,data,valid,monitor):
    folder=root/'runs'/stage;folder.mkdir(parents=True,exist_ok=True);signature=sha256_file(root/'provenance.json')
    if (folder/'completion.json').exists():
        done=json.loads((folder/'completion.json').read_text())
        assert done['provenance_sha256']==signature and done['spec']==spec and done['compared_to']==compared_to
        assert sha256_file(folder/'best.pt')==done['best_sha256'];return
    write(folder/'spec.json',dict(stage=stage,spec=spec,compared_to=compared_to,provenance_sha256=signature))
    model=model_for(spec,cfg).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    history=[];best=float('inf');best_epoch=0;steps=0;start=0;seed(cfg['seed']+151000)
    last=folder/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        assert saved['spec']==spec and saved['provenance_sha256']==signature
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];steps=saved['steps'];start=saved['epoch']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    censoring=fit_censoring(data.rows)
    write(folder/'metadata.json',dict(parameters=sum(p.numel() for p in model.parameters()),train_patients=len(data),validation_patients=len(valid),spec=spec,max_epochs=4))
    for epoch in range(start,4):
        began=time.time();model.train();opt.zero_grad(set_to_none=True);total=0.
        order=np.random.default_rng(cfg['seed']+epoch).permutation(len(data))
        for j,i in enumerate(order):
            loss=training_loss(model,data,int(i),epoch,cfg,spec)
            assert torch.isfinite(loss)
            group=min(cfg['accumulate_cases'],len(order)-(j//cfg['accumulate_cases'])*cfg['accumulate_cases'])
            (loss/group).backward();total+=float(loss.detach())
            if (j+1)%cfg['accumulate_cases']==0 or j+1==len(order):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True);steps+=1
                emit(root,state='training',stage=stage,epoch=epoch+1,patients_done=j+1,patients_total=len(data),optimizer_steps=steps,objective=total/(j+1),gradient_norm=float(norm))
        scores,_,_=evaluate(model,valid,cfg,censoring,['full'],root,stage+'/validation')
        monitored,_,_=evaluate(model,monitor,cfg,censoring,['full'],root,stage+'/train_monitor')
        selection=scores['full']['nll'];history.append(dict(epoch=epoch+1,train_objective=total/len(data),
            selection_nll=selection,validation=scores,monitor_full=monitored['full'],seconds=time.time()-began))
        if selection<best:
            best=selection;best_epoch=epoch+1;atomic_save(folder/'best.pt',dict(model=model.state_dict(),epoch=best_epoch,spec=spec,provenance_sha256=signature))
        atomic_save(last,dict(model=model.state_dict(),optimizer=opt.state_dict(),history=history,best=best,best_epoch=best_epoch,epoch=epoch+1,steps=steps,
            spec=spec,provenance_sha256=signature,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        write(folder/'history.json',history);plot_learning(folder,history);report(root)
        emit(root,state='epoch_completed',stage=stage,epoch=epoch+1,full_nll=selection,full_c_index=scores['full']['c_index'],best_epoch=best_epoch)
    checkpoint=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(checkpoint['model'],strict=True)
    scores,predictions,usage=evaluate(model,valid,cfg,censoring,['zero','0.2','full'],root,stage+'/fixed_best')
    monitored,_,_=evaluate(model,monitor,cfg,censoring,['full'],root,stage+'/fixed_best_train_monitor')
    write(folder/'inner_metrics.json',scores);write(folder/'monitor_metrics.json',monitored)
    write(folder/'predictions_private.json',dict(validation=predictions,usage=usage))
    write(folder/'mechanism.json',mechanism(model,data))
    reference=json.loads((root/'baseline_reference.json').read_text())['metrics'] if compared_to=='baseline' else json.loads((root/'runs'/compared_to/'inner_metrics.json').read_text())
    choice=decision(scores,reference)
    write(folder/'completion.json',dict(state='completed',stage=stage,spec=spec,compared_to=compared_to,decision=choice,
        epochs=4,best_epoch=best_epoch,optimizer_steps=steps,best_sha256=sha256_file(folder/'best.pt'),provenance_sha256=signature,strict_reload=True,outer_predictions=0,at=time.time()))
    report(root);emit(root,state='fit_completed',stage=stage,decision=choice,best_epoch=best_epoch)
    del model,opt;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();root=args.root;cfg=json.loads((root/'config.json').read_text());verify(root)
    assert cfg['max_epochs']==4 and cfg['cancers']==['CRC'] and cfg['stages']==STAGES
    split=json.loads((root/'split_private.json').read_text());tr,va=split['train'],split['validation']
    assert len(tr)==292 and len(va)==147 and not {r['canonical_case_key'] for r in tr}&{r['canonical_case_key'] for r in va}
    assert all(r['cancer']=='CRC' for r in tr+va)
    torch.set_num_threads(cfg['cpu_threads']);torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data=CachedDataset(tr,root,cfg,cfg['train_cache_gib']);valid=CachedDataset(va,root,cfg,cfg['validation_cache_gib'])
    ix=sorted(range(len(tr)),key=lambda i:hashlib.sha256(tr[i]['canonical_case_key'].encode()).hexdigest())[:64];monitor=DatasetSubset(data,ix)
    report(root)
    if not args.execute:preflight(root,cfg,data);return
    proof=json.loads((root/'preflight_completed.json').read_text());assert proof['provenance_sha256']==sha256_file(root/'provenance.json')
    retained=copy.deepcopy(DEFAULT);compared_to='baseline'
    for stage in STAGES:
        spec=propose(stage,retained);fit(stage,spec,compared_to,root,cfg,data,valid,monitor)
        choice=json.loads((root/'runs'/stage/'completion.json').read_text())
        if choice['decision']['retain']:retained=spec;compared_to=stage
    reference=json.loads((root/'baseline_reference.json').read_text())
    path=Path(json.loads((root/'provenance.json').read_text())['baseline_checkpoint']) if compared_to=='baseline' else root/'runs'/compared_to/'best.pt'
    write(root/'final_model.json',dict(stage=compared_to,spec=retained,checkpoint=str(path),checkpoint_sha256=sha256_file(path),
        baseline_reused=compared_to=='baseline',development_only=True,external_validation_completed=False))
    write(root/'suite_completed.json',dict(status='completed',new_fits=4,epochs_per_fit=4,retained_stage=compared_to,
        final_model_sha256=sha256_file(root/'final_model.json'),provenance_sha256=sha256_file(root/'provenance.json'),outer_predictions=0,at=time.time()))
    report(root);emit(root,state='completed',new_fits=4,retained_stage=compared_to,outer_predictions=0)


if __name__=='__main__':
    try:main()
    except Exception:
        import sys
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1]);root.mkdir(parents=True,exist_ok=True)
            error=traceback.format_exc();write(root/'failure.json',dict(traceback=error,at=time.time()));emit(root,state='failed',error=error)
        raise
