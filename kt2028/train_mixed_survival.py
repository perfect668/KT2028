"""Mixed single-head training entry. Default preflight; no assets => no model/run."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np

from .censored_metrics import fit_censoring, survival_metrics
from .mixed_survival_data import inspect_asset
from .uni_encoder import sha256_file


def progress(message):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ')+message,flush=True)


def metric_text(value):
    return 'NA' if value is None else f'{value:.5f}'


def asset_report(documents, roots, contract, verify_content=True):
    good, missing, invalid, assets = 0, [], [], {}
    seen = set()
    for doc in documents:
        for patient in doc['patients']:
            for slide in patient['slides']:
                sid=slide['slide_id']
                if sid in seen: raise ValueError('A slide occurs in more than one patient/split')
                seen.add(sid)
                folder=Path(roots[patient['cohort']])/slide['asset_id']
                if not (folder/'receipt.json').is_file():
                    missing.append(dict(cohort=patient['cohort'],slide_id=sid,reason='missing_5x20x_receipt'))
                    continue
                try:
                    receipt=inspect_asset(folder,slide,contract['feature_protocol'])
                    if verify_content:
                        from .prepare_tcga import verify_output
                        verify_output(folder,receipt)
                    assets[sid]=sha256_file(folder/'receipt.json')
                    good+=1
                except Exception as error:
                    invalid.append(dict(cohort=patient['cohort'],slide_id=sid,
                        error=type(error).__name__+': '+str(error)[:300]))
                if len(seen)%50==0:
                    progress(f'输入核查：已检查{len(seen)}张，合格{good}张；尚未开始训练')
    progress(f'输入核查结束：合格{good}张，缺失{len(missing)}张，无效{len(invalid)}张')
    return dict(expected_slides=len(seen),verified_slides=good,missing=missing,invalid=invalid,
                feature_content_checked=verify_content,
                assets_sha256=hashlib.sha256(json.dumps(assets,sort_keys=True).encode()).hexdigest())


def evaluate(model,dataset,modes,device,edges,censoring):
    import torch
    from .fusion_data import move
    from .survival import discrete_nll
    model.eval(); scores={}
    with torch.no_grad():
        for mode in modes:
            progress(f'验证观察模式={mode}，共{len(dataset)}名患者')
            curves=[]; losses=[]
            for i in range(len(dataset)):
                inputs,target=dataset.sample(i,mode,0)
                out=model(**move(inputs,device))
                curves.append(out['survival'].cpu().numpy()); losses.append(float(discrete_nll(out['logits'],target)))
                if (i+1)%8==0 or i+1==len(dataset):
                    progress(f'验证模式={mode}，已完成{i+1}/{len(dataset)}名患者')
            curves=np.stack(curves)
            pooled=survival_metrics(dataset.rows,curves,edges,censoring)
            by_source={}
            for source in sorted({r['cohort'] for r in dataset.rows}):
                ix=[i for i,r in enumerate(dataset.rows) if r['cohort']==source]
                value=survival_metrics([dataset.rows[i] for i in ix],curves[ix],edges,censoring)
                value['nll']=float(np.mean(np.asarray(losses)[ix])); by_source[source]=value
            scores[mode]=dict(nll=float(np.mean(losses)),pooled=pooled,by_source=by_source)
            five_year=pooled['by_day'][str(int(edges[-1]))]
            progress(f"验证模式={mode}：NLL={scores[mode]['nll']:.5f}，"
                     f"C-index={metric_text(pooled['c_index'])}，"
                     f"五年AUROC={metric_text(five_year['auroc'])}，"
                     f"五年Brier={metric_text(five_year['brier'])}，"
                     f"IBS={metric_text(pooled['integrated_brier_annual_grid'])}")
            print(json.dumps(dict(event='validation_mode_complete',mode=mode,nll=scores[mode]['nll'],
                                  metrics=scores[mode])),flush=True)
    return scores


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--train-manifest',type=Path,required=True)
    ap.add_argument('--validation-manifest',type=Path,required=True)
    ap.add_argument('--surgen-features',type=Path)
    ap.add_argument('--tcga-features',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--preflight-report',type=Path,required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    progress('读取已登记配置和患者名单；先复核资产，未通过则不训练')
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['schema']!='KT2028-mixed-survival-predictor-1' or cfg['old_checkpoint_reuse']:
        raise ValueError('Require registered fresh mixed protocol')
    documents=[json.loads(p.read_text(encoding='utf-8')) for p in (args.train_manifest,args.validation_manifest)]
    lock=json.loads((args.train_manifest.parent/'split_lock.json').read_text(encoding='utf-8'))
    if lock['schema']!='KT2028-mixed-split-lock-1' or lock['config_sha256']!=sha256_file(args.config):
        raise ValueError('Split registration/configuration lock differs')
    for path,name in ((args.train_manifest,'train_manifest'),(args.validation_manifest,'validate_manifest')):
        if sha256_file(path)!=lock['manifest_sha256'][name]:
            raise ValueError('Patient split changed after registration')
    for doc,partition in zip(documents,('train','validate')):
        if doc['partition']!=partition or doc['data_contract']!=cfg['data_contract']:
            raise ValueError('Split or clinical/feature contract mismatch')
        if doc['sources']['config_sha256']!=sha256_file(args.config):
            raise ValueError('Configuration changed since registered split')
    cohorts=cfg.get('cohorts',['surgen','tcga'])
    if not cohorts or len(set(cohorts))!=len(cohorts) or not set(cohorts)<={'surgen','tcga'}:
        raise ValueError('Invalid registered cohorts')
    for doc in documents:
        if {r['cohort'] for r in doc['patients']}!=set(cohorts):
            raise ValueError('Manifest cohorts differ from registered configuration')
    if 'surgen' in cohorts and args.surgen_features is None:
        raise ValueError('SurGen feature root required for registered cohort')
    train_ids={r['canonical_case_key'] for r in documents[0]['patients']}
    val_ids={r['canonical_case_key'] for r in documents[1]['patients']}
    if train_ids&val_ids or len(train_ids)!=len(documents[0]['patients']) or len(val_ids)!=len(documents[1]['patients']):
        raise ValueError('Patient overlap or duplicates')
    roots=dict(surgen=str(args.surgen_features),tcga=str(args.tcga_features))
    assets=asset_report(documents,roots,cfg['data_contract'])
    provenance=dict(config_sha256=sha256_file(args.config),
        train_manifest_sha256=sha256_file(args.train_manifest),
        validation_manifest_sha256=sha256_file(args.validation_manifest),assets=assets,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
        old_checkpoint_reused=False,external_test_dataset_secured=False,
        source_shas={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
    if assets['missing'] or assets['invalid']:
        report=dict(provenance,status='BLOCKED_MISSING_OR_INVALID_ASSETS',training_started=False,
                    optimizer_steps=0,epochs=0,training_run_created=False)
        args.preflight_report.parent.mkdir(parents=True,exist_ok=True)
        with args.preflight_report.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
        print(json.dumps(dict(status=report['status'],verified=assets['verified_slides'],
            missing=len(assets['missing']),invalid=len(assets['invalid']),training_started=False)),flush=True)
        raise SystemExit(2)
    # Heavy/model imports and any model access occur only after ALL assets pass.
    import torch
    from .mixed_survival_data import MixedSurvivalDataset
    from .survival_network import SurvivalPredictor
    from .train_survival import objective, readiness_check
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    train=MixedSurvivalDataset(args.train_manifest,roots,contract=cfg['data_contract'],seed=cfg['seed'])
    val=MixedSurvivalDataset(args.validation_manifest,roots,contract=cfg['data_contract'],purpose='model_selection',seed=cfg['seed'])
    censoring=fit_censoring(train.rows)  # No validation observations used in fitted censoring.
    model=SurvivalPredictor(**cfg['architecture']).to(args.device)
    progress('资产齐备，执行真实Train前向/梯度/重载检查；此检查不更新参数')
    # Check one informative real training patient from EACH source, including multi-slide if available.
    checks={}
    for cohort in cohorts:
        eligible=[i for i,r in enumerate(train.rows) if r['cohort']==cohort and
                  (r['survival_target']['event'] or r['survival_target']['completed_intervals']>0)]
        i=max(eligible,key=lambda i:(len(train.rows[i]['slides']),len(train.case(i).coarse)))
        checks[cohort]=readiness_check(model,train,i,args.device,cfg['loss_weights'])
    report=dict(provenance,status='READY_NOT_STARTED',checks=checks,training_started=False,
                optimizer_steps=0,epochs=0,training_run_created=False)
    args.preflight_report.parent.mkdir(parents=True,exist_ok=True)
    with args.preflight_report.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    if not args.execute:
        print(json.dumps(dict(status='READY_NOT_STARTED',checks=checks)),flush=True)
        return
    if args.output.exists(): raise FileExistsError('Preserve earlier run')
    if subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).parent,text=True).strip():
        raise ValueError('Commit registered source before fitting')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    args.output.mkdir(parents=True)
    (args.output/'run.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    progress(f"正式训练开始：Train={len(train)}人，Val={len(val)}人，最多{cfg['max_epochs']}轮；输出={args.output}")
    best,stale,best_epoch=float('inf'),0,-1
    for epoch in range(cfg['max_epochs']):
        progress(f"第{epoch+1}/{cfg['max_epochs']}轮开始")
        model.train(); order=np.random.default_rng(cfg['seed']+epoch).permutation(len(train))
        optimizer.zero_grad(set_to_none=True); total=0.; accumulate=cfg['accumulate_cases']
        for j,i in enumerate(order):
            loss=objective(model,train,int(i),epoch,args.device,cfg['loss_weights'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training objective')
            group_size=min(accumulate,len(order)-(j//accumulate)*accumulate)
            (loss/group_size).backward(); total+=float(loss.detach())
            if (j+1)%accumulate==0 or j+1==len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                progress(f'第{epoch+1}轮训练：{j+1}/{len(order)}名患者，平均训练总损失={total/(j+1):.5f}')
        scores=evaluate(model,val,cfg['validation_modes'],args.device,cfg['data_contract']['interval_edges_days'],censoring)
        selection=float(np.mean([scores[m]['nll'] for m in cfg['validation_modes']]))
        with (args.output/'epochs.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=epoch+1,train_objective=total/len(train),selection_nll=selection,validation=scores))+'\n')
        if selection<best:
            best,stale,best_epoch=selection,0,epoch+1
            temporary=args.output/'best.partial.pt'
            torch.save(dict(model=model.state_dict(),epoch=best_epoch,selection_nll=best,provenance=provenance),temporary)
            temporary.replace(args.output/'best.pt')
        else: stale+=1
        progress(f"第{epoch+1}轮结束：五模式平均验证NLL={selection:.5f}；"
                 f"最佳第{best_epoch}轮/NLL={best:.5f}；连续未改善{stale}/{cfg['patience']}轮")
        if stale>=cfg['patience']: break
    saved=torch.load(args.output/'best.pt',map_location=args.device,weights_only=True)
    model.load_state_dict(saved['model'],strict=True)
    (args.output/'completion.json').write_text(json.dumps(dict(completed_epochs=epoch+1,
        selected_epoch=best_epoch,checkpoint_sha256=sha256_file(args.output/'best.pt'),strict_reload=True,
        test_used=False,rl_started=False),indent=2),encoding='utf-8')
    progress(f'训练完成：实际{epoch+1}轮，保留第{best_epoch}轮；权重严格重载通过，未运行Test/RL')


if __name__=='__main__': main()
