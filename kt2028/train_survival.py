"""Default = readiness checks only. Fitting requires explicit --execute; no Test."""
import argparse
import hashlib
import io
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import zarr

from .fusion_data import move
from .prepare_survival import contract_hash
from .survival import discrete_nll, common_censor_metrics, validate_edges
from .survival_data import SurvivalDataset
from .survival_network import SurvivalPredictor
from .uni_encoder import sha256_file


def progress(phase, completed=None, total=None, detail=''):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    fraction = f' {completed}/{total} ({100*completed/total:.1f}%)' if total else ''
    print(f'[{stamp}] {phase}{fraction} {detail}'.rstrip(), flush=True)


def metric_text(value):
    return '不可计算' if value is None else f'{value:.5f}'


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def tree_sha256(folder):
    """Hash an existing feature array's encoded chunks without model acquisition."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    if not files:
        raise ValueError('Missing fine feature array files')
    h = hashlib.sha256()
    for path in files:
        h.update(path.relative_to(folder).as_posix().encode()+b'\0')
        h.update(sha256_file(path).encode()+b'\n')
    return h.hexdigest()


def validate_config(cfg):
    if cfg['schema'] != 'KT2028-survival-predictor-1':
        raise ValueError('Require registered survival config')
    c = cfg['data_contract']
    edges = validate_edges(c['interval_edges_days'])
    if edges[-1] != c['administrative_censor_day'] or len(edges)-1 != cfg['architecture']['intervals']:
        raise ValueError('Head/window/interval mismatch')
    if set(cfg['loss_weights']) != {'partial', 'full', 'coarse_aux', 'fine_aux'} or any(v <= 0 for v in cfg['loss_weights'].values()):
        raise ValueError('Require positive registered loss weights')
    if not np.isclose(sum(cfg['loss_weights'].values()), 1.):
        raise ValueError('Loss weights must sum to1')
    if any(not isinstance(cfg[k], int) or cfg[k] <= 0 for k in ('max_epochs', 'patience', 'accumulate_cases')):
        raise ValueError('Require positive epochs/patience/accumulation')
    if any(not np.isfinite(cfg[k]) or cfg[k] <= 0 for k in ('learning_rate', 'weight_decay', 'gradient_clip')):
        raise ValueError('Invalid optimizer configuration')
    if cfg['validation_modes'] != ['zero', '0.1', '0.2', '0.4', 'full']:
        raise ValueError('Keep the registered fixed selection observations')
    if cfg['old_checkpoint_reuse'] or cfg['test_used_for_model_selection']:
        raise ValueError('No binary predecessor or Test model selection')


def objective(model, dataset, i, epoch, device, weights):
    inputs, target = dataset.sample(i, 'random', epoch)
    out = model(**move(inputs, device))
    loss_partial = discrete_nll(out['logits'], target)
    full_inputs, _ = dataset.sample(i, 'full', epoch)
    full = model(**move(full_inputs, device), return_aux=True)
    if full['fine_aux_logits'] is None:
        raise ValueError('Registered survival case has no acquirable fine')
    losses = {'partial':loss_partial, 'full':discrete_nll(full['logits'], target),
              'coarse_aux':discrete_nll(full['coarse_aux_logits'], target),
              'fine_aux':discrete_nll(full['fine_aux_logits'], target)}
    return sum(weights[k]*v for k, v in losses.items())


def evaluate(model, dataset, modes, device, edges):
    if dataset.rows and any(r['survival_target']['event'] not in (0, 1) for r in dataset.rows):
        raise ValueError('Invalid evaluation targets')
    model.eval()
    scores = {}
    with torch.no_grad():
        for mode in modes:
            progress(f'验证：观察模式 {mode}', 0, len(dataset))
            times, events, curves, losses, regions, patches, fractions = [], [], [], [], [], [], []
            for i in range(len(dataset)):
                inputs, target = dataset.sample(i, mode, epoch=0)
                out = model(**move(inputs, device))
                losses.append(float(discrete_nll(out['logits'], target)))
                times.append(target['time_days']); events.append(target['event'])
                curves.append(out['survival'].cpu().numpy())
                n = sum(len(x) for x in inputs['observed_fine'].values())
                regions.append(len(inputs['observed_fine'])); patches.append(n)
                fractions.append(n/max(1, inputs['candidate_patch_count']))
                if (i+1)%16 == 0 or i+1 == len(dataset):
                    progress(f'验证：观察模式 {mode}', i+1, len(dataset))
            metrics = common_censor_metrics(times, events, curves, edges)
            scores[mode] = dict(metrics, cases=len(times), nll=float(np.mean(losses)),
                                mean_observed_regions=float(np.mean(regions)), mean_observed_patches=float(np.mean(patches)),
                                mean_patch_fraction=float(np.mean(fractions)))
            last = metrics['by_day'][str(int(edges[-1]))]
            progress(f'验证模式 {mode} 完成', detail=f"NLL={scores[mode]['nll']:.5f}，C-index={metric_text(metrics['c_index'])}，五年AUROC={metric_text(last['auroc'])}，五年Brier={last['brier']:.5f}")
    return scores


def fingerprint(args, cfg, train, validation):
    if {r['canonical_case_key'] for r in train.rows} & {r['canonical_case_key'] for r in validation.rows}:
        raise ValueError('Train/Validation case overlap')
    assets, counts, maximum = {}, {}, (-1, -1)
    for name, dataset in [('train', train), ('validate', validation)]:
        progress(f'启动前资产核对：{name}', 0, len(dataset))
        n_regions, n_patches = 0, 0
        for i, row in enumerate(dataset.rows):
            case = dataset.case(i)
            if not np.any(case.counts > 0):
                raise ValueError('No eligible fine for registered case')
            fine_path = Path(case.receipt['fine_path'])
            group = zarr.open_group(str(fine_path), mode='r')
            coords = np.asarray(group['coords'][:])
            if hashlib.sha256(coords.tobytes()).hexdigest() != case.receipt['fine_coords_sha256']:
                raise ValueError('Fine coordinates changed since coarse registration')
            if group['features'].shape != (len(coords), 1024) or str(group['features'].dtype) != 'float32':
                raise ValueError('Official fine array shape/dtype mismatch')
            assets[row['canonical_case_key']] = dict(coarse=case.receipt['regions_sha256'],
                                                     fine_coords=case.receipt['fine_coords_sha256'],
                                                     fine_array_tree=tree_sha256(fine_path/'features'))
            n_regions += len(case.coarse); n_patches += int(case.counts.sum())
            if name == 'train' and len(case.coarse) > maximum[0]:
                maximum = (len(case.coarse), i)
            if (i+1)%16 == 0 or i+1 == len(dataset):
                progress(f'启动前资产核对：{name}', i+1, len(dataset))
        counts[name] = dict(cases=len(dataset), regions=n_regions, candidate_fine_patches=n_patches)
    filenames = ('survival.py', 'prepare_survival.py', 'survival_data.py', 'survival_network.py', 'train_survival.py',
                 'fusion_data.py', 'fusion_network.py', 'network.py', 'uni_encoder.py')
    source = Path(__file__).parent
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source.parent, text=True).strip()
    provenance = dict(schema='KT2028-survival-run-1', config=cfg, config_sha256=sha256_file(args.config),
                      contract_sha256=contract_hash(cfg['data_contract']),
                      train_manifest_sha256=sha256_file(args.train_manifest),
                      validation_manifest_sha256=sha256_file(args.validation_manifest),
                      assets_sha256=hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest(),
                      source_commit=commit, source_shas={n:sha256_file(source/n) for n in filenames},
                      data_summary=counts, validation_role='checkpoint selection;not independent Test',
                      test_used=False, binary_checkpoint_reused=False)
    return provenance, maximum[1]


def readiness_check(model, train, i, device, weights):
    """Real Train forward/backward + in-memory reload; no optimizer or model update."""
    model.eval()
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.zero_grad(set_to_none=True)
    states = {}
    for mode in ('zero', '0.2', 'full'):
        inputs, target = train.sample(i, mode, epoch=0)
        with torch.no_grad():
            out = model(**move(inputs, device), return_aux=True)
            finite = all(torch.isfinite(out[k]).all() for k in ('logits', 'hazards', 'survival'))
            monotone = bool(torch.all(out['survival'][1:] <= out['survival'][:-1]))
            if not finite or not monotone or out['logits'].shape != (model.intervals,):
                raise ValueError('Readiness forward/curve check failed')
        states[mode] = dict(observed_regions=len(inputs['observed_fine']),
                            observed_fine_patches=sum(len(v) for v in inputs['observed_fine'].values()),
                            interval_output=list(out['logits'].shape), finite=True, survival_nonincreasing=True)
    loss = objective(model, train, i, 0, device, weights)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite registered objective')
    loss.backward()
    groups = {}
    required = ('coarse_projection', 'region_position', 'coarse_encoder', 'coarse_pool', 'fine_projection',
                'fine_position', 'local_attention', 'fine_encoder', 'fine_pool', 'coarse_gate', 'fine_gate',
                'patient_fusion', 'survival_head', 'fine_only_query')
    for group in required:
        gradients = [p.grad for n,p in model.named_parameters() if n.split('.')[0] == group and p.grad is not None]
        finite = bool(gradients) and all(torch.isfinite(g).all() for g in gradients)
        nonzero = any(torch.any(g != 0) for g in gradients)
        if not finite or not nonzero:
            raise ValueError(f'Required gradient path disconnected: {group}')
        groups[group] = dict(finite=True, nonzero=True)
    if any(not torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()):
        raise ValueError('Readiness check changed model state')
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    clone = SurvivalPredictor(dim=model.dim, heads=model.local_attention.num_heads,
                              dropout=model.local_attention.dropout, intervals=model.intervals, fusion=model.fusion_mode).to(device).eval()
    clone.load_state_dict(torch.load(stream, map_location=device, weights_only=True), strict=True)
    inputs, _ = train.sample(i, 'full', epoch=0)
    with torch.no_grad():
        a = model(**move(inputs, device))['logits']
        b = clone(**move(inputs, device))['logits']
    if not torch.allclose(a, b, atol=1e-6, rtol=1e-6):
        raise ValueError('Strict reload output differs')
    model.zero_grad(set_to_none=True)
    return dict(states=states, gradient_paths=groups, model_state_unchanged=True, strict_in_memory_reload=True,
                optimizer_steps=0, epochs=0, checkpoint_written=False, validation_predictions=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', required=True, type=Path)
    p.add_argument('--validation-manifest', required=True, type=Path)
    p.add_argument('--regions', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--preflight-report', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--execute', action='store_true', help='Explicitly authorize fitting; omission NEVER trains')
    args = p.parse_args()
    cfg = json.loads(args.config.read_text()); validate_config(cfg)
    progress('读取已登记的413例生存任务', detail='拟合开关已开启' if args.execute else '仅就绪检查，不训练')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    train = SurvivalDataset(args.train_manifest, args.regions, contract=cfg['data_contract'], seed=cfg['seed'])
    validation = SurvivalDataset(args.validation_manifest, args.regions, contract=cfg['data_contract'], purpose='model_selection', seed=cfg['seed'])
    provenance, i = fingerprint(args, cfg, train, validation)
    model = SurvivalPredictor(**cfg['architecture']).to(args.device)
    if not args.execute:
        progress('真实Train就绪检查', detail='零/部分/full前向与无参数更新的梯度检查')
        if str(args.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        checks = readiness_check(model, train, i, args.device, cfg['loss_weights'])
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize()
            checks['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        receipt = dict(provenance, status='ready_for_supervised_predictor_training;NOT_STARTED', checks=checks,
                       runtime=dict(torch=torch.__version__, device=args.device), training_run_created=False)
        if args.preflight_report:
            write_new_json(args.preflight_report, receipt)
        print(json.dumps(receipt, indent=2))
        return
    # The first optimizer, output run and epoch loop occur only AFTER the guard.
    if args.output.exists():
        raise FileExistsError('Preserve prior runs; require a new output directory')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).parent.parent, text=True).strip():
        raise ValueError('Commit the source/config before an actual training run')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    args.output.mkdir(parents=True)
    write_new_json(args.output/'run.json', provenance)
    progress('开始预测器监督训练', detail=f"Train={len(train)}，Val={len(validation)}，最多{cfg['max_epochs']}轮，早停耐心{cfg['patience']}；不训练RL")
    progress('指标说明', detail='训练总损失/NLL/Brier越低越好；C-index/AUROC越高越好；Validation用于选择checkpoint')
    best, stale, completed, best_epoch = float('inf'), 0, 0, -1
    for epoch in range(cfg['max_epochs']):
        epoch_start = time.perf_counter()
        progress(f"第{epoch+1}/{cfg['max_epochs']}轮：训练", 0, len(train))
        model.train()
        order = np.random.default_rng(cfg['seed']+epoch).permutation(len(train))
        total = 0.
        optimizer.zero_grad(set_to_none=True)
        accumulate = cfg['accumulate_cases']
        for j,i in enumerate(order):
            loss = objective(model, train, int(i), epoch, args.device, cfg['loss_weights'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite survival objective')
            group_size = min(accumulate, len(order)-(j//accumulate)*accumulate)
            (loss/group_size).backward()
            total += float(loss.detach())
            if (j+1)%accumulate == 0 or j+1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                progress(f"第{epoch+1}/{cfg['max_epochs']}轮：训练", j+1, len(order),
                         f'累计训练总损失={total/(j+1):.5f}，用时{time.perf_counter()-epoch_start:.0f}秒')
        scores = evaluate(model, validation, cfg['validation_modes'], args.device, cfg['data_contract']['interval_edges_days'])
        selection = float(np.mean([scores[m]['nll'] for m in cfg['validation_modes']]))
        event = dict(epoch=epoch, train_objective=total/len(train), selection_nll=selection, validation=scores,
                     all_task_parameters_trainable=True)
        with (args.output/'epochs.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        progress(f'第{epoch+1}轮汇总', detail=f'训练总损失={total/len(train):.5f}，验证五模式平均NLL={selection:.5f}')
        completed = epoch+1
        if selection < best:
            best, stale = selection, 0
            best_epoch = epoch
            temp = args.output/'best.partial.pt'
            torch.save(dict(provenance, checkpoint_schema='KT2028-survival-checkpoint-1', model=model.state_dict(),
                            epoch=epoch, selection_nll=selection, validation=scores), temp)
            temp.replace(args.output/'best.pt')
        else:
            stale += 1
        progress('checkpoint状态', detail=f"最佳第{best_epoch+1}轮，NLL={best:.5f}；未改善{stale}/{cfg['patience']}轮")
        if stale >= cfg['patience']:
            break
    saved = torch.load(args.output/'best.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(saved['model'], strict=True)
    write_new_json(args.output/'completion.json', dict(completed_epochs=completed, selected_epoch=saved['epoch'],
                   checkpoint_sha256=sha256_file(args.output/'best.pt'), best_selection_nll=best,
                   strict_reload=True, test_used=False))
    progress('预测器训练结束', detail=f"实际{completed}轮，选第{saved['epoch']+1}轮，最佳验证NLL={best:.5f}；输出={args.output}")


if __name__ == '__main__':
    main()
