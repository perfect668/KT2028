"""W160: five fixed CRC outer folds; independent early stopping and receipts."""
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

from .censored_metrics import fit_censoring
from .observed_cache import CachedDataset, DatasetSubset
from .run_component_ablation import atomic_save, evaluate, seed
from .run_controlled_survival import inner_folds
from .run_multicancer import split_rows, write
from .run_refinement import model_for, training_loss
from .uni_encoder import sha256_file

SPEC = dict(identity_coarse_gate=True, fine_attention_scale=None,
            normalize_fine_kv=False, exposure_likelihood=True)
SHARDS = {0: [0, 3], 1: [1, 4], 2: [2]}


def emit(folder, **value):
    value['updated_unix'] = time.time()
    write(folder / 'status.json', value)
    print(json.dumps(value, allow_nan=False), flush=True)


def make_splits(rows, seed_value=41):
    """Keep the original outer partitions and use one larger internal split."""
    assert len({r['canonical_case_key'] for r in rows}) == len(rows)
    result = []
    for fold in range(5):
        tr, va, te = split_rows(rows, fold, seed_value)
        development = [r for r in tr + va if r['cancer'] == 'CRC']
        outer = [r for r in te if r['cancer'] == 'CRC']
        ti, vi = inner_folds(development, seed_value)[0]
        training, valid = [development[i] for i in ti], [development[i] for i in vi]
        ids = [{r['canonical_case_key'] for r in x} for x in (training, valid, outer)]
        assert all(not ids[i] & ids[j] for i in range(3) for j in range(i))
        assert set.union(*ids) == {r['canonical_case_key'] for r in rows if r['cancer'] == 'CRC'}
        result.append(dict(train=training, validation=valid, outer=outer))
    held = [r['canonical_case_key'] for s in result for r in s['outer']]
    assert len(held) == len(set(held)) == sum(r['cancer'] == 'CRC' for r in rows)
    return result


def verify(root):
    provenance = json.loads((root / 'provenance.json').read_text())
    for name, digest in provenance['files'].items():
        assert sha256_file(root / name) == digest, name
    return sha256_file(root / 'provenance.json')


def plot_learning(folder, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, key, title in zip(axes, ['nll', 'c_index', 'integrated_brier_annual_grid'],
                              ['Annual NLL (lower)', 'C-index (higher)', 'IPCW IBS (lower)']):
        ax.plot([h['epoch'] for h in history], [h['monitor_full'][key] for h in history], label='Fixed Train subset')
        ax.plot([h['epoch'] for h in history], [h['validation']['full'][key] for h in history], label='Inner validation')
        ax.set(title=title, xlabel='Epoch'); ax.legend(fontsize=8)
    fig.suptitle(folder.name + ' | R3 identity gate + exposure likelihood')
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(folder / ('learning_curves.' + ext), dpi=160)
    plt.close(fig)


def fit(root, fold, cfg, signature):
    folder = root / 'folds' / str(fold)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completion.json').exists():
        done = json.loads((folder / 'completion.json').read_text())
        assert done['provenance_sha256'] == signature
        assert done['best_sha256'] == sha256_file(folder / 'best.pt')
        return
    local_cfg = copy.deepcopy(cfg); local_cfg['seed'] = cfg['seed'] + fold
    split = json.loads((root / 'splits_private.json').read_text())[fold]
    data = CachedDataset(split['train'], root, local_cfg, cfg['train_cache_gib'])
    valid = CachedDataset(split['validation'], root, local_cfg, cfg['validation_cache_gib'])
    indices = sorted(range(len(data)), key=lambda i: hashlib.sha256(data.rows[i]['canonical_case_key'].encode()).hexdigest())[:64]
    monitor = DatasetSubset(data, indices)
    model = model_for(SPEC, local_cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history = []; best = float('inf'); best_epoch = 0; steps = 0; start = 0
    seed(local_cfg['seed'] + 151000)
    if (folder / 'last.pt').exists():
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['provenance_sha256'] == signature and saved['spec'] == SPEC
        model.load_state_dict(saved['model'], strict=True); opt.load_state_dict(saved['optimizer'])
        history, best, best_epoch, steps, start = [saved[k] for k in ['history', 'best', 'best_epoch', 'steps', 'epoch']]
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    censoring = fit_censoring(data.rows)
    write(folder / 'metadata.json', dict(spec=SPEC, fold=fold, seed=local_cfg['seed'],
          parameters=sum(p.numel() for p in model.parameters()),
          sizes={k: len(v) for k, v in split.items()}, max_epochs=cfg['max_epochs'], patience=cfg['patience']))
    for epoch in range(start, cfg['max_epochs']):
        began = time.time(); model.train(); opt.zero_grad(set_to_none=True); total = 0.
        order = np.random.default_rng(local_cfg['seed'] + epoch).permutation(len(data))
        for j, i in enumerate(order):
            loss = training_loss(model, data, int(i), epoch, local_cfg, SPEC)
            assert torch.isfinite(loss)
            group = min(cfg['accumulate_cases'], len(order) - (j // cfg['accumulate_cases']) * cfg['accumulate_cases'])
            (loss / group).backward(); total += float(loss.detach())
            if (j + 1) % cfg['accumulate_cases'] == 0 or j + 1 == len(order):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                opt.step(); opt.zero_grad(set_to_none=True); steps += 1
                emit(folder, state='training', fold=fold, epoch=epoch + 1, patients_done=j + 1,
                     patients_total=len(data), optimizer_steps=steps, objective=total / (j + 1), gradient_norm=float(norm))
        scores, _, _ = evaluate(model, valid, cfg, censoring, ['full'], folder, 'inner_validation')
        monitored, _, _ = evaluate(model, monitor, cfg, censoring, ['full'], folder, 'train_monitor')
        selection = scores['full']['nll']
        history.append(dict(epoch=epoch + 1, train_objective=total / len(data), selection_nll=selection,
                            validation=scores, monitor_full=monitored['full'], seconds=time.time() - began))
        if selection < best:
            best, best_epoch = selection, epoch + 1
            atomic_save(folder / 'best.pt', dict(model=model.state_dict(), epoch=best_epoch,
                        spec=SPEC, provenance_sha256=signature))
        atomic_save(folder / 'last.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(), history=history,
                    best=best, best_epoch=best_epoch, epoch=epoch + 1, steps=steps, spec=SPEC,
                    provenance_sha256=signature, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        write(folder / 'history.json', history); plot_learning(folder, history)
        emit(folder, state='epoch_completed', fold=fold, epoch=epoch + 1, optimizer_steps=steps,
             full_nll=selection, full_c_index=scores['full']['c_index'], best_epoch=best_epoch)
        if epoch + 1 - best_epoch >= cfg['patience']:
            break
    checkpoint = torch.load(folder / 'best.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    # Outer outcomes are first used here, after all fitting and selection have ended.
    outer = CachedDataset(split['outer'], root, local_cfg, cfg['validation_cache_gib'])
    scores, predictions, usage = evaluate(model, outer, cfg, censoring, ['zero', '0.2', 'full'], folder, 'fixed_best_outer')
    write(folder / 'outer_metrics.json', scores)
    write(folder / 'predictions_private.json', dict(outer=predictions, usage=usage))
    write(folder / 'completion.json', dict(state='completed', fold=fold, epochs=len(history), best_epoch=best_epoch,
          optimizer_steps=steps, best_sha256=sha256_file(folder / 'best.pt'), provenance_sha256=signature,
          spec=SPEC, strict_reload=True, outer_patients=len(outer), at=time.time()))
    emit(folder, state='completed', fold=fold, epochs=len(history), optimizer_steps=steps, best_epoch=best_epoch)
    del model, opt, data, valid, monitor, outer
    import gc
    gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True, type=Path)
    p.add_argument('--worker', required=True, type=int, choices=range(3)); p.add_argument('--execute', action='store_true')
    args = p.parse_args(); root = args.root
    signature = verify(root); cfg = json.loads((root / 'config.json').read_text())
    assert cfg['plan'] == '0.40/W160' and cfg['spec'] == SPEC and cfg['max_epochs'] == 30 and cfg['patience'] == 8
    torch.set_num_threads(cfg['cpu_threads']); torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    assert args.execute, 'Run the separately registered preflight before --execute'
    proof = json.loads((root / 'preflight_completed.json').read_text())
    assert proof['provenance_sha256'] == signature and proof['finite_backward'] and proof['strict_reload']
    worker = root / ('worker_' + str(args.worker)); worker.mkdir(exist_ok=True)
    emit(worker, state='started', worker=args.worker, folds=SHARDS[args.worker])
    try:
        for fold in SHARDS[args.worker]:
            fit(root, fold, cfg, signature)
        write(worker / 'completion.json', dict(state='completed', at=time.time(), folds=SHARDS[args.worker]))
        emit(worker, state='completed', worker=args.worker)
    except Exception:
        error = traceback.format_exc(); write(worker / 'failure.json', dict(traceback=error, at=time.time()))
        emit(worker, state='failed', error=error); raise


if __name__ == '__main__':
    main()
