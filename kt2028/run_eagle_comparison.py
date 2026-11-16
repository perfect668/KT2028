"""W145 registered four-cancer fold-0, official-source EAGLE survival adaptation."""
import argparse
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .eagle_source_adapter import SourceEAGLE, grid_coordinates, pack_grids, pool_canvas, survival_objective
from .run_multicancer import split_rows, write
from .mixed_survival_data import inspect_asset
from .survival import discrete_nll
from .censored_metrics import fit_censoring, survival_metrics
from .run_dsca_comparison import constant_reference, plot
from .uni_encoder import sha256_file


def emit(root, **value):
    value['at'] = time.time()
    write(root/'status.json', value)
    print(json.dumps(value), flush=True)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class FineDataset:
    def __init__(self, rows, root, cfg):
        self.rows, self.root, self.cfg, self.cache = rows, root, cfg, {}

    def __len__(self):
        return len(self.rows)

    def sample(self, index):
        import zarr
        if index not in self.cache:
            parts = []
            count = 0
            for slide in self.rows[index]['slides']:
                folder = self.root/'features'/slide['asset_id']
                receipt = inspect_asset(folder, slide, self.cfg['data_contract']['feature_protocol'])
                with np.load(folder/'geometry.npz', allow_pickle=False) as geometry:
                    xy = grid_coordinates(geometry['fine_bbox_um'])
                store = zarr.open_group(str(folder/'fine.zarr'), mode='r')
                values = np.asarray(store['features'][:], dtype=np.float32)
                if not np.isfinite(values).all() or len(values) != receipt['fine']['rows']:
                    raise ValueError('Invalid/uncommitted frozen feature store')
                parts.append((xy, values))
                count += len(values)
            cube, mask = pack_grids(parts)
            original_shape = list(mask.shape)
            cube, mask, levels = pool_canvas(cube, mask, self.cfg['maximum_canvas_cells'])
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
            self.cache[index] = (cube, mask, dict(original_patches=count, original_shape=original_shape,
                                                canvas_shape=list(mask.shape), pooling_levels=levels,
                                                occupied_cells=int(mask.sum())))
        cube, mask, meta = self.cache[index]
        return cube, mask, dict(self.rows[index]['survival_target']), meta


def evaluate(model, data, cfg, censoring, device):
    model.eval()
    curves, losses = [], []
    with torch.no_grad():
        for i in range(len(data)):
            cube, mask, target, _ = data.sample(i)
            output = model(cube.to(device), mask.to(device))
            curves.append(output['survival'].cpu().numpy())
            losses.append(float(discrete_nll(output['logits'], target)))
    values = np.stack(curves)
    metrics = survival_metrics(data.rows, values, cfg['data_contract']['interval_edges_days'], censoring)
    metrics['nll'] = float(np.mean(losses))
    return metrics, values.tolist()


def overview(root, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    arms = [('R3', root/'baseline', '', '#7c98a5'),
            ('DSCA', root/'dsca_baseline', '_dsca_official_recipe', '#b58c63'),
            ('EAGLE adapted', root, '_eagle_survival', '#4c8c6b')]
    for index, cancer in enumerate(cfg['cancers']):
        for offset, (label, base, suffix, color) in enumerate(arms):
            path = base/'runs'/('fold0_'+cancer+suffix)/'outer_metrics.json'
            if not path.exists():
                continue
            metric = json.loads(path.read_text())
            if label == 'R3':
                metric = metric['full']['by_cancer'][cancer]
            results.append(dict(cancer=cancer, arm=label, metrics=metric))
            for ax, key in zip(axes, ['c_index', 'nll', 'integrated_brier_annual_grid']):
                if metric.get(key) is not None:
                    ax.bar(index+(offset-1)*.25, metric[key], width=.24, color=color,
                           label=label if index == 0 else None)
    for ax, title in zip(axes, ['C-index (higher)', 'NLL (lower)', 'IBS (lower)']):
        ax.set(title=title, xticks=range(len(cfg['cancers'])), xticklabels=cfg['cancers'])
    axes[0].set_ylim(0, 1)
    if results:
        axes[0].legend(fontsize=7)
    fig.suptitle('Same outer fold 0 and clinical target; missing EAGLE bars = not complete')
    for ext in ['png', 'svg']:
        fig.savefig(root/('comparison.'+ext), dpi=170)
    plt.close(fig)
    write(root/'comparison.json', results)


def attention_plot(model, data, index, folder, device):
    import matplotlib.pyplot as plt
    cube, mask, _, meta = data.sample(index)
    model.eval()
    with torch.no_grad():
        out = model(cube.to(device), mask.to(device))
    attention = out['attention'].float().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].imshow(mask, cmap='gray', interpolation='nearest')
    axes[0].set_title('Available feature cells / packed canvas')
    axes[1].imshow(np.log10(attention+1e-10), cmap='magma', interpolation='nearest')
    axes[1].set_title('Log attention / fixed inner-validation example')
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle('Model attention; not a pathological annotation or tissue diagnosis')
    for ext in ['png', 'svg']:
        fig.savefig(folder/('attention_example.'+ext), dpi=170)
    plt.close(fig)
    write(folder/'attention_metadata.json', dict(**meta, background_attention=float(out['background_attention']),
          case_selection='lowest SHA256 canonical key in inner validation; no outcome selection'))


def fit(root, cfg, cancer, partitions):
    folder = root/'runs'/('fold0_'+cancer+'_eagle_survival')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completion.json').exists():
        previous = json.loads((folder/'completion.json').read_text())
        if sha256_file(folder/'best.pt') != previous['checkpoint_sha256']:
            raise ValueError('Completed checkpoint changed')
        emit(root, state='skipping_completed', cancer=cancer)
        return
    train, val, outer = ([r for r in part if r['cancer'] == cancer] for part in partitions)
    write(folder/'split_private.json', dict(train=train, inner_validation=val, outer=outer))
    data, valid = FineDataset(train, root, cfg), FineDataset(val, root, cfg)
    censoring = fit_censoring(train)
    reference = constant_reference(train, val, cfg, censoring)
    write(folder/'constant_reference.json', reference)
    seed(cfg['seed'])
    device = 'cuda'
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    model = SourceEAGLE(root/'official', dim=cfg['embedding_dim'], topk=cfg['topk'], tile=cfg['tile']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['peak_learning_rate'], weight_decay=cfg['weight_decay'])
    maximum, warmup = cfg['max_epochs'], cfg['warmup_epochs']
    def lr(epoch):
        if epoch < warmup:
            return (epoch+1)/warmup
        return .5*(1+np.cos(np.pi*(epoch-warmup)/(maximum-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr)
    best, best_epoch, history, first_epoch = float('inf'), 0, [], 0
    if (folder/'last.pt').exists():
        state = torch.load(folder/'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best, best_epoch, history, first_epoch = state['best'], state['best_epoch'], state['history'], state['epoch']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        emit(root, state='resumed', cancer=cancer, completed_epochs=first_epoch)
    else:
        # Resource check uses the actual largest dense training canvas, not a small toy.
        sizes = []
        for row in train:
            area = 0
            for slide in row['slides']:
                with np.load(root/'features'/slide['asset_id']/'geometry.npz') as g:
                    xy = grid_coordinates(g['fine_bbox_um'])
                    area += int(np.prod(xy.max(0)+1))
            sizes.append(area)
        index = int(np.argmax(sizes))
        cube, mask, target, meta = data.sample(index)
        state_before = {n: p.detach().clone() for n, p in model.named_parameters()}
        started = time.time()
        with torch.random.fork_rng(devices=[0]):
            model.train()
            torch.cuda.reset_peak_memory_stats()
            out = model(cube.to(device), mask.to(device), auxiliary=True)
            loss, _ = survival_objective(out, target, cfg['neighborhood_weight'], cfg['background_weight'])
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite largest-case preflight')
            opt.zero_grad(set_to_none=True)
        assert all(torch.equal(state_before[n], p) for n, p in model.named_parameters())
        write(folder/'preflight.json', dict(**meta, training_mode=True, unchanged_parameters=True,
              optimizer_steps=0, largest_train_dense_area=True, finite_gradient=True,
              seconds=time.time()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        del state_before, out, loss, cube, mask
        seed(cfg['seed'])
    emit(root, state='starting_fit', cancer=cancer, train=len(train), inner_validation=len(val),
         parameters=sum(p.numel() for p in model.parameters()), completed_epochs=first_epoch)
    for epoch in range(first_epoch, maximum):
        model.train()
        opt.zero_grad(set_to_none=True)
        totals = dict(bag_nll=0., neighborhood=0., background=0., total=0.)
        started = time.time()
        order = np.random.default_rng(cfg['seed']+epoch).permutation(len(train))
        for j, i in enumerate(order):
            cube, mask, target, _ = data.sample(int(i))
            out = model(cube.to(device), mask.to(device), auxiliary=True)
            loss, terms = survival_objective(out, target, cfg['neighborhood_weight'], cfg['background_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite survival objective')
            group = min(cfg['accumulate'], len(order)-(j//cfg['accumulate'])*cfg['accumulate'])
            (loss/group).backward()
            totals['total'] += float(loss.detach())
            for key, value in terms.items():
                totals[key] += float(value.detach())
            if (j+1) % cfg['accumulate'] == 0 or j+1 == len(order):
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient')
                opt.step()
                opt.zero_grad(set_to_none=True)
                emit(root, state='training', cancer=cancer, epoch=epoch+1, patients=j+1, total=len(order),
                     train_objective=totals['total']/(j+1), bag_nll=totals['bag_nll']/(j+1),
                     elapsed_seconds=time.time()-started)
            del out, loss, cube, mask, terms
        emit(root, state='inner_validation', cancer=cancer, epoch=epoch+1)
        metric, _ = evaluate(model, valid, cfg, censoring, device)
        record = dict(epoch=epoch+1, train_objective=totals['total']/len(train),
                      train_terms={k: v/len(train) for k, v in totals.items()}, validation=metric,
                      selection_nll=metric['nll'], learning_rate=opt.param_groups[0]['lr'], seconds=time.time()-started)
        history.append(record)
        if metric['nll'] < best:
            best, best_epoch = metric['nll'], epoch+1
            torch.save(dict(model=model.state_dict(), epoch=best_epoch), folder/'best.partial.pt')
            (folder/'best.partial.pt').replace(folder/'best.pt')
        scheduler.step()
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                     best=best, best_epoch=best_epoch, history=history, epoch=epoch+1,
                     python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        torch.save(state, folder/'last.partial.pt')
        (folder/'last.partial.pt').replace(folder/'last.pt')
        write(folder/'history.json', history)
        plot(folder, history, reference)
        emit(root, state='epoch_complete', cancer=cancer, epoch=epoch+1, best_epoch=best_epoch,
             inner_nll=metric['nll'], inner_c=metric['c_index'], seconds=record['seconds'])
    model.load_state_dict(torch.load(folder/'best.pt', map_location=device, weights_only=True)['model'], strict=True)
    inner, predictions = evaluate(model, valid, cfg, censoring, device)
    write(folder/'inner_metrics.json', inner)
    write(folder/'inner_predictions_private.json', dict(rows=val, predictions=predictions))
    examples = sorted(range(len(val)), key=lambda i: hashlib.sha256(val[i]['canonical_case_key'].encode()).hexdigest())[:6]
    plot(folder, history, reference, 'inner validation', [predictions[i] for i in examples])
    attention_plot(model, valid, examples[0], folder, device)
    emit(root, state='fixed_best_outer_evaluation', cancer=cancer, best_epoch=best_epoch)
    outer_metrics, predictions = evaluate(model, FineDataset(outer, root, cfg), cfg, censoring, device)
    write(folder/'outer_metrics.json', outer_metrics)
    write(folder/'outer_predictions_private.json', dict(rows=outer, predictions=predictions))
    write(folder/'completion.json', dict(cancer=cancer, epochs=len(history), best_epoch=best_epoch,
          checkpoint_sha256=sha256_file(folder/'best.pt'), parameters=sum(p.numel() for p in model.parameters()),
          inner_metrics=inner, outer_metrics=outer_metrics, outer_used_for_selection=False))
    overview(root, cfg)
    del model, opt, state, data, valid
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = args.root
    if not args.execute:
        raise RuntimeError('Registered experiment requires explicit --execute')
    cfg = json.loads((root/'config.json').read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    official = json.loads((root/'official_manifest.json').read_text())
    for name, expected in official['files'].items():
        if sha256_file(root/'official'/name) != expected:
            raise ValueError('Official file changed: '+name)
    sources = {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    provenance = dict(config_sha256=sha256_file(root/'config.json'), patients_sha256=sha256_file(root/'patients_private.json'),
                      official_manifest_sha256=sha256_file(root/'official_manifest.json'), source_sha256=sources)
    if (root/'provenance.json').exists():
        assert json.loads((root/'provenance.json').read_text()) == provenance
    else:
        write(root/'provenance.json', provenance)
    rows = json.loads((root/'patients_private.json').read_text())
    partitions = split_rows(rows, 0, cfg['seed'])
    overview(root, cfg)
    for cancer in cfg['cancers']:
        fit(root, cfg, cancer, partitions)
    emit(root, state='completed', runs=len(cfg['cancers']))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        import sys
        if '--root' in sys.argv:
            emit(Path(sys.argv[sys.argv.index('--root')+1]), state='failed', error=traceback.format_exc())
        raise
