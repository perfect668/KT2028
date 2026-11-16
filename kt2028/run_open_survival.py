"""W147: six native-Cox fits on existing CRC/LUAD fold0 Train/inner only."""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .open_survival_adapter import make_model, make_panther, cox_breslow, labels, baseline_curves, risk_metrics
from .run_multicancer import split_rows, write
from .mixed_survival_data import inspect_asset
from .uni_encoder import sha256_file
from .censored_metrics import fit_censoring, survival_metrics


def emit(folder, **value):
    value['at'] = time.time()
    write(folder/'status.json', value)
    print(json.dumps(value), flush=True)


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class Bags:
    def __init__(self, rows, root, cfg):
        import zarr
        self.rows, self.root, self.cfg = rows, root, cfg
        self.cache = collections.OrderedDict(); self.bytes = 0
        self.arrays, self.counts = [], []
        for row in rows:
            arrays = []
            for slide in row['slides']:
                folder = root/'features'/slide['asset_id']
                receipt = inspect_asset(folder, slide, cfg['data_contract']['feature_protocol'])
                a = zarr.open_group(str(folder/'fine.zarr'), mode='r')['features']
                assert a.shape == (receipt['fine']['rows'], 1024)
                arrays.append(a)
            self.arrays.append(arrays); self.counts.append(sum(a.shape[0] for a in arrays))

    def get(self, index, count=None, rng=None):
        if index in self.cache:
            values = self.cache.pop(index); self.cache[index] = values
        else:
            values = np.concatenate([np.asarray(a[:], dtype=np.float32) for a in self.arrays[index]])
            if not np.isfinite(values).all() or not len(values):
                raise ValueError('Nonfinite/empty frozen feature bag')
            while self.cache and self.bytes+values.nbytes > self.cfg['ram_cache_gib']*1024**3:
                _, old = self.cache.popitem(last=False); self.bytes -= old.nbytes
            if values.nbytes <= self.cfg['ram_cache_gib']*1024**3:
                self.cache[index] = values; self.bytes += values.nbytes
        if count is not None and len(values) > count:
            return torch.from_numpy(values[rng.choice(len(values), count, replace=False)].copy())
        return torch.from_numpy(values)


def predict_one(model, method, x, device, step, steps, gradient=False):
    x = x.to(device)
    if method == 'PANTHER':
        return model.forward_no_loss(x.unsqueeze(0))['logits'].reshape(())
    if method == 'nnMIL':
        fn = lambda y: model(y.unsqueeze(0), is_cox=True)['logits'].reshape(())
    else:
        fn = lambda y: model.forward_no_loss([y, steps, step])['logits'].reshape(())
    if gradient:
        # Recompute each patient while retaining a genuine multi-patient Cox risk set.
        # CPU saved tensors preserve the full differentiable OT graph without truncation.
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return checkpoint(fn, x, use_reentrant=False, preserve_rng_state=True)
    return fn(x)


def evaluate(model, method, bags, indices, representations, device, step, steps, folder, phase):
    model.eval(); risk = []
    with torch.no_grad():
        for n, i in enumerate(indices):
            x = representations[i] if representations is not None else bags.get(i)
            r = predict_one(model, method, x, device, step, steps)
            if not torch.isfinite(r):
                raise ValueError('Nonfinite survival risk')
            risk.append(float(r))
            if (n+1)%50 == 0:
                emit(folder, state=phase, patients=n+1, total=len(indices), step=step)
    return np.asarray(risk)


def panther_representations(bags, n_train, official, folder, device):
    from sklearn.cluster import KMeans
    prototype = folder/'prototypes.npy'
    # Equal patient allocation, only Train features. Same 1.6m upper budget as source recipe.
    if not prototype.exists():
        count = math.ceil(1600000/n_train); rng = np.random.default_rng(41)
        sample = []
        for i in range(n_train):
            sample.append(bags.get(i, count, rng).numpy().copy())
            if (i+1)%50 == 0:
                emit(folder, state='train_only_prototype_sampling', patients=i+1, total=n_train)
        patches = np.concatenate(sample)[:1600000]
        emit(folder, state='train_only_kmeans', patches=len(patches), clusters=16, n_init=5)
        km = KMeans(n_clusters=16, n_init=5, max_iter=100, random_state=41).fit(patches)
        np.save(prototype, km.cluster_centers_.astype(np.float32))
        write(folder/'prototype_provenance.json', dict(train_patients=n_train, sampled_patches=len(patches),
              fit_partition='train_only', n_init=5, max_iter=100, seed=41, sha256=sha256_file(prototype)))
        del sample, patches, km
    encoder = make_panther(official, prototype).to(device).eval()
    reps = []
    with torch.no_grad():
        for i in range(len(bags.rows)):
            x = bags.get(i).to(device)
            rep, _ = encoder(x.unsqueeze(0))
            if not torch.isfinite(rep).all():
                raise ValueError('Nonfinite GMM representation')
            reps.append(rep.squeeze(0).cpu())
            if (i+1)%25 == 0:
                emit(folder, state='gmm_representation', patients=i+1, total=len(bags.rows))
    result = torch.stack(reps)
    torch.save(result, folder/'representations_private.pt')
    return result


def plot(folder, method, cancer, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ep = [h['epoch'] for h in history]
    ax[0].plot(ep, [h['batch_cox'] for h in history], label='Train batch Cox')
    ax[0].plot(ep, [h['inner']['cox_partial_nll_per_event'] for h in history], label='Inner full-set Cox')
    ax[0].set(title='Different risk-set sizes: trends only', xlabel='Epoch', ylabel='Partial NLL/event')
    ax[0].legend(fontsize=7)
    ax[1].plot(ep, [h['inner']['c_index'] for h in history])
    ax[1].axhline(.5, ls='--', color='gray'); ax[1].set(ylim=(0,1), title='Inner C-index (higher)', xlabel='Epoch')
    ax[2].plot(ep, [h['lr'] for h in history]); ax[2].set(title='Learning rate', xlabel='Epoch')
    fig.suptitle(f'{method} / {cancer} / fixed fold0 inner validation only')
    for ext in ['png', 'svg']:
        fig.savefig(folder/('learning_curves.'+ext), dpi=170)
    plt.close(fig)


def run(root, method, cancer, execute):
    provenance = json.loads((root/'provenance.json').read_text())
    for name, expected in provenance['code'].items():
        assert sha256_file(root/name) == expected, 'Registered code changed: '+name
    assert sha256_file(root/'config.json') == provenance['config_sha256']
    assert sha256_file(root/'patients_private.json') == provenance['patients_sha256']
    for record in json.loads((root/'source_manifest.json').read_text()):
        assert sha256_file(root/'official'/record['repo'].split('/')[-1]/record['path']) == record['sha256']
    cfg = json.loads((root/'config.json').read_text())
    arm = cfg['methods'][method]
    folder = root/'runs'/f'fold0_{cancer}_{method}'; folder.mkdir(parents=True, exist_ok=True)
    completion = folder/'completion.json'
    if completion.exists():
        done = json.loads(completion.read_text())
        assert done['best_sha256'] == sha256_file(folder/'best.pt')
        emit(folder, state='already_completed_sha_verified'); return
    seed(cfg['seed']); torch.set_num_threads(cfg['cpu_threads'])
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    device = torch.device('cuda')
    rows = json.loads((root/'patients_private.json').read_text())
    train, inner, outer = split_rows(rows, 0, cfg['seed'])
    train = [r for r in train if r['cancer'] == cancer]; inner = [r for r in inner if r['cancer'] == cancer]
    # Deliberately do not construct any outer feature loader or predictions.
    del outer
    write(folder/'split_private.json', dict(train=train, inner_validation=inner))
    write(folder/'run_config.json', dict(method=method, cancer=cancer, **arm, config_sha256=sha256_file(root/'config.json')))
    data = Bags(train+inner, root, cfg); n_train = len(train)
    indices = list(range(n_train)); inner_indices = list(range(n_train, len(data.rows)))
    count = max(1, int(np.median(data.counts[:n_train])*.5))
    write(folder/'data_summary.json', dict(train_patients=n_train, inner_patients=len(inner),
          train_events=sum(r['survival_target']['event'] for r in train), inner_events=sum(r['survival_target']['event'] for r in inner),
          train_patches=sum(data.counts[:n_train]), inner_patches=sum(data.counts[n_train:]),
          nnmil_sampled_patch_limit=count, maximum_train_patches=max(data.counts[:n_train]), outer_predictions=0))
    reps = None
    if method == 'PANTHER':
        cache = folder/'representations_private.pt'
        reps = torch.load(cache, weights_only=True) if cache.exists() else panther_representations(data, n_train, root/'official', folder, device)
        assert reps.shape == (len(data.rows), 32784) and torch.isfinite(reps).all()
        # The first 16 allcat entries are each patient's mixture probabilities.
        # Aggregate fixed-size pi, never the variable-length patch assignment axis.
        pi = reps[:, :16]
        assert (pi >= 0).all() and torch.allclose(pi.sum(1), torch.ones(len(pi)), atol=1e-4)
        write(folder/'prototype_occupancy.json', dict(train_mean=pi[:n_train].mean(0).tolist(),
              inner_mean=pi[n_train:].mean(0).tolist(), definition='patient-mean GMM mixture probability',
              prototypes_sha256=sha256_file(folder/'prototypes.npy')))
        # Identical head initialization whether the fixed representation is cached or freshly computed.
        seed(cfg['seed'])
    model = make_model(method, root/'official').to(device)
    batch_size = arm['batch_size']; steps = math.ceil(n_train/batch_size)
    # Real largest Train input: finite gradient, no update, strict reload, RNG restored afterwards.
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    largest = int(np.argmax(data.counts[:n_train])); rng = np.random.default_rng(cfg['seed'])
    x = reps[largest] if reps is not None else data.get(largest, count if method == 'nnMIL' else None, rng)
    model.train(); torch.cuda.reset_peak_memory_stats()
    risk = predict_one(model, method, x, device, 0, steps, gradient=True)
    risk.square().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.parameters())
    assert all(torch.equal(v, state[k]) for k,v in model.state_dict().items())
    peak = torch.cuda.max_memory_allocated(); model.zero_grad(set_to_none=True)
    model.load_state_dict(state, strict=True); torch.set_rng_state(torch_rng); torch.cuda.set_rng_state_all(cuda_rng)
    write(folder/'preflight.json', dict(real_train_largest_patches=data.counts[largest],
          actual_patches=None if method == 'PANTHER' else len(x), representation_dimensions=len(x) if method == 'PANTHER' else None,
          finite_gradients=True, parameters_unchanged=True, optimizer_steps=0, peak_cuda_bytes=peak,
          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if not execute:
        emit(folder, state='preflight_only'); return
    if method == 'nnMIL':
        decay, no_decay = [], []
        for p in model.parameters():
            (no_decay if p.ndim < 2 else decay).append(p)
        optimizer = torch.optim.AdamW([{'params':decay, 'weight_decay':arm['weight_decay']},
                                      {'params':no_decay,'weight_decay':0}], lr=arm['lr'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=arm['lr'], weight_decay=arm['weight_decay'])
    history=[]; best=math.inf; best_epoch=0; start=0; global_step=0
    last = folder/'last.pt'
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)
        assert saved['config_sha256'] == sha256_file(root/'config.json')
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        history=saved['history']; best=saved['best']; best_epoch=saved['best_epoch']; start=saved['epoch']; global_step=saved['global_step']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    for epoch in range(start, arm['max_epochs']):
        began=time.time(); model.train(); order=np.random.permutation(n_train); losses=[]
        for b in range(steps):
            warm = arm['warmup_epochs']*steps; total=arm['max_epochs']*steps
            lr=arm['lr']*(min(1., (global_step+1)/max(1,warm)) if global_step<warm else
                         .5*(1+math.cos(math.pi*(global_step-warm)/max(1,total-warm))))
            for group in optimizer.param_groups: group['lr']=lr
            chosen=order[b*batch_size:(b+1)*batch_size]; optimizer.zero_grad(set_to_none=True)
            pred=[]
            for i in chosen:
                x = reps[i] if reps is not None else data.get(int(i), count if method=='nnMIL' else None, np.random)
                pred.append(predict_one(model, method, x, device, global_step, steps, gradient=True))
            t,e=labels([train[i] for i in chosen], device)
            loss=cox_breslow(torch.stack(pred),t,e)
            if not torch.isfinite(loss): raise ValueError('Nonfinite Cox loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError('Nonfinite training gradients')
            optimizer.step(); global_step+=1; losses.append(float(loss.detach()))
            emit(folder,state='training',method=method,cancer=cancer,epoch=epoch+1,batch=b+1,batches=steps,cox=float(loss.detach()))
        val_risk=evaluate(model,method,data,inner_indices,reps,device,global_step-1,steps,folder,'inner_validation')
        metrics=risk_metrics(val_risk,inner)
        h=dict(epoch=epoch+1,batch_cox=float(np.mean(losses)),inner=metrics,lr=lr,seconds=time.time()-began)
        history.append(h); val=metrics['cox_partial_nll_per_event']
        if val < best:
            best=val;best_epoch=epoch+1
            torch.save(dict(model=model.state_dict(),epoch=best_epoch,step=global_step-1,config_sha256=sha256_file(root/'config.json')),folder/'best.pt')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,global_step=global_step,
             history=history,best=best,best_epoch=best_epoch,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
             torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config_sha256=sha256_file(root/'config.json')),last)
        write(folder/'history.json',history); plot(folder,method,cancer,history)
        emit(folder,state='epoch_complete',method=method,cancer=cancer,**h,best_epoch=best_epoch)
        if arm['patience'] and epoch+1>=arm['minimum_epochs'] and epoch+1-best_epoch>=arm['patience']: break
    saved=torch.load(folder/'best.pt',map_location=device,weights_only=False); model.load_state_dict(saved['model'],strict=True)
    train_risk=evaluate(model,method,data,indices,reps,device,saved['step'],steps,folder,'fixed_best_train_baseline')
    val_risk=evaluate(model,method,data,inner_indices,reps,device,saved['step'],steps,folder,'fixed_best_inner')
    curves=baseline_curves(train_risk,train,val_risk,cfg['data_contract']['interval_edges_days'])
    metrics=risk_metrics(val_risk,inner)
    write(folder/'train_metrics.json', risk_metrics(train_risk,train))
    probability=survival_metrics(inner,curves,cfg['data_contract']['interval_edges_days'],fit_censoring(train))
    write(folder/'inner_metrics.json',dict(**metrics,annual_probability_metrics=probability,baseline_fit='train_only',best_epoch=best_epoch))
    write(folder/'predictions_private.json',dict(train_risk=train_risk.tolist(),inner_risk=val_risk.tolist(),inner_survival=curves.tolist()))
    write(completion,dict(state='completed',method=method,cancer=cancer,epochs=len(history),best_epoch=best_epoch,
          best_sha256=sha256_file(folder/'best.pt'),config_sha256=sha256_file(root/'config.json'),outer_predictions=0,at=time.time()))
    emit(folder,state='completed',method=method,cancer=cancer,best_epoch=best_epoch,inner=metrics)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--method',choices=['nnMIL','PANTHER','OTSurv'],required=True);p.add_argument('--cancer',choices=['CRC','LUAD'],required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    try:run(args.root,args.method,args.cancer,args.execute)
    except Exception:
        folder=args.root/'runs'/f'fold0_{args.cancer}_{args.method}';folder.mkdir(parents=True,exist_ok=True)
        write(folder/'failure.json',dict(state='failed',traceback=traceback.format_exc(),at=time.time()))
        raise


if __name__=='__main__':main()
