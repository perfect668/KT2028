"""R2 coarse anchor and true-fine fusion training; Test has no entry point."""
import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from .fusion_data import FusionDataset, move
from .fusion_network import CrossScalePredictor
from .prepare_coarse_uni import write_json
from .uni_encoder import sha256_file


def evaluate(model, dataset, modes, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for mode in modes:
            labels, probabilities, losses, observed_regions, observed_patches, patch_fractions = [], [], [], [], [], []
            for i in range(len(dataset)):
                inputs, target = dataset.sample(i, mode, epoch=0)
                out = model(**move(inputs, device))
                labels.append(float(target))
                probabilities.append(float(out['logit'].sigmoid()))
                losses.append(float(F.binary_cross_entropy_with_logits(out['logit'], target.to(device))))
                n_patches = sum(len(v) for v in inputs['observed_fine'].values())
                observed_regions.append(len(inputs['observed_fine']))
                observed_patches.append(n_patches)
                patch_fractions.append(n_patches/max(1, inputs['candidate_patch_count']))
            labels, probabilities = np.asarray(labels), np.asarray(probabilities)
            results[mode] = dict(cases=len(labels), bce=float(np.mean(losses)),
                                 auroc=float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
                                 brier=float(np.mean((labels-probabilities)**2)),
                                 mean_observed_regions=float(np.mean(observed_regions)),
                                 mean_observed_patches=float(np.mean(observed_patches)),
                                 mean_patch_fraction=float(np.mean(patch_fractions)))
    return results


def objective(model, dataset, i, stage, epoch, device):
    inputs, target = dataset.sample(i, 'zero' if stage == 'coarse' else 'random', epoch)
    target = target.to(device)
    out = model(**move(inputs, device))
    partial = F.binary_cross_entropy_with_logits(out['logit'], target)
    if stage == 'coarse':
        return partial
    full_inputs, _ = dataset.sample(i, 'full', epoch)
    full = model(**move(full_inputs, device))
    if full['fine_aux_logit'] is None:
        raise ValueError('Fusion training case has no eligible real fine')
    full_loss = F.binary_cross_entropy_with_logits(full['logit'], target)
    aux = F.binary_cross_entropy_with_logits(full['fine_aux_logit'], target)
    return .5*partial+.5*full_loss+.2*aux


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', required=True, type=Path)
    p.add_argument('--validation-manifest', required=True, type=Path)
    p.add_argument('--regions', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--stage', choices=('coarse', 'fusion'), required=True)
    p.add_argument('--predecessor', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['schema'] != 'KT2028-R2-predictor' or cfg['fine_loss_weights'] != [.5, .5, .2]:
        raise ValueError('Unregistered R2 predictor/objective configuration')
    if any(not isinstance(cfg[k], int) or cfg[k] <= 0 for k in ('max_epochs', 'patience', 'accumulate_cases')):
        raise ValueError('Positive epoch/patience/accumulation counts required')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    train = FusionDataset(args.train_manifest, args.regions, seed=cfg['seed'])
    if not len(train):
        raise ValueError('Empty Train')
    model = CrossScalePredictor(**cfg['architecture']).to(args.device)
    if not args.execute:
        inputs, _ = train.sample(0, 'zero')
        with torch.no_grad():
            model.eval(); result = model(**move(inputs, args.device))
        print(json.dumps(dict(status='real_Train_input_forward_only', coarse_shape=list(inputs['coarse'].shape),
                              zero_equals_coarse=bool(torch.equal(result['logit'], result['coarse_logit'])),
                              fitted=False, validation_used=False, test_used=False)))
        return
    if args.output.exists():
        raise FileExistsError('Use a new run directory; preserve all prior results')
    validation = FusionDataset(args.validation_manifest, args.regions, purpose='model_selection', seed=cfg['seed'])
    if not len(validation) or {r['canonical_case_key'] for r in train.rows} & {r['canonical_case_key'] for r in validation.rows}:
        raise ValueError('Empty/overlapping Train and Validation cases')
    asset_hashes = {}
    for dataset in (train, validation):
        for i, row in enumerate(dataset.rows):
            case = dataset.case(i)
            asset_hashes[row['canonical_case_key']] = case.receipt['regions_sha256']
    import hashlib
    asset_sha = hashlib.sha256(json.dumps(asset_hashes, sort_keys=True).encode()).hexdigest()
    model_sha_keys = ('train_manifest_sha256', 'validation_manifest_sha256', 'assets_sha256', 'config_sha256', 'source_shas')
    provenance = dict(stage=args.stage, config=cfg, config_sha256=sha256_file(args.config),
                      train_manifest_sha256=sha256_file(args.train_manifest),
                      validation_manifest_sha256=sha256_file(args.validation_manifest), assets_sha256=asset_sha,
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      source_shas={n: sha256_file(Path(__file__).with_name(n)) for n in
                                   ('fusion_network.py', 'fusion_data.py', 'train_fusion.py')},
                      test_used=False, validation_role='checkpoint_selection; not independent Test',
                      predecessor_sha256=sha256_file(args.predecessor) if args.predecessor else None)
    if args.stage == 'fusion':
        if not args.predecessor:
            raise ValueError('Fusion requires fitted coarse anchor')
        prior = torch.load(args.predecessor, map_location=args.device, weights_only=True)
        if prior['stage'] != 'coarse' or any(prior[k] != provenance[k] for k in model_sha_keys):
            raise ValueError('Coarse anchor data/config differs')
        model.load_state_dict(prior['model'], strict=True)
        model.freeze_coarse()
    elif args.predecessor:
        raise ValueError('Coarse anchor starts independently')
    if args.stage == 'coarse':
        model.requires_grad_(False); model.coarse.requires_grad_(True)
    parameters = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    args.output.mkdir()
    write_json(args.output/'run.json', provenance)
    modes = ['zero'] if args.stage == 'coarse' else ['zero', *cfg['validation_fractions'], 'full']
    select_modes = ['zero'] if args.stage == 'coarse' else [*cfg['validation_fractions'], 'full']
    anchor = {k: v.detach().cpu().clone() for k, v in model.coarse.state_dict().items()}
    best, stale, completed = float('inf'), 0, 0
    for epoch in range(cfg['max_epochs']):
        model.train()
        order = np.random.default_rng(cfg['seed']+epoch).permutation(len(train))
        total = 0.
        optimizer.zero_grad(set_to_none=True)
        accumulate = cfg['accumulate_cases']
        for j, i in enumerate(order):
            loss = objective(model, train, int(i), args.stage, epoch, args.device)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite R2 objective')
            group_start = (j//accumulate)*accumulate
            group_size = min(accumulate, len(order)-group_start)
            (loss/group_size).backward()
            total += float(loss.detach())
            if (j+1) % accumulate == 0 or j+1 == len(order):
                torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        scores = evaluate(model, validation, modes, args.device)
        selection = float(np.mean([scores[m]['bce'] for m in select_modes]))
        if args.stage == 'fusion' and any(not torch.equal(anchor[k], v.detach().cpu()) for k, v in model.coarse.state_dict().items()):
            raise ValueError('Frozen coarse anchor changed')
        event = dict(epoch=epoch, train_objective=total/len(train), selection_bce=selection,
                     validation=scores, coarse_anchor_frozen=args.stage == 'fusion')
        with (args.output/'epochs.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
        completed = epoch+1
        if selection < best:
            best, stale = selection, 0
            temp = args.output/'best.partial.pt'
            torch.save(dict(provenance, model=model.state_dict(), epoch=epoch, selection_bce=selection), temp)
            temp.replace(args.output/'best.pt')
        else:
            stale += 1
        if stale >= cfg['patience']:
            break
    saved = torch.load(args.output/'best.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(saved['model'], strict=True)
    write_json(args.output/'completion.json', dict(completed_epochs=completed, selected_epoch=saved['epoch'],
               checkpoint_sha256=sha256_file(args.output/'best.pt'), best_selection_bce=best,
               strict_reload=True, test_used=False))


if __name__ == '__main__':
    main()
