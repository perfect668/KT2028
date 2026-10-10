"""Explicit staged predictor entry. Without --execute, only CPU preflight runs."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss
from torch.utils.data import DataLoader

from .network import EvidencePredictor, TargetedStateUpdater
from .training_data import TrainRegionDataset, move_inputs, ragged_collate
from .training_stages import configure_stage, stage_objective


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alignment_summary(details):
    """Report alignment and actual TSU supervision separately from prognosis metrics."""
    return {'cases': len(details),
            'alignment_mse': sum(d['alignment_mse'] for d in details) / len(details),
            'tsu_mse': sum(d['tsu_mse'] for d in details) / len(details),
            'tsu_pairs': sum(d['tsu_pairs'] for d in details),
            'cases_with_tsu_pairs': sum(d['tsu_pairs'] > 0 for d in details)}


def validate_config(config):
    for name in ('epochs', 'batch_cases', 'max_tsu_pairs', 'seed', 'latent_dim', 'attention_branches'):
        value = config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == 'seed' else 1):
            raise ValueError(f'Invalid {name}')
    for name in ('learning_rate', 'weight_decay', 'tsu_threshold', 'endpoint_probability', 'gradient_clip'):
        if not np.isfinite(config[name]):
            raise ValueError(f'Invalid {name}')
    if config['learning_rate'] <= 0 or config['weight_decay'] < 0 or config['gradient_clip'] <= 0:
        raise ValueError('Invalid optimizer configuration')
    if not -1 <= config['tsu_threshold'] <= 1 or not 0 < config['endpoint_probability'] < 0.5:
        raise ValueError('Invalid TSU/sampling configuration')
    if config['selection_rule'] != 'minimum_fixed_validation_objective_first_tie':
        raise ValueError('Explicit fixed Validation selection rule required')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('full', 'align', 'partial'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--validation-manifest', required=True)
    parser.add_argument('--regions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--predecessor', help='Previous-stage checkpoint from this pipeline')
    parser.add_argument('--execute', action='store_true', help='Actually optimize and select on Validation')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    if not args.execute:
        # No Validation data loaded and no optimizer constructed in default mode.
        from .prepare_training import main as preflight
        import sys
        sys.argv = [sys.argv[0], '--manifest', args.train_manifest, '--regions', args.regions,
                    '--output', args.output, '--device', 'cpu']
        return preflight()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new run directory; preserve prior runs')
    seed = config['seed']
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = EvidencePredictor(config['latent_dim'], config['attention_branches']).to(args.device)
    updater = TargetedStateUpdater(config['latent_dim'], config['tsu_threshold']).to(args.device)
    if args.stage != 'full':
        if not args.predecessor:
            raise ValueError('Later stages require their fitted predecessor checkpoint')
        prior = torch.load(args.predecessor, map_location=args.device, weights_only=True)
        expected = {'align': 'full', 'partial': 'align'}[args.stage]
        if prior['stage'] != expected or prior['train_manifest_sha256'] != digest(args.train_manifest):
            raise ValueError('Wrong predecessor stage or Train manifest')
        if prior['validation_manifest_sha256'] != digest(args.validation_manifest):
            raise ValueError('Validation manifest changed')
        for name in ('latent_dim', 'attention_branches', 'tsu_threshold'):
            if prior['config'][name] != config[name]:
                raise ValueError('Architecture differs from predecessor')
        model.load_state_dict(prior['model'])
        updater.load_state_dict(prior['updater'])
    elif args.predecessor:
        raise ValueError('Full stage starts independently; do not silently warm-start')
    evidence = 'random' if args.stage == 'partial' else 'full'
    common = {'seed': seed, 'endpoint_probability': config['endpoint_probability']}
    train = TrainRegionDataset(args.train_manifest, args.regions, evidence=evidence, **common)
    validation_modes = ('zero', 'random', 'full') if args.stage == 'partial' else ('full',)
    validation = [TrainRegionDataset(args.validation_manifest, args.regions, evidence=mode,
                   purpose='model_selection', **common) for mode in validation_modes]
    if {r['canonical_case_key'] for r in train.rows} & {r['canonical_case_key'] for r in validation[0].rows}:
        raise ValueError('Train and Validation case overlap')
    if not len(train) or not len(validation[0]):
        raise ValueError('Empty Train or Validation manifest')
    parameters = configure_stage(model, updater, args.stage)
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    config_sha = digest(args.config)
    region_config_shas = set()
    regional_fingerprints = {}
    for dataset in (train, *validation):
        for row in dataset.rows:
            receipt = json.loads((Path(args.regions) / row['slides'][0]['slide_id'] / 'receipt.json').read_text())
            if receipt['status'] != 'completed':
                raise ValueError('Incomplete training/selection regional asset')
            asset = Path(args.regions) / row['slides'][0]['slide_id'] / 'regions.npz'
            if digest(asset) != receipt['regions_sha256']:
                raise ValueError('Regional asset hash mismatch')
            region_config_shas.add(receipt['config_sha256'])
            regional_fingerprints[asset.parent.name] = [receipt['regions_sha256'], receipt['fine_coords_sha256']]
    if len(region_config_shas) != 1:
        raise ValueError('Mixed regional preparation configurations')
    regional_assets_sha = hashlib.sha256(json.dumps(regional_fingerprints, sort_keys=True).encode()).hexdigest()
    if args.stage != 'full' and prior['regional_assets_sha256'] != regional_assets_sha:
        raise ValueError('Regional assets differ from predecessor run')
    output.mkdir(parents=True)
    provenance = {'stage': args.stage, 'config': config, 'config_sha256': config_sha,
                  'source_commit': source_commit, 'train_manifest_sha256': digest(args.train_manifest),
                  'validation_manifest_sha256': digest(args.validation_manifest),
                  'regional_config_sha256': next(iter(region_config_shas)),
                  'regional_assets_sha256': regional_assets_sha,
                  'device': args.device, 'torch_version': str(torch.__version__),
                  'predecessor_sha256': digest(args.predecessor) if args.predecessor else None,
                  'selection': config['selection_rule'], 'test_used': False}
    (output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n')
    best = float('inf')
    for epoch in range(config['epochs']):
        configure_stage(model, updater, args.stage)
        train.set_epoch(epoch)
        loader = DataLoader(train, batch_size=config['batch_cases'], shuffle=True,
                            generator=torch.Generator().manual_seed(seed + epoch), collate_fn=ragged_collate)
        train_total = 0.0
        train_details = []
        pair_rng = torch.Generator().manual_seed(seed + epoch)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            objectives = [stage_objective(model, updater, move_inputs(s['inputs'], args.device),
                       s['target'].to(args.device), args.stage, max_pairs=config['max_tsu_pairs'],
                       generator=pair_rng) for s in batch]
            losses = [value for value, _ in objectives]
            if args.stage == 'align':
                train_details.extend(details for _, details in objectives)
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            train_total += sum(float(v.detach()) for v in losses)
        model.eval()
        updater.eval()
        scores = []
        validation_metrics = {}
        with torch.no_grad():
            for dataset in validation:
                total = 0.0
                labels, probabilities = [], []
                alignment_details = []
                # Fixed masks and pair sampling across epochs, not a moving selection set.
                pair_rng = torch.Generator().manual_seed(seed)
                for s in dataset:
                    value, details = stage_objective(model, updater, move_inputs(s['inputs'], args.device),
                               s['target'].to(args.device), args.stage, max_pairs=config['max_tsu_pairs'],
                               generator=pair_rng)
                    total += float(value)
                    if args.stage == 'align':
                        alignment_details.append(details)
                    if 'logit' in details:
                        labels.append(int(s['target']))
                        probabilities.append(float(details['logit'].sigmoid()))
                scores.append(total / len(dataset))
                if alignment_details:
                    validation_metrics[dataset.evidence] = alignment_summary(alignment_details)
                if labels:
                    validation_metrics[dataset.evidence] = {
                        'auroc': float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
                        'brier': float(brier_score_loss(labels, probabilities)),
                        'bce': total / len(dataset), 'cases': len(dataset)}
        selection_score = sum(scores) / len(scores)
        if not np.isfinite(selection_score):
            raise FloatingPointError('Nonfinite selection objective')
        with (output / 'epochs.jsonl').open('a') as file:
            event = {'epoch': epoch, 'train_objective': train_total / len(train),
                     'validation_objective': selection_score, 'validation_modes': list(validation_modes),
                     'validation_metrics': validation_metrics}
            if train_details:
                event['train_alignment'] = alignment_summary(train_details)
            file.write(json.dumps(event) + '\n')
            print(json.dumps(event), flush=True)
        if selection_score < best:
            best = selection_score
            checkpoint = {**provenance, 'epoch': epoch, 'selection_objective': best,
                          'model': model.state_dict(), 'updater': updater.state_dict()}
            temp = output / 'best.partial.pt'
            torch.save(checkpoint, temp)
            temp.replace(output / 'best.pt')
    (output / 'completion.json').write_text(json.dumps({'completed_epochs': config['epochs'],
                     'best_checkpoint_sha256': digest(output / 'best.pt'), 'test_used': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
