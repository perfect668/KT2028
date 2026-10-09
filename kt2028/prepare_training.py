"""Real Train batch/backward preflight; no optimizer steps or fitted checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .network import EvidencePredictor, TargetedStateUpdater
from .training_data import TrainRegionDataset, move_inputs, ragged_collate
from .training_stages import configure_stage, stage_objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--regions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--case-count', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.case_count <= 0:
        raise ValueError('Positive check size required')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve earlier preflight receipts')
    torch.set_num_threads(2)
    torch.manual_seed(41)
    model = EvidencePredictor().to(args.device)
    updater = TargetedStateUpdater().to(args.device)
    receipts = []
    for stage, evidence in (('full', 'full'), ('align', 'full'), ('partial', 'random'),
                            ('partial', 'zero'), ('partial', 'intermediate')):
        dataset = TrainRegionDataset(args.manifest, args.regions,
                                     evidence='full' if evidence == 'intermediate' else evidence, seed=41)
        subset = torch.utils.data.Subset(dataset, range(min(args.case_count, len(dataset))))
        batch = next(iter(DataLoader(subset, batch_size=args.case_count, collate_fn=ragged_collate)))
        if evidence == 'intermediate':
            for sample in batch:
                ids = list(sample['inputs']['observed_fine'])
                if len(ids) < 2:
                    raise ValueError('Intermediate preflight needs at least two acquirable regions')
                sample['inputs']['observed_fine'] = {i: sample['inputs']['observed_fine'][i] for i in ids[::2]}
                sample['selected_patch_count'] = sum(len(v) for v in sample['inputs']['observed_fine'].values())
        model.zero_grad(set_to_none=True)
        updater.zero_grad(set_to_none=True)
        parameters = configure_stage(model, updater, stage)
        losses = []
        pairs = 0
        for sample in batch:
            inputs = move_inputs(sample['inputs'], args.device)
            loss, details = stage_objective(model, updater, inputs, sample['target'].to(args.device),
                                           stage, generator=torch.Generator().manual_seed(41))
            assert torch.isfinite(loss)
            losses.append(loss)
            pairs += details['tsu_pairs']
        torch.stack(losses).mean().backward()
        grads = [p.grad for p in parameters if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert all(p.grad is None for p in (*model.parameters(), *updater.parameters()) if not p.requires_grad)
        receipts.append({'stage': stage, 'evidence': evidence, 'cases': len(batch),
                         'region_counts': [len(s['inputs']['coarse']) for s in batch],
                         'acquired_patch_counts': [s['selected_patch_count'] for s in batch],
                         'finite_loss_and_gradients': True, 'frozen_parameters_without_gradients': True,
                         'tsu_supervision_pairs': pairs})
    report = {'scope': 'Real Train ragged loading and staged backward software preflight',
              'device': args.device, 'official_train_cases': len(dataset),
              'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'checks': receipts, 'random_initialization': True, 'optimizer_steps': 0,
              'checkpoint_saved': False, 'validation_or_test_prediction': False,
              'limitations': 'Gradient checks do not demonstrate prognosis quality; optimizer/checkpoint/Validation execution of the staged runner has not been exercised'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
