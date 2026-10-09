"""CPU forward checks on prepared Train assets; no fitted predictions/metrics."""
import argparse
import json
from pathlib import Path

import torch

from .data import RegionCase
from .network import EvidencePredictor, TargetedStateUpdater, SharedActorCritic
from .episode import AcquisitionEpisode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True, help='Three registered Train inputs only')
    parser.add_argument('--regions', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(31)
    model = EvidencePredictor().eval()
    updater = TargetedStateUpdater().eval()
    actor = SharedActorCritic().eval()
    records = []
    samples = json.loads(Path(args.samples).read_text())
    # A common software-check reference, derived solely from these Train inputs.
    check_n_ref = max(int(RegionCase(Path(args.regions) / s['slide_id']).counts.sum()) for s in samples)
    for sample in samples:
        case = RegionCase(Path(args.regions) / sample['slide_id'])
        # Explicit software-test units only; not the future experimental protocol.
        episode = AcquisitionEpisode(case, model, n_ref=check_n_ref, budget_patches=int(case.counts.sum()),
                                     cost_preference=0, updater=updater)
        before = episode.visible_state()
        assert not before['prediction']['observed'].any()
        region = int(before['legal'][:-1].nonzero()[0])
        after, _, done, _ = episode.step(region)
        with torch.no_grad():
            dist, value = actor(**after)
        assert after['prediction']['observed'].sum() == 1
        assert dist.probs[region] == 0 and dist.probs[-1] > 0
        assert torch.isfinite(after['prediction']['logit']) and torch.isfinite(value)
        if not done:
            episode.step(len(case.coarse))
        records.append({'regions': len(case.coarse), 'acquired_patch_count': int(case.counts[region]),
                        'zero_fine_forward': True, 'one_acquisition_forward': True,
                        'repeat_masked': True, 'stop_legal': True})
    receipt = {'scope': 'CPU software forward checks on three registered Train images',
               'random_initialization': True, 'training': False, 'prognosis_evaluation': False,
               'software_check_n_ref': check_n_ref,
               'protocol_not_selected': 'Common three-Train reference, lambda=0 and full-candidate budget are check-only inputs',
               'cases': records}
    Path(args.output).write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
