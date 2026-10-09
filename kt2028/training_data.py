"""Supervised Train loader; outcome targets stay outside acquisition inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from .data import RegionCase


class TrainRegionDataset(Dataset):
    def __init__(self, manifest, regions, *, evidence='full', seed=0,
                 purpose='train', endpoint_probability=0.25):
        self.manifest_path = Path(manifest)
        document = json.loads(self.manifest_path.read_text())
        # Filename is not accepted as proof of split: verify manifest contents.
        required_partition = {'train': 'train', 'model_selection': 'validate'}.get(purpose)
        if required_partition is None or document.get('partition') != required_partition or document.get('endpoint') != 'five_year_death':
            raise ValueError('Require official Train for fitting or explicit Validation for selection; no Test loader')
        if evidence not in ('full', 'zero', 'random'):
            raise ValueError('Unknown evidence mode')
        self.rows = document['patients']
        self.regions = Path(regions)
        self.evidence, self.seed, self.epoch = evidence, int(seed), 0
        if not 0 < endpoint_probability < 0.5:
            raise ValueError('Partial sampling must cover zero, full and intermediate evidence')
        self.endpoint_probability = endpoint_probability
        keys = [r['canonical_case_key'] for r in self.rows]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate official case key')
        for row in self.rows:
            if len(row['slides']) != 1:
                raise ValueError('Multi-slide clinical aggregation needs an explicit extension')
            if row.get('label') not in (0, 1):
                raise ValueError('Missing binary outcome target')

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        slide = row['slides'][0]
        case = RegionCase(self.regions / slide['slide_id'])
        if Path(case.receipt['fine_path']) != Path(slide['features_path']):
            raise ValueError('Fine asset differs from official manifest')
        eligible = np.flatnonzero(case.counts > 0)
        if self.evidence == 'zero':
            selected = np.array([], dtype=np.int64)
        elif self.evidence == 'full':
            selected = eligible
        else:
            # Stateless per-case/epoch RNG: reproducible across loader worker counts.
            key = f'{self.seed}:{self.epoch}:{row["canonical_case_key"]}'.encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
            category = rng.random()
            if category < self.endpoint_probability:
                k = 0
            elif category < 2 * self.endpoint_probability or len(eligible) < 2:
                k = len(eligible)
            else:
                k = int(rng.integers(1, len(eligible)))
            selected = rng.permutation(eligible)[:k]
        fine = {}
        if len(selected):
            index_groups = [case.indices[case.ptr[i]:case.ptr[i+1]] for i in selected]
            indices = np.concatenate(index_groups)
            group = zarr.open_group(case.receipt['fine_path'], mode='r')
            values = np.asarray(group['features'].oindex[indices, :], dtype=np.float32)
            if values.shape != (len(indices), 1024) or not np.isfinite(values).all():
                raise ValueError('Invalid training UNI values')
            offset = 0
            for i, ids in zip(selected, index_groups):
                fine[int(i)] = torch.from_numpy(values[offset:offset+len(ids)].copy())
                offset += len(ids)
        inputs = {'coarse': torch.from_numpy(case.coarse), 'boxes': torch.from_numpy(case.boxes),
                  'observed_fine': fine}
        return {'inputs': inputs, 'target': torch.tensor(float(row['label'])),
                'case_key': row['canonical_case_key'], 'selected_patch_count': sum(len(v) for v in fine.values()),
                'candidate_patch_count': int(case.counts.sum())}


def ragged_collate(samples):
    """A list of case bags: no fabricated patches and no cross-case attention."""
    return list(samples)


def move_inputs(inputs, device):
    return {'coarse': inputs['coarse'].to(device), 'boxes': inputs['boxes'].to(device),
            'observed_fine': {i: bag.to(device) for i, bag in inputs['observed_fine'].items()}}
