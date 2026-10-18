"""R2 per-case inputs; read feature values only for actually observed regions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from .uni_encoder import sha256_file


class FusionCase:
    def __init__(self, folder):
        folder = Path(folder)
        self.receipt = json.loads((folder/'receipt.json').read_text())
        if self.receipt['status'] != 'completed' or self.receipt['schema'] != 'KT2028-R2-coarse-UNI':
            raise ValueError('Require completed R2 coarse asset')
        if sha256_file(folder/'regions.npz') != self.receipt['regions_sha256']:
            raise ValueError('R2 asset checksum mismatch')
        with np.load(folder/'regions.npz', allow_pickle=False) as data:
            self.coarse, self.boxes = data['coarse'].copy(), data['bbox_um'].copy()
            self.counts, self.ptr = data['fine_counts'].copy(), data['fine_ptr'].copy()
            self.indices, self.relative = data['fine_indices'].copy(), data['fine_relative_xy'].copy()
        r = len(self.coarse)
        if self.coarse.shape != (r, 1024) or self.boxes.shape != (r, 4) or not np.isfinite(self.coarse).all():
            raise ValueError('Invalid R2 coarse input')
        if len(self.ptr) != r+1 or self.ptr[-1] != len(self.indices) or not np.array_equal(np.diff(self.ptr), self.counts):
            raise ValueError('Invalid R2 fine CSR')
        if self.relative.shape != (len(self.indices), 2) or not np.isfinite(self.relative).all():
            raise ValueError('Invalid R2 relative fine positions')
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError('Duplicate fine acquisition ownership')

    def inputs(self, selected):
        selected = [int(i) for i in selected]
        if len(set(selected)) != len(selected) or any(i < 0 or i >= len(self.coarse) or self.counts[i] <= 0 for i in selected):
            raise ValueError('Invalid observed region list')
        fine, relative = {}, {}
        if selected:
            group = zarr.open_group(self.receipt['fine_path'], mode='r')
            spans = [tuple(self.ptr[i:i+2]) for i in selected]
            chosen = np.concatenate([self.indices[a:b] for a, b in spans])
            # One indexed read avoids repeatedly decoding the same Zarr chunks.
            # It still requests ONLY selected indices, not the complete fine array.
            values = np.asarray(group['features'].oindex[chosen, :], dtype=np.float32)
            if values.shape != (len(chosen), 1024) or not np.isfinite(values).all():
                raise ValueError('Invalid real observed fine')
            offset = 0
            for i, (a, b) in zip(selected, spans):
                fine[i] = torch.from_numpy(values[offset:offset+b-a].copy())
                relative[i] = torch.from_numpy(self.relative[a:b].copy())
                offset += b-a
        return dict(coarse=torch.from_numpy(self.coarse.copy()), boxes=torch.from_numpy(self.boxes.copy()),
                    observed_fine=fine, relative_xy=relative,
                    candidate_patch_count=int(self.counts.sum()))


class FusionDataset:
    def __init__(self, manifest, regions, *, purpose='train', seed=41):
        document = json.loads(Path(manifest).read_text())
        expected = {'train': 'train', 'model_selection': 'validate'}.get(purpose)
        if expected is None or document['partition'] != expected or document['endpoint'] != 'five_year_death':
            raise ValueError('R2 fitting uses Train, selection uses explicit Validation; no Test')
        self.rows, self.regions, self.seed = document['patients'], Path(regions), seed
        keys = [r['canonical_case_key'] for r in self.rows]
        if len(keys) != len(set(keys)) or any(len(r['slides']) != 1 or r['label'] not in (0, 1) for r in self.rows):
            raise ValueError('Invalid official case mapping/targets')
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def case(self, i):
        row = self.rows[i]
        slide = row['slides'][0]
        if i not in self._cache:
            if len(self._cache) >= 16:
                self._cache.pop(next(iter(self._cache)))
            value = FusionCase(self.regions/slide['slide_id'])
            if Path(value.receipt['fine_path']) != Path(slide['features_path']):
                raise ValueError('R2 fine source differs from official manifest')
            self._cache[i] = value
        return self._cache[i]

    def sample(self, i, mode, epoch=0):
        case = self.case(i)
        eligible = np.flatnonzero(case.counts > 0)
        if mode == 'zero':
            selected = []
        elif mode == 'full':
            selected = eligible
        else:
            key = f'{self.seed}:{epoch}:{self.rows[i]["canonical_case_key"]}:{mode}'.encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
            if mode == 'random':
                category = rng.random()
                k = 0 if category < .25 else len(eligible) if category < .5 else int(rng.integers(1, len(eligible))) if len(eligible) > 1 else len(eligible)
            else:
                fraction = float(mode)
                if not 0 < fraction < 1:
                    raise ValueError('Fractional observation must be between zero/full')
                k = min(len(eligible), max(1, int(np.ceil(fraction*len(eligible))))) if len(eligible) else 0
            selected = rng.permutation(eligible)[:k]
        return case.inputs(selected), torch.tensor(float(self.rows[i]['label']))


def move(inputs, device):
    result = dict(coarse=inputs['coarse'].to(device), boxes=inputs['boxes'].to(device),
                observed_fine={i: x.to(device) for i, x in inputs['observed_fine'].items()},
                relative_xy={i: x.to(device) for i, x in inputs['relative_xy'].items()},
                candidate_patch_count=inputs['candidate_patch_count'])
    if 'slide_indices' in inputs:
        result['slide_indices'] = inputs['slide_indices'].to(device)
    return result
