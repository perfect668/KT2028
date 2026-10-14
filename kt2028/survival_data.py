"""Survival targets stay separate from observed-image inputs; Train/Val only."""
import json
from pathlib import Path

from .fusion_data import FusionDataset
from .prepare_survival import contract_hash
from .survival import encode_target


class SurvivalDataset(FusionDataset):
    """Reuse checked fine access/sampling/cache, replacing only target protocol."""
    def __init__(self, manifest, regions, *, contract, purpose='train', seed=41):
        doc = json.loads(Path(manifest).read_text())
        expected = {'train':'train', 'model_selection':'validate'}.get(purpose)
        if doc['schema'] != 'KT2028-survival-manifest-1' or expected is None or doc['partition'] != expected:
            raise ValueError('Only registered Train/Validation survival inputs; no Test entry')
        if doc['endpoint'] != contract['endpoint'] or doc['contract'] != contract or doc['contract_sha256'] != contract_hash(contract):
            raise ValueError('Survival manifest/config contract differs')
        if doc['clinical_source_sha256'] != contract['clinical_source_sha256']:
            raise ValueError('Clinical source provenance differs')
        self.rows, self.regions, self.seed, self._cache = doc['patients'], Path(regions), seed, {}
        keys = [r['canonical_case_key'] for r in self.rows]
        if len(keys) != len(set(keys)) or len(keys) != contract['expected_cases'][expected]:
            raise ValueError('Duplicate or unexpected survival cohort')
        if sum(r['survival_target']['event'] for r in self.rows) != contract['expected_events'][expected]:
            raise ValueError('Unexpected survival event count')
        for row in self.rows:
            target = row['survival_target']
            if len(row['slides']) != 1 or row['label'] != target['event']:
                raise ValueError('Single-slide survival/event mapping differs')
            if target != encode_target(target['time_days'], target['event'], contract['interval_edges_days']):
                raise ValueError('Survival target encoding differs')
            if not target['event'] and target['time_days'] != contract['administrative_censor_day']:
                raise ValueError('Earlier censoring requires a different evaluation contract')

    def sample(self, i, mode, epoch=0):
        # Parent returns image inputs and a binary tensor; discard the latter.
        inputs, _ = super().sample(i, mode, epoch)
        return inputs, dict(self.rows[i]['survival_target'])
