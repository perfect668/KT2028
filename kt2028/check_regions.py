"""Check real prepared arrays and acquisition semantics; never calculate prognosis metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from .data import RegionCase


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=Path,required=True)
    ap.add_argument('--regions',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();rows=[]
    for sample in json.loads(args.samples.read_text()):
        folder=args.regions/sample['slide_id'];case=RegionCase(folder)
        expected=case.receipt['regions_sha256']
        if hashlib.sha256((folder/'regions.npz').read_bytes()).hexdigest()!=expected:raise ValueError('Asset hash mismatch')
        initial=case.visible_state()
        assert initial['observed_fine']=={} and not initial['observed'].any() and initial['spent_patch_count']==0
        assert 'label' not in initial and np.isfinite(initial['coarse']).all()
        eligible=np.flatnonzero(initial['acquirable']);seen=[]
        for i in eligible:
            evidence=case.acquire(int(i));seen.extend(evidence['indices'].tolist())
            assert evidence['fine'].shape==(evidence['new_patch_cost'],1024)
            assert np.isfinite(evidence['fine']).all()
            assert case.visible_state()['observed'][i]
            assert not case.visible_state()['acquirable'][i]
        assert len(seen)==len(set(seen))==case.patch_cost==int(case.counts.sum())
        if len(eligible):
            try:case.acquire(int(eligible[0]))
            except ValueError:pass
            else:raise AssertionError('Duplicate acquire accepted')
        for i in np.flatnonzero(case.counts==0):
            try:case.acquire(int(i))
            except ValueError:pass
            else:raise AssertionError('Empty acquire accepted')
        rows.append({'regions':len(case.coarse),'acquirable_regions':len(eligible),'patches_acquired':case.patch_cost,
                     'initial_zero_fine':True,'unique_acquisition':True,'duplicate_and_empty_actions_rejected':True})
    report={'scope':'real asset/acquisition software check; no trained model or prognosis metric',
            'cases_checked':len(rows),'checks':rows,'no_training_or_evaluation':True}
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print('REGION_CHECKS '+json.dumps(report),flush=True)

if __name__=='__main__':main()
