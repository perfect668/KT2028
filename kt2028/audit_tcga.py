"""Independent TCGA feature/geometry audit and patient-grouped candidate index.

This is an input-asset audit, not a prognosis evaluation or a train/test split.
Incomplete snapshots remain explicitly incomplete. Raw SVS are never removed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .uni_encoder import sha256_file


def validate_geometry(grid, native_mpp_xy):
    coarse, fine = grid['coarse_bbox_um'], grid['fine_bbox_um']
    parents = grid['fine_parent_indices']
    n, m = len(coarse), len(fine)
    if n < 1 or m < 1 or coarse.shape != (n, 4) or fine.shape != (m, 4):
        raise ValueError('Empty/malformed physical boxes')
    if parents.shape != (m,) or parents.dtype.kind not in 'iu' or np.any(parents < 0) or np.any(parents >= n):
        raise ValueError('Invalid fine-parent index')
    if not np.isfinite(coarse).all() or not np.isfinite(fine).all():
        raise ValueError('Nonfinite physical coordinates')
    if np.any(coarse[:, 2:] <= coarse[:, :2]) or np.any(fine[:, 2:] <= fine[:, :2]):
        raise ValueError('Nonpositive spatial extent')
    if np.any(coarse < 0) or np.any(fine < 0):
        raise ValueError('Negative physical coordinates')
    if np.any(fine[:, :2] < coarse[parents, :2]-1e-7) or np.any(fine[:, 2:] > coarse[parents, 2:]+1e-7):
        raise ValueError('Fine box extends outside its declared coarse region')
    counts = np.bincount(parents.astype(np.int64), minlength=n)
    if np.any(counts > 16) or np.any(np.diff(parents.astype(np.int64)) < 0):
        raise ValueError('Non-nested or unordered fine ownership')
    np.testing.assert_array_equal(grid['fine_counts'], counts)
    np.testing.assert_array_equal(grid['fine_ptr'], np.r_[0, np.cumsum(counts)])
    for label, boxes, field in [('coarse', coarse, 448), ('fine', fine, 112)]:
        if np.any(boxes[:, 2:]-boxes[:, :2] > field+1e-7):
            raise ValueError('Physical patch exceeds configured field')
        xy = grid[label+'_coords']
        np.testing.assert_array_equal(xy, np.floor(boxes[:, :2]/native_mpp_xy).astype(np.int64))
        if len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError('Duplicate patch location')
    relative = ((fine[:, :2]-coarse[parents, :2])+56)/448
    np.testing.assert_allclose(grid['fine_relative_xy'], relative, atol=1e-6, rtol=0)
    for key in ['coarse_image_valid_fraction', 'fine_image_valid_fraction']:
        values = grid[key]
        if not np.isfinite(values).all() or np.any(values <= 0) or np.any(values > 1):
            raise ValueError('Invalid image-valid fraction')


def audit_slide(folder, sample, config, verify_hashes):
    import zarr
    from .prepare_tcga import verify_output
    receipt = json.loads((folder/'receipt.json').read_text())
    if receipt['status'] != 'completed' or receipt['file_id'] != sample['id']:
        raise ValueError('Missing completion or source UUID mismatch')
    if receipt['source_md5'] != sample['md5'] or receipt['source_size'] != sample['size']:
        raise ValueError('Feature source disagrees with pinned GDC manifest')
    if receipt['patient_ids'] != sample['patient_ids'] or receipt['project'] != sample['project']:
        raise ValueError('Feature/patient mapping mismatch')
    if receipt['config_sha256'] != sha256_file(config) or not receipt['original_uni_frozen']:
        raise ValueError('Scale/encoder protocol mismatch')
    settings = json.loads(config.read_text())
    if receipt['weight_sha256'] != settings['uni']['sha256']:
        raise ValueError('Encoder weight fingerprint mismatch')
    if sha256_file(folder/'geometry.npz') != receipt['geometry_sha256']:
        raise ValueError('Geometry checksum mismatch')
    with np.load(folder/'geometry.npz', allow_pickle=False) as geometry:
        validate_geometry(geometry, np.asarray(receipt['slide_metadata']['native_mpp_xy']))
        for name, mpp in [('coarse', 2.0), ('fine', 0.5)]:
            group = zarr.open_group(str(folder/(name+'.zarr')), mode='r')
            rows = len(geometry[name+'_coords'])
            if group['features'].shape != (rows, 1024) or str(group['features'].dtype) != 'float32':
                raise ValueError('Unexpected feature shape or dtype')
            if group.attrs['status'] != 'completed' or group.attrs['completed_rows'] != rows:
                raise ValueError('Uncommitted feature rows')
            if group.attrs['mpp_um'] != mpp or group.attrs['source_md5'] != sample['md5']:
                raise ValueError('Stored physical scale or source mismatch')
            np.testing.assert_array_equal(group['coords'][:], geometry[name+'_coords'])
            np.testing.assert_array_equal(group['bbox_um'][:], geometry[name+'_bbox_um'])
            if name == 'coarse':
                np.testing.assert_array_equal(group['fine_ptr'][:], geometry['fine_ptr'])
                np.testing.assert_array_equal(group['fine_counts'][:], geometry['fine_counts'])
            else:
                np.testing.assert_array_equal(group['parent_indices'][:], geometry['fine_parent_indices'])
                np.testing.assert_array_equal(group['relative_xy'][:], geometry['fine_relative_xy'])
    if verify_hashes:
        verify_output(folder, receipt)
    return dict(file_id=sample['id'], status='verified' if verify_hashes else 'metadata_checked',
                feature_folder='features/'+sample['id'],
                coarse_rows=receipt['coarse']['rows'], fine_rows=receipt['fine']['rows'],
                receipt_sha256=sha256_file(folder/'receipt.json'),
                feature_hashes={name: receipt[name]['features_sha256'] for name in ['coarse', 'fine']},
                output_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--clinical', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--verify-feature-hashes', action='store_true')
    ap.add_argument('--final', action='store_true')
    args = ap.parse_args()
    if args.final and not args.verify_feature_hashes:
        raise ValueError('Final audit requires feature content hashes')
    samples = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    clinical = json.loads(args.clinical.read_text(encoding='utf-8-sig'))
    if len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate GDC source UUID')
    clinical_by_patient = {r['patient_id']: r for r in clinical}
    if len(clinical_by_patient) != len(clinical):
        raise ValueError('Duplicate clinical patient')
    expected_by_patient, slides = {}, {}
    for sample in samples:
        if len(sample['patient_ids']) != 1 or len(sample['case_ids']) != 1:
            raise ValueError('Ambiguous slide/patient mapping')
        pid = sample['patient_ids'][0]
        if pid not in clinical_by_patient or sample['case_ids'][0] != clinical_by_patient[pid]['case_id']:
            raise ValueError('Image/clinical case mismatch')
        expected_by_patient.setdefault(pid, []).append(sample['id'])
        folder = args.root/'features'/sample['id']
        if (folder/'receipt.json').exists():
            try:
                slides[sample['id']] = audit_slide(folder, sample, args.config, args.verify_feature_hashes)
            except Exception as error:
                slides[sample['id']] = dict(file_id=sample['id'], status='audit_failed',
                    error=type(error).__name__+': '+str(error)[:400])
        elif (folder/'failure.json').exists():
            slides[sample['id']] = dict(file_id=sample['id'], status='extraction_failed',
                                        error=json.loads((folder/'failure.json').read_text())['error'])
        else:
            slides[sample['id']] = dict(file_id=sample['id'], status='pending')
    if set(expected_by_patient) != set(clinical_by_patient):
        raise ValueError('Clinical cohort and image cohort differ')
    good = {'verified', 'metadata_checked'}
    patients = []
    for pid, ids in sorted(expected_by_patient.items()):
        r = clinical_by_patient[pid]
        patients.append(dict(patient_id=pid, case_id=r['case_id'], project=r['project'],
            slides=[slides[sid] for sid in ids],
            all_expected_slides_ready=all(slides[sid]['status'] in good for sid in ids),
            os_candidate=dict(time_days=r['os_time_days'], event=r['os_event'],
                time_days_5y=r['time_days_5y'], event_5y=r['event_5y'],
                eligible=r['eligible_candidate'], exclusion_reason=r['exclusion_reason'], flags=r['flags']),
            split=None))
    counts = Counter(r['status'] for r in slides.values())
    summary = dict(schema='KT2028-TCGA-feature-dataset-audit-1', at_unix=time.time(),
        expected_files=len(samples), expected_patients=len(patients), slide_status_counts=dict(counts),
        completed_files=sum(counts[k] for k in good),
        completed_patients=sum(p['all_expected_slides_ready'] for p in patients),
        complete_os_candidate_patients=sum(p['all_expected_slides_ready'] and p['os_candidate']['eligible'] for p in patients),
        coarse_patches=sum(s.get('coarse_rows', 0) for s in slides.values() if s['status'] in good),
        fine_patches=sum(s.get('fine_rows', 0) for s in slides.values() if s['status'] in good),
        completed_output_bytes=sum(s.get('output_bytes', 0) for s in slides.values() if s['status'] in good),
        feature_content_hashes_checked=args.verify_feature_hashes,
        feature_dataset_complete=counts['verified'] == len(samples),
        patient_split_assigned=False, model_training_ready_claimed=False,
        manifest_sha256=sha256_file(args.manifest), clinical_sha256=sha256_file(args.clinical),
        config_sha256=sha256_file(args.config), audit_code_sha256=sha256_file(Path(__file__)))
    from .extract_dual_scale import atomic_json
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output/'patient_index.json', patients)
    atomic_json(args.output/'audit_summary.json', summary)
    print(json.dumps(summary), flush=True)
    if args.final and not summary['feature_dataset_complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
