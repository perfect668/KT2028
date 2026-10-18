"""Materialize the user-approved available-slide cohort without splitting/training."""
import argparse
import copy
import json
from pathlib import Path

from .uni_encoder import sha256_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--patient-index', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    samples = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    patients = json.loads(args.patient_index.read_text(encoding='utf-8-sig'))
    audit = json.loads(args.audit.read_text(encoding='utf-8-sig'))
    if not audit['feature_content_hashes_checked'] or not audit['specimen_provenance_checked']:
        raise ValueError('Requires content and specimen audit')
    if sha256_file(args.manifest) != audit['manifest_sha256']:
        raise ValueError('Source manifest changed since audit')
    by_id = {s['id']: s for s in samples}
    slides = {s['file_id']: s for p in patients for s in p['slides']}
    if len(by_id) != len(samples) or set(by_id) != set(slides):
        raise ValueError('Duplicate or mismatching slide coverage')
    selected, deferred, available, modeling = [], [], [], []
    for p in patients:
        kept = []
        for s in p['slides']:
            if s['status'] == 'verified':
                if by_id[s['file_id']]['patient_ids'] != [p['patient_id']]:
                    raise ValueError('Patient mapping mismatch')
                selected.append(by_id[s['file_id']])
                kept.append(s)
            elif s['status'] == 'extraction_failed' and 'Missing native MPP' in s.get('error', ''):
                deferred.append(dict(source=by_id[s['file_id']], reason='missing_native_mpp',
                                     source_retained=True, feature_extraction_complete=False))
            else:
                raise ValueError('Unexpected slide status; do not silently exclude')
        if kept:
            row = copy.deepcopy(p)
            row['slides'] = kept
            row['deferred_file_ids'] = [s['file_id'] for s in p['slides'] if s not in kept]
            row['all_selected_slides_ready'] = True
            row['split'] = None
            available.append(row)
            if p['all_expected_slides_ready'] and p['os_candidate']['eligible']:
                modeling.append(row)
    if len(selected) != audit['completed_files'] or len(modeling) != audit['complete_os_candidate_patients']:
        raise ValueError('Audit and cohort counts differ')
    summary = dict(schema='KT2028-TCGA-available-cohort-1',
        decision='2026-10-18 user: use 600 slides; retain 25 pending MPP',
        source_files=len(samples), selected_files=len(selected), deferred_files=len(deferred),
        available_patients=len(available), os_candidate_patients=len(modeling),
        full_os_events=sum(p['os_candidate']['event'] == 1 for p in modeling),
        five_year_events=sum(p['os_candidate']['event_5y'] == 1 for p in modeling),
        selected_cohort_assets_complete=True, original_625_complete=False,
        patient_split_assigned=False, model_training_ready_claimed=False,
        manifest_sha256=sha256_file(args.manifest), patient_index_sha256=sha256_file(args.patient_index),
        audit_sha256=sha256_file(args.audit), builder_sha256=sha256_file(Path(__file__)))
    payloads = dict(samples=selected, deferred_slides=deferred, patient_index=available,
                    os_modeling_candidates=modeling, summary=summary)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        path = args.output/(name+'.json')
        data = json.dumps(value, indent=2, ensure_ascii=False)+'\n'
        if path.exists() and path.read_text(encoding='utf-8') != data:
            raise ValueError('Preserve existing cohort; output differs: '+name)
        with path.open('w', encoding='utf-8', newline='\n') as f:
            f.write(data)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
