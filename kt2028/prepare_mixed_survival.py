"""Build a new, source-stratified development split; never modify old cohorts."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .uni_encoder import sha256_file


def target(t, e, edges):
    t = float(t)
    if e not in (0, 1) or not math.isfinite(t) or not 0 < t <= edges[-1]:
        raise ValueError('Invalid observed survival time/event')
    b = next((i for i, x in enumerate(edges[1:]) if t <= x), -1)
    return dict(time_days=t, event=e, event_bin=b if e else -1,
                completed_intervals=b if e else sum(x <= t for x in edges[1:]))


def split_patients(rows, seed, validation_fraction):
    strata = {}
    if len({p['canonical_case_key'] for p in rows}) != len(rows):
        raise ValueError('Duplicate patient identity')
    all_slides = [s['slide_id'] for p in rows for s in p['slides']]
    if len(set(all_slides)) != len(all_slides):
        raise ValueError('Slide shared by multiple patients')
    for p in rows:
        strata.setdefault((p['cohort'], p['survival_target']['event']), []).append(p)
    train, val = [], []
    for key, values in sorted(strata.items()):
        values = sorted(values, key=lambda p: hashlib.sha256(
            f"{seed}:{p['canonical_case_key']}".encode()).hexdigest())
        n = int(math.floor(len(values)*validation_fraction+.5))
        if not 0 < n < len(values):
            raise ValueError('Too few patients in a source/event stratum')
        val.extend(values[:n]); train.extend(values[n:])
    return sorted(train, key=lambda p:p['canonical_case_key']), sorted(val, key=lambda p:p['canonical_case_key'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--surgen-cohort', type=Path, required=True)
    ap.add_argument('--tcga-candidates', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    edges = cfg['data_contract']['interval_edges_days']
    rows, excluded, removed_slides, old_sources = [], [], [], {}
    for partition in ('train', 'validate', 'test'):
        path = args.surgen_cohort/(partition+'_survival_manifest.json')
        old_sources[partition] = sha256_file(path)
        doc = json.loads(path.read_text(encoding='utf-8'))
        for p in doc['patients']:
            t = p['survival_target']
            rows.append(dict(canonical_case_key='surgen:'+p['canonical_case_key'], cohort='surgen',
                original_patient_id=p['canonical_case_key'], original_partition=partition,
                original_followup_assumption='published common five-year administration; individual dates unverified',
                survival_target=target(t['time_days'], t['event'], edges),
                slides=[dict(slide_id=s['slide_id'], asset_id=s['slide_id']) for s in p['slides']]))
    tcga = json.loads(args.tcga_candidates.read_text(encoding='utf-8'))
    for p in tcga:
        if not p['all_expected_slides_ready'] or not p['os_candidate']['eligible']:
            raise ValueError('Input must be the previously audited complete OS candidates')
        if p['os_candidate']['flags']:
            excluded.append(dict(patient_id=p['patient_id'], cohort='tcga',
                reason='clinical_flag_pending_source_resolution', flags=p['os_candidate']['flags']))
            continue
        primary = [s for s in p['slides'] if s['biospecimen']['primary_diagnostic_candidate']]
        removed_slides.extend(dict(file_id=s['file_id'], reason='nonprimary_specimen_not_baseline_evidence')
                              for s in p['slides'] if s not in primary)
        if not primary:
            excluded.append(dict(patient_id=p['patient_id'], cohort='tcga', reason='no_primary_slide'))
            continue
        t = p['os_candidate']
        rows.append(dict(canonical_case_key='tcga:'+p['patient_id'], cohort='tcga',
            original_patient_id=p['patient_id'], original_partition=None,
            original_followup_assumption='GDC diagnosis-relative observed death/last-follow-up',
            survival_target=target(t['time_days_5y'], t['event_5y'], edges),
            slides=[dict(slide_id=s['file_id'], asset_id=s['file_id'],
                         receipt_sha256=s['receipt_sha256'], feature_hashes=s['feature_hashes']) for s in primary]))
    train, val = split_patients(rows, cfg['seed'], cfg['validation_fraction'])
    def counts(values):
        return dict(patients=len(values), events=sum(p['survival_target']['event'] for p in values),
            cohorts=dict(Counter(p['cohort'] for p in values)), slides=sum(len(p['slides']) for p in values),
            early_censored_zero_annual_nll=sum(not p['survival_target']['event'] and
                p['survival_target']['completed_intervals']==0 for p in values))
    sources = dict(surgen_manifests=old_sources, tcga_candidates_sha256=sha256_file(args.tcga_candidates),
                   config_sha256=sha256_file(args.config))
    documents = {}
    for partition, values in [('train', train), ('validate', val)]:
        documents[partition+'_manifest'] = dict(schema='KT2028-mixed-survival-manifest-1',
            partition=partition, data_contract=cfg['data_contract'], sources=sources, patients=values)
    documents['exclusions_private'] = dict(clinical_review=excluded, deferred_nonprimary_slides=removed_slides)
    documents['summary'] = dict(schema='KT2028-mixed-survival-preparation-1',
        source_candidate_patients=413+len(tcga), selected=counts(rows), train=counts(train), validate=counts(val),
        internal_test_patients=0, external_test_dataset_secured=False,
        clinical_review_exclusions=len(excluded), nonprimary_slides_deferred=len(removed_slides),
        split_rule='one seed; SHA256 ordering within cohort x five-year event; 90/10 rounded per stratum',
        historical_surgen_split_retained_in_sources=True,
        old_weights_reused=False, validation_is_independent_test=False,
        identity_check='source namespaces and unique slide IDs; no cross-source genomic identity proof',
        sources=sources, code_sha256=sha256_file(Path(__file__)), training_started=False)
    manifest_hashes = {name:hashlib.sha256((json.dumps(documents[name],ensure_ascii=False,indent=2)+'\n').encode('utf-8')).hexdigest()
                       for name in ('train_manifest','validate_manifest')}
    documents['summary']['manifest_sha256'] = manifest_hashes
    documents['split_lock'] = dict(schema='KT2028-mixed-split-lock-1',
        config_sha256=sha256_file(args.config), manifest_sha256=manifest_hashes,
        train_patients=len(train), validate_patients=len(val), seed=cfg['seed'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in documents.items():
        path = args.output/(name+'.json')
        data = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
        if path.exists() and path.read_text(encoding='utf-8') != data:
            raise ValueError('Preserve previous prepared split: '+str(path))
        with path.open('w', encoding='utf-8', newline='\n') as f:
            f.write(data)
    print(json.dumps(documents['summary']))


if __name__ == '__main__':
    main()
