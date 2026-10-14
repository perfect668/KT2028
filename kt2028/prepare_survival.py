"""Derive the authorized 413-case cohort; never change official source manifests."""
import argparse
import hashlib
import json
import re
import csv
from pathlib import Path

from .survival import encode_target, validate_edges
from .uni_encoder import sha256_file


def canonical_number(value):
    match = re.fullmatch(r'SR386_40X_HE_T(\d+)', value)
    return int(match.group(1)) if match else int(value)


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def classify_record(raw, binary, contract):
    if raw['died_within_5_years'] != str(binary):
        raise ValueError('Official binary and clinical source flag disagree')
    value = raw['days_till_death'].strip()
    edges = contract['interval_edges_days']
    if binary == 0:
        if value != 'Alive':
            raise ValueError('Require published Alive literal for admin censoring')
        return encode_target(contract['administrative_censor_day'], 0, edges), None
    try:
        t = float(value)
    except ValueError as exc:
        raise ValueError('Unexpected nonnumeric task death time') from exc
    if t == 0:
        return None, 'zero_death_day_pending_confirmation'
    if t > contract['administrative_censor_day']:
        return None, 'death_day_beyond_five_year_window'
    return encode_target(t, 1, edges), None


def prepare(clinical, manifests, config, output):
    cfg = json.loads(Path(config).read_text())
    if cfg['schema'] != 'KT2028-survival-predictor-1':
        raise ValueError('Require registered survival configuration')
    contract = cfg['data_contract']
    edges = validate_edges(contract['interval_edges_days'])
    if edges[-1] != contract['administrative_censor_day'] or len(edges)-1 != cfg['architecture']['intervals']:
        raise ValueError('Window/head/interval contract differs')
    clinical_sha = sha256_file(clinical)
    if clinical_sha != contract['clinical_source_sha256']:
        raise ValueError('Clinical source version differs; do not silently rederive labels')
    with Path(clinical).open(newline='', encoding='utf-8-sig') as f:
        raw_rows = list(csv.DictReader(f))
    raw = {canonical_number(r['case_id']): r for r in raw_rows}
    if len(raw) != len(raw_rows):
        raise ValueError('Duplicate clinical case identity')
    outputs, seen, exclusions, summary = {}, set(), [], {}
    for path in manifests:
        doc = json.loads(Path(path).read_text())
        split = doc['partition']
        if split not in ('train', 'validate', 'test') or split in outputs or doc['endpoint'] != 'five_year_death':
            raise ValueError('Require one official manifest per original partition')
        patients = []
        for p in doc['patients']:
            number = canonical_number(p['canonical_case_key'])
            if number in seen or number not in raw or len(p['slides']) != 1 or p['label'] not in (0, 1):
                raise ValueError('Invalid/overlapping case or slide mapping')
            seen.add(number)
            target, reason = classify_record(raw[number], p['label'], contract)
            if reason:
                exclusions.append(dict(case_key=p['canonical_case_key'], partition=split,
                                       reason=reason, original_death_day=raw[number]['days_till_death']))
            else:
                patients.append(dict(p, survival_target=target))
        n_events = sum(p['survival_target']['event'] for p in patients)
        if len(patients) != contract['expected_cases'][split] or n_events != contract['expected_events'][split]:
            raise ValueError('Derived cohort counts differ from adopted413; no automatic correction')
        outputs[split] = dict(schema='KT2028-survival-manifest-1', endpoint=contract['endpoint'], partition=split,
                              clinical_source_sha256=clinical_sha, official_manifest_sha256=sha256_file(path),
                              contract=contract, contract_sha256=contract_hash(contract), patients=patients)
        summary[split] = dict(cases=len(patients), events=n_events, censored=len(patients)-n_events,
                              removed=len(doc['patients'])-len(patients))
    if set(outputs) != {'train', 'validate', 'test'} or len(seen) != 423 or len(exclusions) != 10:
        raise ValueError('Require all423 official cases and exactly10 registered exclusions')
    if Path(output).exists():
        raise FileExistsError('Preserve an existing derived cohort; use a new preparation directory')
    Path(output).mkdir(parents=True)
    for split, doc in outputs.items():
        (Path(output)/f'{split}_survival_manifest.json').write_text(json.dumps(doc, indent=2)+'\n')
    (Path(output)/'excluded_cases_private.json').write_text(json.dumps(exclusions, indent=2)+'\n')
    receipt = dict(schema='KT2028-survival-cohort-1', clinical_source_sha256=clinical_sha,
                   config_sha256=sha256_file(config), contract_sha256=contract_hash(contract), partitions=summary,
                   prepare_source_sha256=sha256_file(__file__),
                   total=413, official_sources_modified=False,
                   censoring_assumption=contract['censoring_basis'], test_scope='cohort preparation only;no prediction/selection',
                   manifest_sha256={s:sha256_file(Path(output)/f'{s}_survival_manifest.json') for s in outputs})
    (Path(output)/'cohort_summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clinical', required=True, type=Path)
    p.add_argument('--official-manifest', action='append', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(prepare(args.clinical, args.official_manifest, args.config, args.output), indent=2))


if __name__ == '__main__':
    main()
