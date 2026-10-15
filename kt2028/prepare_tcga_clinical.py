"""Auditable candidate overall-survival table; preserve censoring and raw sources.

No split creation, model fitting, or replacement of the existing SurGen labels.
GDC times are relative to initial diagnosis. Alive before five years is censored,
never automatically labeled a five-year survivor.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .uni_encoder import sha256_file


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) and n >= 0 else None


def derive(case, images):
    demographic = case.get('demographic') or {}
    vital = demographic.get('vital_status')
    death = number(demographic.get('days_to_death'))
    follow_up = []
    for d in case.get('diagnoses', []):
        value = number(d.get('days_to_last_follow_up'))
        if value is not None:
            follow_up.append(dict(source='diagnoses.days_to_last_follow_up',
                                  entity_id=d.get('diagnosis_id'), days=value))
    for f in case.get('follow_ups', []):
        value = number(f.get('days_to_follow_up'))
        if value is not None:
            follow_up.append(dict(source='follow_ups.days_to_follow_up',
                                  entity_id=f.get('follow_up_id'), days=value))
    flags = []
    if demographic.get('age_is_obfuscated'):
        flags.append('age_obfuscated_timeline_may_be_compressed')
    reason, time_days, event = None, None, None
    if vital == 'Dead':
        if death is None:
            reason = 'dead_missing_valid_death_days'
        else:
            time_days, event = death, 1
            if follow_up and max(f['days'] for f in follow_up) > death:
                flags.append('follow_up_later_than_death_review')
    elif vital == 'Alive':
        if death is not None:
            reason = 'alive_with_death_days_conflict'
        elif follow_up:
            time_days, event = max(f['days'] for f in follow_up), 0
        else:
            reason = 'alive_missing_follow_up'
    else:
        reason = 'unknown_vital_status'
    if reason is None and time_days <= 0:
        reason = 'nonpositive_observation_time'
    if any(flag.endswith('_review') for flag in flags) and reason is None:
        reason = 'conflicting_time_records_review'
    eligible = reason is None
    # Never overwrite a missing/dead/short-follow-up outcome with survival to five years.
    label5 = None
    if eligible:
        if event and time_days <= 1825:
            label5 = 1
        elif time_days >= 1825:
            label5 = 0
    return dict(patient_id=case['submitter_id'], case_id=case['case_id'],
                project=case['project']['project_id'], file_ids=images,
                vital_status=vital, source_death_days=death, follow_up_records=follow_up,
                os_time_days=time_days, os_event=event, eligible_candidate=eligible,
                exclusion_reason=reason, flags=flags,
                time_days_5y=min(time_days, 1825) if eligible else None,
                event_5y=int(event == 1 and time_days <= 1825) if eligible else None,
                binary_death_5y_if_observable=label5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cases = json.loads(args.cases.read_text(encoding='utf-8-sig'))['data']['hits']
    samples = json.loads(args.samples.read_text(encoding='utf-8-sig'))
    links = {}
    for s in samples:
        if len(s['case_ids']) != 1 or len(s['patient_ids']) != 1:
            raise ValueError('Ambiguous file/patient mapping; preserve and resolve before building labels')
        links.setdefault(s['case_ids'][0], []).append(s['id'])
    mapping = {c['case_id']: c for c in cases}
    if len(mapping) != len(cases) or set(links)-set(mapping):
        raise ValueError('Duplicate or missing clinical case')
    rows = [derive(mapping[cid], ids) for cid, ids in sorted(links.items())]
    if len({r['patient_id'] for r in rows}) != len(rows):
        raise ValueError('Nonunique patient submitter IDs')
    summary = dict(schema='KT2028-TCGA-clinical-candidate-1', patients_with_images=len(rows),
                   image_files=len(samples), eligible_candidates=sum(r['eligible_candidate'] for r in rows),
                   full_os_events=sum(r['os_event'] == 1 and r['eligible_candidate'] for r in rows),
                   five_year_events=sum(r['event_5y'] == 1 for r in rows),
                   five_year_censored=sum(r['event_5y'] == 0 for r in rows),
                   five_year_binary_unobservable_among_eligible=sum(
                       r['eligible_candidate'] and r['binary_death_5y_if_observable'] is None for r in rows),
                   exclusions=dict(Counter(r['exclusion_reason'] for r in rows if not r['eligible_candidate'])),
                   flags=dict(Counter(flag for r in rows for flag in r['flags'])),
                   cases_sha256=sha256_file(args.cases), samples_sha256=sha256_file(args.samples),
                   code_sha256=sha256_file(Path(__file__)),
                   rule='Dead: demographic.days_to_death; Alive: maximum nonnegative diagnosis/follow-up days; '
                        'strictly positive duration; ambiguous records quarantined; original times preserved',
                   endpoint='overall survival from initial diagnosis; not cancer-specific survival',
                   references=['https://docs.gdc.cancer.gov/Data_Submission_Portal/Users_Guide/Best_Practices/',
                               'https://docs.gdc.cancer.gov/API/Release_Notes/API_Release_Notes/'],
                   no_patient_split_or_training=True, separate_from_surgen=True)
    args.output.mkdir(parents=True, exist_ok=True)
    table = args.output/'candidate_os.json'
    payload = json.dumps(rows, indent=2)
    if table.exists() and table.read_text() != payload:
        raise ValueError('Existing clinical derivation differs; preserve it')
    table.write_text(payload, encoding='utf-8', newline='\n')
    (args.output/'clinical_summary.json').write_text(json.dumps(summary, indent=2),
                                                   encoding='utf-8', newline='\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
