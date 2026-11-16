import unittest
from kt2028.run_r3_fivefold import make_splits, SHARDS, SPEC
from kt2028.run_multicancer import split_rows


class FiveFoldChecks(unittest.TestCase):
    def test_partitions_cover_once_and_preserve_original_outer(self):
        rows = [dict(canonical_case_key=f'case_{i}', cancer='CRC' if i < 75 else 'LUAD',
                     survival_target=dict(event=i % 3 == 0)) for i in range(100)]
        splits = make_splits(rows)
        for fold, split in enumerate(splits):
            original = split_rows(rows, fold, 41)[2]
            self.assertEqual({r['canonical_case_key'] for r in split['outer']},
                             {r['canonical_case_key'] for r in original if r['cancer'] == 'CRC'})
            self.assertTrue(all(r['cancer'] == 'CRC' for part in split.values() for r in part))
        self.assertEqual(sorted(f for fs in SHARDS.values() for f in fs), list(range(5)))

    def test_no_duplicate_patient_allowed(self):
        row = dict(canonical_case_key='duplicate', cancer='CRC', survival_target=dict(event=1))
        with self.assertRaises(AssertionError):
            make_splits([row, row])

    def test_locked_candidate(self):
        self.assertTrue(SPEC['identity_coarse_gate'] and SPEC['exposure_likelihood'])
        self.assertIsNone(SPEC['fine_attention_scale'])
        self.assertFalse(SPEC['normalize_fine_kv'])
