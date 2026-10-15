import unittest
from kt2028.prepare_tcga_clinical import derive, link_biospecimens


def case(vital, death=None, follow=None):
    return dict(submitter_id='test', case_id='case', project={'project_id':'TCGA-COAD'},
                demographic=dict(vital_status=vital, days_to_death=death),
                diagnoses=[dict(days_to_last_follow_up=follow)], follow_ups=[])


class ClinicalCensoring(unittest.TestCase):
    def test_short_followup_is_not_five_year_survival(self):
        r = derive(case('Alive', follow=400), ['slide'])
        self.assertTrue(r['eligible_candidate'])
        self.assertEqual((r['time_days_5y'], r['event_5y']), (400, 0))
        self.assertIsNone(r['binary_death_5y_if_observable'])

    def test_late_death_is_censored_at_horizon(self):
        r = derive(case('Dead', death=2000, follow=1500), ['slide'])
        self.assertEqual((r['os_time_days'], r['os_event']), (2000, 1))
        self.assertEqual((r['time_days_5y'], r['event_5y']), (1825, 0))

    def test_no_imputation_for_missing_followup(self):
        self.assertFalse(derive(case('Alive'), ['slide'])['eligible_candidate'])

    def test_zero_death_is_quarantined(self):
        self.assertEqual(derive(case('Dead', death=0), ['slide'])['exclusion_reason'],
                         'nonpositive_observation_time')

    def test_updated_followup_used(self):
        c = case('Alive', follow=100)
        c['follow_ups'] = [dict(days_to_follow_up=300)]
        self.assertEqual(derive(c, ['slide'])['os_time_days'], 300)


class BiospecimenLink(unittest.TestCase):
    def setUp(self):
        self.sample = dict(id='file', filename='TCGA-AA-0001-01Z-00-DX1.test.svs',
                           patient_ids=['patient'], case_ids=['case'])
        self.hit = dict(file_id='file', file_name=self.sample['filename'],
            associated_entities=[dict(entity_type='slide', entity_id='slide', entity_submitter_id='barcode')],
            cases=[dict(case_id='case', submitter_id='patient', samples=[
                dict(sample_id='normal', sample_type='Solid Tissue Normal', portions=[]),
                dict(sample_id='tumor', sample_type='Primary Tumor', portions=[
                    dict(slides=[dict(slide_id='slide', submitter_id='barcode')])])])])

    def test_follow_associated_slide_not_first_sample(self):
        row = link_biospecimens([self.hit], [self.sample])[0]
        self.assertEqual(row['specimen_id'], 'tumor')
        self.assertTrue(row['primary_diagnostic_candidate'])

    def test_disagreeing_barcode_rejected(self):
        self.hit['cases'][0]['samples'][1]['sample_type'] = 'Metastatic'
        with self.assertRaises(ValueError):
            link_biospecimens([self.hit], [self.sample])

    def test_metastatic_is_kept_but_flagged(self):
        self.sample['filename'] = self.sample['filename'].replace('-01Z-', '-06Z-')
        self.hit['file_name'] = self.sample['filename']
        self.hit['cases'][0]['samples'][1]['sample_type'] = 'Metastatic'
        row = link_biospecimens([self.hit], [self.sample])[0]
        self.assertFalse(row['primary_diagnostic_candidate'])
        self.assertTrue(row['review_flags'])


if __name__ == '__main__':
    unittest.main()
