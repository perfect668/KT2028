import unittest
from kt2028.prepare_tcga_clinical import derive


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


if __name__ == '__main__':
    unittest.main()
