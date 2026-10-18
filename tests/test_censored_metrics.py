import unittest
import numpy as np
from kt2028.censored_metrics import CensoringKM, fit_censoring, survival_metrics
from kt2028.prepare_mixed_survival import split_patients, target


def row(key, t, e, cohort='a'):
    return dict(canonical_case_key=key, cohort=cohort, slides=[dict(slide_id=key)],
                survival_target=target(t,e,[0,365,730]))


class CensoringTests(unittest.TestCase):
    def test_reverse_km_ties(self):
        km = CensoringKM([10,10,20,30],[1,0,1,0])
        self.assertEqual(km.probability(10,left=True),1)
        self.assertAlmostEqual(km.probability(10),2/3)
        self.assertIsNone(km.probability(31))

    def test_common_admin_matches_plain_brier(self):
        rows=[row('a',100,1),row('b',500,1),row('c',730,0),row('d',730,0)]
        s=np.array([[.2,.1],[.8,.2],[.9,.8],[.7,.5]])
        m=survival_metrics(rows,s,[0,365,730],fit_censoring(rows))
        for j, y in enumerate(([1,0,0,0],[1,1,0,0])):
            self.assertAlmostEqual(m['by_day'][str([365,730][j])]['brier'],np.mean((1-s[:,j]-y)**2))
        self.assertEqual(m['by_day']['730']['known_survivors'],2)

    def test_early_censor_not_five_year_survivor(self):
        train=[row('a',100,0),row('b',500,1),row('c',730,0)]
        val=[row('d',100,0),row('e',200,1),row('f',730,0)]
        m=survival_metrics(val,np.full((3,2),.5),[0,365,730],fit_censoring(train))
        self.assertEqual(m['by_day']['365']['censored_before_endpoint'],1)
        self.assertAlmostEqual(m['by_day']['365']['brier'],.25)
        self.assertEqual(m['by_day']['365']['auroc'],.5)

    def test_unsupported_is_missing_not_clipped(self):
        train=[row('a',100,0),row('b',200,0)]
        val=[row('c',500,1),row('d',730,0)]
        m=survival_metrics(val,np.full((2,2),.5),[0,365,730],fit_censoring(train))
        self.assertIsNone(m['by_day']['730']['brier'])
        self.assertGreater(m['by_day']['730']['unsupported_ipcw_observations'],0)

    def test_source_specific_censoring(self):
        rows=[row('a',100,0),row('b',730,0),row('c',730,0,'b')]
        km=fit_censoring(rows)
        self.assertEqual(km['b'].probability(365),1)
        self.assertEqual(km['a'].probability(365),.5)

    def test_split_is_patient_disjoint_and_order_invariant(self):
        rows=[row(str(i),100 if i%2 else 730,i%2, 'a' if i<40 else 'b') for i in range(80)]
        a,b=split_patients(rows,41,.1)
        c,d=split_patients(rows[::-1],41,.1)
        self.assertEqual(a,c); self.assertEqual(b,d)
        self.assertEqual(len(b),8)
        self.assertFalse({p['canonical_case_key'] for p in a}&{p['canonical_case_key'] for p in b})


if __name__=='__main__':
    unittest.main()
