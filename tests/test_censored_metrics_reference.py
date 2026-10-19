"""Analytic software fixtures checked against scikit-survival, not model results."""
import importlib.util
import unittest
import numpy as np
from kt2028.censored_metrics import CensoringKM, fit_censoring, survival_metrics
from tests.test_censored_metrics import row


@unittest.skipUnless(importlib.util.find_spec('sksurv'), 'Reference library unavailable')
class ReferenceMetricsTests(unittest.TestCase):
    def test_reverse_km_matches_reference_with_ties(self):
        from sksurv.nonparametric import kaplan_meier_estimator
        t=np.array([100,100,200,400,730,730.]); e=np.array([1,0,1,0,0,0],bool)
        x,g=kaplan_meier_estimator(e,t,reverse=True)
        own=CensoringKM(t,e)
        np.testing.assert_array_equal(own.times,x)
        np.testing.assert_allclose(own.after,g,rtol=0,atol=1e-15)

    def test_brier_auc_and_concordance_match_reference(self):
        from sksurv.metrics import brier_score,cumulative_dynamic_auc,concordance_index_censored
        from sksurv.util import Surv
        train=[row('a',100,1),row('b',100,0),row('c',200,1),row('d',400,0),row('e',730,0)]
        val=[row('v1',80,1),row('v2',100,1),row('v3',150,0),row('v4',500,1),row('v5',730,0)]
        curves=np.array([[.4,.1],[.5,.2],[.8,.6],[.9,.4],[.95,.8]])
        a=Surv.from_arrays([p['survival_target']['event'] for p in train],
                          [p['survival_target']['time_days'] for p in train])
        b=Surv.from_arrays([p['survival_target']['event'] for p in val],
                          [p['survival_target']['time_days'] for p in val])
        times=np.array([365,np.nextafter(730.,0)])
        _,bs=brier_score(a,b,curves,times)
        auc,_=cumulative_dynamic_auc(a,b,1-curves,times)
        m=survival_metrics(val,curves,[0,365,730],fit_censoring(train))
        for j,day in enumerate(('365','730')):
            self.assertAlmostEqual(m['by_day'][day]['brier'],bs[j],places=13)
            self.assertAlmostEqual(m['by_day'][day]['auroc'],auc[j],places=13)
        c=concordance_index_censored(b['event'],b['time'],-curves.mean(axis=1))
        self.assertAlmostEqual(m['c_index'],c[0],places=13)

    def test_closed_admin_endpoint_matches_shifted_censor_reference(self):
        from sksurv.metrics import brier_score,cumulative_dynamic_auc
        from sksurv.util import Surv
        train=[row('a',100,1),row('b',100,0),row('c',200,1),row('d',400,0),row('e',730,0)]
        val=[row('v1',80,1),row('v2',100,1),row('v3',150,0),row('v4',730,1),row('v5',730,0)]
        curves=np.array([[.4,.1],[.5,.2],[.8,.6],[.9,.4],[.95,.8]])
        def reference_y(rows):
            # Numeric encoding of observation through the CLOSED admin endpoint;
            # original death times and clinical source tables are not changed.
            e=np.array([p['survival_target']['event'] for p in rows],bool)
            t=np.array([p['survival_target']['time_days'] for p in rows])
            t[(~e)&(t==730)]=np.nextafter(730.,np.inf)
            return Surv.from_arrays(e,t)
        a,b=reference_y(train),reference_y(val)
        _,bs=brier_score(a,b,curves,[365,730])
        auc,_=cumulative_dynamic_auc(a,b,1-curves,[365,730])
        m=survival_metrics(val,curves,[0,365,730],fit_censoring(train))
        for j,day in enumerate(('365','730')):
            self.assertAlmostEqual(m['by_day'][day]['brier'],bs[j],places=13)
            self.assertAlmostEqual(m['by_day'][day]['auroc'],auc[j],places=13)


if __name__=='__main__': unittest.main()
