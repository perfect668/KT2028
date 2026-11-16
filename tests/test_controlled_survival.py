"""Guard patient isolation, Train-only preprocessing and the convex objective."""
import numpy as np
import torch
from kt2028.run_controlled_survival import inner_folds, transform_fit, fit_ridge, cancer_pools
from kt2028.open_survival_adapter import cox_breslow


def main():
    torch.set_num_threads(2)
    rows=[dict(canonical_case_key=f'private-{i}',survival_target=dict(event=int(i%3==0),time_days=i+1)) for i in range(60)]
    splits=inner_folds(rows);assert splits==inner_folds(rows)
    for tr,va in splits:assert not set(tr)&set(va) and len(set(tr+va))==60
    assert sorted(i for _,va in splits for i in va)==list(range(60))
    assert sorted(sum(rows[i]['survival_target']['event'] for i in va) for _,va in splits)==[6,7,7]
    mixed=[dict(r,cancer='CRC' if i<25 else 'LUAD') for i,r in enumerate(rows)]
    pools=cancer_pools(mixed)
    assert [len(pools[c]) for c in ['CRC','LUAD']]==[25,35]
    for c,pool in pools.items():
        assert all(r['cancer']==c for r in pool)
        for tr,va in inner_folds(pool):assert {pool[i]['canonical_case_key'] for i in tr+va}=={r['canonical_case_key'] for r in pool}
    x=np.random.default_rng(4).normal(size=(20,12));tr=list(range(15));y,a=transform_fit(x,tr,4)
    altered=x.copy();altered[15:]*=1000;z,b=transform_fit(altered,tr,4)
    np.testing.assert_allclose(y[tr],z[tr]);np.testing.assert_allclose(a['scaler'].mean_,b['scaler'].mean_)
    np.testing.assert_allclose(a['pca'].components_,b['pca'].components_)
    rr=[dict(survival_target=dict(time_days=float(i+1),event=1)) for i in range(12)]
    xx=np.arange(12,dtype=float).reshape(-1,1)/12
    weak,w=fit_ridge(xx,rr,.001);strong,s=fit_ridge(xx,rr,.1)
    assert weak[0]<0 and strong[0]<0 and np.linalg.norm(strong)<np.linalg.norm(weak)
    assert max(w['gradient_inf'],s['gradient_inf'])<1e-5
    risk=torch.tensor([.1,.3,-.2],dtype=torch.float64);t=torch.tensor([1.,2.,3.]);e=torch.ones(3)
    torch.testing.assert_close(cox_breslow(risk,t,e),cox_breslow(risk+37,t,e))
    print('Patient isolation/event balance, exact cancer pool isolation, Train-only scaling/PCA, convex gradient/convergence/shrinkage and Cox shift invariance: PASSED')


if __name__=='__main__':main()
