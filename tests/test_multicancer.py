import unittest
import numpy as np
import torch
from kt2028.multicancer_network import CancerPredictor
from kt2028.run_multicancer import split_rows
from kt2028.fusion_data import move


class MulticancerChecks(unittest.TestCase):
    def test_nested_split_patient_isolation_and_outer_coverage(self):
        rows=[dict(canonical_case_key=f'{c}-{i}',cancer=c,survival_target={'event':i%2})
              for c in ('CRC','STAD','LUAD','PAAD') for i in range(30)]
        outer=[]
        for fold in range(5):
            tr,va,te=split_rows(rows,fold,41)
            self.assertEqual(len(tr)+len(va)+len(te),len(rows))
            outer += [r['canonical_case_key'] for r in te]
            self.assertEqual(split_rows(rows,fold,41),(tr,va,te))
        self.assertEqual(len(set(outer)),len(rows))
        self.assertEqual(len(outer),len(rows))

    def test_conditioned_heads_and_gradients(self):
        torch.set_num_threads(2)
        inputs=dict(coarse=torch.randn(3,1024),boxes=torch.tensor([[0,0,448,448],[448,0,896,448],[0,448,448,896.]]),
            observed_fine={0:torch.randn(2,1024),1:torch.randn(1,1024)},relative_xy={0:torch.rand(2,2),1:torch.rand(1,2)},
            candidate_patch_count=3,cancer_id=2)
        for kind in ('shared','moe'):
            model=CancerPredictor(kind,dim=16,heads=4,dropout=0.).eval()
            out=model(**move(inputs,'cpu'),return_aux=True)
            self.assertEqual(tuple(out['logits'].shape),(5,))
            self.assertTrue(torch.all(out['survival'][1:]<=out['survival'][:-1]))
            (out['logits'].sum()+out['fine_aux_logits'].sum()+out['coarse_aux_logits'].sum()).backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            if kind=='moe':
                route=np.array(model.routing('cpu'))
                np.testing.assert_allclose(route.sum(1),1,atol=1e-6)
                self.assertTrue((route>0).all())


if __name__=='__main__':unittest.main()
