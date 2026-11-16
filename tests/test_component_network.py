"""Source parity, spatial isolation and censored-output invariants for W151."""
import copy
import importlib.util
import io
import os
import unittest
from pathlib import Path

import torch

from kt2028.component_network import (ComponentPredictor, SpatialContext, SubspacePool,
                                     lattice, make_predictor, mixture_logits)
from kt2028.survival_network import SurvivalPredictor
from kt2028.survival import discrete_nll, encode_target, curve_from_logits


class ComponentChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.source = Path(os.environ['EAGLE_SOURCE'])
        self.arch = dict(dim=16, heads=4, dropout=.25, intervals=5, fusion='cross_gate')

    def case(self, full=True):
        boxes = torch.tensor([[0,0,448,448], [448,0,896,448], [0,896,448,1344],
                              [0,0,448,448], [448,448,896,896]], dtype=torch.float32)
        ids = [0,2,3] if full else []
        return dict(coarse=torch.randn(5,1024), boxes=boxes,
                    observed_fine={i:torch.randn(2,1024) for i in ids},
                    relative_xy={i:torch.rand(2,2) for i in ids},
                    candidate_patch_count=12, slide_indices=torch.tensor([0,0,0,1,1]))

    def test_factory_baseline_is_original_and_initial_parameters_paired(self):
        cfg=dict(architecture=self.arch, mase_tile=3)
        torch.manual_seed(41); original=SurvivalPredictor(**self.arch)
        torch.manual_seed(41); base=make_predictor('baseline',cfg,self.source)
        torch.manual_seed(41); sub=make_predictor('subspace',cfg,self.source)
        self.assertIs(type(base), SurvivalPredictor)
        for key,value in original.state_dict().items():
            torch.testing.assert_close(base.state_dict()[key], value, atol=0, rtol=0)
            torch.testing.assert_close(sub.state_dict()[key], value, atol=0, rtol=0)
        torch.manual_seed(41); spatial=make_predictor('mase',cfg,self.source)
        for key,value in base.state_dict().items():
            if not key.startswith('coarse_encoder.'):
                torch.testing.assert_close(spatial.state_dict()[key],value,atol=0,rtol=0)

    def test_parameter_budget(self):
        arch=dict(self.arch,dim=256)
        base=SurvivalPredictor(**arch)
        sub=ComponentPredictor('subspace',**arch)
        mase=ComponentPredictor('mase',self.source,**arch)
        count=lambda m:sum(p.numel() for p in m.parameters())
        self.assertEqual(count(base),2340615)
        self.assertEqual(count(sub),2340615)
        self.assertEqual(count(mase),1817544)

    def test_nn_attention_and_gradients_match_official_on_full_values(self):
        path=Path(os.environ['NNMIL_SOURCE'])/'network_architecture/models/simple_mil.py'
        spec=importlib.util.spec_from_file_location('official_nn_for_test',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        pool=SubspacePool(256).eval()
        official=mod.SimpleMIL(256,128,1,dropout=False).eval()
        for a,b in [(official.V,pool.a),(official.U,pool.b),(official.w,pool.score)]:
            a.load_state_dict(b.state_dict())
        indices=pool.indices(torch.zeros(7,256))
        expected_indices=official._cover_indices(256,128,'cpu')
        self.assertEqual(len(indices),5)
        self.assertEqual(len(torch.unique(torch.cat(indices))),256)
        for a,b in zip(indices,expected_indices):torch.testing.assert_close(a,b)
        x=torch.randn(7,256,requires_grad=True)
        y=x.detach().clone().requires_grad_(True)
        weights=pool.attend(x,indices[0])
        reference=official._attend(y[None],indices[0])[0].T
        torch.testing.assert_close(weights,reference,atol=1e-7,rtol=1e-6)
        actual=weights@x;expected=reference@y
        torch.testing.assert_close(actual,expected)
        probe=torch.randn_like(actual)
        (actual*probe).sum().backward();(expected*probe).sum().backward()
        torch.testing.assert_close(x.grad,y.grad)
        for a,b in [(pool.a,official.V),(pool.b,official.U),(pool.score,official.w)]:
            torch.testing.assert_close(a.weight.grad,b.weight.grad)

    def test_spatial_gather_matches_official_and_respects_physical_holes(self):
        spatial=SpatialContext(self.source,tile=2).eval()
        boxes=torch.tensor([[0,0,448,448],[896,448,1344,896]],dtype=torch.float32)
        values=torch.randn(2,16,requires_grad=True)
        canvas=torch.zeros(2,3,16);canvas[0,0]=values[0];canvas[1,2]=values[1]
        expected=spatial.mase(canvas[None,None])[0,0]
        actual=spatial(values,boxes,None)
        torch.testing.assert_close(actual,expected[[0,1],[0,2]],atol=2e-6,rtol=2e-5)
        with self.assertRaises(ValueError):lattice(boxes[[0,0]])
        with self.assertRaises(ValueError):lattice(boxes+torch.tensor([[0,0,0,0],[3,0,3,0]]))

    def test_spatial_slides_are_independent(self):
        spatial=SpatialContext(self.source,tile=3).eval()
        case=self.case()
        x=torch.randn(5,16)
        expected=spatial(x,case['boxes'],case['slide_indices'])
        changed=x.clone();changed[3:]+=100
        actual=spatial(changed,case['boxes'],case['slide_indices'])
        torch.testing.assert_close(actual[:3],expected[:3],atol=0,rtol=0)

    def test_coordinate_permutation_keeps_predictions(self):
        case=self.case();perm=torch.tensor([3,2,0,4,1])
        inverse={int(old):new for new,old in enumerate(perm)}
        changed=dict(case,coarse=case['coarse'][perm],boxes=case['boxes'][perm],
            slide_indices=case['slide_indices'][perm],
            observed_fine={inverse[i]:v for i,v in case['observed_fine'].items()},
            relative_xy={inverse[i]:v for i,v in case['relative_xy'].items()})
        for arm in ['mase','subspace']:
            model=ComponentPredictor(arm,self.source,tile=3,**self.arch).eval()
            torch.testing.assert_close(model(**case)['survival'],model(**changed)['survival'],atol=2e-6,rtol=2e-5)

    def test_zero_subspace_matches_original_and_aux_curves_reload(self):
        torch.manual_seed(41);base=SurvivalPredictor(**self.arch).eval()
        torch.manual_seed(41);sub=ComponentPredictor('subspace',**self.arch).eval()
        zero=self.case(False)
        torch.testing.assert_close(base(**zero)['logits'],sub(**zero)['logits'],atol=0,rtol=0)
        case=self.case()
        for arm in ['mase','subspace']:
            model=ComponentPredictor(arm,self.source,tile=3,**self.arch).eval()
            result=model(**case,return_aux=True)
            buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
            clone=ComponentPredictor(arm,self.source,tile=3,**self.arch).eval()
            clone.load_state_dict(torch.load(buf,weights_only=True),strict=True)
            torch.testing.assert_close(result['survival'],clone(**case)['survival'],atol=0,rtol=0)
            self.assertTrue(torch.all(result['survival'][1:]<=result['survival'][:-1]))
            self.assertIsNotNone(result['fine_aux_logits'])

    def test_mixture_is_average_survival_and_censor_likelihood(self):
        for source in [torch.randn(5,5),torch.tensor([[1000.,-1000.,0.,-1000.,1000.]]*5)]:
            actual=mixture_logits(source)
            _,curve=curve_from_logits(actual)
            expected=curve_from_logits(source.double())[1].mean(0)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(curve,expected,atol=2e-15,rtol=2e-13)
            for day in [1,365,366,1825]:
                for event in [0,1]:
                    target=encode_target(day,event,[0,365,730,1095,1460,1825])
                    loss=discrete_nll(actual,target)
                    self.assertTrue(torch.isfinite(loss))
                    if not event and day<365:self.assertEqual(float(loss),0.)

    def test_training_objective_reaches_all_active_modules(self):
        target=encode_target(800,1,[0,365,730,1095,1460,1825])
        for arm in ['mase','subspace']:
            model=ComponentPredictor(arm,self.source,tile=3,**self.arch).train()
            out=model(**self.case(),return_aux=True)
            loss=sum(discrete_nll(out[k],target) for k in ['logits','coarse_aux_logits','fine_aux_logits'])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertIsNone(model._spatial_boxes)


if __name__=='__main__':unittest.main()
