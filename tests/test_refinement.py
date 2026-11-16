import copy
import io
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from kt2028.refined_survival_network import RefinedSurvivalPredictor
from kt2028.survival_network import SurvivalPredictor
from kt2028.exposure_survival import exposure_nll
from kt2028.survival import encode_target,discrete_nll,curve_from_logits


class RefinementChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(41)
        self.arch=dict(dim=16,heads=4,dropout=.25,intervals=5,fusion='cross_gate')
        self.edges=[0,365,730,1095,1460,1825]
        self.case=dict(coarse=torch.randn(5,1024),boxes=torch.tensor([[i*448.,0,(i+1)*448.,448] for i in range(5)]),
            observed_fine={0:torch.randn(3,1024),3:torch.randn(2,1024)},
            relative_xy={0:torch.rand(3,2),3:torch.rand(2,2)},candidate_patch_count=16,
            slide_indices=torch.tensor([0,0,0,1,1],dtype=torch.long))

    def test_disabled_repairs_are_bitwise_legacy(self):
        torch.manual_seed(41);a=SurvivalPredictor(**self.arch).eval()
        torch.manual_seed(41);b=RefinedSurvivalPredictor(**self.arch).eval()
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
        for empty in [False,True]:
            x=dict(self.case,observed_fine={},relative_xy={}) if empty else self.case
            torch.testing.assert_close(a(**x)['logits'],b(**x)['logits'],atol=0,rtol=0)

    def test_neutral_gate_preserves_coarse_and_is_trainable(self):
        m=RefinedSurvivalPredictor(identity_coarse_gate=True,**self.arch).eval()
        a=m(**self.case);zero=m(**dict(self.case,observed_fine={},relative_xy={}))
        torch.testing.assert_close(a['coarse_gates'],torch.ones_like(a['coarse_gates']),atol=0,rtol=0)
        torch.testing.assert_close(a['coarse_patient'],zero['coarse_patient'],atol=0,rtol=0)
        a['logits'].square().sum().backward()
        self.assertTrue(m.coarse_gate[0].weight.grad.abs().sum()>0)

    def test_scaled_attention_exact_one_and_fused_path_respects_scale(self):
        torch.manual_seed(41);a=SurvivalPredictor(**self.arch).eval()
        torch.manual_seed(41);b=RefinedSurvivalPredictor(fine_attention_scale=1.,**self.arch).eval()
        for training in [False,True]:
            a.train(training);b.train(training)
            torch.manual_seed(99);x=a(**self.case,return_aux=True)
            torch.manual_seed(99);y=b(**self.case,return_aux=True)
            torch.testing.assert_close(x['logits'],y['logits'],atol=1e-6,rtol=1e-5)
        b.eval();layer=b.fine_encoder.layers[0];value=torch.randn(1,7,16)
        with torch.no_grad():
            layer.attention_scale.zero_()
            expected=value+layer._ff_block(layer.norm2(value))
            for fastpath in [True,False]:
                torch.backends.mha.set_fastpath_enabled(fastpath)
                torch.testing.assert_close(b.fine_encoder(value),expected,atol=1e-6,rtol=1e-5)
        torch.backends.mha.set_fastpath_enabled(True)

    def test_all_repairs_gradient_reload_and_auxiliary_isolation(self):
        m=RefinedSurvivalPredictor(identity_coarse_gate=True,fine_attention_scale=.1,normalize_fine_kv=True,**self.arch).train()
        before=copy.deepcopy(m.state_dict());a=m(**self.case,return_aux=True)
        target=encode_target(800,1,self.edges)
        sum(discrete_nll(a[k],target) for k in ['logits','coarse_aux_logits','fine_aux_logits']).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertTrue(m.fine_encoder.layers[0].attention_scale.grad.abs().sum()>0)
        self.assertTrue(m.fine_kv_norm.weight.grad.abs().sum()>0)
        for k,v in before.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
        m.eval();buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
        clone=RefinedSurvivalPredictor(identity_coarse_gate=True,fine_attention_scale=.1,normalize_fine_kv=True,**self.arch).eval()
        clone.load_state_dict(torch.load(buf,weights_only=True),strict=True)
        torch.testing.assert_close(m(**self.case)['survival'],clone(**self.case)['survival'],atol=0,rtol=0)
        a=m(**self.case,return_aux=True);b=m(**dict(self.case,coarse=self.case['coarse']*17),return_aux=True)
        torch.testing.assert_close(a['fine_aux_logits'],b['fine_aux_logits'],atol=1e-6,rtol=1e-5)

    def test_padding_and_permutation_with_kv_normalization(self):
        m=RefinedSurvivalPredictor(fine_attention_scale=.1,normalize_fine_kv=True,**self.arch).eval()
        q=torch.randn(2,1,16);v=torch.randn(2,4,16);mask=torch.tensor([[False,False,True,True],[False,False,False,True]])
        pos=torch.randn(2,16);a=m.encode_fine(q,v,mask,pos)
        changed=v.clone();changed[mask]=10000
        torch.testing.assert_close(a,m.encode_fine(q,changed,mask,pos),atol=0,rtol=0)
        x=dict(self.case,observed_fine={i:v.flip(0) for i,v in self.case['observed_fine'].items()},
               relative_xy={i:v.flip(0) for i,v in self.case['relative_xy'].items()})
        torch.testing.assert_close(m(**self.case)['logits'],m(**x)['logits'],atol=1e-6,rtol=1e-5)

    def test_exposure_matches_exponential_density_and_censoring_boundaries(self):
        rate=.7
        l=torch.full((5,),float(np.log(np.expm1(rate))),dtype=torch.float64)
        for time in [1,120,365,365.1,730,800,1460,1825]:
            for event in [0,1]:
                target=encode_target(time,event,self.edges)
                expected=rate*time/365-event*np.log(rate)
                self.assertAlmostEqual(float(exposure_nll(l,target,self.edges)),expected,places=11)
                curve=curve_from_logits(l)[1]
                torch.testing.assert_close(curve,torch.exp(-rate*torch.arange(1,6,dtype=torch.float64)))

    def test_early_censor_now_has_gradient_and_extremes_are_finite(self):
        target=encode_target(120,0,self.edges);l=torch.zeros(5,requires_grad=True)
        self.assertEqual(float(discrete_nll(l,target)),0.)
        loss=exposure_nll(l,target,self.edges);g,=torch.autograd.grad(loss,l)
        self.assertGreater(float(loss),0);self.assertGreater(float(g[0]),0);self.assertEqual(int(torch.count_nonzero(g)),1)
        for value in [-1000.,-100.,0.,100.,1000.]:
            l=torch.full((5,),value,requires_grad=True)
            loss=exposure_nll(l,encode_target(800,1,self.edges),self.edges)
            g,=torch.autograd.grad(loss,l)
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(g).all())


if __name__=='__main__':unittest.main()
