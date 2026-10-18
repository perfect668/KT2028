"""Optional Torch software checks; synthetic tensors are never experiment results."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch unavailable in local document runtime')
class MultiSlideTests(unittest.TestCase):
    def inputs(self):
        import torch
        return dict(coarse=torch.randn(4,1024),boxes=torch.tensor(
            [[0,0,448,448],[448,0,896,448],[0,0,448,448],[448,0,896,448]],dtype=torch.float32),
            observed_fine={0:torch.randn(2,1024),2:torch.randn(3,1024)},
            relative_xy={0:torch.rand(2,2),2:torch.rand(3,2)},candidate_patch_count=5)

    def test_single_slide_membership_preserves_previous_output(self):
        import torch
        from kt2028.survival_network import SurvivalPredictor
        torch.manual_seed(41); m=SurvivalPredictor(dim=32,heads=4,dropout=0).eval()
        x=self.inputs()
        with torch.no_grad():
            a=m(**x)['logits']; b=m(**x,slide_indices=torch.zeros(4,dtype=torch.int64))['logits']
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_coarse_context_does_not_mix_unrelated_slide_coordinates(self):
        import torch
        from kt2028.survival_network import SurvivalPredictor
        torch.manual_seed(41); m=SurvivalPredictor(dim=32,heads=4,dropout=0).eval()
        x=self.inputs(); ids=torch.tensor([0,0,1,1])
        with torch.no_grad():
            a=m(**x,slide_indices=ids)['coarse_tokens']
            x['coarse'][2:]+=10
            b=m(**x,slide_indices=ids)['coarse_tokens']
        torch.testing.assert_close(a[:2],b[:2],atol=0,rtol=0)
        self.assertFalse(torch.equal(a[2:],b[2:]))


if __name__=='__main__': unittest.main()
