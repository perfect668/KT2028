import copy
import hashlib
import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from kt2028.fast_module_network import (FastModulePredictor,JointRegionalPool,TransportPool,
                                        check_prototype_source)
from kt2028.survival_network import SurvivalPredictor
from kt2028.survival import encode_target,discrete_nll
from kt2028.observed_cache import CachedCase,RegionCache


class FastChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(41)
        self.source=Path(os.environ['OT_SOURCE'])
        self.proto=torch.randn(16,1024)
        self.arch=dict(dim=16,heads=4,dropout=.25,intervals=5,fusion='cross_gate')

    def model(self,arm):
        return FastModulePredictor(arm,prototypes=self.proto,ot_source=self.source,**self.arch)

    def inputs(self):
        return dict(coarse=torch.randn(4,1024),boxes=torch.tensor([[0.,0,448,448],[448,0,896,448],[0,448,448,896],[448,448,896,896]]),
            observed_fine={0:torch.randn(3,1024),2:torch.randn(2,1024)},relative_xy={0:torch.rand(3,2),2:torch.rand(2,2)},
            candidate_patch_count=16,slide_indices=torch.zeros(4,dtype=torch.long))

    def test_initialization_and_zero_observation_are_paired(self):
        case=self.inputs();case.update(observed_fine={},relative_xy={})
        torch.manual_seed(41);base=SurvivalPredictor(**self.arch).eval()
        for arm in ['joint','morphology','transport']:
            torch.manual_seed(41);model=self.model(arm).eval()
            for key,value in base.state_dict().items():
                if arm=='transport' and key.startswith('fine_pool.'):continue
                torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)
            torch.testing.assert_close(base(**case)['logits'],model(**case)['logits'],atol=0,rtol=0)

    def test_joint_preserves_all_coarse_and_renormalizes_observed_fine(self):
        pool=JointRegionalPool(8).eval()
        c=torch.randn(5,8);f=torch.randn(2,8);ids=torch.tensor([1,4])
        a,b=pool(c,f,ids);wc,wf=pool.last_weights
        torch.testing.assert_close(wc.sum(),torch.tensor(1.))
        torch.testing.assert_close(wf.sum(),torch.tensor(1.))
        torch.testing.assert_close(a,wc@c);torch.testing.assert_close(b,wf@f)
        self.assertTrue(torch.all(wc>0))
        perm=torch.tensor([4,0,3,1,2]);inverse={int(old):new for new,old in enumerate(perm)}
        x,y=pool(c[perm],f,torch.tensor([inverse[1],inverse[4]]))
        torch.testing.assert_close(a,x);torch.testing.assert_close(b,y)

    def test_auxiliary_fine_never_uses_coarse_values(self):
        x=self.inputs();changed=dict(x,coarse=x['coarse']*19)
        for arm in ['joint','morphology','transport']:
            model=self.model(arm).eval()
            a=model(**x,return_aux=True);b=model(**changed,return_aux=True)
            torch.testing.assert_close(a['fine_aux_logits'],b['fine_aux_logits'],atol=1e-6,rtol=1e-5)

    def test_real_branches_finite_gradients_and_strict_reload(self):
        target=encode_target(800,1,[0,365,730,1095,1460,1825]);case=self.inputs()
        for arm in ['joint','morphology','transport']:
            model=self.model(arm).train();out=model(**case,return_aux=True)
            loss=sum(discrete_nll(out[k],target) for k in ['logits','coarse_aux_logits','fine_aux_logits'])
            loss.backward();self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            model.eval();buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
            clone=self.model(arm).eval();clone.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
            torch.testing.assert_close(model(**case)['survival'],clone(**case)['survival'],atol=0,rtol=0)

    def test_ot_matches_pinned_source_at_full_mass_and_gradients(self):
        pool=TransportPool(8,self.source).eval()
        official=importlib.import_module('_kt2028_w154_ot_source.ot_attn').OT_Attn()
        a=torch.randn(11,8,requires_grad=True);b=torch.randn(16,8,requires_grad=True)
        x=a.detach().clone().requires_grad_(True);y=b.detach().clone().requires_grad_(True)
        actual=pool.transport(a,b);expected,_=official(x,y,iterations=10,iterations_per_epoch=1)
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)
        probe=torch.randn_like(actual);(actual*probe).sum().backward();(expected*probe).sum().backward()
        torch.testing.assert_close(a.grad,x.grad,atol=1e-6,rtol=1e-4)
        torch.testing.assert_close(b.grad,y.grad,atol=1e-6,rtol=1e-4)
        flat=pool.transport(torch.zeros(3,8),torch.zeros(16,8))
        self.assertTrue(torch.isfinite(flat).all())

    def test_prototypes_reject_other_training_patients(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'prototypes.npy';m=Path(d)/'provenance.json';rows=[{'canonical_case_key':'train-a'}]
            np.save(p,self.proto.numpy())
            doc=dict(fit_partition='this_inner_training_only',patients=1,
                training_ids_sha256=hashlib.sha256(json.dumps(['train-a']).encode()).hexdigest(),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            m.write_text(json.dumps(doc));torch.testing.assert_close(check_prototype_source(p,m,rows),self.proto)
            with self.assertRaises(AssertionError):check_prototype_source(p,m,[{'canonical_case_key':'heldout'}])

    def test_observed_cache_reads_only_requested_regions_and_zero_reads_none(self):
        real=self.inputs();requested=[]
        original=SimpleNamespace(coarse=real['coarse'].numpy(),boxes=real['boxes'].numpy(),counts=np.array([3,0,2,0]),
                                 slide_indices=real['slide_indices'].numpy())
        def read(ids):
            requested.append(list(ids));return dict(observed_fine={i:real['observed_fine'][i] for i in ids},relative_xy={i:real['relative_xy'][i] for i in ids})
        original.inputs=read
        cache=RegionCache(.01);wrapped=CachedCase(original,cache,'train-a')
        empty=wrapped.inputs([]);self.assertEqual(requested,[]);self.assertEqual(empty['observed_fine'],{})
        first=wrapped.inputs([2]);again=wrapped.inputs([2]);self.assertEqual(requested,[[2]])
        torch.testing.assert_close(first['observed_fine'][2],again['observed_fine'][2])
        both=wrapped.inputs([0,2]);self.assertEqual(requested,[[2],[0]])
        self.assertEqual(set(both['observed_fine']),{0,2})
        cache.limit=0;wrapped.inputs([2]);self.assertEqual(cache.bytes,0)


if __name__=='__main__':unittest.main()
