"""Run on the server with DSCA_SOURCE pointing to the pristine pinned source."""
import os
import copy
import unittest
from pathlib import Path
import torch
from kt2028.dsca_source_adapter import (SourceDSCA,source_network,pack_inputs,
    source_likelihood,source_optimizer,save_optimizer,load_optimizer)
from kt2028.survival import discrete_nll


class DSCASourceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41);torch.set_num_threads(1)
        self.source=Path(os.environ['DSCA_SOURCE'])

    def packed(self,n=3):
        return dict(coarse=torch.randn(1,n,1024),fine=torch.randn(1,n*16,1024),
                    coords=torch.tensor([[[i,0] for i in range(n)]]),valid=torch.ones(n,16,dtype=torch.bool))

    def test_full_grid_matches_unmodified_source_forward_and_backward(self):
        adapter=SourceDSCA(self.source,dropout=0.,dim=16).eval()
        raw=source_network(self.source,dropout=0.,dim=16).eval()
        raw.load_state_dict(adapter.network.state_dict(),strict=True)
        p=self.packed();actual=adapter.forward_packed(p)['hazards']
        expected=raw(p['fine'],p['coarse'],p['coords'])[0]
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
        actual.sum().backward();expected.sum().backward()
        for (n,a),(m,b) in zip(adapter.network.named_parameters(),raw.named_parameters()):
            self.assertEqual(n,m)
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)

    def test_masked_children_have_no_value_or_gradient_effect(self):
        model=SourceDSCA(self.source,dropout=0.,dim=16).eval();p=self.packed()
        p['valid'][0,1:]=False;p['valid'][1,:]=False
        a=model.forward_packed(p)['hazards']
        changed={k:v.clone() for k,v in p.items()}
        values=changed['fine'].reshape(3,16,1024);values[~p['valid']]=1e6
        changed['fine'].requires_grad_(True)
        b=model.forward_packed(changed)['hazards'];torch.testing.assert_close(a,b)
        b.sum().backward()
        self.assertEqual(float(changed['fine'].grad.reshape(3,16,1024)[~p['valid']].abs().sum()),0.)
        p['valid'][:]=False;self.assertTrue(torch.isfinite(model.forward_packed(p)['hazards']).all())

    def test_grid_slots_use_nominal_field_even_at_clipped_edge(self):
        model=SourceDSCA(self.source,dropout=0.,dim=16)
        features=torch.randn(2,1024)
        x=dict(coarse=torch.randn(1,1024),boxes=torch.tensor([[0.,0.,210.,448.]]),observed_fine={0:features},
            relative_xy={0:torch.tensor([[.125,.125],[.375,.125]])},candidate_patch_count=2,slide_indices=torch.tensor([0]))
        p=pack_inputs(x,model.official)
        self.assertEqual(p['valid'].nonzero().tolist(),[[0,0],[0,1]])
        torch.testing.assert_close(p['fine'][0,:2],features)
        x['relative_xy'][0][1]=x['relative_xy'][0][0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):pack_inputs(x,model.official)

    def test_censor_adapter_matches_current_nll_at_every_boundary(self):
        model=SourceDSCA(self.source,dropout=0.,dim=16)
        logits=torch.tensor([-.8,-.2,.1,.6,1.2])
        targets=[dict(event=1,event_bin=k,completed_intervals=k) for k in range(5)]
        targets += [dict(event=0,event_bin=-1,completed_intervals=k) for k in range(6)]
        hazards=logits.sigmoid().repeat(len(targets),1)
        source=model.official.SurvMLE(alpha=.1)
        actual=source_likelihood(source,hazards,targets,0.)
        expected=torch.stack([discrete_nll(logits,t) for t in targets]).mean()
        torch.testing.assert_close(actual,expected)
        weighted=torch.stack([discrete_nll(logits,t)*(1 if t['event'] else .9) for t in targets]).mean()
        torch.testing.assert_close(source_likelihood(source,hazards,targets,.1),weighted)

    def test_official_lookahead_restart_preserves_next_update(self):
        model=SourceDSCA(self.source,dropout=0.,dim=16)
        opt=source_optimizer(model,model.official)
        def step(m,o):
            o.base_optimizer.zero_grad(set_to_none=True)
            sum(p.square().sum() for p in m.parameters()).backward();o.step()
        for _ in range(7):step(model,opt)
        other=SourceDSCA(self.source,dropout=0.,dim=16);other.load_state_dict(model.state_dict(),strict=True)
        other_opt=source_optimizer(other,other.official)
        load_optimizer(other_opt,other,copy.deepcopy(save_optimizer(opt,model)),'cpu')
        for _ in range(5):step(model,opt);step(other,other_opt)
        for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
