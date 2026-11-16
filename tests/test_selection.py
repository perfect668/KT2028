import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from kt2028.selection_network import GatedLocalAttention,RegionSelector,choose,predictor


class SelectionTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(41);torch.set_num_threads(1)

    def test_open_gate_reproduces_attention_and_padding_is_masked(self):
        base=torch.nn.MultiheadAttention(16,4,batch_first=True,dropout=0.).eval()
        gated=GatedLocalAttention.from_attention(base).eval()
        with torch.no_grad():gated.output_gate.bias.fill_(30)
        q=torch.randn(2,1,16);kv=torch.randn(2,4,16);mask=torch.tensor([[False,False,True,True],[False,True,True,True]])
        expected=base(q,kv,kv,key_padding_mask=mask,need_weights=False)[0]
        torch.testing.assert_close(gated(q,kv,kv,key_padding_mask=mask)[0],expected,atol=1e-6,rtol=1e-5)
        changed=kv.clone();changed[mask]=10000
        torch.testing.assert_close(gated(q,changed,changed,key_padding_mask=mask)[0],expected,atol=1e-6,rtol=1e-5)

    def test_budget_eligibility_and_log_probability(self):
        scores=torch.tensor([.2,.7,-4.,1.3],requires_grad=True);counts=torch.tensor([1,2,0,3])
        ids,lp,en=choose(scores,counts,.5,False)
        self.assertEqual(ids,[3,1]);self.assertEqual(len(set(ids)),2)
        expected=scores[3]-torch.logsumexp(scores[[0,1,3]],0)+scores[1]-torch.logsumexp(scores[[0,1]],0)
        torch.testing.assert_close(lp,expected)
        (-lp-.001*en).backward();self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(float(scores.grad[2]),0.)

    def test_selector_only_updates_selector(self):
        policy=RegionSelector(16);tokens=torch.randn(5,16,requires_grad=True);patient=torch.randn(16,requires_grad=True)
        boxes=torch.tensor([[i*448.,0,(i+1)*448.,448] for i in range(5)])
        scores=policy(tokens,patient,boxes,torch.tensor([1,2,0,4,5]),torch.zeros(5,dtype=torch.long))
        ids,lp,entropy=choose(scores,torch.tensor([1,2,0,4,5]),.5,True)
        (-lp-.001*entropy).backward()
        self.assertIsNone(tokens.grad);self.assertIsNone(patient.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()))
        self.assertNotIn(2,ids)

    def test_parameter_count_and_strict_gate_reload(self):
        plain=predictor(dim=128,heads=4);gated=predictor(True,dim=128,heads=4)
        self.assertEqual(sum(p.numel() for p in plain.parameters()),761351)
        self.assertEqual(sum(p.numel() for p in gated.parameters()),761351+516)
        clone=predictor(True,dim=128,heads=4);clone.load_state_dict(gated.state_dict(),strict=True)

    def test_completed_restart_skips_training_and_rejects_changed_checkpoint(self):
        from kt2028.run_selection_ablation import warmup,completed_folder
        from kt2028.uni_encoder import sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'runs'/'CRC_r3_128';folder.mkdir(parents=True)
            checkpoint=folder/'best.pt';checkpoint.write_bytes(b'completed checkpoint')
            (folder/'completion.json').write_text(json.dumps(dict(checkpoint_sha256=sha256_file(checkpoint),outer_used=False,development_only=True)))
            with patch('kt2028.run_selection_ablation.predictor',side_effect=AssertionError('Must not rebuild completed model')):
                self.assertEqual(warmup(root,'CRC',None,None,None,None),folder)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):completed_folder(root,'CRC_r3_128')


if __name__=='__main__':unittest.main()
