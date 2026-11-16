import json
import tempfile
import unittest
from pathlib import Path

import torch

from kt2028.repaired_model import load_repaired_model,digest
from kt2028.refined_survival_network import RefinedSurvivalPredictor


class ArtifactChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(41)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.arch=dict(dim=16,heads=4,dropout=.25,intervals=5,fusion='cross_gate')
        self.spec=dict(identity_coarse_gate=True,fine_attention_scale=.1,normalize_fine_kv=True,exposure_likelihood=False)
        self.model=RefinedSurvivalPredictor(**{k:v for k,v in self.spec.items() if k!='exposure_likelihood'},**self.arch).eval()
        torch.save(dict(model=self.model.state_dict(),epoch=4,spec=self.spec),self.root/'best.pt')
        self.proof=dict(status='final_verified',retained_stage_replayed='residual',selected_spec=self.spec,
            architecture=self.arch,interval_edges_days=[0,365,730,1095,1460,1825],feature_protocol={'feature_dimension':1024},
            baseline_fresh_reload={'checkpoint_sha256':'baseline-unused'},final_fresh_reload={'checkpoint_sha256':digest(self.root/'best.pt')})
        (self.root/'proof.json').write_text(json.dumps(self.proof))
        source=Path(__file__).resolve().parents[1]/'kt2028'
        self.manifest=dict(schema='KT2028-verified-r3-repair-1',stage='residual',refinement=self.spec,
            architecture=self.arch,interval_edges_days=self.proof['interval_edges_days'],feature_protocol=self.proof['feature_protocol'],
            checkpoint='best.pt',checkpoint_sha256=digest(self.root/'best.pt'),verification_receipt='proof.json',
            verification_receipt_sha256=digest(self.root/'proof.json'),network_source_sha256={n:digest(source/n) for n in ['repaired_model.py','refined_survival_network.py','survival_network.py']})
        self.write()

    def tearDown(self):self.tmp.cleanup()
    def write(self):(self.root/'manifest.json').write_text(json.dumps(self.manifest))

    def test_loads_exact_eval_model_and_registered_contract(self):
        model,contract=load_repaired_model(self.root/'manifest.json')
        self.assertFalse(model.training);self.assertEqual(contract['refinement'],self.spec)
        for k,v in self.model.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)
        self.assertTrue(torch.load(self.root/'best.pt',weights_only=True)['epoch']==4)

    def test_rejects_unverified_receipt_and_changed_gate_semantics(self):
        self.manifest['refinement']=dict(self.spec,identity_coarse_gate=False);self.write()
        with self.assertRaisesRegex(ValueError,'selected_spec'):load_repaired_model(self.root/'manifest.json')
        self.manifest['refinement']=self.spec;self.proof['status']='training'
        (self.root/'proof.json').write_text(json.dumps(self.proof));self.manifest['verification_receipt_sha256']=digest(self.root/'proof.json');self.write()
        with self.assertRaisesRegex(ValueError,'not been verified'):load_repaired_model(self.root/'manifest.json')

    def test_rejects_corruption_and_network_source_mismatch(self):
        self.manifest['network_source_sha256']['survival_network.py']='0'*64;self.write()
        with self.assertRaisesRegex(ValueError,'source differs'):load_repaired_model(self.root/'manifest.json')
        self.manifest['network_source_sha256'].pop('survival_network.py');self.write()
        with (self.root/'best.pt').open('ab') as f:f.write(b'corruption')
        with self.assertRaisesRegex(ValueError,'content changed'):load_repaired_model(self.root/'manifest.json')

    def test_verified_candidate_cannot_impersonate_another_variant(self):
        self.proof['verified_variants']={'ranking':dict(status='verified',stage='residual',refinement=self.spec,
            architecture=self.arch,interval_edges_days=self.proof['interval_edges_days'],feature_protocol=self.proof['feature_protocol'],
            fresh_reload=self.proof['final_fresh_reload'])}
        (self.root/'proof.json').write_text(json.dumps(self.proof))
        self.manifest['verification_receipt_sha256']=digest(self.root/'proof.json');self.manifest['verification_variant']='ranking';self.write()
        model,_=load_repaired_model(self.root/'manifest.json')
        self.assertFalse(model.training)
        self.manifest['stage']='gate';self.write()
        with self.assertRaisesRegex(ValueError,'stage'):load_repaired_model(self.root/'manifest.json')
        self.manifest['verification_variant']='not_verified';self.write()
        with self.assertRaisesRegex(ValueError,'not been verified'):load_repaired_model(self.root/'manifest.json')


if __name__=='__main__':unittest.main()
