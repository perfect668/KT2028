"""Fail-closed software fixtures; no feature values, model, fitting, or GPU."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from kt2028.prepare_mixed_survival import target
from kt2028.train_mixed_survival import main
from kt2028.uni_encoder import sha256_file


class MixedPreflightTests(unittest.TestCase):
    def fixture(self, root, overlap=False):
        cfg=root/'config.json'
        cfg.write_text(json.dumps(dict(schema='KT2028-mixed-survival-predictor-1',
                                      old_checkpoint_reuse=False,data_contract={})))
        paths=[]
        for split,key in [('train','one'),('validate','one' if overlap else 'two')]:
            p=root/(split+'.json')
            p.write_text(json.dumps(dict(partition=split,data_contract={},
                sources=dict(config_sha256=sha256_file(cfg)), patients=[dict(
                    canonical_case_key=key,cohort='surgen',slides=[dict(slide_id=key,asset_id=key)])])))
            paths.append(p)
        (root/'split_lock.json').write_text(json.dumps(dict(schema='KT2028-mixed-split-lock-1',
            config_sha256=sha256_file(cfg),manifest_sha256=dict(
                train_manifest=sha256_file(paths[0]),validate_manifest=sha256_file(paths[1])))))
        return ['preflight','--config',str(cfg),'--train-manifest',str(paths[0]),
            '--validation-manifest',str(paths[1]),'--surgen-features',str(root/'features'),
            '--tcga-features',str(root/'features'),'--output',str(root/'run'),
            '--preflight-report',str(root/'report.json')]

    def test_missing_assets_create_no_model_or_run(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); argv=self.fixture(root)
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error: main()
            self.assertEqual(error.exception.code,2)
            report=json.loads((root/'report.json').read_text())
            self.assertEqual(len(report['assets']['missing']),2)
            self.assertFalse(report['training_started'])
            self.assertEqual(report['optimizer_steps'],0)
            self.assertFalse((root/'run').exists())

    def test_patient_overlap_rejected_before_model_access(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(sys,'argv',self.fixture(Path(d),True)):
                with self.assertRaisesRegex(ValueError,'overlap'): main()

    def test_early_censor_and_event_boundary_encoding(self):
        self.assertEqual(target(100,0,[0,365,730])['completed_intervals'],0)
        self.assertEqual(target(365,1,[0,365,730])['event_bin'],0)
        self.assertEqual(target(366,1,[0,365,730])['event_bin'],1)

    def test_modified_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); argv=self.fixture(root)
            with (root/'validate.json').open('a') as f: f.write(' ')
            with patch.object(sys,'argv',argv):
                with self.assertRaisesRegex(ValueError,'changed after registration'): main()


if __name__=='__main__': unittest.main()
