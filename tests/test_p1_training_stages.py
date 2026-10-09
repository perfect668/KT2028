import json

import numpy as np
import pytest
import torch

from kt2028.network import EvidencePredictor, TargetedStateUpdater
from kt2028.training_data import TrainRegionDataset, ragged_collate
from kt2028.training_stages import configure_stage, stage_objective


def test_train_manifest_permission_and_multislide_guard(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'partition': 'test', 'endpoint': 'five_year_death', 'patients': []}))
    with pytest.raises(ValueError):
        TrainRegionDataset(path, tmp_path)
    with pytest.raises(ValueError):
        TrainRegionDataset(path, tmp_path, purpose='model_selection')
    path.write_text(json.dumps({'partition': 'train', 'endpoint': 'five_year_death',
                               'patients': [{'canonical_case_key': 'fixture', 'label': 0, 'slides': [{}, {}]}]}))
    with pytest.raises(ValueError):
        TrainRegionDataset(path, tmp_path)


@pytest.mark.parametrize('stage', ['full', 'align', 'partial'])
def test_staged_gradient_permissions(stage):
    torch.manual_seed(3)
    model, updater = EvidencePredictor(dim=16, branches=2), TargetedStateUpdater(dim=16)
    inputs = {'coarse': torch.ones(3, 512),
              'boxes': torch.tensor([[i, 0, i+1, 1] for i in range(3)], dtype=torch.float32),
              'observed_fine': {0: torch.randn(2, 1024), 1: torch.randn(5, 1024)}}
    configure_stage(model, updater, stage)
    loss, details = stage_objective(model, updater, inputs, torch.tensor(1.), stage)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in (*model.parameters(), *updater.parameters()) if not p.requires_grad)
    if stage == 'align':
        assert details['tsu_pairs'] == 2
        assert any(p.grad is not None for p in updater.parameters())
        assert model.fine_projection[0].weight.grad is None
    if stage == 'partial':
        assert model.coarse_projection[0].weight.grad is None
        assert model.fine_projection[0].weight.grad is None
        assert all(p.grad is None for p in updater.parameters())
    if stage == 'full':
        assert model.fine_projection[0].weight.grad is not None


def test_ragged_batch_preserves_case_boundaries():
    bags = [{'inputs': {'coarse': torch.zeros(2, 512)}}, {'inputs': {'coarse': torch.zeros(7, 512)}}]
    batch = ragged_collate(bags)
    assert [len(s['inputs']['coarse']) for s in batch] == [2, 7]


def test_zero_fine_loader_never_opens_uni(tmp_path, monkeypatch):
    manifest = {'partition': 'train', 'endpoint': 'five_year_death',
                'patients': [{'canonical_case_key': 'fixture', 'label': 1,
                              'slides': [{'slide_id': 'fixture', 'features_path': '/never/open'}]}]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    folder = tmp_path / 'fixture'
    folder.mkdir()
    (folder / 'receipt.json').write_text(json.dumps({'status': 'completed', 'fine_path': '/never/open'}))
    np.savez(folder / 'regions.npz', coarse=np.zeros((2, 512), dtype=np.float32),
             bbox_um=np.array([[0,0,1,1],[1,0,2,1]], dtype=np.float32), fine_counts=np.array([2,0]),
             fine_ptr=np.array([0,2,2]), fine_indices=np.array([0,1]), fine_image_valid_fraction=np.ones(2))
    def forbidden(*args, **kwargs):
        raise AssertionError('Unobserved UNI was read')
    monkeypatch.setattr('kt2028.training_data.zarr.open_group', forbidden)
    sample = TrainRegionDataset(path, tmp_path, evidence='zero')[0]
    assert sample['inputs']['observed_fine'] == {}
    assert sample['selected_patch_count'] == 0 and sample['target'] == 1


def test_training_config_rejects_ambiguous_checkpoint_rule():
    from kt2028.train_predictor import validate_config
    config={'epochs':30,'batch_cases':2,'max_tsu_pairs':128,'seed':41,'latent_dim':256,
            'attention_branches':4,'learning_rate':1e-4,'weight_decay':1e-4,'tsu_threshold':0.9,
            'endpoint_probability':0.25,'gradient_clip':5.,'selection_rule':'test_score'}
    with pytest.raises(ValueError):
        validate_config(config)
    config['selection_rule']='minimum_fixed_validation_objective_first_tie'
    validate_config(config)
