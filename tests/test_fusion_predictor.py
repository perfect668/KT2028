"""Software fixtures verify observation permissions and anchor semantics."""
import pytest
import torch
from kt2028.fusion_network import CrossScalePredictor


def sample():
    torch.manual_seed(9)
    c = torch.randn(3, 1024)
    b = torch.tensor([[0., 0., 896., 896.], [896., 0., 1792., 896.], [0., 896., 896., 1792.]])
    return dict(coarse=c, boxes=b, observed_fine={}, relative_xy={}, candidate_patch_count=9)


def model():
    torch.set_num_threads(2)
    return CrossScalePredictor(dim=32, heads=4, dropout=0).eval()


def test_zero_fine_is_exact_coarse_and_no_aux():
    m, x = model(), sample()
    out = m(**x)
    assert torch.equal(out['logit'], out['coarse_logit'])
    assert out['fine_aux_logit'] is None and len(out['observed_tokens']) == 0


def test_fine_never_replaces_coarse_and_aux_excludes_coarse():
    m, x = model(), sample()
    before = m(**x)
    x['observed_fine'] = {1: torch.randn(4, 1024)}
    x['relative_xy'] = {1: torch.rand(4, 2)}
    after = m(**x)
    assert torch.equal(before['coarse_tokens'], after['coarse_tokens'])
    auxiliary = after['fine_aux_logit']
    x['coarse'] = x['coarse']+2
    changed = m(**x)
    assert torch.equal(auxiliary, changed['fine_aux_logit'])
    assert not torch.equal(after['coarse_logit'], changed['coarse_logit'])


def test_fine_permutation_and_position_pairing_invariant():
    m, x = model(), sample()
    x['observed_fine'] = {0: torch.randn(2, 1024), 2: torch.randn(5, 1024)}
    x['relative_xy'] = {0: torch.rand(2, 2), 2: torch.rand(5, 2)}
    with torch.no_grad():
        m.delta[-1].weight.fill_(.03)
    before = m(**x)['logit']
    for i in x['observed_fine']:
        x['observed_fine'][i] = x['observed_fine'][i].flip(0)
        x['relative_xy'][i] = x['relative_xy'][i].flip(0)
    assert torch.allclose(before, m(**x)['logit'], atol=1e-6)


def test_missing_position_or_invalid_observation_rejected():
    m, x = model(), sample()
    x['observed_fine'] = {3: torch.randn(2, 1024)}
    with pytest.raises(ValueError):
        m(**x)


def test_fusion_optimizes_without_changing_anchor():
    m, x = model(), sample()
    m.freeze_coarse(); m.train()
    assert not m.coarse.training
    x['observed_fine'] = {0: torch.randn(3, 1024)}
    x['relative_xy'] = {0: torch.rand(3, 2)}
    old = {k: v.clone() for k, v in m.coarse.state_dict().items()}
    old_attention = m.local_attention.in_proj_weight.detach().clone()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.001)
    for _ in range(3):
        opt.zero_grad()
        out = m(**x)
        (out['logit'].square()+out['fine_aux_logit'].square()).backward()
        opt.step()
    assert all(torch.equal(v, m.coarse.state_dict()[k]) for k, v in old.items())
    assert not torch.equal(old_attention, m.local_attention.in_proj_weight)


def test_loader_does_not_read_nonobserved_nan_features(tmp_path):
    import json
    import numpy as np
    import zarr
    from kt2028.fusion_data import FusionCase
    from kt2028.uni_encoder import sha256_file
    fine_path = tmp_path/'software_fine_fixture.zarr'
    fine = zarr.open_group(str(fine_path), mode='w', zarr_format=2)
    values = np.ones((4, 1024), dtype=np.float32)
    values[2:] = np.nan
    fine.create_array('features', data=values)
    folder = tmp_path/'software_coarse_fixture'
    folder.mkdir()
    np.savez_compressed(folder/'regions.npz', coarse=np.ones((2, 1024), dtype=np.float32),
                        bbox_um=np.array([[0, 0, 896, 896], [896, 0, 1792, 896]], dtype=np.float32),
                        fine_counts=np.array([2, 2]), fine_ptr=np.array([0, 2, 4]),
                        fine_indices=np.arange(4), fine_relative_xy=np.zeros((4, 2), dtype=np.float32))
    (folder/'receipt.json').write_text(json.dumps(dict(schema='KT2028-R2-coarse-UNI', status='completed',
                      regions_sha256=sha256_file(folder/'regions.npz'), fine_path=str(fine_path))))
    case = FusionCase(folder)
    assert case.inputs([])['observed_fine'] == {}
    assert torch.isfinite(case.inputs([0])['observed_fine'][0]).all()
    with pytest.raises(ValueError, match='observed fine'):
        case.inputs([1])
