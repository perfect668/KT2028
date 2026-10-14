"""Software fixtures, no fitting or exported performance claims."""
import math
import json
import copy
import pytest
import numpy as np
import torch

from kt2028.survival import encode_target, discrete_nll, curve_from_logits, common_censor_metrics
from kt2028.prepare_survival import classify_record
from kt2028.survival_network import SurvivalPredictor


EDGES = [0,365,730,1095,1460,1825]


def evidence():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    return dict(coarse=torch.randn(3,1024),
                boxes=torch.tensor([[0.,0.,896.,896.],[896.,0.,1792.,896.],[0.,896.,896.,1792.]]),
                observed_fine={0:torch.randn(2,1024),2:torch.randn(4,1024)},
                relative_xy={0:torch.rand(2,2),2:torch.rand(4,2)}, candidate_patch_count=9)


def model():
    return SurvivalPredictor(dim=32, heads=4, dropout=0).eval()


def test_likelihood_event_and_censor_match_dsca_alpha_zero():
    logits = torch.logit(torch.tensor([.1,.2,.3,.4,.5]))
    event = encode_target(800,1,EDGES)
    censor = encode_target(1825,0,EDGES)
    assert float(discrete_nll(logits,event)) == pytest.approx(-math.log(.9*.8*.3),rel=1e-6)
    assert float(discrete_nll(logits,censor)) == pytest.approx(-math.log(.9*.8*.7*.6*.5),rel=1e-6)
    # Independently emulate the upstream hazard-domain gather formula (DSCA c=1-event).
    for target in (event,censor):
        h=logits.sigmoid();s=torch.cat((torch.ones(1),torch.cumprod(1-h,0)))
        b=target['event_bin'] if target['event'] else target['completed_intervals']-1
        ref=-torch.log(s[b])-torch.log(h[b]) if target['event'] else -torch.log(s[b+1])
        assert torch.allclose(discrete_nll(logits,target),ref)


def test_boundary_encoding_and_source_exclusions():
    assert encode_target(365,1,EDGES)['event_bin']==0
    assert encode_target(366,1,EDGES)['event_bin']==1
    assert encode_target(1825,0,EDGES)['completed_intervals']==5
    contract={'interval_edges_days':EDGES,'administrative_censor_day':1825}
    alive={'died_within_5_years':'0','days_till_death':'Alive'}
    assert classify_record(alive,0,contract)[0]['event']==0
    for day,reason in [('0','zero_death_day_pending_confirmation'),('1851','death_day_beyond_five_year_window')]:
        assert classify_record({'died_within_5_years':'1','days_till_death':day},1,contract)[1]==reason
    with pytest.raises(ValueError):classify_record({'died_within_5_years':'1','days_till_death':'Err:502'},1,contract)
    with pytest.raises(ValueError):encode_target(0,1,EDGES)


def test_stable_loss_and_monotone_curve_at_extreme_logits():
    x=torch.tensor([-1000.,1000.,0.,-1000.,1000.],requires_grad=True)
    loss=discrete_nll(x,encode_target(800,1,EDGES));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    _,s=curve_from_logits(x)
    assert torch.all(s[1:]<=s[:-1]) and torch.isfinite(s).all()


def test_metrics_reject_early_censor_and_keep_continuous_probabilities():
    curves=np.array([[.7,.5,.3,.2,.1],[.99,.98,.97,.96,.95]])
    scores=common_censor_metrics([300,1825],[1,0],curves,EDGES)
    assert scores['c_index']==1 and scores['comparable_pairs']==1
    assert scores['by_day']['365']['brier']==pytest.approx((.7**2+.01**2)/2)
    with pytest.raises(ValueError,match='common administrative'):
        common_censor_metrics([300,1000],[1,0],curves,EDGES)


def test_zero_observation_is_coarse_shared_head_and_no_fine_branch():
    x=evidence();x['observed_fine']={};x['relative_xy']={}
    m=model();out=m(**x,return_aux=True)
    assert torch.equal(out['logits'],out['coarse_aux_logits']) and out['fine_aux_logits'] is None
    assert out['observed_tokens'].shape==(0,32)
    assert out['coarse_tokens'].shape==(3,32) and out['logits'].shape==(5,)


def test_pair_permutation_invariance_and_independent_fine_only_auxiliary():
    m=model();x=evidence();a=m(**x,return_aux=True)
    y=copy.deepcopy(x)
    for i in y['observed_fine']:
        y['observed_fine'][i]=y['observed_fine'][i].flip(0)
        y['relative_xy'][i]=y['relative_xy'][i].flip(0)
    b=m(**y,return_aux=True)
    assert torch.allclose(a['logits'],b['logits'],atol=1e-6)
    y['coarse']=y['coarse']+3
    c=m(**y,return_aux=True)
    assert torch.allclose(a['fine_aux_logits'],c['fine_aux_logits'],atol=1e-6)
    assert not torch.allclose(a['logits'],c['logits'])


def test_fine_aux_supervises_shared_main_trunk_without_parameter_updates():
    m=model();x=evidence();before={k:v.clone() for k,v in m.state_dict().items()}
    out=m(**x,return_aux=True)
    discrete_nll(out['fine_aux_logits'],encode_target(800,1,EDGES)).backward()
    for prefix in ['fine_projection','fine_position','local_attention','fine_encoder','fine_pool','survival_head']:
        assert any(p.grad is not None and torch.any(p.grad!=0) for n,p in m.named_parameters() if n.startswith(prefix+'.'))
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def test_invalid_observation_keys_counts_and_position_rejected():
    x=evidence();m=model();x['relative_xy'].pop(0)
    with pytest.raises(ValueError):m(**x)
    x=evidence();x['candidate_patch_count']=1
    with pytest.raises(ValueError):m(**x)
    x=evidence();x['relative_xy'][0][0,0]=2
    with pytest.raises(ValueError):m(**x)


def test_test_manifest_is_refused_before_asset_access(tmp_path):
    from kt2028.survival_data import SurvivalDataset
    p=tmp_path/'test.json'
    p.write_text(json.dumps({'schema':'KT2028-survival-manifest-1','partition':'test'}))
    with pytest.raises(ValueError,match='no Test'):
        SurvivalDataset(p,tmp_path,contract={},purpose='model_selection')


def test_batched_selected_fine_read_preserves_region_row_order(tmp_path):
    import zarr
    from kt2028.fusion_data import FusionCase
    from kt2028.uni_encoder import sha256_file
    path=tmp_path/'fine.zarr';g=zarr.open_group(str(path),mode='w',zarr_format=2)
    values=np.arange(6*1024,dtype=np.float32).reshape(6,1024)
    values[2]=np.nan  # Hidden region must not contaminate acquisition.
    g.create_array('features',data=values)
    folder=tmp_path/'coarse';folder.mkdir()
    np.savez_compressed(folder/'regions.npz',coarse=np.ones((3,1024),np.float32),
        bbox_um=np.zeros((3,4),np.float32),fine_counts=np.array([2,1,3]),fine_ptr=np.array([0,2,3,6]),
        fine_indices=np.array([1,0,2,5,3,4]),fine_relative_xy=np.zeros((6,2),np.float32))
    (folder/'receipt.json').write_text(json.dumps({'status':'completed','schema':'KT2028-R2-coarse-UNI',
        'regions_sha256':sha256_file(folder/'regions.npz'),'fine_path':str(path)}))
    out=FusionCase(folder).inputs([2,0])
    assert np.array_equal(out['observed_fine'][2].numpy(),values[[5,3,4]])
    assert np.array_equal(out['observed_fine'][0].numpy(),values[[1,0]])
    assert set(out)=={'coarse','boxes','observed_fine','relative_xy','candidate_patch_count'}
