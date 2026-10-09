import numpy as np
import pytest
from kt2028.regions import map_fine,rgb_regions,crop_region
from PIL import Image

CONFIG={'region_field_um':4.,'region_stride_um':4.,'fine_input_pixels':2,
        'fine_target_mpp_um':1.,'input_pixels':224,'minimum_tissue_fraction':.01,
        'tissue_channel_spread_min':10,'tissue_mean_min':20,'tissue_mean_max':245}


def test_rgb_candidates_do_not_depend_on_fine_and_exclude_white():
    rgb=np.full((8,8,3),255,dtype=np.uint8);rgb[:4,:4]=[180,80,160]
    regions=rgb_regions(rgb,[8,8],[1,1],CONFIG)
    assert [r['cell'] for r in regions]==[[0,0]]


def test_unique_ownership_keeps_partial_edge_and_records_unavailable_patches():
    rgb=np.tile(np.array([180,80,160],dtype=np.uint8),(8,8,1))
    regions=rgb_regions(rgb,[8,8],[1,1],CONFIG)
    coords=np.array([[0,0],[3,3],[7,7],[8,8]])
    m=map_fine(coords,regions,[8,8],[1,1],CONFIG)
    assert m['patch_region'].tolist()==[0,3,3,-1]
    np.testing.assert_allclose(m['fine_image_valid_fraction'],[1,1,.25,0])
    assert len(m['fine_indices'])==len(np.unique(m['fine_indices']))==3
    assert m['fine_counts'].sum()==3


def test_coarse_field_has_white_edge_padding_without_additional_center_crop():
    image=Image.new('RGB',(5,5),(180,80,160))
    candidate={'bbox_um':[4,4,5,5]}
    crop=crop_region(image,candidate,[1,1],CONFIG)
    assert crop.size==(224,224)
    assert crop.getpixel((200,200))==(255,255,255)
    assert crop.getpixel((10,10))==(180,80,160)


def test_overlapping_grid_not_silently_accepted():
    with pytest.raises(ValueError):rgb_regions(np.zeros((8,8,3),np.uint8),[8,8],[1,1],{**CONFIG,'region_stride_um':2})
