"""Outcome-independent nested physical grid for R2; no old fine coordinates."""
import math

import numpy as np

from .regions import rgb_regions, tissue_mask


def build_nested_grid(rgb, image_wh_um, mask_mpp_xy, l0_mpp_xy, config):
    pixels = config['input_pixels']
    coarse_field = pixels * config['coarse_mpp_um']
    fine_field = pixels * config['fine_mpp_um']
    ratio = coarse_field / fine_field
    if pixels != 224 or not math.isclose(ratio, round(ratio)) or ratio < 1:
        raise ValueError('This nested-grid protocol requires 224px and integral scale ratio')
    if not all(math.isfinite(v) and v > 0 for v in [*image_wh_um, *mask_mpp_xy, *l0_mpp_xy]):
        raise ValueError('Invalid physical geometry')
    settings = dict(config, region_field_um=coarse_field, region_stride_um=coarse_field)
    parents = rgb_regions(rgb, image_wh_um, mask_mpp_xy, settings)
    mask = tissue_mask(rgb, settings)
    children = []
    ownership = []
    fractions = []
    width, height = image_wh_um
    for parent_id, parent in enumerate(parents):
        x, y = parent['bbox_um'][:2]
        for row in range(round(ratio)):
            for col in range(round(ratio)):
                x0, y0 = x + col*fine_field, y + row*fine_field
                x1, y1 = min(x0+fine_field, width), min(y0+fine_field, height)
                if x0 >= width or y0 >= height:
                    continue
                px0, py0 = math.floor(x0/mask_mpp_xy[0]), math.floor(y0/mask_mpp_xy[1])
                px1 = min(math.ceil(x1/mask_mpp_xy[0]), mask.shape[1])
                py1 = min(math.ceil(y1/mask_mpp_xy[1]), mask.shape[0])
                part = mask[py0:py1, px0:px1]
                if part.size and part.mean() >= config['minimum_tissue_fraction']:
                    children.append([x0, y0, x1, y1])
                    ownership.append(parent_id)
                    fractions.append((x1-x0)*(y1-y0)/fine_field**2)
    coarse_bbox = np.asarray([p['bbox_um'] for p in parents], dtype=np.float64)
    fine_bbox = np.asarray(children, dtype=np.float64).reshape(-1, 4)
    parent_indices = np.asarray(ownership, dtype=np.int64)
    fine_counts = np.bincount(parent_indices, minlength=len(parents)).astype(np.int64)
    if fine_counts.max(initial=0) > round(ratio)**2:
        raise ValueError('Non-unique or invalid fine ownership')
    relative = ((fine_bbox[:, :2] - coarse_bbox[parent_indices, :2]) + fine_field/2) / coarse_field
    return {
        'coarse_bbox_um': coarse_bbox, 'fine_bbox_um': fine_bbox,
        'coarse_coords': np.floor(coarse_bbox[:, :2]/np.asarray(l0_mpp_xy)).astype(np.int64),
        'fine_coords': np.floor(fine_bbox[:, :2]/np.asarray(l0_mpp_xy)).astype(np.int64),
        'fine_parent_indices': parent_indices, 'fine_relative_xy': relative.astype(np.float32),
        'fine_counts': fine_counts, 'fine_ptr': np.r_[0, np.cumsum(fine_counts)],
        'fine_image_valid_fraction': np.asarray(fractions, dtype=np.float32),
        'coarse_image_valid_fraction': np.asarray([p['image_valid_fraction'] for p in parents], dtype=np.float32),
        'coarse_tissue_fraction': np.asarray([p['tissue_fraction'] for p in parents], dtype=np.float32)
    }
