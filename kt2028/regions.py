"""Physical RGB-defined regions and unique ownership of cached fine patches."""
import math
import numpy as np
from PIL import Image


def tissue_mask(rgb, config):
    x = rgb.astype(np.float32); mean = x.mean(-1)
    return ((x.max(-1)-x.min(-1) >= config['tissue_channel_spread_min']) &
            (mean > config['tissue_mean_min']) & (mean < config['tissue_mean_max']))


def rgb_regions(rgb, image_wh_um, mpp_xy, config):
    field = config['region_field_um']
    if field != config['region_stride_um']: raise ValueError('This contract requires disjoint spatial cells')
    mask = tissue_mask(rgb, config)
    candidates = []
    width, height = image_wh_um
    for row in range(math.ceil(height/field)):
        for col in range(math.ceil(width/field)):
            x0,y0 = col*field,row*field
            x1,y1 = min(x0+field,width),min(y0+field,height)
            px0,py0 = math.floor(x0/mpp_xy[0]),math.floor(y0/mpp_xy[1])
            px1,py1 = min(math.ceil(x1/mpp_xy[0]),mask.shape[1]),min(math.ceil(y1/mpp_xy[1]),mask.shape[0])
            patch_mask = mask[py0:py1,px0:px1]
            fraction = float(patch_mask.mean()) if patch_mask.size else 0.0
            if fraction >= config['minimum_tissue_fraction']:
                candidates.append({'cell':[col,row],'bbox_um':[x0,y0,x1,y1],
                                   'tissue_fraction':fraction,'image_valid_fraction':(x1-x0)*(y1-y0)/field**2})
    if not candidates: raise ValueError('No RGB tissue regions')
    return candidates


def map_fine(coords, candidates, image_wh_pixels, l0_mpp_xy, config):
    """Preserve boundary patches; assign by center of their valid image intersection."""
    if coords.ndim != 2 or coords.shape[1] != 2 or np.any(coords < 0): raise ValueError('Invalid coords')
    pixel_size = np.floor(config['fine_input_pixels']*config['fine_target_mpp_um']/np.array(l0_mpp_xy))
    if np.any(pixel_size <= 0): raise ValueError('Invalid physical fine field')
    lo = coords.astype(np.float64)
    hi = np.minimum(lo+pixel_size,np.asarray(image_wh_pixels))
    valid_sizes = np.maximum(hi-lo,0)
    valid_fraction = np.prod(valid_sizes,axis=1)/np.prod(pixel_size)
    centers_um = (lo+valid_sizes/2)*np.array(l0_mpp_xy)
    cells = np.floor(centers_um/config['region_field_um']).astype(np.int64)
    lookup = {tuple(c['cell']):i for i,c in enumerate(candidates)}
    ownership = np.array([lookup.get(tuple(cell),-1) if valid_fraction[i] > 0 else -1
                          for i,cell in enumerate(cells)],dtype=np.int64)
    groups = [np.flatnonzero(ownership==i) for i in range(len(candidates))]
    counts = np.array([len(g) for g in groups],dtype=np.int64)
    ptr = np.r_[0,np.cumsum(counts)]
    indices = np.concatenate(groups).astype(np.int64)
    if len(np.unique(indices)) != len(indices): raise ValueError('Duplicate fine acquisition ownership')
    return {'fine_indices':indices,'fine_ptr':ptr,'fine_counts':counts,'patch_region':ownership,
            'fine_image_valid_fraction':valid_fraction.astype(np.float32),
            'fine_pixel_size_xy':pixel_size.astype(np.int64)}


def crop_region(rgb_image, candidate, mpp_xy, config):
    # Fixed physical field, white padding at image boundaries, no additional center crop.
    x0,y0,_,_ = candidate['bbox_um'];field=config['region_field_um']
    extent = (x0/mpp_xy[0],y0/mpp_xy[1],(x0+field)/mpp_xy[0],(y0+field)/mpp_xy[1])
    n=config['input_pixels']
    return rgb_image.transform((n,n),Image.Transform.EXTENT,extent,
                               resample=Image.Resampling.BILINEAR,fillcolor=(255,255,255))
