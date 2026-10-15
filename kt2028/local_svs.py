"""Physical, bounded OpenSlide sampling of local SVS; no learned preprocessing."""
import math
from pathlib import Path

import numpy as np
from PIL import Image


class LocalSVS:
    def __init__(self, path, maximum_upsampling_fraction=0.02):
        import openslide
        self.path = Path(path)
        self.slide = openslide.OpenSlide(str(self.path))
        props = self.slide.properties
        try:
            self.mpp = tuple(float(props[k]) for k in
                             [openslide.PROPERTY_NAME_MPP_X, openslide.PROPERTY_NAME_MPP_Y])
        except (KeyError, ValueError):
            self.slide.close()
            raise ValueError('Missing native MPP; do not guess from pyramid level or objective')
        if not all(math.isfinite(x) and x > 0 for x in self.mpp):
            self.slide.close()
            raise ValueError('Invalid native MPP')
        self.wh = self.slide.dimensions
        self.image_wh_um = tuple(a*b for a, b in zip(self.wh, self.mpp))
        self.maximum_upsampling_fraction = maximum_upsampling_fraction

    def close(self):
        self.slide.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def select_level(self, target_mpp):
        if max(self.mpp) > target_mpp*(1+self.maximum_upsampling_fraction):
            raise ValueError('Native resolution insufficient for requested physical sampling')
        candidates = [i for i, d in enumerate(self.slide.level_downsamples)
                      if max(self.mpp)*d <= target_mpp*(1+self.maximum_upsampling_fraction)]
        if not candidates:
            raise ValueError('No sufficiently resolved pyramid level')
        return max(candidates, key=lambda i: self.slide.level_downsamples[i])

    def read_patch(self, x_um, y_um, target_mpp, pixels=224):
        if min(x_um, y_um) < 0 or not all(math.isfinite(x) for x in [x_um, y_um, target_mpp]):
            raise ValueError('Invalid physical patch position')
        level = self.select_level(target_mpp)
        d = self.slide.level_downsamples[level]
        xy = np.asarray([x_um, y_um])/np.asarray(self.mpp)
        origin = np.floor(xy).astype(int)
        offset = (xy-origin)/d
        size = pixels*target_mpp/np.asarray(self.mpp)/d
        read_wh = np.ceil(offset+size).astype(int)+1
        if int(np.prod(read_wh)) > 4_000_000:
            raise ValueError('ROI allocation exceeds bound')
        rgba = self.slide.read_region(tuple(origin), level, tuple(read_wh))
        rgb = Image.new('RGB', rgba.size, 'white')
        rgb.paste(rgba, mask=rgba.getchannel('A'))
        extent = (*offset, *(offset+size))
        return rgb.transform((pixels, pixels), Image.Transform.EXTENT, tuple(extent),
                             resample=Image.Resampling.BILINEAR, fillcolor='white')

    def thumbnail(self, maximum_edge):
        rgba = self.slide.get_thumbnail((maximum_edge, maximum_edge)).convert('RGBA')
        rgb = Image.new('RGB', rgba.size, 'white')
        rgb.paste(rgba, mask=rgba.getchannel('A'))
        return rgb

    def metadata(self):
        import openslide
        return dict(dimensions=list(self.wh), native_mpp_xy=list(self.mpp),
                    image_wh_um=list(self.image_wh_um),
                    level_downsamples=list(self.slide.level_downsamples),
                    level_dimensions=[list(x) for x in self.slide.level_dimensions],
                    objective_power=self.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER),
                    icc_profile_present=self.slide.color_profile is not None,
                    color_policy='decoded RGB; no additional stain normalization or ICC transform',
                    sampling='physical EXTENT bilinear; transparent/outside padded white',
                    maximum_native_upsampling_fraction=self.maximum_upsampling_fraction)
