"""Spatial invariants: native resolution, physical coverage, boundary padding."""
import unittest
from unittest.mock import patch
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from kt2028.local_svs import LocalSVS
from kt2028.dual_scale_geometry import build_nested_grid


class FakeSlide:
    properties = {'mpp-x': '0.25', 'mpp-y': '0.25', 'objective': '40'}
    dimensions = (1024, 1024)
    level_downsamples = [1, 4, 16]
    level_dimensions = [(1024, 1024), (256, 256), (64, 64)]
    color_profile = None

    def __init__(self, _):
        self.last_read = None

    def close(self):
        pass

    def read_region(self, origin, level, wh):
        self.last_read = (origin, level, wh)
        result = Image.new('RGBA', wh, (255, 255, 255, 0))
        w = min(wh[0], max(0, int((1024-origin[0])/self.level_downsamples[level])))
        h = min(wh[1], max(0, int((1024-origin[1])/self.level_downsamples[level])))
        if w and h:
            result.paste((150, 30, 100, 255), (0, 0, w, h))
        return result


class LocalSVSTest(unittest.TestCase):
    def setUp(self):
        self.mock = patch.dict(sys.modules, {'openslide': SimpleNamespace(OpenSlide=FakeSlide,
            PROPERTY_NAME_MPP_X='mpp-x', PROPERTY_NAME_MPP_Y='mpp-y', PROPERTY_NAME_OBJECTIVE_POWER='objective')})
        self.mock.start()

    def tearDown(self):
        self.mock.stop()

    def test_level_is_chosen_by_mpp_not_level_number(self):
        with LocalSVS('fake') as r:
            self.assertEqual(r.select_level(2), 1)  # level 2 would undersample 5x
            self.assertEqual(r.select_level(.5), 0)

    def test_never_invent_native_resolution(self):
        with LocalSVS('fake') as r:
            with self.assertRaises(ValueError):
                r.select_level(.1)

    def test_fixed_physical_coverage_and_boundary_padding(self):
        with LocalSVS('fake') as r:
            im = r.read_patch(200, 200, .5)
            self.assertEqual(im.size, (224, 224))
            self.assertEqual(im.mode, 'RGB')
            self.assertEqual(im.getpixel((5, 5)), (150, 30, 100))
            self.assertEqual(im.getpixel((220, 220)), (255, 255, 255))
            self.assertEqual(r.slide.last_read[0], (800, 800))
            self.assertEqual(r.slide.last_read[2], (449, 449))

    def test_geometry_children_are_inside_their_parent(self):
        rgb = np.full((112, 112, 3), [150, 30, 100], dtype=np.uint8)
        config = dict(input_pixels=224, coarse_mpp_um=2., fine_mpp_um=.5,
                      minimum_tissue_fraction=.01, tissue_channel_spread_min=10,
                      tissue_mean_min=20, tissue_mean_max=245)
        g = build_nested_grid(rgb, (448, 448), (4, 4), (.25, .25), config)
        self.assertEqual(g['fine_counts'].tolist(), [16])
        np.testing.assert_array_equal(g['fine_ptr'], [0, 16])
        self.assertTrue(np.all(g['fine_bbox_um'][:, :2] >= 0))
        self.assertTrue(np.all(g['fine_bbox_um'][:, 2:] <= 448))
        self.assertEqual(len(np.unique(g['fine_coords'], axis=0)), 16)


if __name__ == '__main__':
    unittest.main()
