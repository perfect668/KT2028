import copy
import unittest
import numpy as np

from kt2028.audit_tcga import validate_geometry
from kt2028.dual_scale_geometry import build_nested_grid


class GeometryAudit(unittest.TestCase):
    def setUp(self):
        rgb = np.full((112, 112, 3), [150, 30, 100], dtype=np.uint8)
        self.g = build_nested_grid(rgb, (448, 448), (4, 4), (.25, .25),
            dict(input_pixels=224, coarse_mpp_um=2., fine_mpp_um=.5,
                 minimum_tissue_fraction=.01, tissue_channel_spread_min=10,
                 tissue_mean_min=20, tissue_mean_max=245))

    def test_valid_mapping(self):
        validate_geometry(self.g, np.array([.25, .25]))

    def test_wrong_parent_rejected(self):
        self.g['fine_parent_indices'][0] = 1
        with self.assertRaises(ValueError):
            validate_geometry(self.g, np.array([.25, .25]))

    def test_cross_parent_extent_rejected(self):
        self.g['fine_bbox_um'][0, 2] = 449
        with self.assertRaises(ValueError):
            validate_geometry(self.g, np.array([.25, .25]))

    def test_wrong_csr_rejected(self):
        self.g['fine_ptr'][-1] = 15
        with self.assertRaises(AssertionError):
            validate_geometry(self.g, np.array([.25, .25]))

    def test_wrong_level0_coordinates_rejected(self):
        self.g['fine_coords'][0, 0] = 1
        with self.assertRaises(AssertionError):
            validate_geometry(self.g, np.array([.25, .25]))


if __name__ == '__main__':
    unittest.main()
