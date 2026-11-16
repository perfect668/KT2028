"""Meaningful invariants for pristine EAGLE operations and survival integration."""
import os
import copy
import io
import unittest
import numpy as np
import torch

from kt2028.eagle_source_adapter import (SourceEAGLE, tiled_mase, survival_objective,
                                        grid_coordinates, pack_grids, pool_canvas)
from kt2028.survival import discrete_nll, encode_target


class EagleChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.model = SourceEAGLE(os.environ['EAGLE_SOURCE'], feature_dim=8, dim=8, tile=4, topk=3)

    def test_tiled_mase_eval_and_gradients_match_official(self):
        a = self.model.network.mase.eval()
        b = copy.deepcopy(a)
        x = torch.randn(1, 1, 9, 11, 8, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        expected = a(x)
        actual = tiled_mase(b, y, tile=4)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        weight = torch.randn_like(actual)
        (actual*weight).sum().backward()
        (expected*weight).sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=2e-4)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-4, rtol=3e-4)

    def test_training_dropout_is_one_channel_mask_for_entire_canvas(self):
        module = self.model.network.mase.train()
        tiled = copy.deepcopy(module)
        x = torch.randn(1, 1, 9, 11, 8)
        torch.manual_seed(19)
        expected = module(x)
        torch.manual_seed(19)
        actual = tiled_mase(tiled, x, tile=4, use_checkpoint=True)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        expected.sum().backward()
        actual.sum().backward()
        for a, b in zip(module.parameters(), tiled.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-4, rtol=4e-4)

    def test_full_attention_logits_neighbors_and_gradients_match_official(self):
        model = self.model.eval()
        raw = copy.deepcopy(model.network).eval()
        cube = torch.randn(9, 11, 8)
        mask = torch.rand(9, 11) > .3
        expected, bg, neighbor, alpha = raw(cube[None], mask)
        actual = model(cube, mask, auxiliary=True)
        torch.testing.assert_close(actual['logits'], expected[0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual['background_attention'], bg)
        torch.testing.assert_close(actual['neighbor_logits'], neighbor, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual['neighbor_attention'], alpha[0])
        actual['logits'].sum().backward()
        expected.sum().backward()
        for a, b in zip(model.network.parameters(), raw.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=5e-4)

    def test_survival_neighbor_likelihood_all_boundaries(self):
        for day in [1, 364, 365, 366, 730, 1825]:
            for event in [0, 1]:
                target = encode_target(day, event, [0, 365, 730, 1095, 1460, 1825])
                local = torch.randn(9, 5, requires_grad=True)
                weights = torch.softmax(torch.randn(9), 0)
                out = dict(logits=torch.randn(5), neighbor_logits=local, neighbor_attention=weights,
                           background_attention=torch.tensor(.2))
                loss, terms = survival_objective(out, target)
                expected = sum(discrete_nll(row, target)*weight for row, weight in zip(local, weights))
                torch.testing.assert_close(terms['neighborhood'], expected)
                torch.testing.assert_close(loss, discrete_nll(out['logits'], target)+expected+.2)
                if not event and day < 365:
                    self.assertEqual(float(terms['bag_nll']), 0.)
                    self.assertEqual(float(terms['neighborhood'].detach()), 0.)

    def test_geometry_and_packing_preserve_features_and_odd_edges(self):
        boxes = np.array([[112, 224, 224, 336], [336, 224, 448, 336], [112, 448, 224, 560]])
        xy = grid_coordinates(boxes)
        np.testing.assert_array_equal(xy, [[0, 0], [2, 0], [0, 2]])
        with self.assertRaises(ValueError):
            grid_coordinates(np.r_[boxes, boxes[:1]])
        a = np.ones((3, 8), dtype='float32')
        b = np.full((2, 8), 3., dtype='float32')
        cube, mask = pack_grids([(xy, a), (np.array([[0, 0], [0, 1]]), b)])
        self.assertEqual(int(mask.sum()), 5)
        np.testing.assert_array_equal(np.sort(cube[mask][:, 0].numpy()), [1, 1, 1, 3, 3])
        border = torch.zeros(3, 3, 8)
        border[2, 2] = 7
        m = torch.zeros(3, 3, dtype=torch.bool)
        m[2, 2] = True
        pooled, m, levels = pool_canvas(border, m, maximum=4)
        self.assertEqual(levels, 1)
        self.assertEqual(float(pooled[1, 1, 0]), 7.)
        self.assertTrue(m[1, 1])

    def test_strict_checkpoint_reload_reproduces_survival(self):
        model = self.model.eval()
        cube, mask = torch.randn(9, 11, 8), torch.ones(9, 11, dtype=torch.bool)
        expected = model(cube, mask)['survival']
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        clone = SourceEAGLE(os.environ['EAGLE_SOURCE'], feature_dim=8, dim=8, tile=4, topk=3).eval()
        clone.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        torch.testing.assert_close(expected, clone(cube, mask)['survival'], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
