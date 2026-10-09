"""Software contract checks; controlled inputs are not medical experiments."""

import unittest

import numpy as np

from msrlsurv.environment import AcquisitionEnv, log_loss
from msrlsurv.regions import allocate_regions, representative_patches, spatial_regions


class SpatialTests(unittest.TestCase):
    def test_partition_covers_each_patch_once(self):
        coords = np.asarray([(x, y) for x in range(11) for y in range(7)])
        groups = spatial_regions(coords, 13)
        self.assertEqual(len(groups), 13)
        np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(len(coords)))
        for group in groups:
            chosen = representative_patches(coords, group, 4)
            self.assertLessEqual(len(chosen), 4)
            self.assertEqual(len(chosen), len(set(chosen)))
            self.assertTrue(set(chosen).issubset(set(group)))

    def test_every_slide_is_retained(self):
        allocation = allocate_regions([1, 2, 80], 64)
        self.assertEqual(sum(allocation), 64)
        self.assertTrue(all(1 <= n <= c for n, c in zip(allocation, [1, 2, 80])))

    def test_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            spatial_regions(np.asarray([[1, 1], [1, 1]]), 2)


class EnvironmentTests(unittest.TestCase):
    def make_env(self, *, label=None, budget=4, regions=None):
        regions = regions if regions is not None else [np.ones((2, 3)), np.full((2, 3), 2)]
        calls = []

        def predict(patches, total):
            calls.append(patches.copy())
            return {0: 0.3, 2: 0.5, 4: 0.7}[len(patches)]

        env = AcquisitionEnv(regions, np.ones((2, 2)), predict,
                             label=label, budget_patches=budget, cost_unit=4, max_regions=2)
        return env, calls

    def test_reset_has_no_fine_and_stop_is_legal(self):
        env, calls = self.make_env()
        self.assertEqual(calls[0].shape, (0, 3))
        self.assertFalse(env.observation()["fine"].any())
        self.assertTrue(env.action_masks()[env.stop_action])
        self.assertIsNone(env.step(env.stop_action).reward)

    def test_predictor_only_receives_purchased_features(self):
        env, calls = self.make_env()
        transition = env.step(1)
        np.testing.assert_array_equal(calls[-1], np.full((2, 3), 2))
        self.assertFalse(transition.observation["fine"][0].any())
        self.assertIsNone(transition.reward)

    def test_label_cannot_change_initial_observation(self):
        a, _ = self.make_env(label=0)
        b, _ = self.make_env(label=1)
        for key in a.observation():
            np.testing.assert_array_equal(a.observation()[key], b.observation()[key])

    def test_reward_sign_and_telescoping_cost(self):
        env, _ = self.make_env(label=1)
        first = env.step(0)
        self.assertGreater(first.reward, 0)
        self.assertFalse(env.action_masks()[0])
        second = env.step(1)
        self.assertTrue(second.terminated)
        self.assertAlmostEqual(first.reward + second.reward,
                               log_loss(1, 0.3) - log_loss(1, 0.7) - 0.25)

    def test_budget_blocks_acquisition_and_repeated_step(self):
        env, _ = self.make_env(budget=1)
        self.assertFalse(env.action_masks()[:2].any())
        with self.assertRaises(ValueError):
            env.step(0)
        env.step(env.stop_action)
        with self.assertRaises(RuntimeError):
            env.step(env.stop_action)

    def test_hidden_region_changes_do_not_change_state(self):
        a, _ = self.make_env()
        b, _ = self.make_env(regions=[np.ones((2, 3)), np.full((2, 3), 1000)])
        a.step(0)
        b.step(0)
        for key in a.observation():
            np.testing.assert_array_equal(a.observation()[key], b.observation()[key])


if __name__ == "__main__":
    unittest.main()
