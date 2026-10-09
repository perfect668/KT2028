"""CPU tensor checks, automatically skipped when PyTorch is unavailable."""

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from msrlsurv.predictor import PartialABMIL


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this runtime")
class PredictorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = PartialABMIL(input_dim=8, hidden_dim=4, attention_dim=3, prior=0.3).eval()
        self.x = torch.randn(1, 9, 8)
        self.total = torch.tensor([9])

    def test_unobserved_features_are_invisible(self):
        mask = torch.tensor([[True, True, False, False, False, False, False, False, False]])
        changed = self.x.clone()
        changed[~mask] = torch.nan
        torch.testing.assert_close(self.model(self.x, mask, self.total)["logits"],
                                   self.model(changed, mask, self.total)["logits"])

    def test_empty_mask_returns_prior_and_zero_attention(self):
        result = self.model(self.x, torch.zeros((1, 9), dtype=torch.bool), self.total)
        torch.testing.assert_close(result["probabilities"], torch.tensor([0.3]))
        self.assertFalse(result["attention"].any())

    def test_zero_length_bag_returns_prior(self):
        result = self.model(self.x[:, :0], torch.zeros((1, 0), dtype=torch.bool), self.total)
        torch.testing.assert_close(result["probabilities"], torch.tensor([0.3]))

    def test_patch_permutation_does_not_change_prediction(self):
        mask = torch.ones((1, 9), dtype=torch.bool)
        order = torch.randperm(9)
        torch.testing.assert_close(self.model(self.x, mask, self.total)["logits"],
                                   self.model(self.x[:, order], mask[:, order], self.total)["logits"])

    def test_incremental_and_complete_predictions_agree(self):
        state = self.model.empty_state()
        for start, end in [(0, 2), (2, 5), (5, 9)]:
            state, probability = self.model.acquire(state, self.x[0, start:end], 9)
            mask = torch.arange(9).unsqueeze(0) < end
            expected = self.model(self.x, mask, self.total)["probabilities"][0]
            torch.testing.assert_close(probability, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(state.patch_count, 9)

    def test_incremental_dropout_is_rejected(self):
        self.model.train()
        with self.assertRaises(RuntimeError):
            self.model.acquire(self.model.empty_state(), self.x[0, :2], 9)


if __name__ == "__main__":
    unittest.main()
