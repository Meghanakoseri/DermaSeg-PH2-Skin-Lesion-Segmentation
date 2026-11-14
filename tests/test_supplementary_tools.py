"""Small synthetic tests for the new validation-only and boundary utilities.

Run from the repository root with: python -m unittest tests/test_supplementary_tools.py
No PH2 files, checkpoints, training runs, or test-set labels are used.
"""
import unittest
import numpy as np
import torch

from scripts.boundary_metrics_from_predictions import scores
from scripts.validation_robustness import corruptions


class BoundaryMetricTests(unittest.TestCase):
    def test_identical_masks_have_perfect_boundary_f1(self):
        mask = np.zeros((48, 48), dtype=bool)
        mask[9:37, 11:34] = True
        precision, recall, f1, _, _ = scores(mask, mask, tolerance=2)
        self.assertAlmostEqual(precision, 1.0, places=12)
        self.assertAlmostEqual(recall, 1.0, places=12)
        self.assertAlmostEqual(f1, 1.0, places=12)


class ValidationCorruptionTests(unittest.TestCase):
    def test_corruptions_preserve_tensor_shape_and_names(self):
        torch.manual_seed(7)
        images = torch.rand(2, 3, 32, 40)
        outputs = corruptions(images, seed=2026)
        expected = {
            "clean", "brightness_0.8", "brightness_1.2",
            "gaussian_noise_std_0.03", "3x3_mean_blur", "translation_4px"
        }
        self.assertEqual(set(outputs), expected)
        for tensor in outputs.values():
            self.assertEqual(tuple(tensor.shape), tuple(images.shape))
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertGreaterEqual(float(tensor.min()), 0.0)
            self.assertLessEqual(float(tensor.max()), 1.0)

    def test_noise_is_deterministic_for_same_seed(self):
        images = torch.rand(1, 3, 16, 16)
        a = corruptions(images, seed=123)["gaussian_noise_std_0.03"]
        b = corruptions(images, seed=123)["gaussian_noise_std_0.03"]
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
