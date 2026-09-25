"""Regression and block-control invariants for pooled scalp-mode analysis."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Q2'))
from analyze_pooled_spatial_modes import coefficient, load_matrices


class PooledModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.post, cls.pre, cls.y, cls.blocks, cls.groups, cls.ids = load_matrices()

    def test_shapes_and_block_counts(self):
        self.assertEqual(self.post.shape, (275, 9))
        self.assertEqual(self.pre.shape, (275, 9))
        self.assertEqual(len(np.unique(self.blocks)), 20)
        self.assertTrue(np.isfinite(self.post).all())

    def test_blockwise_offsets_do_not_change_coefficients(self):
        original = coefficient(self.post, self.y, self.blocks)
        shifted = self.post.copy()
        for b in np.unique(self.blocks):
            shifted[self.blocks == b] += np.arange(9) * (b + 1)
        np.testing.assert_allclose(coefficient(shifted, self.y, self.blocks),
                                   original, atol=1e-12)

    def test_swapping_cue_coding_changes_only_sign(self):
        original = coefficient(self.post, self.y, self.blocks)
        np.testing.assert_allclose(coefficient(self.post, -self.y, self.blocks),
                                   -original, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
