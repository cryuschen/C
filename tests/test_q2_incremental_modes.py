import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Q2'))
from analyze_incremental_modes import load_residuals
from analyze_pooled_spatial_modes import coefficient


class IncrementalModesTests(unittest.TestCase):
    def test_nine_residual_modes_and_label_sign(self):
        x, y, blocks, groups, ids, r2 = load_residuals()
        self.assertEqual(x.shape, (275, 9))
        self.assertEqual(len(np.unique(blocks)), 20)
        self.assertEqual(len(r2), 36)
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(coefficient(x, -y, blocks),
                                   -coefficient(x, y, blocks))


if __name__ == '__main__':
    unittest.main()
