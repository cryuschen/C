import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Q2'))
from evaluate_residual_9d import decode


class ResidualNineTests(unittest.TestCase):
    def test_held_labels_cannot_change_scores(self):
        rng = np.random.default_rng(906)
        x = rng.normal(size=(16, 18))
        y = np.tile([-1, 1], 8)
        blocks = np.array([0, 0, 1, 1, 5, 5, 6, 6,
                           10, 10, 11, 11, 15, 15, 16, 16])
        groups = np.repeat(['A1', 'A2', 'B1', 'B2'], 4)
        ids = np.arange(1, 17)
        a = decode(x, y, blocks, groups, ids)
        altered = y.copy()
        altered[blocks == 0] *= -1
        b = decode(x, altered, blocks, groups, ids)
        for model in a.model.unique():
            aa = a[(a.block == 0) & (a.model == model)].logit.to_numpy()
            bb = b[(b.block == 0) & (b.model == model)].logit.to_numpy()
            np.testing.assert_allclose(aa, bb)


if __name__ == '__main__':
    unittest.main()
