import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Q2'))
from evaluate_covariance_geometry import decode, log_covariance_features


class CovarianceGeometryTests(unittest.TestCase):
    def test_log_covariance_is_baseline_invariant(self):
        rng = np.random.default_rng(77)
        x = rng.normal(size=(5, 3, 50))
        a = log_covariance_features(x)
        b = log_covariance_features(x + np.array([1., 2., 3.])[None, :, None])
        self.assertEqual(a.shape, (5, 6))
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_held_labels_do_not_change_held_scores(self):
        rng = np.random.default_rng(78)
        x = rng.normal(size=(16, 12))
        y = np.tile([-1, 1], 8)
        blocks = np.array([0, 0, 1, 1, 5, 5, 6, 6,
                           10, 10, 11, 11, 15, 15, 16, 16])
        groups = np.repeat(['A1', 'A2', 'B1', 'B2'], 4)
        ids = np.arange(1, 17)
        before = decode(x, y, blocks, groups, ids)
        switched = y.copy()
        switched[blocks == 0] *= -1
        after = decode(x, switched, blocks, groups, ids)
        for model in before.model.unique():
            a = before[(before.block == 0) & (before.model == model)].logit.to_numpy()
            b = after[(after.block == 0) & (after.model == model)].logit.to_numpy()
            np.testing.assert_allclose(a, b)


if __name__ == '__main__':
    unittest.main()
