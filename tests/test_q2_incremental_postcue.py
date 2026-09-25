import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Q2'))
from evaluate_incremental_postcue import predict, residualized_post, train_only_scale


class IncrementalPostcueTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(925)
        x = rng.normal(size=(16, 4))
        y = np.tile([-1, 1], 8)
        blocks = np.array([0, 0, 1, 1, 5, 5, 6, 6,
                           10, 10, 11, 11, 15, 15, 16, 16])
        groups = np.repeat(['A1', 'A2', 'B1', 'B2'], 4)
        ids = np.arange(1, 17)
        return x, y, blocks, groups, ids

    def test_past_regression_uses_only_outer_training(self):
        x, _, blocks, groups, _ = self.fixture()
        train = blocks != 0
        z = train_only_scale(x, groups, blocks, 0)
        residual = residualized_post(z, train)
        changed = x.copy()
        changed[~train] += 100
        zz = train_only_scale(changed, groups, blocks, 0)
        other = residualized_post(zz, train)
        np.testing.assert_allclose(residual[train], other[train])

    def test_held_labels_do_not_change_held_scores(self):
        x, y, blocks, groups, ids = self.fixture()
        first = predict(x, y, blocks, groups, ids)
        changed = y.copy()
        changed[blocks == 0] *= -1
        second = predict(x, changed, blocks, groups, ids)
        for name in first.model.unique():
            a = first[(first.block == 0) & (first.model == name)].logit.to_numpy()
            b = second[(second.block == 0) & (second.model == name)].logit.to_numpy()
            np.testing.assert_allclose(a, b)


if __name__ == '__main__':
    unittest.main()
