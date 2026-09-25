import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Q2'))
from evaluate_pooled_discrimination import FEATURES, fold_standardized, predict


class PooledDiscriminationTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(16, 9))
        groups = np.repeat(['A1', 'A2', 'B1', 'B2'], 4)
        blocks = np.array([0, 0, 1, 1, 5, 5, 6, 6,
                           10, 10, 11, 11, 15, 15, 16, 16])
        y = np.tile([-1, 1], 8)
        ids = np.arange(1, 17)
        return x, y, blocks, groups, ids

    def test_outer_scaling_does_not_use_held_block(self):
        x, _, blocks, groups, _ = self.fixture()
        z = fold_standardized(x, blocks, groups, 0)
        changed = x.copy()
        changed[blocks == 0] += 200
        after = fold_standardized(changed, blocks, groups, 0)
        np.testing.assert_allclose(z[blocks != 0], after[blocks != 0])

    def test_held_block_label_does_not_change_its_score(self):
        x, y, blocks, groups, ids = self.fixture()
        base = predict(x, y, blocks, groups, ids)
        switched = y.copy()
        switched[blocks == 0] *= -1
        altered = predict(x, switched, blocks, groups, ids)
        for feature in FEATURES:
            a = base[(base.block == 0) & (base.model == feature)].logit.to_numpy()
            b = altered[(altered.block == 0) & (altered.model == feature)].logit.to_numpy()
            np.testing.assert_allclose(a, b)


if __name__ == '__main__':
    unittest.main()
