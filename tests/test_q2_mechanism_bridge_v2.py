"""Checks for the neural-vs-ocular model comparison and short-guard variant."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Q2'))
from compare_neural_eye_models import decode, fit_gain, select_ridge, templates
from q2model.data import WEIGHTS, independent_data
from q2model.data_variants import filter_tail_ratio, independent_data_variant


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = independent_data(ROOT, 'A1')
        cls.shapes = templates()

    def test_templates_have_matching_weighted_norm_and_distinct_shapes(self):
        for h in self.shapes.values():
            self.assertAlmostEqual(float(np.sum(WEIGHTS * h ** 2)), 1., places=10)
        self.assertGreater(float(np.linalg.norm(self.shapes['neural_mass'] -
                                                self.shapes['ocular_step'])), 1.)

    def test_test_labels_do_not_change_its_model_score_or_ridge(self):
        h = self.shapes['neural_mass']
        a = decode(self.ds, h)
        changed = self.ds.y.copy()
        changed[self.ds.blocks == 0] *= -1
        b = decode(self.ds, h, changed)
        ar = [row for row in a if row['fold'] == 0]
        br = [row for row in b if row['fold'] == 0]
        np.testing.assert_allclose([row['logit'] for row in ar],
                                   [row['logit'] for row in br], atol=1e-12)
        training = self.ds.subset(self.ds.blocks != 0)
        self.assertEqual(select_ridge(training, h), select_ridge(training.labels(
            training.y.copy()), h))

    def test_gain_is_regularized_and_zero_signal_remains_zero(self):
        h = self.shapes['neural_mass']
        delta = np.stack([h, 2 * h, -h])
        np.testing.assert_allclose(fit_gain(delta, h, 0), delta, atol=1e-10)
        self.assertLess(np.linalg.norm(fit_gain(delta, h, 1)),
                        np.linalg.norm(delta))
        np.testing.assert_array_equal(fit_gain(np.zeros_like(delta), h, .1),
                                      np.zeros_like(delta))

    def test_short_guard_has_small_filter_tail_and_more_valid_trials(self):
        self.assertLess(filter_tail_ratio(.3, 8), 1e-4)
        extended = independent_data_variant(ROOT, 'A1', .3, 8)
        self.assertEqual(len(extended.y), 84)
        self.assertTrue(np.isin(self.ds.ids, extended.ids).all())


if __name__ == '__main__':
    unittest.main()
