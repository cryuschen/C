"""Behavioral checks for the exploratory second-question direction analysis."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Q2'))
from optimize_direction_features import (
    CANDIDATES, audit_inputs, contrast_scores, cross_validate,
    feature_bank, permute_blocks, simulated_contrast,
)
from q2model.data import independent_data


class DirectionOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = independent_data(ROOT, 'A1')
        cls.bank = feature_bank(cls.ds.x, .203125)

    def test_feature_definition_and_input_audit(self):
        self.assertEqual(self.bank.shape, (67, 18))
        self.assertTrue(np.isfinite(self.bank).all())
        self.assertEqual(len(CANDIDATES['kernel_lateral3']), 3)
        self.assertEqual(len(audit_inputs()), 7)

    def test_outer_test_label_change_cannot_change_its_predictions(self):
        ds = self.ds
        original, choices = cross_validate(self.bank, ds.y, ds.blocks, ds.ids)
        changed = ds.y.copy()
        changed[ds.blocks == 0] *= -1
        rerun, altered_choices = cross_validate(self.bank, changed, ds.blocks, ds.ids)
        a = [r for r in original if r['fold'] == 0]
        b = [r for r in rerun if r['fold'] == 0]
        self.assertEqual([r['chosen'] for r in a], [r['chosen'] for r in b])
        np.testing.assert_array_equal([r['prediction'] for r in a],
                                      [r['prediction'] for r in b])
        np.testing.assert_allclose([r['logit'] for r in a], [r['logit'] for r in b])
        self.assertEqual(choices[0]['chosen'], altered_choices[0]['chosen'])
        for row in choices:
            self.assertFalse(set(row['train_ids']) & set(row['test_ids']))

    def test_block_permutation_preserves_counts(self):
        shuffled = permute_blocks(self.ds.y, self.ds.blocks, np.random.default_rng(5))
        self.assertFalse(np.array_equal(shuffled, self.ds.y))
        for block in np.unique(self.ds.blocks):
            use = self.ds.blocks == block
            self.assertEqual(int((shuffled[use] == 1).sum()),
                             int((self.ds.y[use] == 1).sum()))

    def test_illustrative_bridge_uses_matching_archived_trials(self):
        contrast = simulated_contrast()
        self.assertEqual(contrast.shape, (3, 269))
        self.assertTrue(np.isfinite(contrast).all())
        self.assertGreater(float(np.linalg.norm(contrast)), 0.)
        rows, traces = contrast_scores(self.ds, contrast)
        self.assertEqual(len(rows), 15)
        self.assertEqual(len(traces), 5)
        self.assertEqual({r['model'] for r in rows},
                         {'zero_direction', 'illustrative_forward', 'effective_M1'})


if __name__ == '__main__':
    unittest.main()
