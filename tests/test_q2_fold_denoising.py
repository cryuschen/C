"""Regression and adversarial isolation tests for the paired Q2 pipeline."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/q2-fold-tests-mpl')
import sys
from pathlib import Path
import unittest
from dataclasses import replace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Q2'))
from q2model.data import independent_data
from q2model.model import train_model
from q2model.fold_denoising import (DenoiserCache, prepare_fold, fit_prepared_model,
                                    q1_module)
from run_q2_fold_denoising import run_fold, paired_intervals


class FoldDenoisingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = independent_data(ROOT, 'A1')
        cls.cache = DenoiserCache()

    def test_baseline_matches_existing_nested_selection(self):
        tr, te, pairs, _, _ = prepare_fold(self.ds, 0, 'baseline', self.cache)
        expected = train_model(tr, quick=True)
        actual = fit_prepared_model(tr, pairs, quick=True)
        self.assertEqual((actual.rank, actual.lam), (expected.rank, expected.lam))
        np.testing.assert_allclose(actual.predict(), expected.predict(), atol=1e-12)
        np.testing.assert_allclose(actual.features(te.x), expected.features(te.x), atol=1e-12)

    def test_all_nested_reference_ids_are_training_only(self):
        tr, te, pairs, d, audits = prepare_fold(self.ds, 0, 'denoised', self.cache)
        self.assertTrue(set(d.audit['reference_ids']) <= set(tr.ids))
        self.assertFalse(set(d.audit['train_ids']) & set(te.ids))
        for ((it, iv), row) in zip(pairs, audits):
            a = row['denoiser']
            self.assertEqual(set(a['train_ids']), set(it.ids))
            self.assertFalse(set(a['train_ids']) & (set(iv.ids) | set(te.ids)))
            self.assertTrue(set(a['reference_ids']) <= set(it.ids))
            self.assertTrue(set(a['selector_train_ids']) <= set(it.ids))
            self.assertTrue(set(a['selector_validation_ids']) <= set(it.ids))
        self.assertEqual(sorted(np.concatenate([iv.ids for _, iv in pairs])), sorted(tr.ids))

    def test_adapter_matches_new_q1_and_batch_independence(self):
        tr = self.ds.subset(self.ds.blocks != 0)
        te = self.ds.subset(self.ds.blocks == 0)
        d = self.cache.fit(tr); q = q1_module()
        expected = q.apply_v7(te.x, te.y, d.model, d.candidate, d.mean_delta)
        np.testing.assert_array_equal(d.transform(te.x), expected)
        np.testing.assert_array_equal(d.transform(te.x[:1]), d.transform(te.x)[:1])
        changed = te.x.copy(); changed[1:] *= 100
        np.testing.assert_array_equal(d.transform(te.x)[:1], d.transform(changed)[:1])

    def test_test_labels_do_not_affect_full_predictions(self):
        altered = self.ds.y.copy(); altered[self.ds.blocks == 0] *= -1
        a = run_fold(self.ds, 0, 'denoised', self.cache, quick=True)
        b = run_fold(self.ds.labels(altered), 0, 'denoised', self.cache, quick=True)
        for key in ('after', 'features', 'theta', 'predicted_erp', 'score', 'prediction'):
            np.testing.assert_array_equal(a[-1][key], b[-1][key])

    def test_test_eeg_cannot_change_training_or_inner_preprocessing(self):
        x = self.ds.x.copy(); x[self.ds.blocks == 0] = 1e6
        a = prepare_fold(self.ds, 0, 'denoised', self.cache)
        b = prepare_fold(replace(self.ds, x=x), 0, 'denoised', self.cache)
        np.testing.assert_array_equal(a[0].x, b[0].x)
        for pa, pb in zip(a[2], b[2]):
            for da, db in zip(pa, pb):
                np.testing.assert_array_equal(da.x, db.x)
        self.assertEqual(a[3].audit, b[3].audit)

    def test_identical_methods_have_zero_paired_difference(self):
        rows = []
        for branch in ('baseline', 'denoised'):
            for i in range(20):
                y = -1 if i % 2 == 0 else 1
                rows.append(dict(dataset='test', branch=branch, model='test',
                                 fold=i//4, trial_id=i, truth=y, prediction=y, score=(y+1)/2))
        _, delta = paired_intervals(rows, n_boot=20)
        for metric in ('BA', 'AUC'):
            for suffix in ('change', 'change_low', 'change_high'):
                self.assertEqual(delta[0][metric+'_'+suffix], 0)


if __name__ == '__main__':
    unittest.main()
