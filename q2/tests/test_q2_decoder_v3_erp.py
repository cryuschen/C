"""Data-flow checks for the pre-registered fixed ERP template decoder."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'q2'))
from decoder_v3_erp import (N_DCT, POST_MASK, TIMES, dct_features, decode_one,
                            permute_within_blocks, remove_past_prediction)


class DecoderV3ERPTests(unittest.TestCase):
    def test_lateral_dct_tracks_f3_minus_f4(self):
        x = np.zeros((2, 3, 64))
        x[0, 1] = 2
        x[0, 2] = -2
        x[1] = -x[0]
        feature = dct_features(x)
        self.assertEqual(feature.shape, (2, 3 * N_DCT))
        self.assertGreater(feature[0, 2 * N_DCT], 0)
        self.assertAlmostEqual(feature[0, 2 * N_DCT], -feature[1, 2 * N_DCT])
        self.assertTrue(np.all(TIMES[POST_MASK] >= 0))

    def test_outer_test_labels_do_not_change_its_scores(self):
        rng = np.random.default_rng(9)
        labels = np.tile([-1, 1, -1, 1], 5)
        data = dict(pre=rng.normal(size=(20, 18)), post=rng.normal(size=(20, 18)),
                    blocks=np.repeat(np.arange(5), 4), ids=np.arange(1, 21))
        original, _ = decode_one(data, labels)
        changed = labels.copy()
        changed[data['blocks'] == 0] *= -1
        altered, _ = decode_one(data, changed)
        for name in original:
            np.testing.assert_allclose(original[name][data['blocks'] == 0],
                                       altered[name][data['blocks'] == 0])

    def test_past_regression_fits_training_only(self):
        rng = np.random.default_rng(10)
        pre = rng.normal(size=(60, 18))
        post = .5 * pre + rng.normal(scale=.2, size=(60, 18))
        train_a, test_a = remove_past_prediction(pre[:50], pre[50:], post[:50], post[50:])
        train_b, _ = remove_past_prediction(pre[:50], pre[50:] * 100,
                                            post[:50], post[50:] * -100)
        np.testing.assert_allclose(train_a, train_b)
        self.assertLess(np.linalg.norm(train_a), np.linalg.norm(post[:50]))

    def test_permutation_preserves_each_block_class_counts(self):
        labels = np.array([-1, -1, 1, 1] * 5)
        data = {'A1':dict(y=labels, blocks=np.repeat(np.arange(5), 4))}
        shuffled = permute_within_blocks(data, np.random.default_rng(11))['A1']
        for block in range(5):
            index = data['A1']['blocks'] == block
            self.assertEqual(int(np.sum(labels[index] == 1)), int(np.sum(shuffled[index] == 1)))


if __name__ == '__main__':
    unittest.main()
