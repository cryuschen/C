"""Actual problem-image extraction and forward-response invariants."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Q2'))
from q2model.stimulus_shape import cue_images, shape_inputs
from q2model.mechanism import neural_forward


class StimulusModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cues = cue_images(ROOT)
        cls.encoded = shape_inputs(cls.cues)

    def test_triangle_mirrors_around_shared_fixation(self):
        cues = self.cues
        self.assertEqual(cues['source_size'], (154, 137))
        self.assertEqual(cues['center_x'], 70)
        self.assertGreater(cues['circle'].sum(), 0.)
        self.assertGreater(cues['right_triangle'].sum(), 0.)
        np.testing.assert_allclose(cues['right_triangle'].sum(),
                                   cues['left_triangle'].sum(), atol=1e-10)
        right_y, right_x = np.nonzero(cues['right_triangle'])
        for y, x in zip(right_y, right_x):
            self.assertAlmostEqual(cues['right_triangle'][y, x],
                                   cues['left_triangle'][y, 2 * cues['center_x'] - x])
        np.testing.assert_allclose(cues['left'] - cues['left_triangle'], cues['circle'])
        np.testing.assert_allclose(cues['right'] - cues['right_triangle'], cues['circle'])

    def test_spatial_response_is_mirror_selective_but_not_oracle_label(self):
        response = self.encoded['activation']
        self.assertGreater(response[0, 0], response[0, 1])
        self.assertGreater(response[1, 1], response[1, 0])
        np.testing.assert_allclose(response[0], response[1, ::-1], atol=1e-12)
        np.testing.assert_allclose(self.encoded['selectivity'][0],
                                   -self.encoded['selectivity'][1], atol=1e-12)

    def test_neural_mass_generates_finite_distinct_scalp_curves(self):
        fwd = neural_forward(self.encoded['inputs'])
        self.assertTrue(np.isfinite(fwd['eeg']).all())
        self.assertLess(float(max(fwd['eigenvalues'].real)), 0.)
        self.assertGreater(float(np.max(np.abs(fwd['eeg'][1] - fwd['eeg'][0]))), .05)


if __name__ == '__main__':
    unittest.main()
