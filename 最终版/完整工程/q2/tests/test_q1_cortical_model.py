"""Data-lineage checks for the current Q2 model."""
from dataclasses import replace
import sys
import unittest
from pathlib import Path

import numpy as np

Q2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Q2))
import q1_cortical_model as model


class Q1CorticalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recordings = model.load_q1_results()

    def test_q1_lineage_and_feature_shape(self):
        self.assertEqual([len(item.cues) for item in self.recordings], [90, 95, 94, 92])
        x, y, groups, blocks, trial_ids, names = model.assemble(self.recordings, "before")
        self.assertEqual(x.shape, (371, 24))
        self.assertEqual(len(names), 24)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(set(np.unique(y)), {-1, 1})
        self.assertEqual(set(np.unique(groups)), {0, 1, 2, 3})
        self.assertEqual(set(np.unique(blocks)), {0, 1, 2, 3, 4})
        self.assertEqual(len(trial_ids), 371)

    def test_primary_representation_does_not_read_v7_waveforms(self):
        baseline = model.assemble(self.recordings, "before")[0]
        changed = [replace(item, v7=np.full_like(item.v7, 1e9))
                   for item in self.recordings]
        np.testing.assert_array_equal(model.assemble(changed, "before")[0], baseline)

    def test_cascade_bases_are_fixed_and_finite(self):
        bases, names = model.gamma_basis(self.recordings[0].times_ms)
        self.assertEqual(bases.shape, (269, 6))
        self.assertEqual(len(names), 6)
        self.assertTrue(np.isfinite(bases).all())
        self.assertTrue(np.allclose(bases[self.recordings[0].times_ms < 0], 0))


if __name__ == "__main__":
    unittest.main()
