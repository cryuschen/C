import unittest
import numpy as np
from Q1.independent_validation import synthetic,unseen_contamination,reference,conservative_output,shrink

class IndependentRecoveryTests(unittest.TestCase):
    def test_zero_noise_and_zero_strength_are_exact(self):
        y=np.tile([-1,1],10);x=synthetic(y,12);model=reference(x,y)
        for family in ('asymmetric_burst','direction_step'):
            np.testing.assert_array_equal(unseen_contamination(x,y,7,0,family),x)
        np.testing.assert_array_equal(conservative_output(x,y,model,'unused',0,2,1),x)
        np.testing.assert_array_equal(shrink(x,y,model,0),x)

    def test_gate_prevents_small_deviation_changes(self):
        y=np.tile([-1,1],10);x=synthetic(y,12);model=reference(x,y)
        candidate=x+100
        np.testing.assert_array_equal(conservative_output(x,y,model,'unused',.25,2,1e10,candidate),x)

    def test_shrink_uses_training_reference(self):
        y=np.tile([-1,1],10);x=synthetic(y,12);model=reference(x,y)
        np.testing.assert_allclose(shrink(x*10,y,model,1),np.stack([model[0][c] for c in y]))
