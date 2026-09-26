import unittest
from dataclasses import replace
import numpy as np
from q3.decision import target_match,simulate,SCENARIOS

class MatchingTests(unittest.TestCase):
    def test_layout_swap_reverses_evidence(self):
        cues=np.array([-1,-1,1,1]);layouts=np.array([[-1,1],[1,-1],[-1,1],[1,-1]])
        e=target_match(cues,layouts)
        np.testing.assert_allclose(e,[-1,1,1,-1],atol=1e-12)
        np.testing.assert_allclose(target_match(cues,layouts[:,::-1]),-e,atol=1e-12)
        np.testing.assert_allclose(target_match(cues,np.ones((4,2),int)),0,atol=1e-12)

    def test_matching_decisions_scored_from_layout(self):
        s,rows,_,_=simulate(n=40,gamma=20,sigma=0,evidence_mode='matching')
        self.assertEqual(s['correct_rate'],1.)
        for r in rows:
            expected=1 if r['target_right_shape']==r['cue'] else -1
            self.assertEqual(r['choice'],expected)

    def test_low_evidence_is_single_factor(self):
        nominal=SCENARIOS[0];low=next(s for s in SCENARIOS if s[0]=='low_evidence')
        self.assertEqual(nominal[2],low[2]);self.assertEqual(nominal[4:],low[4:]);self.assertNotEqual(nominal[3],low[3])
