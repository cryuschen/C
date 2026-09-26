import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from q3.common import FS,TIME,Trials,load_raw
from q3.events import build_trials
from q3.model import Candidate,states,design_bank,predict,penalties
from q3.main import past_slopes

def attach_past(ds):
    past_slopes(ds.key,ds)

from q3.fit import Bank


class RevisedModelTests(unittest.TestCase):
    def setUp(self):
        self.e=dict(cue=1,cue_duration_s=52/FS,target_s=568/FS)

    def test_target_does_not_write_into_memory(self):
        a=states(self.e,Candidate('N2',target_duration=.1))
        b=states(dict(self.e,target_s=2.7),Candidate('N2',target_duration=.4))
        np.testing.assert_array_equal(a[1],b[1])
        self.assertGreater(np.max(abs(a[0]-b[0])),.01)
        np.testing.assert_allclose(a[2][TIME<2.2],b[2][TIME<2.2],atol=1e-12)

    def test_retrieval_gate_ablation(self):
        c=Candidate('N2');off=replace(c,model='N2_no_retrieval')
        a=states(self.e,c);b=states(self.e,off)
        np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_array_equal(a[2][TIME<self.e['target_s']],b[2][TIME<self.e['target_s']])
        self.assertGreater(np.max(abs(a[2]-b[2])),1e-4)

    def test_prediction_ignores_response_and_click(self):
        c=Candidate('N2');coef=np.arange(27).reshape(9,3)
        a=predict(self.e,c,coef)
        b=predict(dict(self.e,response_s=4.9,click_side=-1,correctness=False),c,coef)
        np.testing.assert_array_equal(a,b)

    def test_past_context_is_strictly_pre_cue(self):
        ds,_=build_trials('A2');attach_past(ds);first=dict(ds.events[0]);expected=np.array(first['past_slopes']).copy()
        raw,labels,path=load_raw('A2');changed=raw.copy();changed[:3,first['cue_sample']:]+=1e6
        with patch('q3.main.load_raw',return_value=(changed,labels,path)):
            attach_past(ds)
        np.testing.assert_array_equal(expected,ds.events[0]['past_slopes'])

    def test_outer_fit_excludes_test_eeg_and_test_context(self):
        ds,_=build_trials('A2');attach_past(ds)
        ix=np.concatenate([np.flatnonzero(ds.blocks==b)[:1] for b in range(5)])
        one=Trials('A2',[dict(ds.events[i]) for i in ix],ds.x[ix].copy(),ds.baseline_scale[ix],ds.mode,ds.gap)
        two=Trials('A2',[dict(e) for e in one.events],one.x.copy(),one.baseline_scale,one.mode,one.gap)
        two.x[two.blocks==4]*=1e4
        for e in two.events:
            if e['block']==4:e['past_slopes']=(np.ones((3,3))*1e6).tolist()
        params=[Candidate('N2'),Candidate('N2',tau_m=2.,direction_penalty=10)]
        a=Bank(one,params,True).select((0,1,2,3),'N2')
        b=Bank(two,params,True).select((0,1,2,3),'N2')
        np.testing.assert_array_equal(a['coef'],b['coef']);self.assertEqual(a['selection'],b['selection'])

    def test_penalties_and_no_projection(self):
        c=Candidate('N2',direction_penalty=10)
        p=penalties(c,18)
        np.testing.assert_array_equal(np.flatnonzero(p==10),[1,4,7])
        h=design_bank([replace(c,model='N2_no_projection')],self.e)[0]
        self.assertEqual(np.max(abs(h[:,3:5])),0.)
        self.assertGreater(np.max(abs(h[:,6:])),0.)


if __name__=='__main__':unittest.main()
