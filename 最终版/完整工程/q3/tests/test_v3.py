import unittest
from unittest.mock import patch
from pathlib import Path
import json
import numpy as np
from q3.common import load_raw,FS,TIME,causal_filter
from q3.events import audit_events,window_policy,build_trials,product_limit
from q3.model import Candidate,states,predict
from q3.decision import simulate,aligned_evidence,REFERENCE

ROOT=Path(__file__).resolve().parents[2]


class ResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=load_raw('A2')[0]
        cls.events=audit_events('A2',cls.raw)

    def test_real_labels_remain_unknown(self):
        for key in ('A1','A2','B1','B2'):
            ev=audit_events(key,load_raw(key)[0])
            self.assertEqual(len(ev),100)
            self.assertTrue(all(e['correctness'] is None and e['timeout'] is None for e in ev))
            status='explicit_click' if key.endswith('2') else 'platform_end_proxy'
            self.assertTrue(all(e['response_status']==status for e in ev))

    def test_wrong_correct_unknown_same_window(self):
        e=self.events[10]
        windows=[window_policy(dict(e,correctness=x)) for x in (True,False,None)]
        self.assertEqual(len({r['end_s'] for r in windows}),1)
        self.assertTrue(all(r['window_status']=='response_minus_gap' for r in windows))

    def test_missing_marker_keeps_candidate_window_not_timeout(self):
        x=self.raw.copy();e=self.events[10]
        x[8,e['click_sample']]=0
        r=window_policy(audit_events('A2',x)[10])
        self.assertTrue(r['valid_events']);self.assertEqual(r['end_s'],r['target_s']+2)
        self.assertIsNone(r['timeout']);self.assertIsNone(r['behavior_event'])
        self.assertFalse(r['behavior_survival_eligible'])

    def test_missing_response_can_retain_quality_eligible_EEG(self):
        ds,_=build_trials('A2');e=ds.events[0];x=self.raw.copy()
        x[8,e['click_sample']]=0
        # A fully declared synthetic clean EEG fixture tests retention, not biology.
        end=e['target_sample']+2*FS
        x[:3,e['cue_sample']-25*FS:end]=0
        out,rows=build_trials('A2',raw=x)
        r=next(a for a in rows if a['trial_id']==e['trial_id'])
        self.assertTrue(r['retained']);self.assertIn(e['trial_id'],out.ids)

    def test_late_click_is_administrative_not_experimental_timeout(self):
        e=self.events[0];r=window_policy(e,horizon=.5)
        self.assertEqual(r['behavior_event'],0);self.assertEqual(r['behavior_duration_s'],.5)
        self.assertTrue(r['behavior_survival_eligible']);self.assertIsNone(r['timeout'])

    def test_proxy_not_survival_event(self):
        e=audit_events('A1',load_raw('A1')[0])[10];r=window_policy(e)
        self.assertFalse(r['behavior_survival_eligible']);self.assertIsNone(r['behavior_event'])

    def test_missing_target_not_invented(self):
        x=self.raw.copy();e=self.events[10]
        x[8,e['target_sample']:e['platform_end_sample']]=0
        r=window_policy(audit_events('A2',x)[10])
        self.assertIsNone(r['end_s']);self.assertFalse(r['valid_events'])

    def test_future_EEG_cannot_change_past(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(3,2048));y=x.copy();y[:,1024:]+=1e7
        np.testing.assert_array_equal(causal_filter(x)[:,:1024],causal_filter(y)[:,:1024])

    def test_product_limit_censoring_not_failure(self):
        r=product_limit([1,1.5,2],[1,0,1])
        np.testing.assert_allclose([x['survival'] for x in r],[2/3,2/3,0])


class UnifiedModelTests(unittest.TestCase):
    def test_neural_states_ignore_click_and_truth(self):
        e=dict(cue=1,cue_duration_s=52/FS,target_s=568/FS)
        changed=dict(e,click_side=-1,response_s=4.3,correctness=False,timeout=True)
        co=np.arange(27).reshape(9,3)
        np.testing.assert_array_equal(predict(e,REFERENCE,co),predict(changed,REFERENCE,co))

    def test_target_not_written_to_memory(self):
        e=dict(cue=1,cue_duration_s=52/FS,target_s=568/FS)
        m=states(e,REFERENCE)[1];m2=states(dict(e,target_s=3.),REFERENCE)[1]
        np.testing.assert_array_equal(m,m2)

    def test_mirror_equivalence_used_in_balanced_simulation(self):
        e=dict(cue=1,cue_duration_s=52/FS,target_s=568/FS)
        _,a=aligned_evidence(REFERENCE,e);_,b=aligned_evidence(REFERENCE,dict(e,cue=-1))
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_no_drift_no_noise_is_censored_not_zero_RT(self):
        s,rows,paths,curves=simulate(gamma=0,sigma=0,n=40)
        self.assertEqual(s['unresolved_rate'],1.)
        self.assertIsNone(s['censor_aware_median_s'])
        self.assertEqual(s['restricted_mean_decision_time_s'],2.)
        self.assertTrue(all(np.isnan(r['decision_time_s']) for r in rows))
        self.assertTrue(all(r['observation_time_s']==2 and r['event_observed']==0 for r in rows))

    def test_negative_evidence_produces_wrong_decisions_without_filtering(self):
        s,rows,_,_=simulate(gamma=-20,sigma=0,n=40)
        self.assertGreater(s['error_rate'],.99)
        self.assertTrue(all(r['choice']==-r['correct_side'] for r in rows))

    def test_probabilities_and_first_passage_bounds(self):
        s,rows,_,curves=simulate(n=400)
        self.assertAlmostEqual(s['correct_rate']+s['error_rate']+s['unresolved_rate'],1)
        for r in rows:
            if r['event_observed']:self.assertTrue(0<r['decision_time_s']<=s['deadline_s'])
            else:self.assertTrue(np.isnan(r['decision_time_s']))
        for r in curves:self.assertAlmostEqual(r['correct_cdf']+r['error_cdf']+r['unresolved_survival'],1.)

    def test_V2_fitted_waveform_reproduced(self):
        ds,_=build_trials('A2');choices=json.loads((ROOT/'q3_result/reference/A2_stimulus_choices.json').read_text(encoding='utf8'))
        f=next(c for c in choices if c['model']=='N2' and c['held_block']==0)
        i=int(np.flatnonzero(ds.blocks==0)[0]);old=np.load(ROOT/'q3_result/reference/A2_stimulus_waves.npz')
        p=predict(ds.events[i],Candidate(**f['parameters']),np.array(f['coef']))
        np.testing.assert_allclose(p,old['N2'][i],atol=1e-8,rtol=1e-8)


if __name__=='__main__':unittest.main()
