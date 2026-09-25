import unittest
from dataclasses import replace
import numpy as np
from q3.data import (FS,TIME,load_raw,audit_events,causal_filter,stage_masks,sample_weights,build_trials,prefix_quality_trials)
from q3.model import Candidate,visual_response,slow_states,design_bank,predict
from q3.behavior import smoother,predict_all_labels


class EventTests(unittest.TestCase):
    def test_400_events_and_no_invented_correctness(self):
        count=0
        for key in ('A1','A2','B1','B2'):
            d,_,_=load_raw(key);events=audit_events(key,d);count+=len(events)
            self.assertEqual(len(events),100)
            self.assertTrue(all(e['correctness'] is None and e['timeout'] is None for e in events))
            if key.endswith('2'):
                self.assertTrue(all(e['click_sample']==e['platform_end_sample'] for e in events))
                self.assertTrue(all(e['response_status']=='explicit_click' for e in events))
            else:self.assertTrue(all(e['click_sample'] is None and e['response_status']=='platform_end_proxy' for e in events))
            if key=='B1':self.assertEqual([e['trial_id'] for e in events if e['cue_platform_disagree']],[49,87])
        self.assertEqual(count,400)

    def test_missing_click_is_unknown_not_timeout(self):
        d,_,_=load_raw('A2');d=d.copy();events=audit_events('A2',d);click=events[0]['click_sample']
        d[8,click]=0
        e=audit_events('A2',d)[0]
        self.assertEqual(e['response_status'],'unknown');self.assertIsNone(e['timeout'])
        self.assertIsNone(e['correctness']);self.assertFalse(e['valid_events'])

    def test_future_samples_cannot_modify_filtered_past(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(3,4000));changed=x.copy();changed[:,2500:]+=1e5
        np.testing.assert_array_equal(causal_filter(x)[:,:2500],causal_filter(changed)[:,:2500])
        np.testing.assert_array_equal(causal_filter(x[:,:2500]),causal_filter(x)[:,:2500])

    def test_windows_and_weights(self):
        ds,audit=build_trials('A2')
        for i,e in enumerate(ds.events):
            active=TIME<e['end_s'];self.assertTrue(np.isnan(ds.x[i,~active]).all())
            w=sample_weights(e['target_s'],e['end_s']);self.assertAlmostEqual(w.sum(),1.)
            for m in stage_masks(e['target_s'],e['end_s']):self.assertAlmostEqual(w[m].sum(),1/3)

    def test_behavior_future_and_window_length_do_not_enter_prefix(self):
        from unittest.mock import patch
        original,labels,path=load_raw('A2')
        events,x,_=prefix_quality_trials('A2');first=events[0]
        self.assertLessEqual(first['prefix_last_included_after_target_s'],.6)
        changed=original.copy();start=first['prefix_end_sample'];stop=first['response_sample']+50
        changed[:3,start:stop]+=1e7
        with patch('q3.data.load_raw',return_value=(changed,labels,path)):
            ev2,x2,_=prefix_quality_trials('A2')
        j=next(i for i,e in enumerate(ev2) if e['trial_id']==first['trial_id'])
        np.testing.assert_array_equal(x[0],x2[j])
        for key in ('saturation_fraction','PTP','max_jump','retained'):
            self.assertEqual(first[key],ev2[j][key])


class ModelTests(unittest.TestCase):
    def test_response_does_not_enter_prediction(self):
        e=dict(cue=1,cue_duration_s=52/FS,target_s=568/FS,response_s=3.5,click_side=1,correctness=True)
        changed=dict(e,response_s=4.5,click_side=-1,correctness=False)
        c=Candidate('M2');coef=np.arange(27).reshape(9,3)
        np.testing.assert_array_equal(predict(e,c,coef),predict(changed,c,coef))

    def test_stability_and_numerical_convergence(self):
        c=Candidate('M2',1.,.3,.05,.1,1.);t=np.arange(0,4,1/256)
        v=np.tile(np.exp(-t)[:,None],(1,3));m,p=slow_states(v,[c],2.,t)
        tf=np.arange(0,4,1/512);vf=np.tile(np.exp(-tf)[:,None],(1,3));mf,pf=slow_states(vf,[c],2.,tf)
        self.assertLess(np.max(abs(m-mf[:,::2])),.002)
        self.assertLess(np.max(abs(p-pf[:,::2])),.004)
        with self.assertRaises(ValueError):slow_states(v,[replace(c,feedback=1.)],2.,t)

    def test_no_memory_ablation_and_no_direct_projection(self):
        e=dict(cue=-1,cue_duration_s=52/FS,target_s=568/FS)
        a=design_bank([Candidate('M0'),Candidate('M1'),Candidate('M2_no_direct_memory')],e)
        self.assertTrue(np.all(a[0,:,3:]==0));self.assertTrue(np.all(a[1,:,6:]==0))
        self.assertTrue(np.all(a[2,:,3:6]==0));self.assertGreater(np.max(abs(a[2,:,6:])),0)

    def test_q2_visual_compatibility_before_target(self):
        from q2.q2model.generative import simulate
        from q3.model import shape_inputs
        expected=simulate(shape_inputs()[1],duration=52/FS)
        actual=visual_response(1,52,568)
        ix=(TIME>=0)&(TIME<2.)
        ref=np.stack([np.interp(TIME[ix],expected['t'],s) for s in expected['q'][0,2]],axis=1)
        np.testing.assert_allclose(actual[ix],ref,atol=1e-5,rtol=3e-3)

    def test_ridge_map_training_only(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(20,4));train=np.arange(15);test=np.arange(15,20)
        h=smoother(x,train,test,.1);y=rng.normal(size=20);changed=y.copy();changed[test]=1e8
        np.testing.assert_array_equal(h@y[train],h@changed[train])
        self.assertEqual(h.shape,(5,15))

    def test_test_eeg_cannot_change_outer_fit(self):
        from q3.data import Trials
        from q3.fit import Bank
        ds,_=build_trials('A2');idx=np.concatenate([np.flatnonzero(ds.blocks==b)[:2] for b in range(5)])
        original=Trials('A2',[ds.events[i] for i in idx],ds.x[idx].copy(),ds.baseline_scale[idx],'causal',.1)
        modified=Trials('A2',original.events,original.x.copy(),original.baseline_scale,'causal',.1)
        modified.x[modified.blocks==4]*=1e4
        params=[Candidate('M0'),Candidate('M2')]
        a=Bank(original,params).select((0,1,2,3),'M2')
        b=Bank(modified,params).select((0,1,2,3),'M2')
        np.testing.assert_array_equal(a['coef'],b['coef']);self.assertEqual(a['selection'],b['selection'])

    def test_test_rt_cannot_change_outer_rt_prediction(self):
        from q3.fit import LAMBDAS
        from types import SimpleNamespace
        rng=np.random.default_rng(17);features=rng.normal(size=(25,3));blocks=np.repeat(np.arange(5),5)
        ds=SimpleNamespace(blocks=blocks,events=[None]*25);maps={}
        for held in range(5):
            train=np.flatnonzero(blocks!=held);test=np.flatnonzero(blocks==held);inner=[]
            for b in range(5):
                if b==held:continue
                ti=np.flatnonzero((blocks!=held)&(blocks!=b));vi=np.flatnonzero(blocks==b)
                inner.append((ti,vi,np.stack([smoother(features,ti,vi,l) for l in LAMBDAS])))
            for name in ('ordinary','cognitive'):
                maps[(held,name)]=(train,test,np.stack([smoother(features,train,test,l) for l in LAMBDAS]),inner)
        y=rng.normal(.3,.1,(1,25));changed=y.copy();changed[:,blocks==0]+=2
        a,_=predict_all_labels(ds,maps,y);b,_=predict_all_labels(ds,maps,changed)
        for name in a:np.testing.assert_array_equal(a[name][:,blocks==0],b[name][:,blocks==0])


if __name__=='__main__':unittest.main()
