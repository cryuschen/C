"""Mechanistic identities and train/test separation for the new Q2 pipeline."""
import os
import sys
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'q2'))
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'q2_result')
import numpy as np
from q2model.data import TIMES,WEIGHTS,independent_data,means
from q2model.generative import (encode_shapes,simulate,observation_sources,ERPBank,Candidate,
                               empirical_basis,delta_metrics,wave_metrics)
from q2model.validation import prepare_decoder,feature_fold,lda,standardize


class Q2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds=independent_data(ROOT,'A1')
        cls.bank=ERPBank([Candidate('test',1.,.1,empirical_basis('step'))],lambdas=[.1,1.])
        cls.prepared=prepare_decoder(cls.ds,cls.bank,np.zeros(len(cls.ds.y)))
        cls.train=cls.ds.blocks!=4;cls.test=~cls.train
        cls.shape=encode_shapes(ROOT)

    def test_mirrored_encoding_and_perturbations(self):
        a=self.shape['inputs']
        np.testing.assert_allclose(a[0,:2],a[1,:2][::-1],atol=1e-10)
        self.assertGreater(a[0,0],a[0,1])
        self.assertLess(a[2,1],a[1,1])
        self.assertLess(a[3,1],a[1,1])

    def test_rest_and_shape_abolition(self):
        sim=simulate(np.zeros((1,3)),max_step=.004)
        self.assertLess(sim['eigenmax'],0)
        self.assertLess(abs(sim['q']).max(),1e-10)
        f=np.tile(self.shape['inputs'][:2].mean(0),(2,1))
        sim=simulate(f,max_step=.004)
        np.testing.assert_allclose(sim['q'][0],sim['q'][1],atol=1e-10)
        self.assertLess(abs(sim['q'][...,sim['t']<0]).max(),1e-10)

    def test_integration_convergence(self):
        a=simulate(self.shape['inputs'][:1],max_step=.004)
        b=simulate(self.shape['inputs'][:1],max_step=.002)
        self.assertLess(np.linalg.norm(a['q']-b['q'])/np.linalg.norm(b['q']),1e-4)
        self.assertEqual(observation_sources(b).shape,(1,len(TIMES),9))

    def test_cached_statistics_equal_direct_fit(self):
        ds=self.ds;b=self.bank
        p=b.prepare(ds.x)
        direct=b.stats(means(ds.x[self.train],ds.y[self.train]))
        cached=b.prepared_stats(p,ds.y,self.train)
        np.testing.assert_allclose(direct,cached,atol=1e-10,rtol=1e-10)

    def test_test_labels_do_not_change_predictions(self):
        d=self.prepared;y=self.ds.y.copy()
        f,m=feature_fold(d,self.bank,y,self.train,self.test)
        y[self.test]*=-1
        g,n=feature_fold(d,self.bank,y,self.train,self.test)
        self.assertEqual(m,n)
        for name in f:
            np.testing.assert_allclose(lda(f[name][0],y[self.train],f[name][1]),
                                       lda(g[name][0],y[self.train],g[name][1]),atol=1e-10)

    def test_test_waveforms_do_not_change_training(self):
        altered=self.ds.x.copy();altered[self.test]*=100
        new=prepare_decoder(replace(self.ds,x=altered),self.bank,np.zeros(len(self.ds.y)))
        f,m=feature_fold(self.prepared,self.bank,self.ds.y,self.train,self.test)
        g,n=feature_fold(new,self.bank,self.ds.y,self.train,self.test)
        self.assertEqual(m,n)
        for name in f:
            np.testing.assert_allclose(f[name][0],g[name][0],equal_nan=True)

    def test_reduced_features_and_contrast_selection_ignore_outer_test(self):
        from q2model.generative import select_contrast_shrinkage
        f,_=feature_fold(self.prepared,self.bank,self.ds.y,self.train,self.test)
        self.assertEqual(f['projection6'][1].shape[1],6)
        self.assertEqual(f['direction3'][1].shape[1],3)
        np.testing.assert_allclose(f['direction3'][1],f['projection6'][1][:,[1,3,5]])
        a,loss,_=select_contrast_shrinkage(self.bank,self.prepared['projected'],self.ds,self.train)
        altered=self.ds.x.copy();altered[self.test]*=1000
        labels=self.ds.y.copy();labels[self.test]*=-1
        ds2=replace(self.ds,x=altered,y=labels)
        b,loss2,records=select_contrast_shrinkage(self.bank,self.bank.prepare(altered),ds2,self.train)
        self.assertEqual(a,b);np.testing.assert_allclose(loss,loss2)
        for r in records:
            self.assertFalse(set(r['train_ids'])&set(self.ds.ids[self.test]))
            self.assertFalse(set(r['validation_ids'])&set(self.ds.ids[self.test]))

    def test_no_q1_path_in_strict_decoder(self):
        # Decoder receives raw-block Dataset only and must not open a V7 file.
        with patch('numpy.load',side_effect=AssertionError('Unexpected Q1 read')):
            feature_fold(self.prepared,self.bank,self.ds.y,self.train,self.test)

    def test_symmetric_observation_constraint(self):
        b=ERPBank(self.bank.candidates,lambdas=[1.],symmetric=True)
        obs=means(self.ds.x,self.ds.y);pred=b.predict(b.fit(obs,0),0)
        np.testing.assert_allclose(pred[:,1],pred[:,2],atol=1e-10)

    def test_preprocessing_blocks_and_past_are_disjoint(self):
        from q2model.data import load_raw,FS
        raw,on,_,_=load_raw(ROOT,'A1')
        edges=[0]+[(int(on[j-1])+int(on[j]))//2 for j in (20,40,60,80)]+[raw.shape[1]]
        for trial,block in zip(self.ds.ids,self.ds.blocks):
            self.assertGreaterEqual(on[trial-1]-int(24.25*FS),edges[block])
            self.assertLessEqual(on[trial-1]+205,edges[block+1]-24*FS)

    def test_metric_definitions_and_missing_values(self):
        y=np.stack([np.tile(TIMES,(3,1)),np.tile(2*TIMES,(3,1))])
        self.assertAlmostEqual(delta_metrics(y,y)['S_delta'],1.)
        self.assertAlmostEqual(wave_metrics(y,y)['R2'],1.)
        a=np.array([[np.nan,2.],[np.nan,3.]]);b=np.array([[1e9,np.nan]])
        aa,bb=standardize(a,b)
        self.assertTrue(np.isfinite(aa).all() and np.isfinite(bb).all())


if __name__=='__main__':unittest.main()
