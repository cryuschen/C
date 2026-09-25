"""Q2 tests focus on isolation, algebra, optimizer behavior, and full permutation replay."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Q2'))
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from q2model.data import B,WEIGHTS,FS,TIMES,means,independent_data
from q2model.model import KernelFactory,coefficients,predict_erp,fit_erp,allowed
from q2model.mechanism import spatial_encoding,neural_forward
from q2model.validation import evaluate_split,permutation_job,holm,metrics
ROOT=Path(__file__).resolve().parents[1]


class Q2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds=independent_data(ROOT,'A1')

    def test_spatial_coordinates_reversible(self):
        np.testing.assert_allclose(B.T@B,np.eye(3),atol=1e-14)
        x=np.random.default_rng(1).normal(size=(3,269))
        np.testing.assert_allclose(B@(B.T@x),x,atol=1e-14)

    def test_raw_blocks_have_disjoint_filter_support_and_exact_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'audit').mkdir()
            ds=independent_data(ROOT,'A1',out=directory)
            audit=pd.read_csv(Path(directory,'audit/A1_events_guard24.csv'))
            self.assertEqual(len(ds.x),67)
            self.assertEqual(len(audit),100)
            blocks=audit.groupby('block')[['filter_start','filter_end_exclusive']].first()
            for i in range(4):self.assertEqual(blocks.iloc[i,1],blocks.iloc[i+1,0])
            valid=audit[audit.retained]
            self.assertTrue((valid.epoch_start>=valid.filter_start+24*FS).all())
            self.assertTrue((valid.epoch_end_exclusive<=valid.filter_end_exclusive-24*FS).all())
            np.testing.assert_array_equal(valid.trial_id.to_numpy(),ds.ids)

    def test_kernel_scaling_projection_closure_and_padding(self):
        theta=[65.,50.,135.,350.];f=KernelFactory();h=f.kernel(theta,3)
        np.testing.assert_allclose(np.einsum('tr,t,tr->r',h,WEIGHTS,h),np.ones(3),atol=1e-12)
        h2=KernelFactory(nfft=32768).kernel(theta,3)
        self.assertLess(np.max(abs(h-h2)),2e-4)
        a=np.arange(9).reshape(3,3)+1;d=a/3
        aa,dd=coefficients(predict_erp(a,d,h),h,0)
        np.testing.assert_allclose(aa,a,atol=1e-8);np.testing.assert_allclose(dd,d,atol=1e-8)
        # Discrete causal transfer has no wrap-around tail above numerical tolerance.
        from scipy.fft import irfft
        response=irfft(f.exp_transfer(20)*f.alpha_transfer(50),n=f.nfft)
        self.assertLess(np.max(abs(response[-64:])),1e-12)
        self.assertAlmostEqual(response.sum(),1,places=12)

    def test_model_fit_refines_grid_and_respects_constraints(self):
        h=KernelFactory().kernel([77,45,140,370],3)
        erp=predict_erp(np.array([[25,10,8],[5,8,4],[10,4,9]]),np.ones((3,3))*2,h)
        f=fit_erp(erp,.203125,3,.01,quick=True)
        self.assertTrue(allowed(f.theta,3))
        self.assertLess(f.diagnostics['objective'],f.diagnostics['grid_objective']*.9999)
        self.assertTrue(any(t['success'] for t in f.diagnostics['starts']))

    def test_testing_directions_do_not_affect_predictions(self):
        train=self.ds.subset(self.ds.blocks!=0);test=self.ds.subset(self.ds.blocks==0)
        a=evaluate_split(train,test,True,False)
        b=evaluate_split(train,test.labels(-test.y),True,False)
        np.testing.assert_allclose(a['features'],b['features'])
        np.testing.assert_allclose(a['predictions']['mechanism_selected'][1],b['predictions']['mechanism_selected'][1])
        for row in a['fit'].diagnostics['inner_validation']:
            self.assertFalse(set(row['train_ids'])&set(row['validation_ids']))
            self.assertFalse(set(row['train_ids'])&set(test.ids))

    def test_permutation_retrains_five_outer_models_and_resumes_identically(self):
        import q2model.validation as validation
        original=validation.train_model;calls=[]
        def track(ds,*args,**kwargs):
            calls.append((ds.ids.copy(),ds.y.copy()))
            return original(ds,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory,patch.object(validation,'train_model',side_effect=track):
            a=permutation_job(self.ds,0,42,True,directory)
            self.assertEqual(len(calls),5)
            b=permutation_job(self.ds,0,42,True,directory)
            self.assertEqual(len(calls),5)
            self.assertEqual(a['BA'],b['BA'])
            np.testing.assert_allclose(a['scores'],b['scores'])
            self.assertEqual(len(a['folds']),5)
            y_by_id=dict(zip(a['trial_ids'],a['permuted_truth']))
            for block in range(5):
                mask=self.ds.blocks==block
                self.assertEqual(sum(y_by_id[int(i)]==1 for i in self.ds.ids[mask]),int((self.ds.y[mask]==1).sum()))
            self.assertTrue(any(not np.array_equal(y,self.ds.y[np.isin(self.ds.ids,ids)]) for ids,y in calls))
        with tempfile.TemporaryDirectory() as directory:
            c=permutation_job(self.ds,0,42,True,directory)
            np.testing.assert_allclose(a['scores'],c['scores'],atol=1e-12)

    def test_mechanism_mirror_and_stable_equilibrium(self):
        s=spatial_encoding()
        np.testing.assert_array_equal(s['images'][0],s['images'][1][:,::-1])
        self.assertLess(s['eta'][0],0);self.assertGreater(s['eta'][1],0)
        self.assertAlmostEqual(s['eta'][0],-s['eta'][1],places=8)
        self.assertLess(s['inputs'][2,1],s['inputs'][1,1])
        d=neural_forward(s['inputs'][:2])
        self.assertLess(max(d['eigenvalues'].real),0)
        self.assertTrue(np.isfinite(d['eeg']).all())

    def test_missing_class_and_multiple_comparisons(self):
        self.assertTrue(np.isnan(metrics(np.ones(5),np.ones(5),np.ones(5))['AUC']))
        self.assertTrue(np.isnan(metrics(np.ones(5),np.ones(5),np.ones(5))['BA']))
        with self.assertRaises(ValueError):means(self.ds.x,np.ones(len(self.ds.x)))
        np.testing.assert_allclose(holm([.01,.04,.03,.2]),[.04,.09,.09,.2])




class Q2CompletedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.out=ROOT/'Q2_result'
        path=cls.out/'manifest.json'
        if not path.exists():raise unittest.SkipTest('Run the full Q2 pipeline first')
        cls.manifest=json.loads(path.read_text())
        if cls.manifest.get('status')!='complete' or cls.manifest['config']['quick']:
            raise unittest.SkipTest('Full Q2 run has not completed')

    def test_full_counts_and_every_saved_hash(self):
        import json
        from q2model.data import sha
        self.assertEqual(self.manifest['config']['permutations'],999)
        self.assertEqual(self.manifest['config']['refit_bootstrap'],200)
        for p,digest in self.manifest['source'].items():self.assertEqual(sha(ROOT/p),digest,p)
        for p,digest in self.manifest['inputs'].items():self.assertEqual(sha(ROOT/p),digest,p)
        for p,digest in self.manifest['output_hashes'].items():self.assertEqual(sha(self.out/p),digest,p)
        for key in ['A1','A2','B1','B2']:
            files=list((self.out/'checkpoints/permutations'/key).glob('*.json'))
            self.assertEqual(len(files),999)
            for file in files:
                r=json.loads(file.read_text());self.assertEqual(r['status'],'complete')
                self.assertEqual(len(r['folds']),5)
                calculated=metrics(np.array(r['permuted_truth']),np.array(r['predictions']),np.array(r['scores']))
                self.assertAlmostEqual(calculated['BA'],r['BA'],places=12)
            self.assertEqual(len(list((self.out/'checkpoints/bootstrap'/key).glob('*.json'))),200)

    def test_metrics_and_corrected_significance_recomputed(self):
        summary=pd.read_csv(self.out/'validation/classification_summary.csv')
        tests=pd.read_csv(self.out/'validation/permutation_tests.csv');raw=[]
        for row in tests.itertuples():
            preds=pd.read_csv(self.out/'validation/blocked'/f'{row.dataset}_predictions.csv')
            for model,g in preds.groupby('model'):
                self.assertEqual(g.trial_id.nunique(),len(g))
                calculated=metrics(g.truth.to_numpy(),g.prediction.to_numpy(),g.score.to_numpy())
                saved=summary[(summary.dataset==row.dataset)&(summary.model==model)].iloc[0]
                for k in ['BA','AUC','recall_left','recall_right']:self.assertAlmostEqual(calculated[k],saved[k],places=12)
            perm=pd.read_csv(self.out/'validation/permutations'/f'{row.dataset}.csv')
            p=(1+(perm.BA>=row.observed_BA).sum())/1000
            self.assertAlmostEqual(p,row.p_raw,places=12);raw.append(p)
        np.testing.assert_allclose(holm(raw),tests.p_holm.to_numpy(),atol=1e-12)

    def test_all_nested_lineages_and_figure_pairs(self):
        import json
        for file in (self.out/'validation/blocked').glob('*/fold_*.json'):
            r=json.loads(file.read_text());train=set(r['train_ids']);test=set(r['test_ids'])
            self.assertFalse(train&test)
            self.assertTrue(all((t-1)//20!=r['fold'] for t in train))
            self.assertTrue(all((t-1)//20==r['fold'] for t in test))
            for entry in r['fit']['diagnostics']['inner_validation']:
                a=set(entry['train_ids']);b=set(entry['validation_ids'])
                self.assertFalse(a&b);self.assertFalse((a|b)&test);self.assertEqual(a|b,train)
        figures=list((self.out/'figures').glob('*.png'))
        self.assertGreaterEqual(len(figures),16)
        for f in figures:self.assertTrue(f.with_suffix('.pdf').exists())


if __name__=='__main__':unittest.main()
