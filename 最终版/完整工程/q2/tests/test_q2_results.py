"""Independent audits of a saved complete run (skip when outputs are absent)."""
import json
import hashlib
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'q2_result'


@unittest.skipUnless((OUT/'Q2_运行清单.json').exists(),'Run Q2/main.py first')
class SavedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((OUT/'Q2_运行清单.json').read_text(encoding='utf8'))
        cls.null=pd.read_csv(OUT/'Q2_置换分布.csv')
        cls.metrics=pd.read_csv(OUT/'Q2_表4_特征判别.csv')

    def test_raw_and_q1_inputs_unchanged(self):
        for path,expected in self.manifest['input_hashes'].items():
            p=ROOT/Path(path.replace('\\','/'))
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),expected,path)

    def test_all_fold_training_and_test_ids_disjoint(self):
        for filename in ('Q2_机制内层选择.json','Q2_判别内层选择.json'):
            records=json.loads((OUT/filename).read_text(encoding='utf8'))
            for r in records:
                self.assertFalse(set(r['train_ids']) & set(r['test_ids']))
                self.assertTrue(all((i-1)//20==r['held_block'] for i in r['test_ids']))
                self.assertTrue(all((i-1)//20!=r['held_block'] for i in r['train_ids']))

    def test_null_counts_and_p_values(self):
        primary=self.manifest.get('primary_features',['amplitude','peak_latency','spatial','mechanism','covariance'])
        for kind,total in [('block_permutation',self.manifest['permutations']),
                           ('circular_shift',self.manifest['circular_shifts'])]:
            null=self.null[self.null.kind==kind]
            self.assertEqual(len(null),total*len(self.metrics))
            self.assertFalse(null.duplicated(['iteration','dataset','features']).any())
            for r in self.metrics.itertuples():
                n=null[(null.dataset==r.dataset)&(null.features==r.features)]
                expected=(1+sum(n.BA>=r.BA))/(total+1)
                self.assertAlmostEqual(getattr(r,kind+'_p'),expected,places=10)
                if r.features in primary:
                    mx=null[(null.dataset==r.dataset)&null.features.isin(primary)].groupby('iteration').BA.max()
                    expected=(1+sum(mx>=r.BA))/(total+1)
                    self.assertAlmostEqual(getattr(r,kind+'_maxT_p'),expected,places=10)

    def test_saved_no_shape_and_symmetric_predictions(self):
        with np.load(OUT/'Q2_机制波形.npz') as w:
            for key in ('A1','A2','B1','B2'):
                h=w[key+'_no_shape_predicted']
                np.testing.assert_allclose(h[:,0],h[:,1],atol=1e-10)
                h=w[key+'_symmetric_readout_predicted']
                np.testing.assert_allclose(h[:,:,1],h[:,:,2],atol=1e-10)


if __name__=='__main__':unittest.main()
