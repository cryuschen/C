"""验证数值含义与数据隔离；使用 unittest，不依赖额外测试框架。"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import EEG_P300_artifact_correction_v4 as v

class SignalTests(unittest.TestCase):
    def test_no_positive_peak_is_missing_not_zero(self):
        values=v.v3.positive_p300_features(-np.ones(len(v.T)))
        self.assertEqual(values[:2],(0.,0.))
        self.assertTrue(np.isnan(values[2]) and np.isnan(values[3]))

    def test_zero_injection_and_safety_fallback(self):
        rng=np.random.default_rng(4)
        x=v.v3.robust_baseline_correct(rng.normal(size=(12,3,len(v.T))))
        cues=np.tile([-1,1],6)
        np.testing.assert_allclose(v.inject(x,cues,82,0),x,atol=1e-14)
        result=v.correct(x[0],x.mean(0),np.ones(len(v.T)),np.ones(len(v.T)),'不校正')
        np.testing.assert_array_equal(result,x[0])
        for candidate in v.CANDIDATES:
            result=v.correct(x[0],x.mean(0),np.ones(len(v.T)),np.ones(len(v.T)),candidate)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_allclose(np.median(result[:,:v.N],axis=1),0,atol=1e-12)

    def test_boundary_rejection_keeps_ids_and_cues_aligned(self):
        d=np.zeros((10,1500))
        d[7,[10,600,1490]]=[1,-1,1]
        labels=v.CHANNELS+['a','b','c','ECG','VisCue','action','time']
        with patch.object(v.v3,'load_mat_file',return_value=(256,labels,d)):
            raw,x,cues,scores,audit,sensitivity=v.read_dataset('mock.mat')
        self.assertEqual(cues.tolist(),[-1])
        self.assertEqual(audit.loc[audit.role=='折外评价','trial_id'].tolist(),[2])
        self.assertEqual((audit.role=='边界剔除').sum(),2)
        self.assertEqual(raw.shape,x.shape)

    def test_three_stage_summary_has_identical_denominator(self):
        frames={k:pd.DataFrame({'amp_error_after':a}) for k,a in {
            '预处理':[1.,np.nan,3.],'V3':[2.,4.,np.nan],'V4':[3.,5.,6.]}.items()}
        rows=v.paired_summary(frames,'test')
        self.assertEqual([r['paired_rows'] for r in rows],[1,1,1])
        self.assertEqual([r['mean'] for r in rows],[1.,2.,3.])

    def test_semisynthetic_latency_uses_common_valid_rows(self):
        x=np.ones((4,3,len(v.T)))
        y=x.copy();y[:,0]=-1
        result=v.paired_latencies(x,{'预处理':x,'V3':y,'V4':x},np.array([-1,-1,1,1]))
        self.assertEqual([r['latency_pairs'] for r in result.values()],[4,4,4])

class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]/'eeg_v4_results'
        if not (cls.root/'汇总与说明/运行清单.json').exists():
            raise unittest.SkipTest('请先运行完整V4流水线')

    def test_all_trials_accounted_and_reference_disjoint(self):
        expected=[90,95,94,92]
        for folder,n in zip(sorted(self.root.glob('受试者*')),expected):
            audit=pd.read_csv(folder/'完整事件与试次审计.csv')
            self.assertEqual(len(audit),100)
            self.assertEqual((audit.role=='折外评价').sum(),n)
            waves=np.load(folder/'可复核波形.npz')
            self.assertEqual(len(waves['before']),n)
            roles=pd.read_csv(folder/'逐折训练参考评价清单.csv')
            evaluated=[]
            for fold,g in roles.groupby('fold'):
                train=set(g[g.role!='折外评价'].trial_id)
                test=set(g[g.role=='折外评价'].trial_id)
                self.assertFalse(train&test)
                self.assertEqual(len(train|test),n)
                evaluated.extend(test)
            self.assertEqual(sorted(evaluated),sorted(waves['trial_ids']))
            self.assertTrue(np.isfinite(waves['v4']).all())

    def test_metrics_recomputed_from_saved_waves(self):
        for folder in self.root.glob('受试者*'):
            w=np.load(folder/'可复核波形.npz')
            df=pd.read_csv(folder/'逐方向逐通道评价指标.csv')
            for c in (-1,1):
                ix=w['cues']==c
                ref=w['reference_per_trial'][ix].mean(0)
                for ch,name in enumerate(v.CHANNELS):
                    for stage,key in [('预处理','before'),('V3','v3'),('V4','v4')]:
                        avg=w[key][ix,ch].mean(0)
                        expected=np.abs(avg[v.P]-ref[ch,v.P]).mean()
                        row=df[(df.cue==c)&(df.channel==name)&(df.stage==stage)].iloc[0]
                        self.assertAlmostEqual(row.MAE_after,expected,places=9)
                        self.assertEqual(row.n_trials,int(ix.sum()))

    def test_simulation_scenarios_paired(self):
        df=pd.read_csv(self.root/'半合成验证/多场景逐折配对验证.csv')
        self.assertEqual(len(df),4*5*3*4*3)
        for _,g in df.groupby(['dataset','fold','seed','level']):
            self.assertEqual(set(g.stage),{'预处理','V3','V4'})
            self.assertEqual(g.latency_pairs.nunique(),1)
        zero=df[(df.stage=='预处理')&(df.level==0)]
        np.testing.assert_allclose(zero.RMSE,0,atol=1e-12)

    def test_original_baseline_is_unchanged(self):
        frozen=self.root/'基线快照'
        self.assertEqual(v.sha(frozen/'EEG_P300_artifact_correction_v3.py'),v.sha(v.ROOT/'EEG_P300_artifact_correction_v3.py'))
        for p in (frozen/'现有V3结果').rglob('*'):
            if p.is_file():
                self.assertEqual(v.sha(p),v.sha(v.ROOT/'eeg_v3_results'/p.relative_to(frozen/'现有V3结果')))

if __name__=='__main__':unittest.main()
