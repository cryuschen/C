"""第一问终版：折外来源、计算口径和关键取舍的回归检查。"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import EEG_P300_artifact_correction_v7 as final


class FinalOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / 'eeg_v7_results'
        if not (cls.root / '汇总与说明/运行清单.json').exists():
            raise unittest.SkipTest('请先完整运行 V7 流水线')

    def test_all_saved_trials_reconstruct_v7_from_fold_policy(self):
        decisions = pd.read_csv(self.root / '汇总与说明/逐折V7策略与参考数量.csv')
        for folder in sorted(self.root.glob('受试者*')):
            w = np.load(folder / '可复核波形.npz')
            dataset = 'VisualCog' + folder.name[3] + '_Task-' + ('1' if folder.name.endswith('一') else '2')
            selected = decisions[decisions.dataset == dataset]
            self.assertEqual(len(selected), 5)
            np.testing.assert_allclose(w['v6'], w['before'] + .15 * (w['v4'] - w['before']), atol=1e-10)
            for _, decision in selected.iterrows():
                fold = int(decision.fold)
                mask = w['folds'] == fold
                self.assertTrue(mask.any())
                if decision.V7_policy == '保守比例':
                    np.testing.assert_allclose(w['v7'][mask], w['v6'][mask], atol=1e-10)
                else:
                    self.assertEqual(decision.V4_candidate, '保守双分量')
                    for cue, index in ((-1, 0), (1, 1)):
                        chosen = mask & (w['cues'] == cue)
                        expected = w['before'][chosen] - .25 * (w['before'][chosen] - w['v4'][chosen])
                        expected += .10 * w['fold_mean_deltas'][fold - 1, index]
                        np.testing.assert_allclose(w['v7'][chosen], expected, atol=1e-10)

    def test_metrics_and_trial_lineage(self):
        for folder in sorted(self.root.glob('受试者*')):
            w = np.load(folder / '可复核波形.npz')
            audit = pd.read_csv(folder / '完整事件与试次审计.csv')
            metrics = pd.read_csv(folder / '逐方向逐通道评价指标.csv')
            self.assertEqual(len(audit), 100)
            self.assertEqual((audit.role == '折外评价').sum(), len(w['before']))
            self.assertEqual(set(w['folds']), {1, 2, 3, 4, 5})
            for cue in (-1, 1):
                chosen = w['cues'] == cue
                reference = w['reference_per_trial'][chosen].mean(axis=0)
                for ch, name in enumerate(final.CHANNELS):
                    row = metrics[(metrics.stage == 'V7') & (metrics.cue == cue) &
                                  (metrics.channel == name)].iloc[0]
                    erp = w['v7'][chosen, ch].mean(axis=0)
                    mae = np.abs(erp[final.WINDOW] - reference[ch, final.WINDOW]).mean()
                    self.assertAlmostEqual(row.MAE_after, mae, places=9)
                    self.assertEqual(row.n_trials, int(chosen.sum()))

    def test_benchmark_improvement_and_replay_manifest(self):
        bench = pd.read_csv(self.root / '半合成验证/按数据组和幅度汇总.csv')
        for dataset, group in bench.groupby('dataset'):
            for level in (1., 2.):
                rows = group[group.level == level].set_index('stage')
                self.assertLess(rows.loc['V7', 'normalized_RMSE'],
                                rows.loc['V6', 'normalized_RMSE'], (dataset, level))
        spatial = pd.read_csv(self.root / '汇总与说明/左右刺激与额区空间差异指标.csv')
        ratios = spatial[(spatial.stage == 'V7') & (spatial.quantity == 'left_minus_right_ERP')]
        self.assertTrue((ratios.groupby('dataset').retention_ratio.mean() > .8).all())
        manifest = json.loads((self.root / '汇总与说明/运行清单.json').read_text())
        self.assertEqual(manifest['source']['EEG_P300_artifact_correction_v7.py'],
                         final.sha(Path(final.__file__)))
        for relative, digest in manifest['csv_sha256'].items():
            self.assertEqual(final.sha(self.root / relative), digest)
        self.assertEqual(len(list(self.root.rglob('*.png'))), 36)


if __name__ == '__main__':
    unittest.main()
