"""V6 输出的关键可复核约束；运行完整流水线后执行。"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import EEG_P300_artifact_correction_v6 as v6


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / 'eeg_v6_results'
        if not (cls.root / '汇总与说明/运行清单.json').exists():
            raise unittest.SkipTest('请先运行完整 V6 流水线')

    def test_saved_waves_are_conservative_fold_outputs(self):
        for folder in sorted(self.root.glob('受试者*')):
            w = np.load(folder / '可复核波形.npz')
            np.testing.assert_allclose(w['v6'], w['before'] + .15 * (w['v4'] - w['before']), atol=1e-10)
            self.assertEqual(len(w['folds']), len(w['cues']))
            self.assertEqual(set(w['folds']), {1, 2, 3, 4, 5})
            self.assertTrue(np.isfinite(w['v6']).all())

    def test_reported_mae_recomputes_from_saved_waves(self):
        for folder in sorted(self.root.glob('受试者*')):
            w = np.load(folder / '可复核波形.npz')
            df = pd.read_csv(folder / '逐方向逐通道评价指标.csv')
            for cue in (-1, 1):
                selected = w['cues'] == cue
                reference = w['reference_per_trial'][selected].mean(axis=0)
                for ch, name in enumerate(v6.CHANNELS):
                    row = df[(df.stage == 'V6') & (df.cue == cue) & (df.channel == name)].iloc[0]
                    mean_erp = w['v6'][selected, ch].mean(axis=0)
                    mae = np.abs(mean_erp[v6.WINDOW] - reference[ch, v6.WINDOW]).mean()
                    self.assertAlmostEqual(row.MAE_after, mae, places=9)

    def test_synthetic_scenarios_and_manifest_are_complete(self):
        result = pd.read_csv(self.root / '半合成验证/多场景逐折配对验证.csv')
        self.assertEqual(len(result), 4 * 5 * 3 * 4 * 4)
        for _, group in result.groupby(['dataset', 'fold', 'seed', 'level']):
            self.assertEqual(set(group.stage), set(v6.STAGES))
            self.assertEqual(group.n_trials.nunique(), 1)
        zero = result[(result.stage == '预处理') & (result.level == 0)]
        np.testing.assert_allclose(zero.RMSE, 0, atol=1e-12)
        manifest = json.loads((self.root / '汇总与说明/运行清单.json').read_text())
        for relative, digest in manifest['csv_sha256'].items():
            self.assertEqual(v6.sha(self.root / relative), digest)


if __name__ == '__main__':
    unittest.main()
