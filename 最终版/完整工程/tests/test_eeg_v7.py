"""第一问终版：折外来源、计算口径和关键取舍的回归检查。"""
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import EEG_P300_artifact_correction_v7 as final


class FinalOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / 'eeg_v7_results'
        if not (cls.root / '汇总与说明/运行清单.json').exists():
            raise unittest.SkipTest('请先完整运行 V7 流水线')

    def test_v7_imports_without_legacy_modules(self):
        with TemporaryDirectory() as directory:
            script = Path(directory) / 'EEG_P300_artifact_correction_v7.py'
            shutil.copyfile(final.__file__, script)
            process = subprocess.run([sys.executable, '-I', str(script), '--help'],
                                     cwd=directory, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_all_saved_trials_reconstruct_v7_from_fold_policy(self):
        decisions = pd.read_csv(self.root / '汇总与说明/逐折V7策略与参考数量.csv')
        for folder in sorted(self.root.glob('受试者*')):
            w = np.load(folder / '可复核波形.npz')
            dataset = 'VisualCog' + folder.name[3] + '_Task-' + ('1' if folder.name.endswith('一') else '2')
            selected = decisions[decisions.dataset == dataset]
            self.assertEqual(len(selected), 5)
            self.assertEqual(set(w.files), {'raw', 'before', 'candidate_output', 'v7', 'cues',
                                            'trial_ids', 'folds', 'fold_mean_deltas',
                                            'reference_per_trial', 'times_ms'})
            for _, decision in selected.iterrows():
                fold = int(decision.fold)
                mask = w['folds'] == fold
                self.assertTrue(mask.any())
                if decision.V7_policy == '保守比例':
                    expected = w['before'][mask] + .15 * (w['candidate_output'][mask] - w['before'][mask])
                    np.testing.assert_allclose(w['v7'][mask], expected, atol=1e-10)
                else:
                    self.assertEqual(decision.selected_candidate, '保守双分量')
                    for cue, index in ((-1, 0), (1, 1)):
                        chosen = mask & (w['cues'] == cue)
                        expected = w['before'][chosen] - .25 * (w['before'][chosen] - w['candidate_output'][chosen])
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

    def test_benchmark_baseline_and_replay_manifest(self):
        bench = pd.read_csv(self.root / '半合成验证/按数据组和幅度汇总.csv')
        self.assertEqual(set(bench.stage), {'未校正', 'V7'})
        for dataset, group in bench.groupby('dataset'):
            self.assertEqual(set(group.level), {0., .5, 1., 2.})
            zero = group[group.level == 0].set_index('stage')
            self.assertAlmostEqual(zero.loc['未校正', 'normalized_RMSE'], 0., places=12)
            self.assertGreater(zero.loc['V7', 'normalized_RMSE'], 0.)
        spatial = pd.read_csv(self.root / '汇总与说明/左右刺激与额区空间差异指标.csv')
        ratios = spatial[(spatial.stage == 'V7') & (spatial.quantity == 'left_minus_right_ERP')]
        self.assertTrue((ratios.groupby('dataset').retention_ratio.mean() > .8).all())
        manifest = json.loads((self.root / '汇总与说明/运行清单.json').read_text())
        self.assertEqual(set(manifest['source']), {'EEG_P300_artifact_correction_v7.py'})
        self.assertEqual(manifest['source']['EEG_P300_artifact_correction_v7.py'],
                         final.sha(Path(final.__file__)))
        for relative, digest in manifest['csv_sha256'].items():
            self.assertEqual(final.sha(self.root / relative), digest)
        self.assertEqual(len([p for p in self.root.rglob('*.png') if not p.name.startswith('独立验证_')]), 44)
        summary = pd.read_csv(self.root / '汇总与说明/预处理与V7共同配对指标汇总.csv')
        self.assertEqual(set(summary.stage), {'预处理', 'V7'})
        for folder in sorted(self.root.glob('受试者*')):
            metrics = pd.read_csv(folder / '逐方向逐通道评价指标.csv')
            bootstrap = pd.read_csv(folder / '试次重采样区间.csv')
            self.assertEqual(set(metrics.stage), {'预处理', 'V7'})
            self.assertEqual(set(bootstrap.stage), {'预处理', 'V7', 'V7减预处理'})

    def test_group_intervals_and_temporal_contrast_recompute_from_saved_trials(self):
        intervals = pd.read_csv(self.root / '汇总与说明/四组核心指标重采样区间.csv')
        stability = pd.read_csv(self.root / '汇总与说明/左右差分前后时段稳定性.csv')
        for folder in sorted(self.root.glob('受试者*')):
            dataset = 'VisualCog' + folder.name[3] + '_Task-' + ('1' if folder.name.endswith('一') else '2')
            w = np.load(folder / '可复核波形.npz')
            cues = w['cues']
            reference = {cue: w['reference_per_trial'][cues == cue].mean(axis=0) for cue in (-1, 1)}
            mae = {}
            snr = {}
            erps = {}
            for stage, key in [('预处理', 'before'), ('V7', 'v7')]:
                per_row_mae = []
                per_row_snr = []
                erps[stage] = {}
                for cue in (-1, 1):
                    trials = w[key][cues == cue]
                    erps[stage][cue] = trials.mean(axis=0)
                    per_row_mae.extend(np.abs(erps[stage][cue][:, final.WINDOW] -
                                              reference[cue][:, final.WINDOW]).mean(axis=1))
                    per_row_snr.extend(final.snr_proxy_db(trials[:, ch]) for ch in range(3))
                mae[stage] = np.mean(per_row_mae)
                snr[stage] = np.mean(per_row_snr)
            rows = intervals[(intervals.dataset == dataset) & (intervals.stage == 'V7')].set_index('metric')
            self.assertAlmostEqual(rows.loc['proxy_MAE_reduction_pct', 'point'],
                                   100 * (mae['预处理'] - mae['V7']) / mae['预处理'], places=9)
            self.assertAlmostEqual(rows.loc['SNR_proxy_gain_dB', 'point'],
                                   snr['V7'] - snr['预处理'], places=9)
            self.assertLess(rows.loc['proxy_MAE_reduction_pct', 'low'],
                            rows.loc['proxy_MAE_reduction_pct', 'point'])
            self.assertGreater(rows.loc['proxy_MAE_reduction_pct', 'low'], 0)
            self.assertLess(rows.loc['SNR_proxy_gain_dB', 'low'], 0)
            self.assertGreater(rows.loc['SNR_proxy_gain_dB', 'high'], 0)
            self.assertEqual(set(rows.n_boot), {800})
            order = np.argsort(w['trial_ids'])
            first = np.zeros(len(cues), dtype=bool)
            first[order[:len(cues)//2]] = True
            differences = []
            for mask in (first, ~first):
                left = w['v7'][mask & (cues == -1)].mean(axis=0)
                right = w['v7'][mask & (cues == 1)].mean(axis=0)
                differences.append((left - right)[:, final.WINDOW].ravel())
            row = stability[(stability.dataset == dataset) & (stability.stage == 'V7') &
                            (stability.channel == '三通道合并')].iloc[0]
            self.assertAlmostEqual(row.contrast_correlation,
                                   np.corrcoef(*differences)[0, 1], places=9)
            self.assertEqual(row.first_left_n, int(np.sum(first & (cues == -1))))
            self.assertEqual(row.second_right_n, int(np.sum(~first & (cues == 1))))


if __name__ == '__main__':
    unittest.main()
