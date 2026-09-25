"""Independent metric checks and known-truth controls for direction optimization."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_direction_features import (ROOT, classification_metrics, cross_validate,
                                         feature_bank)
from q2model.data import independent_data, save_csv, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'Q2_direction_optimization')
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / 'manifest.json').read_text())
    digest = hashlib.sha256((ROOT / 'Q2/optimize_direction_features.py').read_bytes()).hexdigest()
    if digest != manifest['algorithm_sha256']:
        raise ValueError('Analysis source changed since the run')
    recorded = pd.read_csv(out / 'classification.csv')
    predictions = pd.read_csv(out / 'oof_predictions.csv')
    null = pd.read_csv(out / 'permutation_null.csv')
    p_table = pd.read_csv(out / 'permutation_summary.csv')
    checked = []
    for key in ('A1', 'A2', 'B1', 'B2'):
        frame = predictions[predictions.dataset == key]
        for record in classification_metrics(frame.to_dict('records')):
            saved = recorded[(recorded.dataset == key) & (recorded.model == record['model'])].iloc[0]
            for field in ('BA', 'AUC', 'balanced_logloss'):
                if not np.isclose(record[field], saved[field], rtol=0, atol=1e-12):
                    raise ValueError(f'{key}: {field} mismatch')
        selected = recorded[(recorded.dataset == key) &
                            (recorded.model == 'nested_selected')].iloc[0]
        values = null[null.dataset == key].BA.to_numpy()
        if len(values) != manifest['permutations']:
            raise ValueError(f'{key}: incomplete permutation null')
        p = (1 + np.sum(values >= selected.BA)) / (len(values) + 1)
        saved_p = p_table[p_table.dataset == key].iloc[0]
        if not np.isclose(p, saved_p.p_raw, atol=1e-12):
            raise ValueError(f'{key}: permutation p mismatch')
        checked.append(dict(dataset=key, BA=float(selected.BA), p_raw=float(p)))
    order = sorted(checked, key=lambda row: row['p_raw'])
    current = 0.
    for i, row in enumerate(order):
        current = max(current, min(1., (4 - i) * row['p_raw']))
        saved = p_table[p_table.dataset == row['dataset']].iloc[0]
        if not np.isclose(current, saved.p_holm, atol=1e-12):
            raise ValueError(f'{row["dataset"]}: Holm p mismatch')

    # Fixed OOF-model block bootstrap is descriptive, and excludes model refits.
    rng = np.random.default_rng(20260925)
    intervals = []
    for key in ('A1', 'A2', 'B1', 'B2'):
        part = predictions[(predictions.dataset == key) &
                           (predictions.model == 'nested_selected')]
        blocks = sorted(part.fold.unique())
        draws = []
        for _ in range(2000):
            sampled = pd.concat([part[part.fold == b] for b in rng.choice(blocks, len(blocks))])
            left = sampled[sampled.truth == -1]
            right = sampled[sampled.truth == 1]
            draws.append(.5 * ((left.prediction == -1).mean() +
                                (right.prediction == 1).mean()))
        lo, hi = np.quantile(draws, [.025, .975])
        intervals.append(dict(dataset=key, BA_low=float(lo), BA_high=float(hi),
                              method='fixed_OOF_time_block_bootstrap', n_boot=2000))
    save_csv(intervals, out / 'block_intervals.csv')

    feature_names = [f'kernel_{mode}_{j}' for mode in ('common', 'midline', 'lateral')
                     for j in (1, 2, 3)] + [
        f'window_{mode}_{window}' for mode in ('common', 'midline', 'lateral')
        for window in ('50_250', '250_500', '500_750')]
    features, effects = [], []
    for key in ('A1', 'A2', 'B1', 'B2'):
        ds = independent_data(ROOT, key)
        values = feature_bank(ds.x, .203125)
        ids_in_predictions = set(predictions[predictions.dataset == key].trial_id)
        if ids_in_predictions != set(ds.ids):
            raise ValueError(f'{key}: exported feature IDs differ from OOF predictions')
        features.extend(dict(dataset=key, trial_id=int(i), block=int(b),
                             **dict(zip(feature_names, row)))
                        for i, b, row in zip(ds.ids, ds.blocks, values))
        for j, name in enumerate(feature_names):
            left, right = values[ds.y == -1, j], values[ds.y == 1, j]
            pooled = np.sqrt((((left - left.mean()) ** 2).sum() +
                              ((right - right.mean()) ** 2).sum()) /
                             (len(left) + len(right) - 2))
            effects.append(dict(dataset=key, feature=name,
                                right_minus_left=float(right.mean() - left.mean()),
                                pooled_sd=float(pooled),
                                standardized_difference=float((right.mean() - left.mean()) /
                                                              max(pooled, 1e-12)),
                                interpretation='descriptive_in_sample_not_neural_specific'))
    save_csv(features, out / 'feature_values.csv')
    save_csv(effects, out / 'feature_effects.csv')

    # Known-truth scenarios use the same features/inner selection. The eye-only
    # scenario has zero neural direction effect by construction.
    synthetic = []
    with np.load(ROOT / 'Q2_result/synthetic/ground_truth.npz') as w:
        y = w['y'].copy()
        blocks = np.repeat(np.arange(5), 20)
        for scenario in ('model_assumed', 'eye_only_no_neural_direction',
                         'background_drift', 'time_parameter_drift'):
            bank = feature_bank(w[scenario], .203125)
            rows, _ = cross_validate(bank, y, blocks, np.arange(1, 101))
            synthetic.extend(dict(scenario=scenario, **record)
                             for record in classification_metrics(rows))
    save_csv(synthetic, out / 'synthetic_controls.csv')
    save_json(dict(status='complete', source_sha256=digest, inputs_verified=True,
                   metrics_recomputed=True, permutation_p_recomputed=True,
                   holm_recomputed=True, synthetic_truth_source='Q2_result/synthetic/ground_truth.npz',
                   visual_review='pending_manual_inspection'), out / 'verification.json')
    print(pd.DataFrame(checked).to_string(index=False))
    print(pd.DataFrame(synthetic)[lambda d: d.model == 'nested_selected'][
        ['scenario', 'BA', 'AUC']].to_string(index=False))


if __name__ == '__main__':
    main()
