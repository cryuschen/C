"""Nine post-cue scalp modes after removing past-only linear predictors.

Exploratory association scan. Ridge residualization uses EEG only, with no cue
labels; permutation is within the 20 independent temporal blocks.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from q2model.data import NAMES, TIMES, independent_data, window_features, save_csv, save_json
from evaluate_pre_control import PRE_WINDOWS
from analyze_pooled_spatial_modes import FEATURE_NAMES, coefficient
from optimize_direction_features import ROOT, audit_inputs, permute_blocks

RIDGE = 10.0


def load_residuals():
    residuals, labels, blocks, groups, ids, r2 = [], [], [], [], [], []
    for group, key in enumerate(NAMES):
        ds = independent_data(ROOT, key)
        post = window_features(ds.x)
        pre = window_features(ds.prestim_only, times=TIMES[:64], windows=PRE_WINDOWS)
        pre = (pre - pre.mean(0)) / np.maximum(pre.std(0), 1e-8)
        post = (post - post.mean(0)) / np.maximum(post.std(0), 1e-8)
        coef = np.linalg.solve(pre.T @ pre + RIDGE * np.eye(9), pre.T @ post)
        fitted = pre @ coef
        residual = post - fitted
        r2.extend(dict(dataset=key, feature=name,
                       r2_from_past=float(1 - np.sum(residual[:, j] ** 2) /
                                          np.sum(post[:, j] ** 2)))
                  for j, name in enumerate(FEATURE_NAMES))
        residual /= np.maximum(residual.std(0), 1e-8)
        residuals.append(residual)
        labels.append(ds.y)
        blocks.append(ds.blocks + 5 * group)
        groups.extend([key] * len(ds.y))
        ids.extend(ds.ids.tolist())
    return (np.vstack(residuals), np.concatenate(labels), np.concatenate(blocks),
            np.asarray(groups), np.asarray(ids), pd.DataFrame(r2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'Q2_incremental_modes_v1')
    parser.add_argument('--permutations', type=int, default=9999)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    checked = audit_inputs()
    x, y, blocks, groups, ids, r2 = load_residuals()
    observed = coefficient(x, y, blocks)
    rng = np.random.default_rng(2026092505)
    null = np.empty((args.permutations, len(FEATURE_NAMES)))
    for i in range(args.permutations):
        shuffled = permute_blocks(y, blocks, rng)
        null[i] = coefficient(x, shuffled, blocks)
    maxima = np.max(np.abs(null), axis=1)
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        rows.append(dict(feature=name, beta=float(observed[j]),
                         p_two_sided=float((1 + (np.abs(null[:, j]) >= abs(observed[j])).sum()) /
                                           (args.permutations + 1)),
                         p_maxT_nine=float((1 + (maxima >= abs(observed[j])).sum()) /
                                           (args.permutations + 1)), n=len(y)))
    by_dataset = []
    for key in NAMES:
        use = groups == key
        effect = coefficient(x[use], y[use], blocks[use])
        by_dataset.extend(dict(dataset=key, feature=name, beta=float(value))
                          for name, value in zip(FEATURE_NAMES, effect))
    save_csv(rows, args.output / 'feature_statistics.csv')
    save_csv(by_dataset, args.output / 'per_dataset_effects.csv')
    save_csv(r2.to_dict('records'), args.output / 'past_explained_variance.csv')
    save_csv([dict(dataset=str(g), trial_id=int(i), block=int(b), truth=int(label),
                   **dict(zip(FEATURE_NAMES, vector)))
              for g, i, b, label, vector in zip(groups, ids, blocks, y, x)],
             args.output / 'residualized_features.csv')
    np.savez_compressed(args.output / 'permutation_null.npz',
                        observed=observed, null=null, maxima=maxima)
    save_json(dict(inputs=checked, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   ridge=RIDGE, n_permutations=args.permutations, seed=2026092505,
                   feature_names=FEATURE_NAMES,
                   caveat='label-blind residualization uses all EEG; this is an association scan, not held-out decoding'),
              args.output / 'manifest.json')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
