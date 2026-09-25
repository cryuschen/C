"""Log-Euclidean covariance features for left/right cue discrimination.

Unlike ERP window means, these six features describe joint Fz/F3/F4
fluctuations. This is an exploratory alternative inspired by Riemannian EEG
classifiers, with strict outer-block preprocessing and a past-only control.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from q2model.data import NAMES, TIMES, independent_data, save_csv, save_json
from evaluate_incremental_postcue import metrics, train_only_scale
from optimize_direction_features import ROOT, audit_inputs, permute_blocks, score_lda

POST = (TIMES >= .05) & (TIMES < .75)
SHRINKAGE = .1
TRIU = np.triu_indices(3)


def log_covariance_features(epochs: np.ndarray) -> np.ndarray:
    """Six coordinates of the symmetric matrix logarithm of each 3x3 covariance."""
    if epochs.ndim != 3 or epochs.shape[1] != 3 or epochs.shape[2] < 4:
        raise ValueError('Expected trials x three electrodes x samples')
    centered = epochs - epochs.mean(axis=-1, keepdims=True)
    covariance = np.einsum('nct,ndt->ncd', centered, centered) / (epochs.shape[-1] - 1)
    trace = np.trace(covariance, axis1=1, axis2=2) / 3
    covariance = (1 - SHRINKAGE) * covariance + SHRINKAGE * trace[:, None, None] * np.eye(3)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        raise ValueError('Covariance is not positive definite')
    logs = np.einsum('nij,nj,nkj->nik', eigenvectors, np.log(eigenvalues), eigenvectors)
    feature = logs[:, TRIU[0], TRIU[1]]
    feature[:, TRIU[0] != TRIU[1]] *= np.sqrt(2)
    return feature


def load_features():
    entries = []
    for group, key in enumerate(NAMES):
        ds = independent_data(ROOT, key)
        post = log_covariance_features(ds.x[:, :, POST])
        pre = log_covariance_features(ds.prestim_only)
        entries.append((key, np.column_stack([pre, post]), ds.y,
                        ds.blocks + 5 * group, ds.ids))
    return (np.vstack([a[1] for a in entries]),
            np.concatenate([a[2] for a in entries]),
            np.concatenate([a[3] for a in entries]),
            np.concatenate([np.repeat(a[0], len(a[2])) for a in entries]),
            np.concatenate([a[4] for a in entries]))


def decode(x, y, blocks, groups, ids):
    rows = []
    for held in np.unique(blocks):
        train, test = blocks != held, blocks == held
        z = train_only_scale(x, groups, blocks, held)
        pre, post = z[:, :6], z[:, 6:]
        bank = {'pre_cov6': pre, 'post_cov6': post,
                'post_minus_pre_cov6': post - pre}
        for name, feature in bank.items():
            score = score_lda(feature[train], y[train], feature[test])
            rows.extend(dict(dataset=str(group), block=int(held), trial_id=int(i),
                             truth=int(label), model=name, logit=float(s),
                             prediction=1 if s >= 0 else -1)
                        for group, i, label, s in zip(groups[test], ids[test], y[test], score))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'Q2_covariance_geometry_v1')
    parser.add_argument('--permutations', type=int, default=1999)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    checked = audit_inputs()
    x, y, blocks, groups, ids = load_features()
    pred = decode(x, y, blocks, groups, ids)
    observed = metrics(pred)
    rng = np.random.default_rng(2026092507)
    null = []
    for iteration in range(args.permutations):
        shuffled = permute_blocks(y, blocks, rng)
        null.extend(dict(iteration=iteration + 1, **row)
                    for row in metrics(decode(x, shuffled, blocks, groups, ids)).to_dict('records'))
    null_frame = pd.DataFrame(null)
    maximum = null_frame.groupby('iteration').BA.max().to_numpy()
    observed['p_raw'] = [float((1 + np.sum(null_frame[null_frame.model == row.model].BA.to_numpy()
                                             >= row.BA)) / (args.permutations + 1))
                         for row in observed.itertuples()]
    observed['p_maxT_three'] = [float((1 + np.sum(maximum >= row.BA)) /
                                      (args.permutations + 1))
                                for row in observed.itertuples()]
    groupwise = []
    for key, part in pred.groupby('dataset'):
        groupwise.extend(dict(dataset=key, **row) for row in metrics(part).to_dict('records'))
    save_csv(pred.to_dict('records'), args.output / 'oof_predictions.csv')
    save_csv(observed.to_dict('records'), args.output / 'pooled_metrics.csv')
    save_csv(groupwise, args.output / 'per_dataset_metrics.csv')
    save_csv(null, args.output / 'permutation_null.csv')
    save_json(dict(inputs=checked, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   permutations=args.permutations, seed=2026092507,
                   covariance_shrinkage=SHRINKAGE,
                   features='log-Euclidean 3x3 covariance: 6 symmetric coordinates',
                   post_window='50-750 ms', pre_window='-250 to 0 ms, past-only filter',
                   limitation='exploratory development data; covariance cannot identify neural source'),
              args.output / 'manifest.json')
    print(observed.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
