"""Does post-cue lateral EEG add direction information beyond strict past EEG?

Exploratory analysis on previously inspected recordings. All scaling,
residualization and classifier fitting are confined to outer training blocks.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from q2model.data import NAMES, TIMES, independent_data, window_features, save_csv, save_json
from evaluate_pre_control import PRE_WINDOWS
from optimize_direction_features import ROOT, audit_inputs, permute_blocks, score_lda

RIDGE = 10.0


def load_paired():
    values = []
    for index, key in enumerate(NAMES):
        ds = independent_data(ROOT, key)
        post = window_features(ds.x)[:, 6]
        pre = window_features(ds.prestim_only, times=TIMES[:64],
                              windows=PRE_WINDOWS)[:, 6:9]
        values.append((key, np.column_stack([pre, post]), ds.y,
                       ds.blocks + 5 * index, ds.ids))
    return (np.vstack([item[1] for item in values]),
            np.concatenate([item[2] for item in values]),
            np.concatenate([item[3] for item in values]),
            np.concatenate([np.repeat(item[0], len(item[2])) for item in values]),
            np.concatenate([item[4] for item in values]))


def train_only_scale(x, groups, blocks, held):
    result = np.empty_like(x)
    for key in NAMES:
        current = groups == key
        train = current & (blocks != held)
        mu = x[train].mean(axis=0)
        sd = np.maximum(x[train].std(axis=0), 1e-8)
        result[current] = np.clip((x[current] - mu) / sd, -8, 8)
    return result


def residualized_post(z, train):
    """Label-blind ridge projection of post on the three pre windows."""
    pre = z[:, :3]
    post = z[:, 3]
    a = np.column_stack([np.ones(train.sum()), pre[train]])
    penalty = RIDGE * np.diag([0.0, 1.0, 1.0, 1.0])
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ post[train])
    return post - np.column_stack([np.ones(len(z)), pre]) @ coef


def predict(x, y, blocks, groups, ids):
    records = []
    for held in np.unique(blocks):
        train, test = blocks != held, blocks == held
        z = train_only_scale(x, groups, blocks, held)
        residual = residualized_post(z, train)
        bank = {'pre_lateral_3': z[:, :3],
                'post_lateral_early': z[:, 3:4],
                'post_given_pre': residual[:, None]}
        for name, values in bank.items():
            score = score_lda(values[train], y[train], values[test])
            records.extend(dict(dataset=str(group), block=int(held), trial_id=int(i),
                                truth=int(label), model=name, logit=float(s),
                                prediction=1 if s >= 0 else -1)
                           for group, i, label, s in zip(groups[test], ids[test], y[test], score))
    return pd.DataFrame(records)


def metrics(rows):
    result = []
    for name, part in rows.groupby('model'):
        y = part.truth.to_numpy()
        pred = part.prediction.to_numpy()
        logits = part.logit.to_numpy()
        result.append(dict(model=name, n=len(part),
                           BA=float(.5 * ((pred[y == -1] == -1).mean() +
                                           (pred[y == 1] == 1).mean())),
                           AUC=float(roc_auc_score(y, logits)),
                           balanced_logloss=float(.5 * (
                               np.logaddexp(0, logits[y == -1]).mean() +
                               np.logaddexp(0, -logits[y == 1]).mean()))))
    return pd.DataFrame(result).sort_values('model').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'Q2_incremental_postcue_v1')
    parser.add_argument('--permutations', type=int, default=1999)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    checked = audit_inputs()
    x, y, blocks, groups, ids = load_paired()
    rows = predict(x, y, blocks, groups, ids)
    observed = metrics(rows)
    rng = np.random.default_rng(2026092504)
    null = []
    for iteration in range(args.permutations):
        shuffled = permute_blocks(y, blocks, rng)
        null.extend(dict(iteration=iteration + 1, **row)
                    for row in metrics(predict(x, shuffled, blocks, groups, ids)).to_dict('records'))
    null_frame = pd.DataFrame(null)
    maximum = null_frame.groupby('iteration').BA.max().to_numpy()
    observed['p_raw'] = [float((1 + np.sum(null_frame[null_frame.model == row.model].BA.to_numpy()
                                             >= row.BA)) / (args.permutations + 1))
                         for row in observed.itertuples()]
    observed['p_maxT_three'] = [float((1 + np.sum(maximum >= row.BA)) /
                                      (args.permutations + 1))
                                for row in observed.itertuples()]
    groupwise = []
    for key, part in rows.groupby('dataset'):
        groupwise.extend(dict(dataset=key, **row) for row in metrics(part).to_dict('records'))
    save_csv(rows.to_dict('records'), args.output / 'oof_predictions.csv')
    save_csv(observed.to_dict('records'), args.output / 'pooled_metrics.csv')
    save_csv(groupwise, args.output / 'per_dataset_metrics.csv')
    save_csv(null, args.output / 'permutation_null.csv')
    save_json(dict(inputs=checked, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   permutations=args.permutations, seed=2026092504, ridge=RIDGE,
                   statement='post_given_pre removes linear information predictable from three past-only lateral windows',
                   limitation='already inspected development data, not independent confirmation'),
              args.output / 'manifest.json')
    print(observed.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
