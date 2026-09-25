"""Blocked decoding of the full nine post-cue modes after past-only control.

The complete nine-dimensional representation is fixed before this run. This
remains exploratory because the underlying recordings have been inspected.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from q2model.data import NAMES, TIMES, independent_data, window_features, save_csv, save_json
from evaluate_pre_control import PRE_WINDOWS
from evaluate_incremental_postcue import metrics, train_only_scale
from optimize_direction_features import ROOT, audit_inputs, permute_blocks, score_lda

RIDGE = 10.0


def load_paired_nine():
    entries = []
    for group, key in enumerate(NAMES):
        ds = independent_data(ROOT, key)
        pre = window_features(ds.prestim_only, times=TIMES[:64], windows=PRE_WINDOWS)
        post = window_features(ds.x)
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
        tr, te = blocks != held, blocks == held
        z = train_only_scale(x, groups, blocks, held)
        pre, post = z[:, :9], z[:, 9:]
        fit = np.linalg.solve(pre[tr].T @ pre[tr] + RIDGE * np.eye(9),
                              pre[tr].T @ post[tr])
        residual = post - pre @ fit
        for name, feature in (('pre_9', pre), ('post_9', post),
                              ('post_residual_9', residual)):
            logits = score_lda(feature[tr], y[tr], feature[te])
            rows.extend(dict(dataset=str(group), block=int(held), trial_id=int(i),
                             truth=int(label), model=name, logit=float(score),
                             prediction=1 if score >= 0 else -1)
                        for group, i, label, score in zip(groups[te], ids[te], y[te], logits))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'Q2_residual_9d_v1')
    parser.add_argument('--permutations', type=int, default=1999)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    checked = audit_inputs()
    x, y, blocks, groups, ids = load_paired_nine()
    pred = decode(x, y, blocks, groups, ids)
    observed = metrics(pred)
    rng = np.random.default_rng(2026092506)
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
                   permutations=args.permutations, seed=2026092506, ridge=RIDGE,
                   feature='three scalp modes times three post-cue windows, jointly residualized on nine past-only windows',
                   note='all preprocessing and regression fit on outer training blocks'),
              args.output / 'manifest.json')
    print(observed.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
