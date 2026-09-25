"""Exploratory pooled, block-held-out discrimination of scalp spatial modes.

The feature family has been inspected on these data. Permutation p-values
describe internal consistency only, not independent confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2model.data import NAMES, independent_data, window_features, save_csv, save_json
from optimize_direction_features import ROOT, audit_inputs, permute_blocks, score_lda

FEATURES = {
    'early_lateral': (6,),
    'lateral_3windows': (6, 7, 8),
    'all_9windows': tuple(range(9)),
}


def load_features():
    rows = []
    for group, key in enumerate(NAMES):
        ds = independent_data(ROOT, key)
        rows.append((key, window_features(ds.x), ds.y, ds.blocks + 5 * group, ds.ids))
    x = np.vstack([row[1] for row in rows])
    y = np.concatenate([row[2] for row in rows])
    blocks = np.concatenate([row[3] for row in rows])
    groups = np.concatenate([np.repeat(row[0], len(row[2])) for row in rows])
    ids = np.concatenate([row[4] for row in rows])
    return x, y, blocks, groups, ids


def fold_standardized(x, blocks, groups, held):
    """Within-recording nuisance scale uses only the outer training portion."""
    z = np.empty_like(x)
    for group in NAMES:
        group_mask = groups == group
        train = group_mask & (blocks != held)
        mu = x[train].mean(0)
        sd = np.maximum(x[train].std(0), 1e-8)
        z[group_mask] = np.clip((x[group_mask] - mu) / sd, -8, 8)
    return z


def predict(x, y, blocks, groups, ids):
    rows = []
    for held in np.unique(blocks):
        tr, te = blocks != held, blocks == held
        z = fold_standardized(x, blocks, groups, held)
        for name, columns in FEATURES.items():
            logit = score_lda(z[tr][:, columns], y[tr], z[te][:, columns])
            rows.extend(dict(dataset=str(group), block=int(held), trial_id=int(i),
                             truth=int(label), model=name, logit=float(score),
                             prediction=1 if score >= 0 else -1)
                        for group, i, label, score in zip(groups[te], ids[te], y[te], logit))
    return pd.DataFrame(rows)


def metrics(rows):
    scores = []
    for model, group in rows.groupby('model'):
        y = group.truth.to_numpy()
        pred = group.prediction.to_numpy()
        scores.append(dict(model=model, n=len(group),
                           BA=float(.5 * ((pred[y == -1] == -1).mean() +
                                           (pred[y == 1] == 1).mean())),
                           AUC=float(roc_auc_score(y, group.logit))))
    return pd.DataFrame(scores).sort_values('model').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'Q2_pooled_discrimination_v2')
    parser.add_argument('--permutations', type=int, default=999)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    checked = audit_inputs()
    x, y, blocks, groups, ids = load_features()
    result = predict(x, y, blocks, groups, ids)
    observed = metrics(result)
    rng = np.random.default_rng(2026092502)
    null = []
    for iteration in range(args.permutations):
        shuffled = permute_blocks(y, blocks, rng)
        scores = metrics(predict(x, shuffled, blocks, groups, ids))
        null.extend(dict(iteration=iteration + 1, **row)
                    for row in scores.to_dict('records'))
    null_frame = pd.DataFrame(null)
    maximum = null_frame.groupby('iteration').BA.max().to_numpy()
    observed['p_raw'] = [float((1 + np.sum(null_frame[null_frame.model == row.model].BA.to_numpy()
                                             >= row.BA)) / (args.permutations + 1))
                         for row in observed.itertuples()]
    observed['p_maxT_three'] = [float((1 + np.sum(maximum >= row.BA)) /
                                      (args.permutations + 1))
                                for row in observed.itertuples()]
    save_csv(result.to_dict('records'), args.output / 'oof_predictions.csv')
    save_csv(observed.to_dict('records'), args.output / 'pooled_metrics.csv')
    save_csv(null, args.output / 'permutation_null.csv')
    per_dataset = []
    for group, frame in result.groupby('dataset'):
        per_dataset.extend(dict(dataset=group, **row)
                           for row in metrics(frame).to_dict('records'))
    save_csv(per_dataset, args.output / 'per_dataset_metrics.csv')
    save_json(dict(inputs=checked, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   n_permutations=args.permutations, seed=2026092502,
                   feature_family=list(FEATURES),
                   standardization='per recording, outer training EEG only',
                   validation='leave one of 20 contiguous time blocks out; train on the other 19',
                   caveat='all four recordings were previously used for feature development'),
              args.output / 'manifest.json')
    print(observed.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
