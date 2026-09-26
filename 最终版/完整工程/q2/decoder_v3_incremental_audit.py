"""Paired accuracy audit for the fixed V3 decoder's post-cue increment."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from q2model.data import save_json


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(.5 * ((pred[y == -1] == -1).mean() + (pred[y == 1] == 1).mean()))


def audit(folder: Path, repetitions: int = 5000) -> dict:
    pred = pd.read_csv(folder / 'oof_predictions.csv')
    null = pd.read_csv(folder / 'permutation_null.csv')
    if len(pred) != 275 or pred.duplicated(['dataset', 'trial_id']).any():
        raise ValueError('OOF predictions are incomplete or duplicated')
    y = pred.truth.to_numpy()
    pre = pred.past_only_prediction.to_numpy()
    residual = pred.post_given_past_prediction.to_numpy()
    delta = balanced_accuracy(y, residual) - balanced_accuracy(y, pre)
    block_indices = []
    for dataset, group in pred.groupby('dataset', sort=True):
        for block, part in group.groupby('block', sort=True):
            block_indices.append((dataset, block, part.index.to_numpy()))
    by_dataset = {key: [indices for group, _, indices in block_indices if group == key]
                  for key in sorted(pred.dataset.unique())}
    rng = np.random.default_rng(2026092513)
    draws = np.empty(repetitions)
    for iteration in range(repetitions):
        sampled = []
        for blocks in by_dataset.values():
            sampled.extend(blocks[index] for index in rng.integers(len(blocks), size=len(blocks)))
        indices = np.concatenate(sampled)
        draws[iteration] = (balanced_accuracy(y[indices], residual[indices]) -
                            balanced_accuracy(y[indices], pre[indices]))
    pivot = null[null.dataset == 'pooled'].pivot(index='iteration', columns='model', values='BA')
    null_delta = (pivot.post_given_past - pivot.past_only).to_numpy()
    return dict(observed_delta_BA=delta, paired_block_bootstrap_95pct=np.quantile(draws,[.025,.975]),
                one_sided_unconditional_permutation_p=float((1 + np.sum(null_delta >= delta)) /
                                                            (len(null_delta) + 1)),
                bootstraps=repetitions, permutations=len(null_delta),
                note='Permutation difference tests no cue association, not the full conditional '
                     'independence null; post_given_past is linearly residualized on strict past EEG.',
                input_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                              for name in ('oof_predictions.csv','permutation_null.csv')})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--bootstraps', type=int, default=5000)
    args = parser.parse_args()
    result = audit(args.folder, args.bootstraps)
    save_json(result, args.folder / 'incremental_audit.json')
    print(result)


if __name__ == '__main__':
    main()
