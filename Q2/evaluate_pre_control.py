"""Past-only negative control for the pooled direction decoder.

Every feature sample is filtered from signal strictly before cue onset.
The first/second/third windows cover -250:-167, -167:-83, -83:0 ms.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from q2model.data import NAMES, TIMES, independent_data, window_features, save_csv, save_json
from evaluate_pooled_discrimination import FEATURES, metrics, predict
from optimize_direction_features import ROOT, audit_inputs, permute_blocks

PRE_WINDOWS = [(-.25, -.167), (-.167, -.083), (-.083, 0.)]


def load_pre():
    entries = []
    for group, key in enumerate(NAMES):
        ds = independent_data(ROOT, key)
        feature = window_features(ds.prestim_only, times=TIMES[:64], windows=PRE_WINDOWS)
        entries.append((key, feature, ds.y, ds.blocks + 5 * group, ds.ids))
    return (np.vstack([entry[1] for entry in entries]),
            np.concatenate([entry[2] for entry in entries]),
            np.concatenate([entry[3] for entry in entries]),
            np.concatenate([np.repeat(entry[0], len(entry[2])) for entry in entries]),
            np.concatenate([entry[4] for entry in entries]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'Q2_pooled_pre_control_v2')
    parser.add_argument('--permutations', type=int, default=1999)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    checked = audit_inputs()
    x, y, blocks, groups, ids = load_pre()
    observed = metrics(predict(x, y, blocks, groups, ids))
    rng = np.random.default_rng(2026092503)
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
    save_csv(observed.to_dict('records'), args.output / 'pooled_metrics.csv')
    save_csv(null, args.output / 'permutation_null.csv')
    save_json(dict(inputs=checked, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   n_permutations=args.permutations, seed=2026092503,
                   windows=PRE_WINDOWS, feature_family=list(FEATURES),
                   filter='independent_data prestim_only ends at cue onset'),
              args.output / 'manifest.json')
    print(observed.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
