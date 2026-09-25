"""Block-stratified association of scalp-mode windows with triangle direction.

The nine candidates were explored after prior inspection, so even max-T values
are internal-development evidence rather than an external confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2model.data import NAMES, TIMES, independent_data, window_features, save_csv, save_json
from optimize_direction_features import ROOT, audit_inputs

FEATURE_NAMES = [f'{mode}_{window}' for mode in ('common', 'midline', 'lateral')
                 for window in ('50_250', '250_500', '500_750')]
PRE_WINDOWS = [(-.25, -.167), (-.167, -.083), (-.083, 0.)]


def load_matrices():
    post, pre, labels, blocks, groups, ids = [], [], [], [], [], []
    for g, key in enumerate(NAMES):
        ds = independent_data(ROOT, key)
        a = window_features(ds.x)
        b = window_features(ds.prestim_only, times=TIMES[:64], windows=PRE_WINDOWS)
        # This scaling uses EEG only and is fixed before any label permutation.
        scale = np.maximum(a.std(axis=0), 1e-8)
        post.append(a / scale)
        pre.append(b / scale)
        labels.append(ds.y)
        blocks.append(ds.blocks + 5 * g)
        groups.extend([key] * len(ds.y))
        ids.extend(ds.ids.tolist())
    return (np.vstack(post), np.vstack(pre), np.concatenate(labels),
            np.concatenate(blocks), np.asarray(groups), np.asarray(ids))


def block_residual(x, blocks):
    output = np.asarray(x, dtype=float).copy()
    for block in np.unique(blocks):
        mask = blocks == block
        output[mask] -= output[mask].mean(axis=0)
    return output


def coefficient(x, labels, blocks):
    a = block_residual(x, blocks)
    b = block_residual(labels, blocks)
    return a.T @ b / (b @ b)


def make_plot(out, effects, stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, feature, title in zip(axes, ('lateral_50_250', 'lateral_250_500'),
                                  ('侧化模态 50–250 ms', '侧化模态 250–500 ms')):
        frame = effects[effects.feature == feature]
        values = [float(frame[frame.dataset == k].beta.iloc[0]) for k in NAMES]
        total = float(stats[(stats.feature == feature) & (stats.period == 'post')].beta.iloc[0])
        ax.bar(range(5), values + [total], color=['#89A9BB'] * 4 + ['#D98126'])
        ax.axhline(0, color='#555555', linewidth=.8)
        ax.set_xticks(range(5), list(NAMES) + ['合并'])
        ax.set_title(title); ax.set_ylabel('标准化特征对方向的回归系数')
    fig.suptitle('额区空间模态的方向关联；合并值为块内调整，非神经来源证明')
    fig.savefig(out / 'pooled_lateral_effects.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'Q2_pooled_mode_analysis')
    parser.add_argument('--permutations', type=int, default=9999)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    checked = audit_inputs()
    post, pre, y, blocks, groups, ids = load_matrices()
    rng = np.random.default_rng(20260925)
    observed = {'post': coefficient(post, y, blocks),
                'pre_past_only': coefficient(pre, y, blocks)}
    null = {period: np.zeros((args.permutations, 9)) for period in observed}
    for iteration in range(args.permutations):
        shuffled = y.copy()
        for block in np.unique(blocks):
            idx = np.flatnonzero(blocks == block)
            shuffled[idx] = rng.permutation(shuffled[idx])
        null['post'][iteration] = coefficient(post, shuffled, blocks)
        null['pre_past_only'][iteration] = coefficient(pre, shuffled, blocks)
    stats, maxima = [], []
    for period, values in observed.items():
        distribution = null[period]
        max_abs = np.max(np.abs(distribution), axis=1)
        maxima.extend(dict(period=period, permutation=i + 1, max_abs_beta=float(v))
                      for i, v in enumerate(max_abs))
        for j, name in enumerate(FEATURE_NAMES):
            p = ((1 + np.sum(np.abs(distribution[:, j]) >= abs(values[j]))) /
                 (args.permutations + 1))
            adjusted = ((1 + np.sum(max_abs >= abs(values[j]))) /
                        (args.permutations + 1))
            stats.append(dict(period=period, feature=name, beta=float(values[j]),
                              p_two_sided=float(p), p_maxT_nine=float(adjusted),
                              n=len(y)))
    effects, jackknife = [], []
    for key in NAMES:
        use = groups == key
        estimates = coefficient(post[use], y[use], blocks[use])
        effects.extend(dict(dataset=key, feature=name, beta=float(value), n=int(use.sum()))
                       for name, value in zip(FEATURE_NAMES, estimates))
        other = ~use
        estimates = coefficient(post[other], y[other], blocks[other])
        jackknife.extend(dict(excluded=key, feature=name, beta=float(value), n=int(other.sum()))
                         for name, value in zip(FEATURE_NAMES, estimates))
    save_csv(stats, args.output / 'feature_statistics.csv')
    save_csv(maxima, args.output / 'permutation_maxima.csv')
    np.savez_compressed(args.output / 'permutation_coefficients.npz',
                        observed_post=observed['post'],
                        observed_pre=observed['pre_past_only'],
                        null_post=null['post'],
                        null_pre=null['pre_past_only'])
    save_csv(effects, args.output / 'per_dataset_effects.csv')
    save_csv(jackknife, args.output / 'leave_one_dataset_out.csv')
    save_csv([dict(dataset=k, trial_id=int(i), block=int(b), truth=int(label),
                   **dict(zip(FEATURE_NAMES, vector)))
              for k, i, b, label, vector in zip(groups, ids, blocks, y, post)],
             args.output / 'standardized_features.csv')
    stats_frame = pd.DataFrame(stats)
    make_plot(args.output, pd.DataFrame(effects), stats_frame)
    save_json(dict(inputs=checked, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   permutations=args.permutations, seed=20260925,
                   code='direction: left=-1, right=+1',
                   standardization='within-dataset feature SD without labels',
                   block_adjustment='20 independent time blocks',
                   multiplicity='max abs coefficient among 9 post or 9 pre features',
                   caveat='candidate family informed by prior development data'),
              args.output / 'manifest.json')
    print(stats_frame[(stats_frame.period == 'post') &
                      (stats_frame.feature.str.startswith('lateral'))].to_string(index=False))


if __name__ == '__main__':
    main()
