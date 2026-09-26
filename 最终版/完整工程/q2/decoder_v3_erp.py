"""Fixed low-frequency ERP matched filter with a strict past-only control.

This is an exploratory analysis of the repeatedly inspected four MAT files.
All features for a held-out block are transformed using its training blocks;
every label-dependent ERP template is refit under each within-block permutation.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.metrics import roc_auc_score

from q2model.data import NAMES, TIMES, independent_data, modes, save_csv, save_json


ROOT = Path(__file__).resolve().parents[1]
POST_MASK = (TIMES >= .05) & (TIMES < .75)
N_DCT = 6
SHRINK = .8  # fixed isotropic shrinkage of within-class covariance
PRE_TO_POST_RIDGE_FACTOR = 1.0  # penalty = number of training trials
MODELS = ('past_only', 'post_erp', 'post_given_past')
MODES = ('common', 'midline', 'lateral')


def dct_features(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal low-frequency coefficients, 3 scalp modes x 6 time modes."""
    if x.ndim != 3 or x.shape[1] != 3:
        raise ValueError('Expected trials x Fz/F3/F4 x time')
    segment = x if mask is None else x[..., mask]
    if segment.shape[-1] < N_DCT:
        raise ValueError('Not enough time samples for DCT')
    return dct(modes(segment), type=2, norm='ortho', axis=-1)[..., :N_DCT].reshape(len(x), -1)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-8)
    return np.clip((train - center) / scale, -8, 8), np.clip((test - center) / scale, -8, 8)


def remove_past_prediction(pre_train: np.ndarray, pre_test: np.ndarray,
                           post_train: np.ndarray, post_test: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Label-blind, train-only ridge regression of post waveform on past EEG."""
    if pre_train.shape[0] != post_train.shape[0] or pre_test.shape[0] != post_test.shape[0]:
        raise ValueError('Paired EEG features required')
    penalty = PRE_TO_POST_RIDGE_FACTOR * len(pre_train)
    transform = np.linalg.solve(pre_train.T @ pre_train + penalty * np.eye(pre_train.shape[1]),
                                pre_train.T @ post_train)
    return post_train - pre_train @ transform, post_test - pre_test @ transform


def erp_score(train: np.ndarray, labels: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Shrinkage-whitened right-minus-left ERP template, balanced prior."""
    if set(np.unique(labels)) != {-1, 1}:
        raise ValueError('Both cue directions required in training')
    left, right = train[labels == -1], train[labels == 1]
    mu_left, mu_right = left.mean(axis=0), right.mean(axis=0)
    residual = np.vstack((left - mu_left, right - mu_right))
    cov = residual.T @ residual / max(len(train) - 2, 1)
    identity_scale = np.trace(cov) / cov.shape[0]
    regularized = (1 - SHRINK) * cov + SHRINK * max(identity_scale, 1e-6) * np.eye(cov.shape[0])
    weight = np.linalg.solve(regularized, mu_right - mu_left)
    return (test - .5 * (mu_left + mu_right)) @ weight


def load_datasets() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for index, key in enumerate(NAMES):
        ds = independent_data(ROOT, key)
        out[key] = dict(pre=dct_features(ds.prestim_only), post=dct_features(ds.x, POST_MASK),
                        y=ds.y, blocks=ds.blocks, ids=ds.ids,
                        global_blocks=ds.blocks + 5 * index)
    return out


def decode_one(data: dict[str, np.ndarray], labels: np.ndarray,
               return_features: bool = False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Five contiguous-block folds; no outer-test labels enter any fitting step."""
    scores = {name: np.full(len(labels), np.nan) for name in MODELS}
    features = {name: np.full_like(data['post'], np.nan) for name in MODELS} if return_features else {}
    for held in np.unique(data['blocks']):
        train, test = data['blocks'] != held, data['blocks'] == held
        pre_train, pre_test = standardize(data['pre'][train], data['pre'][test])
        post_train, post_test = standardize(data['post'][train], data['post'][test])
        residual_train, residual_test = remove_past_prediction(
            pre_train, pre_test, post_train, post_test)
        fold = {'past_only': (pre_train, pre_test), 'post_erp': (post_train, post_test),
                'post_given_past': (residual_train, residual_test)}
        for name, (x_train, x_test) in fold.items():
            scores[name][test] = erp_score(x_train, labels[train], x_test)
            if return_features:
                features[name][test] = x_test
    if any(not np.isfinite(arr).all() for arr in scores.values()):
        raise ValueError('Incomplete outer-fold predictions')
    return scores, features


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = np.where(scores >= 0, 1, -1)
    return dict(n=int(len(y)), BA=float(.5 * ((pred[y == -1] == -1).mean() +
                                              (pred[y == 1] == 1).mean())),
                AUC=float(roc_auc_score(y, scores)),
                recall_left=float((pred[y == -1] == -1).mean()),
                recall_right=float((pred[y == 1] == 1).mean()))


def evaluate(data: dict[str, dict[str, np.ndarray]],
             labels: dict[str, np.ndarray], return_features: bool = False):
    score_map, feature_map = {}, {}
    for key, group in data.items():
        score_map[key], feature_map[key] = decode_one(group, labels[key], return_features)
    rows = []
    for key in ('pooled', *NAMES):
        for name in MODELS:
            yy = np.concatenate([labels[k] for k in NAMES]) if key == 'pooled' else labels[key]
            ss = np.concatenate([score_map[k][name] for k in NAMES]) if key == 'pooled' else score_map[key][name]
            rows.append(dict(dataset=key, model=name, **metrics(yy, ss)))
    return pd.DataFrame(rows), score_map, feature_map


def permute_within_blocks(data: dict[str, dict[str, np.ndarray]],
                          rng: np.random.Generator) -> dict[str, np.ndarray]:
    labels = {}
    for key, group in data.items():
        labels[key] = group['y'].copy()
        for block in np.unique(group['blocks']):
            index = np.flatnonzero(group['blocks'] == block)
            labels[key][index] = rng.permutation(labels[key][index])
    return labels


def block_bootstrap(data, score_map, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in ('pooled', *NAMES):
        for model in MODELS:
            vals = []
            for _ in range(n_boot):
                yy, ss = [], []
                for key in NAMES if dataset == 'pooled' else (dataset,):
                    group = data[key]
                    blocks = np.unique(group['blocks'])
                    chosen = rng.choice(blocks, len(blocks), replace=True)
                    index = np.concatenate([np.flatnonzero(group['blocks'] == b) for b in chosen])
                    yy.append(group['y'][index]); ss.append(score_map[key][model][index])
                y = np.concatenate(yy); score = np.concatenate(ss)
                pred = np.where(score >= 0, 1, -1)
                vals.append(float(.5 * ((pred[y == -1] == -1).mean() +
                                           (pred[y == 1] == 1).mean())))
            low, high = np.quantile(vals, [.025, .975])
            rows.append(dict(dataset=dataset, model=model, BA_low=float(low), BA_high=float(high),
                             bootstraps=n_boot, resampling_unit='contiguous_time_block'))
    return pd.DataFrame(rows)


def make_plot(output: Path, observed: pd.DataFrame, bootstrap: pd.DataFrame,
              null: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'figure.facecolor':'white',
                         'axes.facecolor':'white','axes.edgecolor':'#333333',
                         'axes.linewidth':.9,'axes.spines.top':True,
                         'axes.spines.right':True,'axes.axisbelow':True,
                         'axes.grid':True,'grid.color':'#D9DEE3',
                         'grid.linestyle':'--','grid.linewidth':.7,
                         'grid.alpha':.75,'legend.frameon':True})
    colors = {'past_only': '#8A8A8A', 'post_erp': '#2F75A5', 'post_given_past': '#E08932'}
    joined = observed
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    ax = axes[0]
    groups = list(NAMES) + ['pooled']
    for j, model in enumerate(MODELS):
        part = joined[joined.model == model].set_index('dataset').loc[groups]
        xx = np.arange(len(groups)) + (j - 1) * .23
        yy = part.BA.to_numpy()
        ax.bar(xx, yy, width=.21, color=colors[model],
               label={'past_only': '刺激前', 'post_erp': '刺激后ERP', 'post_given_past': '刺激后增量'}[model])
        ax.errorbar(xx, yy, yerr=np.vstack([yy - part.BA_low.to_numpy(), part.BA_high.to_numpy() - yy]),
                    fmt='none', color='black', capsize=2, lw=.8)
    ax.axhline(.5, ls='--', color='#333333', lw=.9)
    ax.set_xticks(np.arange(len(groups)), [f'{key}\n(n={int(joined[joined.dataset==key].n.iloc[0])})' for key in groups])
    ax.set_ylabel('留出平衡准确率（95%区间）')
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title('18维时空特征的方向判别')
    ax = axes[1]
    vals = null[(null.dataset == 'pooled') & (null.model == 'post_given_past')].BA.to_numpy()
    ax.hist(vals, bins=30, color='#C9D7DF', edgecolor='white')
    line = float(observed[(observed.dataset == 'pooled') &
                          (observed.model == 'post_given_past')].BA.iloc[0])
    ax.axvline(line, color=colors['post_given_past'], lw=2, label=f'实测值={line:.3f}')
    ax.set_xlabel('置换平衡准确率'); ax.set_ylabel('频数')
    ax.set_title('刺激后增量特征的块内置换检验')
    ax.legend(fontsize=9)
    fig.suptitle('左右三角提示方向的留出判别')
    fig.savefig(output / 'decoder_v3_erp.png', dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'q2_result/results/decoder')
    parser.add_argument('--permutations', type=int, default=1999)
    parser.add_argument('--bootstraps', type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Use a new output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_datasets()
    true_labels = {key: group['y'] for key, group in data.items()}
    observed, scores, features = evaluate(data, true_labels, return_features=True)
    predictions, feature_rows = [], []
    feature_names = [f'{mode}_dct{j}' for mode in MODES for j in range(N_DCT)]
    for key, group in data.items():
        for i, label in enumerate(group['y']):
            row = dict(dataset=key, trial_id=int(group['ids'][i]), block=int(group['blocks'][i]),
                       truth=int(label))
            frow = row.copy()
            for name in MODELS:
                value = float(scores[key][name][i])
                row[f'{name}_score'] = value
                row[f'{name}_prediction'] = 1 if value >= 0 else -1
                frow.update({f'{name}_{dim}':float(v) for dim, v in zip(feature_names, features[key][name][i])})
            predictions.append(row); feature_rows.append(frow)
    save_csv(predictions, args.output / 'oof_predictions.csv')
    save_csv(feature_rows, args.output / 'oof_features.csv')
    rng = np.random.default_rng(2026092511)
    null_rows = []
    for iteration in range(args.permutations):
        permuted = permute_within_blocks(data, rng)
        result, _, _ = evaluate(data, permuted)
        null_rows.extend(dict(iteration=iteration + 1, **row) for row in result.to_dict('records'))
        if (iteration + 1) % 200 == 0:
            print(f'permutations {iteration + 1}/{args.permutations}', flush=True)
    null = pd.DataFrame(null_rows)
    null['iteration'] = null.iteration.astype(int)
    pool_max = null[null.dataset == 'pooled'].groupby('iteration').BA.max().to_numpy()
    group_max = null[null.dataset != 'pooled'].groupby('iteration').BA.max().to_numpy()
    observed['p_raw'] = [float((1 + np.sum(null[(null.dataset == row.dataset) &
                                              (null.model == row.model)].BA.to_numpy() >= row.BA)) /
                               (args.permutations + 1)) for row in observed.itertuples()]
    observed['p_family'] = [float((1 + np.sum((pool_max if row.dataset == 'pooled' else group_max)
                                             >= row.BA)) / (args.permutations + 1))
                            for row in observed.itertuples()]
    observed['family'] = ['pooled_3_models' if dataset == 'pooled' else 'four_groups_x_3_models'
                          for dataset in observed.dataset]
    boot = block_bootstrap(data, scores, args.bootstraps, 2026092512)
    observed = observed.merge(boot, on=['dataset', 'model'])
    pre_ba = float(observed[(observed.dataset == 'pooled') & (observed.model == 'past_only')].BA.iloc[0])
    residual_ba = float(observed[(observed.dataset == 'pooled') &
                                 (observed.model == 'post_given_past')].BA.iloc[0])
    save_csv(observed.to_dict('records'), args.output / 'metrics.csv')
    save_csv(null.to_dict('records'), args.output / 'permutation_null.csv')
    make_plot(args.output, observed, boot, null)
    inputs = {str(Path('data') / f'VisualCog{key[0]}_Task-{key[1]}.mat'):
              hashlib.sha256((ROOT / 'data' / f'VisualCog{key[0]}_Task-{key[1]}.mat').read_bytes()).hexdigest()
              for key in NAMES}
    save_json(dict(inputs=inputs, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   data_source='raw Fz/F3/F4 only; q2model.data.independent_data, guard 24 seconds',
                   feature='orthonormal DCT coefficients 0-5 of 3 fixed scalp modes',
                   post_window_ms=[50, 750], past_window_ms=[-250, 0],
                   past_only='independently filtered data ending before VisCue onset',
                   models=MODELS, shrinkage=SHRINK,
                   nuisance_ridge_penalty='n_training_trials times identity',
                   validation='each recording leave 1 of 5 contiguous time blocks out',
                   permutations=args.permutations, permutation_seed=2026092511,
                   bootstrap_unit='contiguous time block within recording', bootstraps=args.bootstraps,
                   pooled_residual_minus_past_BA=residual_ba-pre_ba,
                   limitation='same four MAT previously inspected; internal exploratory evidence only; '
                              'linear past residualization does not exclude nonlinear confounds or eye movements'),
              args.output / 'manifest.json')
    print(observed.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
