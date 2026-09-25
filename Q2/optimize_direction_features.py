"""Exploratory Q2 mechanism bridge and nested, label-free-at-inference features.

All decisions involving cue labels are repeated inside each outer training split,
including in every within-block permutation. This is a development-data analysis;
its candidate family was designed after inspecting the archived Q2 results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2model.data import B, NAMES, TIMES, WEIGHTS, independent_data, means, modes, preprocess, save_csv, save_json, window_features
from q2model.model import KernelFactory

ROOT = Path(__file__).resolve().parents[1]
THETA_MS = (80., 50., 160., 400.)  # fixed plausible delays; not estimated from test EEG
CANDIDATES = {
    'kernel_all9': tuple(range(9)),
    'kernel_lateral3': (6, 7, 8),
    'kernel_midline_lateral6': (3, 4, 5, 6, 7, 8),
    'window_all9': tuple(range(9, 18)),
    'window_lateral3': (15, 16, 17),
    'window_middle3': (10, 13, 16),
}


def model_features(x: np.ndarray, duration: float) -> np.ndarray:
    """Three scalp modes x three fixed neural response kernels, no cue input."""
    h = KernelFactory(duration).kernel(THETA_MS, 3)
    inv = np.linalg.inv(h.T @ (WEIGHTS[:, None] * h) + .1 * np.eye(3))
    return np.einsum('nmt,tr,rs->nms', modes(x), WEIGHTS[:, None] * h, inv).reshape(len(x), 9)


def feature_bank(x: np.ndarray, duration: float) -> np.ndarray:
    return np.column_stack((model_features(x, duration), window_features(x)))


def score_lda(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Fixed diagonal shrinkage discriminant with balanced class prior."""
    if set(np.unique(y_train)) != {-1, 1}:
        raise ValueError('Both cue classes are required in each training split')
    mu = x_train.mean(axis=0)
    scale = np.maximum(x_train.std(axis=0), 1e-8)
    z = np.clip((x_train - mu) / scale, -8, 8)
    te = np.clip((x_test - mu) / scale, -8, 8)
    left, right = z[y_train == -1], z[y_train == 1]
    ml, mr = left.mean(0), right.mean(0)
    var = ((left - ml) ** 2).sum(0) + ((right - mr) ** 2).sum(0)
    var /= max(len(z) - 2, 1)
    var = .75 * var + .25  # shrink toward unit variance
    weight = (mr - ml) / np.maximum(var, .25)
    return (te - .5 * (ml + mr)) @ weight


def balanced_logloss(y: np.ndarray, logit: np.ndarray) -> float:
    loss = np.logaddexp(0., -y * logit)
    return float(.5 * (loss[y == -1].mean() + loss[y == 1].mean()))


def select_candidate(bank: np.ndarray, y: np.ndarray, blocks: np.ndarray) -> tuple[str, dict]:
    """Four inner held-out time blocks, using outer-training trials only."""
    results = {}
    for name, columns in CANDIDATES.items():
        values = []
        for block in np.unique(blocks):
            tr, va = blocks != block, blocks == block
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            logit = score_lda(bank[tr][:, columns], y[tr], bank[va][:, columns])
            values.append(balanced_logloss(y[va], logit))
        results[name] = float(np.mean(values)) if values else float('inf')
    winner = min(CANDIDATES, key=lambda k: (results[k], len(CANDIDATES[k]), k))
    return winner, results


def cross_validate(bank: np.ndarray, y: np.ndarray, blocks: np.ndarray,
                   ids: np.ndarray, selected_only: bool = False):
    rows, choices = [], []
    for block in np.unique(blocks):
        tr, te = blocks != block, blocks == block
        chosen, losses = select_candidate(bank[tr], y[tr], blocks[tr])
        choices.append(dict(fold=int(block), chosen=chosen, inner_balanced_logloss=losses,
                            train_ids=ids[tr].tolist(), test_ids=ids[te].tolist()))
        names = ('nested_selected',) if selected_only else (
            'nested_selected', 'kernel_all9', 'kernel_lateral3', 'window_all9')
        for name in names:
            cols = CANDIDATES[chosen if name == 'nested_selected' else name]
            logit = score_lda(bank[tr][:, cols], y[tr], bank[te][:, cols])
            rows.extend(dict(fold=int(block), trial_id=int(i), truth=int(label),
                             model=name, chosen=chosen if name == 'nested_selected' else name,
                             logit=float(v), probability_right=float(expit(v)),
                             prediction=1 if v >= 0 else -1)
                        for i, label, v in zip(ids[te], y[te], logit))
    return rows, choices


def classification_metrics(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    out = []
    for name, part in frame.groupby('model', sort=True):
        y = part.truth.to_numpy(); p = part.prediction.to_numpy()
        left = float((p[y == -1] == -1).mean()); right = float((p[y == 1] == 1).mean())
        out.append(dict(model=name, n=len(y), BA=.5 * (left + right),
                        AUC=float(roc_auc_score(y, part.logit)),
                        recall_left=left, recall_right=right,
                        balanced_logloss=balanced_logloss(y, part.logit.to_numpy())))
    return out


def permute_blocks(y: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = y.copy()
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        out[idx] = rng.permutation(out[idx])
    return out


def simulated_contrast() -> np.ndarray:
    """Process the pre-existing, unfitted neural-mass simulation like the EEG."""
    with np.load(ROOT / 'Q2_result/mechanism/neural_forward.npz') as w:
        time = w['t'].copy(); simulated = w['eeg'][:2].copy()
    padded = np.pad(simulated, ((0, 0), (0, 0), (2560, 2560)))
    filtered = preprocess(padded)[:, :, 2560:-2560]
    aligned = np.stack([np.interp(TIMES, time, filtered[d, c]) for d in range(2) for c in range(3)])
    aligned = aligned.reshape(2, 3, len(TIMES))
    aligned -= np.median(aligned[:, :, :64], axis=-1, keepdims=True)
    return aligned[1] - aligned[0]


def contrast_scores(ds, sim_delta: np.ndarray):
    """One arbitrary-unit gain calibrated on outer training; test cues score only."""
    rows, traces = [], []
    denominator = np.sum(WEIGHTS[None, :] * sim_delta ** 2)
    if denominator <= 1e-16:
        raise ValueError('Simulated direction contrast vanished after preprocessing')
    for block in np.unique(ds.blocks):
        tr = ds.subset(ds.blocks != block); te = ds.subset(ds.blocks == block)
        training = np.diff(means(tr.x, tr.y), axis=0)[0]
        observed = np.diff(means(te.x, te.y), axis=0)[0]
        gain = float(np.sum(WEIGHTS[None, :] * sim_delta * training) / denominator)
        m1_path = ROOT / 'Q2_result/validation/blocked' / ds.key / f'fold_{block}.npz'
        with np.load(m1_path) as archived:
            if not np.array_equal(archived['test_ids'], te.ids):
                raise ValueError(f'{ds.key} fold {block}: archived M1 trial IDs differ')
            m1 = np.diff(archived['prediction_M1'], axis=0)[0]
        for model, predicted in [('zero_direction', np.zeros_like(observed)),
                                 ('illustrative_forward', gain * sim_delta),
                                 ('effective_M1', m1)]:
            err = float(np.sum(WEIGHTS[None, :] * (observed - predicted) ** 2) / 3)
            energy = float(np.sum(WEIGHTS[None, :] * observed ** 2) / 3)
            rows.append(dict(dataset=ds.key, fold=int(block), model=model,
                             contrast_MSE=err, observed_energy=energy, gain=gain,
                             train_cosine=float(np.sum(WEIGHTS[None, :] * training * sim_delta) /
                                                np.sqrt(max(1e-16, denominator *
                                                            np.sum(WEIGHTS[None, :] * training ** 2))))))
        traces.append(dict(fold=int(block), observed=observed, simulated=gain * sim_delta, m1=m1))
    return rows, traces


def audit_inputs() -> dict:
    manifest = json.loads((ROOT / 'Q2_result/manifest.json').read_text())
    verified = {}
    for key in ('A1', 'A2', 'B1', 'B2'):
        name = f'data/VisualCog{key[0]}_Task-{key[1]}.mat'
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if digest != manifest['inputs'][name]:
            raise ValueError(f'Raw input differs from archived experiment: {name}')
        verified[name] = digest
    for name in ('Q2/q2model/data.py', 'Q2/q2model/model.py',
                 'Q2/q2model/mechanism.py'):
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if digest != manifest['source'][name]:
            raise ValueError(f'Core source differs from archived experiment: {name}')
        verified[name] = digest
    return verified


def make_figure(output: Path, datasets: dict, trace_map: dict, classification: list[dict], permutation: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout='constrained')
    for ax, key in zip(axes.flat, datasets):
        tr = trace_map[key]
        # Spatial mode 2 is (F3-F4)/sqrt(2); gray curves expose block instability.
        for row in tr:
            ax.plot(TIMES * 1000, np.einsum('c,ct->t', B[:, 2], row['observed']),
                    color='#AAAAAA', alpha=.45, lw=.9)
        for item, color, label in [('observed', '#222222', 'observed mean'),
                                    ('simulated', '#D98126', 'forward model'),
                                    ('m1', '#2877A8', 'effective model')]:
            curve = np.mean([np.einsum('c,ct->t', B[:, 2], r[item]) for r in tr], axis=0)
            ax.plot(TIMES * 1000, curve, color=color, lw=2, label=label)
        ax.axvline(0, color='#555555', lw=.8); ax.set_title(key)
        ax.set_xlim(0, 750); ax.set_ylabel('F3−F4 mode, original units')
    axes[1, 0].set_xlabel('Time after VisCue (ms)'); axes[1, 1].set_xlabel('Time after VisCue (ms)')
    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Held-out right minus left contrast; gray = individual time blocks')
    fig.savefig(output / '01_forward_heldout.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    table = pd.DataFrame(classification)
    for ax, key in zip(axes.flat, datasets):
        g = table[table.dataset == key]
        names = ['nested_selected', 'kernel_all9', 'kernel_lateral3', 'window_all9']
        values = [float(g[g.model == name].BA.iloc[0]) for name in names]
        ax.bar(range(4), values, color=['#2877A8', '#88AFC3', '#D98126', '#9A9A9A'])
        ax.axhline(.5, color='#333333', ls='--', lw=1)
        ax.set_xticks(range(4), ['nested', 'kernel 9', 'lateral 3', 'window 9'])
        ax.set_ylim(0, 1); ax.set_title(f'{key}  p={permutation[key]["p_raw"]:.3f}')
        ax.set_ylabel('Balanced accuracy')
        for i, value in enumerate(values): ax.text(i, value + .025, f'{value:.2f}', ha='center', fontsize=8)
    fig.suptitle('Held-out cue decoding; p includes inner feature selection and retraining')
    fig.savefig(output / '02_feature_decoding.png', dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'Q2_direction_optimization')
    ap.add_argument('--permutations', type=int, default=999)
    ap.add_argument('--seed', type=int, default=20260925)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Use a new output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    verified = audit_inputs()
    sim_delta = simulated_contrast()
    datasets = {key: independent_data(ROOT, key) for key in NAMES}
    all_predictions, all_choices, all_classification = [], [], []
    all_contrast, all_perm, trace_map = [], [], {}
    permutation_summary = {}
    for key, ds in datasets.items():
        # Fixed temporal shape; duration and EEG from this record only, cue-free features.
        bank = feature_bank(ds.x, .203125)
        rows, choices = cross_validate(bank, ds.y, ds.blocks, ds.ids)
        observed = classification_metrics(rows)
        target_ba = next(v['BA'] for v in observed if v['model'] == 'nested_selected')
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, ord(key[0]), int(key[1])]))
        null = []
        for index in range(args.permutations):
            shuffled = permute_blocks(ds.y, ds.blocks, rng)
            permutation_rows, _ = cross_validate(bank, shuffled, ds.blocks, ds.ids, selected_only=True)
            value = next(v['BA'] for v in classification_metrics(permutation_rows)
                         if v['model'] == 'nested_selected')
            null.append(value)
            all_perm.append(dict(dataset=key, permutation=index + 1, BA=value))
        p = (1 + sum(v >= target_ba for v in null)) / (args.permutations + 1)
        permutation_summary[key] = dict(dataset=key, observed_BA=target_ba,
                                        p_raw=p, permutations=args.permutations,
                                        null_mean=float(np.mean(null)),
                                        null_95=float(np.quantile(null, .95)))
        contrast, trace_map[key] = contrast_scores(ds, sim_delta)
        all_contrast += contrast
        all_predictions += [dict(dataset=key, **r) for r in rows]
        all_choices += [dict(dataset=key, **r) for r in choices]
        all_classification += [dict(dataset=key, **r) for r in observed]
        print(key, f'n={len(ds.x)}', f'nested BA={target_ba:.3f}', f'p={p:.3f}', flush=True)
    ordered = sorted(permutation_summary, key=lambda k: permutation_summary[k]['p_raw'])
    previous = 0.
    for rank, key in enumerate(ordered):
        adjusted = min(1., (len(ordered) - rank) * permutation_summary[key]['p_raw'])
        previous = max(previous, adjusted)
        permutation_summary[key]['p_holm'] = previous
    grouped = pd.DataFrame(all_contrast).groupby(['dataset', 'model'], as_index=False).agg(
        contrast_MSE=('contrast_MSE', 'mean'), observed_energy=('observed_energy', 'mean'))
    grouped['contrast_RMSE'] = np.sqrt(grouped.contrast_MSE)
    grouped['S_delta'] = 1 - grouped.contrast_MSE / grouped.observed_energy
    save_csv(all_predictions, args.output / 'oof_predictions.csv')
    save_json(all_choices, args.output / 'inner_selections.json')
    save_csv(all_classification, args.output / 'classification.csv')
    save_csv(all_contrast, args.output / 'forward_fold_metrics.csv')
    grouped.to_csv(args.output / 'forward_summary.csv', index=False, encoding='utf-8-sig')
    save_csv(all_perm, args.output / 'permutation_null.csv')
    save_csv(list(permutation_summary.values()), args.output / 'permutation_summary.csv')
    save_json(dict(inputs=verified, algorithm_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   theta_ms=THETA_MS, candidates=CANDIDATES, seed=args.seed,
                   permutations=args.permutations, feature_duration='fixed_0.203125_seconds',
                   note='exploratory candidate family informed by archived results'),
              args.output / 'manifest.json')
    make_figure(args.output, datasets, trace_map, all_classification, permutation_summary)


if __name__ == '__main__':
    main()
