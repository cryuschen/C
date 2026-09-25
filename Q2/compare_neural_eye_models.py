"""Test whether a fitted neural-mass response shape transfers better than an ocular step.

Both templates have exactly one temporal degree of freedom and three scalp gains.
This is a model comparison, not identification of the physiological source.
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
from q2model.data import (NAMES, TIMES, WEIGHTS, independent_data, means,
                          preprocess, save_csv, save_json)
from optimize_direction_features import ROOT, audit_inputs, permute_blocks, score_lda

LAMBDAS = (.01, .1, 1., 10.)


def filtered_template(t: np.ndarray, signal: np.ndarray) -> np.ndarray:
    full = np.pad(signal[None, :], ((0, 0), (2560, 2560)))
    filtered = preprocess(full)[0, 2560:-2560]
    out = np.interp(TIMES, t, filtered)
    out -= np.median(out[:64])
    norm = np.sqrt(np.sum(WEIGHTS * out * out))
    if norm < 1e-10:
        raise ValueError('Template vanished after filtering')
    return out / norm


def templates() -> dict[str, np.ndarray]:
    with np.load(ROOT / 'Q2_result/mechanism/neural_forward.npz') as w:
        t = w['t'].copy()
        q = w['dipoles'].copy()
    neural = filtered_template(t, q[1, 0] - q[0, 0])
    # Symmetric left/right shape drive gives rank-one direction dipole activity.
    ocular = filtered_template(t, expit((t - .18) / .02))
    return {'neural_mass': neural, 'ocular_step': ocular}


def fit_gain(delta: np.ndarray, h: np.ndarray, lam: float) -> np.ndarray:
    gain = (delta * WEIGHTS[None, :]) @ h / (1. + lam)
    return gain[:, None] * h[None, :]


def mse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum((obs - pred) ** 2 * WEIGHTS[None, :]) / 3)


def select_ridge(ds, h: np.ndarray) -> tuple[float, list]:
    choices = []
    for lam in LAMBDAS:
        errors = []
        for block in np.unique(ds.blocks):
            tr = ds.subset(ds.blocks != block)
            te = ds.subset(ds.blocks == block)
            target = means(te.x, te.y)[1] - means(te.x, te.y)[0]
            train_delta = means(tr.x, tr.y)[1] - means(tr.x, tr.y)[0]
            errors.append(mse(target, fit_gain(train_delta, h, lam)))
        choices.append(dict(lambda_gain=lam, inner_MSE=float(np.mean(errors))))
    best = min(choices, key=lambda row: (row['inner_MSE'], -row['lambda_gain']))
    return float(best['lambda_gain']), choices


def decode(ds, h: np.ndarray, y_override=None):
    """Three-channel matched-filter coefficients; test cue used only for scoring."""
    features = np.einsum('nct,t,t->nc', ds.x, WEIGHTS, h)
    labels = ds.y if y_override is None else np.asarray(y_override)
    rows = []
    for block in np.unique(ds.blocks):
        tr, te = ds.blocks != block, ds.blocks == block
        logit = score_lda(features[tr], labels[tr], features[te])
        rows.extend(dict(fold=int(block), trial_id=int(i), truth=int(y),
                         logit=float(s), prediction=1 if s >= 0 else -1)
                    for i, y, s in zip(ds.ids[te], labels[te], logit))
    return rows


def score_metrics(rows):
    frame = pd.DataFrame(rows)
    left = frame[frame.truth == -1]; right = frame[frame.truth == 1]
    return dict(BA=float(.5 * ((left.prediction == -1).mean() +
                               (right.prediction == 1).mean())),
                AUC=float(roc_auc_score(frame.truth, frame.logit)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'Q2_mechanism_bridge_v2')
    parser.add_argument('--permutations', type=int, default=999)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    checked_inputs = audit_inputs()
    shapes = templates()
    contrast, choices, predictions, summaries, permutation_rows = [], [], [], [], []
    for key in NAMES:
        ds = independent_data(ROOT, key)
        outer_scores = {}
        for name, h in shapes.items():
            for block in np.unique(ds.blocks):
                tr = ds.subset(ds.blocks != block)
                te = ds.subset(ds.blocks == block)
                lam, inner = select_ridge(tr, h)
                train_delta = means(tr.x, tr.y)[1] - means(tr.x, tr.y)[0]
                test_delta = means(te.x, te.y)[1] - means(te.x, te.y)[0]
                pred = fit_gain(train_delta, h, lam)
                contrast.append(dict(dataset=key, fold=int(block), model=name,
                                     contrast_MSE=mse(test_delta, pred),
                                     zero_MSE=mse(test_delta, np.zeros_like(test_delta)),
                                     lambda_gain=lam))
                choices.append(dict(dataset=key, fold=int(block), model=name,
                                    train_ids=tr.ids.tolist(), test_ids=te.ids.tolist(),
                                    selected_lambda=lam, inner=inner))
            rows = decode(ds, h)
            observed = score_metrics(rows)
            rng = np.random.default_rng(np.random.SeedSequence([20260925, ord(key[0]),
                                                                 int(key[1]), 1 if name == 'neural_mass' else 2]))
            null = []
            for index in range(args.permutations):
                permuted = permute_blocks(ds.y, ds.blocks, rng)
                value = score_metrics(decode(ds, h, permuted))['BA']
                null.append(value)
                permutation_rows.append(dict(dataset=key, model=name,
                                             iteration=index + 1, BA=value))
            p = (1 + sum(x >= observed['BA'] for x in null)) / (args.permutations + 1)
            outer_scores[name] = dict(dataset=key, model=name, n=len(ds.y),
                                      **observed, p_raw=p)
            predictions.extend(dict(dataset=key, model=name, **row) for row in rows)
        summaries.extend(outer_scores.values())
        print(key, outer_scores, flush=True)
    save_csv(contrast, args.output / 'contrast_fold_metrics.csv')
    c = pd.DataFrame(contrast).groupby(['dataset', 'model'], as_index=False).agg(
        contrast_MSE=('contrast_MSE', 'mean'), zero_MSE=('zero_MSE', 'mean'))
    c['contrast_RMSE'] = np.sqrt(c.contrast_MSE)
    c['S_delta'] = 1 - c.contrast_MSE / c.zero_MSE
    c.to_csv(args.output / 'contrast_summary.csv', index=False, encoding='utf-8-sig')
    save_csv(summaries, args.output / 'decoding_summary.csv')
    save_csv(predictions, args.output / 'oof_predictions.csv')
    save_csv(permutation_rows, args.output / 'permutation_null.csv')
    save_json(choices, args.output / 'inner_choices.json')
    np.savez_compressed(args.output / 'templates.npz', times=TIMES, **shapes)
    save_json(dict(inputs=checked_inputs, algorithm_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_simulation='Q2_result/mechanism/neural_forward.npz',
                   lambda_candidates=LAMBDAS, permutations=args.permutations,
                   eye_template='sigmoid step at 180 ms, 20 ms transition',
                   note='exploratory model comparison on previously inspected development data'),
              args.output / 'manifest.json')


if __name__ == '__main__':
    main()
