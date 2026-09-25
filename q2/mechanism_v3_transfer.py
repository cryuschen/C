"""Train-only transfer test of an image-driven neural-mass direction response.

The image-to-shape and E/I simulation are fixed before reading EEG.  The model
only estimates three scalp gains per fixed temporal basis, with ridge selected
inside each outer training set.  This is a prediction test, not source recovery.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from q2model.data import NAMES, TIMES, WEIGHTS, independent_data, means, save_csv, save_json
from q2model.mechanism import neural_forward
from q2model.stimulus_shape import DOCX, MEDIA, cue_images, shape_inputs
from v3_support import ROOT, audit_inputs, filtered_template, mse

LAMBDAS = (.1, .3, 1., 3., 10., 30., 100.)


def make_bases() -> tuple[dict[str, np.ndarray], dict]:
    cue = cue_images(ROOT)
    encoded = shape_inputs(cue)
    forward = neural_forward(encoded['inputs'])
    t = forward['t']
    # Mirrored shape-selective source populations have opposite cue contrasts.
    # An effective frontal path may have several positive propagation delays.
    source_delta = forward['dipoles'][1, 0] - forward['dipoles'][0, 0]
    neural = np.column_stack([
        filtered_template(t, np.interp(t - delay, t, source_delta, left=0., right=0.))
        for delay in (0., .09, .18)
    ])
    # Eye rotation and task preparation can cause smooth step/ramp potentials.
    # These bases have the same number of temporal components and free gains.
    ocular = np.column_stack([
        filtered_template(t, expit((t - latency) / .025))
        for latency in (.08, .18, .32)
    ])
    ramp = np.column_stack([
        filtered_template(t, np.maximum(t - latency, 0.))
        for latency in (0., .18, .35)
    ])
    base = {"visual_neural_mass": neural,
            "ocular_step": ocular,
            "slow_ramp": ramp}
    for h in base.values():
        if not np.isfinite(h).all() or h.shape != (len(TIMES), 3):
            raise ValueError('Invalid basis')
    meta = dict(shape_activation=encoded['activation'].tolist(),
                shape_selectivity=encoded['selectivity'].tolist(),
                frontal_delays_ms=[0, 90, 180],
                eye_step_latency_ms=[80, 180, 320],
                ramp_latency_ms=[0, 180, 350],
                cue_schematic=True)
    return base, meta


def project(delta: np.ndarray, h: np.ndarray, lam: float) -> np.ndarray:
    """Three channel-wise ridge gains for a three-component temporal basis."""
    gram = h.T @ (WEIGHTS[:, None] * h)
    target = (delta * WEIGHTS[None, :]) @ h
    coef = np.linalg.solve(gram + lam * np.eye(h.shape[1]), target.T).T
    return coef @ h.T


def contrast(ds) -> np.ndarray:
    avg = means(ds.x, ds.y)
    return avg[1] - avg[0]


def select_lambda(train, h: np.ndarray) -> tuple[float, list[dict]]:
    options = []
    for lam in LAMBDAS:
        scores = []
        for held in np.unique(train.blocks):
            inner_train = train.subset(train.blocks != held)
            inner_test = train.subset(train.blocks == held)
            pred = project(contrast(inner_train), h, lam)
            scores.append(mse(contrast(inner_test), pred))
        options.append(dict(lambda_gain=lam, inner_MSE=float(np.mean(scores))))
    chosen = min(options, key=lambda item: (item['inner_MSE'], -item['lambda_gain']))
    return float(chosen['lambda_gain']), options


def evaluate_dataset(ds, bases: dict[str, np.ndarray]) -> tuple[list[dict], list[dict], dict]:
    folds, choices, arrays = [], [], {}
    for model, h in bases.items():
        for held in np.unique(ds.blocks):
            tr = ds.subset(ds.blocks != held)
            te = ds.subset(ds.blocks == held)
            lam, opts = select_lambda(tr, h)
            observed = contrast(te)
            predicted = project(contrast(tr), h, lam)
            folds.append(dict(dataset=ds.key, model=model, held_block=int(held),
                              n_train=int(len(tr.y)), n_test=int(len(te.y)),
                              selected_lambda=lam, MSE=mse(observed, predicted),
                              zero_MSE=mse(observed, np.zeros_like(observed))))
            choices.append(dict(dataset=ds.key, model=model, held_block=int(held),
                                train_ids=tr.ids.tolist(), test_ids=te.ids.tolist(),
                                selected_lambda=lam, inner=opts))
            arrays[f'{model}_{int(held)}_observed'] = observed
            arrays[f'{model}_{int(held)}_predicted'] = predicted
    return folds, choices, arrays


def summary_metrics(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['dataset', 'model'], as_index=False).agg(
        MSE=('MSE', 'mean'), zero_MSE=('zero_MSE', 'mean'))
    summary['S_delta'] = 1 - summary.MSE / summary.zero_MSE
    summary['RMSE'] = np.sqrt(summary.MSE)
    return summary


def prestim_control(ds) -> tuple[list[dict], list[dict]]:
    """Past-only polynomial contrast transfer, fitted entirely before cue onset.

    This tests whether a simple direction-linked baseline shape transfers across
    blocks.  A null result cannot rule out non-linear preparation or eye motion.
    """
    axis = np.linspace(-1., 1., ds.prestim_only.shape[-1])
    h = np.polynomial.legendre.legvander(axis, 2)
    h /= np.sqrt(np.mean(h*h, axis=0))[None, :]

    def pre_contrast(part):
        x = part.prestim_only
        return x[part.y == 1].mean(axis=0) - x[part.y == -1].mean(axis=0)

    def fit(z, lam):
        gram = h.T @ h / len(h)
        gains = np.linalg.solve(gram + lam*np.eye(3), (z @ h / len(h)).T).T
        return gains @ h.T

    rows, choices = [], []
    for held in np.unique(ds.blocks):
        train = ds.subset(ds.blocks != held)
        test = ds.subset(ds.blocks == held)
        options = []
        for lam in LAMBDAS:
            losses = []
            for inner in np.unique(train.blocks):
                tr = train.subset(train.blocks != inner)
                va = train.subset(train.blocks == inner)
                losses.append(float(np.mean((pre_contrast(va) - fit(pre_contrast(tr), lam))**2)))
            options.append(dict(lambda_gain=lam, inner_MSE=float(np.mean(losses))))
        selected = min(options, key=lambda item: (item['inner_MSE'], -item['lambda_gain']))
        observed = pre_contrast(test)
        predicted = fit(pre_contrast(train), selected['lambda_gain'])
        rows.append(dict(dataset=ds.key, model='prestim_only_polynomial', held_block=int(held),
                         n_train=int(len(train.y)), n_test=int(len(test.y)),
                         selected_lambda=selected['lambda_gain'],
                         MSE=float(np.mean((observed-predicted)**2)),
                         zero_MSE=float(np.mean(observed**2))))
        choices.append(dict(dataset=ds.key, held_block=int(held),
                            train_ids=train.ids.tolist(), test_ids=test.ids.tolist(),
                            selected_lambda=selected['lambda_gain'], inner=options))
    return rows, choices


def bootstrap_intervals(rows: list[dict], pre_rows: list[dict]) -> pd.DataFrame:
    """Descriptive fold-resampling ranges; five folds are not independent studies."""
    frame = pd.concat((pd.DataFrame(rows), pd.DataFrame(pre_rows)), ignore_index=True)
    rng = np.random.default_rng(20260925)
    result = []
    for (dataset, model), group in frame.groupby(['dataset', 'model']):
        e = group.MSE.to_numpy()
        z = group.zero_MSE.to_numpy()
        picks = rng.integers(0, len(group), size=(10000, len(group)))
        sampled = 1 - e[picks].mean(axis=1) / z[picks].mean(axis=1)
        low, high = np.quantile(sampled, [.025, .975])
        result.append(dict(dataset=dataset, model=model, n_folds=len(group),
                           bootstrap_95_low=float(low), bootstrap_95_high=float(high)))
    return pd.DataFrame(result)


def plot_comparison(summary: pd.DataFrame, pre_summary: pd.DataFrame,
                    intervals: pd.DataFrame, counts: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = list(NAMES)
    fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(11, 6.8),
                                     gridspec_kw={'height_ratios': [3.5, 1.4]},
                                     layout='constrained')
    names = ['visual_neural_mass', 'ocular_step', 'slow_ramp']
    colors = ['#1767a2', '#bf6a22', '#687484']
    x = np.arange(len(keys))
    for j, name in enumerate(names):
        subset = summary.set_index(['dataset','model'])
        ax.bar(x + (j-1)*.23, [subset.loc[(k,name),'S_delta'] for k in keys],
               width=.21, label=name, color=colors[j])
    ax.plot(x, [pre_summary.set_index('dataset').loc[k,'S_delta'] for k in keys],
            color='#8f3b8f', marker='d', linestyle='--', label='prestim only control')
    ax.axhline(0., color='black', linewidth=.9)
    ax.set_xticks(x, [f'{k}\nn={counts[k]}' for k in keys])
    ax.set_ylabel('Held-out direction contrast S_delta')
    ax.set_title('VisCue 50–750 ms, three equal-weight windows: held-out right-minus-left EEG')
    ax.legend(fontsize=8, ncol=2)
    table_ax.set_axis_off()
    lookup = intervals.set_index(['dataset', 'model'])
    columns = ['visual_neural_mass', 'ocular_step', 'slow_ramp',
               'prestim_only_polynomial']
    cells = [[f"[{lookup.loc[(k,m),'bootstrap_95_low']:.2f}, "
              f"{lookup.loc[(k,m),'bootstrap_95_high']:.2f}]"
              for m in columns] for k in keys]
    tab = table_ax.table(cellText=cells, rowLabels=keys,
                         colLabels=['Neural', 'Ocular', 'Ramp', 'Pre only'],
                         loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.25)
    table_ax.set_title('95% descriptive fold-resampling ranges (5 blocks per group)',
                       fontsize=9, pad=4)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'q2_result/results/mechanism')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = audit_inputs()
    bases, meta = make_bases()
    rows, choices, pre_rows, pre_choices = [], [], [], []
    np.savez_compressed(args.output / 'fixed_bases.npz', times=TIMES, **bases)
    for key in NAMES:
        ds = independent_data(ROOT, key)
        r, c, a = evaluate_dataset(ds, bases)
        rows.extend(r)
        choices.extend(c)
        pr, pc = prestim_control(ds)
        pre_rows.extend(pr)
        pre_choices.extend(pc)
        np.savez_compressed(args.output / f'{key}_heldout_waveforms.npz', times=TIMES, **a)
    summary = summary_metrics(rows)
    pre_summary = summary_metrics(pre_rows)
    intervals = bootstrap_intervals(rows, pre_rows)
    pooled = pd.DataFrame(rows).groupby('model', as_index=False).agg(
        MSE=('MSE', 'mean'), zero_MSE=('zero_MSE', 'mean'))
    pooled['S_delta'] = 1 - pooled.MSE / pooled.zero_MSE
    pooled['RMSE'] = np.sqrt(pooled.MSE)
    pooled.insert(0, 'dataset', 'pooled_equal_fold')
    save_csv(rows, args.output / 'heldout_folds.csv')
    summary.to_csv(args.output / 'heldout_summary.csv', index=False, encoding='utf-8-sig')
    pooled.to_csv(args.output / 'pooled_summary.csv', index=False, encoding='utf-8-sig')
    save_csv(pre_rows, args.output / 'prestim_folds.csv')
    pre_summary.to_csv(args.output / 'prestim_summary.csv', index=False, encoding='utf-8-sig')
    intervals.to_csv(args.output / 'fold_resampling_ranges.csv', index=False, encoding='utf-8-sig')
    save_json(choices, args.output / 'inner_choices.json')
    save_json(pre_choices, args.output / 'prestim_inner_choices.json')
    counts = pd.DataFrame(rows).query("model == 'visual_neural_mass'").groupby('dataset').n_test.sum().to_dict()
    plot_comparison(summary, pre_summary, intervals, counts,
                    args.output / 'heldout_comparison.png')
    save_json(dict(inputs=inputs, model=meta,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_docx=DOCX, source_media=MEDIA,
                   lambdas=list(LAMBDAS),
                   caveat='Effective scalp gains are estimated; sources and ocular activity are not identified.'),
              args.output / 'manifest.json')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
