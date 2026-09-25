#!/usr/bin/env python3
"""Paired Q2 comparison with Q1 correction refitted in every nested split."""
import os
for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[var] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/q2-fold-denoising-mpl')
import argparse
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
from q2model.data import (NAMES, TIMES, means, window_features, independent_data,
                          save_json, save_csv, sha)
from q2model.model import waveform_metrics
from q2model.validation import classify, fit_record, metrics
from q2model.fold_denoising import DenoiserCache, prepare_fold, fit_prepared_model

ROOT = Path(__file__).resolve().parents[1]


def standardized_difference(x, y):
    left, right = x[y == -1], x[y == 1]
    if min(len(left), len(right)) < 2:
        return np.full(x.shape[1], np.nan)
    pooled = ((len(left)-1)*left.var(0, ddof=1)+(len(right)-1)*right.var(0, ddof=1))/(len(x)-2)
    return (right.mean(0)-left.mean(0))/np.sqrt(np.maximum(pooled, 1e-12))


def run_fold(ds, block, branch, cache, quick=False, out=None):
    start = time.monotonic()
    tr, te, pairs, denoiser, inner_audits = prepare_fold(ds, block, branch, cache)
    fit = fit_prepared_model(tr, pairs, quick)
    train_features, test_features = fit.features(tr.x), fit.features(te.x)
    pred, score, scaler, classifier = classify(train_features, tr.y, test_features)
    train_window, test_window = window_features(tr.x), window_features(te.x)
    wp, ws, _, _ = classify(train_window, tr.y, test_window)
    predictions = {'mechanism_selected': (pred, score), 'window_mean_9': (wp, ws)}
    rows = []
    for name, (p, s) in predictions.items():
        rows.extend(dict(dataset=ds.key, branch=branch, fold=int(block), trial_id=int(i),
                         truth=int(y), model=name, prediction=int(a), score=float(b))
                    for i, y, a, b in zip(te.ids, te.y, p, s))
    raw_test = ds.subset(ds.blocks == block)
    wave, channel_wave = [], []
    for target_name, obs in [('common_baseline', means(raw_test.x, raw_test.y)),
                             ('branch_output', means(te.x, te.y))]:
        for name, prediction in [('mechanism', fit.predict()), ('training_template', means(tr.x, tr.y))]:
            detail, summary = waveform_metrics(obs, prediction)
            wave.append(dict(dataset=ds.key, branch=branch, fold=int(block), model=name,
                             target=target_name, **summary))
            channel_wave.extend(dict(dataset=ds.key, branch=branch, fold=int(block), model=name,
                                     target=target_name, **r) for r in detail)
    # Fixed 9-window features have identical definitions in every rank/fold.
    train_effect = standardized_difference(train_window, tr.y)
    test_effect = standardized_difference(test_window, te.y)
    feature_rows = [dict(dataset=ds.key, branch=branch, fold=int(block), feature=j+1,
                         spatial_mode=('common', 'midline', 'lateral')[j//3],
                         window=('50_250', '250_500', '500_750')[j%3],
                         train_effect=float(a), test_effect=float(b),
                         same_sign=bool(a*b > 0)) for j, (a, b) in enumerate(zip(train_effect, test_effect))]
    record = dict(dataset=ds.key, branch=branch, fold=int(block),
                  train_ids=tr.ids.tolist(), test_ids=te.ids.tolist(),
                  fit=fit_record(fit), inner_denoisers=inner_audits,
                  denoiser=denoiser.audit if denoiser else None,
                  seconds=time.monotonic()-start)
    arrays = dict(test_ids=te.ids, truth=te.y, before=raw_test.x, after=te.x,
                  prediction=pred, score=score, features=test_features,
                  standardized_features=scaler.transform(test_features),
                  classifier_mean=scaler.mean_, classifier_scale=scaler.scale_,
                  classifier_coef=classifier.coef_, classifier_intercept=classifier.intercept_,
                  h=fit.h, theta=fit.theta, A=fit.A, D=fit.D, predicted_erp=fit.predict(),
                  observed_erp=means(te.x, te.y), common_observed_erp=means(raw_test.x, raw_test.y))
    if denoiser:
        arrays.update({f'denoiser_{k}': v for k, v in denoiser.arrays().items()})
    if out:
        dest = Path(out)/'folds'/ds.key/branch
        dest.mkdir(parents=True, exist_ok=True)
        save_json(record, dest/f'fold_{block}.json')
        np.savez_compressed(dest/f'fold_{block}.npz', **arrays)
    return rows, wave, channel_wave, feature_rows, record, arrays


def paired_intervals(predictions, n_boot=2000, seed=20260925):
    """Same sampled time blocks for both methods; models are held fixed."""
    frame = pd.DataFrame(predictions)
    summary, differences = [], []
    rng = np.random.default_rng(seed)
    for (key, model), group in frame.groupby(['dataset', 'model'], sort=False):
        stages = {b: g.sort_values('trial_id').reset_index(drop=True) for b, g in group.groupby('branch')}
        a, b = stages['baseline'], stages['denoised']
        for column in ('trial_id', 'truth', 'fold'):
            np.testing.assert_array_equal(a[column], b[column])
        point = {k: metrics(g.truth.to_numpy(), g.prediction.to_numpy(), g.score.to_numpy())
                 for k, g in stages.items()}
        blocks = np.unique(a.fold)
        indices = {block: np.flatnonzero(a.fold.to_numpy() == block) for block in blocks}
        draws = {k: [] for k in stages}
        # All resamples preserve pairing; five blocks provide limited precision.
        for _ in range(n_boot):
            idx = np.concatenate([indices[v] for v in rng.choice(blocks, len(blocks))])
            for k, g in stages.items():
                y = g.iloc[idx]
                m = metrics(y.truth.to_numpy(), y.prediction.to_numpy(), y.score.to_numpy())
                draws[k].append([m['BA'], m['AUC']])
        for k in stages:
            ci = np.nanquantile(draws[k], [.025, .975], axis=0)
            summary.append(dict(dataset=key, model=model, branch=k, **point[k],
                                BA_low=ci[0, 0], BA_high=ci[1, 0], AUC_low=ci[0, 1], AUC_high=ci[1, 1]))
        changes = np.asarray(draws['denoised'])-np.asarray(draws['baseline'])
        ci = np.nanquantile(changes, [.025, .975], axis=0)
        differences.append(dict(dataset=key, model=model, n=len(a),
                                BA_change=point['denoised']['BA']-point['baseline']['BA'],
                                BA_change_low=ci[0, 0], BA_change_high=ci[1, 0],
                                AUC_change=point['denoised']['AUC']-point['baseline']['AUC'],
                                AUC_change_low=ci[0, 1], AUC_change_high=ci[1, 1],
                                n_boot=n_boot, interval='paired_time_blocks_fixed_models'))
    return summary, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'Q2_fold_denoising_results')
    parser.add_argument('--quick', action='store_true', help='Debug grid only; use separate output')
    parser.add_argument('--datasets', nargs='+', choices=list(NAMES), default=list(NAMES))
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT or out.is_relative_to(ROOT/'data'):
        raise ValueError('Unsafe output directory')
    sources = [Path(__file__), ROOT/'EEG_P300_artifact_correction_unsupervised.py',
               *sorted((ROOT/'Q2/q2model').glob('*.py'))]
    config = dict(datasets=args.datasets, quick=args.quick, seed=args.seed,
                  bootstrap=args.bootstrap, guard_seconds=24., branches=['baseline', 'denoised'],
                  selected_grid_points=3 if args.quick else 4,
                  denoiser_selector='unchanged_Q1_training_label_assisted',
                  permutation_tests=False, full_parameter_bootstrap=False)
    source = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    inputs = {f'data/VisualCog{k[0]}_Task-{k[1]}.mat': sha(ROOT/'data'/f'VisualCog{k[0]}_Task-{k[1]}.mat') for k in args.datasets}
    signature = hashlib.sha256(json.dumps(dict(config=config, source=source, inputs=inputs), sort_keys=True).encode()).hexdigest()
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not args.resume or signature != manifest['signature']:
            raise RuntimeError('Existing output or changed code/config: use a new directory')
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError('Output directory is not empty')
        out.mkdir(parents=True, exist_ok=True)
        manifest = dict(signature=signature, config=config, source=source, inputs=inputs,
                        created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                        numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    manifest['status'] = 'running'; save_json(manifest, manifest_path)
    (out/'audit').mkdir(exist_ok=True)
    predictions, waves, wave_channels, feature_rows, records = [], [], [], [], []
    try:
        for key in args.datasets:
            ds = independent_data(ROOT, key, 24., out)
            print(f'{key}: {len(ds.x)} identical retained trials for both branches', flush=True)
            cache = DenoiserCache(args.seed)
            for block in np.unique(ds.blocks):
                for branch in ('baseline', 'denoised'):
                    start = time.monotonic()
                    # Resume validates provenance but recomputes folds, preventing
                    # partially written folds from being accepted silently.
                    result = run_fold(ds, int(block), branch, cache, args.quick, out)
                    p, w, c, f, record, _ = result
                    predictions.extend(p); waves.extend(w); wave_channels.extend(c)
                    feature_rows.extend(f); records.append(record)
                    print(f'{key} block {block} {branch}: {time.monotonic()-start:.1f}s; '
                          f'rank={record["fit"]["rank"]}; '
                          f'correction={record["denoiser"]["candidate"] if record["denoiser"] else "none"}', flush=True)
            save_csv(predictions, out/'predictions.csv')
        summary, differences = paired_intervals(predictions, args.bootstrap, args.seed)
        for data, name in [(predictions, 'predictions'), (summary, 'classification_summary'),
                           (differences, 'paired_changes'), (waves, 'waveform_metrics'),
                           (wave_channels, 'waveform_channel_metrics'), (feature_rows, 'feature_stability')]:
            save_csv(data, out/(name+'.csv'))
        save_json(records, out/'all_fold_records.json')
        manifest['status'] = 'comparison_complete'
        manifest['counts'] = {key: sum(r['dataset']==key for r in records) for key in args.datasets}
        manifest['output_hashes'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
                                     if p.is_file() and p != manifest_path}
        save_json(manifest, manifest_path)
        print(pd.DataFrame(summary).query('model == "mechanism_selected"').to_string(index=False), flush=True)
        print('Comparison complete: '+str(out), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc)); save_json(manifest, manifest_path)
        raise


if __name__ == '__main__':
    main()
