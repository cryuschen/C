"""Rebuild the Q2 visual forward model from the figure embedded in the problem."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2model.data import NAMES, TIMES, WEIGHTS, independent_data, means, save_csv, save_json
from q2model.mechanism import neural_forward
from q2model.stimulus_shape import DOCX, MEDIA, cue_images, shape_inputs
from compare_neural_eye_models import (filtered_template, fit_gain, mse, select_ridge,
                                       decode, score_metrics)
from optimize_direction_features import ROOT, audit_inputs, permute_blocks


def plot(out: Path, cues: dict, encoded: dict, forward: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
    for ax, key, title in zip(axes[0, :2], ('left', 'right'), ('左向刺激', '右向刺激')):
        ax.imshow(cues[key], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(title); ax.set_axis_off()
        ax.set_xlim(45, 90); ax.set_ylim(105, 25)
    act = encoded['activation']
    axes[0, 2].bar([0, 1], act[0], width=.35, label='左向输入')
    axes[0, 2].bar([.35, 1.35], act[1], width=.35, label='右向输入')
    axes[0, 2].set_xticks([.18, 1.18], ['左向选择群', '右向选择群'])
    axes[0, 2].set_ylabel('归一化空间组合响应'); axes[0, 2].legend(fontsize=8)
    axes[0, 2].set_title('形状驱动：模型计算值')
    t = forward['t'] * 1000
    for c, ax in enumerate(axes[1]):
        ax.plot(t, forward['eeg'][0, c], label='左向')
        ax.plot(t, forward['eeg'][1, c], label='右向')
        ax.set_xlim(0, 750); ax.set_xlabel('刺激后时间 (ms)')
        ax.set_ylabel('仿真电位（任意单位）')
        ax.set_title(('Fz', 'F3', 'F4')[c])
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('原题嵌入图像 → 空间选择性 → 神经群体 → 头皮模拟；非实测源定位')
    fig.savefig(out / 'actual_cue_forward.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'Q2_stimulus_model_v2')
    parser.add_argument('--permutations', type=int, default=999)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = audit_inputs()
    cues = cue_images(ROOT)
    encoded = shape_inputs(cues)
    forward = neural_forward(encoded['inputs'])
    for key in ('left', 'right'):
        mask = cues[key]
        rgb = np.full((*mask.shape, 3), 240, dtype=np.uint8)
        rgb[:, :, 0] = np.clip(240 * (1 - mask), 0, 255)
        rgb[:, :, 1] = np.clip(240 * (1 - mask), 0, 255)
        rgb[:, :, 2] = np.clip(240 + 15 * mask, 0, 255)
        Image.fromarray(rgb).save(args.output / f'{key}_cue.png')
    save_csv([dict(stimulus=name, f_left=float(row[0]), f_right=float(row[1]),
                   common=float(row[2]), signed_selectivity=float(eta))
              for name, row, eta in zip(encoded['names'], encoded['inputs'],
                                        encoded['selectivity'])], args.output / 'shape_inputs.csv')
    np.savez_compressed(args.output / 'forward.npz', **forward,
                        left_mask=cues['left'], right_mask=cues['right'],
                        inputs=encoded['inputs'], selectivity=encoded['selectivity'])
    plot(args.output, cues, encoded, forward)
    h = filtered_template(forward['t'], forward['dipoles'][1, 0] - forward['dipoles'][0, 0])
    rows, decoded, null = [], [], []
    for key in NAMES:
        ds = independent_data(ROOT, key)
        for block in np.unique(ds.blocks):
            tr = ds.subset(ds.blocks != block); te = ds.subset(ds.blocks == block)
            lam, _ = select_ridge(tr, h)
            delta_tr = means(tr.x, tr.y)[1] - means(tr.x, tr.y)[0]
            delta_te = means(te.x, te.y)[1] - means(te.x, te.y)[0]
            rows.append(dict(dataset=key, fold=int(block), lambda_gain=lam,
                             MSE=mse(delta_te, fit_gain(delta_tr, h, lam)),
                             zero_MSE=mse(delta_te, np.zeros_like(delta_te))))
        pred = decode(ds, h)
        observed = score_metrics(pred)
        rng = np.random.default_rng(np.random.SeedSequence([20260925, ord(key[0]),
                                                             int(key[1]), 17]))
        values = []
        for i in range(args.permutations):
            perm = permute_blocks(ds.y, ds.blocks, rng)
            value = score_metrics(decode(ds, h, perm))['BA']
            values.append(value)
            null.append(dict(dataset=key, iteration=i + 1, BA=value))
        decoded.append(dict(dataset=key, n=len(ds.y), **observed,
                            p_raw=(1 + sum(v >= observed['BA'] for v in values)) /
                                  (args.permutations + 1)))
        print(key, decoded[-1], flush=True)
    summary = pd.DataFrame(rows).groupby('dataset', as_index=False).agg(
        MSE=('MSE', 'mean'), zero_MSE=('zero_MSE', 'mean'))
    summary['S_delta'] = 1 - summary.MSE / summary.zero_MSE
    summary['RMSE'] = np.sqrt(summary.MSE)
    save_csv(rows, args.output / 'heldout_contrast_folds.csv')
    summary.to_csv(args.output / 'heldout_contrast_summary.csv', index=False,
                   encoding='utf-8-sig')
    save_csv(decoded, args.output / 'decoding_summary.csv')
    save_csv(null, args.output / 'permutation_null.csv')
    with ZipFile(ROOT / DOCX) as archive:
        media_hash = hashlib.sha256(archive.read(MEDIA)).hexdigest()
    save_json(dict(inputs=inputs, source_docx_sha256=hashlib.sha256((ROOT / DOCX).read_bytes()).hexdigest(),
                   source_media=MEDIA, source_media_sha256=media_hash,
                   source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   shape_code_sha256=hashlib.sha256((ROOT / 'Q2/q2model/stimulus_shape.py').read_bytes()).hexdigest(),
                   n_permutations=args.permutations,
                   caveat='embedded diagram is a schematic cue, not a calibrated display screenshot'),
              args.output / 'manifest.json')


if __name__ == '__main__':
    main()
