"""Plot the held-out direction score and its block-permutation reference."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.output / 'per_dataset_metrics.csv')
    summary = pd.read_csv(args.output / 'pooled_metrics.csv')
    null = pd.read_csv(args.output / 'permutation_null.csv')
    chosen = 'early_lateral'
    effect = frame[frame.model == chosen].set_index('dataset').loc[['A1', 'A2', 'B1', 'B2']]
    total = summary[summary.model == chosen].iloc[0]
    maximum = null.groupby('iteration').BA.max()
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].bar(range(5), [*effect.BA, total.BA], color=['#7B9FB3'] * 4 + ['#D77C27'])
    axes[0].axhline(.5, color='#555555', linestyle='--', linewidth=1)
    axes[0].set_xticks(range(5), ['A1', 'A2', 'B1', 'B2', '合并'])
    axes[0].set_ylim(.35, .7)
    axes[0].set_ylabel('留出时间块平衡准确率')
    axes[0].set_title('F3−F4 侧化均值，50–250 ms')
    axes[1].hist(maximum, bins=30, color='#93A8B0', edgecolor='white')
    axes[1].axvline(total.BA, color='#C95D22', linewidth=2,
                    label=f'观测 BA={total.BA:.3f}')
    axes[1].set_xlabel('三候选流程最大 BA（分块置换）')
    axes[1].set_ylabel('置换次数')
    axes[1].set_title(f'内部开发数据；maxT p={total.p_maxT_three:.3f}')
    axes[1].legend()
    fig.suptitle('可识别性检验；不能判定信号的神经来源')
    fig.savefig(args.output / 'pooled_discrimination.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
