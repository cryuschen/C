"""Plot post-cue direction decoding before and after past-only control."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    by_group = pd.read_csv(args.output / 'per_dataset_metrics.csv')
    overall = pd.read_csv(args.output / 'pooled_metrics.csv')
    groups = ['A1', 'A2', 'B1', 'B2', '合并']
    models = [('pre_lateral_3', '仅刺激前'),
              ('post_lateral_early', '刺激后原特征'),
              ('post_given_pre', '扣除刺激前可预测部分')]
    values = []
    for model, _ in models:
        a = by_group[by_group.model == model].set_index('dataset').loc[groups[:-1]].BA.tolist()
        a.append(float(overall[overall.model == model].BA.iloc[0]))
        values.append(a)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(9.5, 4.8), layout='constrained')
    centers = np.arange(len(groups))
    for index, ((_, label), result) in enumerate(zip(models, values)):
        ax.bar(centers + (index - 1) * .23, result, width=.22,
               label=label, color=('#7996A6', '#D97F28', '#6F8C65')[index])
    ax.axhline(.5, color='#555555', linestyle='--', linewidth=1)
    ax.set_xticks(centers, groups)
    ax.set_ylim(.4, .7)
    ax.set_ylabel('留出时间块平衡准确率')
    ax.set_xlabel('探索性内部数据；剩余特征只控制与刺激前三窗口的线性相关',
                  labelpad=12, fontsize=9)
    ax.set_title('刺激前控制后，早期额区侧化的方向判别接近随机')
    ax.legend(loc='upper left', fontsize=9)
    fig.savefig(args.output / 'incremental_postcue.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
