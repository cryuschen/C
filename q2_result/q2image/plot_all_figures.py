# -*- coding: utf-8 -*-
"""
Q2 绘图独立模块 (Q2 Visualization Suite)
=========================================
该文件将 Q2 问中所有的绘图代码整合为独立的执行文件。
无需重新运行繁重的模型训练或脑电预处理，直接从 q2_result 中已保存的
数值结果（npz / csv / json）读取数据，完整复现 q2_result 里的所有主图、附图及子模型分析图。

包含图表：
---------
1. Q2_图1_机制与形状编码 (.png, .pdf)
2. Q2_图2_皮层内部响应 及 补图 (A1, A2, B1, B2) (.png, .pdf)
3. Q2_附图_LGN (A1, A2, B1, B2) (.png, .pdf)
4. Q2_图3_ERP拟合与留出 及 补图 (A1, A2, B1, B2) (.png, .pdf)
5. Q2_附图_Q1衔接 (A1, A2, B1, B2) (.png, .pdf)
6. Q2_图4_左右差异波 (.png, .pdf)
7. Q2_图5_空间侧化 (.png, .pdf)
8. Q2_图6_消融与敏感性 (.png, .pdf)
9. Q2_图7_特征与判别 及 补图 (A1, A2, B1, B2) (.png, .pdf)
10. Q2_附图_判别汇总 (.png, .pdf)
11. results/decoder/decoder_v3_erp.png (VisCue 留出解码与置换检验)
12. results/mechanism/heldout_comparison.png (机制迁移与各模型留出对比及区间表)
13. q1_model/第二问_Q1数据新模型.png (基于Q1数据的级联机制模型与协方差判别)

使用方法：
---------
直接运行本脚本即可生成全部图表：
    python plot_all_figures.py
或指定结果目录：
    python plot_all_figures.py --result-dir ../
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ==========================================
# 常量与样式配置
# ==========================================
BLUE = '#2667a5'
RED = '#c44e52'
GREY = '#687782'
GREEN = '#28816b'

CHANNELS = ('Fz', 'F3', 'F4')
NAMES = ('A1', 'A2', 'B1', 'B2')

FEATURE_NAMES = {
    'amplitude': '均值振幅',
    'peak_latency': '峰幅与潜伏期',
    'spatial': '加入侧化',
    'mechanism': '加入机制',
    'covariance': '协方差基线',
    'past_only': '刺激前',
    'previous_cue': '前次标签',
    'post_given_past': '扣除过去'
}

MODEL_NAMES = {
    'full': '完整模型',
    'no_shape': '取消形状',
    'no_recurrence': '取消跨群复发',
    'symmetric_readout': '对称观测',
    'gamma': '六时间核',
    'step': '平滑阶跃',
    'ramp': '慢斜坡',
    'zero': '零方向差'
}

FS = 256
TIMES = np.arange(-64, 205) / FS


def setup_matplotlib():
    """初始化中文字体及统一的图表样式"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
    chosen = next((f for f in candidates if f in available), None)
    if chosen is not None:
        plt.rcParams['font.family'] = chosen
    plt.rcParams.update({
        'axes.unicode_minus': False,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 240,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def save_figure(fig, out_dir, name, copy_dirs=None, dpi=240, save_pdf=True):
    """保存 PNG 及可选的 PDF 格式，支持同步备份到指定目录"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    fig.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=dpi)
    pdf_path = None
    if save_pdf:
        pdf_path = out_dir / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    if copy_dirs:
        for cdir in copy_dirs:
            cdir = Path(cdir)
            cdir.mkdir(parents=True, exist_ok=True)
            target_png = cdir / f"{name}.png"
            if target_png.resolve() != png_path.resolve():
                shutil.copy2(png_path, target_png)
            if save_pdf and pdf_path and pdf_path.exists():
                target_pdf = cdir / f"{name}.pdf"
                if target_pdf.resolve() != pdf_path.resolve():
                    shutil.copy2(pdf_path, target_pdf)
    plt.close(fig)


def decorate(ax, ylabel=True):
    """为 ERP 曲线子图添加统一的标准时间窗标记与坐标标注"""
    ax.axvline(0, color=GREY, lw=0.6)
    ax.axhline(0, color=GREY, lw=0.5, alpha=0.5)
    ax.axvspan(250, 500, color='#e5d6a6', alpha=0.24)
    ax.set_xlim(-250, 800)
    ax.set_xlabel('刺激后时间 / ms')
    if ylabel:
        ax.set_ylabel('记录幅值单位')


# ==========================================
# 各图表绘制函数
# ==========================================

def plot_fig1_mechanism_and_shape(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """图1: 从三角空间结构到额区 EEG 的可计算路径"""
    shape = np.load(res_dir / 'Q2_形状输入.npz')
    fig = plt.figure(figsize=(12, 6.1), layout='constrained')
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.3])
    labels = ['左三角', '右三角', '缺失尖角（模拟）', '位置打乱（模拟）']
    for i, label in enumerate(labels):
        ax = fig.add_subplot(gs[0, i])
        im = shape['images'][i]
        ys, xs = np.nonzero(shape['images'][:2].sum(0))
        ax.imshow(im[max(0, ys.min() - 5):ys.max() + 6, max(0, xs.min() - 5):xs.max() + 6],
                  cmap='Blues', vmin=0, vmax=1)
        ax.set_title(label + f"\n选择群输入 ({shape['inputs'][i, 0]:.3f}, {shape['inputs'][i, 1]:.3f})")
        ax.axis('off')
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        ('局部对比与\n边缘空间组合', .07),
        ('LGN\n三路中继', .25),
        ('早期视觉 → 形状整合 → 额区\n每级三个形状/共同 E–I 群体', .52),
        ('突触等效源\n共享观测矩阵 L', .78),
        ('Fz\nF3 / F4', .94)
    ]
    for text, x in boxes:
        ax.text(x, .62, text, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=.6', fc='#eef3f7', ec=BLUE))
    for x1, x2 in ((.14, .2), (.3, .36), (.68, .72), (.84, .9)):
        ax.annotate('', xy=(x2, .62), xytext=(x1, .62),
                    arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))
    ax.text(.5, .16, '左右条件共享网络与观测参数；差异由图像输入进入。\n形状偏好 ≠ 视野位置 ≠ 半球定位；示意图像素不是校准刺激记录。',
            ha='center', va='center')
    fig.suptitle('图1  从三角空间结构到额区 EEG 的可计算路径', fontsize=13)
    save_figure(fig, target_dir, 'Q2_图1_机制与形状编码', copy_dirs=copy_dirs)
    shape.close()


def plot_fig2_cortical_responses(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """图2: 皮层内部响应（主图及各组补图）"""
    w = np.load(res_dir / 'Q2_机制波形.npz')
    for key in NAMES:
        suffix = '' if key == 'A1' else f'_补图_{key}'
        t = w[f'{key}_neural_t'] * 1000
        s = w[f'{key}_neural_state']
        q = w[f'{key}_neural_q']
        fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, layout='constrained')
        for stage, name in enumerate(('早期视觉', '形状整合', '额区响应')):
            for group, style, gname in [(0, '-', '左形状偏好群')]:
                for cond, color in enumerate((BLUE, RED)):
                    axes[stage, 0].plot(t, s[cond, 3 + stage * 3 + group], color=color, ls=style, lw=1)
                    axes[stage, 1].plot(t, s[cond, 12 + stage * 3 + group], color=color, ls=style, lw=1)
                    axes[stage, 2].plot(t, q[cond, stage, group], color=color, ls=style, lw=1,
                                       label=f'{("左刺激", "右刺激")[cond]} / {gname}')
            for col, ct in enumerate(('兴奋群 E', '抑制群 I', '突触等效源 q')):
                axes[stage, col].set_title(name + ' · ' + ct)
                axes[stage, col].set_xlim(-100, 1100)
                axes[stage, col].axvspan(0, 203.125, color=GREY, alpha=.1)
                axes[stage, col].set_xlabel('时间 / ms')
                axes[stage, col].set_ylabel('模型单位')
        axes[0, 2].legend(fontsize=6, ncol=2)
        fig.suptitle(f'图2  {key} 同一左形状偏好群对左右刺激的响应（模型内部量）', fontsize=12)
        save_figure(fig, target_dir, f'Q2_图2_皮层内部响应{suffix}', copy_dirs=copy_dirs)
    w.close()


def plot_fig_lgn(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """附图: LGN 中继动态 (A1, A2, B1, B2)"""
    w = np.load(res_dir / 'Q2_机制波形.npz')
    for key in NAMES:
        t = w[f'{key}_neural_t'] * 1000
        s = w[f'{key}_neural_state']
        fig, ax = plt.subplots(figsize=(7, 3), layout='constrained')
        for cond, color in enumerate((BLUE, RED)):
            for g, style in enumerate(('-', '--', ':')):
                ax.plot(t, s[cond, g], color=color, ls=style, label=f'{("左", "右")[cond]}刺激·群{g+1}')
        ax.set(xlim=(-100, 600), xlabel='时间 / ms', ylabel='模型单位', title=f'{key} LGN 中继动态')
        ax.legend(ncol=3)
        save_figure(fig, target_dir, f'Q2_附图_LGN_{key}', copy_dirs=copy_dirs)
    w.close()


def plot_fig3_erp_and_heldout(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """图3: 左右 ERP 的描述拟合与独立时间块预测（主图及各组补图）"""
    w = np.load(res_dir / 'Q2_机制波形.npz')
    ms = 1000 * w['times_s']
    for key in NAMES:
        suffix = '' if key == 'A1' else f'_补图_{key}'
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, layout='constrained')
        pairs = [(w[f'{key}_strict_all_observed'], w[f'{key}_strict_all_predicted'], '同样本描述拟合'),
                 (w[f'{key}_full_observed'].mean(0), w[f'{key}_full_predicted'].mean(0), '五个留出块等权汇总')]
        for row, (obs, pred, title) in enumerate(pairs):
            for ch in range(3):
                ax = axes[row, ch]
                for cond, color in enumerate((BLUE, RED)):
                    ax.plot(ms, obs[cond, ch], color=color, label=f'{("左", "右")[cond]}·实测')
                    ax.plot(ms, pred[cond, ch], color=color, ls='--', label=f'{("左", "右")[cond]}·模型')
                decorate(ax)
                ax.set_title(f'{title} · {CHANNELS[ch]}')
        axes[0, 0].legend(ncol=2)
        fig.suptitle(f'图3  {key} 左右 ERP 的描述拟合与独立时间块预测', fontsize=12)
        save_figure(fig, target_dir, f'Q2_图3_ERP拟合与留出{suffix}', copy_dirs=copy_dirs)
    w.close()


def plot_fig_q1_transfer(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """附图: Q1 衔接对比 (A1, A2, B1, B2)"""
    w = np.load(res_dir / 'Q2_机制波形.npz')
    ms = 1000 * w['times_s']
    for key in NAMES:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
        for row, stage in enumerate(('q1_before', 'q1_v7')):
            for ch in range(3):
                ax = axes[row, ch]
                for cond, color in enumerate((BLUE, RED)):
                    ax.plot(ms, w[f'{key}_{stage}_observed'][cond, ch], color=color)
                    ax.plot(ms, w[f'{key}_{stage}_predicted'][cond, ch], color=color, ls='--')
                decorate(ax)
                ax.set_title(f'{stage} · {CHANNELS[ch]}')
        fig.suptitle(f'{key} Q1 衔接：实线为条件均值，虚线为同样本描述拟合（不计入预测证据）')
        save_figure(fig, target_dir, f'Q2_附图_Q1衔接_{key}', copy_dirs=copy_dirs)
    w.close()


def plot_fig4_direction_diff(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """图4: 四组数据的方向差分留出预测"""
    w = np.load(res_dir / 'Q2_机制波形.npz')
    summary = pd.read_csv(res_dir / 'Q2_机制汇总.csv')
    ms = 1000 * w['times_s']
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True, layout='constrained')
    for r, key in enumerate(NAMES):
        obs = w[f'{key}_full_observed'].mean(0)
        pred = w[f'{key}_full_predicted'].mean(0)
        ci = w[f'{key}_delta_ci']
        score = summary[(summary.dataset == key) & (summary.model == 'full')].iloc[0]
        for ch in range(3):
            ax = axes[r, ch]
            ax.fill_between(ms, ci[0, ch], ci[1, ch], color=BLUE, alpha=.17, label='实测点态95%区间')
            ax.plot(ms, obs[1, ch] - obs[0, ch], color=BLUE, label='实测 R−L')
            ax.plot(ms, pred[1, ch] - pred[0, ch], color=RED, ls='--', label='留出预测 R−L')
            decorate(ax)
            ax.set_title(f'{key} · {CHANNELS[ch]}' + (f'    SΔ={score.S_delta:+.3f}' if ch == 0 else ''))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('图4  四组数据的方向差分留出预测；阴影不是模型预测区间', fontsize=12)
    save_figure(fig, target_dir, 'Q2_图4_左右差异波', copy_dirs=copy_dirs)
    w.close()


def plot_fig5_spatial_lateralization(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """图5: 额区侧化的实测与预测"""
    w = np.load(res_dir / 'Q2_机制波形.npz')
    data = np.load(res_dir / 'Q2_分析数据.npz')
    ms = 1000 * w['times_s']
    fig, axes = plt.subplots(4, 3, figsize=(12, 10), layout='constrained')
    for row, key in enumerate(NAMES):
        obs = w[f'{key}_full_observed'].mean(0)
        pred = w[f'{key}_full_predicted'].mean(0)
        x = data[f'{key}_x']
        y = data[f'{key}_y']
        mask = (TIMES >= .25) & (TIMES < .5)
        amp = x[..., mask].mean(-1)
        floor = max(np.quantile(abs(amp[:, 1]) + abs(amp[:, 2]), .1), 1e-8)
        trial_li = (amp[:, 2] - amp[:, 1]) / np.maximum(abs(amp[:, 1]) + abs(amp[:, 2]), floor)
        for cond, color in enumerate((BLUE, RED)):
            for waves, style, label in ((obs, '-', '实测'), (pred, '--', '预测')):
                axes[row, 0].plot(ms, waves[cond, 2] - waves[cond, 1], color=color, ls=style,
                                 label=f'{("左", "右")[cond]}·{label}')
                a = waves[cond]
                li = (a[2] - a[1]) / np.maximum(abs(a[1]) + abs(a[2]), floor)
                axes[row, 1].plot(ms, li, color=color, ls=style)
        decorate(axes[row, 0])
        decorate(axes[row, 1], False)
        axes[row, 1].set_ylabel('稳定化侧化指数')
        axes[row, 0].set_title(f'{key} · F4−F3')
        axes[row, 1].set_title(f'{key} · 时变侧化指数')
        axes[row, 2].boxplot([trial_li[y == -1], trial_li[y == 1]], tick_labels=['左', '右'], showfliers=False)
        axes[row, 2].set_title(f'{key} · 250–500 ms 试次 LI')
        axes[row, 2].axhline(0, color=GREY, lw=.5)
    axes[0, 0].legend(ncol=2)
    fig.suptitle('图5  额区侧化的实测与预测；稳定项为全数据描述值', fontsize=12)
    save_figure(fig, target_dir, 'Q2_图5_空间侧化', copy_dirs=copy_dirs)
    w.close()
    data.close()


def plot_fig6_ablation_and_sensitivity(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """图6: 模型比较、机制消融和敏感性"""
    summary = pd.read_csv(res_dir / 'Q2_机制汇总.csv')
    fixed = pd.read_csv(res_dir / 'Q2_固定参数消融.csv')
    sensitivity = pd.read_csv(res_dir / 'Q2_参数敏感性.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    model_order = list(MODEL_NAMES)
    for j, key in enumerate(NAMES):
        vals = summary[summary.dataset == key].set_index('model').loc[model_order]
        offset = (j - 1.5) * .18
        axes[0, 0].bar(np.arange(len(model_order)) + offset, vals.S_delta, width=.18, label=key)
    axes[0, 0].set_xticks(np.arange(len(model_order)), [MODEL_NAMES[m] for m in model_order], rotation=30, ha='right')
    axes[0, 0].set_title('训练内重新拟合后的留出 SΔ')
    axes[0, 0].axhline(0, color=GREY, lw=.8)
    axes[0, 0].legend(ncol=4)

    agg = fixed.groupby(['dataset', 'intervention'])[['delta_MSE', 'full_delta_MSE']].mean()
    for j, ab in enumerate(('no_shape', 'no_recurrence', 'symmetric_readout')):
        val = agg.xs(ab, level=1)
        relative = (val.delta_MSE - val.full_delta_MSE) / val.full_delta_MSE
        axes[0, 1].bar(np.arange(4) + (j - 1) * .24, relative, width=.24, label=MODEL_NAMES[ab])
    axes[0, 1].set_xticks(range(4), NAMES)
    axes[0, 1].set_title('固定参数干预：差分 MSE 相对变化')
    axes[0, 1].axhline(0, color=GREY, lw=.8)
    axes[0, 1].legend(fontsize=7)

    for j, p in enumerate(('time_scale', 'recurrence', 'F4_readout')):
        s = sensitivity[sensitivity.parameter == p].groupby('factor').S_delta.mean()
        axes[1, 0].plot(s.index, s.values, marker='o', label=p)
    axes[1, 0].set(title='参数 ±20%：20 折 SΔ 等权均值', xlabel='乘数', ylabel='SΔ')
    axes[1, 0].legend()

    vals = summary[summary.model == 'full']
    axes[1, 1].errorbar(np.arange(4), vals.S_delta,
                        yerr=np.array([np.maximum(vals.S_delta - vals.S_low, 0),
                                       np.maximum(vals.S_high - vals.S_delta, 0)]),
                        fmt='o', color=BLUE, capsize=4)
    axes[1, 1].set_xticks(range(4), vals.dataset)
    axes[1, 1].axhline(0, color=GREY, lw=.8)
    axes[1, 1].set_title('完整模型：时间块重采样95%范围')
    axes[1, 1].set_ylabel('SΔ')
    fig.suptitle('图6  模型比较、机制消融和敏感性；负收益和无改善均保留', fontsize=12)
    save_figure(fig, target_dir, 'Q2_图6_消融与敏感性', copy_dirs=copy_dirs)


def plot_fig7_features_and_decoding(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """图7: 特征可视化与未知方向判别（主图及各组补图）"""
    pca = pd.read_csv(res_dir / 'Q2_训练参考PCA.csv')
    null = pd.read_csv(res_dir / 'Q2_置换分布.csv')
    metric = pd.read_csv(res_dir / 'Q2_表4_特征判别.csv')
    for key in NAMES:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
        points = pca[pca.dataset == key]
        for direction, color in ((-1, BLUE), (1, RED)):
            for role, marker in (('train', '.'), ('test', '^')):
                p = points[(points.cue == direction) & (points.role == role)]
                axes[0, 0].scatter(p.PC1, p.PC2, c=color, marker=marker,
                                   s=22 if role == 'train' else 45,
                                   alpha=.45 if role == 'train' else .95,
                                   label=f'{("左" if direction == -1 else "右")}·{role}')
        axes[0, 0].legend(ncol=2, fontsize=7)
        axes[0, 0].set(title=f'{key} 固定第5块留出 PCA', xlabel='PC1', ylabel='PC2')
        m = metric[metric.dataset == key].set_index('features').loc[list(FEATURE_NAMES)]
        axes[0, 1].errorbar(np.arange(len(m)), m.BA,
                            yerr=[np.maximum(m.BA - m.BA_low, 0), np.maximum(m.BA_high - m.BA, 0)],
                            fmt='o', capsize=3)
        axes[0, 1].axhline(.5, color=GREY, ls='--')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_xticks(range(len(m)), [FEATURE_NAMES[x] for x in m.index], rotation=35, ha='right')
        axes[0, 1].set(title=f'{key} 全部外层留出 BA', ylabel='平衡准确率')

        row = m.loc['mechanism']
        cm = np.array([[row.TN, row.FP], [row.FN, row.TP]])
        axes[1, 0].imshow(cm, cmap='Blues')
        axes[1, 0].set_xticks([0, 1], ['预测左', '预测右'])
        axes[1, 0].set_yticks([0, 1], ['实际左', '实际右'])
        for i in range(2):
            for j in range(2):
                axes[1, 0].text(j, i, str(int(cm[i, j])), ha='center', va='center', fontsize=15,
                                color='white' if cm[i, j] > (cm.max() + cm.min()) / 2 else 'black')
        axes[1, 0].set_title(f'{key} 机制特征混淆矩阵')

        n = null[(null.dataset == key) & (null.features == 'mechanism') & (null.kind == 'block_permutation')]
        axes[1, 1].hist(n.BA, bins=20, color=GREY, alpha=.65)
        axes[1, 1].axvline(row.BA, color=RED, lw=2)
        axes[1, 1].set(title=f'块内置换：maxT p={row.block_permutation_maxT_p:.3f}', xlabel='置换 BA', ylabel='次数')
        fig.suptitle(f'图7  {key} 特征可视化与未知方向判别（二维散点不作为显著性证据）', fontsize=12)
        suffix = '' if key == 'A1' else f'_补图_{key}'
        save_figure(fig, target_dir, f'Q2_图7_特征与判别{suffix}', copy_dirs=copy_dirs)


def plot_fig_pooled_summary(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """附图: 逐组与合并留出判别结果折线图"""
    metric = pd.read_csv(res_dir / 'Q2_表4_特征判别.csv')
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    sub = None
    for j, key in enumerate((*NAMES, 'pooled')):
        sub = metric[(metric.dataset == key) &
                     metric.features.isin(['amplitude', 'spatial', 'mechanism', 'past_only', 'post_given_past'])]
        ax.plot(range(len(sub)), sub.BA, marker='o', label=key)
    ax.set_xticks(range(len(sub)), [FEATURE_NAMES[s] for s in sub.features])
    ax.axhline(.5, color=GREY, ls='--')
    ax.set(ylabel='平衡准确率', title='逐组与合并留出判别结果')
    ax.legend(ncol=5)
    save_figure(fig, target_dir, 'Q2_附图_判别汇总', copy_dirs=copy_dirs)


# ==========================================
# 子模块深度图表 (decoder / mechanism / q1)
# ==========================================

def plot_decoder_v3_erp(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """子模型: decoder_v3_erp.png (VisCue 留出解码与置换检验)"""
    decoder_dir = res_dir / 'results' / 'decoder'
    if not (decoder_dir / 'metrics.csv').exists():
        return
    observed = pd.read_csv(decoder_dir / 'metrics.csv')
    null = pd.read_csv(decoder_dir / 'permutation_null.csv')
    colors = {'past_only': '#8A8A8A', 'post_erp': '#2F75A5', 'post_given_past': '#E08932'}
    models = ('past_only', 'post_erp', 'post_given_past')
    joined = observed
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')

    ax = axes[0]
    groups = list(NAMES) + ['pooled']
    for j, model in enumerate(models):
        part = joined[joined.model == model].set_index('dataset').loc[groups]
        xx = np.arange(len(groups)) + (j - 1) * .23
        yy = part.BA.to_numpy()
        ax.bar(xx, yy, width=.21, color=colors[model], label=model)
        ax.errorbar(xx, yy, yerr=np.vstack([yy - part.BA_low.to_numpy(), part.BA_high.to_numpy() - yy]),
                    fmt='none', color='black', capsize=2, lw=.8)
    ax.axhline(.5, ls='--', color='#333333', lw=.9)
    ax.set_xticks(np.arange(len(groups)), [f'{key}\n(n={int(joined[joined.dataset==key].n.iloc[0])})' for key in groups])
    ax.set_ylabel('Held-out balanced accuracy (95% block bootstrap CI)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title('Six DCT coefficients × three scalp modes')

    ax = axes[1]
    vals = null[(null.dataset == 'pooled') & (null.model == 'post_given_past')].BA.to_numpy()
    ax.hist(vals, bins=30, color='#C9D7DF', edgecolor='white')
    line = float(observed[(observed.dataset == 'pooled') &
                          (observed.model == 'post_given_past')].BA.iloc[0])
    ax.axvline(line, color=colors['post_given_past'], lw=2, label=f'observed = {line:.3f}')
    ax.set_xlabel('Permutation balanced accuracy')
    ax.set_ylabel('Number of permutations')
    ax.set_title('Past-adjusted post-cue ERP, within-block null')
    ax.legend(fontsize=9)
    fig.suptitle('VisCue 50–750 ms post cue vs strictly past-only −250–0 ms; exploratory')

    save_figure(fig, target_dir, 'decoder_v3_erp', copy_dirs=copy_dirs, dpi=180, save_pdf=False)


def plot_mechanism_comparison(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """子模型: heldout_comparison.png (各模型留出对比及重采样范围表)"""
    mech_dir = res_dir / 'results' / 'mechanism'
    if not (mech_dir / 'heldout_summary.csv').exists():
        return
    summary = pd.read_csv(mech_dir / 'heldout_summary.csv')
    pre_summary = pd.read_csv(mech_dir / 'prestim_summary.csv')
    intervals = pd.read_csv(mech_dir / 'fold_resampling_ranges.csv')

    counts = {'A1': 67, 'A2': 70, 'B1': 69, 'B2': 69}
    manifest_path = res_dir / 'Q2_运行清单.json'
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding='utf8'))
            if 'counts' in m:
                counts = m['counts']
        except Exception:
            pass

    keys = list(NAMES)
    fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(11, 6.8),
                                       gridspec_kw={'height_ratios': [3.5, 1.4]},
                                       layout='constrained')
    names = ['visual_neural_mass', 'ocular_step', 'slow_ramp']
    colors = ['#1767a2', '#bf6a22', '#687484']
    x = np.arange(len(keys))
    for j, name in enumerate(names):
        subset = summary.set_index(['dataset', 'model'])
        ax.bar(x + (j - 1) * .23, [subset.loc[(k, name), 'S_delta'] for k in keys],
               width=.21, label=name, color=colors[j])
    ax.plot(x, [pre_summary.set_index('dataset').loc[k, 'S_delta'] for k in keys],
            color='#8f3b8f', marker='d', linestyle='--', label='prestim only control')
    ax.axhline(0., color='black', linewidth=.9)
    ax.set_xticks(x, [f'{k}\nn={counts.get(k, "")}' for k in keys])
    ax.set_ylabel('Held-out direction contrast S_delta')
    ax.set_title('VisCue 50–750 ms, three equal-weight windows: held-out right-minus-left EEG')
    ax.legend(fontsize=8, ncol=2)

    table_ax.set_axis_off()
    lookup = intervals.set_index(['dataset', 'model'])
    columns = ['visual_neural_mass', 'ocular_step', 'slow_ramp', 'prestim_only_polynomial']
    cells = [[f"[{lookup.loc[(k, m), 'bootstrap_95_low']:.2f}, "
              f"{lookup.loc[(k, m), 'bootstrap_95_high']:.2f}]"
              for m in columns] for k in keys]
    tab = table_ax.table(cellText=cells, rowLabels=keys,
                         colLabels=['Neural', 'Ocular', 'Ramp', 'Pre only'],
                         loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.25)
    table_ax.set_title('95% descriptive fold-resampling ranges (5 blocks per group)',
                       fontsize=9, pad=4)

    save_figure(fig, target_dir, 'heldout_comparison', copy_dirs=copy_dirs, dpi=180, save_pdf=False)


def plot_q1_cortical_model(res_dir: Path, target_dir: Path, copy_dirs: list = None):
    """子模型: 第二问_Q1数据新模型.png (基于Q1数据的机制模型与协方差判别)"""
    q1_dir = res_dir / 'q1_model'
    if not (q1_dir / '机制波形.npz').exists():
        return
    waves = np.load(q1_dir / '机制波形.npz')
    metrics = pd.read_csv(q1_dir / '判别指标.csv')
    null_df = pd.read_csv(q1_dir / '置换分布.csv')
    mani = json.loads((q1_dir / '运行清单.json').read_text(encoding='utf8'))
    observed_ba = float(mani.get('observed_BA', 0.5453))
    null = null_df['BA'].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax = axes[0, 0]
    ax.axis("off")
    chain = ["三角边缘对比", "LGN中继", "V1方向选择", "皮层复发/情境更新", "Fz·F3·F4"]
    for index, label in enumerate(chain):
        x = .08 + index * .21
        ax.text(x, .55, label, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=.35", fc="#e8f1f5", ec="#4f7185"))
        if index < len(chain) - 1:
            ax.annotate("", xy=(x + .13, .55), xytext=(x + .08, .55),
                        arrowprops=dict(arrowstyle="->", color="#4f7185"))
    ax.text(.5, .22, "固定级联时间核 + 训练数据估计的头皮增益", ha="center", fontsize=10)
    ax.set_title("A  LGN→皮层→头皮的计算路径")

    ax = axes[0, 1]
    times_ms = waves['times_ms']
    chosen = (times_ms >= 50) & (times_ms < 750)
    for channel, color in zip(range(3), ("#2f75a5", "#cf6a32", "#3c8c64")):
        ax.plot(times_ms[chosen], waves["A1_observed"][channel, chosen],
                color=color, lw=1.4, label=f"{CHANNELS[channel]} 实测")
        ax.plot(times_ms[chosen], waves["A1_fitted"][channel, chosen],
                color=color, lw=1.1, ls="--", label=f"{CHANNELS[channel]} 模型")
    ax.axhline(0, color="#999", lw=.6)
    ax.set_title("B  A1右减左条件差：Q1 V7与机制拟合")
    ax.set_xlabel("提示后时间（ms）")
    ax.set_ylabel("原始电位单位")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    order = list(NAMES) + ["pooled"]
    width = .35
    x = np.arange(len(order))
    for offset, stage, label, color in [(-width / 2, "before", "Q1预处理（主分析）", "#2f75a5"),
                                         (width / 2, "v7", "Q1 V7（标签知情敏感性）", "#cf6a32")]:
        values = metrics[metrics.stage == stage].set_index("dataset").loc[order, "BA"]
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.axhline(.5, color="#555", ls="--", lw=1)
    ax.set_xticks(x, order)
    ax.set_ylim(.35, .72)
    ax.set_ylabel("留出平衡准确率")
    ax.set_title("C  连续时间块留出判别")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(null, bins=24, color="#9cb8c6", edgecolor="white")
    ax.axvline(observed_ba, color="#b14832", lw=2, label=f"观察值 {observed_ba:.3f}")
    ax.set_xlabel("块内置换平衡准确率")
    ax.set_ylabel("次数")
    ax.set_title("D  主分析固定流程的置换分布")
    ax.legend()
    fig.suptitle("第二问：基于Q1结果的级联机制与头皮空间协方差判别模型", fontsize=15)

    save_figure(fig, target_dir, '第二问_Q1数据新模型', copy_dirs=copy_dirs, dpi=180, save_pdf=False)
    waves.close()


# ==========================================
# 主生成执行流程
# ==========================================

def render_all_figures(result_dir=None, image_dir=None, sync_original=True):
    """
    执行所有图表的生成
    
    参数:
    -----
    result_dir: q2_result 结果根目录（默认根据脚本位置自动探测）
    image_dir:  q2image 目标保存目录（默认本脚本所在目录）
    sync_original: 是否同时覆盖更新 q2_result 对应原位置的图片文件
    """
    setup_matplotlib()

    script_path = Path(__file__).resolve()
    if image_dir is None:
        image_dir = script_path.parent
    else:
        image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    if result_dir is None:
        # 常见布局: q2_result/q2image/plot_all_figures.py -> parent 为 q2_result
        if (image_dir.parent / 'Q2_机制波形.npz').exists():
            result_dir = image_dir.parent
        elif (Path.cwd() / 'q2_result' / 'Q2_机制波形.npz').exists():
            result_dir = Path.cwd() / 'q2_result'
        else:
            result_dir = image_dir.parent
    else:
        result_dir = Path(result_dir)

    print(f"[Q2 绘图引擎] 数据输入源目录: {result_dir.resolve()}")
    print(f"[Q2 绘图引擎] q2image 输出目录: {image_dir.resolve()}")
    if sync_original:
        print(f"[Q2 绘图引擎] 同时同步更新 q2_result 原位置图片: 是")

    # 1. 主报告正文图与附图 (生成在 q2_result 根目录并同步至 q2image)
    target_main = result_dir if sync_original else image_dir
    backup_main = [image_dir] if sync_original else []

    print("-> 正在绘制图1 (机制与形状编码)...")
    plot_fig1_mechanism_and_shape(result_dir, target_main, copy_dirs=backup_main)

    print("-> 正在绘制图2 (皮层内部响应及各组补图)...")
    plot_fig2_cortical_responses(result_dir, target_main, copy_dirs=backup_main)

    print("-> 正在绘制附图 (LGN 中继动态)...")
    plot_fig_lgn(result_dir, target_main, copy_dirs=backup_main)

    print("-> 正在绘制图3 (ERP 拟合与留出及各组补图)...")
    plot_fig3_erp_and_heldout(result_dir, target_main, copy_dirs=backup_main)

    print("-> 正在绘制附图 (Q1 衔接对比)...")
    plot_fig_q1_transfer(result_dir, target_main, copy_dirs=backup_main)

    print("-> 正在绘制图4 (四组左右差异波)...")
    plot_fig4_direction_diff(result_dir, target_main, copy_dirs=backup_main)

    print("-> 正在绘制图5 (空间侧化与 LI 指数)...")
    plot_fig5_spatial_lateralization(result_dir, target_main, copy_dirs=backup_main)

    print("-> 正在绘制图6 (消融与敏感性分析)...")
    plot_fig6_ablation_and_sensitivity(result_dir, target_main, copy_dirs=backup_main)

    print("-> 正在绘制图7 (特征可视化与判别及各组补图)...")
    plot_fig7_features_and_decoding(result_dir, target_main, copy_dirs=backup_main)

    print("-> 正在绘制附图 (判别汇总)...")
    plot_fig_pooled_summary(result_dir, target_main, copy_dirs=backup_main)

    # 2. 子模块分析图 (分别生成在对应子目录并同步至 q2image)
    print("-> 正在绘制 decoder_v3_erp 图...")
    target_dec = (result_dir / 'results' / 'decoder') if sync_original else image_dir
    plot_decoder_v3_erp(result_dir, target_dec, copy_dirs=backup_main)

    print("-> 正在绘制 heldout_comparison 图...")
    target_mech = (result_dir / 'results' / 'mechanism') if sync_original else image_dir
    plot_mechanism_comparison(result_dir, target_mech, copy_dirs=backup_main)

    print("-> 正在绘制 第二问_Q1数据新模型 图...")
    target_q1 = (result_dir / 'q1_model') if sync_original else image_dir
    plot_q1_cortical_model(result_dir, target_q1, copy_dirs=backup_main)

    print("\n[成功] Q2 所有绘图已全部生成完成！")
    print(f"所有生成的图片均已完备存放于: {image_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Q2 独立绘图脚本")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="q2_result 结果目录路径，默认为脚本所在上一级目录")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="图片保存输出目录，默认为脚本所在目录 q2image")
    parser.add_argument("--no-sync", action="store_true",
                        help="不向 q2_result 原位置同步图片，仅保存于 image-dir")
    args = parser.parse_args()

    render_all_figures(
        result_dir=args.result_dir,
        image_dir=args.image_dir,
        sync_original=(not args.no_sync)
    )


if __name__ == "__main__":
    main()
