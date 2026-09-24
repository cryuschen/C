#!/usr/bin/env python3
"""生成第一问 V7 顶刊标准三栏伪影分解图 (Three-panel Artifact Decomposition Plots)

展示内容：
1. 第一栏 (顶部)：原始脑电记录 (基线对齐 Raw)
2. 第二栏 (中间)：V7 校正后纯净脑电信号 (V7 Clean)
3. 第三栏 (底部)：算法提取并剥离的伪影成分 (Removed Artifact = Raw - V7)

覆盖四组数据 (受试者 A/B · 项目一/二)，各包含 75% 较高伪影分位试次与 50% 中位试次。
输出路径：output/三栏伪影分解对比图/
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# 设置安全的 Matplotlib 缓存路径
os.environ.setdefault('MPLCONFIGDIR', '/tmp/eeg-v7-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 字体与排版样式配置
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Noto Sans SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'eeg_v7_results'
OUTPUT_DIR = ROOT / 'output/三栏伪影分解对比图'
CHANNELS = ['Fz', 'F3', 'F4']
N_PRE = 64  # 刺激前 250 ms 基线点数 (256 Hz 采样率下 64 点)

# 高级学术配色方案
COLORS = {
    'raw': '#455A64',       # 灰蓝色：原始未加工记录
    'clean': '#00796B',     # 青绿色：V7 校正后纯净脑电
    'artifact': '#C62828',  # 铁红色：提取出的伪影与噪声
    'p300_span': '#FFF3CD', # 淡黄色：250-500 ms 视觉响应分析窗
}


def plot_decomposition_for_dataset(dataset_folder_name, quantile, quantile_label, filename):
    """为指定数据集生成三通道、左右方向的三栏分解图"""
    data_path = RESULTS_DIR / dataset_folder_name / '可复核波形.npz'
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")
        
    npz_data = np.load(data_path)
    raw = npz_data['raw']
    v7 = npz_data['v7']
    cues = npz_data['cues']
    trial_ids = npz_data['trial_ids']
    times = npz_data['times_ms']
    
    # 读取预先指定的固定示例试次编号
    fixed_csv = RESULTS_DIR / dataset_folder_name / '固定示例试次编号.csv'
    fixed_df = pd.read_csv(fixed_csv)
    
    # 查找该分位数下的左侧(-1)与右侧(+1)试次
    left_trial_id = int(fixed_df[(fixed_df['quantile'] == quantile) & (fixed_df['cue'] == -1)]['trial_id'].iloc[0])
    right_trial_id = int(fixed_df[(fixed_df['quantile'] == quantile) & (fixed_df['cue'] == 1)]['trial_id'].iloc[0])
    
    ix_left = int(np.where(trial_ids == left_trial_id)[0][0])
    ix_right = int(np.where(trial_ids == right_trial_id)[0][0])
    
    # 创建 6 行 × 3 列的画布
    # 上 3 行：左侧视觉刺激 (Fz, F3, F4)
    # 下 3 行：右侧视觉刺激 (Fz, F3, F4)
    fig, axes = plt.subplots(6, 3, figsize=(15, 16.5), sharex=True)
    
    blocks = [
        ('左侧视觉提示 (VisCue = -1, 左视野)', ix_left, left_trial_id, 0),
        ('右侧视觉提示 (VisCue = +1, 右视野)', ix_right, right_trial_id, 3)
    ]
    
    for cue_title, ix, tid, row_offset in blocks:
        # 基线对齐的原始信号
        raw_bc = raw[ix] - np.median(raw[ix, :, :N_PRE], axis=1, keepdims=True)
        clean = v7[ix]
        artifact = raw_bc - clean
        
        # 逐通道绘制
        for ch, ch_name in enumerate(CHANNELS):
            # 确定当前通道在处理前后的纵坐标范围，保持对比客观
            y_max = max(np.max(np.abs(raw_bc[ch])), np.max(np.abs(clean[ch]))) * 1.15
            
            # --- Row 1: 原始记录 ---
            ax_raw = axes[row_offset, ch]
            ax_raw.plot(times, raw_bc[ch], color=COLORS['raw'], lw=1.2, label='原始通道 (基线对齐)')
            ax_raw.axvline(0, color='black', lw=0.8, ls='--')
            ax_raw.axvspan(250, 500, color=COLORS['p300_span'], alpha=0.6, label='P300 分析窗 (250–500 ms)')
            ax_raw.set_title(f'{cue_title} · {ch_name} · 试次 #{tid} [原始]', fontsize=10.5, fontweight='bold', pad=4)
            ax_raw.grid(True, alpha=0.25, ls=':')
            if ch == 0:
                ax_raw.set_ylabel('原始记录\n(电位单位)', fontsize=9.5, fontweight='bold')
            if row_offset == 0 and ch == 0:
                ax_raw.legend(loc='upper right', fontsize=8, framealpha=0.9)
                
            # --- Row 2: V7 去噪后 ---
            ax_clean = axes[row_offset + 1, ch]
            ax_clean.plot(times, clean[ch], color=COLORS['clean'], lw=1.5, label='V7 校正后脑电')
            ax_clean.axvline(0, color='black', lw=0.8, ls='--')
            ax_clean.axvspan(250, 500, color=COLORS['p300_span'], alpha=0.6)
            ax_clean.set_title(f'{ch_name} · V7 去噪后信号', fontsize=10.5, fontweight='bold', pad=4)
            ax_clean.grid(True, alpha=0.25, ls=':')
            if ch == 0:
                ax_clean.set_ylabel('V7 纯净信号\n(电位单位)', fontsize=9.5, fontweight='bold')
            if row_offset == 0 and ch == 0:
                ax_clean.legend(loc='upper right', fontsize=8, framealpha=0.9)
                
            # --- Row 3: 剥离的纯伪影分量 ---
            ax_art = axes[row_offset + 2, ch]
            ax_art.plot(times, artifact[ch], color=COLORS['artifact'], lw=1.1, label='提取伪影 (Raw - V7)')
            ax_art.axvline(0, color='black', lw=0.8, ls='--')
            ax_art.axvspan(250, 500, color=COLORS['p300_span'], alpha=0.6)
            ax_art.set_title(f'{ch_name} · 剥离伪影成分 (Raw - V7)', fontsize=10.5, fontweight='bold', pad=4)
            ax_art.grid(True, alpha=0.25, ls=':')
            if ch == 0:
                ax_art.set_ylabel('滤除伪影分量\n(电位单位)', fontsize=9.5, fontweight='bold')
            if row_offset == 0 and ch == 0:
                ax_art.legend(loc='upper right', fontsize=8, framealpha=0.9)
            if row_offset == 3:
                ax_art.set_xlabel('相对刺激提示时间 (ms)', fontsize=10, fontweight='bold')

    # 主标题与副标题
    dataset_display = dataset_folder_name.replace('_', ' · ')
    fig.suptitle(f'{dataset_display} · {quantile_label}三栏伪影分解图\n[顶刊标准：原始信号 = V7纯净脑电 + 提取伪影成分]',
                 fontsize=13.5, fontweight='bold', y=0.995)
    
    fig.tight_layout(rect=[0, 0.01, 1, 0.985])
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / dataset_folder_name / filename
    fig.savefig(out_file, dpi=200)
    print(f"成功生成并写入 eeg_v7_results: {out_file.relative_to(ROOT)}")
    
    # 同时在 output/三栏伪影分解对比图 保存带完整命名的副本以方便集中查阅
    backup_dir = ROOT / 'output/三栏伪影分解对比图'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / f"{dataset_folder_name}_{filename}"
    fig.savefig(backup_file, dpi=200)
    plt.close(fig)


def main():
    datasets = [
        '受试者A_项目一',
        '受试者A_项目二',
        '受试者B_项目一',
        '受试者B_项目二'
    ]
    
    print("开始生成第一问 V7 顶刊标准三栏伪影分解图...")
    for ds in datasets:
        # 1. 典型较高伪影试次 (75% 分位)
        plot_decomposition_for_dataset(
            ds,
            quantile=0.75,
            quantile_label='典型较高伪影试次 (75% 分位)',
            filename='较高伪影试次_三栏分解图.png'
        )
        # 2. 中位伪影试次 (50% 分位)
        plot_decomposition_for_dataset(
            ds,
            quantile=0.50,
            quantile_label='中位伪影试次 (50% 分位)',
            filename='中位伪影试次_三栏分解图.png'
        )
        
    print(f"\n全部图表生成完毕！保存在目录：{OUTPUT_DIR.resolve()}")


if __name__ == '__main__':
    main()
