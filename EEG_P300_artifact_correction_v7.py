#!/usr/bin/env python3
"""第一问 V7 可复核实验：按训练折所选候选强度调节校正，同时保留方向响应。

运行：python EEG_P300_artifact_correction_v7.py --output eeg_v7_results
V7 属四组数据上的开发实验；任何试次区间都不是跨受试者泛化验证。
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/eeg-v7-mpl')
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline
import scipy.io as sio
from scipy.stats import median_abs_deviation
from scipy.signal import find_peaks, butter, sosfiltfilt, iirnotch, filtfilt, savgol_filter, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT=Path(__file__).resolve().parent

# 与原 V3 相同的时间轴、校正参数和绘图设置。
FS_EXPECTED = 256
PRE_SEC = 0.25      # 刺激前 250 ms 基线
POST_SEC = 0.80     # 刺激后 800 ms 分析窗

N_PRE = round(PRE_SEC * FS_EXPECTED)
N_POST = round(POST_SEC * FS_EXPECTED)
TIMES = np.arange(-N_PRE, N_POST) / FS_EXPECTED
TIMES_MS = TIMES * 1000.0

P300_MASK = (TIMES >= 0.25) & (TIMES <= 0.50)
CHANNEL_NAMES = ["Fz", "F3", "F4"]

# V3 核心去噪参数
V3_PARAMS = {
    "win_blink": 41,        # 垂直眼电平滑窗长 (约 160 ms)
    "win_saccade": 31,      # 水平扫视平滑窗长 (约 120 ms)
    "gate_z_v": 2.2,        # 垂直软门控阈值
    "gate_z_h": 2.2,        # 水平软门控阈值
    "gate_width": 1.2,      # 软门控过渡带宽度
    "reg_lambda": 0.05,     # 岭回归正则化系数，防止过减
    "max_beta_v": 1.8,      # 垂直通道最大增益
    "max_beta_h": 1.5,      # 水平通道最大增益
}

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# V7 原有时间轴为毫秒；V4 内部别名保持原值。
TIMES=TIMES_MS
WINDOW=P300_MASK
CHANNELS=CHANNEL_NAMES
T=TIMES_MS
P=P300_MASK
N=N_PRE

# 原 V4 训练内候选集与诊断候选集。
CANDIDATES = {
    '不校正': (0., 2.2, 1.5, False),
    '中位数基线': (1., 2.2, 1.5, False),
    '保守双分量': (.5, 3.5, .5, False),
    '仅垂直分量': (.5, 3.5, 0., False),
    '时域保护': (.75, 3.5, .5, True),
}

DIAGNOSTIC_CANDIDATES = {
    **CANDIDATES,
    '诊断_仅提高门控': (1., 3.5, 1.5, False),
    '诊断_仅减弱校正': (.5, 2.2, 1.5, False),
    '诊断_仅限制水平上限': (1., 2.2, .5, False),
    '诊断_关闭水平分量': (1., 2.2, 0., False),
    '诊断_关闭垂直分量': (1., 2.2, 1.5, False),
}

STAGES=('预处理','V3','V4','V6','V7')
PLOT_STAGES=('预处理','V6','V7')
COLORS={'原始':'#a8adb4','预处理':'#596e83','V3':'#c87927','V4':'#6b61a8','V6':'#8b89b4','V7':'#007f89','代理参考':'#30343a'}
LINES={'原始':':','预处理':'--','V3':'-.','V4':':','V6':':','V7':'-','代理参考':':'}



# 从 V3/V4 精确整合的运行时实现；阶段名称继续用于同口径比较。
def load_mat_file(filepath):
    """读取赛题 .mat 文件"""
    mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    fs = int(mat["SampleRate"])
    labels = list(mat["DataLabel"])
    data = mat["data"]
    return fs, labels, data

def extract_viscue_events(viscue):
    """根据 VisCue 通道提取事件上升沿与方向标签 (-1: 左三角, +1: 右三角)"""
    nonzero = (viscue != 0)
    onset_mask = nonzero & np.r_[True, ~nonzero[:-1]]
    onsets = np.where(onset_mask)[0]
    cue_values = viscue[onsets].astype(int)
    return onsets, cue_values

def segment_epochs(signal, onsets, n_pre=N_PRE, n_post=N_POST):
    """截取 [-N_PRE, N_POST] 试次片段，shape: (trials, channels, time)"""
    epochs = []
    n_samples = signal.shape[-1]
    for onset in onsets:
        start = onset - n_pre
        end = onset + n_post
        if start < 0 or end > n_samples:
            continue
        epochs.append(signal[..., start:end])
    return np.stack(epochs, axis=0)

def preprocess_continuous_eeg(data_3ch, fs=FS_EXPECTED):
    """
    对连续 3 通道 EEG 进行预处理：
    1. 60 Hz 陷波滤波（抑制实测强工频干扰）
    2. 0.1–30 Hz 四阶 Butterworth 零相位带通
    """
    # 60 Hz 陷波
    b_notch, a_notch = iirnotch(w0=60.0, Q=30.0, fs=fs)
    notched = filtfilt(b_notch, a_notch, data_3ch, axis=-1)

    # 0.1-30 Hz 带通 (4阶 Butterworth, 前后双向零相位)
    sos = butter(4, [0.1, 30.0], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, notched, axis=-1)
    return filtered

def robust_baseline_correct(epochs, n_pre=N_PRE):
    """
    使用刺激前 [-250, 0] ms 的稳健中位数/截断均值作为基线，扣除慢漂移偏置
    """
    baseline = np.median(epochs[:, :, :n_pre], axis=2, keepdims=True)
    return epochs - baseline

def robust_zscore(x):
    """基于 Median 与 MAD 的稳健 Z-score 标准化"""
    med = np.median(x)
    mad = median_abs_deviation(x, scale="normal")
    if mad < 1e-12:
        mad = np.std(x) + 1e-12
    return (x - med) / mad

def detect_bad_trials(raw_epochs, filtered_epochs):
    """
    识别不可恢复严重坏试次：
    1. 原始采样点发生 ±1000 硬件饱和截幅连续超过 15 个点
    2. 峰峰值超过 1800 原始幅值单位
    3. 相邻点发生物理不可能的跳跃
    """
    n_trials = len(raw_epochs)
    peak_to_peak = np.ptp(filtered_epochs, axis=2).max(axis=1)

    # 硬件饱和检测 (|raw| >= 999.0)
    sat_points = np.sum(np.abs(raw_epochs) >= 999.0, axis=(1, 2))

    # 最大跳跃与最大绝对值
    max_jump = np.max(np.abs(np.diff(filtered_epochs, axis=2)), axis=(1, 2))
    max_abs = np.max(np.abs(filtered_epochs), axis=(1, 2))

    # 综合伪影得分
    z_ptp = robust_zscore(np.log1p(peak_to_peak))
    z_jump = robust_zscore(np.log1p(max_jump))
    z_abs = robust_zscore(np.log1p(max_abs))
    artifact_score = np.maximum(z_ptp, 0) + np.maximum(z_jump, 0) + np.maximum(z_abs, 0)

    irrecoverable = (sat_points >= 15) | (peak_to_peak > 1800) | (max_jump > 600)
    return irrecoverable, artifact_score, peak_to_peak

def build_clean_reference_model(epochs, cues, irrecoverable, artifact_scores, ref_ratio=0.50):
    """
    在训练试次中为左(-1)与右(+1)刺激分别构建低伪影 ERP 模板及动态残差尺度
    """
    templates = {}
    scales_v = {}
    scales_h = {}
    ref_mask = np.zeros(len(epochs), dtype=bool)

    for cond in (-1, 1):
        valid_idx = np.where((cues == cond) & (~irrecoverable))[0]
        if len(valid_idx) == 0:
            continue
        # 按伪影得分排序，选取最干净的前 ref_ratio 试次
        sorted_idx = valid_idx[np.argsort(artifact_scores[valid_idx])]
        k = max(8, int(np.ceil(ref_ratio * len(sorted_idx))))
        chosen = sorted_idx[:k]
        ref_mask[chosen] = True

        # 稳健中位数模板 (shape: 3, time)
        tpl = np.median(epochs[chosen], axis=0)
        templates[cond] = tpl

        # 计算纯净试次的残差与伪影参考
        res = epochs[chosen] - tpl[None, :, :]
        # 垂直分量: 三通道中位数
        res_v = np.median(res, axis=1)
        # 水平分量: (F4 - F3) 偶极差分
        res_h = 0.5 * (res[:, 2, :] - res[:, 1, :])

        scale_v = median_abs_deviation(res_v, axis=0, scale="normal")
        scale_h = median_abs_deviation(res_h, axis=0, scale="normal")

        scale_v = np.maximum(scale_v, 0.5 * np.median(scale_v) + 1e-6)
        scale_h = np.maximum(scale_h, 0.5 * np.median(scale_h) + 1e-6)

        scales_v[cond] = scale_v
        scales_h[cond] = scale_h

    return templates, scales_v, scales_h, ref_mask

def correct_single_epoch_v3(epoch, template, scale_v, scale_h, params=V3_PARAMS):
    """
    V3 核心自适应去噪：
    1. 从单试次中扣除对应的先验 ERP 模板，得到纯残差 residual
    2. 分离垂直眼电分量 V(t) 与水平扫视分量 H(t)
    3. 分别进行多尺度平滑与动态自适应软门控
    4. 采用带约束岭回归自适应估计各通道的消除系数
    5. 重构无伪影信号并恢复基线
    """
    residual = epoch - template  # shape: (3, time)

    # 1. 垂直同相分量参考 (眨眼/头动)
    ref_v = np.median(residual, axis=0)
    ref_v_smooth = savgol_filter(ref_v, window_length=params["win_blink"], polyorder=3)

    # 2. 水平反相偶极分量参考 (左右扫视 eye movement)
    ref_h = 0.5 * (residual[2] - residual[1])
    ref_h_smooth = savgol_filter(ref_h, window_length=params["win_saccade"], polyorder=3)

    # 3. 动态软门控
    z_v = np.abs(ref_v_smooth) / (scale_v + 1e-6)
    gate_v = np.clip((z_v - params["gate_z_v"]) / params["gate_width"], 0, 1)
    gate_v = savgol_filter(gate_v, window_length=15, polyorder=2)
    gate_v = np.clip(gate_v, 0, 1)
    art_v = ref_v_smooth * gate_v

    z_h = np.abs(ref_h_smooth) / (scale_h + 1e-6)
    gate_h = np.clip((z_h - params["gate_z_h"]) / params["gate_width"], 0, 1)
    gate_h = savgol_filter(gate_h, window_length=15, polyorder=2)
    gate_h = np.clip(gate_h, 0, 1)
    art_h = ref_h_smooth * gate_h

    corrected = epoch.copy()

    # 4. 通道自适应岭回归消除
    X = np.column_stack([art_v, art_h])  # shape: (T, 2)
    lambda_eye = params["reg_lambda"] * np.eye(2)
    XtX = X.T @ X + lambda_eye

    for ch in range(3):
        y = residual[ch]
        # 解岭回归系数: beta = (X^T X + lambda I)^(-1) X^T y
        beta = np.linalg.solve(XtX, X.T @ y)

        # 垂直分量在额叶均为正偏转
        beta_v = float(np.clip(beta[0], 0, params["max_beta_v"]))
        # 水平分量允许正负（F3 为负，F4 为正）
        beta_h = float(np.clip(beta[1], -params["max_beta_h"], params["max_beta_h"]))

        # 若是 Fz（中线），水平扫视分量理论为 0，抑制 beta_h
        if ch == 0:
            beta_h = float(np.clip(beta_h, -0.3, 0.3))

        corrected[ch] = epoch[ch] - (beta_v * art_v + beta_h * art_h)

    # 5. 重新校正刺激前基线
    corrected -= np.mean(corrected[:, :N_PRE], axis=1, keepdims=True)
    return corrected

def compute_bootstrap_ci(epochs_matrix, n_boot=500, ci=95):
    """
    对试次集合进行 Bootstrap 重采样，计算逐时间点的 95% 置信区间
    shape: (trials, time) -> (time,), (time,)
    """
    n_trials = len(epochs_matrix)
    if n_trials < 5:
        mean_sig = np.mean(epochs_matrix, axis=0)
        return mean_sig, mean_sig

    rng = np.random.default_rng(42)
    boot_means = np.zeros((n_boot, epochs_matrix.shape[1]))
    for b in range(n_boot):
        sample_idx = rng.choice(n_trials, size=n_trials, replace=True)
        boot_means[b] = np.mean(epochs_matrix[sample_idx], axis=0)

    alpha = (100 - ci) / 2.0
    lower = np.percentile(boot_means, alpha, axis=0)
    upper = np.percentile(boot_means, 100 - alpha, axis=0)
    return lower, upper

def evaluate_spatial_metrics(ep_before, ep_after, cues, dataset_name):
    """左右条件差分与 F3−F4 不对称性；保留率以去噪前为分母。"""
    rows = []
    erps = {}
    for stage, epochs in (('before', ep_before), ('after', ep_after)):
        erps[stage] = {cond: np.mean(epochs[cues == cond], axis=0) for cond in (-1, 1)}
    for ch, name in enumerate(CHANNEL_NAMES):
        before = erps['before'][-1][ch, P300_MASK] - erps['before'][1][ch, P300_MASK]
        after = erps['after'][-1][ch, P300_MASK] - erps['after'][1][ch, P300_MASK]
        rows.append({'dataset': dataset_name, 'quantity': 'left_minus_right_ERP', 'channel_or_cue': name,
                     'RMS_before': float(np.sqrt(np.mean(before ** 2))),
                     'RMS_after': float(np.sqrt(np.mean(after ** 2))),
                     'retention_ratio': float(np.linalg.norm(after) / (np.linalg.norm(before) + 1e-12)),
                     'waveform_correlation': safe_corr(before, after)})
    for cond in (-1, 1):
        before = erps['before'][cond][1, P300_MASK] - erps['before'][cond][2, P300_MASK]
        after = erps['after'][cond][1, P300_MASK] - erps['after'][cond][2, P300_MASK]
        rows.append({'dataset': dataset_name, 'quantity': 'F3_minus_F4',
                     'channel_or_cue': 'left' if cond == -1 else 'right',
                     'RMS_before': float(np.sqrt(np.mean(before ** 2))),
                     'RMS_after': float(np.sqrt(np.mean(after ** 2))),
                     'retention_ratio': float(np.linalg.norm(after) / (np.linalg.norm(before) + 1e-12)),
                     'waveform_correlation': safe_corr(before, after),
                     'P300_positive_mean_difference_before': positive_p300_features(erps['before'][cond][1])[0]
                     - positive_p300_features(erps['before'][cond][2])[0],
                     'P300_positive_mean_difference_after': positive_p300_features(erps['after'][cond][1])[0]
                     - positive_p300_features(erps['after'][cond][2])[0]})
    return rows

def positive_p300_features(erp):
    """250–500 ms 正向均值、正面积、正峰值和峰潜伏期。幅值沿用数据原始单位。"""
    segment = np.asarray(erp)[P300_MASK]
    positive = np.maximum(segment, 0.0)
    peak_index = int(np.argmax(segment))
    peak = float(segment[peak_index])
    # 窗内无正峰时，不给负波指定 P300 峰值或潜伏期。
    positive_peak = peak if peak > 0 else np.nan
    peak_latency = float(TIMES_MS[P300_MASK][peak_index]) if peak > 0 else np.nan
    return (
        float(np.mean(positive)),
        float(np.trapezoid(positive, TIMES_MS[P300_MASK])),
        positive_peak,
        peak_latency,
    )

def safe_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def shape_distance(a, b):
    """P300 窗内 z 标准化后的平均绝对距离。"""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.mean(np.abs((a - np.mean(a)) / np.std(a) -
                                (b - np.mean(b)) / np.std(b))))

def snr_proxy_db(trials):
    """ERP 功率 / 试次相对 ERP 的残差功率；残差包含真实试次变异。"""
    post_mask = (TIMES_MS >= 0) & (TIMES_MS <= 500)
    x = trials[:, post_mask]
    erp = np.mean(x, axis=0)
    signal_power = np.mean(erp ** 2)
    residual_power = np.mean((x - erp[None, :]) ** 2)
    return float(10 * np.log10((signal_power + 1e-12) / (residual_power + 1e-12)))

def edge_band_power_ratio(trials):
    """1–3 Hz 和 20–30 Hz 占 1–30 Hz 功率比；仅为频谱描述量。"""
    freq, psd = welch(trials, fs=FS_EXPECTED, axis=-1,
                      nperseg=trials.shape[-1], detrend='linear')
    mean_psd = np.mean(psd, axis=0)
    def power(lo, hi):
        mask = (freq >= lo) & (freq <= hi)
        return float(np.trapezoid(mean_psd[mask], freq[mask]))
    total = power(1, 30)
    return (power(1, 3) + power(20, 30)) / (total + 1e-12)

def evaluate_metrics(ep_before, ep_after, cues, ref_templates):
    """按刺激方向及通道计算真实数据的描述量与低伪影模板代理误差。"""
    records = []
    for cond in (-1, 1):
        idx = np.where(cues == cond)[0]
        if len(idx) == 0:
            continue
        avg_before = np.mean(ep_before[idx], axis=0)
        avg_after = np.mean(ep_after[idx], axis=0)
        ref = ref_templates[cond]

        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            trials_before = ep_before[idx, ch_idx, :]
            trials_after = ep_after[idx, ch_idx, :]
            ref_p3 = ref[ch_idx, P300_MASK]
            bef_p3 = avg_before[ch_idx, P300_MASK]
            aft_p3 = avg_after[ch_idx, P300_MASK]

            mae_before = float(np.mean(np.abs(bef_p3 - ref_p3)))
            mae_after = float(np.mean(np.abs(aft_p3 - ref_p3)))

            corr_before = safe_corr(bef_p3, ref_p3)
            corr_after = safe_corr(aft_p3, ref_p3)

            rmse_before = float(np.sqrt(np.mean((bef_p3 - ref_p3) ** 2)))
            rmse_after = float(np.sqrt(np.mean((aft_p3 - ref_p3) ** 2)))

            pos_ref, auc_ref, _, _ = positive_p300_features(ref[ch_idx])
            pos_bef, auc_bef, amp_bef, lat_bef = positive_p300_features(avg_before[ch_idx])
            pos_aft, auc_aft, amp_aft, lat_aft = positive_p300_features(avg_after[ch_idx])
            _, _, amp_ref, lat_ref = positive_p300_features(ref[ch_idx])

            records.append({
                "cue": cond,
                "channel": ch_name,
                "MAE_before": mae_before,
                "MAE_after": mae_after,
                "RMSE_before": rmse_before,
                "RMSE_after": rmse_after,
                "corr_before": corr_before,
                "corr_after": corr_after,
                "shape_distance_before": shape_distance(bef_p3, ref_p3),
                "shape_distance_after": shape_distance(aft_p3, ref_p3),
                "baseline_RMS_before": float(np.mean(np.sqrt(np.mean(trials_before[:, :N_PRE] ** 2, axis=1)))),
                "baseline_RMS_after": float(np.mean(np.sqrt(np.mean(trials_after[:, :N_PRE] ** 2, axis=1)))),
                "trial_PTP_median_before": float(np.median(np.ptp(trials_before, axis=1))),
                "trial_PTP_median_after": float(np.median(np.ptp(trials_after, axis=1))),
                "SNR_proxy_dB_before": snr_proxy_db(trials_before),
                "SNR_proxy_dB_after": snr_proxy_db(trials_after),
                "edge_band_ratio_before": edge_band_power_ratio(trials_before),
                "edge_band_ratio_after": edge_band_power_ratio(trials_after),
                "amp_error_before": abs(amp_bef - amp_ref),
                "amp_error_after": abs(amp_aft - amp_ref),
                "latency_error_before_ms": abs(lat_bef - lat_ref),
                "latency_error_after_ms": abs(lat_aft - lat_ref),
                "P300_positive_mean_before": pos_bef,
                "P300_positive_mean_after": pos_aft,
                "P300_positive_mean_ref": pos_ref,
                "P300_positive_mean_error_before": abs(pos_bef - pos_ref),
                "P300_positive_mean_error_after": abs(pos_aft - pos_ref),
                "P300_AUC_before_unit_ms": auc_bef,
                "P300_AUC_after_unit_ms": auc_aft,
                "P300_AUC_ref_unit_ms": auc_ref,
                "P300_AUC_error_before_unit_ms": abs(auc_bef - auc_ref),
                "P300_AUC_error_after_unit_ms": abs(auc_aft - auc_ref),
                "P300_amp_before": amp_bef,
                "P300_amp_after": amp_aft,
                "P300_lat_before_ms": lat_bef,
                "P300_lat_after_ms": lat_aft
            })
    return pd.DataFrame(records)

def correct(epoch, tpl, sv, sh, candidate):
    strength, gate, max_h, protect = DIAGNOSTIC_CANDIDATES[candidate]
    if strength == 0:
        return epoch.copy()
    params = dict(V3_PARAMS, gate_z_v=gate, gate_z_h=gate, max_beta_h=max_h)
    if candidate == '诊断_关闭垂直分量':
        params['max_beta_v'] = 0.
    # 撤销 V3 最后的均值重定位，再统一使用中位数基线。
    full = correct_single_epoch_v3(epoch, tpl, sv, sh, params)
    full -= np.median(full[:, :N], axis=1, keepdims=True)
    delta = epoch - full
    if protect:
        # 仅保护训练模板的时间形状；不将测试条件 ERP 人为恢复到原幅度。
        taper = np.clip((T - 0) / 80, 0, 1) * np.clip((750 - T) / 120, 0, 1)
        basis = savgol_filter(tpl, 25, 3, axis=-1) * taper
        u, s, _ = np.linalg.svd(basis.T, full_matrices=False)
        keep = s > max(s[0] * .10, 1e-9)
        q = u[:, keep]
        delta -= (delta @ q) @ q.T
    result = epoch - strength * delta
    return result - np.median(result[:, :N], axis=1, keepdims=True)

def inject(background, cues, seed, level):
    """已知加性污染，正负眨眼、扫视和短暂运动；保留真实背景原波形为恢复目标。"""
    rng = np.random.default_rng(seed)
    t = T / 1000
    out = background.copy()
    scale = max(float(np.median(np.std(background, axis=-1))), 1.)
    for i in range(len(out)):
        center = rng.uniform(-.12, .65)
        width = rng.uniform(.025, .13)
        amp = level * scale * rng.uniform(2, 6)
        blink = amp * rng.choice([-1, 1]) * np.exp(-.5*((t-center)/width)**2)
        saccade = amp * .7 * cues[i] * (np.tanh((t-center)/.035)-np.tanh((t-center-.18)/.045))/2
        motion = amp*.3*np.sin(2*np.pi*rng.uniform(7,15)*t)*np.exp(-.5*((t-center)/.035)**2)
        artifact = np.array([1.1,1.,.9])[:,None]*blink + np.array([.05,-1,1])[:,None]*saccade + motion
        # 污染与基线操作作为已知变换；level=0 时精确返回背景。
        out[i] += artifact
    return robust_baseline_correct(out)

def apply_model(x, cues, model, candidate):
    tpl, sv, sh, _ = model
    if candidate == 'V3':
        return np.stack([correct_single_epoch_v3(e, tpl[c], sv[c], sh[c]) for e,c in zip(x,cues)])
    return np.stack([correct(e,tpl[c],sv[c],sh[c],candidate) for e,c in zip(x,cues)])

def known_errors(target, recovered, cues):
    scale = max(float(np.sqrt(np.mean(target**2))), 1e-9)
    rmse = float(np.sqrt(np.mean((target-recovered)**2)))
    truth_diff = target[cues==-1].mean(0)-target[cues==1].mean(0)
    rec_diff = recovered[cues==-1].mean(0)-recovered[cues==1].mean(0)
    contrast = float(np.sqrt(np.mean((truth_diff[:,P]-rec_diff[:,P])**2)))
    pos = []
    lat = []
    for c in (-1,1):
        for ch in range(3):
            a=positive_p300_features(target[cues==c,ch].mean(0))
            b=positive_p300_features(recovered[cues==c,ch].mean(0))
            pos.append(abs(a[0]-b[0]))
            if np.isfinite(a[3]) and np.isfinite(b[3]): lat.append(abs(a[3]-b[3]))
    return dict(RMSE=rmse, normalized_RMSE=rmse/scale, contrast_RMSE=contrast,
                positive_mean_error=np.mean(pos), latency_error_ms=np.mean(lat) if lat else np.nan,
                latency_pairs=len(lat), objective=(rmse+.5*contrast+np.mean(pos))/scale)

def select_candidate(x, cues, seed):
    train, val = train_test_split(np.arange(len(x)), test_size=.35, stratify=cues, random_state=seed)
    # MAD/评分的归一化也仅在内层训练中估计，不能沿用全数据的归一化。
    train_scores = detect_bad_trials(np.zeros_like(x[train]), x[train])[1]
    val_scores = detect_bad_trials(np.zeros_like(x[val]), x[val])[1]
    model = build_clean_reference_model(x[train],cues[train],np.zeros(len(train),bool),train_scores)
    # 内层验证背景也固定为相对低伪影试次，但不进入模板。
    idx = np.concatenate([val[cues[val]==c][np.argsort(val_scores[cues[val]==c],kind='stable')[:max(4,len(val[cues[val]==c])//2)]] for c in (-1,1)])
    bg, cs=x[idx], cues[idx]
    rows=[]
    for level in (0.,1.,2.):
        noisy=inject(bg,cs,seed+1000,level)
        for cand in CANDIDATES:
            r=known_errors(bg,apply_model(noisy,cs,model,cand),cs)
            rows.append(dict(candidate=cand,level=level,**r))
    df=pd.DataFrame(rows)
    # 清洁背景失真占 50%，两种污染各 25%；候选相同时偏向更弱校正。
    means=df.assign(weight=np.where(df.level==0,.5,.25))
    values=means.assign(loss=means.objective*means.weight).groupby('candidate',sort=False).loss.sum()
    return values.idxmin(),rows

def read_dataset(path):
    fs,labels,data=load_mat_file(path)
    if fs!=256 or list(labels[:3])!=CHANNELS or not str(labels[7]).startswith('VisCue'):
        raise ValueError(f'输入采样率或标签不匹配：{path}')
    if not np.all(np.isfinite(data[:3])): raise ValueError('EEG 包含非有限数值')
    onsets,cues=extract_viscue_events(data[7])
    if not np.isin(cues,[-1,1]).all(): raise ValueError('未知刺激方向')
    bounded=(onsets>=N)&(onsets+N_POST<=data.shape[1])
    event_ids=np.arange(1,len(onsets)+1)
    raw=segment_epochs(data[:3],onsets[bounded])
    x=robust_baseline_correct(segment_epochs(preprocess_continuous_eeg(data[:3],fs),onsets[bounded]))
    bad,scores,ptp=detect_bad_trials(raw,x)
    sat=(np.abs(raw)>=999).sum(axis=(1,2))
    jump=np.abs(np.diff(x,axis=-1)).max(axis=(1,2))
    audit=pd.DataFrame(dict(trial_id=event_ids, onset_sample=onsets, cue=cues,boundary_ok=bounded))
    for key,vals in dict(rejected=bad,artifact_score=scores,PTP=ptp,saturated_samples=sat,max_jump=jump).items():
        audit.loc[bounded,key]=vals
    audit['role']=np.where(~bounded,'边界剔除',np.where(audit.rejected.fillna(True),'质量剔除','折外评价'))
    # 不调整当前剔除集；只报告固定阈值 ±10% 的敏感性。
    sensitivity=[]
    for factor in (.9,1.,1.1):
        rejected=(sat>=max(1,round(15*factor)))|(ptp>1800*factor)|(jump>600*factor)
        sensitivity.append(dict(threshold_factor=factor,rejected=int(rejected.sum()),usable=int((~rejected).sum())))
    return raw[~bad],x[~bad],cues[bounded][~bad],scores[~bad],audit,sensitivity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(data,path):
    pd.DataFrame(data).to_csv(path,index=False,encoding='utf-8-sig')


def prestim_slopes(epochs):
    """逐试次逐通道 OLS 刺激前斜率，单位原始单位/ms；不使用刺激后样本。"""
    t=TIMES[:N_PRE]
    center=t-t.mean()
    x=epochs[:,:,:N_PRE]
    return ((x-x.mean(axis=2,keepdims=True))*center).sum(axis=2)/(center@center)


def conservative_blend(output,input_epochs,fraction):
    """只采用一部分校正量，避免无伪影真值时过度改写背景。"""
    return input_epochs + fraction * (output - input_epochs)


def selected_policy(candidate):
    """训练折选弱双分量时增强试次残差校正；其余候选沿用 V6。"""
    return '训练均值保护' if candidate == '保守双分量' else '保守比例'


def training_mean_delta(x_train,cues_train,model,candidate):
    delta=x_train-apply_v4(x_train,cues_train,model,candidate)
    return {cue:delta[cues_train==cue].mean(axis=0) for cue in (-1,1)}


def apply_v7(x,cues,model,candidate,mean_delta):
    v4=apply_v4(x,cues,model,candidate)
    if selected_policy(candidate)=='保守比例':
        return conservative_blend(v4,x,.15)
    delta=x-v4
    result=np.empty_like(x)
    for cue in (-1,1):
        mask=cues==cue
        result[mask]=x[mask]-.25*delta[mask]+.10*mean_delta[cue]
    return result


def apply_v3(x,cues,model):
    tpl,sv,sh,_=model
    return np.stack([correct_single_epoch_v3(e,tpl[c],sv[c],sh[c]) for e,c in zip(x,cues)])


def apply_v4(x,cues,model,candidate):
    return apply_model(x,cues,model,candidate)


def positive_presence(erp):
    return np.isfinite(positive_p300_features(erp)[2])


def extended_metrics(before,methods,cues,reference):
    frames=[]
    for name,x in methods.items():
        df=evaluate_metrics(before,x,cues,reference)
        df.insert(0,'stage',name)
        df['n_trials']=[int((cues==c).sum()) for c in df.cue]
        df['positive_peak_valid']=np.isfinite(df.P300_amp_after)
        df['reference_positive_peak_valid']=[positive_presence(reference[int(c)][CHANNELS.index(ch)]) for c,ch in zip(df.cue,df.channel)]
        df['latency_error_valid']=np.isfinite(df.latency_error_after_ms)
        frames.append(df)
    return pd.concat(frames,ignore_index=True)


def summarize_metrics(df,dataset):
    rows=[]
    columns=[c[:-6] for c in df if c.endswith('_after')]
    columns+=['latency_error_ms','P300_AUC_unit_ms','P300_AUC_error_unit_ms','P300_lat_ms']
    special={'latency_error_ms':'latency_error_after_ms','P300_AUC_unit_ms':'P300_AUC_after_unit_ms',
             'P300_AUC_error_unit_ms':'P300_AUC_error_after_unit_ms','P300_lat_ms':'P300_lat_after_ms'}
    for metric in dict.fromkeys(columns):
        col=special.get(metric,metric+'_after')
        if col not in df:continue
        value=np.stack([df[df.stage==s][col].to_numpy(float) for s in STAGES])
        common=np.isfinite(value).all(axis=0)
        for i,stage in enumerate(STAGES):
            rows.append(dict(dataset=dataset,metric=metric,stage=stage,
                             mean=float(value[i,common].mean()) if common.any() else np.nan,
                             paired_rows=int(common.sum()),valid_rows=int(np.isfinite(value[i]).sum()),total_rows=6))
    return rows


def bootstrap_metrics(methods,cues,reference,n_boot=400):
    """同一方向复用同一批重采样索引；模型和代理参考保持固定。"""
    rows=[]
    for condition in (-1,1):
        ids=np.flatnonzero(cues==condition)
        rng=np.random.default_rng(20260924+(condition+1))
        draw=rng.integers(0,len(ids),(n_boot,len(ids)))
        for ch,channel in enumerate(CHANNELS):
            reference_window=reference[condition][ch,WINDOW]
            boot={}
            for stage,x in methods.items():
                erp=x[ids,ch][draw].mean(axis=1)
                w=erp[:,WINDOW]
                boot[stage]={
                    'proxy_MAE':np.mean(np.abs(w-reference_window),axis=1),
                    'positive_mean':np.maximum(w,0).mean(axis=1),
                }
            observed={stage:{
                'proxy_MAE':float(np.mean(np.abs(x[ids,ch].mean(axis=0)[WINDOW]-reference_window))),
                'positive_mean':float(np.maximum(x[ids,ch].mean(axis=0)[WINDOW],0).mean()),
            } for stage,x in methods.items()}
            for metric in ('proxy_MAE','positive_mean'):
                for stage in STAGES:
                    arr=boot[stage][metric]
                    lo,hi=np.quantile(arr,[.025,.975])
                    rows.append(dict(cue=condition,channel=channel,metric=metric,stage=stage,
                                     point=observed[stage][metric],low=lo,high=hi,
                                     n_trials=len(ids),n_boot=n_boot))
                for baseline in ('预处理','V3','V4','V6'):
                    arr=boot['V7'][metric]-boot[baseline][metric]
                    lo,hi=np.quantile(arr,[.025,.975])
                    rows.append(dict(cue=condition,channel=channel,metric=metric,stage='V7减'+baseline,
                                     point=observed['V7'][metric]-observed[baseline][metric],
                                     low=lo,high=hi,n_trials=len(ids),n_boot=n_boot))
    return rows


def group_uncertainty(methods,cues,reference,n_boot=800,seed=20260924):
    """同试次分方向重采样的组均值区间；模型、剔除集及代理参考固定。"""
    selected=('预处理','V6','V7')
    post=(TIMES>=0)&(TIMES<=500)
    rng=np.random.default_rng(seed)
    erps={s:{} for s in selected}
    mae={s:[] for s in selected}
    snr={s:[] for s in selected}
    observed_erps={s:{} for s in selected}
    observed_mae={s:[] for s in selected}
    observed_snr={s:[] for s in selected}
    for cue in (-1,1):
        ids=np.flatnonzero(cues==cue)
        counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=n_boot)
        weights=counts/len(ids)
        for stage in selected:
            trials=methods[stage][ids]
            mean=np.einsum('bn,nct->bct',weights,trials,optimize=True)
            second=np.einsum('bn,nct->bct',weights,trials[:,:,post]**2,optimize=True)
            erps[stage][cue]=mean
            mae[stage].append(np.mean(np.abs(mean[:,:,WINDOW]-reference[cue][:,WINDOW][None,:,:]),axis=2))
            signal=np.mean(mean[:,:,post]**2,axis=2)
            residual=np.maximum(np.mean(second-mean[:,:,post]**2,axis=2),0)
            snr[stage].append(10*np.log10((signal+1e-12)/(residual+1e-12)))
            full=trials.mean(axis=0)
            observed_erps[stage][cue]=full
            observed_mae[stage].extend(np.mean(np.abs(full[:,WINDOW]-reference[cue][:,WINDOW]),axis=1))
            observed_snr[stage].extend(snr_proxy_db(trials[:,ch]) for ch in range(3))
    mae={s:np.concatenate(mae[s],axis=1).mean(axis=1) for s in selected}
    snr={s:np.concatenate(snr[s],axis=1).mean(axis=1) for s in selected}
    observed_mae={s:float(np.mean(observed_mae[s])) for s in selected}
    observed_snr={s:float(np.mean(observed_snr[s])) for s in selected}
    retention={};observed_retention={}
    base_norm=np.linalg.norm(erps['预处理'][-1][:,:,WINDOW]-erps['预处理'][1][:,:,WINDOW],axis=2)
    observed_base_norm=np.linalg.norm(observed_erps['预处理'][-1][:,WINDOW]-
                                      observed_erps['预处理'][1][:,WINDOW],axis=1)
    for stage in ('V6','V7'):
        norm=np.linalg.norm(erps[stage][-1][:,:,WINDOW]-erps[stage][1][:,:,WINDOW],axis=2)
        retention[stage]=np.mean(norm/(base_norm+1e-12),axis=1)
        one=np.linalg.norm(observed_erps[stage][-1][:,WINDOW]-
                           observed_erps[stage][1][:,WINDOW],axis=1)
        observed_retention[stage]=float(np.mean(one/(observed_base_norm+1e-12)))
    rows=[]
    for stage in ('V6','V7'):
        values={
            'proxy_MAE_reduction_pct':(100*(mae['预处理']-mae[stage])/mae['预处理'],
                                       100*(observed_mae['预处理']-observed_mae[stage])/observed_mae['预处理']),
            'SNR_proxy_gain_dB':(snr[stage]-snr['预处理'],observed_snr[stage]-observed_snr['预处理']),
            'left_right_retention_ratio':(retention[stage],observed_retention[stage]),
        }
        for metric,(samples,point) in values.items():
            low,high=np.quantile(samples,[.025,.975])
            rows.append(dict(stage=stage,metric=metric,point=point,low=low,high=high,
                             n_trials=len(cues),n_boot=n_boot))
    for metric,samples,point in (
        ('V7_minus_V6_proxy_MAE',mae['V7']-mae['V6'],observed_mae['V7']-observed_mae['V6']),
        ('V7_minus_V6_SNR_proxy_dB',snr['V7']-snr['V6'],observed_snr['V7']-observed_snr['V6']),
    ):
        low,high=np.quantile(samples,[.025,.975])
        rows.append(dict(stage='V7减V6',metric=metric,point=point,low=low,high=high,
                         n_trials=len(cues),n_boot=n_boot))
    return rows


def temporal_contrast_stability(methods,cues,trial_ids):
    """按原始事件顺序前后对半，检验左减右差分是否在时段间重复。"""
    order=np.argsort(trial_ids)
    first=np.zeros(len(cues),bool);first[order[:len(order)//2]]=True
    halves=(first,~first)
    counts={(part,cue):int(np.sum(mask&(cues==cue)))
            for part,mask in enumerate(halves) for cue in (-1,1)}
    if min(counts.values())<2:
        raise ValueError('时段稳定性分析需要每半段至少两个左右刺激试次')
    rows=[]
    for stage,x in methods.items():
        diff=[]
        for mask in halves:
            left=x[mask&(cues==-1)].mean(axis=0)
            right=x[mask&(cues==1)].mean(axis=0)
            diff.append((left-right)[:,WINDOW])
        for channel,one,two in [('三通道合并',diff[0].ravel(),diff[1].ravel())]+[
                (name,diff[0][ch],diff[1][ch]) for ch,name in enumerate(CHANNELS)]:
            rows.append(dict(stage=stage,channel=channel,
                             first_left_n=counts[(0,-1)],first_right_n=counts[(0,1)],
                             second_left_n=counts[(1,-1)],second_right_n=counts[(1,1)],
                             contrast_correlation=safe_corr(one,two),
                             second_first_norm_ratio=float(np.linalg.norm(two)/(np.linalg.norm(one)+1e-12))))
    return rows


def spline_fit(erp):
    """固定 50 ms 结点的三次样条；约束自由度，避免近似逐点插值。"""
    mask=(TIMES>=50)&(TIMES<=750)
    t=TIMES[mask]
    y=erp[mask]
    knots=np.arange(100.,750.,50.)
    spline=LSQUnivariateSpline(t,y,knots,k=3)
    fitted=spline(TIMES)
    residual=erp-fitted
    win=np.flatnonzero(WINDOW)
    segment=fitted[win]
    prominence=max(1.,.10*float(np.ptp(segment)))
    peaks,properties=find_peaks(segment,prominence=prominence)
    peaks=peaks[(peaks>2)&(peaks<len(segment)-3)&(segment[peaks]>0)]
    if len(peaks):
        best=peaks[np.argmax(segment[peaks])]
        peak=float(segment[best]);lat=float(TIMES[win[best]])
        status='窗内局部正峰，非生理确认'
    else:
        peak=np.nan;lat=np.nan;status='无合格窗内局部正峰'
    return fitted,dict(fit_family='fixed_knot_cubic_spline',knot_spacing_ms=50.,
                       fit_RMSE_50_750=float(np.sqrt(np.mean(residual[mask]**2))),
                       fit_RMSE_250_500=float(np.sqrt(np.mean(residual[WINDOW]**2))),
                       positive_mean=float(np.mean(np.maximum(segment,0))),
                       positive_area_unit_ms=float(np.trapezoid(np.maximum(segment,0),TIMES[WINDOW])),
                       local_positive_peak=peak,local_peak_latency_ms=lat,peak_status=status)


def make_axes(title,subtitle):
    fig,axes=plt.subplots(3,2,figsize=(13.5,9.5),sharex=True,layout='constrained')
    fig.suptitle(title+'\n'+subtitle,fontsize=14)
    for ax in axes.flat:
        ax.axvline(0,color='#aeb5bc',lw=.75)
        ax.axhline(0,color='#cbd0d5',lw=.6)
        ax.axvspan(250,500,color='#d5ebed',alpha=.27)
        ax.set_xlabel('相对提示时间（ms）')
        ax.set_ylabel('原始电位单位')
        ax.spines[['top','right']].set_visible(False)
    return fig,axes


def shared_row_limits(axes,values,margin=.06):
    for row in range(3):
        combined=np.concatenate([np.asarray(v).ravel() for v in values[row]])
        low,high=float(np.nanmin(combined)),float(np.nanmax(combined))
        pad=max((high-low)*margin,1.)
        for col in range(2): axes[row,col].set_ylim(low-pad,high+pad)


def draw_lines(ax,signals,mask=None,legend=True):
    mask=np.ones(len(TIMES),bool) if mask is None else mask
    for label,y in signals.items():
        ax.plot(TIMES[mask],np.asarray(y)[mask],label=label,color=COLORS[label],
                ls=LINES[label],lw=1.5 if label=='V7' else 1.)
    if legend:ax.legend(loc='upper left',fontsize=8,ncol=2)


def save_fig(fig,path):
    fig.savefig(path,dpi=170)
    plt.close(fig)


def plot_condition(name,methods,cues,refs,out,zoom=False):
    title=name+(' · 250–500 ms 正向响应窗' if zoom else ' · 左右刺激三通道 ERP')
    subtitle='各图同一有效试次，同行左右共享纵轴；代理参考只作描述' if zoom else '三阶段同一试次；阴影是V7逐点95%试次重采样区间'
    fig,axes=make_axes(title,subtitle)
    scale={k:[] for k in range(3)}
    for col,c in enumerate((-1,1)):
        n=int((cues==c).sum())
        for ch,channel in enumerate(CHANNELS):
            ax=axes[ch,col]
            curves={k:x[cues==c,ch].mean(axis=0) for k,x in methods.items()}
            if zoom:curves['代理参考']=refs[c][ch]
            draw_lines(ax,curves,WINDOW if zoom else None)
            if not zoom:
                lower,upper=compute_bootstrap_ci(methods['V7'][cues==c,ch],n_boot=400)
                ax.fill_between(TIMES,lower,upper,color=COLORS['V7'],alpha=.14,label='V7逐点95%试次区间')
                ax.legend(loc='upper left',fontsize=8,ncol=2)
                scale[ch].extend([*curves.values(),lower,upper])
            else:scale[ch].extend([v[WINDOW] for v in curves.values()])
            ax.set_title(f'{channel} · {"左" if c==-1 else "右"}刺激 · n={n}')
            if zoom:ax.set_xlim(250,500)
    shared_row_limits(axes,scale)
    save_fig(fig,out/('P300分析窗对比.png' if zoom else '左右刺激三通道ERP.png'))


def plot_examples(name,raw,methods,cues,scores,trial_ids,out,quantile,filename):
    fig,axes=make_axes(name+' · 固定试次原始/处理对照',f'同方向伪影评分第{int(100*quantile)}百分位；原始试次编号已标注，纵轴不截断')
    scale={k:[] for k in range(3)}
    selected=[]
    for col,c in enumerate((-1,1)):
        ids=np.flatnonzero(cues==c)
        order=ids[np.argsort(scores[ids],kind='stable')]
        ix=order[round(quantile*(len(order)-1))]
        selected.append(dict(cue=c,quantile=quantile,trial_id=int(trial_ids[ix])))
        for ch,channel in enumerate(CHANNELS):
            original=raw[ix,ch]-np.median(raw[ix,ch,:N_PRE])
            curves={'原始':original,**{k:x[ix,ch] for k,x in methods.items()}}
            draw_lines(axes[ch,col],curves)
            axes[ch,col].set_title(f'{channel} · {"左" if c==-1 else "右"} · 原始试次 #{trial_ids[ix]}')
            scale[ch].extend(curves.values())
    shared_row_limits(axes,scale)
    save_fig(fig,out/filename)
    return selected


def fixed_example_ids(cues,scores,trial_ids):
    """无绘图模式也保存同一套预先定义的示例编号。"""
    rows=[]
    for quantile in (.50,.75):
        for cue in (-1,1):
            ids=np.flatnonzero(cues==cue)
            order=ids[np.argsort(scores[ids],kind='stable')]
            ix=order[round(quantile*(len(order)-1))]
            rows.append(dict(cue=cue,quantile=quantile,trial_id=int(trial_ids[ix])))
    return rows


def plot_heatmap(name,methods,cues,trial_ids,out,focus=False):
    stages=list(methods)
    all_abs=np.concatenate([np.abs(x[:,0]).ravel() for x in methods.values()])
    full=float(all_abs.max())
    limit=float(np.quantile(all_abs,.98)) if focus else full
    exceed=float(np.mean(all_abs>limit))*100
    fig,axes=plt.subplots(2,len(stages),figsize=(17,8),layout='constrained')
    for row,c in enumerate((-1,1)):
        ids=np.flatnonzero(cues==c)
        for col,stage in enumerate(stages):
            ax=axes[row,col]
            image=ax.imshow(methods[stage][ids,0,:],aspect='auto',origin='lower',interpolation='nearest',
                            cmap='RdBu_r',vmin=-limit,vmax=limit,extent=[TIMES[0],TIMES[-1],.5,len(ids)+.5])
            ticks=np.unique(np.linspace(0,len(ids)-1,5,dtype=int))
            ax.set_yticks(ticks+1,[str(i) for i in trial_ids[ids[ticks]]])
            ax.set_title(f'{"左" if c==-1 else "右"}刺激 · {stage} · n={len(ids)}')
            ax.set_xlabel('相对提示时间（ms）')
            ax.set_ylabel('原始试次编号')
    fig.colorbar(image,ax=axes,label='Fz 原始电位单位')
    subtitle=(f'细节视图：图内各阶段共同98%绝对值分位 ±{limit:.1f}；{exceed:.2f}%像素截色' if focus
              else f'完整色域：图内各阶段共用 ±{limit:.1f}，无截色')
    fig.suptitle(name+' · Fz 试次时间热力图\n'+subtitle)
    save_fig(fig,out/('Fz热力图_细节视图.png' if focus else 'Fz热力图_完整色域.png'))


def plot_spatial(name,methods,cues,out):
    fig,axes=make_axes(name+' · 左右与额区差分','左列为左减右；右列为F3减F4及左右交互；各面板内三阶段同尺度；差分可能包含残余眼动')
    scale={k:[] for k in range(3)}
    for ch,channel in enumerate(CHANNELS):
        signals={s:x[cues==-1,ch].mean(0)-x[cues==1,ch].mean(0) for s,x in methods.items()}
        draw_lines(axes[ch,0],signals)
        axes[ch,0].set_title(channel+' · 左减右')
        scale[ch].extend(signals.values())
    for row,c in enumerate((-1,1)):
        signals={s:x[cues==c,1].mean(0)-x[cues==c,2].mean(0) for s,x in methods.items()}
        draw_lines(axes[row,1],signals)
        axes[row,1].set_title(('左' if c==-1 else '右')+'刺激 · F3减F4')
        scale[row].extend(signals.values())
    signals={s:(x[cues==-1,1].mean(0)-x[cues==-1,2].mean(0))
             -(x[cues==1,1].mean(0)-x[cues==1,2].mean(0)) for s,x in methods.items()}
    draw_lines(axes[2,1],signals)
    axes[2,1].set_title('左右交互 · (F3−F4)左减右')
    # 差分类型不同，不强制左右列同一幅度；每图方法共享其自身坐标。
    save_fig(fig,out/'左右与F3-F4差分.png')


def plot_fits(name,methods,cues,out):
    fig,axes=make_axes(name+' · V7 三通道分方向响应曲线拟合',
                       '固定50 ms结点三次样条；灰色为残差；窗内局部正峰只是描述量，不确认生理P300')
    records=[]
    for col,c in enumerate((-1,1)):
        n=int((cues==c).sum())
        for ch,channel in enumerate(CHANNELS):
            erp=methods['V7'][cues==c,ch].mean(0)
            fitted,fit=spline_fit(erp)
            ax=axes[ch,col]
            ax.plot(TIMES,erp,color=COLORS['V7'],label='V7 ERP')
            mask=(TIMES>=50)&(TIMES<=750)
            ax.plot(TIMES[mask],fitted[mask],color='#32373c',ls='--',label='样条拟合')
            ax.plot(TIMES[mask],(erp-fitted)[mask],color='#9a938d',lw=.8,label='残差')
            if np.isfinite(fit['local_peak_latency_ms']):
                ax.scatter([fit['local_peak_latency_ms']],[fit['local_positive_peak']],s=14,color='#32373c',zorder=3)
            ax.legend(fontsize=8,loc='upper left')
            ax.set_title(f'{channel} · {"左" if c==-1 else "右"} · n={n} · 窗内拟合RMSE={fit["fit_RMSE_250_500"]:.2f}\n'+fit['peak_status'],fontsize=10)
            records.append(dict(cue=c,channel=channel,n_trials=n,**fit))
    save_fig(fig,out/'分方向样条拟合与残差.png')
    save_csv(records,out/'分方向样条拟合参数.csv')
    return records


def stage_summary_markdown(data,dataset):
    subset=pd.DataFrame(data)
    subset=subset[subset.dataset==dataset]
    result=[]
    for metric in ('MAE','corr','SNR_proxy_dB','baseline_RMS','trial_PTP_median','edge_band_ratio',
                   'P300_positive_mean_error','P300_AUC_error_unit_ms'):
        rows=subset[subset.metric==metric].set_index('stage')
        if len(rows)==len(STAGES):
            result.append('| '+metric+' | '+' | '.join(f'{rows.loc[s,"mean"]:.3f}' for s in STAGES)+' | '+str(int(rows.loc['V7','paired_rows']))+' |')
    return result


def write_report(out,datasets,summary,spatial,benchmark,fitrows,group_intervals,stability):
    summary=pd.DataFrame(summary)
    spatial=pd.DataFrame(spatial)
    benchmark=pd.DataFrame(benchmark)
    fit=pd.DataFrame(fitrows)
    intervals=pd.DataFrame(group_intervals)
    stability=pd.DataFrame(stability)
    lines=['# 第一问：脑电预处理、伪影校正与有效视觉响应拟合（终稿实验报告）','',
    '## 赛题对应与结果定位','',
    '针对项目一与项目二、Fz/F3/F4原始记录，完成视觉提示事件提取、固定质量筛查、连续信号预处理、折外校正、左右方向分层响应估计及曲线拟合。刺激方向从VisCue读取，仅用于已知条件下的离线分析；本结果不是未知刺激解码器。',
    '', '核心结论：V7按训练折选型控制校正强度；四组代理MAE点估计下降约5.4%–10.0%，同试次重采样的组均值区间均在零以上。SNR代理点估计略升，但四组增量区间均跨零，不能称稳定提高。左右差分幅度保留约0.81–0.95，却不能证明保留的是神经特征；四组1倍和2倍半合成恢复误差较V6略低，弱注入与零注入仍有背景改动。',
    '', '## 数据与数学定义','',
    '- 四组输入各100个提示事件；只用原始Fz、F3、F4与VisCue，绝不使用机器处理后的FzDecon/F3Decon/F4Decon作为输入。采样率256 Hz；试次窗−250至796.875 ms，分析窗250–500 ms。幅度单位沿用“原始电位单位”，不假定µV。',
    '- 固定预处理：连续信号60 Hz陷波、0.1–30 Hz零相位带通、刺激前250 ms中位数基线校正。提高高通截止频率会改变慢ERP的幅值和潜伏期，因此未通过抬高截止频率强行消除项目二的慢变化。',
    '- 严重坏试次规则沿用V3：原始跨通道采样绝对值≥999的总点数≥15，或滤波后峰峰值>1800，或相邻点跳变>600；这些是数据量纲下的工程阈值。没有证据表明“15点”是连续硬件饱和，指标解释按实际计算。',
    '- 五折外层分层划分、固定随机种子42。每折仅用训练试次估计方向条件模板、残差尺度和选型；当前折测试试次不进入自己的模板。被剔除试次不进入任何ERP、拟合或指标。',
    '- V3为原双分量校正；V4为训练内半合成选型；V6为取V4校正量15%的保守基线。V7仍由外折训练试次选择V4候选。选到“保守双分量”时，采用训练均值保护策略；其他候选沿用V6。0.15/0.25等比例是在查看四组开发结果后确定，不是独立受试者验证出的最优值。',
    '- 训练均值保护只从测试试次扣除25%的V4估计改动，再加回该方向训练试次平均V4改动的10%；这旨在更多抑制试次间波动，同时限制方向平均响应偏移。平均偏移仍可能改变真实视觉特征；它不是神经源分离。刺激前斜率仅作为诊断量保存，不进入V7修正。',
    '- 代理参考为当前外折训练集中同方向低伪影半数试次的逐点中位数；每个评价试次对应训练模板按条件平均用于指标。它与校正模型相关，不是独立无噪声真值。预处理/V3/V4/V6/V7在相同试次、参考和窗口内比较，六个方向×通道行等权；缺失指标使用五阶段共同有效行配对，不补零。',
    '- 曲线拟合采用内部结点相隔50 ms的三次样条，固定自由度以避免逐点插值；输出50–750 ms和250–500 ms残差RMSE。只有250–500 ms内的局部正峰才报告峰时刻；正面积可在无局部峰时报告。它仅是额区正向响应表征，不能称已经在中央/顶区观察到典型P300。',
    '', '### 校正模型与选型目标','',
    '对每个训练折和方向，以低伪影半数试次的逐点中位数构造三通道模板 T；单试次残差为 r_c(t)=x_c(t)−T_c(t)。垂直候选参考取三通道残差中位数，水平候选参考取 (r_F4−r_F3)/2；两者经平滑、相对于训练残差尺度的软门控后组成矩阵 A。对每通道求受系数边界约束的岭回归 β_c≈(AᵀA+λI)⁻¹Aᵀr_c，再从 x_c 中减去 Aβ_c。V4 候选还包括门控强度、水平分量上限、时域保护和不校正，具体参数保存在源码与逐折策略CSV。',
    'V4 的训练内验证在0、1、2倍人工污染上选择候选：L=(RMSE_全窗+0.5×RMSE_左右差分+正向均值绝对误差)/背景RMS；三档权重依次为0.5、0.25、0.25。记δ_i=x_i−x_V4,i，m_y=同方向外折训练试次δ的均值。V6及V7保守分支为x_i−0.15δ_i；V7训练均值保护分支为x_i−0.25δ_i+0.10m_y。m_y仅由外折训练试次产生。分支规则和比例经过四组数据开发，外折数字是本数据集模型分析，不是独立泛化精度。',
    '分方向 ERP 为同一方向有效试次的逐点算术均值。曲线用 f(t)=Σ_j θ_j B_j(t) 拟合，其中 B_j 为固定50 ms内部结点的三次B样条基函数，θ通过50–750 ms普通最小二乘估计；拟合残差 RMSE 用相同 ERP 点计算，是描述性拟合误差，不是独立预测误差。',
    '', '## 试次与训练参考对账','',
    '| 数据组 | 事件 | 边界剔除 | 质量剔除 | 可用并折外评价 | 左 / 右 | 每折训练参考数 |',
    '|---|---:|---:|---:|---:|---|---|']
    for ds,d in datasets.items():
        a=d['audit'];c=d['cues']
        refs=d['reference_counts']
        lines.append(f'| {d["name"]} | {len(a)} | {int((a.role=="边界剔除").sum())} | {int((a.role=="质量剔除").sum())} | {len(c)} | {int((c==-1).sum())} / {int((c==1).sum())} | '+', '.join(map(str,refs))+' |')
    lines += ['', '训练参考在不同外折可以重复出现，上表不能将每折参考数简单相加。各组“完整事件与试次审计.csv”“逐折训练参考评价清单.csv”“可复核波形.npz”保留原始试次编号、折号和每折训练校正均值。',
             '', '## 五阶段同口径指标','',
             '表中“V3”指按共同训练评分和参考流程重算的V3校正，数值不得与旧版不同评价口径CSV直接相减。MAE及相关针对代理参考；SNR为ERP功率/试次残差功率的代理值，残差也含真实试次差异。',
             '', '| 数据组 | 指标 | 预处理 | V3 | V4 | V6 | V7 | 共同有效行数/6 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds,d in datasets.items():
        for row in stage_summary_markdown(summary,ds):
            lines.append('| '+d['name']+' | '+row[2:])
    lines += ['', '### 视觉形状与空间差分','',
              '左减右、F3减F4均用相同有效试次计算。差分范数比以预处理差分为分母；接近1只说明幅度更接近处理前，不能区分保留神经信息与保留方向相关眼动。',
              '', '| 数据组 | V3左右差分比 | V4左右差分比 | V6左右差分比 | V7左右差分比 | V7差分波形相关均值 |',
              '|---|---:|---:|---:|---:|---:|']
    for ds,d in datasets.items():
        g=spatial[(spatial.dataset==ds)&(spatial.quantity=='left_minus_right_ERP')]
        vals=[g[g.stage==s].retention_ratio.mean() for s in ('V3','V4','V6','V7')]
        corr=g[g.stage=='V7'].waveform_correlation.mean()
        lines.append(f'| {d["name"]} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} | {corr:.3f} |')
    lines += ['', '逐通道左右差分如下，避免三通道均值掩盖局部退步。比值和波形相关都是相对预处理，而不是相对无伪影真值。',
              '', '| 数据组 | 通道 | V7左右差分比 | V7波形相关 |', '|---|---|---:|---:|']
    for ds,d in datasets.items():
        g=spatial[(spatial.dataset==ds)&(spatial.quantity=='left_minus_right_ERP')&(spatial.stage=='V7')]
        for ch in CHANNELS:
            row=g[g.channel_or_cue==ch].iloc[0]
            lines.append(f'| {d["name"]} | {ch} | {row.retention_ratio:.3f} | {row.waveform_correlation:.3f} |')
    lines += ['', 'F3/F4空间差分的完整数值见“左右刺激与额区空间差异指标.csv”。',
              '', '### 组均值区间与时段稳定性','',
              '下表的95%区间按左右方向分别对同一批试次重采样，V6/V7和预处理共用索引；已选模型、试次剔除集与训练代理参考固定。它量化本组试次抽样波动，不覆盖跨时段相关性、开发调参或跨受试者不确定性。',
              '', '| 数据组 | V7代理MAE降幅% [95%区间] | V7 SNR代理增量dB [95%区间] | V7左右差分比 [95%区间] | 前后半段差分相关 |',
              '|---|---:|---:|---:|---:|']
    for ds,d in datasets.items():
        g=intervals[(intervals.dataset==ds)&(intervals.stage=='V7')].set_index('metric')
        t=stability[(stability.dataset==ds)&(stability.stage=='V7')&
                    (stability.channel=='三通道合并')].iloc[0]
        def cell(metric,fmt):
            row=g.loc[metric]
            return f'{row.point:{fmt}} [{row.low:{fmt}}, {row.high:{fmt}}]'
        lines.append(f'| {d["name"]} | {cell("proxy_MAE_reduction_pct",".1f")} | '
                     f'{cell("SNR_proxy_gain_dB","+.3f")} | '
                     f'{cell("left_right_retention_ratio",".3f")} | '
                     f'{t.contrast_correlation:.3f} |')
    lines += ['', '四组SNR代理增量的组均值区间均跨零；它只支持“点估计略升”，不支持稳定增益。试次按原始事件编号前后对半，每半段分别计算左右ERP差分；A项目一与B项目二的三通道合并差分相关为负，说明方向差异的时间稳定性尚未确立。这个检查也不能将真实视觉响应与方向相关眼动分离。逐通道时段相关、每半段左右样本数及V7−V6差异区间见汇总CSV。',
              '', '### 正峰缺失、样条拟合与不确定性','',
              '逐方向逐通道CSV列出每阶段`positive_peak_valid`、`reference_positive_peak_valid`与潜伏期有效性。无正峰不赋予零潜伏期。样条拟合全部可计算，但“可拟合”不等于“有可信生理P300”。',
              '', '| 数据组 | V7有合格窗内局部正峰 / 6 | V7正峰检出行 / 6 | 五阶段峰误差共同配对行 / 6 |',
              '|---|---:|---:|---:|']
    for ds,d in datasets.items():
        f=fit[fit.dataset==ds]
        m=d['metrics'];v=m[m.stage=='V7']
        paired=summary[(summary.dataset==ds)&(summary.metric=='amp_error')&(summary.stage=='V7')]
        lines.append(f'| {d["name"]} | {int((f.peak_status=="窗内局部正峰，非生理确认").sum())} | {int(v.positive_peak_valid.sum())} | {int(paired.paired_rows.iloc[0]) if len(paired) else 0} |')
    lines += ['', '逐方向逐通道区间文件进行400次试次重采样，组均值区间文件进行800次分方向配对重采样；均固定处理后的波形与代理参考。区间不能推断人群疗效或诊断能力，也不等于独立受试者验证。',
              '', '## 半合成闭环验证','',
              '在四组各外折测试试次上人工加入随机时刻、宽度、极性的眨眼样、扫视样和运动样扰动；污染前的预处理实测波形是可计算恢复目标，仍可能含原有伪影。固定三个测试随机种子和0/0.5/1/2倍注入；0倍只检验无新污染时算法改动背景的程度。每组每幅度、每算法另存原始单位RMSE、按背景RMS归一化RMSE、左右差分恢复误差和正向均值误差。',
              '', '| 数据组 | 幅度 | 未校正归一化RMSE | V3 | V4 | V6 | V7 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for ds,d in datasets.items():
        for level in (0.,.5,1.,2.):
            g=benchmark[(benchmark.dataset==ds)&(benchmark.level==level)]
            values=[g[g.stage==s].normalized_RMSE.mean() for s in STAGES]
            lines.append(f'| {d["name"]} | {level:g} | '+' | '.join(f'{x:.3f}' for x in values)+' |')
    lines += ['', 'V7在四组1倍和2倍场景较V6均有小幅降低，但0倍背景改动通常略大。A项目一在0.5倍轻扰动时，V7恢复RMSE仍高于未校正；这说明分支规则不能保证所有伪影强度都获益。0倍注入的非零恢复误差是算法对背景本身的改动，不能解释为恢复收益。不能将半合成结果推广为真实无噪声脑电恢复精度；没有专门的眼电或独立真值。各试次重复注入也不能当作独立受试者。',
              '', '## 图表检查与解释','',
              '- 所有五阶段指标共享试次；主图只展示预处理、V6、V7，以减少线条遮挡。左右面板同一通道共享纵轴。置信带来自试次重采样，注明为逐点区间。',
              '- 典型波形同时给50%与75%伪影评分试次，不把75%试次称普通典型；标注原始编号。两图都使用完整幅度，不截断原始大波形。',
              '- 热力图同时给完整共用色域与共用98%分位细节视图；细节图显式写出截色像素比例。不得凭细节图单独宣称大波形消失。',
              '- 空间图右列依次为左刺激、右刺激的F3−F4及二者差；各面板内三阶段共享尺度，跨不同差分类型不强行共用纵轴。',
              '- 跨项目图按左、右方向分别比较，各方向写明样本数；拟合图显示残差和窗内拟合RMSE。',
              '- 汇总图以四组为行、三项指标为列，并加入V6/V7的组均值试次重采样区间；不同单位与意义的指标不合成单一综合分数。',
              '', '## 结论边界和可复现性','',
              'V7是四组数据上经过开发调试的折外实验结果：操作层面测试试次未进入其折的模板或策略判断，但开发者已查看全部四组数据，故这些数字是开发集结果，不应称为全新受试者独立验证。三额区通道没有眼电通道，无法从这份数据单独证明前额共同缓慢变化是眼电还是神经慢电位；前后半段差分不稳定进一步限制“形状特征得到可靠保留”的结论。',
              '', '要把结果用于未知刺激分类，需要在完全独立数据上重新建立不使用测试Cue的校正和分类流程；当前曲线是已知条件下的视觉响应描述。数模论文可用本结果讨论可观测信号和方法取舍，不能声称神经源唯一识别、临床诊断准确率或人群泛化。',
              '', '运行 `python EEG_P300_artifact_correction_v7.py --output eeg_v7_results`，读取各组“可复核波形.npz”和运行清单JSON重算。源程序、四份.mat文件以及数值结果的SHA256均保存。',
              '', '## 方法出处','',
              '- [Tanner等：不适当高通滤波可能使认知ERP产生人为效应](https://pmc.ncbi.nlm.nih.gov/articles/PMC4506207/)。这里据此保留0.1 Hz基线滤波并把更强滤波仅作为未采用的备选。',
              '- [Luck等：伪影校正与剔除的ERP评价框架](https://pmc.ncbi.nlm.nih.gov/articles/11021170/)。这里据此同时报告噪声代理量、响应特征与局限，而非只凭一种误差证明去噪。']
    (out/'第一问终稿实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot_cross_task(out,datasets):
    folder=out/'跨项目对比';folder.mkdir(exist_ok=True)
    for subject in 'AB':
        fig,axes=make_axes('受试者'+subject+' · V7按方向跨项目ERP',
                           '同通道左右共享纵轴；项目一/二仅描述同受试者条件差异，不作为跨人群效应')
        scale={ch:[] for ch in range(3)}
        for col,c in enumerate((-1,1)):
            for ch,channel in enumerate(CHANNELS):
                ax=axes[ch,col]
                for task,color,style in ((1,'#007f89','-'),(2,'#c87927','--')):
                    d=datasets[f'VisualCog{subject}_Task-{task}']
                    y=d['stages']['V7'][d['cues']==c,ch].mean(0)
                    ax.plot(TIMES,y,color=color,ls=style,label=f'项目{task} · n={(d["cues"]==c).sum()}')
                    scale[ch].append(y)
                ax.legend(fontsize=8)
                ax.set_title(channel+' · '+('左' if c==-1 else '右')+'刺激')
        shared_row_limits(axes,scale)
        save_fig(fig,folder/f'受试者{subject}_左右分方向跨项目.png')


def plot_benchmark(out,synthetic):
    folder=out/'半合成验证'
    frame=pd.DataFrame(synthetic)
    fig,axes=plt.subplots(2,2,figsize=(12.5,8),layout='constrained',sharex=True)
    for ax,(ds,g) in zip(axes.flat,frame.groupby('dataset',sort=False)):
        for stage in PLOT_STAGES:
            s=g[g.stage==stage].groupby('level').normalized_RMSE.mean()
            ax.plot(s.index,s.values,color=COLORS[stage],ls=LINES[stage],marker='o',label=stage)
        ax.set_title(ds+' · 背景样本与外折固定')
        ax.set_xlabel('人工注入幅度倍数（0表示未注入）')
        ax.set_ylabel('恢复RMSE / 背景RMS')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8,ncol=2)
    fig.suptitle('四组各自归一化：已知注入扰动恢复与零注入背景改动\n污染前为实测预处理背景，可能已含伪影；试次重用不表示独立受试者')
    save_fig(fig,folder/'四组多幅度半合成验证.png')



# 单文件重建提交结果中的三栏伪影分解图；绘制规则沿用原独立脚本。
DECOMPOSITION_COLORS = {
    'raw': '#455A64',       # 灰蓝色：原始未加工记录
    'clean': '#00796B',     # 青绿色：V7 校正后纯净脑电
    'artifact': '#C62828',  # 铁红色：提取出的伪影与噪声
    'p300_span': '#FFF3CD', # 淡黄色：250-500 ms 视觉响应分析窗
}
DECOMPOSITION_OUTPUT_DIR = ROOT / 'output/三栏伪影分解对比图'

def plot_decomposition_for_dataset(results_dir, dataset_folder_name, quantile, quantile_label, filename):
    """为指定数据集生成三通道、左右方向的三栏分解图"""
    data_path = results_dir / dataset_folder_name / '可复核波形.npz'
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    npz_data = np.load(data_path)
    raw = npz_data['raw']
    v7 = npz_data['v7']
    cues = npz_data['cues']
    trial_ids = npz_data['trial_ids']
    times = npz_data['times_ms']

    # 读取预先指定的固定示例试次编号
    fixed_csv = results_dir / dataset_folder_name / '固定示例试次编号.csv'
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
            ax_raw.plot(times, raw_bc[ch], color=DECOMPOSITION_COLORS['raw'], lw=1.2, label='原始通道 (基线对齐)')
            ax_raw.axvline(0, color='black', lw=0.8, ls='--')
            ax_raw.axvspan(250, 500, color=DECOMPOSITION_COLORS['p300_span'], alpha=0.6, label='P300 分析窗 (250–500 ms)')
            ax_raw.set_title(f'{cue_title} · {ch_name} · 试次 #{tid} [原始]', fontsize=10.5, fontweight='bold', pad=4)
            ax_raw.grid(True, alpha=0.25, ls=':')
            if ch == 0:
                ax_raw.set_ylabel('原始记录\n(电位单位)', fontsize=9.5, fontweight='bold')
            if row_offset == 0 and ch == 0:
                ax_raw.legend(loc='upper right', fontsize=8, framealpha=0.9)

            # --- Row 2: V7 去噪后 ---
            ax_clean = axes[row_offset + 1, ch]
            ax_clean.plot(times, clean[ch], color=DECOMPOSITION_COLORS['clean'], lw=1.5, label='V7 校正后脑电')
            ax_clean.axvline(0, color='black', lw=0.8, ls='--')
            ax_clean.axvspan(250, 500, color=DECOMPOSITION_COLORS['p300_span'], alpha=0.6)
            ax_clean.set_title(f'{ch_name} · V7 去噪后信号', fontsize=10.5, fontweight='bold', pad=4)
            ax_clean.grid(True, alpha=0.25, ls=':')
            if ch == 0:
                ax_clean.set_ylabel('V7 纯净信号\n(电位单位)', fontsize=9.5, fontweight='bold')
            if row_offset == 0 and ch == 0:
                ax_clean.legend(loc='upper right', fontsize=8, framealpha=0.9)

            # --- Row 3: 剥离的纯伪影分量 ---
            ax_art = axes[row_offset + 2, ch]
            ax_art.plot(times, artifact[ch], color=DECOMPOSITION_COLORS['artifact'], lw=1.1, label='提取伪影 (Raw - V7)')
            ax_art.axvline(0, color='black', lw=0.8, ls='--')
            ax_art.axvspan(250, 500, color=DECOMPOSITION_COLORS['p300_span'], alpha=0.6)
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

    DECOMPOSITION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / dataset_folder_name / filename
    fig.savefig(out_file, dpi=200)
    print(f"成功生成并写入 eeg_v7_results: {out_file}")

    # 同时在 output/三栏伪影分解对比图 保存带完整命名的副本以方便集中查阅
    backup_dir = ROOT / 'output/三栏伪影分解对比图'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / f"{dataset_folder_name}_{filename}"
    fig.savefig(backup_file, dpi=200)
    plt.close(fig)


def plot_outcome_dashboard(out,datasets,group_intervals):
    """三个独立坐标上的点与分方向试次重采样区间，便于在A4页面阅读。"""
    frame=pd.DataFrame(group_intervals)
    names=[item['name'] for item in datasets.values()]
    metrics=(('proxy_MAE_reduction_pct','代理 MAE 降幅','% 相对预处理',0,(-8,24)),
             ('SNR_proxy_gain_dB','SNR 代理增量','dB 相对预处理',0,(-1.2,1.8)),
             ('left_right_retention_ratio','左右差分幅度比','比值，相对预处理',1,(0,1.45)))
    fig,axes=plt.subplots(3,1,figsize=(8.5,8.2),layout='constrained')
    for ax,(metric,title,xlabel,baseline,limits) in zip(axes,metrics):
        ax.axvline(baseline,color='#657480',ls='--',lw=1,zorder=0)
        for i,ds in enumerate(datasets):
            for stage,offset,marker in (('V6',-.13,'s'),('V7',.13,'o')):
                row=frame[(frame.dataset==ds)&(frame.stage==stage)&(frame.metric==metric)].iloc[0]
                y=i+offset
                ax.errorbar(row.point,y,xerr=[[row.point-row.low],[row.high-row.point]],
                            fmt=marker,color=COLORS[stage],markersize=6,capsize=3,
                            elinewidth=1.7,label=stage if i==0 else None,zorder=3)
        ax.set_title(title,fontsize=12,pad=10)
        ax.set_xlabel(xlabel,fontsize=10)
        ax.set_xlim(*limits)
        ax.set_yticks(range(len(names)),names)
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=.17)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='lower right',frameon=False,ncol=2,fontsize=9)
    fig.suptitle('四组数据：V6 / V7 点估计与95%试次重采样区间\n'
                 '模型及训练代理参考固定；区间仅表示本组试次抽样波动',fontsize=13)
    save_fig(fig,out/'汇总与说明/去噪与特征保留联合评价.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'eeg_v7_results')
    parser.add_argument('--no-plots',action='store_true',help='生成数值副本供重复性核验')
    args=parser.parse_args()
    out=args.output.resolve()
    if out==ROOT or any(out.is_relative_to(p) for p in (ROOT/'data',ROOT/'eeg_v2_results',ROOT/'eeg_v3_results',ROOT/'eeg_v4_results',ROOT/'eeg_v5_results',ROOT/'eeg_v6_results')):
        raise ValueError('输出路径不得覆盖原始数据与旧结果')
    out.mkdir(parents=True,exist_ok=True)
    summary_dir=out/'汇总与说明';summary_dir.mkdir(exist_ok=True)
    synth_dir=out/'半合成验证';synth_dir.mkdir(exist_ok=True)
    manifest={'inputs':{},'source':{Path(__file__).name:sha(Path(__file__))},
              'python':platform.python_version(),'numpy':np.__version__,'fold_seed':42,'test_seeds':[2027,2039,2053],
              'unit':'原始电位单位','V7_policy':{'保守双分量':'训练均值保护', '其他候选':'保守比例'},
              'fractions':{'V6':.15,'V7_protected_trial':.25,'V7_protected_train_mean':.10}}
    datasets={};summary=[];spatial=[];synthetic=[];fits=[];strategies=[];slope_diagnostics=[]
    group_intervals=[];stability=[]
    for subject in 'AB':
        for task in (1,2):
            ds=f'VisualCog{subject}_Task-{task}'
            name=f'受试者{subject}_项目'+('一' if task==1 else '二')
            input_path=ROOT/'data'/(ds+'.mat')
            manifest['inputs'][input_path.name]=sha(input_path)
            dest=out/name;dest.mkdir(exist_ok=True)
            raw,x,cues,scores,audit,sensitivity=read_dataset(input_path)
            trial_ids=audit.loc[audit.role=='折外评价','trial_id'].to_numpy(int)
            n=len(x);v3=np.empty_like(x);v4=np.empty_like(x);v6=np.empty_like(x);v7=np.empty_like(x);refs_trial=np.empty_like(x)
            folds=np.zeros(n,dtype=int);reference_counts=[];fold_roles=[]
            fold_mean_deltas=np.empty((5,2,3,len(TIMES)),dtype=float)
            cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
            for fold,(train,test) in enumerate(cv.split(x,cues),1):
                train_scores=detect_bad_trials(raw[train],x[train])[1]
                model=build_clean_reference_model(x[train],cues[train],np.zeros(len(train),bool),train_scores)
                v4_choice,_=select_candidate(x[train],cues[train],100+fold)
                mean_delta=training_mean_delta(x[train],cues[train],model,v4_choice)
                policy=selected_policy(v4_choice)
                median_slope=float(np.median(prestim_slopes(x[train])))
                fold_mean_deltas[fold-1,0]=mean_delta[-1]
                fold_mean_deltas[fold-1,1]=mean_delta[1]
                for j in test:
                    refs_trial[j]=model[0][cues[j]]
                    fold_roles.append(dict(fold=fold,trial_id=int(trial_ids[j]),role='折外评价'))
                reference_ids=set(trial_ids[train[model[3]]])
                assert not reference_ids.intersection(trial_ids[test])
                reference_counts.append(len(reference_ids))
                for j in train:
                    fold_roles.append(dict(fold=fold,trial_id=int(trial_ids[j]),role='训练参考' if trial_ids[j] in reference_ids else '训练非参考'))
                v3[test]=apply_v3(x[test],cues[test],model)
                v4[test]=apply_v4(x[test],cues[test],model,v4_choice)
                v6[test]=conservative_blend(v4[test],x[test],.15)
                v7[test]=apply_v7(x[test],cues[test],model,v4_choice,mean_delta)
                folds[test]=fold
                strategies.append(dict(dataset=ds,fold=fold,train_n=len(train),test_n=len(test),
                                       train_median_pre_slope=median_slope,V4_candidate=v4_choice,
                                       V7_policy=policy,trial_fraction=.25 if policy=='训练均值保护' else .15,
                                       mean_fraction=.10 if policy=='训练均值保护' else 0.,
                                       reference_count=len(reference_ids)))
                # 同试次、同注入、同训练折，复用训练折均值修正，不拟合测试试次ERP。
                for seed in (2027,2039,2053):
                    for level in (0.,.5,1.,2.):
                        contaminated=inject(x[test],cues[test],seed+fold,level)
                        s3=apply_v3(contaminated,cues[test],model)
                        s4=apply_v4(contaminated,cues[test],model,v4_choice)
                        s6=conservative_blend(s4,contaminated,.15)
                        s7=apply_v7(contaminated,cues[test],model,v4_choice,mean_delta)
                        for stage,recovered in zip(STAGES,(contaminated,s3,s4,s6,s7)):
                            result=known_errors(x[test],recovered,cues[test])
                            synthetic.append(dict(dataset=ds,fold=fold,seed=seed,level=level,stage=stage,
                                                  n_trials=len(test),V7_policy=policy,**result))
            assert (folds>0).all()
            audit.loc[audit.role=='折外评价','fold']=folds
            save_csv(audit,dest/'完整事件与试次审计.csv')
            save_csv(fold_roles,dest/'逐折训练参考评价清单.csv')
            save_csv(sensitivity,dest/'坏试次阈值敏感性.csv')
            refs={c:refs_trial[cues==c].mean(axis=0) for c in (-1,1)}
            stages=dict(zip(STAGES,(x,v3,v4,v6,v7)))
            metrics=extended_metrics(x,stages,cues,refs)
            save_csv(metrics,dest/'逐方向逐通道评价指标.csv')
            summary.extend(summarize_metrics(metrics,ds))
            save_csv(bootstrap_metrics(stages,cues,refs),dest/'试次重采样区间.csv')
            group_intervals.extend(dict(dataset=ds,**row) for row in
                                   group_uncertainty(stages,cues,refs,seed=20260924+len(datasets)))
            stability.extend(dict(dataset=ds,**row) for row in
                             temporal_contrast_stability(stages,cues,trial_ids))
            for stage,y in stages.items():
                spatial.extend([dict(stage=stage,**z) for z in evaluate_spatial_metrics(x,y,cues,ds)])
                for c in (-1,1):
                    for ch,channel in enumerate(CHANNELS):
                        early=y[cues==c,ch][:,(TIMES>=0)&(TIMES<200)].mean()
                        late=y[cues==c,ch][:,(TIMES>=600)&(TIMES<800)].mean()
                        slope_diagnostics.append(dict(dataset=ds,stage=stage,cue=c,channel=channel,
                                                      n_trials=int((cues==c).sum()),late_minus_early=late-early,
                                                      mean_pre_slope=float(prestim_slopes(y)[cues==c,ch].mean())))
            np.savez_compressed(dest/'可复核波形.npz',raw=raw,before=x,v3=v3,v4=v4,v6=v6,v7=v7,cues=cues,
                                trial_ids=trial_ids,folds=folds,fold_mean_deltas=fold_mean_deltas,
                                reference_per_trial=refs_trial,times_ms=TIMES)
            save_csv(fixed_example_ids(cues,scores,trial_ids),dest/'固定示例试次编号.csv')
            dataset=dict(name=name,audit=audit,cues=cues,stages=stages,metrics=metrics,reference_counts=reference_counts)
            datasets[ds]=dataset
            if not args.no_plots:
                shown={stage:stages[stage] for stage in PLOT_STAGES}
                plot_condition(name,shown,cues,refs,dest)
                plot_condition(name,shown,cues,refs,dest,zoom=True)
                plot_examples(name,raw,shown,cues,scores,trial_ids,dest,.50,'中位伪影试次_全幅对照.png')
                plot_examples(name,raw,shown,cues,scores,trial_ids,dest,.75,'较高伪影试次_全幅对照.png')
                plot_heatmap(name,shown,cues,trial_ids,dest,focus=False)
                plot_heatmap(name,shown,cues,trial_ids,dest,focus=True)
                plot_spatial(name,shown,cues,dest)
                fits.extend([dict(dataset=ds,**z) for z in plot_fits(name,stages,cues,dest)])
            else:
                local_fits=[]
                for c in (-1,1):
                    for ch,channel in enumerate(CHANNELS):
                        _,fit=spline_fit(stages['V7'][cues==c,ch].mean(axis=0))
                        local_fits.append(dict(cue=c,channel=channel,n_trials=int((cues==c).sum()),**fit))
                save_csv(local_fits,dest/'分方向样条拟合参数.csv')
                fits.extend([dict(dataset=ds,**z) for z in local_fits])
            print(f'{name}：{len(audit)}事件，{n}可用；V7各折策略 '+', '.join(r['V7_policy'] for r in strategies[-5:]),flush=True)
    save_csv(summary,summary_dir/'五阶段共同配对指标汇总.csv')
    save_csv(spatial,summary_dir/'左右刺激与额区空间差异指标.csv')
    save_csv(strategies,summary_dir/'逐折V7策略与参考数量.csv')
    save_csv(slope_diagnostics,summary_dir/'刺激前斜率与刺激后慢变化.csv')
    save_csv(group_intervals,summary_dir/'四组核心指标重采样区间.csv')
    save_csv(stability,summary_dir/'左右差分前后时段稳定性.csv')
    save_csv(synthetic,synth_dir/'多场景逐折配对验证.csv')
    bench=pd.DataFrame(synthetic)
    save_csv(bench.groupby(['dataset','level','stage'],sort=False)[['RMSE','normalized_RMSE','contrast_RMSE','positive_mean_error','latency_error_ms']].mean().reset_index(),
             synth_dir/'按数据组和幅度汇总.csv')
    if fits:save_csv(fits,summary_dir/'分方向样条拟合参数汇总.csv')
    if not args.no_plots:
        plot_cross_task(out,datasets)
        plot_benchmark(out,synthetic)
        plot_outcome_dashboard(out,datasets,group_intervals)
        with plt.rc_context({'font.sans-serif': ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'Noto Sans SC', 'SimHei', 'DejaVu Sans'],
                             'axes.unicode_minus': False}):
            for item in datasets.values():
                plot_decomposition_for_dataset(out,item['name'],.75,'典型较高伪影试次 (75% 分位)',
                                               '较高伪影试次_三栏分解图.png')
                plot_decomposition_for_dataset(out,item['name'],.50,'中位伪影试次 (50% 分位)',
                                               '中位伪影试次_三栏分解图.png')
    write_report(out,datasets,summary,spatial,synthetic,fits,group_intervals,stability)
    manifest['csv_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*.csv'))}
    (summary_dir/'运行清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('完成：'+str(out),flush=True)


if __name__=='__main__':main()
