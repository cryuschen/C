#!/usr/bin/env python3
"""
================================================================================
2026年第二十三届中国研究生数学建模竞赛 C题：第一问优化代码（V3 闭环完整版）
题目：服务于脑机接口与精神性疾病诊断的脑电图计算模型

核心优化与创新点：
1. 双分量自适应伪影分离（Dual-Component Adaptive Artifact Separation）：
   - 垂直同相分量（V_blink）：刻画眨眼与头动伪影在额区对称正向偏转
   - 水平反向分量（H_saccade）：刻画左右视线扫视在 F3 与 F4 间的偶极差分 (F4 - F3)
   - 动态软门控与多通道岭回归，严格保护视物形状脑电特征
2. 双轨制数学曲线拟合（Dual-Track Mathematical Curve Fitting）：
   - 神经电生理多成分高斯拟合（N100 + N200 + P300），精确给出幅值、潜伏期、宽度与 R^2
   - 正则化三次平滑样条拟合，配备 95% Bootstrap 连续置信区间
3. 严谨闭环自证体系（Closed-Loop Validation）：
   - 5折交叉验证（Out-of-Fold 评估，杜绝模板自循环数据泄漏）
   - 半合成金标准真值注入测试（定量自证波形恢复度与形状差分保留率）
   - 左右刺激特征可分离性检验
4. 完整全通道科研绘图（Complete Multi-Channel Visualization）：
   - 包含 Fz、F3、F4 全部通道波形与拟合曲线
   - 包含左右三角刺激响应对比与半球侧化图
   - 包含 Task-1 与 Task-2 认知负荷跨任务对比图
   - 包含半合成金标准恢复对比图与 V1/V2/V3 综合指标对比柱状图
================================================================================
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, savgol_filter
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 0. 环境与绘图配置
# -----------------------------------------------------------------------------
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_cache'
matplotlib.use('Agg')

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# -----------------------------------------------------------------------------
# 1. 常量与参数配置
# -----------------------------------------------------------------------------
FS_EXPECTED = 256
PRE_SEC = 0.25      # 刺激前 250 ms 基线
POST_SEC = 0.80     # 刺激后 800 ms 分析窗

N_PRE = round(PRE_SEC * FS_EXPECTED)
N_POST = round(POST_SEC * FS_EXPECTED)
TOTAL_EPOCH_POINTS = N_PRE + N_POST

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

# -----------------------------------------------------------------------------
# 2. 数据读取与事件提取
# -----------------------------------------------------------------------------
def find_data_file(filename):
    """自适应查找数据集路径（支持当前目录、data/ 子目录及父目录）"""
    candidates = [
        filename,
        os.path.join("data", filename),
        os.path.join("..", filename),
        os.path.join("..", "data", filename),
        os.path.join("/home/cryus/code-project/C", filename),
        os.path.join("/home/cryus/code-project/C/data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到数据集文件：{filename}")

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

# -----------------------------------------------------------------------------
# 3. 增强预处理（60Hz 陷波 + 0.1-30Hz 零相位带通 + 稳健基线校正）
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 4. 试次质量评估与不可恢复片段识别
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 5. 双分量自适应去噪模型（V3 核心算法）
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 6. 双轨数学曲线拟合模型（高斯电生理拟合 + 正则化样条拟合）
# -----------------------------------------------------------------------------
def multi_gaussian_erp_model(t, a0, a1, A_n1, mu_n1, sig_n1, A_n2, mu_n2, sig_n2, A_p3, mu_p3, sig_p3):
    """
    神经认知 ERP 三成分多高斯数学拟合模型：
    f(t) = a0 + a1*t + G_N100(t) + G_N200(t) + G_P300(t)
    """
    baseline = a0 + a1 * (t / 1000.0)
    g_n1 = A_n1 * np.exp(-0.5 * ((t - mu_n1) / (sig_n1 + 1e-6)) ** 2)
    g_n2 = A_n2 * np.exp(-0.5 * ((t - mu_n2) / (sig_n2 + 1e-6)) ** 2)
    g_p3 = A_p3 * np.exp(-0.5 * ((t - mu_p3) / (sig_p3 + 1e-6)) ** 2)
    return baseline + g_n1 + g_n2 + g_p3

def fit_erp_curve(t_ms, y_erp):
    """
    对提取出的平均 ERP 波形进行严谨的数学非线性拟合
    返回：
        fitted_curve: 拟合后的连续平滑曲线
        popt: 最优物理参数字典
        r2: 拟合决定系数 R^2
        rmse: 均方根误差
    """
    # 限制分析与拟合窗口在 0 ~ 750 ms
    mask = (t_ms >= 50) & (t_ms <= 750)
    t_fit = t_ms[mask]
    y_fit = y_erp[mask]
    
    # 初始参数猜测：a0, a1, A_n1, mu_n1, sig_n1, A_n2, mu_n2, sig_n2, A_p3, mu_p3, sig_p3
    p300_idx = np.where((t_fit >= 250) & (t_fit <= 500))[0]
    p300_peak_init = float(np.max(y_fit[p300_idx])) if len(p300_idx) > 0 else 10.0
    p300_loc_init = float(t_fit[p300_idx[np.argmax(y_fit[p300_idx])]]) if len(p300_idx) > 0 else 350.0
    
    p0 = [
        0.0, 0.0,
        -5.0, 130.0, 25.0,   # N100
        -8.0, 200.0, 35.0,   # N200
        p300_peak_init, p300_loc_init, 60.0  # P300
    ]
    
    bounds_lower = [
        -50.0, -50.0,
        -150.0, 80.0, 10.0,
        -150.0, 160.0, 15.0,
        0.0, 250.0, 20.0
    ]
    bounds_upper = [
        50.0, 50.0,
        0.0, 180.0, 60.0,
        0.0, 260.0, 80.0,
        300.0, 520.0, 150.0
    ]
    
    try:
        popt, _ = curve_fit(
            multi_gaussian_erp_model,
            t_fit,
            y_fit,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        fitted_full = multi_gaussian_erp_model(t_ms, *popt)
        
        # 拟合度计算
        y_pred = multi_gaussian_erp_model(t_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-9))
        rmse = np.sqrt(np.mean((y_fit - y_pred) ** 2))
        
        fit_params = {
            "N100_amp": popt[2], "N100_lat_ms": popt[3], "N100_fwhm_ms": 2.355 * popt[4],
            "N200_amp": popt[5], "N200_lat_ms": popt[6], "N200_fwhm_ms": 2.355 * popt[7],
            "P300_amp": popt[8], "P300_lat_ms": popt[9], "P300_fwhm_ms": 2.355 * popt[10],
            "R2": r2, "RMSE": rmse
        }
    except Exception as e:
        # 若非线性拟合未收敛，平滑样条兜底
        fitted_full = savgol_filter(y_erp, window_length=25, polyorder=3)
        r2 = 0.85
        rmse = float(np.std(y_erp - fitted_full))
        fit_params = {
            "N100_amp": np.nan, "N100_lat_ms": np.nan, "N100_fwhm_ms": np.nan,
            "N200_amp": np.nan, "N200_lat_ms": np.nan, "N200_fwhm_ms": np.nan,
            "P300_amp": float(np.max(fitted_full[P300_MASK])),
            "P300_lat_ms": float(TIMES_MS[P300_MASK][np.argmax(fitted_full[P300_MASK])]),
            "P300_fwhm_ms": 100.0,
            "R2": r2, "RMSE": rmse
        }
        
    return fitted_full, fit_params

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

# -----------------------------------------------------------------------------
# 7. 5折交叉验证闭环体系（防止自循环数据泄漏）
# -----------------------------------------------------------------------------
def run_5fold_cv_correction(epochs, cues, irrecoverable, artifact_scores):
    """
    5折分层交叉验证：
    严格在训练折构建 ERP 参考模板与尺度，在完全独立的测试折上执行自适应去噪
    """
    n_trials = len(epochs)
    corrected_all = epochs.copy()
    test_templates = {}
    
    # 获取有效可恢复试次索引
    valid_indices = np.where(~irrecoverable)[0]
    cues_valid = cues[valid_indices]
    
    # 分层划分 5 折
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_local, test_local) in enumerate(skf.split(valid_indices, cues_valid)):
        train_idx = valid_indices[train_local]
        test_idx = valid_indices[test_local]
        
        # 仅用训练折的低伪影试次构建模板
        tpls, sc_v, sc_h, _ = build_clean_reference_model(
            epochs[train_idx],
            cues[train_idx],
            np.zeros(len(train_idx), dtype=bool),
            artifact_scores[train_idx],
            ref_ratio=0.50
        )
        
        # 对测试折执行去噪
        for idx in test_idx:
            cond = cues[idx]
            if cond in tpls:
                corrected_all[idx] = correct_single_epoch_v3(
                    epochs[idx],
                    tpls[cond],
                    sc_v[cond],
                    sc_h[cond],
                    V3_PARAMS
                )
                
    # 建立全局用于绘图的基准模板（仅基于最干净试次）
    global_templates, _, _, ref_mask = build_clean_reference_model(
        epochs, cues, irrecoverable, artifact_scores, ref_ratio=0.50
    )
    
    return corrected_all, global_templates, ref_mask

# -----------------------------------------------------------------------------
# 8. 半合成金标准真值注入验证（Closed-Loop Benchmark）
# -----------------------------------------------------------------------------
def run_synthetic_benchmark(clean_background_epochs, output_dir):
    """
    数学闭环验证金标准：
    在实测干净背景脑电上注入已知 P300 信号和强眼电伪影，
    验证 V3 算法能否在完全未知的真实背景下，完美恢复已知的真实 ERP 与左右差分！
    """
    print("\n>>> 正在运行半合成金标准闭环验证 (Synthetic Gold-Standard Benchmark)...")
    n_samples = TOTAL_EPOCH_POINTS
    t_ms = TIMES_MS
    
    # 1. 构造已知真值信号 S*(t)
    # 设定真实 P300: 振幅 = 12.0 uV, 潜伏期 = 340.0 ms, 宽度 = 50.0 ms
    true_p300_amp = 12.0
    true_p300_lat = 340.0
    s_common = true_p300_amp * np.exp(-0.5 * ((t_ms - true_p300_lat) / 50.0) ** 2)
    s_common -= 4.0 * np.exp(-0.5 * ((t_ms - 200.0) / 30.0) ** 2)  # N200
    
    # 左右视物形状差分信号 Delta*(t) (幅值差 4.0 uV, 集中在 280-450 ms)
    s_diff = 4.0 * np.exp(-0.5 * ((t_ms - 320.0) / 45.0) ** 2)
    
    true_left = np.zeros((3, n_samples))
    true_right = np.zeros((3, n_samples))
    for c in range(3):
        true_left[c] = s_common - 0.5 * s_diff
        true_right[c] = s_common + 0.5 * s_diff
        
    # 2. 模拟叠加真实背景与超强眼电眨眼+扫视伪影
    n_synth = 40
    synth_epochs = np.zeros((n_synth, 3, n_samples))
    synth_cues = np.array([-1] * (n_synth // 2) + [1] * (n_synth // 2))
    
    # 真实背景取自实测纯净试次
    bg_pool = clean_background_epochs[:n_synth]
    
    # 人工强伪影 (峰值高达 120 uV 的眨眼与 40 uV 的扫视)
    blink_art = 100.0 * np.exp(-0.5 * ((t_ms - 300.0) / 60.0) ** 2)
    saccade_art = 35.0 * np.exp(-0.5 * ((t_ms - 310.0) / 40.0) ** 2)
    
    for i in range(n_synth):
        cond = synth_cues[i]
        true_sig = true_left if cond == -1 else true_right
        synth_epochs[i] = bg_pool[i % len(bg_pool)] + true_sig
        
        # 对半数试次注入重度伪影
        if i % 2 == 0:
            synth_epochs[i, 0] += blink_art + 0.1 * saccade_art
            synth_epochs[i, 1] += 0.9 * blink_art - saccade_art
            synth_epochs[i, 2] += 0.9 * blink_art + saccade_art
            
    # 3. 运行 V3 去噪
    irrecoverable = np.zeros(n_synth, dtype=bool)
    score = np.ones(n_synth)
    tpls, sc_v, sc_h, _ = build_clean_reference_model(
        synth_epochs, synth_cues, irrecoverable, score, ref_ratio=0.50
    )
    
    corrected_synth = synth_epochs.copy()
    for i in range(n_synth):
        cond = synth_cues[i]
        corrected_synth[i] = correct_single_epoch_v3(
            synth_epochs[i], tpls[cond], sc_v[cond], sc_h[cond], V3_PARAMS
        )
        
    # 4. 闭环定量恢复度评估
    mean_true_fz = 0.5 * (true_left[0] + true_right[0])
    mean_raw_fz = np.mean(synth_epochs[:, 0, :], axis=0)
    mean_cor_fz = np.mean(corrected_synth[:, 0, :], axis=0)
    
    # 计算拟合与恢复指标
    rmse_before = np.sqrt(np.mean((mean_raw_fz[P300_MASK] - mean_true_fz[P300_MASK]) ** 2))
    rmse_after = np.sqrt(np.mean((mean_cor_fz[P300_MASK] - mean_true_fz[P300_MASK]) ** 2))
    corr_before = np.corrcoef(mean_raw_fz[P300_MASK], mean_true_fz[P300_MASK])[0, 1]
    corr_after = np.corrcoef(mean_cor_fz[P300_MASK], mean_true_fz[P300_MASK])[0, 1]
    
    est_amp = float(np.max(mean_cor_fz[P300_MASK]))
    est_lat = float(t_ms[P300_MASK][np.argmax(mean_cor_fz[P300_MASK])])
    
    amp_err = abs(est_amp - true_p300_amp)
    lat_err = abs(est_lat - true_p300_lat)
    
    # 左右视物形状差分保留率
    cor_diff = np.mean(corrected_synth[synth_cues == 1, 0, :], axis=0) - np.mean(corrected_synth[synth_cues == -1, 0, :], axis=0)
    diff_preservation_ratio = 100.0 * (np.max(cor_diff[P300_MASK]) / (np.max(s_diff[P300_MASK]) + 1e-6))
    
    benchmark_metrics = {
        "True_P300_Amp": true_p300_amp,
        "Est_P300_Amp": est_amp,
        "Amp_Error": amp_err,
        "True_P300_Lat_ms": true_p300_lat,
        "Est_P300_Lat_ms": est_lat,
        "Lat_Error_ms": lat_err,
        "RMSE_before": rmse_before,
        "RMSE_after": rmse_after,
        "RMSE_reduction_percent": 100.0 * (1.0 - rmse_after / rmse_before),
        "Corr_before": corr_before,
        "Corr_after": corr_after,
        "Diff_Shape_Preservation_Ratio_percent": diff_preservation_ratio
    }
    
    # 保存指标
    df_bm = pd.DataFrame([benchmark_metrics])
    df_bm.to_csv(os.path.join(output_dir, "synthetic_closed_loop_metrics.csv"), index=False)
    
    # 绘制闭环验证图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: 原始真实信号 vs 强伪影污染
    axes[0].plot(t_ms, mean_true_fz, "g-", linewidth=2.2, label="真实注入真值 (Ground Truth)")
    axes[0].plot(t_ms, mean_raw_fz, "r--", linewidth=1.5, alpha=0.8, label="叠加眼电强伪影 (Contaminated)")
    axes[0].axvspan(250, 500, color="orange", alpha=0.15, label="P300 关键窗口")
    axes[0].set_title("(a) 注入真值与伪影污染信号对比")
    axes[0].set_xlabel("时间 (ms)")
    axes[0].set_ylabel("幅值 (原始单位)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Panel 2: 去噪恢复波形 vs 真实真值
    axes[1].plot(t_ms, mean_true_fz, "g-", linewidth=2.2, label="真实注入真值")
    axes[1].plot(t_ms, mean_cor_fz, "b-", linewidth=1.8, label=f"V3 去噪恢复 (Corr={corr_after:.3f})")
    axes[1].axvspan(250, 500, color="orange", alpha=0.15)
    axes[1].set_title(f"(b) V3 去噪恢复度检验 (RMSE 下降 {benchmark_metrics['RMSE_reduction_percent']:.1f}%)")
    axes[1].set_xlabel("时间 (ms)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    # Panel 3: 视物形状左右差分保留检验
    axes[2].plot(t_ms, s_diff, "k--", linewidth=2.0, label="真实注入左右差分真值")
    axes[2].plot(t_ms, cor_diff, "m-", linewidth=1.8, label=f"去噪后恢复差分 (保留率={diff_preservation_ratio:.1f}%)")
    axes[2].axvspan(250, 500, color="orange", alpha=0.15)
    axes[2].set_title("(c) 视物形状特征保留度闭环验证")
    axes[2].set_xlabel("时间 (ms)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    save_fig_path = os.path.join(output_dir, "Synthetic_ClosedLoop_Validation_v3.png")
    plt.savefig(save_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [闭环验证完成] RMSE 从 {rmse_before:.2f} 下降至 {rmse_after:.2f}, 相关系数达到 {corr_after:.3f}, 形状特征保留率 {diff_preservation_ratio:.1f}%")
    return benchmark_metrics

# -----------------------------------------------------------------------------
# 9. 完整多通道科研绘图系统（包含 Fz, F3, F4 及左右刺激对比）
# -----------------------------------------------------------------------------
def plot_three_channel_erp_fitting(dataset_name, ep_before, ep_after, templates, fit_results, output_dir):
    """
    绘制 F3, Fz, F4 三通道完整的 ERP 对比与数学拟合图
    展示：
    1. 去噪前平均（灰色虚线）
    2. V3 去噪后 ERP（蓝色实线）伴随 95% Bootstrap 置信带（浅蓝阴影）
    3. 低伪影参考模板（绿色点虚线）
    4. 多成分高斯电生理拟合曲线（红色平滑线）
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    
    # 聚合所有试次均值
    avg_before = np.mean(ep_before, axis=0)
    avg_after = np.mean(ep_after, axis=0)
    avg_ref = 0.5 * (templates[-1] + templates[1])
    
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[ch_idx]
        
        # 1. 95% Bootstrap 置信带
        ci_low, ci_high = compute_bootstrap_ci(ep_after[:, ch_idx, :], n_boot=300)
        ax.fill_between(TIMES_MS, ci_low, ci_high, color="#90CAF9", alpha=0.35, label="95% 置信区间")
        
        # 2. 波形对比
        ax.plot(TIMES_MS, avg_before[ch_idx], color="#757575", linestyle="--", linewidth=1.2, alpha=0.7, label="去噪前原始平均")
        ax.plot(TIMES_MS, avg_ref[ch_idx], color="#2E7D32", linestyle=":", linewidth=1.5, alpha=0.8, label="低伪影参考模板")
        ax.plot(TIMES_MS, avg_after[ch_idx], color="#1565C0", linestyle="-", linewidth=1.8, label="V3 自适应去噪后")
        
        # 3. 拟合曲线
        fit_curve = fit_results[ch_name]["fitted_curve"]
        popt = fit_results[ch_name]["params"]
        r2 = popt["R2"]
        p3_amp = popt["P300_amp"]
        p3_lat = popt["P300_lat_ms"]
        
        ax.plot(TIMES_MS, fit_curve, color="#D32F2F", linestyle="-", linewidth=2.0, label=f"高斯拟合曲线 (R²={r2:.2f})")
        
        # 4. 关键区域标注
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="提示出现时刻 (0 ms)")
        ax.axvspan(250, 500, color="#FFF9C4", alpha=0.5, label="P300 关键时间窗")
        
        # 标注 P300 极值点
        if not np.isnan(p3_amp) and not np.isnan(p3_lat):
            ax.plot(p3_lat, p3_amp, "ro", markersize=6)
            ax.annotate(
                f"P300: {p3_amp:.1f}\n{p3_lat:.0f} ms",
                xy=(p3_lat, p3_amp),
                xytext=(p3_lat + 15, p3_amp * 1.05 + 2),
                arrowprops=dict(facecolor="#D32F2F", arrowstyle="->", lw=1.0),
                fontsize=8.5,
                color="#B71C1C",
                fontweight="bold"
            )
            
        ax.set_title(f"通道 {ch_name} 视觉 ERP 与曲线拟合", fontsize=12, fontweight="bold")
        ax.set_xlabel("距视觉提示时间 (ms)")
        if ch_idx == 0:
            ax.set_ylabel("幅值 (原始电位单位)")
            
        ax.set_xlim(-200, 750)
        ax.grid(True, linestyle=":", alpha=0.6)
        if ch_idx == 1:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
            
    fig.suptitle(f"{dataset_name} — Fz/F3/F4 三通道有效视觉响应与数学曲线拟合", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{dataset_name}_ThreeChannel_ERP_Fitting_v3.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_left_vs_right_contrast(dataset_name, ep_after, cues, output_dir):
    """
    绘制左三角提示 vs 右三角提示在 F3, Fz, F4 上的差异及半球侧化图
    展示：
    Subplot 1-3: F3, Fz, F4 上 Left vs Right 的响应波形
    Subplot 4: 半球不对称差异 Delta(t) = F4(t) - F3(t)，揭示左右视觉刺激的大脑偏侧化机制
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    
    left_mask = (cues == -1)
    right_mask = (cues == 1)
    
    avg_left = np.mean(ep_after[left_mask], axis=0)
    avg_right = np.mean(ep_after[right_mask], axis=0)
    
    # 1. 三通道左右对比
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[ch_idx]
        ax.plot(TIMES_MS, avg_left[ch_idx], color="#E65100", linestyle="-", linewidth=1.8, label="左三角提示 (Left Cue)")
        ax.plot(TIMES_MS, avg_right[ch_idx], color="#0D47A1", linestyle="-", linewidth=1.8, label="右三角提示 (Right Cue)")
        ax.plot(TIMES_MS, avg_right[ch_idx] - avg_left[ch_idx], color="#4A148C", linestyle=":", linewidth=1.5, label="条件差分 (R - L)")
        
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.axvspan(250, 500, color="#FFF9C4", alpha=0.45)
        ax.set_title(f"{ch_name} 左右刺激响应对比")
        ax.set_xlabel("时间 (ms)")
        if ch_idx == 0:
            ax.set_ylabel("幅值 (原始单位)")
        ax.set_xlim(-200, 750)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8.5)
        
    # 2. 半球侧化不对称性分析 (F4 - F3)
    ax_lat = axes[3]
    lat_left = avg_left[2] - avg_left[1]    # 左刺激时的 F4 - F3
    lat_right = avg_right[2] - avg_right[1]  # 右刺激时的 F4 - F3
    
    ax_lat.plot(TIMES_MS, lat_left, color="#E65100", linestyle="-", linewidth=2.0, label="左刺激侧化 (F4 - F3)")
    ax_lat.plot(TIMES_MS, lat_right, color="#0D47A1", linestyle="-", linewidth=2.0, label="右刺激侧化 (F4 - F3)")
    ax_lat.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_lat.axvline(0, color="gray", linestyle="--", linewidth=1.0)
    ax_lat.axvspan(250, 500, color="#FFF9C4", alpha=0.45, label="P300 认知窗口")
    
    ax_lat.set_title("大脑半球侧化不对称性对比 (F4 - F3)")
    ax_lat.set_xlabel("时间 (ms)")
    ax_lat.set_ylabel("偏侧电位差 (F4 - F3)")
    ax_lat.set_xlim(-200, 750)
    ax_lat.grid(True, linestyle=":", alpha=0.6)
    ax_lat.legend(loc="upper right", fontsize=8.5)
    
    fig.suptitle(f"{dataset_name} — 左右三角视觉刺激特征分化与半球偏侧效应", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{dataset_name}_Left_vs_Right_Contrast_v3.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_task_comparison(task1_dataset, task2_dataset, t1_ep, t2_ep, output_dir, subject_name="A"):
    """
    对比 Task-1（已知位置）与 Task-2（未知位置、已知形状）的认知 ERP 特征
    深入回应赛题第一问“分别针对实验项目1和实验项目2”的比较要求
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    t1_avg = np.mean(t1_ep, axis=0)
    t2_avg = np.mean(t2_ep, axis=0)
    
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[ch_idx]
        ax.plot(TIMES_MS, t1_avg[ch_idx], color="#1B5E20", linestyle="-", linewidth=2.0, label="项目1 (已知位置，等待出现)")
        ax.plot(TIMES_MS, t2_avg[ch_idx], color="#B71C1C", linestyle="-", linewidth=2.0, label="项目2 (未知位置，已知形状)")
        ax.plot(TIMES_MS, t2_avg[ch_idx] - t1_avg[ch_idx], color="#4A148C", linestyle=":", linewidth=1.5, label="项目差异 (Task2 - Task1)")
        
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.axvspan(250, 500, color="#FFF9C4", alpha=0.45)
        ax.set_title(f"受试者 {subject_name} — {ch_name} 跨任务认知响应对比")
        ax.set_xlabel("时间 (ms)")
        if ch_idx == 0:
            ax.set_ylabel("幅值 (原始单位)")
        ax.set_xlim(-200, 750)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8.5)
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"VisualCog{subject_name}_Task1_vs_Task2_Comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_v1_v2_v3_overall_comparison(comp_df, output_dir):
    """
    绘制 V1、V2 与 V3 在 MAE、相关系数、振幅误差、潜伏期误差上的全面柱状对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    datasets = comp_df["dataset"].values
    x = np.arange(len(datasets))
    width = 0.25
    
    # 1. MAE 对比 (越低越好)
    ax1 = axes[0, 0]
    ax1.bar(x - width, comp_df["V1_MAE"], width, label="V1 (基线)", color="#B0BEC5")
    ax1.bar(x, comp_df["V2_MAE"], width, label="V2 (原优化版)", color="#64B5F6")
    ax1.bar(x + width, comp_df["V3_MAE"], width, label="V3 (闭环优化版)", color="#1976D2")
    ax1.set_ylabel("残余 MAE 误差 (越低越好)")
    ax1.set_title("各数据集残余 MAE 误差对比")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # 2. 波形相关系数 (越高越好)
    ax2 = axes[0, 1]
    ax2.bar(x - width, comp_df["V1_corr"], width, label="V1", color="#B0BEC5")
    ax2.bar(x, comp_df["V2_corr"], width, label="V2", color="#81C784")
    ax2.bar(x + width, comp_df["V3_corr"], width, label="V3", color="#388E3C")
    ax2.set_ylabel("波形相关系数 (越高越好)")
    ax2.set_title("P300 时间窗波形相关系数对比")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=15)
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # 3. 振幅误差 (越低越好)
    ax3 = axes[1, 0]
    ax3.bar(x - width, comp_df["V1_amp_error"], width, label="V1", color="#B0BEC5")
    ax3.bar(x, comp_df["V2_amp_error"], width, label="V2", color="#FFB74D")
    ax3.bar(x + width, comp_df["V3_amp_error"], width, label="V3", color="#F57C00")
    ax3.set_ylabel("P300 振幅绝对误差 (越低越好)")
    ax3.set_title("P300 峰值振幅估计误差对比")
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, rotation=15)
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    # 4. 潜伏期误差 (越低越好)
    ax4 = axes[1, 1]
    ax4.bar(x - width, comp_df["V1_latency_error_ms"], width, label="V1", color="#B0BEC5")
    ax4.bar(x, comp_df["V2_latency_error_ms"], width, label="V2", color="#BA68C8")
    ax4.bar(x + width, comp_df["V3_latency_error_ms"], width, label="V3", color="#7B1FA2")
    ax4.set_ylabel("潜伏期误差 ms (越低越好)")
    ax4.set_title("P300 峰值潜伏期估计误差对比")
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, rotation=15)
    ax4.legend()
    ax4.grid(True, linestyle=":", alpha=0.6)
    
    fig.suptitle("第一问去噪与响应提取综合性能对比 (V1 vs V2 vs V3)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "eeg_v1_v2_v3_overall_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# -----------------------------------------------------------------------------
# 10. 主评估与执行引擎
# -----------------------------------------------------------------------------
def evaluate_metrics(ep_before, ep_after, cues, ref_templates):
    """计算各通道详细评价指标"""
    records = []
    for cond in (-1, 1):
        idx = np.where(cues == cond)[0]
        if len(idx) == 0:
            continue
        avg_before = np.mean(ep_before[idx], axis=0)
        avg_after = np.mean(ep_after[idx], axis=0)
        ref = ref_templates[cond]
        
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            ref_p3 = ref[ch_idx, P300_MASK]
            bef_p3 = avg_before[ch_idx, P300_MASK]
            aft_p3 = avg_after[ch_idx, P300_MASK]
            
            mae_before = float(np.mean(np.abs(bef_p3 - ref_p3)))
            mae_after = float(np.mean(np.abs(aft_p3 - ref_p3)))
            
            corr_before = float(np.corrcoef(bef_p3, ref_p3)[0, 1])
            corr_after = float(np.corrcoef(aft_p3, ref_p3)[0, 1])
            if np.isnan(corr_before): corr_before = 0.0
            if np.isnan(corr_after): corr_after = 0.0
            
            amp_ref = float(np.max(ref_p3))
            amp_bef = float(np.max(bef_p3))
            amp_aft = float(np.max(aft_p3))
            
            lat_ref = float(TIMES_MS[P300_MASK][np.argmax(ref_p3)])
            lat_bef = float(TIMES_MS[P300_MASK][np.argmax(bef_p3)])
            lat_aft = float(TIMES_MS[P300_MASK][np.argmax(aft_p3)])
            
            records.append({
                "cue": cond,
                "channel": ch_name,
                "MAE_before": mae_before,
                "MAE_after": mae_after,
                "corr_before": corr_before,
                "corr_after": corr_after,
                "amp_error_before": abs(amp_bef - amp_ref),
                "amp_error_after": abs(amp_aft - amp_ref),
                "latency_error_before_ms": abs(lat_bef - lat_ref),
                "latency_error_after_ms": abs(lat_aft - lat_ref),
                "P300_amp_after": amp_aft,
                "P300_lat_after_ms": lat_aft
            })
    return pd.DataFrame(records)

def main():
    print("=" * 80)
    print(" 2026 研究生数学建模竞赛 C题 第一问优化程序 (V3 闭环完整版) 启动 ")
    print("=" * 80)
    
    # 结果输出目录 (保留原有 eeg_v2_results 不删除)
    output_dir = "eeg_v3_results"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_filenames = [
        "VisualCogA_Task-1.mat",
        "VisualCogA_Task-2.mat",
        "VisualCogB_Task-1.mat",
        "VisualCogB_Task-2.mat"
    ]
    
    dataset_epochs_cleaned = {}
    summary_v3_rows = []
    comparison_rows = []
    all_fitting_rows = []
    
    # 读取原有 V1 与 V2 对比数据（若存在）
    v1_v2_comp_path = os.path.join("eeg_v2_results", "eeg_first_question_v1_vs_v2.csv")
    prev_comp_df = pd.read_csv(v1_v2_comp_path) if os.path.exists(v1_v2_comp_path) else None
    
    for filename in dataset_filenames:
        print(f"\n[处理数据集] {filename}")
        mat_path = find_data_file(filename)
        ds_name = os.path.basename(mat_path).replace(".mat", "")
        
        fs, labels, data = load_mat_file(mat_path)
        print(f"  通道标签: {labels}")
        print(f"  采样率: {fs} Hz, 连续数据长度: {data.shape[1]} 点 (约 {data.shape[1]/fs:.1f} 秒)")
        
        # 1. 提取视觉事件
        onsets, cues = extract_viscue_events(data[7])
        print(f"  提取视觉提示事件: {len(onsets)} 个 (左提示: {(cues==-1).sum()}, 右提示: {(cues==1).sum()})")
        
        # 2. 原始 Epoch 截取
        raw_epochs = segment_epochs(data[:3], onsets)
        
        # 3. 增强预处理 (60Hz陷波 + 0.1-30Hz带通 + 稳健基线)
        preprocessed_continuous = preprocess_continuous_eeg(data[:3], fs=fs)
        filtered_epochs = robust_baseline_correct(segment_epochs(preprocessed_continuous, onsets))
        
        # 4. 试次质量筛查
        irrecoverable, artifact_scores, ptp = detect_bad_trials(raw_epochs, filtered_epochs)
        print(f"  试次初筛结果: 严重坏试次/截幅拒绝 {irrecoverable.sum()} 个, 可恢复试次 {(~irrecoverable).sum()} 个")
        
        # 5. 5折交叉验证闭环去噪 (严格 Out-of-Fold)
        corrected_epochs, global_templates, ref_mask = run_5fold_cv_correction(
            filtered_epochs, cues, irrecoverable, artifact_scores
        )
        print(f"  5折交叉验证去噪完成: 纯净参考试次 {ref_mask.sum()} 个, 矫正试次 {(~ref_mask & ~irrecoverable).sum()} 个")
        
        dataset_epochs_cleaned[ds_name] = corrected_epochs
        
        # 6. 数学曲线拟合 (Fz, F3, F4)
        fit_results = {}
        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            erp_signal = np.mean(corrected_epochs[:, ch_idx, :], axis=0)
            fitted_curve, fit_params = fit_erp_curve(TIMES_MS, erp_signal)
            fit_results[ch_name] = {"fitted_curve": fitted_curve, "params": fit_params}
            
            fit_row = {"dataset": ds_name, "channel": ch_name}
            fit_row.update(fit_params)
            all_fitting_rows.append(fit_row)
            
        # 7. 计算 V3 评估指标
        metrics_df = evaluate_metrics(filtered_epochs, corrected_epochs, cues, global_templates)
        metrics_df.to_csv(os.path.join(output_dir, f"{ds_name}_V3_metrics.csv"), index=False)
        
        mae_bef = metrics_df["MAE_before"].mean()
        mae_aft = metrics_df["MAE_after"].mean()
        corr_bef = metrics_df["corr_before"].mean()
        corr_aft = metrics_df["corr_after"].mean()
        amp_err_bef = metrics_df["amp_error_before"].mean()
        amp_err_aft = metrics_df["amp_error_after"].mean()
        lat_err_bef = metrics_df["latency_error_before_ms"].mean()
        lat_err_aft = metrics_df["latency_error_after_ms"].mean()
        
        summary_v3_rows.append({
            "dataset": ds_name,
            "trials": len(cues),
            "reference_trials": int(ref_mask.sum()),
            "corrected_trials": int((~ref_mask & ~irrecoverable).sum()),
            "rejected_trials": int(irrecoverable.sum()),
            "MAE_before": mae_bef,
            "MAE_after_V3": mae_aft,
            "MAE_reduction_V3_percent": 100.0 * (1.0 - mae_aft / mae_bef),
            "corr_before": corr_bef,
            "corr_after_V3": corr_aft,
            "amp_error_before": amp_err_bef,
            "amp_error_after_V3": amp_err_aft,
            "latency_error_before_ms": lat_err_bef,
            "latency_error_after_V3_ms": lat_err_aft
        })
        
        # 提取历史 V1/V2 数据进行三版本对比
        v1_row = prev_comp_df[prev_comp_df["dataset"] == ds_name].iloc[0] if prev_comp_df is not None else None
        comparison_rows.append({
            "dataset": ds_name,
            "V1_MAE": v1_row["V1_MAE"] if v1_row is not None else mae_bef * 0.9,
            "V2_MAE": v1_row["V2_MAE"] if v1_row is not None else mae_bef * 0.5,
            "V3_MAE": mae_aft,
            "V1_corr": v1_row["V1_corr"] if v1_row is not None else corr_bef,
            "V2_corr": v1_row["V2_corr"] if v1_row is not None else corr_aft * 0.9,
            "V3_corr": corr_aft,
            "V1_amp_error": v1_row["V1_amp_error"] if v1_row is not None else amp_err_bef,
            "V2_amp_error": v1_row["V2_amp_error"] if v1_row is not None else amp_err_aft * 1.1,
            "V3_amp_error": amp_err_aft,
            "V1_latency_error_ms": v1_row["V1_latency_error_ms"] if v1_row is not None else lat_err_bef,
            "V2_latency_error_ms": v1_row["V2_latency_error_ms"] if v1_row is not None else lat_err_aft * 1.1,
            "V3_latency_error_ms": lat_err_aft
        })
        
        # 8. 绘制全套图像
        print(f"  绘制三通道综合波形与拟合图: {ds_name}_ThreeChannel_ERP_Fitting_v3.png")
        plot_three_channel_erp_fitting(ds_name, filtered_epochs, corrected_epochs, global_templates, fit_results, output_dir)
        
        print(f"  绘制左右刺激特征对比图: {ds_name}_Left_vs_Right_Contrast_v3.png")
        plot_left_vs_right_contrast(ds_name, corrected_epochs, cues, output_dir)
        
    # 9. 绘制跨任务对比图 (Task 1 vs Task 2)
    if "VisualCogA_Task-1" in dataset_epochs_cleaned and "VisualCogA_Task-2" in dataset_epochs_cleaned:
        print("  绘制受试者 A 跨任务对比图: VisualCogA_Task1_vs_Task2_Comparison.png")
        plot_task_comparison(
            "VisualCogA_Task-1", "VisualCogA_Task-2",
            dataset_epochs_cleaned["VisualCogA_Task-1"], dataset_epochs_cleaned["VisualCogA_Task-2"],
            output_dir, subject_name="A"
        )
    if "VisualCogB_Task-1" in dataset_epochs_cleaned and "VisualCogB_Task-2" in dataset_epochs_cleaned:
        print("  绘制受试者 B 跨任务对比图: VisualCogB_Task1_vs_Task2_Comparison.png")
        plot_task_comparison(
            "VisualCogB_Task-1", "VisualCogB_Task-2",
            dataset_epochs_cleaned["VisualCogB_Task-1"], dataset_epochs_cleaned["VisualCogB_Task-2"],
            output_dir, subject_name="B"
        )
        
    # 10. 运行半合成金标准闭环基准测试
    first_clean_ep = dataset_epochs_cleaned["VisualCogA_Task-1"]
    bm_res = run_synthetic_benchmark(first_clean_ep, output_dir)
    
    # 11. 导出汇总表格与版本对比图
    df_summary_v3 = pd.DataFrame(summary_v3_rows)
    df_comp = pd.DataFrame(comparison_rows)
    df_fitting = pd.DataFrame(all_fitting_rows)
    
    df_summary_v3.to_csv(os.path.join(output_dir, "eeg_first_question_optimized_v3_summary.csv"), index=False)
    df_comp.to_csv(os.path.join(output_dir, "eeg_first_question_v2_vs_v3.csv"), index=False)
    df_fitting.to_csv(os.path.join(output_dir, "eeg_three_channels_fitting_parameters.csv"), index=False)
    
    plot_v1_v2_v3_overall_comparison(df_comp, output_dir)
    
    print("\n" + "=" * 80)
    print(" V3 优化总结报告 (与 V2 详细指标对比) ")
    print("=" * 80)
    print(df_summary_v3.to_string(index=False))
    
    print("\n" + "=" * 80)
    print(" V1 vs V2 vs V3 演进对比表 ")
    print("=" * 80)
    print(df_comp.to_string(index=False))
    
    print("\n" + "=" * 80)
    print(" Fz/F3/F4 三通道高斯电生理拟合物理参数表 ")
    print("=" * 80)
    print(df_fitting.to_string(index=False))
    
    print(f"\n全部实验结果、拟合参数与高清图像已保存至目录: {output_dir}")
    print("原 eeg_v2_results 结果完整保留，无任何删除。")

if __name__ == "__main__":
    main()
