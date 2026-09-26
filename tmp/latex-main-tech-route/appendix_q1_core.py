# 问题一：脑电预处理、双分量校正与样条拟合
import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline
import scipy.io as sio
from scipy.stats import median_abs_deviation
from scipy.signal import find_peaks, butter, sosfiltfilt, iirnotch, filtfilt, savgol_filter, welch
FS_EXPECTED = 256
PRE_SEC = 0.25      # 刺激前 250 ms 基线
POST_SEC = 0.80     # 刺激后 800 ms 分析窗

N_PRE = round(PRE_SEC * FS_EXPECTED)
N_POST = round(POST_SEC * FS_EXPECTED)
TIMES = np.arange(-N_PRE, N_POST) / FS_EXPECTED
TIMES_MS = TIMES * 1000.0

P300_MASK = (TIMES >= 0.25) & (TIMES <= 0.50)
CHANNEL_NAMES = ["Fz", "F3", "F4"]

# 双分量校正参数
CORRECTION_PARAMS = {
    "win_blink": 41,        # 垂直眼电平滑窗长 (约 160 ms)
    "win_saccade": 31,      # 水平扫视平滑窗长 (约 120 ms)
    "gate_z_v": 2.2,        # 垂直软门控阈值
    "gate_z_h": 2.2,        # 水平软门控阈值
    "gate_width": 1.2,      # 软门控过渡带宽度
    "reg_lambda": 0.05,     # 岭回归正则化系数，防止过减
    "max_beta_v": 1.8,      # 垂直通道最大增益
    "max_beta_h": 1.5,      # 水平通道最大增益
}
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

def correct_single_epoch(epoch, template, scale_v, scale_h, params=CORRECTION_PARAMS):
    """
    双分量自适应校正：
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

