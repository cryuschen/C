
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, savgol_filter
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt

# ============================================================
# 第一问：三通道 EEG 的 P300 保真伪影校正（Optimized V2）
# 适用数据：
# VisualCogA_Task-1.mat
# VisualCogA_Task-2.mat
# VisualCogB_Task-1.mat
# VisualCogB_Task-2.mat
#
# 核心思路：
# 1) 0.1–30 Hz 二阶 Butterworth 零相位滤波
# 2) 以 VisCue 上升沿为事件起点，截取 [-250, 800] ms Epoch
# 3) 每个左右刺激条件中选取低伪影试次，构建 ERP 参考模板
# 4) 对污染试次：先减 ERP 模板，再从 Fz/F3/F4 的共同残差中
#    构造 pseudo-artifact reference
# 5) 对公共伪影进行平滑、门控和通道自适应校正
# 6) 在 250–500 ms P300 窗口评估 MAE、相关性、振幅误差、
#    50% 正面积潜伏期误差
#
# 注意：
# - 通道顺序按题目数据：[Fz, F3, F4, FzDecon, F3Decon, F4Decon,
#   ECG, VisCue, Action/TgtAct, TimeStamp]
# - Decon 通道不参与正式模型
# ============================================================

# ----------------------------
# 1. 参数设置
# ----------------------------
# 加载四个数据集
DATA_FILES = [
    "data/VisualCogA_Task-1.mat",
    "data/VisualCogA_Task-2.mat",
    "data/VisualCogB_Task-1.mat",
    "data/VisualCogB_Task-2.mat",
]

FS_EXPECTED = 256 #每秒采多少个点（参数固定）
PRE_SEC = 0.25 # 刺激前保留250ms （可调整）
POST_SEC = 0.80 #刺激后保留800ms （可调整）

N_PRE = round(PRE_SEC * FS_EXPECTED)
N_POST = round(POST_SEC * FS_EXPECTED)

TIMES = np.arange(-N_PRE, N_POST) / FS_EXPECTED

# P300 分析窗口
P300_MASK = (TIMES >= 0.25) & (TIMES <= 0.50)

# 上一版 V1 参数，仅用于对照
V1_PARAMS = {
    "win": 31,
    "gate_z": 3.0,
    "width": 3.0,
    "strength": 1.0,
    "max_beta": 2.5,
}

# 最新优化版 V2 参数
V2_PARAMS = {
    "win": 41,
    "gate_z": 2.5,
    "width": 1.5,
    "strength": 1.0,
    "max_beta": 1.5,
}


# ----------------------------
# 2. 数据读取
# ----------------------------

def load_mat_file(path):
    """
    读取题目 .mat 数据。
    返回：
        fs: 采样率
        labels: 通道标签
        data: shape = (10, N)
    """
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    fs = int(mat["SampleRate"])
    labels = list(mat["DataLabel"])
    data = mat["data"]

    if fs != FS_EXPECTED:
        print(f"[Warning] {path}: SampleRate={fs}, expected={FS_EXPECTED}")

    return fs, labels, data


# ----------------------------
# 3. 视觉事件定位
# ----------------------------

def get_viscue_onsets(viscue):
    """
    VisCue:
        -1 = 左视觉提示
        +1 = 右视觉提示
         0 = 无提示

    找每段非零 VisCue 的起始点。
    """
    nonzero = viscue != 0

    onset_mask = nonzero & np.r_[True, ~nonzero[:-1]]
    onsets = np.where(onset_mask)[0]

    cue_values = viscue[onsets].astype(int)

    return onsets, cue_values


# ----------------------------
# 4. Epoch 截取
# ----------------------------

def extract_epochs(signal, onsets, n_pre=N_PRE, n_post=N_POST):
    """
    signal:
        shape = (channels, time)

    返回：
        shape = (trials, channels, epoch_time)
    """
    ep = []

    for onset in onsets:
        start = onset - n_pre
        end = onset + n_post

        if start < 0 or end > signal.shape[-1]:
            continue

        ep.append(signal[..., start:end])

    return np.stack(ep, axis=0)


# ----------------------------
# 5. 0.1–30 Hz 零相位带通滤波
# ----------------------------

def zero_phase_bandpass(signal, fs=256, low=0.1, high=30.0, order=2):
    """
    二阶 Butterworth 0.1–30 Hz 带通。
    sosfiltfilt 实现前向+反向零相位滤波。
    """
    sos = butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal, axis=-1)


# ----------------------------
# 6. Epoch 基线校正
# ----------------------------

def baseline_correct(ep):
    """
    使用刺激前 [-250, 0] ms 均值作为基线。
    """
    baseline = ep[:, :, :N_PRE].mean(axis=2, keepdims=True)

    return ep - baseline


# ----------------------------
# 7. 鲁棒 Z 分数
# ----------------------------

def robust_z(x):
    """
    基于 median + MAD 的鲁棒标准化。
    """
    med = np.median(x)
    mad = median_abs_deviation(x, scale="normal")

    if mad < 1e-12:
        mad = np.std(x) + 1e-12

    return (x - med) / mad


# ----------------------------
# 8. 伪影评分
# ----------------------------

def compute_artifact_score(ep):
    """
    对每个 trial 计算综合伪影分数。

    使用特征：
    1) 三通道最大 peak-to-peak
    2) 最大绝对幅值
    3) 最大相邻采样点跳变
    4) 三通道共同慢成分幅值
    """

    peak_to_peak = np.ptp(ep, axis=2).max(axis=1)

    max_abs = np.abs(ep).max(axis=(1, 2))

    max_diff = np.abs(
        np.diff(ep, axis=2)
    ).max(axis=(1, 2))

    # 三通道中位数作为 robust common component
    common = np.median(ep, axis=1)

    common_smooth = savgol_filter(
        common,
        window_length=31,
        polyorder=3,
        axis=1
    )

    common_amp = np.abs(common_smooth).max(axis=1)

    features = np.column_stack([
        np.log1p(peak_to_peak),
        np.log1p(max_abs),
        np.log1p(max_diff),
        np.log1p(common_amp),
    ])

    z_features = np.column_stack([
        robust_z(features[:, j])
        for j in range(features.shape[1])
    ])

    # 只把“异常方向”的偏离记入分数
    score = np.maximum(z_features, 0).sum(axis=1)

    return score, peak_to_peak


# ----------------------------
# 9. 构建低伪影 ERP 参考
# ----------------------------

def build_reference_model(
    ep,
    cue,
    irrecoverable,
    artifact_score,
    reference_fraction=0.50
):
    """
    对左刺激和右刺激分别：
    - 在未被严重剔除的 trial 中
    - 选择伪影分数最低的约 50%
    - 用中位数构建低伪影 ERP 参考

    返回：
        templates[value] : shape (3, epoch_time)
        scales[value]    : 每个时间点的 clean common residual MAD
        reference_mask
        artifact_mask
    """

    templates = {}
    scales = {}

    reference_mask = np.zeros(len(ep), dtype=bool)

    for value in (-1, 1):

        idx = np.where(
            (cue == value) & (~irrecoverable)
        )[0]

        # 按伪影分数从小到大
        idx = idx[
            np.argsort(artifact_score[idx])
        ]

        k = max(
            12,
            int(np.ceil(reference_fraction * len(idx)))
        )

        chosen = idx[:k]

        reference_mask[chosen] = True

        # 用中位数而不是均值，增强抗异常能力
        template = np.median(
            ep[chosen],
            axis=0
        )

        templates[value] = template

        # clean residual
        residual = (
            ep[chosen]
            - template[None, :, :]
        )

        # 三通道共同残差
        clean_common = np.median(
            residual,
            axis=1
        )

        # 每个时间点的 MAD
        scale_t = median_abs_deviation(
            clean_common,
            axis=0,
            scale="normal"
        )

        valid = scale_t > 1e-6

        if np.any(valid):
            global_scale = np.median(scale_t[valid])
        else:
            global_scale = np.std(clean_common)

        scale_t = np.maximum(
            scale_t,
            0.5 * global_scale + 1e-6
        )

        scales[value] = scale_t

    artifact_mask = (
        (~reference_mask)
        & (~irrecoverable)
    )

    return (
        templates,
        scales,
        reference_mask,
        artifact_mask
    )


# ----------------------------
# 10. 单个 Epoch 的自适应伪影校正
# ----------------------------

def correct_single_epoch(
    epoch,
    template,
    scale_t,
    params
):
    """
    Optimized V2 核心：

    1) residual = epoch - ERP template
       先保护事件相关 ERP

    2) 对三个通道 residual 取中位数，
       构造 robust common residual

    3) Savitzky-Golay 平滑，
       提取慢变眨眼/运动形态

    4) 只有超过 clean residual 正常范围
       的时间点才激活校正

    5) 对每个通道估计独立 beta，
       自适应减去公共伪影参考
    """

    residual = epoch - template

    # robust common component
    common = np.median(
        residual,
        axis=0
    )

    # 平滑保留慢变伪影形态
    artifact_ref = savgol_filter(
        common,
        window_length=params["win"],
        polyorder=3
    )

    # 与 clean residual 的正常波动比较
    z = np.abs(artifact_ref) / (
        scale_t + 1e-6
    )

    # soft gate
    gate = np.clip(
        (
            z - params["gate_z"]
        )
        / params["width"],
        0,
        1
    )

    gate = savgol_filter(
        gate,
        window_length=15,
        polyorder=2
    )

    gate = np.clip(
        gate,
        0,
        1
    )

    artifact_ref = (
        artifact_ref * gate
    )

    if np.max(
        np.abs(artifact_ref)
    ) < 1e-8:
        return epoch.copy()

    denom = (
        np.sum(
            (artifact_ref * gate) ** 2
        )
        + 1e-9
    )

    corrected = epoch.copy()

    for ch in range(3):

        beta = np.sum(
            (gate * artifact_ref)
            * residual[ch]
        ) / denom

        beta = float(
            np.clip(
                beta,
                0,
                params["max_beta"]
            )
        )

        corrected[ch] = (
            epoch[ch]
            - params["strength"]
            * beta
            * artifact_ref
        )

    # 校正后重新 baseline
    corrected -= corrected[
        :, :N_PRE
    ].mean(
        axis=1,
        keepdims=True
    )

    return corrected


# ----------------------------
# 11. 批量校正污染 trial
# ----------------------------

def correct_artifact_epochs(
    ep,
    cue,
    artifact_mask,
    templates,
    scales,
    params
):
    corrected = ep.copy()

    target_idx = np.where(
        artifact_mask
    )[0]

    for i in target_idx:

        value = cue[i]

        corrected[i] = correct_single_epoch(
            ep[i],
            templates[value],
            scales[value],
            params
        )

    return corrected


# ----------------------------
# 12. P300 特征
# ----------------------------

def p300_amplitude_latency(y):
    """
    P300 正向平均振幅 + 50% 正面积潜伏期
    """

    positive = np.maximum(
        y[P300_MASK],
        0
    )

    amplitude = positive.mean()

    total_area = positive.sum()

    if total_area <= 1e-9:
        return amplitude, np.nan

    cumulative = np.cumsum(positive)

    idx = np.searchsorted(
        cumulative,
        0.5 * total_area
    )

    p300_times = TIMES[P300_MASK]

    idx = min(
        idx,
        len(p300_times) - 1
    )

    latency_ms = (
        p300_times[idx] * 1000
    )

    return amplitude, latency_ms


# ----------------------------
# 13. 模型评价
# ----------------------------

def evaluate_p300(
    ep_before,
    ep_after,
    cue,
    artifact_mask,
    templates
):
    """
    比较污染 trial ERP 与 low-artifact ERP reference。

    指标：
    - MAE
    - 波形相关系数
    - P300 positive mean amplitude error
    - 50% positive area latency error
    """

    records = []

    channel_names = [
        "Fz",
        "F3",
        "F4"
    ]

    for value in (-1, 1):

        idx = np.where(
            (cue == value)
            & artifact_mask
        )[0]

        before_avg = ep_before[
            idx
        ].mean(axis=0)

        after_avg = ep_after[
            idx
        ].mean(axis=0)

        reference = templates[value]

        for ch, ch_name in enumerate(
            channel_names
        ):

            ref_p300 = reference[
                ch, P300_MASK
            ]

            before_p300 = before_avg[
                ch, P300_MASK
            ]

            after_p300 = after_avg[
                ch, P300_MASK
            ]

            # MAE
            mae_before = np.mean(
                np.abs(
                    before_p300
                    - ref_p300
                )
            )

            mae_after = np.mean(
                np.abs(
                    after_p300
                    - ref_p300
                )
            )

            # 波形相关
            corr_before = np.corrcoef(
                before_p300,
                ref_p300
            )[0, 1]

            corr_after = np.corrcoef(
                after_p300,
                ref_p300
            )[0, 1]

            # P300 特征
            amp_ref, lat_ref = (
                p300_amplitude_latency(
                    reference[ch]
                )
            )

            amp_before, lat_before = (
                p300_amplitude_latency(
                    before_avg[ch]
                )
            )

            amp_after, lat_after = (
                p300_amplitude_latency(
                    after_avg[ch]
                )
            )

            records.append({
                "cue": value,
                "channel": ch_name,

                "MAE_before":
                    mae_before,

                "MAE_after":
                    mae_after,

                "corr_before":
                    corr_before,

                "corr_after":
                    corr_after,

                "amp_error_before":
                    abs(
                        amp_before
                        - amp_ref
                    ),

                "amp_error_after":
                    abs(
                        amp_after
                        - amp_ref
                    ),

                "latency_error_before_ms":
                    abs(
                        lat_before
                        - lat_ref
                    )
                    if not (
                        np.isnan(lat_before)
                        or np.isnan(lat_ref)
                    )
                    else np.nan,

                "latency_error_after_ms":
                    abs(
                        lat_after
                        - lat_ref
                    )
                    if not (
                        np.isnan(lat_after)
                        or np.isnan(lat_ref)
                    )
                    else np.nan,
            })

    return pd.DataFrame(records)


# ----------------------------
# 14. 绘图
# ----------------------------

def plot_fz_comparison(
    dataset_name,
    ep,
    corrected_v1,
    corrected_v2,
    cue,
    artifact_mask,
    templates,
    output_dir
):
    """
    Fz 快速对比图。
    左/右刺激分别算 ERP 后等权平均。
    """

    reference_curves = []
    before_curves = []
    v1_curves = []
    v2_curves = []

    for value in (-1, 1):

        idx = (
            (cue == value)
            & artifact_mask
        )

        reference_curves.append(
            templates[value][0]
        )

        before_curves.append(
            ep[idx].mean(axis=0)[0]
        )

        v1_curves.append(
            corrected_v1[
                idx
            ].mean(axis=0)[0]
        )

        v2_curves.append(
            corrected_v2[
                idx
            ].mean(axis=0)[0]
        )

    ref = np.mean(
        reference_curves,
        axis=0
    )

    before = np.mean(
        before_curves,
        axis=0
    )

    v1 = np.mean(
        v1_curves,
        axis=0
    )

    v2 = np.mean(
        v2_curves,
        axis=0
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        TIMES * 1000,
        ref,
        label="Low-artifact ERP reference"
    )

    plt.plot(
        TIMES * 1000,
        before,
        label="Bandpass only"
    )

    plt.plot(
        TIMES * 1000,
        v1,
        label="V1 correction"
    )

    plt.plot(
        TIMES * 1000,
        v2,
        label="Optimized V2"
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.axvspan(
        250,
        500,
        alpha=0.12,
        label="P300 window"
    )

    plt.xlabel(
        "Time from VisCue onset (ms)"
    )

    plt.ylabel(
        "Fz amplitude (original units)"
    )

    plt.title(
        f"{dataset_name} — optimized Fz ERP comparison"
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"{dataset_name}_Fz_optimized_v2.png"
    )

    plt.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ----------------------------
# 15. 主流程
# ----------------------------

def run_analysis(
    data_files,
    output_dir="eeg_v2_results"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    summary_rows = []
    comparison_rows = []

    for path in data_files:

        print("\n" + "=" * 70)
        print(
            f"Processing: {path}"
        )

        dataset_name = os.path.basename(
            path
        ).replace(".mat", "")

        fs, labels, data = (
            load_mat_file(path)
        )

        print(
            "Labels:",
            labels
        )

        # --------------------
        # 视觉事件
        # --------------------

        onsets, cue = (
            get_viscue_onsets(
                data[7]
            )
        )

        print(
            f"Visual trials: {len(onsets)}"
        )

        # --------------------
        # 原始 EEG Epoch
        # --------------------

        raw_ep = extract_epochs(
            data[:3],
            onsets
        )

        # --------------------
        # 0.1–30 Hz 预滤波
        # --------------------

        filtered_eeg = (
            zero_phase_bandpass(
                data[:3],
                fs=fs
            )
        )

        ep = baseline_correct(
            extract_epochs(
                filtered_eeg,
                onsets
            )
        )

        # --------------------
        # 严重异常 trial
        # --------------------

        score, peak_to_peak = (
            compute_artifact_score(ep)
        )

        # 原始设备饱和 ±1000
        saturation_count = np.sum(
            np.isclose(
                np.abs(raw_ep),
                1000.0
            ),
            axis=(1, 2)
        )

        irrecoverable = (
            (saturation_count >= 20)
            | (peak_to_peak > 1800)
        )

        # --------------------
        # 建立参考 ERP
        # --------------------

        (
            templates,
            scales,
            reference_mask,
            artifact_mask
        ) = build_reference_model(
            ep,
            cue,
            irrecoverable,
            score,
            reference_fraction=0.50
        )

        print(
            "Reference trials:",
            reference_mask.sum()
        )

        print(
            "Corrected trials:",
            artifact_mask.sum()
        )

        print(
            "Rejected trials:",
            irrecoverable.sum()
        )

        # --------------------
        # V1
        # --------------------

        corrected_v1 = (
            correct_artifact_epochs(
                ep,
                cue,
                artifact_mask,
                templates,
                scales,
                V1_PARAMS
            )
        )

        # --------------------
        # V2
        # --------------------

        corrected_v2 = (
            correct_artifact_epochs(
                ep,
                cue,
                artifact_mask,
                templates,
                scales,
                V2_PARAMS
            )
        )

        # --------------------
        # 评价 V1
        # --------------------

        metrics_v1 = evaluate_p300(
            ep,
            corrected_v1,
            cue,
            artifact_mask,
            templates
        )

        # --------------------
        # 评价 V2
        # --------------------

        metrics_v2 = evaluate_p300(
            ep,
            corrected_v2,
            cue,
            artifact_mask,
            templates
        )

        # 保存单数据集明细
        metrics_v2.to_csv(
            os.path.join(
                output_dir,
                f"{dataset_name}_V2_metrics.csv"
            ),
            index=False
        )

        # --------------------
        # 汇总表
        # --------------------

        mae_before = (
            metrics_v2[
                "MAE_before"
            ].mean()
        )

        mae_after = (
            metrics_v2[
                "MAE_after"
            ].mean()
        )

        summary_rows.append({
            "dataset":
                dataset_name,

            "trials":
                len(cue),

            "reference_trials":
                int(
                    reference_mask.sum()
                ),

            "corrected_trials":
                int(
                    artifact_mask.sum()
                ),

            "rejected_trials":
                int(
                    irrecoverable.sum()
                ),

            "MAE_before":
                mae_before,

            "MAE_after_V2":
                mae_after,

            "MAE_reduction_V2_percent":
                100
                * (
                    1
                    - mae_after
                    / mae_before
                ),

            "corr_before":
                metrics_v2[
                    "corr_before"
                ].mean(),

            "corr_after_V2":
                metrics_v2[
                    "corr_after"
                ].mean(),

            "amp_error_before":
                metrics_v2[
                    "amp_error_before"
                ].mean(),

            "amp_error_after_V2":
                metrics_v2[
                    "amp_error_after"
                ].mean(),

            "latency_error_before_ms":
                metrics_v2[
                    "latency_error_before_ms"
                ].mean(),

            "latency_error_after_V2_ms":
                metrics_v2[
                    "latency_error_after_ms"
                ].mean(),
        })

        # --------------------
        # V1 vs V2
        # --------------------

        comparison_rows.append({
            "dataset":
                dataset_name,

            "V1_MAE":
                metrics_v1[
                    "MAE_after"
                ].mean(),

            "V2_MAE":
                metrics_v2[
                    "MAE_after"
                ].mean(),

            "V1_corr":
                metrics_v1[
                    "corr_after"
                ].mean(),

            "V2_corr":
                metrics_v2[
                    "corr_after"
                ].mean(),

            "V1_amp_error":
                metrics_v1[
                    "amp_error_after"
                ].mean(),

            "V2_amp_error":
                metrics_v2[
                    "amp_error_after"
                ].mean(),

            "V1_latency_error_ms":
                metrics_v1[
                    "latency_error_after_ms"
                ].mean(),

            "V2_latency_error_ms":
                metrics_v2[
                    "latency_error_after_ms"
                ].mean(),
        })

        # --------------------
        # Fz 图
        # --------------------

        plot_fz_comparison(
            dataset_name,
            ep,
            corrected_v1,
            corrected_v2,
            cue,
            artifact_mask,
            templates,
            output_dir
        )

    # ------------------------
    # 总汇总
    # ------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    summary_file = os.path.join(
        output_dir,
        "eeg_first_question_optimized_v2_summary.csv"
    )

    comparison_file = os.path.join(
        output_dir,
        "eeg_first_question_v1_vs_v2.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    comparison_df.to_csv(
        comparison_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("Optimized V2 Summary")
    print("=" * 70)

    print(
        summary_df.round(3)
    )

    print("\n" + "=" * 70)
    print("V1 vs V2")
    print("=" * 70)

    print(
        comparison_df.round(3)
    )

    print(
        "\nV2 Parameters:"
    )

    print(
        V2_PARAMS
    )

    print(
        "\nResults saved to:",
        output_dir
    )

    return (
        summary_df,
        comparison_df
    )


# ============================================================
# 16. 执行
# ============================================================

if __name__ == "__main__":

    summary_df, comparison_df = run_analysis(
        DATA_FILES,
        output_dir="eeg_v2_results"
    )
