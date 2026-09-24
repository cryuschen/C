"""Build three final Q1 evidence figures from the V7 result files only."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "eeg_v7_results"
FIGURES = ROOT / "figures"
METRICS = ROOT / "metrics"
DATASETS = [
    ("VisualCogA_Task-1", "受试者A 项目一", "受试者A_项目一"),
    ("VisualCogA_Task-2", "受试者A 项目二", "受试者A_项目二"),
    ("VisualCogB_Task-1", "受试者B 项目一", "受试者B_项目一"),
    ("VisualCogB_Task-2", "受试者B 项目二", "受试者B_项目二"),
]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
TEAL = "#087f8c"
GREY = "#687587"


def save_final_metrics():
    intervals = pd.read_csv(SOURCE / "汇总与说明" / "四组核心指标重采样区间.csv")
    intervals = intervals[intervals.stage == "V7"].copy()
    synthetic = pd.read_csv(SOURCE / "半合成验证" / "按数据组和幅度汇总.csv")
    synthetic = synthetic[synthetic.stage.isin(["预处理", "V7"])].copy()
    stability = pd.read_csv(SOURCE / "汇总与说明" / "左右差分前后时段稳定性.csv")
    stability = stability[(stability.stage == "V7") &
                          (stability.channel == "三通道合并")].copy()
    intervals.to_csv(METRICS / "V7_四组指标区间.csv", index=False)
    synthetic.to_csv(METRICS / "V7_半合成恢复.csv", index=False)
    stability.to_csv(METRICS / "V7_左右差分时段稳定性.csv", index=False)
    return intervals, synthetic, stability


def draw_interval(ax, rows, metric, reference, xlabel, title, xlim=None):
    names = [row[0] for row in DATASETS]
    subset = rows[rows.metric == metric].set_index("dataset").loc[names]
    y = np.arange(len(names))
    point = subset.point.to_numpy()
    low = subset.low.to_numpy()
    high = subset.high.to_numpy()
    ax.errorbar(point, y, xerr=[point - low, high - point], fmt="o",
                color=TEAL, ecolor=TEAL, capsize=4, lw=1.6, ms=6)
    ax.axvline(reference, color=GREY, ls="--", lw=1)
    ax.set_yticks(y, [row[1] for row in DATASETS])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", alpha=0.2)


def plot_overview(intervals, stability):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    draw_interval(axes[0, 0], intervals, "proxy_MAE_reduction_pct", 0,
                  "相对预处理 / %", "A  代理参考 MAE 降幅")
    draw_interval(axes[0, 1], intervals, "SNR_proxy_gain_dB", 0,
                  "相对预处理 / dB", "B  ERP 与试次残差的 SNR 代理增量")
    draw_interval(axes[1, 0], intervals, "left_right_retention_ratio", 1,
                  "处理后 / 预处理", "C  左右差分幅度保留比")
    ax = axes[1, 1]
    stab = stability.set_index("dataset").loc[[row[0] for row in DATASETS]]
    values = stab.contrast_correlation.to_numpy()
    y = np.arange(len(values))
    ax.barh(y, values, color=[TEAL if x >= 0 else "#b85c55" for x in values],
            height=0.48)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlim(-1, 1)
    ax.set_yticks(y, [row[1] for row in DATASETS])
    ax.invert_yaxis()
    ax.set_title("D  左右差分的前后半段相关", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("三通道合并波形相关；无区间")
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle("V7 第一问：去噪与视觉提示差异的证据及限制", fontsize=17,
                 fontweight="bold", y=0.99)
    fig.text(0.03, 0.015,
             "A–C 区间仅反映固定处理流程下的组内试次抽样波动；C 接近 1 不能区分神经响应与方向相关眼动。",
             fontsize=9, color="#45515c")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96), h_pad=3.2, w_pad=2.6)
    fig.savefig(FIGURES / "01_去噪与保真证据总览.png", dpi=180)
    plt.close(fig)


def plot_synthetic(synthetic):
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharex=True)
    specs = [("normalized_RMSE", "恢复 RMSE / 背景 RMS"),
             ("contrast_RMSE", "左右差分恢复 RMSE / 原始单位")]
    for i, (name, label, _) in enumerate(DATASETS):
        for j, (metric, ylabel) in enumerate(specs):
            ax = axes[i, j]
            for stage, color, style in [("预处理", GREY, "--"), ("V7", TEAL, "-")]:
                rows = synthetic[(synthetic.dataset == name) &
                                 (synthetic.stage == stage)].sort_values("level")
                ax.plot(rows.level, rows[metric], "o" + style, color=color,
                        lw=1.8, ms=5, label=stage)
            ax.set_title(f"{label} · {ylabel}", fontsize=11, loc="left")
            ax.set_ylabel(ylabel)
            ax.set_xticks([0, 0.5, 1, 2])
            ax.grid(alpha=0.2)
            if i == 0:
                ax.legend(frameon=False)
            if i == 3:
                ax.set_xlabel("人工伪影幅度倍数；0 = 未新增伪影")
    fig.suptitle("V7 半合成闭环：恢复误差与左右差分误差", fontsize=17,
                 fontweight="bold", y=0.995)
    fig.text(0.03, 0.015,
             "目标为注入前的实测预处理波形，可能仍含原有伪影。零注入时 V7 非零误差表示背景被改动。",
             fontsize=9, color="#45515c")
    fig.tight_layout(rect=(0, 0.038, 1, 0.97), h_pad=2.0, w_pad=2.2)
    fig.savefig(FIGURES / "02_半合成去噪与差分恢复.png", dpi=180)
    plt.close(fig)


def contrast_curve(data, cues):
    lateral = data[:, 1, :] - data[:, 2, :]
    return lateral[cues == -1].mean(axis=0) - lateral[cues == 1].mean(axis=0)


def plot_contrasts(stability):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    records = []
    for ax, (name, label, folder) in zip(axes.flat, DATASETS):
        with np.load(SOURCE / folder / "可复核波形.npz") as z:
            times = z["times_ms"]
            cues = z["cues"]
            before = contrast_curve(z["before"], cues)
            after = contrast_curve(z["v7"], cues)
        corr = stability.loc[stability.dataset == name, "contrast_correlation"].iloc[0]
        ax.plot(times, before, color=GREY, ls="--", lw=1.5, label="预处理")
        ax.plot(times, after, color=TEAL, lw=1.9, label="V7")
        ax.axvspan(250, 500, color="#e3d4a3", alpha=0.2)
        ax.axhline(0, color="#aaaaaa", lw=0.8)
        ax.axvline(0, color="#aaaaaa", lw=0.8)
        ax.set_title(f"{label}｜前后半段相关 {corr:+.2f}", loc="left", fontsize=11)
        ax.set_ylabel("(F3−F4)左减右 / 原始单位")
        ax.set_xlim(-250, 800)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        records.extend({"dataset": name, "time_ms": float(t),
                        "preprocessed": float(b), "V7": float(a)}
                       for t, b, a in zip(times, before, after))
    for ax in axes[1]:
        ax.set_xlabel("相对视觉提示时间 / ms")
    fig.suptitle("左右视觉提示的额区差分：处理前后与时段重复性", fontsize=17,
                 fontweight="bold", y=0.99)
    fig.text(0.03, 0.015,
             "灰线是处理前的方向差，不是神经真值；曲线相近只说明观测差异被保留，不能证明保留的是形状特异神经信号。",
             fontsize=9, color="#45515c")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96), h_pad=2.8, w_pad=2.3)
    fig.savefig(FIGURES / "03_左右差分保留与时段重复性.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(METRICS / "V7_左右差分曲线.csv", index=False)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    intervals, synthetic, stability = save_final_metrics()
    plot_overview(intervals, stability)
    plot_synthetic(synthetic)
    plot_contrasts(stability)
    print("Created 3 final figures and 4 V7-only metric files in", ROOT)


if __name__ == "__main__":
    main()
