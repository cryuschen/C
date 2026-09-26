"""Build paper figures 4 and 5 from the fixed V7 figure-data bundle.

Figure 4 uses the predefined median-artifact left trial in A Task 1. Device
Decon is included only for visual comparison and was never a V7 input.
Figure 5 compares raw and V7 arithmetic means on the same retained trials.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "figure_data_v7.npz"
OUTPUT = ROOT / "figures_4_5_v7"

CHANNELS = ("F3", "Fz", "F4")
RAW_ROWS = {"Fz": 0, "F3": 1, "F4": 2}
COLORS = {"Raw": "#8C9AA8", "Decon": "#C5724A", "V7": "#087C87",
          "左提示": "#087C87", "右提示": "#C5724A"}
LINESTYLES = {"Raw": (0, (2, 2)), "Decon": (0, (5, 2)), "V7": "-"}


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "Microsoft YaHei",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.axisbelow": True,
        "grid.color": "#D9DEE3",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "legend.frameon": True,
        "savefig.dpi": 240,
    })


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, np.ndarray]:
    with np.load(DATA, allow_pickle=False) as saved:
        bundle = {key: saved[key].copy() for key in saved.files}
    provenance = json.loads(str(bundle["provenance_json"]))
    assert provenance["kind"] == "committed V7 fixed figure inputs"
    return bundle


def load_v7(subject: str, task: int) -> dict[str, np.ndarray]:
    prefix = f"{subject}{task}_"
    bundle = load_bundle()
    data = {key: bundle[prefix + key] for key in
            ("raw", "v7", "cues", "times_ms", "trial_ids")}
    assert data["raw"].shape == data["v7"].shape
    assert data["raw"].shape[1] == 3
    assert np.array_equal(np.unique(data["cues"]), np.array([-1, 1]))
    assert np.isclose(data["times_ms"][0], -250.0)
    assert np.isclose(data["times_ms"][-1], 796.875)
    return data


def trial_for_figure_4() -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    bundle = load_bundle()
    trial_id = int(bundle["figure4_trial_id"])
    return load_v7("A", 1), trial_id, bundle["figure4_decon"]


def align_baseline(curve: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Display alignment only; the underlying processing remains unchanged."""
    return curve - np.median(curve[times < 0])


def figure_4_data() -> tuple[np.ndarray, int, dict[str, dict[str, np.ndarray]]]:
    v7, trial_id, decon = trial_for_figure_4()
    indices = np.flatnonzero(v7["trial_ids"] == trial_id)
    assert len(indices) == 1
    index = int(indices[0])
    times = v7["times_ms"]
    assert decon.shape == (3, len(times))

    curves: dict[str, dict[str, np.ndarray]] = {}
    for channel in CHANNELS:
        raw_row = RAW_ROWS[channel]
        curves[channel] = {
            "Raw": align_baseline(v7["raw"][index, raw_row], times),
            "Decon": align_baseline(decon[raw_row], times),
            "V7": align_baseline(v7["v7"][index, raw_row], times),
        }
    return times, trial_id, curves


def basic_axes(ax: plt.Axes, times: np.ndarray) -> None:
    ax.axvspan(250, 500, color="#E8F2F4", zorder=0)
    ax.axvline(0, color="#8996A2", lw=0.85, zorder=1)
    ax.axhline(0, color="#B6C0C7", lw=0.7, zorder=1)
    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.set_xticks([-200, 0, 200, 400, 600, 800])
    ax.grid(True)


def save_png(fig: plt.Figure, path: Path) -> None:
    """先完整写入临时文件，再替换目标，避免覆盖期间出现半成品。"""
    temporary = path.with_name(".__plot_tmp.png")
    fig.savefig(temporary, facecolor="white")
    path.unlink(missing_ok=True)
    temporary.replace(path)


def draw_figure_4_channel(ax: plt.Axes, times: np.ndarray, channel: str,
                          curves: dict[str, np.ndarray]) -> None:
    basic_axes(ax, times)
    for label in ("Raw", "Decon", "V7"):
        ax.plot(times, curves[label], color=COLORS[label],
                ls=LINESTYLES[label], lw=1.45 if label != "V7" else 1.8,
                label={"Raw": "降噪前", "Decon": "设备滤波", "V7": "降噪后"}[label],
                zorder=3 if label == "V7" else 2)
    ax.set_ylabel("电位（原始单位）")
    ax.text(0.015, 0.96, channel, transform=ax.transAxes, va="top",
            ha="left", fontsize=11, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78})


def save_figure_4() -> None:
    times, trial_id, curves = figure_4_data()
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.4), sharex=True)
    for ax, channel in zip(axes, CHANNELS):
        draw_figure_4_channel(ax, times, channel, curves[channel])
    axes[0].legend(loc="upper right", frameon=True, ncol=3, fontsize=9)
    axes[-1].set_xlabel("相对提示时间（ms）")
    fig.suptitle(f"图4  A1试次{trial_id}三通道降噪对比", fontsize=15, y=0.98)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.92, bottom=0.09, hspace=0.18)
    save_png(fig, OUTPUT / "图4_Raw_Decon_V7_三通道.png")
    plt.close(fig)

    for channel in CHANNELS:
        fig, ax = plt.subplots(figsize=(8.2, 3.6))
        draw_figure_4_channel(ax, times, channel, curves[channel])
        ax.set_xlabel("相对提示时间（ms）")
        ax.legend(loc="upper right", frameon=True, ncol=3, fontsize=8.4)
        fig.suptitle(f"图4  A1试次{trial_id}{channel}通道降噪对比",
                     fontsize=12, y=0.98)
        fig.subplots_adjust(left=0.13, right=0.985, top=0.84, bottom=0.18)
        save_png(fig, OUTPUT / f"图4_{channel}_Raw_Decon_V7.png")
        plt.close(fig)


def erp_curves(data: dict[str, np.ndarray], channel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    row = RAW_ROWS[channel]
    # Baseline-align each unfiltered source epoch before averaging. The saved
    # V7 epochs already have the same prestimulus median baseline definition.
    times = data["times_ms"]
    raw = data["raw"][:, row]
    raw_aligned = raw - np.median(raw[:, times < 0], axis=1, keepdims=True)
    raw_left = raw_aligned[data["cues"] == -1].mean(axis=0)
    raw_right = raw_aligned[data["cues"] == 1].mean(axis=0)
    v7_left = data["v7"][data["cues"] == -1, row].mean(axis=0)
    v7_right = data["v7"][data["cues"] == 1, row].mean(axis=0)
    n_left = int(np.sum(data["cues"] == -1))
    n_right = int(np.sum(data["cues"] == 1))
    return raw_left, raw_right, v7_left, v7_right, n_left, n_right


def draw_erp(ax: plt.Axes, data: dict[str, np.ndarray], channel: str) -> tuple[int, int]:
    times = data["times_ms"]
    raw_left, raw_right, v7_left, v7_right, n_left, n_right = erp_curves(data, channel)
    basic_axes(ax, times)
    ax.plot(times, raw_left, color="#72A7AD", ls=(0, (4, 2)), lw=1.3,
            label="降噪前 左三角", zorder=2)
    ax.plot(times, raw_right, color="#D8A087", ls=(0, (4, 2)), lw=1.3,
            label="降噪前 右三角", zorder=2)
    ax.plot(times, v7_left, color=COLORS["左提示"], lw=1.85,
            label="降噪后 左三角", zorder=3)
    ax.plot(times, v7_right, color=COLORS["右提示"], lw=1.85,
            label="降噪后 右三角", zorder=3)
    ax.set_title(f"{channel}（左{n_left}次，右{n_right}次）", pad=6)
    return n_left, n_right


def save_figure_5(subject: str) -> None:
    data_by_task = {task: load_v7(subject, task) for task in (1, 2)}
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), sharex=True)
    for row, task in enumerate((1, 2)):
        data = data_by_task[task]
        for col, channel in enumerate(CHANNELS):
            ax = axes[row, col]
            draw_erp(ax, data, channel)
            if col == 0:
                ax.set_ylabel(f"项目{task}\n电位（原始单位）")
            if row == 1:
                ax.set_xlabel("相对提示时间（ms）")
    handles = [
        Line2D([0], [0], color="#72A7AD", ls=(0, (4, 2)), lw=1.5, label="降噪前 左三角"),
        Line2D([0], [0], color=COLORS["左提示"], lw=2, label="降噪后 左三角"),
        Line2D([0], [0], color="#D8A087", ls=(0, (4, 2)), lw=1.5, label="降噪前 右三角"),
        Line2D([0], [0], color=COLORS["右提示"], lw=2, label="降噪后 右三角"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=4, frameon=False)
    fig.suptitle(f"图5{'A' if subject == 'A' else 'B'}  受试者{subject}左右三角提示的ERP对比",
                 fontsize=15, y=0.99)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.125,
                        wspace=0.24, hspace=0.39)
    save_png(fig, OUTPUT / f"图5{'A' if subject == 'A' else 'B'}_受试者{subject}_左右提示ERP.png")
    plt.close(fig)

    for task in (1, 2):
        data = data_by_task[task]
        for channel in CHANNELS:
            fig, ax = plt.subplots(figsize=(7.8, 3.9))
            draw_erp(ax, data, channel)
            ax.set_xlabel("相对提示时间（ms）")
            ax.set_ylabel("电位（原始单位）")
            ax.legend(loc="best", frameon=True, ncol=2, fontsize=8.5)
            fig.suptitle(f"图5{'A' if subject == 'A' else 'B'}  受试者{subject}项目{task}{channel}通道ERP",
                         fontsize=12, y=0.98)
            fig.subplots_adjust(left=0.12, right=0.985, top=0.82, bottom=0.18)
            save_png(fig, OUTPUT / f"图5{'A' if subject == 'A' else 'B'}_项目{task}_{channel}_左右提示ERP.png")
            plt.close(fig)


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="图片输出目录，默认 Q1/figures_4_5_v7")
    OUTPUT = parser.parse_args().output.resolve()
    configure_plotting()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_figure_4()
    for subject in ("A", "B"):
        save_figure_5(subject)
    paths = sorted(OUTPUT.glob("*.png"))
    assert len(paths) == 18, f"Expected 18 images; found {len(paths)}"
    print(f"Created {len(paths)} PNG files in {OUTPUT}")


if __name__ == "__main__":
    main()
