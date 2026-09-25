#!/usr/bin/env python3
"""Q2 model using the auditable Q1 V7 result bundles.

The known-condition mechanism fit uses Q1 ``v7`` epochs.  The unknown-cue
decoder deliberately uses Q1 ``before`` epochs because V7 selects a
direction-specific reference with the true cue.  This separation prevents the
test cue from entering the primary decoder while retaining Q1's trial audit,
filtering, baseline correction, and rejection decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/q2-matplotlib")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "q2_result" / "q1_model"
NAMES = {
    "A1": "受试者A_项目一",
    "A2": "受试者A_项目二",
    "B1": "受试者B_项目一",
    "B2": "受试者B_项目二",
}
CHANNELS = ("Fz", "F3", "F4")
MODE_NAMES = ("共同", "中线", "侧化")
# Rows transform Fz/F3/F4 into common, midline and F3-F4 lateral modes.
SCALP = np.array([
    [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    [2 / np.sqrt(6), -1 / np.sqrt(6), -1 / np.sqrt(6)],
    [0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
])
WINDOWS_MS = ((50, 250), (250, 500), (500, 750), (50, 750))


@dataclass(frozen=True)
class Recording:
    key: str
    before: np.ndarray
    v7: np.ndarray
    cues: np.ndarray
    trial_ids: np.ndarray
    blocks: np.ndarray
    times_ms: np.ndarray
    source: Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_q1_results(root: Path = ROOT) -> list[Recording]:
    recordings = []
    expected_times = None
    for key, folder in NAMES.items():
        source = root / "eeg_v7_results" / folder / "可复核波形.npz"
        with np.load(source) as data:
            required = {"before", "v7", "cues", "trial_ids", "times_ms"}
            if not required.issubset(data.files):
                raise ValueError(f"Q1 result is incomplete: {source}")
            before = data["before"].copy()
            v7 = data["v7"].copy()
            cues = data["cues"].astype(int, copy=True)
            trial_ids = data["trial_ids"].astype(int, copy=True)
            times = data["times_ms"].copy()
        if before.shape != v7.shape or before.shape[1:] != (3, len(times)):
            raise ValueError(f"Unexpected Q1 epoch shape: {source}")
        if len(np.unique(trial_ids)) != len(trial_ids) or set(np.unique(cues)) != {-1, 1}:
            raise ValueError(f"Invalid trial lineage or cues: {source}")
        if not np.isfinite(before).all() or not np.isfinite(v7).all():
            raise ValueError(f"Non-finite Q1 waveform: {source}")
        if expected_times is None:
            expected_times = times
        elif not np.array_equal(times, expected_times):
            raise ValueError("Q1 time axes differ")
        recordings.append(Recording(
            key=key, before=before, v7=v7, cues=cues, trial_ids=trial_ids,
            blocks=(trial_ids - 1) // 20, times_ms=times, source=source,
        ))
    return recordings


def scalp_modes(epochs: np.ndarray) -> np.ndarray:
    return np.einsum("mc,nct->nmt", SCALP, epochs)


def covariance_features(epochs: np.ndarray, times_ms: np.ndarray
                        ) -> tuple[np.ndarray, list[str]]:
    """24 fixed source-mode covariance features from four post-cue windows."""
    modes = scalp_modes(epochs)
    arrays, names = [], []
    for low, high in WINDOWS_MS:
        chosen = (times_ms >= low) & (times_ms < high)
        segment = modes[..., chosen]
        segment = segment - segment.mean(axis=-1, keepdims=True)
        cov = np.einsum("nmt,nkt->nmk", segment, segment) / segment.shape[-1]
        sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-12))
        values = [np.log(sd[:, j] ** 2) for j in range(3)]
        labels = [f"{low}_{high}ms_logvar_{name}" for name in MODE_NAMES]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            values.append(cov[:, a, b] / (sd[:, a] * sd[:, b]))
            labels.append(f"{low}_{high}ms_corr_{MODE_NAMES[a]}_{MODE_NAMES[b]}")
        arrays.append(np.column_stack(values))
        names.extend(labels)
    result = np.concatenate(arrays, axis=1)
    if result.shape[1] != 24 or not np.isfinite(result).all():
        raise ValueError("Invalid covariance representation")
    return result, names


def assemble(recordings: list[Recording], stage: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features, labels, groups, blocks, trial_ids = [], [], [], [], []
    names = None
    for group, rec in enumerate(recordings):
        x, current_names = covariance_features(getattr(rec, stage), rec.times_ms)
        features.append(x)
        labels.append(rec.cues)
        groups.append(np.full(len(rec.cues), group))
        blocks.append(rec.blocks)
        trial_ids.append(rec.trial_ids)
        names = current_names if names is None else names
        if names != current_names:
            raise ValueError("Feature names differ")
    return (np.concatenate(features), np.concatenate(labels), np.concatenate(groups),
            np.concatenate(blocks), np.concatenate(trial_ids), names)


def normalize_by_recording(train_x: np.ndarray, test_x: np.ndarray,
                           train_group: np.ndarray, test_group: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Training-only robust normalization within each recording."""
    train_z, test_z = train_x.copy(), test_x.copy()
    for group in np.unique(train_group):
        reference = train_x[train_group == group]
        center = np.median(reference, axis=0)
        scale = np.quantile(reference, .75, axis=0) - np.quantile(reference, .25, axis=0)
        scale = np.maximum(scale, 1e-8)
        train_z[train_group == group] = np.clip(
            (train_x[train_group == group] - center) / scale, -8, 8)
        test_z[test_group == group] = np.clip(
            (test_x[test_group == group] - center) / scale, -8, 8)
    return train_z, test_z


def classifier(seed: int, trees: int, n_jobs: int = -1) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trees, min_samples_leaf=3, max_features=.7,
        class_weight="balanced", random_state=seed, n_jobs=n_jobs,
    )


def blocked_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   blocks: np.ndarray, trees: int, seed: int,
                   n_jobs: int = -1) -> np.ndarray:
    """Hold the same chronological block out from all four recordings."""
    scores = np.full(len(y), np.nan)
    for held in sorted(np.unique(blocks)):
        train, test = blocks != held, blocks == held
        train_x, test_x = normalize_by_recording(
            x[train], x[test], groups[train], groups[test])
        model = classifier(seed + int(held), trees, n_jobs=n_jobs)
        model.fit(train_x, y[train])
        scores[test] = model.predict_proba(test_x)[:, 1] - .5
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete outer-fold predictions")
    return scores


def balanced_accuracy(y: np.ndarray, scores: np.ndarray) -> float:
    prediction = np.where(scores >= 0, 1, -1)
    return float(.5 * ((prediction[y == -1] == -1).mean() +
                       (prediction[y == 1] == 1).mean()))


def metric_rows(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                stage: str) -> list[dict]:
    rows = []
    prediction = np.where(scores >= 0, 1, -1)
    for key, mask in [("pooled", np.ones(len(y), dtype=bool))] + [
            (name, groups == index) for index, name in enumerate(NAMES)]:
        yy, ss, pp = y[mask], scores[mask], prediction[mask]
        rows.append(dict(
            stage=stage, dataset=key, n=int(mask.sum()),
            BA=balanced_accuracy(yy, ss), AUC=float(roc_auc_score(yy, ss)),
            recall_left=float((pp[yy == -1] == -1).mean()),
            recall_right=float((pp[yy == 1] == 1).mean()),
        ))
    return rows


def permute_within_blocks(y: np.ndarray, groups: np.ndarray, blocks: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    result = y.copy()
    for group in np.unique(groups):
        for block in np.unique(blocks[groups == group]):
            chosen = np.flatnonzero((groups == group) & (blocks == block))
            result[chosen] = rng.permutation(result[chosen])
    return result


def gamma_basis(times_ms: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Fixed LGN-to-cortex cascade kernels; peaks are assumptions, not estimates."""
    positive = np.maximum(times_ms, 0)
    settings = [
        (90, 3, "LGN快速中继"),
        (140, 4, "V1方向选择"),
        (230, 5, "视觉皮层复发"),
        (350, 6, "高级视觉整合"),
        (520, 7, "额区情境更新"),
        (680, 8, "皮层丘脑反馈"),
    ]
    bases = []
    chosen = (times_ms >= 50) & (times_ms < 750)
    for peak, shape, _ in settings:
        z = np.zeros_like(positive, dtype=float)
        mask = positive > 0
        ratio = positive[mask] / peak
        z[mask] = ratio ** shape * np.exp(shape * (1 - ratio))
        z /= np.linalg.norm(z[chosen]) + 1e-12
        bases.append(z)
    return np.column_stack(bases), [item[2] for item in settings]


def fit_mechanism(recordings: list[Recording]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit fixed cascade bases to Q1 V7 right-minus-left condition contrasts."""
    rows, waves = [], {}
    for rec in recordings:
        mask = (rec.times_ms >= 50) & (rec.times_ms < 750)
        bases, _ = gamma_basis(rec.times_ms)
        h = bases[mask]
        observed = rec.v7[rec.cues == 1].mean(0) - rec.v7[rec.cues == -1].mean(0)
        coef = np.linalg.solve(h.T @ h + .1 * np.eye(h.shape[1]),
                               h.T @ observed[:, mask].T)
        fitted = np.zeros_like(observed)
        fitted[:, mask] = (h @ coef).T
        descriptive = 1 - np.mean((observed[:, mask] - fitted[:, mask]) ** 2) / np.mean(
            observed[:, mask] ** 2)
        held_observed, held_predicted = [], []
        for held in sorted(np.unique(rec.blocks)):
            train, test = rec.blocks != held, rec.blocks == held
            train_delta = (rec.v7[train & (rec.cues == 1)].mean(0) -
                           rec.v7[train & (rec.cues == -1)].mean(0))
            test_delta = (rec.v7[test & (rec.cues == 1)].mean(0) -
                          rec.v7[test & (rec.cues == -1)].mean(0))
            fold_coef = np.linalg.solve(h.T @ h + np.eye(h.shape[1]),
                                        h.T @ train_delta[:, mask].T)
            held_observed.append(test_delta[:, mask])
            held_predicted.append((h @ fold_coef).T)
        held_observed = np.stack(held_observed)
        held_predicted = np.stack(held_predicted)
        held_score = 1 - np.mean((held_observed - held_predicted) ** 2) / np.mean(
            held_observed ** 2)
        rows.append(dict(dataset=rec.key, n=len(rec.cues),
                         descriptive_fit_fraction=float(descriptive),
                         held_block_S_delta=float(held_score)))
        waves[f"{rec.key}_observed"] = observed
        waves[f"{rec.key}_fitted"] = fitted
    waves["times_ms"] = recordings[0].times_ms
    waves["bases"] = gamma_basis(recordings[0].times_ms)[0]
    return pd.DataFrame(rows), waves


def make_figure(output: Path, recordings: list[Recording], mechanism: pd.DataFrame,
                waves: dict[str, np.ndarray], metrics: pd.DataFrame,
                null: np.ndarray, observed_ba: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
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
    ax.text(.5, .22, "固定级联时间核 + 训练数据估计的头皮增益",
            ha="center", fontsize=10)
    ax.set_title("A  LGN→皮层→头皮的计算路径")

    ax = axes[0, 1]
    rec = recordings[0]
    chosen = (rec.times_ms >= 50) & (rec.times_ms < 750)
    for channel, color in zip(range(3), ("#2f75a5", "#cf6a32", "#3c8c64")):
        ax.plot(rec.times_ms[chosen], waves[f"A1_observed"][channel, chosen],
                color=color, lw=1.4, label=f"{CHANNELS[channel]} 实测")
        ax.plot(rec.times_ms[chosen], waves[f"A1_fitted"][channel, chosen],
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
    ax.axvline(observed_ba, color="#b14832", lw=2,
               label=f"观察值 {observed_ba:.3f}")
    ax.set_xlabel("块内置换平衡准确率")
    ax.set_ylabel("次数")
    ax.set_title("D  主分析固定流程的置换分布")
    ax.legend()
    fig.suptitle("第二问：基于Q1结果的级联机制与头皮空间协方差判别模型", fontsize=15)
    fig.savefig(output / "第二问_Q1数据新模型.png", dpi=180)
    plt.close(fig)


def save_json(value: dict, path: Path) -> None:
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.generic):
            return clean(x.item())
        return x
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")


def run(output: Path, permutations: int, trees: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    recordings = load_q1_results()
    all_predictions, all_metrics, feature_frames = [], [], []
    primary = None
    for stage_index, stage in enumerate(("before", "v7")):
        x, y, groups, blocks, trial_ids, feature_names = assemble(recordings, stage)
        scores = blocked_scores(x, y, groups, blocks, trees, 202609250 + stage_index * 100)
        all_metrics.extend(metric_rows(y, scores, groups, stage))
        for index in range(len(y)):
            all_predictions.append(dict(
                stage=stage, dataset=list(NAMES)[groups[index]],
                trial_id=int(trial_ids[index]), block=int(blocks[index]),
                truth=int(y[index]), score=float(scores[index]),
                prediction=1 if scores[index] >= 0 else -1,
            ))
        frame = pd.DataFrame(x, columns=feature_names)
        frame.insert(0, "truth", y)
        frame.insert(0, "trial_id", trial_ids)
        frame.insert(0, "dataset", [list(NAMES)[g] for g in groups])
        frame.insert(0, "stage", stage)
        feature_frames.append(frame)
        if stage == "before":
            primary = (x, y, groups, blocks, scores)

    metrics = pd.DataFrame(all_metrics)
    observed_ba = balanced_accuracy(primary[1], primary[4])
    rng = np.random.default_rng(202609252)
    permuted_labels = [permute_within_blocks(primary[1], primary[2], primary[3], rng)
                       for _ in range(permutations)]

    def one_permutation(iteration: int) -> float:
        labels = permuted_labels[iteration]
        scores = blocked_scores(primary[0], labels, primary[2], primary[3],
                                trees, 202700000 + iteration * 10, n_jobs=1)
        return balanced_accuracy(labels, scores)

    null_parts = []
    for start in range(0, permutations, 25):
        stop = min(start + 25, permutations)
        null_parts.extend(Parallel(n_jobs=4, prefer="threads")(
            delayed(one_permutation)(iteration) for iteration in range(start, stop)))
        print(f"permutations {stop}/{permutations}", flush=True)
    null = np.asarray(null_parts)
    p_value = float((1 + np.sum(null >= observed_ba)) / (permutations + 1))

    mechanism, waves = fit_mechanism(recordings)
    pd.DataFrame(all_predictions).to_csv(output / "逐试次留出预测.csv", index=False,
                                         encoding="utf-8-sig")
    metrics.to_csv(output / "判别指标.csv", index=False, encoding="utf-8-sig")
    pd.concat(feature_frames, ignore_index=True).to_csv(
        output / "24维头皮空间协方差特征.csv", index=False, encoding="utf-8-sig")
    mechanism.to_csv(output / "机制拟合与留出检验.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"iteration": np.arange(1, permutations + 1), "BA": null}).to_csv(
        output / "置换分布.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(output / "机制波形.npz", **waves)
    make_figure(output, recordings, mechanism, waves, metrics, null, observed_ba)
    save_json({
        "input": "Q1/eeg_v7_results auditable waveform bundles",
        "input_sha256": {str(rec.source.relative_to(ROOT)): sha256(rec.source)
                          for rec in recordings},
        "mechanism_stage": "v7; known-condition descriptive and held-block fit",
        "decoder_stage": "before; Q1 filtering, baseline and trial rejection without cue-aware V7 correction",
        "decoder": "24 fixed scalp-mode covariance features + ExtraTrees",
        "validation": "hold the same one of five chronological blocks out from every recording",
        "n_trials": int(sum(len(rec.cues) for rec in recordings)),
        "trees": trees,
        "permutations": permutations,
        "permutation_p": p_value,
        "observed_BA": observed_ba,
        "v7_warning": "Q1 V7 uses the true cue to select a direction-specific reference; its decoder row is sensitivity analysis only.",
        "development_warning": "The feature/model family was examined on these four recordings; validation is internal and exploratory.",
        "script_sha256": sha256(Path(__file__)),
    }, output / "运行清单.json")
    print(metrics.to_string(index=False), flush=True)
    print(mechanism.to_string(index=False), flush=True)
    print(f"primary permutation p={p_value:.6f}; output={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()
    if args.permutations < 19 or args.trees < 100:
        parser.error("Use at least 19 permutations and 100 trees")
    run(args.output.expanduser().resolve(), args.permutations, args.trees)


if __name__ == "__main__":
    main()
