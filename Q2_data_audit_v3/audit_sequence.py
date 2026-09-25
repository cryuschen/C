#!/usr/bin/env python3
"""Independent, read-only audit of VisualCog cue timing and past-only controls.

The script writes its own summaries and never modifies the four source MATs or
the existing Q2 analyses. It intentionally does not use channels 4--6 as truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Q2"))
from q2model.data import (NAMES, FS, TIMES, baseline, independent_data,
                          preprocess, window_features)
from optimize_direction_features import score_lda

OUT = Path(__file__).resolve().parent
WINDOWS = [(-.25, -.167), (-.167, -.083), (-.083, 0.)]


def cv_score(x, y, blocks, groups, ids):
    """Same blocked LDA/scaling protocol as Q2 past-only negative control."""
    scores = np.full(len(y), np.nan)
    for held in np.unique(blocks):
        train, test = blocks != held, blocks == held
        z = np.empty_like(x, dtype=float)
        for group in NAMES:
            gm = groups == group
            ref = gm & train
            mu, sd = x[ref].mean(0), np.maximum(x[ref].std(0), 1e-8)
            z[gm] = np.clip((x[gm] - mu) / sd, -8, 8)
        scores[test] = score_lda(z[train], y[train], z[test])
    prediction = np.where(scores >= 0, 1, -1)
    ba = .5 * (np.mean(prediction[y == -1] == -1) +
               np.mean(prediction[y == 1] == 1))
    return float(ba), float(roc_auc_score(y, scores)), scores


def main():
    summary = []
    rows = []
    prep = []
    prep_all = []
    for group_index, key in enumerate(NAMES):
        path = ROOT / "data" / f"VisualCog{key[0]}_Task-{key[1]}.mat"
        mat = loadmat(path, squeeze_me=True)
        data = mat["data"]
        cue, action = data[7], data[8]
        nz = cue != 0
        on = np.flatnonzero(nz & np.r_[True, ~nz[:-1]])
        off = np.flatnonzero(nz & np.r_[~nz[1:], True]) + 1
        labels = cue[on].astype(int)
        action_nz = action != 0
        action_on = np.flatnonzero(action_nz & np.r_[True, ~action_nz[:-1]])
        first_action = action[action_on].astype(int)
        assert len(on) == len(off) == len(action_on) == 100
        assert np.array_equal(data[9], np.arange(len(cue)) / FS)
        assert np.all((action_on - on) / FS > .8)
        dur = (off - on) / FS
        edges = [0] + [(int(on[j - 1]) + int(on[j])) // 2
                       for j in (20, 40, 60, 80)] + [len(cue)]
        guard = np.zeros(100, dtype=bool)
        for block, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
            ix = np.arange(block * 20, (block + 1) * 20)
            guard[ix] = ((on[ix] - 64 >= start + 24 * FS) &
                         (on[ix] + 205 <= end - 24 * FS))
        ds = independent_data(ROOT, key)
        retained = np.zeros(100, dtype=bool)
        retained[ds.ids - 1] = True
        assert np.all(retained <= guard)
        raw_pre = baseline(np.stack([data[:3, o - 64:o] for o in on[ds.ids - 1]]))
        raw_feat = window_features(raw_pre, times=TIMES[:64], windows=WINDOWS)
        filtered_feat = window_features(ds.prestim_only, times=TIMES[:64],
                                        windows=WINDOWS)
        for i in range(100):
            rows.append(dict(dataset=key, trial_id=i + 1, block=i // 20,
                             cue=int(labels[i]), previous_cue=int(labels[i - 1]) if i else 0,
                             second_previous_cue=int(labels[i - 2]) if i > 1 else 0,
                             previous_action=int(first_action[i - 1]) if i else 0,
                             previous_interval_s=float((on[i] - on[i - 1]) / FS) if i else np.nan,
                             cue_duration_s=float(dur[i]),
                             action_delay_s=float((action_on[i] - on[i]) / FS),
                             first_action=int(first_action[i]),
                             guard=bool(guard[i]), retained=bool(retained[i])))
        for j, trial_id in enumerate(ds.ids):
            prep.append(dict(dataset=key, group_index=group_index,
                             block=group_index * 5 + int(ds.blocks[j]),
                             trial_id=int(trial_id), cue=int(ds.y[j]),
                             pre_filtered=filtered_feat[j], pre_raw=raw_feat[j]))
        for i in np.flatnonzero(guard):
            seg = data[:3, on[i] - int(24.25 * FS):on[i]]
            past = preprocess(seg)[:, -64:]
            past -= np.median(past, axis=1, keepdims=True)
            features = window_features(past[None], times=TIMES[:64], windows=WINDOWS)[0]
            prep_all.append(dict(dataset=key, group_index=group_index,
                                 block=group_index * 5 + i // 20,
                                 trial_id=i + 1, cue=int(labels[i]),
                                 pre_filtered=features))
        summary.append(dict(dataset=key, n_cues=100, n_left=int(np.sum(labels == -1)),
                            n_right=int(np.sum(labels == 1)),
                            duration_min_s=float(dur.min()), duration_max_s=float(dur.max()),
                            action_delay_min_s=float(np.min((action_on - on) / FS)),
                            action_delay_max_s=float(np.max((action_on - on) / FS)),
                            n_guard=int(guard.sum()), n_retained=int(retained.sum()),
                            retained_left=int(np.sum(retained & (labels == -1))),
                            retained_right=int(np.sum(retained & (labels == 1))),
                            previous_same_fraction=float(np.mean(labels[1:] == labels[:-1])),
                            action_cue_agree_fraction=float(np.mean(first_action == labels))))
    events = pd.DataFrame(rows)
    pd.DataFrame(summary).to_csv(OUT / "event_summary.csv", index=False)
    events.to_csv(OUT / "trial_events.csv", index=False)
    evaluation = []
    for sample, prefix in [(prep, "retained"), (prep_all, "guard_all")]:
        group = np.array([row["dataset"] for row in sample])
        blocks = np.array([row["block"] for row in sample])
        ids = np.array([row["trial_id"] for row in sample])
        y = np.array([row["cue"] for row in sample])
        event_indices = {k: events.set_index(["dataset", "trial_id"])
                           .loc[list(zip(group, ids)), k].to_numpy()
                         for k in ["previous_cue", "second_previous_cue",
                                   "previous_action", "previous_interval_s"]}
        previous_interval = event_indices["previous_interval_s"].astype(float)
        previous_interval[np.isnan(previous_interval)] = np.nanmedian(previous_interval)
        x_prior = np.column_stack([event_indices["previous_cue"],
                                   event_indices["second_previous_cue"],
                                   event_indices["previous_action"],
                                   previous_interval, ids])
        task = np.array([int(k[1]) for k in group])
        prior_cue_by_task = np.column_stack(
            [event_indices["previous_cue"] * (task == t) for t in (1, 2)])
        prior_cue_by_recording = np.column_stack(
            [event_indices["previous_cue"] * (group == k) for k in NAMES])
        variants = {
            "pre_filtered_9": np.stack([r["pre_filtered"] for r in sample]),
            "previous_cue_only": x_prior[:, [0]],
            "previous_cue_by_task": prior_cue_by_task,
            "previous_cue_by_recording": prior_cue_by_recording,
            "previous_action_only": x_prior[:, [2]],
            "sequence_timing_5": x_prior,
        }
        if prefix == "retained":
            variants["pre_raw_9"] = np.stack([r["pre_raw"] for r in sample])
            variants["pre_filtered_plus_prev_cue_by_task"] = np.column_stack(
                [variants["pre_filtered_9"], prior_cue_by_task])
            variants["pre_filtered_plus_sequence"] = np.column_stack(
                [variants["pre_filtered_9"], x_prior])
        for name, x in variants.items():
            ba, auc, scores = cv_score(x, y, blocks, group, ids)
            evaluation.append(dict(sample=prefix, model=name, n=len(y),
                                   BA=ba, AUC=auc))
            if prefix == "retained":
                for g in NAMES:
                    gm = group == g
                    gg = y[gm]; pp = np.where(scores[gm] >= 0, 1, -1)
                    evaluation.append(dict(sample=f"retained_{g}", model=name,
                                           n=int(gm.sum()),
                                           BA=float(.5 * (np.mean(pp[gg == -1] == -1) +
                                                          np.mean(pp[gg == 1] == 1))),
                                           AUC=float(roc_auc_score(gg, scores[gm]))))
    pd.DataFrame(evaluation).to_csv(OUT / "sequence_controls.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(evaluation).query("sample in ['retained','guard_all']").to_string(index=False))


if __name__ == "__main__":
    main()
