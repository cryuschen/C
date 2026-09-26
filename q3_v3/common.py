"""Auditable events, causal processing and variable-length cognitive windows."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
FS = 256
SEED = 202609253
KEYS = ('A1', 'A2', 'B1', 'B2')
CHANNELS = ('Fz', 'F3', 'F4')
# Fixed horizon independent of a trial's response time. No time warping.
TIME = np.arange(-64, 1281) / FS
STAGES = ('encoding', 'maintenance', 'retrieval')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, Path): return str(x)
        return x
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf8')


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def filter_sos():
    b, a = iirnotch(60, 30, fs=FS)
    return np.vstack([tf2sos(b, a), butter(4, [.1, 30], fs=FS, btype='bandpass', output='sos')])


def causal_filter(x):
    """No sample at t depends on x after t; steady constant initial condition."""
    x = np.asarray(x, dtype=float)
    sos = filter_sos()
    zi = sosfilt_zi(sos)[:, None, :] * x[None, :, :1]
    return sosfilt(sos, x, axis=-1, zi=zi)[0]


def load_raw(key, root=ROOT):
    path = Path(root) / 'data' / f'VisualCog{key[0]}_Task-{key[1]}.mat'
    m = loadmat(path, squeeze_me=True)
    d = m['data']
    if int(m['SampleRate']) != FS or list(m['DataLabel'][:3]) != list(CHANNELS):
        raise ValueError('Unexpected sampling rate / channels')
    if d.shape[0] != 10 or not np.isfinite(d).all():
        raise ValueError('Invalid raw data')
    if not np.allclose(d[9], np.arange(d.shape[1])/FS, atol=1e-10):
        raise ValueError('Invalid sample timestamps')
    return d, m['DataLabel'], path


def runs(a):
    starts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    ends = np.r_[starts[1:], len(a)]
    return [(int(s), int(e), float(a[s])) for s, e in zip(starts, ends) if a[s] != 0]


def stage_masks(target, end):
    active = (TIME >= 0) & (TIME < end)
    return np.array([active & (TIME < .8), active & (TIME >= .8) & (TIME < target),
                     active & (TIME >= target)])


def sample_weights(target, end):
    masks = stage_masks(target, end)
    nonempty = masks.sum(1) > 0
    weights = np.zeros(len(TIME))
    for m in masks[nonempty]: weights[m] = 1 / (m.sum() * nonempty.sum())
    return weights


def quality(x, raw, mask):
    """Q1 thresholds; saturation scaled by duration instead of fixed 15 points."""
    r = raw[:, mask]; f = x[:, mask]
    saturation_fraction = np.mean(np.abs(r) >= 999)
    ptp = np.ptp(f, axis=1).max()
    jump = np.abs(np.diff(f, axis=1)).max()
    bad = saturation_fraction >= 15/(3*269) or ptp > 1800 or jump > 600
    return bool(bad), dict(saturation_fraction=float(saturation_fraction), PTP=float(ptp), max_jump=float(jump))


@dataclass
class Trials:
    key: str
    events: list
    x: np.ndarray  # trial, time, channel; outside permitted endpoint is NaN
    baseline_scale: np.ndarray  # trial, channel, past-only
    mode: str
    gap: float

    @property
    def blocks(self): return np.array([e['block'] for e in self.events])
    @property
    def ids(self): return np.array([e['trial_id'] for e in self.events])
    @property
    def cues(self): return np.array([e['cue'] for e in self.events])
    @property
    def weights(self): return np.stack([sample_weights(e['target_s'], e['end_s']) for e in self.events])


