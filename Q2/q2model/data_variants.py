"""Independent-block preprocessing alternatives for Q2 development experiments.

This module does not alter the archived 0.1 Hz / 24 s experiment. Every filter
is fitted to a disjoint continuous block, and acceptance rules ignore labels.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt

from .data import FS, Dataset, baseline, epochs, load_raw, quality


def filter_block(raw: np.ndarray, highpass_hz: float) -> np.ndarray:
    b, a = iirnotch(60, 30, fs=FS)
    sos = butter(4, [highpass_hz, 30], btype='bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, filtfilt(b, a, raw, axis=-1), axis=-1)


def filter_tail_ratio(highpass_hz: float, seconds: float = 8.) -> float:
    """Absolute impulse-response tail beyond guard / peak response."""
    n = 60 * FS
    impulse = np.zeros((3, n))
    impulse[:, n // 2] = 1.
    response = filter_block(impulse, highpass_hz)[0]
    distance = int(seconds * FS)
    tail = np.max(np.abs(np.r_[response[:n // 2 - distance],
                                response[n // 2 + distance + 1:]]))
    return float(tail / np.max(np.abs(response)))


def independent_data_variant(root, key: str, highpass_hz: float = .5,
                             guard_seconds: float = 8.) -> Dataset:
    if highpass_hz < .3 or guard_seconds < 6:
        raise ValueError('Use a short guard only with a high-pass of at least 0.3 Hz')
    raw, on, y, durations = load_raw(root, key)
    edges = [0] + [(int(on[j - 1]) + int(on[j])) // 2
                   for j in (20, 40, 60, 80)] + [raw.shape[1]]
    pieces = []
    guard = int(round(guard_seconds * FS))
    for block, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        all_ids = np.arange(block * 20, (block + 1) * 20)
        use = all_ids[(on[all_ids] - 64 >= start + guard) &
                      (on[all_ids] + 205 <= end - guard)]
        filtered = filter_block(raw[:, start:end], highpass_hz)
        x = baseline(epochs(filtered, on[use] - start))
        raw_epochs = epochs(raw, on[use])
        bad, artifact = quality(raw_epochs, x)
        prestim = []
        for i in use:
            segment = raw[:, on[i] - guard - 64:on[i]]
            pre = filter_block(segment, highpass_hz)[:, -64:]
            prestim.append(pre - np.median(pre, axis=1, keepdims=True))
        selected = use[~bad]
        pieces.append(Dataset(key, x[~bad], y[selected], selected + 1,
                              np.full(len(selected), block), durations[selected],
                              on[selected], artifact[~bad],
                              np.stack(prestim)[~bad]))
    return Dataset(key, *[np.concatenate([getattr(p, field) for p in pieces])
                          for field in ('x', 'y', 'ids', 'blocks', 'durations',
                                        'onset', 'artifact', 'prestim_only')])
