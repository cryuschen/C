"""Shared V3 helpers, extracted without changing their numerical definitions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from q2model.data import TIMES, WEIGHTS, preprocess


ROOT = Path(__file__).resolve().parents[1]

# Hashes of the four supplied MAT files from the verified V3 run.
EXPECTED_RAW_SHA256 = {
    'data/VisualCogA_Task-1.mat': '02a8167b243815c3d048763528641f002eeef3b3903a31ea130b1f4e0cfc6b33',
    'data/VisualCogA_Task-2.mat': 'c120ef16ed6bd1a2f1e86e0ab030c6306d08e555a71359cf455744a9904cb748',
    'data/VisualCogB_Task-1.mat': 'e6fb2684bddeed0f43e541a52928ce73c690a9b1f91e23a6404e0537f0ef0fc9',
    'data/VisualCogB_Task-2.mat': 'c58c998880682027119548029ba92240604b9ecf4138e6c4ea3efe4522212e9d',
}


def audit_inputs() -> dict[str, str]:
    """Check the supplied recordings and record the relocated V3 sources."""
    verified = {}
    for name, expected in EXPECTED_RAW_SHA256.items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'Raw input differs from verified V3 experiment: {name}')
        verified[name] = digest
    for name in (
        '服务于脑机接口与精神性疾病诊断的脑电图计算模型.docx',
        'q2/q2model/data.py',
        'q2/q2model/mechanism.py',
        'q2/q2model/stimulus_shape.py',
        'q2/v3_support.py',
    ):
        verified[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return verified


def filtered_template(t: np.ndarray, signal: np.ndarray) -> np.ndarray:
    full = np.pad(signal[None, :], ((0, 0), (2560, 2560)))
    filtered = preprocess(full)[0, 2560:-2560]
    out = np.interp(TIMES, t, filtered)
    out -= np.median(out[:64])
    norm = np.sqrt(np.sum(WEIGHTS * out * out))
    if norm < 1e-10:
        raise ValueError('Template vanished after filtering')
    return out / norm


def mse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum((obs - pred) ** 2 * WEIGHTS[None, :]) / 3)


def score_lda(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Fixed diagonal shrinkage discriminant with balanced class prior."""
    if set(np.unique(y_train)) != {-1, 1}:
        raise ValueError('Both cue classes are required in each training split')
    mu = x_train.mean(axis=0)
    scale = np.maximum(x_train.std(axis=0), 1e-8)
    z = np.clip((x_train - mu) / scale, -8, 8)
    te = np.clip((x_test - mu) / scale, -8, 8)
    left, right = z[y_train == -1], z[y_train == 1]
    ml, mr = left.mean(0), right.mean(0)
    var = ((left - ml) ** 2).sum(0) + ((right - mr) ** 2).sum(0)
    var /= max(len(z) - 2, 1)
    var = .75 * var + .25
    weight = (mr - ml) / np.maximum(var, .25)
    return (te - .5 * (ml + mr)) @ weight
