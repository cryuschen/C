"""Fixed, label-independent preprocessing and sample-disjoint temporal blocks."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt

FS = 256
TIMES = np.arange(-64, 205) / FS
CHANNELS = ['Fz', 'F3', 'F4']
B = np.array([[1/np.sqrt(3), 2/np.sqrt(6), 0],
              [1/np.sqrt(3), -1/np.sqrt(6), 1/np.sqrt(2)],
              [1/np.sqrt(3), -1/np.sqrt(6), -1/np.sqrt(2)]])
WINDOWS = [(0.05, .25), (.25, .5), (.5, .75 + 1e-10)]
WEIGHTS = np.zeros(len(TIMES))
for lo, hi in WINDOWS:
    mask = (TIMES >= lo) & (TIMES < hi)
    WEIGHTS[mask] = 1 / mask.sum() / 3
FIT_MASK = WEIGHTS > 0
NAMES = {'A1': '受试者A_项目一', 'A2': '受试者A_项目二',
         'B1': '受试者B_项目一', 'B2': '受试者B_项目二'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding='utf8')
    tmp.replace(path)


def save_csv(rows, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def filters():
    return (*iirnotch(60, 30, fs=FS), butter(4, [.1, 30], btype='bandpass', fs=FS, output='sos'))


def preprocess(x):
    b, a, sos = filters()
    return sosfiltfilt(sos, filtfilt(b, a, x, axis=-1), axis=-1)


def baseline(x):
    return x - np.median(x[..., :64], axis=-1, keepdims=True)


def epochs(x, onsets):
    return np.stack([x[:, i-64:i+205] for i in onsets])


def quality(raw, x):
    sat = (np.abs(raw) >= 999).sum(axis=(1, 2))
    ptp = np.ptp(x, axis=-1).max(axis=1)
    jump = np.abs(np.diff(x, axis=-1)).max(axis=(1, 2))
    bad = (sat >= 15) | (ptp > 1800) | (jump > 600)
    return bad, np.column_stack([sat, ptp, jump])


@dataclass
class Dataset:
    key: str
    x: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    blocks: np.ndarray
    durations: np.ndarray
    onset: np.ndarray
    artifact: np.ndarray
    prestim_only: np.ndarray

    def subset(self, idx):
        return Dataset(self.key, *[getattr(self, k)[idx] for k in
            ('x', 'y', 'ids', 'blocks', 'durations', 'onset', 'artifact', 'prestim_only')])

    def labels(self, y):
        return replace(self, y=np.asarray(y))


def load_raw(root, key):
    p = Path(root) / 'data' / f'VisualCog{key[0]}_Task-{key[1]}.mat'
    m = loadmat(p, squeeze_me=True)
    x = m['data']
    if int(m['SampleRate']) != FS or list(m['DataLabel'][:3]) != CHANNELS:
        raise ValueError(f'{p}: unexpected sampling rate or channel order')
    if x.shape[0] != 10 or not np.isfinite(x).all():
        raise ValueError(f'{p}: invalid data')
    cue = x[7]
    on = np.flatnonzero((cue != 0) & np.r_[True, cue[:-1] == 0])
    off = np.flatnonzero((cue != 0) & np.r_[cue[1:] == 0, True]) + 1
    if len(on) != 100 or len(off) != 100 or not np.isin(cue[on], [-1, 1]).all():
        raise ValueError('Expected 100 complete cues with +/-1 directions')
    if any(not np.all(cue[a:b] == cue[a]) for a, b in zip(on, off)):
        raise ValueError('Cue changes sign within event')
    return x[:3], on, cue[on].astype(int), (off-on)/FS


def independent_data(root, key, guard=24., out=None):
    raw, on, y, durations = load_raw(root, key)
    edges = [0] + [(int(on[j-1])+int(on[j]))//2 for j in (20, 40, 60, 80)] + [raw.shape[1]]
    audit, block_audit, pieces = [], [], []
    for k, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        ids = np.arange(k*20, (k+1)*20)
        valid = (on[ids]-64 >= s+int(guard*FS)) & (on[ids]+205 <= e-int(guard*FS))
        selected = ids[valid]
        filt = preprocess(raw[:, s:e])
        ep = baseline(epochs(filt, on[selected]-s))
        raw_ep = epochs(raw, on[selected])
        bad, art = quality(raw_ep, ep)
        # A strict pre-stimulus control: filter ONLY the preceding 24.25 seconds.
        pre = []
        for i in selected:
            segment = raw[:, on[i]-int((guard+.25)*FS):on[i]]
            past = preprocess(segment)[:, -64:]
            pre.append(past - np.median(past, axis=1, keepdims=True))
        pre = np.stack(pre)
        for j in ids:
            row = dict(dataset=key, trial_id=int(j+1), block=k, onset_sample=int(on[j]),
                       epoch_start=int(on[j]-64), epoch_end_exclusive=int(on[j]+205),
                       cue=int(y[j]), duration_ms=float(durations[j]*1000),
                       filter_start=s, filter_end_exclusive=e, guard_seconds=guard,
                       retained=False, reason='filter_boundary_buffer')
            if j in selected:
                z = int(np.flatnonzero(selected == j)[0])
                row.update(saturated_samples=art[z,0], PTP=art[z,1], max_jump=art[z,2],
                           retained=bool(~bad[z]), reason='quality' if bad[z] else 'retained')
            audit.append(row)
        keep = ~bad; ii = selected[keep]
        pieces.append(Dataset(key, ep[keep], y[ii], ii+1, np.full(len(ii), k),
                              durations[ii], on[ii], art[keep], pre[keep]))
        block_audit.append(dict(dataset=key, block=k, start=s, end_exclusive=e,
                                total=20, buffer_retained=len(selected), retained=int(keep.sum()),
                                left=int((y[ii]==-1).sum()), right=int((y[ii]==1).sum())))
    ds = Dataset(key, *[np.concatenate([getattr(p, f) for p in pieces]) for f in
                       ('x','y','ids','blocks','durations','onset','artifact','prestim_only')])
    if out:
        save_csv(audit, Path(out)/'audit'/f'{key}_events_guard{guard:g}.csv')
        save_csv(block_audit, Path(out)/'audit'/f'{key}_blocks_guard{guard:g}.csv')
        np.savez_compressed(Path(out)/'audit'/f'{key}_independent_guard{guard:g}.npz',
                            **{k:v for k,v in vars(ds).items() if k!='key'}, times=TIMES)
    return ds


def descriptive_data(root, key, stage):
    p = Path(root)/'eeg_v7_results'/NAMES[key]/'可复核波形.npz'
    with np.load(p) as w:
        raw, on, y, durations = load_raw(root, key)
        ids = w['trial_ids'].copy(); idx = ids-1
        if not np.array_equal(y[idx], w['cues']): raise ValueError('Q1 cue lineage mismatch')
        return Dataset(key, w[stage].copy(), y[idx], ids, idx//20, durations[idx],
                       on[idx], np.zeros((len(idx),3)), w[stage][:,:,:64].copy())


def modes(x):
    return np.einsum('cm,nct->nmt', B, x)


def means(x, y):
    if not all(np.any(y==c) for c in (-1,1)): raise ValueError('Both directions required')
    return np.stack([x[y==c].mean(axis=0) for c in (-1,1)])


def window_features(x, times=TIMES, windows=WINDOWS):
    z = modes(x)
    return np.stack([z[..., (times>=a)&(times<b)].mean(-1) for a,b in windows], axis=-1).reshape(len(x),-1)
