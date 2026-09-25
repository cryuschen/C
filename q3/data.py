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


def audit_events(key, d):
    cues = runs(d[7]); markers = runs(d[8]); rows = []
    for i, (on, off, cue) in enumerate(cues):
        next_on = cues[i+1][0] if i+1 < len(cues) else d.shape[1]
        ev = [(s, e, value) for s, e, value in markers if on <= s < next_on]
        platforms = [(s, e, v) for s, e, v in ev if abs(v) == 1]
        clicks = [(s, e, v) for s, e, v in ev if abs(v) == 2]
        valid = len(platforms) == 1 and len(clicks) <= 1
        ts, te, tv = platforms[0] if len(platforms) == 1 else (None, None, None)
        response = clicks[0][0] if len(clicks) == 1 else (te if key.endswith('1') else None)
        source = 'explicit_click' if len(clicks) == 1 else ('platform_end_proxy' if key.endswith('1') and te is not None else 'unknown')
        valid = valid and response is not None and on < ts < response < next_on
        rows.append(dict(dataset=key, trial_id=i+1, block=i//20, cue=int(cue),
            cue_sample=on, cue_end_sample=off, cue_duration_s=(off-on)/FS,
            target_sample=ts, target_status='platform_onset_hypothesis' if ts is not None else 'unknown',
            platform_end_sample=te, platform_sign=tv,
            click_sample=clicks[0][0] if len(clicks)==1 else None,
            click_side=int(np.sign(clicks[0][2])) if len(clicks)==1 else None,
            response_sample=response, response_status=source,
            target_s=(ts-on)/FS if ts is not None else None,
            response_s=(response-on)/FS if response is not None else None,
            rt_s=(response-ts)/FS if response is not None and ts is not None else None,
            correctness=None, correctness_reason='ground_truth_target_mapping_unavailable',
            timeout=None, timeout_reason='deadline_and_timeout_flag_unavailable',
            cue_platform_disagree=bool(tv != cue) if tv is not None else None,
            valid_events=bool(valid), n_platforms=len(platforms), n_clicks=len(clicks)))
    return rows


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


def build_trials(key, gap=.1, mode='causal', pretarget=False, root=ROOT):
    d, labels, path = load_raw(key, root)
    audit = audit_events(key, d)
    onsets = [e['cue_sample'] for e in audit]
    edges = [0] + [(onsets[j-1]+onsets[j])//2 for j in (20,40,60,80)] + [d.shape[1]]
    pieces = {}; kept = []; baseline_scales = []; all_x = []
    for b, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        raw = d[:3, start:stop]
        pieces[b] = causal_filter(raw) if mode == 'causal' else sosfiltfilt(filter_sos(), raw, axis=-1)
    for event in audit:
        e = event.copy(); b = e['block']; start, stop = edges[b:b+2]; on = e['cue_sample']
        end_s = e['target_s'] if pretarget else (e['response_s']-gap if e['response_s'] is not None else None)
        e.update(end_s=end_s, gap_s=gap, filter_mode=mode, pretarget_only=pretarget,
                 filter_start_sample=start, filter_stop_sample=stop, retained=False)
        e['reason'] = 'invalid_events'
        if e['valid_events']:
            end = on + int(np.ceil(end_s*FS))
            e.update(epoch_start_sample=on-64, epoch_end_exclusive=end)
            # Conservative identical buffers for causal and offline sensitivity.
            if on-64 < start+24*FS or end > stop-24*FS or end_s > TIME[-1]:
                e['reason'] = 'block_boundary_24s_buffer'
            else:
                length = end-(on-64)
                raw_epoch = d[:3, on-64:end]
                epoch = pieces[b][:, on-64-start:end-start].copy()
                epoch -= np.median(epoch[:,:64], axis=1, keepdims=True)
                bad, measures = quality(epoch, raw_epoch, np.arange(length)>=64)
                e.update(measures)
                e['reason'] = 'quality' if bad else 'retained'
                if not bad:
                    x = np.full((len(TIME),3), np.nan); x[:length] = epoch.T
                    scale = np.maximum(1.4826*np.median(abs(epoch[:,:64]-np.median(epoch[:,:64],axis=1)[:,None]),axis=1), 1.)
                    all_x.append(x); baseline_scales.append(scale); kept.append(e)
                    e['retained'] = True
        event.update(e)
    if not kept: raise ValueError(f'No valid trials for {key}')
    ds = Trials(key, kept, np.stack(all_x), np.stack(baseline_scales), mode, gap)
    return ds, audit


def prefix_quality_trials(key, root=ROOT):
    """Eligibility depends ONLY on baseline through target+600 ms, not future EEG."""
    d, _, _ = load_raw(key, root); events = audit_events(key, d)
    ons = [r['cue_sample'] for r in events]
    edges=[0]+[(ons[j-1]+ons[j])//2 for j in (20,40,60,80)]+[d.shape[1]]
    arrays=[]; retained=[]; audit=[]
    for e in events:
        row=e.copy(); row.update(retained=False, reason='invalid_events')
        if e['target_s'] is not None:
            on=e['cue_sample']; start,stop=edges[e['block']:e['block']+2]
            # Last included sample MUST be <= 600 ms, not rounded past it.
            end=e['target_sample']+int(np.floor(.6*FS))+1
            row['prefix_end_sample']=end; row['prefix_end_s']=(end-on)/FS
            row['prefix_last_included_after_target_s']=(end-1-e['target_sample'])/FS
            if on-64<start+24*FS or end>stop-24*FS:
                row['reason']='block_boundary_24s_buffer'
            else:
                # Explicit slicing before filtering provides a second future-data barrier.
                f=causal_filter(d[:3,start:end])[:,on-64-start:]
                f-=np.median(f[:,:64],axis=1,keepdims=True)
                bad,measures=quality(f,d[:3,on-64:end],np.arange(f.shape[1])>=64)
                row.update(measures,reason='quality' if bad else 'retained',retained=not bad)
                if not bad:
                    x=np.full((len(TIME),3),np.nan);x[:f.shape[1]]=f.T
                    arrays.append(x);retained.append(row)
        audit.append(row)
    return retained, np.stack(arrays), audit
