"""Response-aware EEG windows. Unknown markers are not clinical timeouts."""
import numpy as np
from .common import FS, TIME, ROOT, Trials, load_raw, runs, causal_filter, quality


def audit_events(key, data):
    cues, markers = runs(data[7]), runs(data[8])
    rows = []
    for i, (on, off, cue) in enumerate(cues):
        nxt = cues[i+1][0] if i+1 < len(cues) else data.shape[1]
        ev = [(s, e, v) for s, e, v in markers if on <= s < nxt]
        platforms = [(s, e, v) for s, e, v in ev if abs(v) == 1]
        clicks = [(s, e, v) for s, e, v in ev if abs(v) == 2]
        ts, te, sign = platforms[0] if len(platforms) == 1 else (None, None, None)
        target_valid = ts is not None and on < ts < nxt
        marker_valid = len(clicks) <= 1
        click = clicks[0][0] if len(clicks) == 1 else None
        if click is not None and (not target_valid or not ts < click < nxt):
            marker_valid = False
        response, status = None, 'unobserved'
        if marker_valid and click is not None:
            response, status = click, 'explicit_click'
        elif marker_valid and key.endswith('1') and target_valid and ts < te < nxt:
            response, status = te, 'platform_end_proxy'
        row = dict(dataset=key, trial_id=i+1, block=i//20, cue=int(cue),
            cue_sample=on, cue_end_sample=off, cue_duration_s=(off-on)/FS,
            next_cue_sample=nxt, record_stop_sample=data.shape[1],
            target_sample=ts, target_s=(ts-on)/FS if target_valid else None,
            target_status='platform_onset_hypothesis' if target_valid else 'unknown',
            platform_end_sample=te, platform_sign=sign,
            click_sample=click, click_side=int(np.sign(clicks[0][2])) if len(clicks)==1 else None,
            response_sample=response, response_s=(response-on)/FS if response is not None else None,
            rt_s=(response-ts)/FS if response is not None else None, response_status=status,
            correctness=None, correctness_reason='target_ground_truth_unavailable',
            timeout=None, timeout_reason='experimental_deadline_unavailable',
            n_platforms=len(platforms), n_clicks=len(clicks),
            cue_platform_disagree=bool(sign!=cue) if sign is not None else None,
            valid_events=bool(target_valid and marker_valid),
            event_issue='none' if target_valid and marker_valid else 'ambiguous_or_invalid_markers')
        rows.append(row)
    return rows


def window_policy(event, horizon=2., gap=.1):
    """All correctness values are eligible. Administrative cutoff != timeout.

    Missing-marker records have a finite EEG window but are not automatically
    valid right-censored behavior: ascertainment of the response is unknown.
    """
    if horizon <= 0 or gap < 0:
        raise ValueError('Invalid window policy')
    e = dict(event)
    e.update(horizon_s=float(horizon), gap_s=gap, retained=False,
             end_s=None, behavior_duration_s=None, behavior_event=None,
             behavior_survival_eligible=False, eeg_administrative_cutoff=False)
    if not e['valid_events']:
        e['window_status']='unusable_events'
        return e
    target = e['target_s']
    administrative = target+horizon
    available = min((e['next_cue_sample']-e['cue_sample'])/FS,
                    (e['record_stop_sample']-e['cue_sample'])/FS, float(TIME[-1]))
    bound = min(administrative, available)
    response = e['response_s']
    # Gap is attached to an actual/proxy response, not subtracted from censoring.
    e['end_s'] = min(bound, response-gap) if response is not None else bound
    e['eeg_administrative_cutoff'] = response is None or bound < response-gap
    e['window_status'] = ('response_minus_gap' if not e['eeg_administrative_cutoff']
                          else 'bounded_observation')
    e['bound_reason'] = 'configured_horizon' if administrative <= available else 'record_or_next_trial'
    duration = max(0., bound-target)
    if e['response_status']=='explicit_click':
        e['behavior_survival_eligible']=duration>0
        e['behavior_event']=int(response <= bound)
        e['behavior_duration_s']=min(response-target,duration)
        e['behavior_status']='observed_click' if e['behavior_event'] else 'administratively_right_censored'
    elif e['response_status']=='platform_end_proxy':
        e['behavior_status']='proxy_only_not_observed_click'
    else:
        e['behavior_status']='unknown_response_or_missing_marker'
    if e['end_s'] <= target:
        e['window_status']='no_posttarget_observation'
    return e


def build_trials(key, horizon=2., gap=.1, raw=None, root=ROOT):
    data = load_raw(key, root)[0] if raw is None else raw
    audited = audit_events(key,data)
    ons=[e['cue_sample'] for e in audited]
    edges=[0]+[(ons[j-1]+ons[j])//2 for j in (20,40,60,80)]+[data.shape[1]]
    pieces={b:causal_filter(data[:3,a:z]) for b,(a,z) in enumerate(zip(edges[:-1],edges[1:]))}
    kept=[];arrays=[];scales=[];rows=[]
    for event in audited:
        e=window_policy(event,horizon,gap);b=e['block'];on=e['cue_sample'];a,z=edges[b:b+2]
        e.update(filter_start_sample=a,filter_stop_sample=z,filter_mode='causal',reason='invalid_events')
        if e['end_s'] is not None and e['end_s']>e['target_s']:
            end=on+int(np.ceil(e['end_s']*FS));length=end-(on-64)
            e.update(epoch_start_sample=on-64,epoch_end_exclusive=end)
            if on-64<a+24*FS or end>z-24*FS:
                e['reason']='block_boundary_24s_buffer'
            else:
                epoch=pieces[b][:,on-64-a:end-a].copy()
                epoch-=np.median(epoch[:,:64],axis=1,keepdims=True)
                bad,measures=quality(epoch,data[:3,on-64:end],np.arange(length)>=64)
                e.update(measures,retained=not bad,reason='quality' if bad else 'retained')
                if not bad:
                    x=np.full((len(TIME),3),np.nan);x[:length]=epoch.T
                    arrays.append(x);kept.append(e)
                    scales.append(np.maximum(1.4826*np.median(abs(epoch[:,:64]),axis=1),1.))
        rows.append(e)
    if not kept: raise ValueError(f'No valid EEG trials for {key}')
    return Trials(key,kept,np.stack(arrays),np.stack(scales),'causal',gap),rows


def product_limit(duration,event):
    """Kaplan-Meier for observed response time or valid administrative censoring."""
    duration=np.asarray(duration,float);event=np.asarray(event,int)
    if len(duration)!=len(event) or np.any(duration<0) or not np.isin(event,[0,1]).all():
        raise ValueError('Invalid survival observations')
    survival=1.;rows=[]
    for t in np.unique(duration):
        risk=int(np.sum(duration>=t));d=int(np.sum((duration==t)&(event==1)))
        cens=int(np.sum((duration==t)&(event==0)))
        survival*=1-d/risk
        rows.append(dict(time_s=float(t),at_risk=risk,events=d,censored=cens,survival=survival))
    return rows
