"""Keep V1 observation windows; add strictly pre-cue background information."""
import numpy as np
from scipy.signal import welch
from q3.data import load_raw, audit_events, causal_filter, FS, TIME


def attach_past(ds):
    raw,_,_=load_raw(ds.key)
    events=audit_events(ds.key,raw);onsets=[e['cue_sample'] for e in events]
    edges=[0]+[(onsets[j-1]+onsets[j])//2 for j in (20,40,60,80)]+[raw.shape[1]]
    pieces={b:causal_filter(raw[:3,a:z]) for b,(a,z) in enumerate(zip(edges[:-1],edges[1:]))}
    for e in ds.events:
        on=e['cue_sample'];start=edges[e['block']];past=pieces[e['block']][:,:on-start]
        slopes=[]
        for tau in (.25,1.,4.):
            n=round(tau*FS)
            slopes.append((past[:,-n:].mean(1)-past[:,-2*n:-n].mean(1))/tau)
        e['past_slopes']=np.array(slopes).tolist()
        e['past_last_sample']=on-1
    return ds


def diagnose(ds,audit):
    # Diagnostics use only trials with a complete 0-3 s scoring window.
    mask=(TIME>=0)&(TIME<3)
    valid=np.isfinite(ds.x[:,mask]).all((1,2));x=ds.x[valid][:,mask]
    f,p=welch(x,FS,nperseg=512,axis=1)
    power=np.trapezoid(p,f,axis=1).sum()
    low=np.trapezoid(p[:,f<=1],f[f<=1],axis=1).sum()/power
    flat=x.reshape(-1,3);cov=np.corrcoef(flat.T)
    return dict(dataset=ds.key,n_trials=len(ds.events),n_diagnostic_trials=len(x),
        power_le_1Hz=float(low),common_energy_fraction=float(3*np.sum(x.mean(2)**2)/np.sum(x*x)),
        corr_Fz_F3=float(cov[0,1]),corr_Fz_F4=float(cov[0,2]),corr_F3_F4=float(cov[1,2]),
        target_delay_min=min(e['target_s'] for e in ds.events),target_delay_max=max(e['target_s'] for e in ds.events),
        n_quality_exclusions=sum(e['reason']=='quality' for e in audit),
        n_boundary_exclusions=sum(e['reason']=='block_boundary_24s_buffer' for e in audit))
