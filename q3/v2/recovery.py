"""Known-source waveform recovery; never interpreted as anatomical recovery."""
import numpy as np
from q3.data import ROOT,build_trials,Trials,TIME,save_csv,save_json,SEED
from q3.v2.model import Candidate,candidates,design_bank,predict
from q3.v2.fit import Bank


def run(out,progress):
    ds,_=build_trials('A2');rng=np.random.default_rng(SEED+201)
    truth=Candidate('N2',tau_m=1.,tau_p=.3,target_duration=.2,direction_penalty=1.)
    h=[design_bank([truth],e)[0] for e in ds.events]
    coef=rng.normal(size=(9,3))*[100.,80.,120.]
    coef[5]=0
    clean=np.stack([a@coef for a in h])
    scale=np.sqrt(np.nanmean(clean**2));coef*=100/scale;clean*=100/scale
    old=np.load(ROOT/'q3_result/A2_waves.npz');background=np.nan_to_num(old['x']-old['M2'])
    background/=np.sqrt(np.mean(background**2))
    rows=[];choices=[];params=[c for c in candidates() if c.model=='N2']
    # Fixed single recovery split avoids claiming population coverage from three scenarios.
    train=(0,1,2,3);test=np.flatnonzero(ds.blocks==4)
    for level in (0.,.5,1.):
        observed=clean+level*100*background;observed[~np.isfinite(ds.x)]=np.nan
        synthetic=Trials('A2',ds.events,observed,ds.baseline_scale,ds.mode,ds.gap)
        bank=Bank(synthetic,params);fit=bank.select(train,'N2')
        pred=np.stack([predict(ds.events[i],fit['candidate'],fit['coef']) for i in test])
        mask=np.isfinite(ds.x[test])&(TIME[None,:,None]>=0)
        error=np.sqrt(np.mean((pred[mask]-clean[test][mask])**2))/np.sqrt(np.mean(clean[test][mask]**2))
        c=fit['candidate']
        exact=(c.tau_m==truth.tau_m and c.tau_p==truth.tau_p and c.target_duration==truth.target_duration)
        rows.append(dict(noise_ratio=level,NRMSE_clean=error,exact_parameters=exact,
            tau_m=c.tau_m,tau_p=c.tau_p,target_duration=c.target_duration,direction_penalty=c.direction_penalty))
        choices.append(dict(noise_ratio=level,truth=truth.record(),selected=c.record(),
            coefficients=fit['coef'],selection=fit['selection'],meta=fit['meta'],train_blocks=train,
            train_ids=fit['train_ids'],test_ids=ds.ids[test]))
        progress(f'V2 recovery noise={level}: clean NRMSE={error:.4f}, exact dynamics={exact}')
    save_csv(out/'recovery.csv',rows);save_json(out/'recovery_choices.json',choices)
