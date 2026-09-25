#!/usr/bin/env python3
"""Independent arithmetic, saved-coefficient and data-boundary audit of V2."""
from pathlib import Path
import sys,json,argparse
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from q3.data import ROOT,TIME,KEYS,sha,save_json,build_trials
from q3.v2.data import attach_past
from q3.v2.model import Candidate,predict


def run(out):
    manifest=json.loads((out/'manifest.json').read_text())
    mismatches=[p for p,h in manifest['source_hashes'].items() if sha(ROOT/p)!=h]
    assert not mismatches,mismatches
    assert manifest['status']=='complete'
    metrics=pd.read_csv(out/'trial_metrics.csv');n=0;maxerr=0.;predmax=0.;interval_rows=0
    for key in KEYS:
        old=np.load(ROOT/f'q3_result/{key}_waves.npz');ds,_=build_trials(key);attach_past(ds)
        for setting in ('stimulus','past'):
            with np.load(out/f'{key}_{setting}_waves.npz') as archive:
                z={name:archive[name] for name in archive.files}
                np.testing.assert_array_equal(z['ids'],old['ids'])
                np.testing.assert_allclose(z['x'],old['x'],equal_nan=True)
                np.testing.assert_array_equal(z['end'],old['end'])
                index={int(t):i for i,t in enumerate(z['ids'])}
                for row in metrics[(metrics.dataset==key)&(metrics.setting==setting)].itertuples():
                    i=index[row.trial_id];active=(TIME>=0)&(TIME<z['end'][i]);target=z['target'][i]
                    stage=(TIME<.8) if row.stage=='encoding' else ((TIME>=.8)&(TIME<target) if row.stage=='maintenance' else TIME>=target)
                    mask=active&stage
                    value=float(np.mean((z['x'][i,mask,row.channel]-z[row.model][i,mask,row.channel])**2))
                    err=abs(value-row.MSE)/max(abs(value),1.)
                    assert err<1e-10
                    maxerr=max(maxerr,err);n+=1
                choices=json.loads((out/f'{key}_{setting}_choices.json').read_text())
                for fit in choices:
                    held=fit['held_block'];assert held not in fit['train_blocks']
                    train_ids=ds.ids[ds.blocks!=held].tolist();assert train_ids==fit['train_ids']
                    c=Candidate(**fit['parameters']);co=np.array(fit['coef'])
                    # Reconstruct every held prediction from saved parameters and raw past.
                    for i in np.flatnonzero(ds.blocks==held):
                        pred=predict(ds.events[i],c,co,context=setting=='past')
                        delta=float(np.max(abs(pred-z[fit['model']][i])))
                        assert delta<1e-8,(key,setting,fit['model'],delta)
                        predmax=max(predmax,delta)
                if setting=='stimulus':np.testing.assert_array_equal(z['V1_M2'],old['M2'])
    iv=pd.read_csv(out/'intervals.csv')
    for row in iv.itertuples():
        part=metrics[(metrics.dataset==row.dataset)&(metrics.setting==row.setting)&(metrics.stage==row.stage)]
        a=part[part.model==row.model].groupby('block').MSE.mean().mean()
        b=part[part.model==row.control].groupby('block').MSE.mean().mean()
        assert abs(1-a/b-row.S)<1e-12
        assert np.isfinite([row.ci_low,row.ci_high]).all() and row.ci_low<=row.ci_high
        interval_rows+=1
    bp=pd.read_csv(out/'behavior_predictions.csv');bm=pd.read_csv(out/'behavior_metrics.csv')
    oldbp=pd.read_csv(ROOT/'q3_result/behavior_predictions.csv')
    assert set(zip(bp.dataset,bp.trial_id))==set(zip(oldbp.dataset,oldbp.trial_id))
    for r in bm.itertuples():
        part=bp if r.dataset=='pooled' else bp[bp.dataset==r.dataset]
        y=part.rt_s.to_numpy();p=part[r.model].to_numpy()
        assert abs(np.mean(abs(y-p))-r.MAE)<1e-12
        assert abs(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)-r.R2)<1e-10
    bf=pd.read_csv(out/'behavior_features.csv');events=pd.read_csv(out/'behavior_audit.csv')
    for r in bf.itertuples():
        e=events[(events.dataset==r.dataset)&(events.trial_id==r.trial_id)].iloc[0]
        assert e.prefix_last_included_after_target_s<=.6
        assert e.prefix_end_s==r.prefix_end_s and e.response_s>r.prefix_end_s
    br=json.loads((out/'behavior_choices.json').read_text())
    for fit in br:
        held=fit['held_block'];assert fit['prefix_only']
        test=set(bp[(bp.dataset==fit['dataset'])&(bp.block==held)].trial_id)
        assert not test.intersection(fit['train_ids'])
    perm=pd.read_csv(out/'behavior_permutation.csv');null=json.loads((out/'behavior_null_distribution.json').read_text())
    ps=[]
    for r in perm.itertuples():
        vals=np.array(null[r.dataset][r.contrast.removeprefix('cognitive_vs_')])
        assert len(vals)==2000
        p=(1+np.sum(vals[1:]>=vals[0]))/2000
        assert abs(p-r.p_raw)<1e-12;ps.append(p)
    order=np.argsort(ps);adj=np.zeros(6);v=0
    for j,i in enumerate(order):v=max(v,(6-j)*ps[i]);adj[i]=min(1,v)
    np.testing.assert_allclose(adj,perm.p_Holm_6,atol=1e-12)
    for ext in ('png','pdf'):assert len(list((out/'figures').glob('*.'+ext)))==6
    result=dict(status='passed',n_metric_rows_recomputed=n,max_relative_error=maxerr,
        max_reconstructed_prediction_error=predmax,n_interval_point_estimates_recomputed=interval_rows,
        same_V1_observations_and_trials=True,n_behavior_trials=len(bp),source_hash_mismatches=mismatches,
        checks=['same V1 raw scoring target and masks','saved coefficients reproduce all outer predictions',
                'outer training ids exclude held block','behavior prefix and prior-version cohort match',
                'behavior metrics, 1999-permutation p and six-way Holm reproduced','six PNG and six PDF exports'],
        caveats=['numerical audit does not replace visual QA','development-set comparison; not external validation'])
    save_json(out/'独立复核.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'q3_result_v2')
    run(p.parse_args().output)
