#!/usr/bin/env python3
"""Run the pre-specified Q3 V2 optimization without changing V1 outputs."""
from pathlib import Path
import sys,os,argparse,time,json,fcntl
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('MPLCONFIGDIR','/tmp/q3-matplotlib')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from q3.data import ROOT,KEYS,TIME,STAGES,SEED,build_trials,save_csv,save_json,sha,stage_masks
from q3.fit import stage_baselines
from q3.validation import metric_summary,contrast_metrics
from q3.v2.model import candidates,predict
from q3.v2.fit import Bank
from q3.v2.data import attach_past,diagnose

MODELS=('N0','N1','N2','N3','N2_no_projection','N2_no_retrieval','N2_fixed_target','N2_equal_penalty')


def progress(s):
    print(time.strftime('%H:%M:%S'),s,flush=True)


def evaluate(ds,context,progress=progress):
    params=candidates();bank=Bank(ds,params,context)
    models=MODELS if not context else ('B0','N0','N1','N2','N3')
    rows=[];choices=[];preds={m:np.full_like(ds.x,np.nan) for m in models}
    setting='past' if context else 'stimulus'
    for held in range(5):
        train=ds.blocks!=held;base=stage_baselines(ds,train)
        for model in models:
            fit=bank.select(tuple(b for b in range(5) if b!=held),model)
            choices.append(dict(dataset=ds.key,setting=setting,held_block=held,model=model,
                parameters=fit['candidate'].record(),**{k:v for k,v in fit.items() if k!='candidate'}))
            for i in np.flatnonzero(ds.blocks==held):
                e=ds.events[i];p=predict(e,fit['candidate'],fit['coef'],context);preds[model][i]=p
                for j,mask in enumerate(stage_masks(e['target_s'],e['end_s'])):
                    for ch in range(3):
                        y=ds.x[i,mask,ch];yp=p[mask,ch];scale=fit['meta']['channel_scale'][ch]
                        rows.append(dict(dataset=ds.key,setting=setting,trial_id=e['trial_id'],block=held,
                            cue=e['cue'],model=model,stage=STAGES[j],channel=ch,n_samples=int(mask.sum()),
                            MSE=float(np.mean((y-yp)**2)),zero_MSE=float(np.mean(y*y)),
                            baseline_MSE=float(np.mean((y-base[j,ch])**2)),normalized_MSE=float(np.mean(((y-yp)/scale)**2))))
        progress(f'{ds.key} {setting}: outer block {held+1}/5')
    return rows,choices,preds


def intervals(rows,n_boot=2000):
    rng=np.random.default_rng(SEED+200);df=pd.DataFrame(rows);results=[]
    for (key,setting,stage),part in df.groupby(['dataset','setting','stage']):
        block=part.groupby(['block','model']).MSE.mean().unstack('model')
        models=list(block.columns)
        pairs=[('N2',m) for m in models if m!='N2']
        if setting=='past':pairs += [('N0','B0')]
        # Do not pool different input information sets for mechanistic support.
        for model,control in pairs:
            a=block[model].to_numpy();b=block[control].to_numpy()
            ix=rng.integers(0,len(a),(n_boot,len(a)))
            boot=1-a[ix].mean(1)/b[ix].mean(1)
            lo,hi=np.quantile(boot,[.025,.975])
            results.append(dict(dataset=key,setting=setting,stage=stage,model=model,control=control,
                S=1-a.mean()/b.mean(),ci_low=lo,ci_high=hi,n_boot=n_boot))
    return results


def eeg(args,out):
    allrows=[];diags=[];allevents=[]
    legacy=pd.read_csv(ROOT/'q3_result/trial_metrics.csv')
    for key in KEYS:
        ds,audit=build_trials(key);attach_past(ds);diags.append(diagnose(ds,audit));allevents+=audit
        old=np.load(ROOT/f'q3_result/{key}_waves.npz')
        np.testing.assert_array_equal(ds.ids,old['ids']);np.testing.assert_allclose(ds.x,old['x'],equal_nan=True)
        for context in (False,True):
            setting='past' if context else 'stimulus'
            progress(f'{key} {setting}: building bank ({len(ds.ids)} identical V1 trials)')
            rows,choices,preds=evaluate(ds,context)
            if not context:
                extra=legacy[(legacy.dataset==key)&(legacy.model=='M2')].copy()
                extra['setting']=setting;extra['model']='V1_M2';rows+=extra.to_dict('records')
                preds['V1_M2']=old['M2']
            allrows+=rows
            save_csv(out/f'{key}_{setting}_metrics.csv',rows)
            save_json(out/f'{key}_{setting}_choices.json',choices)
            save_csv(out/f'{key}_{setting}_contrasts.csv',contrast_metrics(ds,preds))
            np.savez_compressed(out/f'{key}_{setting}_waves.npz',time=TIME,x=ds.x,ids=ds.ids,
                blocks=ds.blocks,cues=ds.cues,target=[e['target_s'] for e in ds.events],
                end=[e['end_s'] for e in ds.events],**preds)
    save_csv(out/'diagnostics.csv',diags);save_csv(out/'event_audit.csv',allevents)
    save_csv(out/'trial_metrics.csv',allrows)
    summaries=[]
    for setting,part in pd.DataFrame(allrows).groupby('setting'):
        s=metric_summary(part);s['setting']=setting;summaries.extend(s.to_dict('records'))
    save_csv(out/'summary.csv',summaries);save_csv(out/'intervals.csv',intervals(allrows))
    # Context value is a prediction benefit, never evidence of memory sources.
    tmp=pd.DataFrame(allrows);tmp['model']=tmp.setting+'_'+tmp.model;tmp['setting']='context_comparison'
    selected=tmp[tmp.model.isin(['past_N2','stimulus_N2','past_N0'])].copy()
    selected['model']=selected.model.map({'past_N2':'N2','stimulus_N2':'stimulus_N2','past_N0':'past_N0'})
    save_csv(out/'context_intervals.csv',intervals(selected))


def snapshot(out,completed):
    paths=[*sorted((ROOT/'q3').rglob('*.py')),ROOT/'q2/q2model/generative.py',
        ROOT/'q2/q2model/stimulus_shape.py',*sorted((ROOT/'data').glob('*.mat')),
        ROOT/'q3_result/manifest.json',ROOT/'q3_result/trial_metrics.csv']
    save_json(out/'manifest.json',dict(status='complete' if set(completed)=={'eeg','behavior','recovery','render'} else 'partial',
        stages=sorted(completed),seed=SEED+200,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        design='same observations/cohort/folds; nested selection; no post-cue input except separately labelled behavior',
        preregistration='engineering specification saved before V2 results; not external preregistration',
        parameters='encoding-only memory; fixed 50 ms delay; no feedback; target duration .1/.2/.4; directional penalties 1/10',
        interpretation='exploratory optimization on previously examined two participants'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','eeg','behavior','recovery','render'],default='all')
    p.add_argument('--output',type=Path,default=ROOT/'q3_result_v2');args=p.parse_args()
    out=args.output.resolve()
    if out==(ROOT/'q3_result').resolve():p.error('V1 output cannot be overwritten')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    completed=set(json.loads((out/'manifest.json').read_text())['stages']) if (out/'manifest.json').exists() else set()
    with threadpool_limits(limits=1):
        for stage in (('eeg','behavior','recovery','render') if args.stage=='all' else (args.stage,)):
            completed.discard(stage);snapshot(out,completed);progress('START '+stage)
            if stage=='eeg':eeg(args,out)
            elif stage=='behavior':
                from q3.v2.behavior import run
                run(out,progress)
            elif stage=='recovery':
                from q3.v2.recovery import run
                run(out,progress)
            elif stage=='render':
                from q3.v2.report import render
                render(out)
            completed.add(stage);snapshot(out,completed);progress('FINISH '+stage)
    lock.close()


if __name__=='__main__':main()
