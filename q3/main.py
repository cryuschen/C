#!/usr/bin/env python3
"""Reproducible Q3 pipeline. Examples are in README.md; no Q1/Q2 output writes."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import fcntl

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

os.environ.setdefault('MPLCONFIGDIR','/tmp/q3-matplotlib')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from q3.data import ROOT,KEYS,TIME,SEED,build_trials,save_csv,save_json,sha,STAGES,stage_masks
from q3.model import candidates,MODELS,Candidate,predict,visual_response,slow_states
from q3.fit import Bank,evaluate,stage_baselines
from q3.validation import metric_summary,cognitive_intervals,contrast_metrics,recovery_experiment,decision_simulation


def progress(message):
    print(time.strftime('%H:%M:%S'),message,flush=True)


def export_eeg(out,key,ds,rows,choices,preds):
    save_csv(out/f'{key}_trial_metrics.csv',rows)
    save_csv(out/f'{key}_summary.csv',metric_summary(rows))
    save_csv(out/f'{key}_contrasts.csv',contrast_metrics(ds,preds))
    save_json(out/f'{key}_choices.json',choices)
    np.savez_compressed(out/f'{key}_waves.npz',time=TIME,x=ds.x,ids=ds.ids,blocks=ds.blocks,cues=ds.cues,
        target=np.array([e['target_s'] for e in ds.events]),end=np.array([e['end_s'] for e in ds.events]),
        **preds)


def main_eeg(args,out):
    allrows=[];all_audit=[]
    for key in KEYS:
        ds,audit=build_trials(key);all_audit+=audit
        progress(f'{key}: {len(ds.events)}/100 long-window trials retained; building full candidate bank')
        bank=Bank(ds,candidates(int(key[1]),args.quick))
        rows,choices,preds,fits=evaluate(bank,MODELS,progress)
        allrows+=rows;export_eeg(out,key,ds,rows,choices,preds)
        save_csv(out/f'{key}_event_audit.csv',audit)
        intervals=cognitive_intervals(rows,args.bootstraps);save_csv(out/f'{key}_cognitive_intervals.csv',intervals)
        # Endpoint sensitivities use exactly the already selected outer-fold model.
        sensitivity=[]
        for gap in (.05,.15,.20):
            other,_=build_trials(key,gap=gap);lookup={e['trial_id']:i for i,e in enumerate(ds.events)}
            for i,e in enumerate(other.events):
                if e['trial_id'] not in lookup:continue
                held=e['block'];base=stage_baselines(ds,ds.blocks!=held)
                for model in MODELS:
                    fit=fits[(held,model)];pred=predict(e,fit['candidate'],fit['coef'])
                    for j,mask in enumerate(stage_masks(e['target_s'],e['end_s'])):
                        if not mask.any():continue
                        sensitivity.append(dict(dataset=key,trial_id=e['trial_id'],block=held,gap_s=gap,
                            model=model,stage=STAGES[j],MSE=float(np.mean((other.x[i,mask]-pred[mask])**2)),
                            method='frozen_main_fit_common_trial_rescore'))
        save_csv(out/f'{key}_endpoint_sensitivity.csv',sensitivity)
        del bank
    save_csv(out/'trial_metrics.csv',allrows);save_csv(out/'event_audit.csv',all_audit)
    save_csv(out/'eeg_summary.csv',metric_summary(allrows))
    save_csv(out/'cognitive_intervals.csv',cognitive_intervals(allrows,args.bootstraps))


def sensitivity_eeg(args,out):
    folder=out/'sensitivity';folder.mkdir(exist_ok=True)
    for mode,keys in [('zero_phase',KEYS),('pretarget',('A1','B1'))]:
        for key in keys:
            ds,audit=build_trials(key,mode='zero_phase' if mode=='zero_phase' else 'causal',pretarget=mode=='pretarget')
            progress(f'{key} {mode}: refitting on {len(ds.events)} trials')
            bank=Bank(ds,candidates(int(key[1]),args.quick))
            rows,choices,preds,fits=evaluate(bank,MODELS[:4],progress)
            export_eeg(folder,f'{key}_{mode}',ds,rows,choices,preds)
            save_csv(folder/f'{key}_{mode}_audit.csv',audit)
            save_csv(folder/f'{key}_{mode}_intervals.csv',cognitive_intervals(rows,args.bootstraps))
            del bank


def behavior(args,out):
    from q3.behavior import run_behavior
    result=run_behavior(args.quick,args.permutations,args.bootstraps,progress)
    for key,value in result.items():
        if key in ('choices','null_distribution'):save_json(out/f'behavior_{key}.json',value)
        else:save_csv(out/f'behavior_{key}.csv',value)


def simulation(args,out):
    ds,_=build_trials('A2')
    recovery,profiles=recovery_experiment(ds,candidates(2,args.quick),args.quick,progress)
    save_json(out/'recovery.json',recovery);save_csv(out/'recovery_profiles.csv',profiles)
    rows,paths,detail=decision_simulation(1000 if args.quick else 5000)
    save_csv(out/'decision_summary.csv',rows);save_csv(out/'decision_trials.csv',detail)
    np.savez_compressed(out/'decision_paths.npz',**{f'{name}_{k}':v for name,items in paths.items() for k,v in items.items()})


def manifest(out,args,completed):
    import scipy,matplotlib
    files=[*sorted((ROOT/'q3').rglob('*.py')),ROOT/'q2/q2model/generative.py',ROOT/'q2/q2model/stimulus_shape.py',
           ROOT/'服务于脑机接口与精神性疾病诊断的脑电图计算模型.docx',*sorted((ROOT/'data').glob('*.mat'))]
    save_json(out/'manifest.json',dict(status='complete' if completed=={'eeg','sensitivity','behavior','simulation','render'} else 'partial',
        completed_stages=sorted(completed),quick=args.quick,seed=SEED,permutations=args.permutations,
        bootstraps=args.bootstraps,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
        versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
        methodological_defaults=dict(causal_filter='60Hz notch Q30 + fourth-order 0.1-30Hz Butterworth',
           blocks=5,block_buffer_s=24,main_response_gap_s=.1,prefix_after_target_s=.6,
           horizon_s=5,target_common_input_duration_s=.2,screening='all-grid nested ridge; top2 pairs nested Huber IRLS',
           unknown_correctness='null',unknown_timeout='null',units='original recording units'),
        limitations=['two previously explored participants','three frontal channels, no EOG or anatomy',
                    'target onset inferred from platform structure','Task-1 response endpoint is a proxy',
                    'no empirical correctness or deadline labels','source dynamics need not be uniquely identifiable'] ))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'q3_result')
    parser.add_argument('--stage',choices=['all','eeg','sensitivity','behavior','simulation','render'],default='all')
    parser.add_argument('--quick',action='store_true',help='smoke only; MUST use a separate output path')
    parser.add_argument('--permutations',type=int,default=1999)
    parser.add_argument('--bootstraps',type=int,default=2000)
    args=parser.parse_args()
    if args.quick:
        if args.output.resolve()==(ROOT/'q3_result').resolve():parser.error('--quick requires a separate --output directory')
        args.permutations=min(args.permutations,39);args.bootstraps=min(args.bootstraps,100)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # Prevent interleaved stage outputs from two invocations in the same folder.
    run_lock=(out/'.run.lock').open('a+')
    try:
        fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Another Q3 process is using {out}; use a different output directory') from exc
    completed=set()
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if old['quick']!=args.quick:raise ValueError('Cannot mix smoke and formal output')
        if old['permutations']!=args.permutations or old['bootstraps']!=args.bootstraps:
            raise ValueError('Use the same resampling settings or a new output path')
        completed=set(old['completed_stages'])
    stages=['eeg','sensitivity','behavior','simulation','render'] if args.stage=='all' else [args.stage]
    with threadpool_limits(limits=1):
        for stage in stages:
            progress(f'START {stage}')
            completed.discard(stage);manifest(out,args,completed)
            if stage=='eeg':main_eeg(args,out)
            elif stage=='sensitivity':sensitivity_eeg(args,out)
            elif stage=='behavior':behavior(args,out)
            elif stage=='simulation':simulation(args,out)
            elif stage=='render':
                from q3.reporting import render_report
                render_report(out)
            completed.add(stage);manifest(out,args,completed)
            progress(f'FINISH {stage}')
    run_lock.close()


if __name__=='__main__':main()
