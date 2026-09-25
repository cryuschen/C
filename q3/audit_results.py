#!/usr/bin/env python3
"""Independent saved-output arithmetic and provenance audit (no model refitting)."""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]


def audit(out):
    errors=[];checks=[];metrics=pd.read_csv(out/'trial_metrics.csv');max_error=0.;total=0
    events=pd.read_csv(out/'event_audit.csv')
    assert len(events)==400 and not events.duplicated(['dataset','trial_id']).any()
    assert events.correctness.isna().all() and events.timeout.isna().all()
    checks.append('400 unique events; unknown correctness/timeout remain empty')
    for key in ('A1','A2','B1','B2'):
        with np.load(out/f'{key}_waves.npz') as z:
            ids=z['ids'];time=z['time'];source=z['x'];target=z['target'];end=z['end'];models=[m for m in z.files if m.startswith('M')]
            saved_predictions={m:z[m] for m in models}
            rows=metrics[metrics.dataset==key].set_index(['trial_id','model','stage','channel'])
            for i,trial in enumerate(ids):
                for stage in ('encoding','maintenance','retrieval'):
                    mask=(time>=0)&(time<end[i])
                    if stage=='encoding':mask&=time<.8
                    elif stage=='maintenance':mask&=(time>=.8)&(time<target[i])
                    else:mask&=time>=target[i]
                    for model in models:
                        for channel in range(3):
                            value=float(np.mean((source[i,mask,channel]-saved_predictions[model][i,mask,channel])**2))
                            recorded=float(rows.loc[(trial,model,stage,channel),'MSE'])
                            max_error=max(max_error,abs(value-recorded)/max(1,abs(recorded)));total+=1
            choices=json.loads((out/f'{key}_choices.json').read_text())
            for choice in choices:
                test_ids=set(ids[z['blocks']==choice['held_block']].tolist())
                assert not test_ids.intersection(choice['train_ids'])
                assert choice['held_block'] not in choice['train_blocks']
        checks.append(f'{key}: saved waveform errors and outer train/test IDs independently verified')
    assert max_error<1e-10
    summary=pd.read_csv(out/'eeg_summary.csv')
    manual=metrics.groupby(['dataset','model','stage']).MSE.mean()
    for r in summary.itertuples():assert np.isclose(r.MSE,manual.loc[(r.dataset,r.model,r.stage)],rtol=1e-12)
    # Different aggregation is intentional: cognitive intervals use equal block weighting.
    increments=pd.read_csv(out/'cognitive_intervals.csv')
    block=metrics.groupby(['dataset','stage','model','block']).MSE.mean()
    for r in increments.itertuples():
        a=block.loc[(r.dataset,r.stage,'M2')].mean();b=block.loc[(r.dataset,r.stage,r.control)].mean()
        assert np.isclose(r.S_cog,1-a/b,rtol=1e-9,atol=1e-12)
    predictions=pd.read_csv(out/'behavior_predictions.csv');bm=pd.read_csv(out/'behavior_metrics.csv')
    for r in bm.itertuples():
        d=predictions if r.dataset=='pooled' else predictions[predictions.dataset==r.dataset]
        y=d.rt_s.to_numpy();p=d[r.model].to_numpy()
        assert r.n==len(d)
        assert np.isclose(r.MAE,np.mean(abs(y-p)),rtol=1e-10)
        assert np.isclose(r.R2,1-np.sum((y-p)**2)/np.sum((y-y.mean())**2),rtol=1e-9,atol=1e-12)
        assert np.isclose(r.spearman,spearmanr(y,p).statistic,rtol=1e-9,atol=1e-12)
    features=pd.read_csv(out/'behavior_features.csv')
    assert not features.duplicated(['dataset','trial_id']).any()
    assert len(features)==len(predictions)
    merged=features.merge(events[['dataset','trial_id','target_s','response_s']],on=['dataset','trial_id'],validate='one_to_one')
    assert (merged.prefix_end_s<merged.response_s).all()
    assert np.allclose(merged.prefix_end_s-merged.target_s,(np.floor(.6*256)+1)/256)
    assert ((merged.prefix_end_s-merged.target_s)-1/256 <= .6+1e-12).all()
    # Every saved permutation p recomputed from the saved full null distribution.
    null=json.loads((out/'behavior_null_distribution.json').read_text())
    perm=pd.read_csv(out/'behavior_permutation.csv');raw=[]
    for r in perm.itertuples():
        control=r.contrast.removeprefix('cognitive_vs_');values=np.array(null[r.dataset][control])
        p=(1+np.sum(values[1:]>=values[0]))/len(values)
        assert len(values)==r.n_permutations+1 and np.isclose(p,r.p_raw)
        raw.append(p)
    order=np.argsort(raw);corrected=np.empty(len(raw));last=0.
    for j,k in enumerate(order):last=max(last,(len(raw)-j)*raw[k]);corrected[k]=min(1.,last)
    assert np.allclose(perm.p_Holm_6,corrected)
    checks.append('RT metrics, prefix endpoints, full permutation distributions and Holm adjustment recomputed')
    decision=pd.read_csv(out/'decision_trials.csv');ds=pd.read_csv(out/'decision_summary.csv')
    for r in ds.itertuples():
        d=decision[decision.scenario==r.scenario]
        assert len(d)==r.n and np.isclose((d.choice==1).mean(),r.correct_rate)
        assert np.isclose((d.choice==-1).mean(),r.error_rate) and np.isclose((d.choice==0).mean(),r.timeout_rate)
        assert np.isclose(r.correct_rate+r.error_rate+r.timeout_rate,1.)
    checks.append('All simulated outcomes reconcile; no synthetic labels are placed in event audit')
    manifest=json.loads((out/'manifest.json').read_text());hash_mismatches=[]
    for path,expected in manifest['source_hashes'].items():
        actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        if actual!=expected:hash_mismatches.append(path)
    assert not hash_mismatches,hash_mismatches
    figures=list((out/'figures').glob('*.png'))
    assert len(figures)==9 and len(list((out/'figures').glob('*.pdf')))==9
    result=dict(status='passed',n_metric_rows_recomputed=total,max_relative_arithmetic_error=max_error,
        n_events=400,n_main_trials=int(events.retained.sum()),n_behavior_trials=len(predictions),
        n_pngs=len(figures),source_hash_mismatches=hash_mismatches,checks=checks,
        caveats=['numeric audit is not visual QA','synthetic recovery is not biological source identification',
                 'only two participants; exploratory statistics; behavior labels partially unknown'])
    (out/'独立复核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'q3_result')
    audit(p.parse_args().output)
