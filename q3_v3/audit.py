"""Independent checks on saved V3 observations, errors and simulated outcomes."""
from pathlib import Path
import sys,json,argparse,hashlib
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from q3_v3.common import ROOT,KEYS,save_json


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(out):
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf8'))
    for path,h in manifest['source_hashes'].items():assert digest(ROOT/path)==h,path
    for path,h in manifest['inherited_hashes'].items():
        assert digest(out/'inherited_v2'/path)==h,path
        assert digest(ROOT/'q3_result_v2'/path)==h,('original V2 changed',path)
    ev=pd.read_csv(out/'event_audit.csv')
    assert len(ev)==400 and not ev.duplicated(['dataset','trial_id']).any()
    assert ev.correctness.isna().all() and ev.timeout.isna().all()
    assert sum(ev.response_status=='explicit_click')==200 and sum(ev.response_status=='platform_end_proxy')==200
    assert not ev.loc[ev.response_status!='explicit_click','behavior_survival_eligible'].any()
    for row in ev[ev.retained].itertuples():
        assert row.epoch_end_exclusive<=row.filter_stop_sample-24*256
        assert row.epoch_start_sample>=row.filter_start_sample+24*256
        assert row.end_s<=row.target_s+2.+1e-12
        if np.isfinite(row.response_s):assert row.end_s<=row.response_s-.1+1e-12
    df=pd.read_csv(out/'trial_metrics.csv');maxerr=0.;count=0
    for key in KEYS:
        for setting in ('stimulus','past'):
            # Decompress each array once, rather than once per metric row.
            with np.load(out/f'{key}_{setting}_waves.npz') as archive:
                z={name:archive[name] for name in archive.files}
            with np.load(out/f'inherited_v2/{key}_{setting}_waves.npz') as archive:
                old={name:archive[name] for name in archive.files}
            np.testing.assert_array_equal(z['ids'],old['ids'])
            np.testing.assert_allclose(z['x'],old['x'],atol=1e-7,rtol=1e-9,equal_nan=True)
            np.testing.assert_array_equal(z['end'],old['end'])
            fits=json.loads((out/f'inherited_v2/{key}_{setting}_choices.json').read_text(encoding='utf8'))
            for f in fits:
                assert f['train_ids']==z['ids'][z['blocks']!=f['held_block']].tolist()
                a,b=z[f['model']],old[f['model']]
                assert np.max(abs(a-b))<1e-4
                assert np.linalg.norm(a-b)/max(np.linalg.norm(b),1.)<1e-7
            lookup={int(t):i for i,t in enumerate(z['ids'])}
            for r in df[(df.dataset==key)&(df.setting==setting)].itertuples():
                i=lookup[r.trial_id];t=z['time'];active=(t>=0)&(t<z['end'][i]);target=z['target'][i]
                stage=(t<.8) if r.stage=='encoding' else ((t>=.8)&(t<target) if r.stage=='maintenance' else t>=target)
                mask=active&stage
                assert mask.sum()==r.n_samples
                value=float(np.mean((z['x'][i,mask,r.channel]-z[r.model][i,mask,r.channel])**2))
                err=abs(value-r.MSE)/max(1.,abs(value));maxerr=max(maxerr,err);assert err<1e-10;count+=1
    trials=pd.read_csv(out/'decision_trials.csv');ds=pd.read_csv(out/'decision_summary.csv')
    cdf=pd.read_csv(out/'decision_cumulative.csv')
    for r in ds.itertuples():
        x=trials[trials.scenario==r.scenario];assert len(x)==r.n
        assert not x.duplicated('simulation_id').any()
        event=x.event_observed==1;cens=~event
        assert x.loc[cens,'decision_time_s'].isna().all()
        assert (x.loc[cens,'observation_time_s']==r.deadline_s).all()
        assert (x.loc[event,'decision_time_s']>0).all() and (x.loc[event,'decision_time_s']<=r.deadline_s).all()
        assert (x.loc[event,'observation_time_s']==x.loc[event,'decision_time_s']).all()
        assert abs(x.observation_time_s.mean()-r.restricted_mean_decision_time_s)<1e-12
        resolved=np.sort(x.loc[event,'decision_time_s'].to_numpy());mi=int(np.ceil(len(x)/2))-1
        if len(resolved)>mi:assert abs(resolved[mi]-r.censor_aware_median_s)<1e-12
        else:assert pd.isna(r.censor_aware_median_s)
        for name in ('correct','error','unresolved'):
            assert abs((x.outcome==name).mean()-getattr(r,name+'_rate'))<1e-12
        assert ((x.choice==x.correct_side)&event).equals(x.outcome=='correct')
        assert ((x.choice==-x.correct_side)&event).equals(x.outcome=='error')
        assert (x.choice==0).equals(x.outcome=='unresolved')
        part=cdf[cdf.scenario==r.scenario]
        np.testing.assert_allclose(part.correct_cdf+part.error_cdf+part.unresolved_survival,1,atol=1e-12)
        assert np.all(np.diff(part.unresolved_survival)<=1e-12)
        for row in part.itertuples():
            assert abs(((x.outcome=='correct')&(x.decision_time_s<=row.time_s)).mean()-row.correct_cdf)<1e-12
            assert abs(((x.outcome=='error')&(x.decision_time_s<=row.time_s)).mean()-row.error_cdf)<1e-12
    numerical=pd.read_csv(out/'decision_time_step_sensitivity.csv')
    delta={k:float(abs(numerical[k].iloc[0]-numerical[k].iloc[1])) for k in ['correct_rate','error_rate','unresolved_rate']}
    # Report simulation-discretization sensitivity, do not hide it behind a biological claim.
    assert max(delta.values())<.04,delta
    intervals=pd.read_csv(out/'window_sensitivity.csv');sm=pd.read_csv(out/'window_sensitivity_trials.csv')
    for r in intervals.itertuples():
        part=sm[(sm.dataset==r.dataset)&(sm.horizon_s==r.horizon_s)&(sm.stage==r.stage)]
        block=part.groupby(['block','model']).MSE.mean().unstack('model')
        assert abs(1-block.N2.mean()/block[r.control].mean()-r.S)<1e-12
    assert len(list((out/'figures').glob('*.png')))==5
    assert len(list((out/'figures').glob('*.pdf')))==5
    reconstruction=pd.read_csv(out/'prediction_reconstruction.csv')
    result=dict(status='passed',n_real_events=len(ev),n_main_EEG_trials=int(ev.retained.sum()),
        n_recomputed_metric_rows=count,max_relative_MSE_error=maxerr,n_decision_simulations=len(trials),
        max_V2_prediction_reconstruction_difference=float(reconstruction.max_absolute_prediction_difference.max()),
        n_new_scenarios=len(ds),source_and_inherited_hashes_match=True,
        original_V2_files_unchanged=True,neural_parameters_refitted=False,
        prior_behavior_permutations_rerun=False,time_step_absolute_rate_difference=delta,
        checks=['real correctness and timeout stay unknown','proxy is not observed click',
                'same V2 observations/trials; frozen prediction reconstruction','outer training excludes held block',
                'saved waveform MSE independently recomputed','censoring is separate from decision time',
                'simulated correctness derived only from simulated truth','CDFs and outcome totals reconcile',
                'window sensitivity points recomputed'],
        caveats=['absence of observed no-response cases limits empirical validation',
                 'marker deletion is synthetic software stress only','numeric audit does not replace visual QA'])
    save_json(out/'独立复核.json',result)
    manifest['status']='numerically_verified'
    manifest['output_hashes']={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in ['manifest.json','独立复核.json','visual_qa.json','测试验收.txt']
        and p.name!='运行日志.txt' and '.mplcache' not in str(p) and 'inherited_v2' not in p.parts}
    save_json(out/'manifest.json',manifest)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'q3_result_v3')
    audit(p.parse_args().output.resolve())
