"""Portable V3 pipeline; writes only q3_result by default."""
from pathlib import Path
import sys,os,json,argparse,shutil,time
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'q3_result/.mplcache'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from q3.common import KEYS,TIME,FS,STAGES,save_json,save_csv,sha,stage_masks,load_raw
from q3.events import build_trials,window_policy,product_limit
from q3.model import Candidate,predict,states
from q3.decision import SCENARIOS,simulate,EVENT,REFERENCE
from q3.reference import prepare_reference


def log(message):print(time.strftime('%H:%M:%S'),message,flush=True)


def snapshot(out):
    reference_hashes=prepare_reference(out)
    paths=[*sorted((ROOT/'q3').rglob('*.py')),ROOT/'q3/README.md',ROOT/'q3/requirements.txt',
           *sorted((ROOT/'q2/q2model').glob('*.py')),*sorted((ROOT/'data').glob('*.mat')),
           ROOT/'服务于脑机接口与精神性疾病诊断的脑电图计算模型.docx',ROOT/'q3/reference_manifest.json']
    save_json(out/'manifest.json',dict(status='running',version='V3',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        reference_hashes=reference_hashes,
        analysis='frozen_V2_EEG_reconstruction_plus_V3_response_handling_and_V2_state_decisions',
        original_data_modified=False,refit_EEG=False,rerun_RT_permutations=False,
        seed=202609263,administrative_horizon_s=2.,horizon_sensitivity=[1.5,2.,2.5]))


def past_slopes(key,ds):
    from q3.common import causal_filter
    from q3.events import audit_events
    raw=load_raw(key)[0];events=audit_events(key,raw);ons=[e['cue_sample'] for e in events]
    edges=[0]+[(ons[j-1]+ons[j])//2 for j in (20,40,60,80)]+[raw.shape[1]]
    for block in range(5):
        a,z=edges[block:block+2];filtered=causal_filter(raw[:3,a:z])
        for e in ds.events:
            if e['block']!=block:continue
            past=filtered[:,:e['cue_sample']-a]
            e['past_slopes']=[((past[:,-round(t*FS):].mean(1)-past[:,-2*round(t*FS):-round(t*FS)].mean(1))/t).tolist() for t in (.25,1.,4.)]


def score(ds,model,pred,setting,horizon):
    rows=[]
    for i,e in enumerate(ds.events):
        for stage,mask in zip(STAGES,stage_masks(e['target_s'],e['end_s'])):
            if not mask.any():continue
            for ch in range(3):
                y=ds.x[i,mask,ch];p=pred[i,mask,ch]
                rows.append(dict(dataset=ds.key,trial_id=e['trial_id'],block=e['block'],
                    setting=setting,horizon_s=horizon,model=model,stage=stage,channel=ch,
                    n_samples=int(mask.sum()),MSE=float(np.mean((y-p)**2))))
    return rows


def eeg(out):
    audits=[];metrics=[];lineage=[];sens=[];survival=[];fixtures=[];fixture_arrays={};reconstruction=[]
    inherited=out/'reference'
    for key in KEYS:
        ds,audit=build_trials(key);past_slopes(key,ds);audits+=audit
        old=np.load(inherited/f'{key}_stimulus_waves.npz')
        np.testing.assert_array_equal(ds.ids,old['ids']);np.testing.assert_allclose(ds.x,old['x'],atol=1e-7,rtol=1e-9,equal_nan=True)
        lineage.append(dict(dataset=key,n=len(ds.ids),same_V2_ids=True,
            max_observation_difference=float(np.nanmax(abs(ds.x-old['x'])))))
        stimulus_predictions={}
        for setting in ('stimulus','past'):
            source=np.load(inherited/f'{key}_{setting}_waves.npz')
            fits=json.loads((inherited/f'{key}_{setting}_choices.json').read_text(encoding='utf8'))
            models=sorted({f['model'] for f in fits});preds={m:np.empty_like(ds.x) for m in models}
            for f in fits:
                c=Candidate(**f['parameters']);coef=np.asarray(f['coef']);held=f['held_block']
                assert f['train_ids']==ds.ids[ds.blocks!=held].tolist()
                for i in np.flatnonzero(ds.blocks==held):
                    preds[f['model']][i]=predict(ds.events[i],c,coef,setting=='past')
            for model,p in preds.items():
                delta=float(np.max(abs(p-source[model])))
                relative=float(np.linalg.norm(p-source[model])/max(np.linalg.norm(source[model]),1.))
                # Cross-platform ODE/BLAS rounding is recorded, not mistaken for a new fit.
                assert delta<1e-4 and relative<1e-7,(key,setting,model,delta,relative)
                reconstruction.append(dict(dataset=key,setting=setting,model=model,
                    max_absolute_prediction_difference=delta,relative_wave_L2_difference=relative))
                metrics+=score(ds,model,p,setting,2.)
            np.savez_compressed(out/f'{key}_{setting}_waves.npz',time=TIME,x=ds.x,ids=ds.ids,
                blocks=ds.blocks,cues=ds.cues,target=[e['target_s'] for e in ds.events],
                end=[e['end_s'] for e in ds.events],**preds)
            if setting=='stimulus':stimulus_predictions=preds
        for horizon in (1.5,2.,2.5):
            alt,aa=build_trials(key,horizon)
            # Every sensitivity comparison uses exactly the main cohort intersection.
            common=sorted(set(ds.ids)&set(alt.ids));indices=[list(alt.ids).index(i) for i in common]
            mainidx=[list(ds.ids).index(i) for i in common]
            from q3.common import Trials
            subset=Trials(key,[alt.events[i] for i in indices],alt.x[indices],alt.baseline_scale[indices],'causal',.1)
            for model in ('N0','N1','N2','N3'):
                sens+=score(subset,model,stimulus_predictions[model][mainidx],'stimulus',horizon)
            selected=[e for e in aa if e['behavior_survival_eligible']]
            # All event-valid explicit-click trials: separate from EEG quality cohort.
            for r in product_limit([e['behavior_duration_s'] for e in selected],[e['behavior_event'] for e in selected]):
                survival.append(dict(dataset=key,horizon_s=horizon,n=len(selected),**r))
        # Controlled missing-marker injection: never joins the real-data audit/metrics.
        if key.endswith('2'):
            raw=load_raw(key)[0];first=ds.events[0];changed=raw.copy()
            on=first['cue_sample'];nxt=first['next_cue_sample']
            mask=(abs(changed[8,on:nxt])==2);changed[8,on:nxt][mask]=0
            stress,sa=build_trials(key,raw=changed)
            row=next(e for e in sa if e['trial_id']==first['trial_id'])
            assert row['response_status']=='unobserved' and row['timeout'] is None
            assert row['end_s']==min(row['target_s']+2.,float(TIME[-1]),(nxt-on)/FS)
            wrong=window_policy(dict(first,correctness=False))
            assert wrong['end_s']==first['end_s']
            fixtures.append(dict(dataset=key,trial_id=first['trial_id'],synthetic_marker_deletion=True,
                real_response_s=first['response_s'],injected_end_s=row['end_s'],retained=row['retained'],
                exclusion_reason=row['reason'],survival_eligible=row['behavior_survival_eligible'],
                correctness_ignored_in_window=True,
                caveat='real EEG may contain actual movement; this is software stress only'))
            if first['trial_id'] in stress.ids:
                i=list(stress.ids).index(first['trial_id']);fixture_arrays[key+'_x']=stress.x[i]
        log(f'{key}: V2所有外层模型重建；主队列 {len(ds.ids)}；事件与窗口审计完成')
    save_csv(out/'event_audit.csv',audits);save_csv(out/'trial_metrics.csv',metrics)
    save_csv(out/'lineage.csv',lineage);save_csv(out/'window_sensitivity_trials.csv',sens)
    save_csv(out/'prediction_reconstruction.csv',reconstruction)
    save_csv(out/'observed_response_survival.csv',survival)
    save_csv(out/'synthetic_marker_stress.csv',fixtures)
    np.savez_compressed(out/'synthetic_marker_stress.npz',time=TIME,**fixture_arrays)
    frame=pd.DataFrame(metrics)
    summary=frame.groupby(['dataset','setting','model','stage']).agg(MSE=('MSE','mean'),n=('trial_id','nunique')).reset_index()
    summary['RMSE']=np.sqrt(summary.MSE);save_csv(out/'summary.csv',summary)
    sf=pd.DataFrame(sens);outrows=[];rng=np.random.default_rng(202609263)
    for (key,horizon,stage),df in sf.groupby(['dataset','horizon_s','stage']):
        b=df.groupby(['block','model']).MSE.mean().unstack('model');ix=rng.integers(0,len(b),(2000,len(b)))
        for control in ('N0','N1','N3'):
            a=b.N2.to_numpy();c=b[control].to_numpy();boot=1-a[ix].mean(1)/c[ix].mean(1)
            lo,hi=np.quantile(boot,[.025,.975]);outrows.append(dict(dataset=key,horizon_s=horizon,
                stage=stage,control=control,S=1-a.mean()/c.mean(),ci_low=lo,ci_high=hi,
                n=df.trial_id.nunique(),n_blocks=len(b),mode='frozen_V2_common_cohort'))
    save_csv(out/'window_sensitivity.csv',outrows)


def decisions(out):
    summaries=[];trials=[];curves=[];waves={};sources={}
    for name,label,c,gamma,sigma,deadline in SCENARIOS:
        sm,rr,pp,cc=simulate(c,gamma,sigma,deadline)
        summaries.append(dict(scenario=name,label=label,**sm))
        trials.extend(dict(scenario=name,**r) for r in rr)
        curves.extend(dict(scenario=name,**r) for r in cc)
        for k,v in pp.items():waves[name+'_'+k]=v
        v,m,p=states(EVENT,c)
        for k,a in [('v',v),('m',m),('p',p)]:sources[name+'_'+k]=a
        log(f'{label}: 正确 {sm["correct_rate"]:.3f}，错误 {sm["error_rate"]:.3f}，未越界 {sm["unresolved_rate"]:.3f}（仅仿真）')
    save_csv(out/'decision_summary.csv',summaries);save_csv(out/'decision_trials.csv',trials)
    save_csv(out/'decision_cumulative.csv',curves)
    np.savez_compressed(out/'decision_paths.npz',**waves)
    np.savez_compressed(out/'cognitive_states.npz',time=TIME,**sources)
    selected=[]
    for key in KEYS:
        fits=json.loads((out/f'reference/{key}_stimulus_choices.json').read_text(encoding='utf8'))
        for f in fits:
            if f['model']!='N2':continue
            c=Candidate(**f['parameters']);sm,_,_,_=simulate(c,n=1000)
            selected.append(dict(dataset=key,held_block=f['held_block'],parameter_origin='V2_EEG_outer_training',**sm))
    save_csv(out/'selected_parameter_decisions.csv',selected)
    # Numerical sensitivity, not empirical behavioral validation.
    numerical=[]
    for dt in (1/256,1/512):
        sm,_,_,_=simulate(n=12000,dt=dt)
        numerical.append(sm)
    save_csv(out/'decision_time_step_sensitivity.csv',numerical)
    save_json(out/'decision_specification.json',dict(reference=REFERENCE.record(),event=EVENT,
        equation='dz = correct_side * gamma * cue * p_cue_contrast(t) / fixed_reference_scale * dt + sigma*dW',
        simulation_mapping_only=True,fitted_observed_behavior=False,nondecision_time_added=False,
        no_response='censored decision time, observation duration stored separately',
        limitation='Target layout-to-action mapping is assumed in simulation; it is not inferred from real EEG.'))


def main():
    p=argparse.ArgumentParser(description='第三题最终入口：默认重建主结果、运行优化对照并完成审计。')
    p.add_argument('--stage',choices=['all','reconstruct','render','optimize','verify'],default='all')
    p.add_argument('--output',type=Path,default=ROOT/'q3_result');args=p.parse_args();out=args.output.resolve()
    protected=[(ROOT/n).resolve() for n in ('q3_result/reference','q3','q2','Q1','data','q2_result','eeg_v7_results')]
    if out==ROOT or any(out==path or path in out.parents for path in protected):
        p.error('输出目录不能覆盖源码、原始数据、冻结参考或其他题结果')
    out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if args.stage=='verify':
            from q3.audit import audit
            audit(out)
            return
        if args.stage=='optimize':
            prepare_reference(out)
            from q3.optimization import run
            run(out)
            log('新增拟合、参数剖面、恢复网格与目标匹配验证完成')
            return
        if args.stage in ('all','reconstruct'):
            snapshot(out);eeg(out);decisions(out)
        from q3.report import render
        render(out)
        if args.stage=='all':
            from q3.optimization import run
            run(out)
        m=json.loads((out/'manifest.json').read_text(encoding='utf8'));m['status']='computed';save_json(out/'manifest.json',m)
        from q3.audit import audit
        audit(out)
    log('V3 计算、报告与数值审计完成')


if __name__=='__main__':main()
