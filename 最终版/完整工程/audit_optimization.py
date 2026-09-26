"""Independent arithmetic audits for the new recovery and cognitive results."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json,hashlib
import numpy as np
import pandas as pd
from EEG_P300_artifact_correction_v7 import known_errors
ROOT=Path(__file__).resolve().parent


def audit():
    count=0;largest=0.
    q1=ROOT/'eeg_v7_results';q2=ROOT/'q2_result';q3=ROOT/'q3_result'
    with ZipFile(ROOT/'优化计划/优化前基线.zip') as zipfile:
        before=pd.read_csv(BytesIO(zipfile.read('q2_result/Q2_表4_特征判别.csv')))
    after=pd.read_csv(q2/'Q2_表4_特征判别.csv')
    merged=before.merge(after,on=['dataset','features'],suffixes=('_before','_after'))
    for col in ('Accuracy','BA','AUC','TN','FP','FN','TP'):
        np.testing.assert_allclose(merged[col+'_before'],merged[col+'_after'],atol=1e-12)
    frame=pd.read_csv(q1/'独立验证_逐折指标.csv')
    choices=json.loads((q1/'独立验证_选择记录.json').read_text())
    with np.load(q1/'独立验证_真值与示例.npz') as z:
        for choice in choices:
            assert not set(choice['train_ids'])&set(choice['test_ids'])
            assert all((i-1)//20==choice['held_block'] for i in choice['test_ids'])
            if choice['held_block']!=0:continue
            key=choice['dataset'];origin=choice['origin'];prefix=key+'_'+origin
            ids=z[prefix+'_ids'];y=z[prefix+'_cues'][np.isin(ids,choice['test_ids'])]
            for family in ('asymmetric_burst','direction_step'):
                target=z[prefix+'_'+family+'_test_target']
                for method in ('uncorrected','V7','shrink','gated'):
                    recovered=z[prefix+'_'+family+'_'+method]
                    result=known_errors(target,recovered,y)
                    row=frame[(frame.dataset==key)&(frame.origin==origin)&(frame.held_block==0)&(frame.family==family)&(frame.seed==20260927)&(frame.level==1)&(frame.method==method)].iloc[0]
                    for name in ('RMSE','normalized_RMSE','contrast_RMSE','positive_mean_error','latency_error_ms'):
                        np.testing.assert_allclose(result[name],row[name],atol=1e-9,rtol=1e-10,equal_nan=True)
                    count+=1
    inner=pd.read_csv(q1/'独立验证_内层选型.csv')
    for choice in choices:
        selected=choice['selected']['gated']
        subset=inner[(inner.dataset==choice['dataset'])&(inner.origin==choice['origin'])&(inner.held_block==choice['held_block'])&(inner.method=='gated')&(inner.alpha==selected['alpha'])&(inner.gate==selected['gate'])&(inner.level==0)]
        assert len(subset)==1
        assert subset.iloc[0].normalized_RMSE<=.05+1e-12 and subset.iloc[0].contrast_normalized<=.05+1e-12
    metrics=pd.read_csv(q3/'优化_新拟合逐试次.csv');events=pd.read_csv(q3/'event_audit.csv')
    fits=json.loads((q3/'优化_新拟合选择.json').read_text())
    for f in fits:
        assert not set(f['train_ids'])&set(f['test_ids'])
        assert all((i-1)//20==f['held_block'] for i in f['test_ids'])
    for key in ('A1','A2','B1','B2'):
        with np.load(q3/'优化_新拟合波形.npz') as archive:
            data={n:archive[n] for n in ('time',key+'_x',key+'_ids',key+'_N2',key+'_N3')}
        t=data['time'];lookup={i:j for j,i in enumerate(data[key+'_ids'])};ev=events[events.dataset==key].set_index('trial_id')
        for r in metrics[metrics.dataset==key].itertuples():
            e=ev.loc[r.trial_id];i=lookup[r.trial_id]
            mask=(t>=0)&(t<e.end_s)
            mask&=(t<.8) if r.stage=='encoding' else ((t>=.8)&(t<e.target_s) if r.stage=='maintenance' else t>=e.target_s)
            assert mask.sum()==r.n_samples
            value=np.mean((data[key+'_x'][i,mask]-data[key+'_'+r.model][i,mask])**2)
            delta=abs(value-r.MSE)/max(1.,abs(value));largest=max(largest,delta);assert delta<1e-10
    trials=pd.read_csv(q3/'优化_目标匹配试次.csv');summary=pd.read_csv(q3/'优化_目标匹配仿真.csv')
    for r in summary.itertuples():
        x=trials[trials.scenario==r.scenario]
        truth=np.where(x.target_right_shape==x.cue,1,-1)
        np.testing.assert_array_equal(x.correct_side,truth)
        for name in ('correct','error','unresolved'):
            np.testing.assert_allclose((x.outcome==name).mean(),getattr(r,name+'_rate'),atol=1e-12)
        assert x[x.event_observed==0].decision_time_s.isna().all()
    result=dict(status='passed',q2_baseline_scores_unchanged=True,q1_saved_recovery_cases_recomputed=count,
        q1_all_training_guardrails_verified=True,q3_new_fit_rows_recomputed=len(metrics),
        q3_max_relative_MSE_error=largest,q3_matching_simulations_recomputed=len(trials),
        caveat='Arithmetic audit; no claim of physiological sources or independent population validation')
    (ROOT/'优化计划/新增结果独立复核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
