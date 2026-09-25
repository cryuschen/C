"""Read-only audit of saved Q1 outputs; writes only beside this script.

Run with numpy, pandas, scipy, scikit-learn and matplotlib available.
Models and selected candidates are replayed, not retuned. Synthetic recovery
targets are measured preprocessed backgrounds, not clean neural ground truth.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/q1-audit-mpl')
import sys, json, hashlib, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
old = load('EEG_P300_artifact_correction_v7')
new = load('EEG_P300_artifact_correction_unsupervised')
VERSIONS = {'old': ('eeg_v7_results', old), 'new': ('eeg_unsupervised_results', new)}
def save(rows, filename):
    pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding='utf-8-sig')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checks = {'source_hashes': {}, 'csv_hashes': {}, 'input_hashes': {}}
for key, (folder, mod) in VERSIONS.items():
    manifest = json.loads((ROOT/folder/'汇总与说明/运行清单.json').read_text())
    checks['source_hashes'][key] = all(sha(ROOT/k)==v for k,v in manifest['source'].items())
    checks['csv_hashes'][key] = all(sha(ROOT/folder/k)==v for k,v in manifest['csv_sha256'].items())
    checks['input_hashes'][key] = all(sha(ROOT/'data'/k)==v for k,v in manifest['inputs'].items())
assert all(all(v.values()) for v in checks.values())
strategies = {v: pd.read_csv(ROOT/f/'汇总与说明/逐折V7策略与参考数量.csv') for v,(f,_) in VERSIONS.items()}
replays=[]; real=[]; spatial=[]; synth=[]; proxy=[]; waves={}
for subject in 'AB':
    for task in (1,2):
        ds=f'VisualCog{subject}_Task-{task}'; key=f'{subject}{task}'
        folder=f'受试者{subject}_项目'+('一' if task==1 else '二')
        arrays={v:dict(np.load(ROOT/f/folder/'可复核波形.npz')) for v,(f,_) in VERSIONS.items()}
        a,b=arrays['old'],arrays['new']
        for field in ('raw','before','cues','trial_ids','folds','times_ms'):
            assert np.array_equal(a[field],b[field]), (key,field)
        x,cues,folds=a['before'],a['cues'],a['folds']
        waves[key]=(x,cues,a['v7'],b['v7'])
        # Same legacy pooled reference for a sensitivity analysis only. This
        # reproduces the original metric, but is NOT a strictly held-out target.
        common_ref={c:a['reference_per_trial'][cues==c].mean(0) for c in (-1,1)}
        for version,y in [('before',x),('old',a['v7']),('new',b['v7'])]:
            met=new.evaluate_metrics(x,y,cues,common_ref)
            real.append(dict(dataset=key,version=version,n=len(x),
                shared_legacy_proxy_MAE=met.MAE_after.mean(),
                shared_legacy_positive_mean_error=met.P300_positive_mean_error_after.mean(),
                SNR_proxy_dB=met.SNR_proxy_dB_after.mean(),
                baseline_RMS=met.baseline_RMS_after.mean(),
                trial_PTP_median=met.trial_PTP_median_after.mean()))
            spatial.extend(dict(version=version,**r) for r in new.evaluate_spatial_metrics(x,y,cues,key))
        for fold in range(1,6):
            train=np.where(folds!=fold)[0]; test=np.where(folds==fold)[0]
            scores=new.detect_bad_trials(a['raw'][train],x[train])[1]
            # Common evaluation reference built from this fold's training ONLY.
            reference=old.build_clean_reference_model(x[train],cues[train],np.zeros(len(train),bool),scores)[0]
            for version,y in [('before',x),('old',a['v7']),('new',b['v7'])]:
                for c in (-1,1):
                    erp=y[test[cues[test]==c]].mean(0)
                    for ch,channel in enumerate(new.CHANNELS):
                        truth=reference[c][ch]
                        proxy.append(dict(dataset=key,fold=fold,cue=c,channel=channel,version=version,
                            MAE=np.abs(erp[ch,new.P]-truth[new.P]).mean(),
                            positive_mean_error=abs(new.positive_p300_features(erp[ch])[0]-new.positive_p300_features(truth)[0])))
            for version,(_,mod) in VERSIONS.items():
                builder=mod.build_clean_reference_model if version=='old' else mod.build_unsupervised_reference_model
                model=builder(x[train],cues[train],np.zeros(len(train),bool),scores)
                choice=strategies[version].query('dataset == @ds and fold == @fold').selected_candidate.iloc[0]
                delta=mod.training_mean_delta(x[train],cues[train],model,choice)
                recovered=mod.apply_v7(x[test],cues[test],model,choice,delta)
                flipped=mod.apply_v7(x[test],-cues[test],model,choice,delta)
                saved_error=np.max(np.abs(recovered-arrays[version]['v7'][test]))
                assert saved_error<1e-8, (key,fold,version,saved_error)
                replays.append(dict(dataset=key,fold=fold,version=version,candidate=choice,
                    replay_max_error=saved_error,test_label_flip_max_change=np.max(np.abs(flipped-recovered)),
                    test_label_flip_RMS_change=np.sqrt(np.mean((flipped-recovered)**2))))
                for seed in (2027,2039,2053):
                    for level in (0.,.5,1.,2.):
                        noisy=mod.inject(x[test],cues[test],seed+fold,level)
                        recovered=mod.apply_v7(noisy,cues[test],model,choice,delta)
                        for stage,out in [(version,recovered)]+([('input',noisy)] if version=='old' else []):
                            r=mod.known_errors(x[test],out,cues[test])
                            spatial_errors=[]; auc_errors=[]; lat_errors=[]
                            for c in (-1,1):
                                te=x[test][cues[test]==c].mean(0)
                                re=out[cues[test]==c].mean(0)
                                spatial_errors.extend(((te[1,new.P]-te[2,new.P])-(re[1,new.P]-re[2,new.P]))**2)
                                for ch in range(3):
                                    ft=mod.positive_p300_features(te[ch]); fr=mod.positive_p300_features(re[ch])
                                    auc_errors.append(abs(ft[1]-fr[1]))
                                    if np.isfinite(ft[3]) and np.isfinite(fr[3]): lat_errors.append(abs(ft[3]-fr[3]))
                            synth.append(dict(dataset=key,fold=fold,seed=seed,level=level,version=stage,
                                F3_F4_RMSE=np.sqrt(np.mean(spatial_errors)),positive_AUC_error=np.mean(auc_errors),
                                latency_error_sum=sum(lat_errors),latency_count=len(lat_errors),**r))
        print(key,'replayed; test-label invariance and 12 injection settings verified',flush=True)
save(replays,'label_flip_and_replay.csv');save(real,'real_metrics_common_reference.csv')
save(spatial,'real_spatial_metrics.csv');save(proxy,'foldwise_common_proxy.csv');save(synth,'synthetic_recomputed.csv')
real=pd.DataFrame(real);spatial=pd.DataFrame(spatial);proxy=pd.DataFrame(proxy);synth=pd.DataFrame(synth)
summary=synth.groupby(['dataset','level','version'],sort=False)[['RMSE','normalized_RMSE','contrast_RMSE','positive_mean_error','F3_F4_RMSE','positive_AUC_error','latency_error_ms']].mean().reset_index()
save(summary,'synthetic_summary.csv')
fold_summary=proxy.groupby(['dataset','version'],sort=False)[['MAE','positive_mean_error']].mean().reset_index()
save(fold_summary,'foldwise_common_proxy_summary.csv')
table=[]
for key in waves:
    rr=real[real.dataset==key].set_index('version'); pp=fold_summary[fold_summary.dataset==key].set_index('version')
    for v in ['old','new']:
        ss=spatial.query('dataset == @key and version == @v and quantity == "left_minus_right_ERP"')
        table.append(dict(dataset=key,version=v,
            common_fold_MAE_reduction_pct=100*(1-pp.loc[v,'MAE']/pp.loc['before','MAE']),
            common_legacy_MAE_reduction_pct=100*(1-rr.loc[v,'shared_legacy_proxy_MAE']/rr.loc['before','shared_legacy_proxy_MAE']),
            SNR_gain_dB=rr.loc[v,'SNR_proxy_dB']-rr.loc['before','SNR_proxy_dB'],
            baseline_RMS_reduction_pct=100*(1-rr.loc[v,'baseline_RMS']/rr.loc['before','baseline_RMS']),
            left_right_amplitude_ratio=ss.retention_ratio.mean(),left_right_shape_correlation=ss.waveform_correlation.mean()))
save(table,'comparison_summary.csv')
print(pd.DataFrame(table).round(5).to_string(index=False),flush=True)
print('Synthetic comparisons: normalized RMSE / contrast RMSE',flush=True)
print(summary[['dataset','level','version','normalized_RMSE','contrast_RMSE','F3_F4_RMSE']].round(5).to_string(index=False),flush=True)
# Chart contract: four recordings, existing synthetic levels, static scientific
# figures. Blue/old vs orange/new; input is gray; marker/line styles redundant.
colors={'input':'#707070','before':'#707070','old':'#2563A6','new':'#D97732'}
styles={'input':':','before':':','old':'--','new':'-'}
labels={'input':'污染输入','before':'预处理','old':'旧版（使用方向标签）','new':'新版（合并方向模板）'}
plt.rcParams.update({'font.sans-serif':['Noto Sans CJK SC','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(4,3,figsize=(13,13),layout='constrained')
for i,key in enumerate(waves):
    for j,(metric,title) in enumerate([('normalized_RMSE','全波形恢复 NRMSE'),('contrast_RMSE','左右差分恢复 RMSE'),('F3_F4_RMSE','F3−F4 恢复 RMSE')]):
        ax=axes[i,j]
        for v in ['input','old','new']:
            s=summary.query('dataset == @key and version == @v').sort_values('level')
            ax.plot(s.level,s[metric],styles[v],color=colors[v],marker={'input':'x','old':'s','new':'o'}[v],label=labels[v])
        ax.set_title(f'{key} · {title}');ax.set_xlabel('注入幅度倍数');ax.set_ylabel('无量纲' if j==0 else '原始电位单位')
        ax.set_xticks([0,.5,1,2]);ax.set_ylim(bottom=0);ax.grid(alpha=.15)
        if i==0 and j==0:ax.legend(fontsize=8)
fig.suptitle('新旧方法的受控污染恢复比较\n每点为 5 折 × 3 随机种子均值；目标是污染前实测背景，仍含原有伪影',fontsize=14)
fig.savefig(OUT/'synthetic_comparison.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(4,3,figsize=(13,12),layout='constrained')
for i,(key,(x,cues,yo,yn)) in enumerate(waves.items()):
    for j,ch in enumerate(new.CHANNELS):
        ax=axes[i,j]
        for v,y in [('before',x),('old',yo),('new',yn)]:
            d=y[cues==-1,j].mean(0)-y[cues==1,j].mean(0)
            ax.plot(new.TIMES,d,color=colors[v],ls=styles[v],label=labels[v])
        ax.axvspan(250,500,color='#808080',alpha=.08);ax.axhline(0,color='#aaa',lw=.6)
        ax.set_title(f'{key} · {ch} · 左减右 ERP');ax.set_xlabel('提示后时间（ms）');ax.set_ylabel('原始电位单位');ax.grid(alpha=.15)
        if i==0 and j==0:ax.legend(fontsize=8)
fig.suptitle('真实数据左右差分：预处理、旧版与新版\n相同有效试次；250–500 ms 灰色分析窗；接近原波形不等于保留真实神经信息',fontsize=14)
fig.savefig(OUT/'real_contrast_comparison.png',dpi=150);plt.close(fig)
checks['max_replay_error']=max(r['replay_max_error'] for r in replays)
checks['new_test_label_flip_max_change']=max(r['test_label_flip_max_change'] for r in replays if r['version']=='new')
checks['old_test_label_flip_max_change']=max(r['test_label_flip_max_change'] for r in replays if r['version']=='old')
checks['scope']='Fixed saved candidates replayed; complete candidate selection and continuous filtering were not rerun.'
checks['files']={p.name:sha(p) for p in OUT.glob('*.csv')}
(OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
