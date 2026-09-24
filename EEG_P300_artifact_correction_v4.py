#!/usr/bin/env python3
"""第一问 V4：训练内选择保守校正，固定折外比较，中文可追溯输出。
运行 python3 EEG_P300_artifact_correction_v4.py --output eeg_v4_results
V3 仅作为被冻结的算法/滤波/指标定义依赖，不调用其 main，不覆盖其输出。
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/eeg-v4-mpl')
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold, train_test_split
import EEG_P300_artifact_correction_v3 as v3
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
T = v3.TIMES_MS
P = v3.P300_MASK
N = v3.N_PRE
CHANNELS = v3.CHANNEL_NAMES
# 固定候选集，不根据四组真实评价结果调整。内层只用训练数据模拟污染选型。
CANDIDATES = {
    '不校正': (0., 2.2, 1.5, False),
    '中位数基线': (1., 2.2, 1.5, False),
    '保守双分量': (.5, 3.5, .5, False),
    '仅垂直分量': (.5, 3.5, 0., False),
    '时域保护': (.75, 3.5, .5, True),
}
# 只用于定位因素的单因素诊断；不进入训练选型，也不据其结果改参数。
DIAGNOSTIC_CANDIDATES = {
    **CANDIDATES,
    '诊断_仅提高门控': (1., 3.5, 1.5, False),
    '诊断_仅减弱校正': (.5, 2.2, 1.5, False),
    '诊断_仅限制水平上限': (1., 2.2, .5, False),
    '诊断_关闭水平分量': (1., 2.2, 0., False),
    '诊断_关闭垂直分量': (1., 2.2, 1.5, False),
}
COLORS = {'原始（基线对齐）':'#a8adb4', '预处理':'#67788a', 'V3':'#d48832', 'V4':'#007d91', '代理参考':'#282c34'}
STYLE = {'原始（基线对齐）':':','预处理':'--','V3':'-.','V4':'-','代理参考':':'}


def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correct(epoch, tpl, sv, sh, candidate):
    strength, gate, max_h, protect = DIAGNOSTIC_CANDIDATES[candidate]
    if strength == 0:
        return epoch.copy()
    params = dict(v3.V3_PARAMS, gate_z_v=gate, gate_z_h=gate, max_beta_h=max_h)
    if candidate == '诊断_关闭垂直分量':
        params['max_beta_v'] = 0.
    # 撤销 V3 最后的均值重定位，再统一使用中位数基线。
    full = v3.correct_single_epoch_v3(epoch, tpl, sv, sh, params)
    full -= np.median(full[:, :N], axis=1, keepdims=True)
    delta = epoch - full
    if protect:
        # 仅保护训练模板的时间形状；不将测试条件 ERP 人为恢复到原幅度。
        taper = np.clip((T - 0) / 80, 0, 1) * np.clip((750 - T) / 120, 0, 1)
        basis = savgol_filter(tpl, 25, 3, axis=-1) * taper
        u, s, _ = np.linalg.svd(basis.T, full_matrices=False)
        keep = s > max(s[0] * .10, 1e-9)
        q = u[:, keep]
        delta -= (delta @ q) @ q.T
    result = epoch - strength * delta
    return result - np.median(result[:, :N], axis=1, keepdims=True)


def inject(background, cues, seed, level):
    """已知加性污染，正负眨眼、扫视和短暂运动；保留真实背景原波形为恢复目标。"""
    rng = np.random.default_rng(seed)
    t = T / 1000
    out = background.copy()
    scale = max(float(np.median(np.std(background, axis=-1))), 1.)
    for i in range(len(out)):
        center = rng.uniform(-.12, .65)
        width = rng.uniform(.025, .13)
        amp = level * scale * rng.uniform(2, 6)
        blink = amp * rng.choice([-1, 1]) * np.exp(-.5*((t-center)/width)**2)
        saccade = amp * .7 * cues[i] * (np.tanh((t-center)/.035)-np.tanh((t-center-.18)/.045))/2
        motion = amp*.3*np.sin(2*np.pi*rng.uniform(7,15)*t)*np.exp(-.5*((t-center)/.035)**2)
        artifact = np.array([1.1,1.,.9])[:,None]*blink + np.array([.05,-1,1])[:,None]*saccade + motion
        # 污染与基线操作作为已知变换；level=0 时精确返回背景。
        out[i] += artifact
    return v3.robust_baseline_correct(out)


def apply_model(x, cues, model, candidate):
    tpl, sv, sh, _ = model
    if candidate == 'V3':
        return np.stack([v3.correct_single_epoch_v3(e, tpl[c], sv[c], sh[c]) for e,c in zip(x,cues)])
    return np.stack([correct(e,tpl[c],sv[c],sh[c],candidate) for e,c in zip(x,cues)])


def known_errors(target, recovered, cues):
    scale = max(float(np.sqrt(np.mean(target**2))), 1e-9)
    rmse = float(np.sqrt(np.mean((target-recovered)**2)))
    truth_diff = target[cues==-1].mean(0)-target[cues==1].mean(0)
    rec_diff = recovered[cues==-1].mean(0)-recovered[cues==1].mean(0)
    contrast = float(np.sqrt(np.mean((truth_diff[:,P]-rec_diff[:,P])**2)))
    pos = []
    lat = []
    for c in (-1,1):
        for ch in range(3):
            a=v3.positive_p300_features(target[cues==c,ch].mean(0))
            b=v3.positive_p300_features(recovered[cues==c,ch].mean(0))
            pos.append(abs(a[0]-b[0]))
            if np.isfinite(a[3]) and np.isfinite(b[3]): lat.append(abs(a[3]-b[3]))
    return dict(RMSE=rmse, normalized_RMSE=rmse/scale, contrast_RMSE=contrast,
                positive_mean_error=np.mean(pos), latency_error_ms=np.mean(lat) if lat else np.nan,
                latency_pairs=len(lat), objective=(rmse+.5*contrast+np.mean(pos))/scale)


def paired_latencies(target, recovered_stages, cues):
    """半合成潜伏期也使用三阶段与已知背景共同有效的六行。"""
    def peaks(x):
        return np.array([v3.positive_p300_features(x[cues==c,ch].mean(0))[3]
                         for c in (-1,1) for ch in range(3)])
    truth = peaks(target)
    values = {k: peaks(x) for k,x in recovered_stages.items()}
    valid = np.isfinite(truth)
    for x in values.values():
        valid &= np.isfinite(x)
    return {k: dict(latency_error_ms=float(np.abs(x[valid]-truth[valid]).mean()) if valid.any() else np.nan,
                    latency_pairs=int(valid.sum())) for k,x in values.items()}


def select_candidate(x, cues, seed):
    train, val = train_test_split(np.arange(len(x)), test_size=.35, stratify=cues, random_state=seed)
    # MAD/评分的归一化也仅在内层训练中估计，不能沿用全数据的归一化。
    train_scores = v3.detect_bad_trials(np.zeros_like(x[train]), x[train])[1]
    val_scores = v3.detect_bad_trials(np.zeros_like(x[val]), x[val])[1]
    model = v3.build_clean_reference_model(x[train],cues[train],np.zeros(len(train),bool),train_scores)
    # 内层验证背景也固定为相对低伪影试次，但不进入模板。
    idx = np.concatenate([val[cues[val]==c][np.argsort(val_scores[cues[val]==c],kind='stable')[:max(4,len(val[cues[val]==c])//2)]] for c in (-1,1)])
    bg, cs=x[idx], cues[idx]
    rows=[]
    for level in (0.,1.,2.):
        noisy=inject(bg,cs,seed+1000,level)
        for cand in CANDIDATES:
            r=known_errors(bg,apply_model(noisy,cs,model,cand),cs)
            rows.append(dict(candidate=cand,level=level,**r))
    df=pd.DataFrame(rows)
    # 清洁背景失真占 50%，两种污染各 25%；候选相同时偏向更弱校正。
    means=df.assign(weight=np.where(df.level==0,.5,.25))
    values=means.assign(loss=means.objective*means.weight).groupby('candidate',sort=False).loss.sum()
    return values.idxmin(),rows


def read_dataset(path):
    fs,labels,data=v3.load_mat_file(path)
    if fs!=256 or list(labels[:3])!=CHANNELS or not str(labels[7]).startswith('VisCue'):
        raise ValueError(f'输入采样率或标签不匹配：{path}')
    if not np.all(np.isfinite(data[:3])): raise ValueError('EEG 包含非有限数值')
    onsets,cues=v3.extract_viscue_events(data[7])
    if not np.isin(cues,[-1,1]).all(): raise ValueError('未知刺激方向')
    bounded=(onsets>=N)&(onsets+v3.N_POST<=data.shape[1])
    event_ids=np.arange(1,len(onsets)+1)
    raw=v3.segment_epochs(data[:3],onsets[bounded])
    x=v3.robust_baseline_correct(v3.segment_epochs(v3.preprocess_continuous_eeg(data[:3],fs),onsets[bounded]))
    bad,scores,ptp=v3.detect_bad_trials(raw,x)
    sat=(np.abs(raw)>=999).sum(axis=(1,2))
    jump=np.abs(np.diff(x,axis=-1)).max(axis=(1,2))
    audit=pd.DataFrame(dict(trial_id=event_ids, onset_sample=onsets, cue=cues,boundary_ok=bounded))
    for key,vals in dict(rejected=bad,artifact_score=scores,PTP=ptp,saturated_samples=sat,max_jump=jump).items():
        audit.loc[bounded,key]=vals
    audit['role']=np.where(~bounded,'边界剔除',np.where(audit.rejected.fillna(True),'质量剔除','折外评价'))
    # 不调整当前剔除集；只报告固定阈值 ±10% 的敏感性。
    sensitivity=[]
    for factor in (.9,1.,1.1):
        rejected=(sat>=max(1,round(15*factor)))|(ptp>1800*factor)|(jump>600*factor)
        sensitivity.append(dict(threshold_factor=factor,rejected=int(rejected.sum()),usable=int((~rejected).sum())))
    return raw[~bad],x[~bad],cues[bounded][~bad],scores[~bad],audit,sensitivity


def paired_summary(tables,dataset):
    # 所有三阶段共同有效，避免比较行集合变化；额外输出每阶段检出率。
    rows=[]
    keys=list(tables)
    metrics=[c[:-6] for c in tables[keys[0]].columns if c.endswith('_after')]
    metrics += ['latency_error_ms','P300_AUC_unit_ms','P300_lat_ms','P300_AUC_error_unit_ms']
    for metric in dict.fromkeys(metrics):
        col={'latency_error_ms':'latency_error_after_ms','P300_AUC_unit_ms':'P300_AUC_after_unit_ms','P300_lat_ms':'P300_lat_after_ms','P300_AUC_error_unit_ms':'P300_AUC_error_after_unit_ms'}.get(metric,metric+'_after')
        if col not in tables[keys[0]]: continue
        vals=np.stack([tables[k][col].to_numpy(float) for k in keys])
        common=np.isfinite(vals).all(0)
        for j,k in enumerate(keys):
            rows.append(dict(dataset=dataset,metric=metric,stage=k,mean=vals[j,common].mean() if common.any() else np.nan,
                             paired_rows=int(common.sum()),total_rows=6,valid_rows=int(np.isfinite(vals[j]).sum())))
    return rows


def interval_metrics(stages,cues,refs,n_boot=300):
    rows=[]
    for c in (-1,1):
        ix=np.flatnonzero(cues==c)
        rng=np.random.default_rng(20260923+c)
        draw=rng.integers(0,len(ix),(n_boot,len(ix)))
        for ch,name in enumerate(CHANNELS):
            ref=refs[c][ch,P]
            boot={}
            for label,x in stages.items():
                erps=x[ix,ch][draw].mean(1)
                boot[label]={'MAE':np.abs(erps[:,P]-ref).mean(1),'positive_mean':np.maximum(erps[:,P],0).mean(1)}
                for metric,values in boot[label].items():
                    lo,hi=np.percentile(values,[2.5,97.5])
                    rows.append(dict(cue=c,channel=name,stage=label,metric=metric,low=lo,high=hi,n=len(ix),bootstrap=n_boot))
            for metric in ('MAE','positive_mean'):
                values=boot['V4'][metric]-boot['V3'][metric]
                lo,hi=np.percentile(values,[2.5,97.5])
                rows.append(dict(cue=c,channel=name,stage='V4减V3',metric=metric,low=lo,high=hi,n=len(ix),bootstrap=n_boot))
    return rows


def axes_grid(title,subtitle,rows=3,cols=2):
    fig,axs=plt.subplots(rows,cols,figsize=(13,3.05*rows),squeeze=False,sharex=True,layout='constrained')
    fig.suptitle(title+'\n'+subtitle,fontsize=13)
    for ax in axs.flat:
        ax.axvline(0,color='#aeb4bb',lw=.7)
        ax.axhline(0,color='#cbd0d5',lw=.6)
        ax.axvspan(250,500,color='#c1dce0',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('提示后时间（ms）')
        ax.set_ylabel('原始电位单位')
    return fig,axs


def finish(fig,path):
    fig.savefig(path,dpi=160)
    plt.close(fig)


def lines(ax,series,mask=None):
    mask=np.ones(len(T),bool) if mask is None else mask
    for label,y in series.items(): ax.plot(T[mask],y[mask],label=label,color=COLORS[label],ls=STYLE[label],lw=1.4 if label=='V4' else 1.)


def plot_dataset(name,raw,stages,cues,scores,ids,refs,out):
    subtitle='相同可用试次；V3/V4 均折外校正；250–500 ms 阴影为分析窗'
    fig,axs=axes_grid(name+' · 左右刺激三通道 ERP',subtitle)
    for col,c in enumerate((-1,1)):
        ix=cues==c
        for ch in range(3):
            ax=axs[ch,col]
            lines(ax,{k:x[ix,ch].mean(0) for k,x in stages.items()})
            lo,hi=v3.compute_bootstrap_ci(stages['V4'][ix,ch],n_boot=300)
            ax.fill_between(T,lo,hi,color=COLORS['V4'],alpha=.16,label='V4 逐点95%试次区间')
            ax.set_title(f'{CHANNELS[ch]} · {"左" if c==-1 else "右"}刺激 · n={ix.sum()}')
            ax.legend(fontsize=8,loc='upper right')
    finish(fig,out/'左右刺激三通道ERP对比图.png')
    fig,axs=axes_grid(name+' · 正向响应分析窗','代理参考来自对应训练折；不等于无噪声真值')
    for col,c in enumerate((-1,1)):
        for ch in range(3):
            ax=axs[ch,col]
            lines(ax,{**{k:x[cues==c,ch].mean(0) for k,x in stages.items()},'代理参考':refs[c][ch]},P)
            ax.set_title(f'{CHANNELS[ch]} · {"左" if c==-1 else "右"}刺激 · n={(cues==c).sum()}')
            ax.legend(fontsize=8)
            ax.set_xlim(250,500)
    finish(fig,out/'P300时间窗放大图.png')
    fig,axs=axes_grid(name+' · 固定典型试次','各方向按原始伪影评分选择75%分位；四阶段共享坐标，无幅度截断')
    selection=[]
    for col,c in enumerate((-1,1)):
        ix=np.flatnonzero(cues==c)
        chosen=ix[np.argsort(scores[ix],kind='stable')[round(.75*(len(ix)-1))]]
        selection.append(dict(cue=c,trial_id=int(ids[chosen])))
        for ch in range(3):
            ax=axs[ch,col]
            lines(ax,{'原始（基线对齐）':raw[chosen,ch]-np.median(raw[chosen,ch,:N]),**{k:x[chosen,ch] for k,x in stages.items()}})
            ax.set_title(f'{CHANNELS[ch]} · {"左" if c==-1 else "右"}刺激 · 原始试次 #{ids[chosen]}')
            ax.legend(fontsize=8)
    finish(fig,out/'典型试次四阶段对比图.png')
    # 所有阶段/方向使用同一完整色域，无裁剪。
    fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    limit=max(np.abs(x[:,0]).max() for x in stages.values())
    for row,c in enumerate((-1,1)):
        for col,(label,x) in enumerate(stages.items()):
            ax=axs[row,col]
            ix=np.flatnonzero(cues==c)
            im=ax.imshow(x[ix,0],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,origin='lower',extent=[T[0],T[-1],.5,len(ix)+.5])
            ax.set_title(f'{"左" if c==-1 else "右"}刺激 · {label} · n={len(ix)}')
            ticks=np.unique(np.linspace(0,len(ix)-1,5,dtype=int))
            ax.set_yticks(ticks+1,[str(i) for i in ids[ix[ticks]]])
            ax.set_ylabel('原始试次编号'); ax.set_xlabel('提示后时间（ms）')
    fig.colorbar(im,ax=axs,label='Fz 原始电位单位')
    fig.suptitle(name+' · Fz 试次热力图\n同一试次、共享完整色域；不裁剪极端幅度')
    finish(fig,out/'试次时间热力图.png')
    fig,axs=axes_grid(name+' · 左右与额区空间差分','左列：左减右；右列：F3减F4；幅度相近不等于神经特征被证实保留')
    for ch in range(3):
        lines(axs[ch,0],{k:x[cues==-1,ch].mean(0)-x[cues==1,ch].mean(0) for k,x in stages.items()})
        axs[ch,0].set_title(CHANNELS[ch]+' · 左减右')
    for row,c in enumerate((-1,1)):
        lines(axs[row,1],{k:x[cues==c,1].mean(0)-x[cues==c,2].mean(0) for k,x in stages.items()})
        axs[row,1].set_title(('左' if c==-1 else '右')+'刺激 · F3减F4')
    fig.delaxes(axs[2,1])
    for ax in axs.flat:
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    finish(fig,out/'左右刺激与空间差分图.png')
    save_csv(selection,out/'典型试次索引.csv')
    # 按方向拟合，数值收敛与参数可靠性分开。
    rows=[]
    fig,axs=axes_grid(name+' · 分方向三通道拟合','拟合质控为工程筛查规则；不构成 P300 生理成分存在的证据')
    for col,c in enumerate((-1,1)):
        for ch in range(3):
            avg=stages['V4'][cues==c,ch].mean(0)
            fit,params=v3.fit_erp_curve(T,avg)
            issues=[]
            if params['fit_status']!='gaussian': issues.append('未收敛')
            else:
                if params['R2']<.5: issues.append('R²不足0.5')
                if params['P300_amp']<max(.5,params['RMSE']*.25): issues.append('幅度弱')
                if not 255<params['P300_lat_ms']<515: issues.append('潜伏期近边界')
                if not 21<params['P300_fwhm_ms']/2.355<149: issues.append('宽度近边界')
                if not np.isfinite(v3.positive_p300_features(avg)[2]): issues.append('无窗内正峰')
            params.update(cue=c,channel=CHANNELS[ch],reliable=not issues,quality_reason='；'.join(issues) or '通过工程筛查')
            rows.append(params)
            ax=axs[ch,col]
            fitmask=(T>=50)&(T<=750)
            ax.plot(T,avg,color=COLORS['V4'],label='V4 ERP')
            ax.plot(T[fitmask],fit[fitmask],color='#282c34',ls='--',label='拟合（50–750 ms）')
            ax.plot(T[fitmask],(avg-fit)[fitmask],color='#a69b8f',lw=.8,label='拟合残差')
            ax.set_title(f'{CHANNELS[ch]} · {"左" if c==-1 else "右"} · n={(cues==c).sum()} · R²={params["R2"]:.2f}\n'+('不支持可靠参数：'+'；'.join(issues) if issues else f'筛查通过：峰潜伏期 {params["P300_lat_ms"]:.0f} ms'),fontsize=10)
            ax.legend(fontsize=8)
    finish(fig,out/'分方向ERP拟合与残差图.png')
    save_csv(rows,out/'分方向ERP拟合参数.csv')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'eeg_v4_results')
    parser.add_argument('--no-plots',action='store_true',help='只重算数值，用于重复性核验')
    args=parser.parse_args()
    out=args.output.resolve()
    if out == ROOT or any(out.is_relative_to(p) for p in [ROOT/'eeg_v3_results',ROOT/'eeg_v2_results',ROOT/'data']):
        raise ValueError('禁止覆盖原始数据或旧版目录')
    out.mkdir(parents=True,exist_ok=True)
    summary_dir=out/'汇总与说明';summary_dir.mkdir(exist_ok=True)
    validation=out/'半合成验证';validation.mkdir(exist_ok=True)
    all_summary=[];all_spatial=[];all_synthetic=[];all_selection=[];all_fit=[];all_ablation=[];all_sensitivity=[]
    datasets={};manifest={'inputs':{},'code':{p.name:sha(p) for p in [Path(__file__),ROOT/'EEG_P300_artifact_correction_v3.py']},
                        'python':platform.python_version(),'numpy':np.__version__,'candidates':CANDIDATES,
                        'split_seed':42,'selection_seeds':'100+fold','test_seeds':[2027,2039,2053],
                        'unit':'原始电位单位','window_ms':[float(T[0]),float(T[-1])],
                        'reference':'逐试次训练折模板，按条件等试次平均；代理参考与校正模板相关，不是真值'}
    for subject in 'AB':
        for task in (1,2):
            ds=f'VisualCog{subject}_Task-{task}'
            name=f'受试者{subject}_项目'+('一' if task==1 else '二')
            path=ROOT/'data'/f'{ds}.mat';manifest['inputs'][path.name]=sha(path)
            dest=out/name;dest.mkdir(exist_ok=True)
            raw,x,cues,scores,audit,sensitivity=read_dataset(path)
            ids=audit.loc[audit.role=='折外评价','trial_id'].to_numpy(int)
            print(f'{name}: {len(audit)}事件，{len(x)}可用',flush=True)
            y3=x.copy();y4=x.copy();ref_trial=x.copy();fold_ids=np.zeros(len(x),int)
            ref_lists={};fold_roles=[]
            ablations={k:x.copy() for k in DIAGNOSTIC_CANDIDATES}
            skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
            for fold,(train,test) in enumerate(skf.split(x,cues),1):
                train_scores=v3.detect_bad_trials(raw[train],x[train])[1]
                model=v3.build_clean_reference_model(x[train],cues[train],np.zeros(len(train),bool),train_scores)
                choice,selection=select_candidate(x[train],cues[train],100+fold)
                for r in selection: all_selection.append(dict(dataset=ds,fold=fold,chosen=choice,**r))
                print(f'  折 {fold}: {choice}',flush=True)
                y3[test]=apply_model(x[test],cues[test],model,'V3')
                y4[test]=apply_model(x[test],cues[test],model,choice)
                for cand in DIAGNOSTIC_CANDIDATES: ablations[cand][test]=apply_model(x[test],cues[test],model,cand)
                for i in test:ref_trial[i]=model[0][cues[i]]
                fold_ids[test]=fold
                reference_ids=ids[train[model[3]]].tolist()
                ref_lists[str(fold)]=reference_ids
                assert not set(ids[test])&set(reference_ids)
                for i in train:fold_roles.append(dict(fold=fold,trial_id=int(ids[i]),role='训练参考' if int(ids[i]) in reference_ids else '训练非参考'))
                for i in test:fold_roles.append(dict(fold=fold,trial_id=int(ids[i]),role='折外评价'))
                # 外层背景仅使用测试折；每折预处理、V3、V4在完全相同污染上配对。
                for seed in (2027,2039,2053):
                    for level in (0.,.5,1.,2.):
                        noisy=inject(x[test],cues[test],seed+fold,level)
                        recovered={'预处理':noisy,'V3':apply_model(noisy,cues[test],model,'V3'),
                                   'V4':apply_model(noisy,cues[test],model,choice)}
                        latency=paired_latencies(x[test],recovered,cues[test])
                        for stage,rec in recovered.items():
                            errors=known_errors(x[test],rec,cues[test])
                            errors.update(latency[stage])
                            all_synthetic.append(dict(dataset=ds,fold=fold,seed=seed,level=level,stage=stage,n=len(test),candidate=choice,**errors))
            assert np.all(fold_ids>0)
            refs={c:ref_trial[cues==c].mean(0) for c in (-1,1)}
            stages={'预处理':x,'V3':y3,'V4':y4}
            tables={k:v3.evaluate_metrics(x,z,cues,refs) for k,z in stages.items()}
            for stage,df in tables.items():
                df['stage']=stage
                df['n_trials']=[int((cues==c).sum()) for c in df.cue]
                df['positive_peak_valid']=np.isfinite(df.P300_amp_after)
                df['reference_positive_peak_valid']=[np.isfinite(v3.positive_p300_features(refs[int(c)][CHANNELS.index(ch)])[2]) for c,ch in zip(df.cue,df.channel)]
                for base in ('corr','amp_error'):
                    df[base+'_pair_valid']=np.isfinite(df[base+'_before'])&np.isfinite(df[base+'_after'])
                df['latency_pair_valid']=np.isfinite(df.latency_error_before_ms)&np.isfinite(df.latency_error_after_ms)
            save_csv(pd.concat(tables.values(),ignore_index=True),dest/'逐方向逐通道评价指标.csv')
            all_summary.extend(paired_summary(tables,ds))
            for stage,z in stages.items():
                all_spatial.extend([dict(stage=stage,**r) for r in v3.evaluate_spatial_metrics(x,z,cues,ds)])
            for candidate,z in ablations.items():
                df=v3.evaluate_metrics(x,z,cues,refs)
                spatial=v3.evaluate_spatial_metrics(x,z,cues,ds)
                all_ablation.append(dict(dataset=ds,candidate=candidate,MAE=df.MAE_after.mean(),corr=df.corr_after.mean(),
                    SNR_proxy_dB=df.SNR_proxy_dB_after.mean(),baseline_RMS=df.baseline_RMS_after.mean(),
                    left_right_retention=np.mean([r['retention_ratio'] for r in spatial[:3]])))
            audit.loc[audit.role=='折外评价','fold']=fold_ids
            save_csv(audit,dest/'完整事件与试次审计.csv')
            save_csv(fold_roles,dest/'逐折训练参考评价清单.csv')
            save_csv(interval_metrics(stages,cues,refs),dest/'试次重采样区间.csv')
            (dest/'训练参考试次编号.json').write_text(json.dumps(ref_lists,ensure_ascii=False,indent=2),encoding='utf-8')
            np.savez_compressed(dest/'可复核波形.npz',raw=raw,before=x,v3=y3,v4=y4,cues=cues,trial_ids=ids,
                                folds=fold_ids,reference_per_trial=ref_trial,times_ms=T)
            all_sensitivity.extend([dict(dataset=ds,**r) for r in sensitivity])
            if not args.no_plots:
                all_fit.extend([dict(dataset=ds,**r) for r in plot_dataset(name,raw,stages,cues,scores,ids,refs,dest)])
            datasets[ds]=dict(name=name,cues=cues,stages=stages,audit=audit,ref_lists=ref_lists)
    save_csv(all_summary,summary_dir/'三阶段共同配对指标汇总.csv')
    save_csv(all_spatial,summary_dir/'左右刺激与额区空间差异指标.csv')
    save_csv(all_synthetic,validation/'多场景逐折配对验证.csv')
    save_csv(all_selection,summary_dir/'训练内部选型记录.csv')
    save_csv(all_ablation,summary_dir/'固定候选消融比较.csv')
    save_csv(all_sensitivity,summary_dir/'坏试次阈值敏感性.csv')
    if all_fit:save_csv(all_fit,summary_dir/'分方向拟合参数汇总.csv')
    synthetic=pd.DataFrame(all_synthetic)
    save_csv(synthetic.groupby(['dataset','level','stage'],sort=False)[['RMSE','normalized_RMSE','contrast_RMSE','positive_mean_error','latency_error_ms']].mean().reset_index(),validation/'多场景验证汇总.csv')
    deltas=[]
    for (ds,level),group in synthetic.groupby(['dataset','level']):
        for metric in ['RMSE','contrast_RMSE','positive_mean_error']:
            wide=group.pivot(index=['fold','seed'],columns='stage',values=metric)
            diff=wide['V4']-wide['V3']
            deltas.append(dict(dataset=ds,level=level,metric=metric,paired_scenarios=len(diff),
                               improved_scenarios=int((diff<0).sum()),median_delta=diff.median(),
                               p10_delta=diff.quantile(.1),p90_delta=diff.quantile(.9)))
    save_csv(deltas,validation/'配对改善与退步分布.csv')
    manifest['output_numerical_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*.csv')) if '基线快照' not in p.parts}
    (summary_dir/'运行清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(datasets,all_summary,all_spatial,synthetic,all_fit,summary_dir)
    if not args.no_plots:
        cross=out/'跨项目对比';cross.mkdir(exist_ok=True)
        for subject in 'AB':
            fig,axs=axes_grid('受试者'+subject+' · 分方向跨项目 V4 ERP','左右条件分别比较，不混合方向；样本数为各项目有效试次')
            for col,c in enumerate((-1,1)):
                for ch in range(3):
                    ax=axs[ch,col]
                    for task,color,style in [(1,'#007d91','-'),(2,'#d48832','--')]:
                        d=datasets[f'VisualCog{subject}_Task-{task}'];ix=d['cues']==c
                        ax.plot(T,d['stages']['V4'][ix,ch].mean(0),color=color,ls=style,label=f'项目{task} · n={ix.sum()}')
                    ax.set_title(CHANNELS[ch]+' · '+('左' if c==-1 else '右'));ax.legend()
            finish(fig,cross/f'受试者{subject}_分方向跨项目对比.png')
        fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
        for ax,metric,label in zip(axs,['RMSE','contrast_RMSE','positive_mean_error'],['已知背景恢复 RMSE','左减右恢复 RMSE','正向均值误差']):
            for k in ('预处理','V3','V4'):
                means=synthetic[synthetic.stage==k].groupby('level')[metric].mean()
                ax.plot(means.index,means.values,marker='o',color=COLORS[k],ls=STYLE[k],label=k)
            ax.set_xlabel('注入幅度系数（0 为未注入）');ax.set_ylabel(label+'（原始单位）');ax.legend();ax.set_ylim(bottom=0)
        fig.suptitle('四组背景 × 五折 × 三个固定种子\n恢复污染前实测波形；背景自身可能含伪影，不是真实无噪声 EEG')
        finish(fig,validation/'多幅度恢复与失真比较.png')
    print('完成：'+str(out),flush=True)


def write_report(datasets,summary,spatial,synthetic,fits,out):
    df=pd.DataFrame(summary);sp=pd.DataFrame(spatial)
    lines_=['# V4 结果说明与验证报告','',
        '本版针对第一问：在固定试次与折外评价下比较保守伪影校正，并完善分方向图表及指标。V3 原文件与已归档结果保持原样。',
        '', '## 比较范围与口径', '',
        '- 输入仅为原始 Fz/F3/F4 与 VisCue。256 Hz；−250 至 796.875 ms；正向分析窗 250–500 ms。单位未确认，统一为原始电位单位。',
        '- 与旧 CSV 的参考口径不同，不能跨文件直接宣称提升。此处预处理、V3、V4 在同一批全部有效试次上重新计算；不筛选有利方向或通道。',
        '- 本表V3指旧校正函数在共同训练评分/参考流程下的重算结果。评分MAD也只在训练子集估计，因此不是旧脚本输出的原封复现；冻结的原始输出另存基线快照。连续预处理为离线双向滤波，不是实时或跨受试者验证。',
        '- 五折固定种子42；训练折内再拆训练/验证，模拟污染后选候选。外层测试种子与选型种子分离。V4 可逐折选择不校正，这是安全回退，已记录。',
        '- 每个评价试次的代理模板来自其训练折；条件内按试次平均模板。评价试次没有直接进入自己的参考，但训练模板与校正相关；代理指标仍不能证明真实神经信号恢复。',
        '- 每组方向×通道六行等权汇总；三阶段共同有效行才计算同一指标均值。正峰、潜伏期缺失不补零。逐行CSV保留有效性与试次数。',
        '- 正峰定义仅为分析窗内最大值>0，不等于显著P300；拟合通过工程阈值也不是生理证据。仅有两个受试者、三个额区电极。',
        '- 95%区间为固定模型和代理模板下的试次重采样区间，不包含模型重拟合或参考估计不确定性，也不是人群置信区间。',
        '', '## 试次对账', '', '| 数据集 | 全部事件 | 边界剔除 | 质量剔除 | 折外评价 | 左 / 右 | 每折参考数 |', '|---|---:|---:|---:|---:|---|---|']
    for ds,d in datasets.items():
        a=d['audit'];c=d['cues']
        lines_.append(f'| {d["name"]} | {len(a)} | {(a.role=="边界剔除").sum()} | {(a.role=="质量剔除").sum()} | {len(c)} | {(c==-1).sum()} / {(c==1).sum()} | '+', '.join(str(len(z)) for z in d['ref_lists'].values())+' |')
    lines_ += ['', '参考试次在不同外折重复使用，因此每折参考数不可相加解释为独立样本数。完整事件清单与逐折角色CSV可以逐项追溯。', '', '## 同口径结果（预处理 / V3 / V4）', '', '| 数据集 | 代理MAE | 代理相关 | 代理SNR dB | 左右差分幅度比 V3 / V4 |','|---|---|---|---|---|']
    for ds,d in datasets.items():
        vals=[]
        for metric in ['MAE','corr','SNR_proxy_dB']:
            vals.append(' / '.join(f'{df[(df.dataset==ds)&(df.metric==metric)&(df.stage==k)]["mean"].iloc[0]:.3f}' for k in ['预处理','V3','V4']))
        ret=[]
        for k in ['V3','V4']:
            ret.append(sp[(sp.dataset==ds)&(sp.stage==k)&(sp.quantity=='left_minus_right_ERP')].retention_ratio.mean())
        lines_.append(f'| {d["name"]} | '+' | '.join(vals)+f' | {ret[0]:.3f} / {ret[1]:.3f} |')
    lines_ += ['', '左右幅度比以预处理ERP为分母。接近1只代表改变较小，不证明保留的一定是神经信息；未校正的伪影也可能被保留。', '', '### 改善与退步逐项列出', '']
    lower={'MAE','RMSE','shape_distance','amp_error','latency_error_ms','P300_positive_mean_error','P300_AUC_error_unit_ms'}
    for ds,d in datasets.items():
        better=[];worse=[]
        for metric in sorted(lower|{'corr','SNR_proxy_dB'}):
            a=df[(df.dataset==ds)&(df.metric==metric)].set_index('stage')
            if len(a)!=3 or not np.isfinite(a.loc['V3','mean']) or not np.isfinite(a.loc['V4','mean']):continue
            diff=a.loc['V4','mean']-a.loc['V3','mean']
            if abs(diff)<1e-8:continue
            good=(diff<0 if metric in lower else diff>0)
            (better if good else worse).append(metric)
        lines_.append(f'- {d["name"]}：相对V3数值改善：'+('、'.join(better) or '无')+'；退步：'+('、'.join(worse) or '无')+'。')
    lines_ += ['', '以上为点估计方向，并非显著性结论；基线RMS与PTP不按越低越好排序。', '', '## 正峰缺失与拟合', '', '| 数据集 | 阶段 | 正峰有效行 / 6 | 三阶段峰误差共同配对数 | 潜伏期误差共同配对数 |','|---|---|---:|---:|---:|']
    for ds,d in datasets.items():
        for k in ['预处理','V3','V4']:
            a=df[(df.dataset==ds)&(df.stage==k)].set_index('metric')
            lines_.append(f'| {d["name"]} | {k} | {int(a.loc["P300_amp","valid_rows"])} | {int(a.loc["amp_error","paired_rows"])} | {int(a.loc["latency_error_ms","paired_rows"])} |')
    if fits:
        lines_ += ['',f'24个方向×通道拟合中，{sum(r["reliable"] for r in fits)}个通过预先写入代码的工程筛查。其余明确标注不支持可靠参数解释。规则：数值收敛、R²≥0.5、幅度≥max(0.5,0.25×拟合RMSE)、潜伏期/宽度不贴边、分析窗内有正峰。阈值是工程规则，未经临床验证。']
    lines_ += ['', '## 多场景半合成验证', '', '四组独立外层测试背景，5折、3种子、0/0.5/1/2注入幅度；随机改变时刻、宽度、极性，包含眨眼样同相、方向相关扫视样及运动脉冲。污染前实测波形作为已知恢复目标；它自身仍可能带有伪影。', '', '| 注入幅度 | 预处理RMSE | V3 RMSE | V4 RMSE | V3 / V4 左右恢复误差 |', '|---:|---:|---:|---:|---|']
    for level,g in synthetic.groupby('level'):
        m=g.groupby('stage')[['RMSE','contrast_RMSE']].mean()
        lines_.append(f'| {level} | {m.loc["预处理","RMSE"]:.3f} | {m.loc["V3","RMSE"]:.3f} | {m.loc["V4","RMSE"]:.3f} | {m.loc["V3","contrast_RMSE"]:.3f} / {m.loc["V4","contrast_RMSE"]:.3f} |')
    lines_ += ['', '以上按数据集、折、种子等权。完整CSV保留每次结果，0幅度专门检测算法自身失真。不能把该测试视为临床泛化验证。', '', '## 算法与已定位限制', '',
        '- V3使用刺激前均值重新定位，V4统一中位数基线。固定候选消融表展示单改基线、减小水平校正、提高门控、时域保护的结果；候选消融使用真实评价数据只作诊断，不参与选型。',
        '- 时域保护只保护训练模板构成的时间子空间，不强制恢复测试集左右差分。可保留与模板相似的伪影，不能据此声称严格分离脑电与眼电。',
        '- 999幅度、15个饱和值、1800峰峰值、600跳变沿用V3；饱和值统计是全通道点数总和，并非连续15点。阈值敏感性只考察后三个阈值±10%，不改变当前剔除集，也不证明硬件饱和机制。',
        '- 峰潜伏期只有3.90625 ms采样分辨率；边缘频段能量比与代理SNR只是描述量。分方向图不代表顶区P300空间分布。',
        '', '## 重现与文件', '',
        '运行 `python3 EEG_P300_artifact_correction_v4.py --output eeg_v4_results`。依赖见 requirements-v4.txt。可用 `--no-plots` 输出数值复核副本。',
        '', '每组6张图、完整事件审计、逐折角色、逐方向指标、试次区间、拟合参数、固定典型试次编号及NPZ波形。跨项目图按方向展示。运行清单包含输入、源码和数值结果哈希。基线快照保存优化前代码与已有中文结果。',
        '', '验证结论：可以作为带限制说明的第一问实验结果使用；代理参考、两名受试者和半合成污染均不足以支持真实无噪声恢复或疾病诊断准确率声明。']
    (out/'第四版结果说明.md').write_text('\n'.join(lines_)+'\n',encoding='utf-8')


if __name__=='__main__':main()
