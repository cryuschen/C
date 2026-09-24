#!/usr/bin/env python3
"""第一问 V5 可复核实验：固定试次上比较 V3、V4、训练折趋势驱动的 V5。

运行：python EEG_P300_artifact_correction_v5.py --output eeg_v5_results
V5 属四组数据上的开发实验；任何试次区间都不是跨受试者泛化验证。
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/eeg-v5-mpl')
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import EEG_P300_artifact_correction_v3 as core
import EEG_P300_artifact_correction_v4 as old

ROOT=Path(__file__).resolve().parent
TIMES=core.TIMES_MS
WINDOW=core.P300_MASK
N_PRE=core.N_PRE
CHANNELS=core.CHANNEL_NAMES
STAGES=('预处理','V3','V4','V5')
COLORS={'原始':'#a8adb4','预处理':'#596e83','V3':'#c87927','V4':'#6b61a8','V5':'#007f89','代理参考':'#30343a'}
LINES={'原始':':','预处理':'--','V3':'-.','V4':':','V5':'-','代理参考':':'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(data,path):
    pd.DataFrame(data).to_csv(path,index=False,encoding='utf-8-sig')


def prestim_slopes(epochs):
    """逐试次逐通道 OLS 刺激前斜率，单位原始单位/ms；不使用刺激后样本。"""
    t=TIMES[:N_PRE]
    center=t-t.mean()
    x=epochs[:,:,:N_PRE]
    return ((x-x.mean(axis=2,keepdims=True))*center).sum(axis=2)/(center@center)


def strategy_from_training(x_train):
    """规则固定于 V5 开发：只看外层训练折刺激前斜率的中位数。"""
    median=float(np.median(prestim_slopes(x_train)))
    if median<.03:
        return 'V3',0.,median
    if median<.075:
        return 'V4',.10,median
    return 'V3',.15,median


def drift_correct(output,input_epochs,alpha):
    if alpha==0:
        return output.copy()
    slopes=prestim_slopes(input_epochs)
    result=output-alpha*slopes[:,:,None]*TIMES[None,None,:]
    return core.robust_baseline_correct(result)


def apply_v3(x,cues,model):
    tpl,sv,sh,_=model
    return np.stack([core.correct_single_epoch_v3(e,tpl[c],sv[c],sh[c]) for e,c in zip(x,cues)])


def apply_v4(x,cues,model,candidate):
    return old.apply_model(x,cues,model,candidate)


def positive_presence(erp):
    return np.isfinite(core.positive_p300_features(erp)[2])


def extended_metrics(before,methods,cues,reference):
    frames=[]
    for name,x in methods.items():
        df=core.evaluate_metrics(before,x,cues,reference)
        df.insert(0,'stage',name)
        df['n_trials']=[int((cues==c).sum()) for c in df.cue]
        df['positive_peak_valid']=np.isfinite(df.P300_amp_after)
        df['reference_positive_peak_valid']=[positive_presence(reference[int(c)][CHANNELS.index(ch)]) for c,ch in zip(df.cue,df.channel)]
        df['latency_error_valid']=np.isfinite(df.latency_error_after_ms)
        frames.append(df)
    return pd.concat(frames,ignore_index=True)


def summarize_metrics(df,dataset):
    rows=[]
    columns=[c[:-6] for c in df if c.endswith('_after')]
    columns+=['latency_error_ms','P300_AUC_unit_ms','P300_AUC_error_unit_ms','P300_lat_ms']
    special={'latency_error_ms':'latency_error_after_ms','P300_AUC_unit_ms':'P300_AUC_after_unit_ms',
             'P300_AUC_error_unit_ms':'P300_AUC_error_after_unit_ms','P300_lat_ms':'P300_lat_after_ms'}
    for metric in dict.fromkeys(columns):
        col=special.get(metric,metric+'_after')
        if col not in df:continue
        value=np.stack([df[df.stage==s][col].to_numpy(float) for s in STAGES])
        common=np.isfinite(value).all(axis=0)
        for i,stage in enumerate(STAGES):
            rows.append(dict(dataset=dataset,metric=metric,stage=stage,
                             mean=float(value[i,common].mean()) if common.any() else np.nan,
                             paired_rows=int(common.sum()),valid_rows=int(np.isfinite(value[i]).sum()),total_rows=6))
    return rows


def bootstrap_metrics(methods,cues,reference,n_boot=400):
    """同一方向复用同一批重采样索引；模型和代理参考保持固定。"""
    rows=[]
    for condition in (-1,1):
        ids=np.flatnonzero(cues==condition)
        rng=np.random.default_rng(20260924+(condition+1))
        draw=rng.integers(0,len(ids),(n_boot,len(ids)))
        for ch,channel in enumerate(CHANNELS):
            reference_window=reference[condition][ch,WINDOW]
            boot={}
            for stage,x in methods.items():
                erp=x[ids,ch][draw].mean(axis=1)
                w=erp[:,WINDOW]
                boot[stage]={
                    'proxy_MAE':np.mean(np.abs(w-reference_window),axis=1),
                    'positive_mean':np.maximum(w,0).mean(axis=1),
                }
            for metric in ('proxy_MAE','positive_mean'):
                for stage in STAGES:
                    arr=boot[stage][metric]
                    lo,hi=np.quantile(arr,[.025,.975])
                    rows.append(dict(cue=condition,channel=channel,metric=metric,stage=stage,
                                     low=lo,high=hi,n_trials=len(ids),n_boot=n_boot))
                for baseline in ('预处理','V3','V4'):
                    arr=boot['V5'][metric]-boot[baseline][metric]
                    lo,hi=np.quantile(arr,[.025,.975])
                    rows.append(dict(cue=condition,channel=channel,metric=metric,stage='V5减'+baseline,
                                     low=lo,high=hi,n_trials=len(ids),n_boot=n_boot))
    return rows


def spline_fit(erp):
    """三次平滑样条只作为响应曲线拟合，不强制三个高斯生理成分。"""
    mask=(TIMES>=50)&(TIMES<=750)
    t=TIMES[mask]
    y=erp[mask]
    second=np.diff(y,n=2)
    noise=np.median(np.abs(second-np.median(second)))/(.6745*np.sqrt(6)+1e-12)
    smooth=max(float(noise),float(np.std(y))*.015,.25)
    spline=UnivariateSpline(t,y,k=3,s=len(y)*smooth*smooth)
    fitted=spline(TIMES)
    residual=erp-fitted
    win=np.flatnonzero(WINDOW)
    segment=fitted[win]
    prominence=max(1.,.10*float(np.ptp(segment)))
    peaks,properties=find_peaks(segment,prominence=prominence)
    peaks=peaks[(peaks>2)&(peaks<len(segment)-3)&(segment[peaks]>0)]
    if len(peaks):
        best=peaks[np.argmax(segment[peaks])]
        peak=float(segment[best]);lat=float(TIMES[win[best]])
        status='窗内局部正峰，非生理确认'
    else:
        peak=np.nan;lat=np.nan;status='无合格窗内局部正峰'
    return fitted,dict(fit_family='cubic_smoothing_spline',smoothing_s=float(spline.get_residual()),
                       noise_scale_used=smooth,fit_RMSE_50_750=float(np.sqrt(np.mean(residual[mask]**2))),
                       fit_RMSE_250_500=float(np.sqrt(np.mean(residual[WINDOW]**2))),
                       positive_mean=float(np.mean(np.maximum(segment,0))),
                       positive_area_unit_ms=float(np.trapezoid(np.maximum(segment,0),TIMES[WINDOW])),
                       local_positive_peak=peak,local_peak_latency_ms=lat,peak_status=status)


def make_axes(title,subtitle):
    fig,axes=plt.subplots(3,2,figsize=(13.5,9.5),sharex=True,layout='constrained')
    fig.suptitle(title+'\n'+subtitle,fontsize=14)
    for ax in axes.flat:
        ax.axvline(0,color='#aeb5bc',lw=.75)
        ax.axhline(0,color='#cbd0d5',lw=.6)
        ax.axvspan(250,500,color='#d5ebed',alpha=.27)
        ax.set_xlabel('相对提示时间（ms）')
        ax.set_ylabel('原始电位单位')
        ax.spines[['top','right']].set_visible(False)
    return fig,axes


def shared_row_limits(axes,values,margin=.06):
    for row in range(3):
        combined=np.concatenate([np.asarray(v).ravel() for v in values[row]])
        low,high=float(np.nanmin(combined)),float(np.nanmax(combined))
        pad=max((high-low)*margin,1.)
        for col in range(2): axes[row,col].set_ylim(low-pad,high+pad)


def draw_lines(ax,signals,mask=None,legend=True):
    mask=np.ones(len(TIMES),bool) if mask is None else mask
    for label,y in signals.items():
        ax.plot(TIMES[mask],np.asarray(y)[mask],label=label,color=COLORS[label],
                ls=LINES[label],lw=1.5 if label=='V5' else 1.)
    if legend:ax.legend(loc='upper left',fontsize=8,ncol=2)


def save_fig(fig,path):
    fig.savefig(path,dpi=170)
    plt.close(fig)


def plot_condition(name,methods,cues,refs,out,zoom=False):
    title=name+(' · 250–500 ms 正向响应窗' if zoom else ' · 左右刺激三通道 ERP')
    subtitle='各图同一有效试次，同行左右共享纵轴；代理参考只作描述' if zoom else '四阶段同一试次；阴影是V5逐点95%试次重采样区间'
    fig,axes=make_axes(title,subtitle)
    scale={k:[] for k in range(3)}
    for col,c in enumerate((-1,1)):
        n=int((cues==c).sum())
        for ch,channel in enumerate(CHANNELS):
            ax=axes[ch,col]
            curves={k:x[cues==c,ch].mean(axis=0) for k,x in methods.items()}
            if zoom:curves['代理参考']=refs[c][ch]
            draw_lines(ax,curves,WINDOW if zoom else None)
            if not zoom:
                lower,upper=core.compute_bootstrap_ci(methods['V5'][cues==c,ch],n_boot=400)
                ax.fill_between(TIMES,lower,upper,color=COLORS['V5'],alpha=.14,label='V5逐点95%试次区间')
                ax.legend(loc='upper left',fontsize=8,ncol=2)
                scale[ch].extend([*curves.values(),lower,upper])
            else:scale[ch].extend([v[WINDOW] for v in curves.values()])
            ax.set_title(f'{channel} · {"左" if c==-1 else "右"}刺激 · n={n}')
            if zoom:ax.set_xlim(250,500)
    shared_row_limits(axes,scale)
    save_fig(fig,out/('P300分析窗对比.png' if zoom else '左右刺激三通道ERP.png'))


def plot_examples(name,raw,methods,cues,scores,trial_ids,out,quantile,filename):
    fig,axes=make_axes(name+' · 固定试次原始/处理对照',f'同方向伪影评分第{int(100*quantile)}百分位；原始试次编号已标注，纵轴不截断')
    scale={k:[] for k in range(3)}
    selected=[]
    for col,c in enumerate((-1,1)):
        ids=np.flatnonzero(cues==c)
        order=ids[np.argsort(scores[ids],kind='stable')]
        ix=order[round(quantile*(len(order)-1))]
        selected.append(dict(cue=c,quantile=quantile,trial_id=int(trial_ids[ix])))
        for ch,channel in enumerate(CHANNELS):
            original=raw[ix,ch]-np.median(raw[ix,ch,:N_PRE])
            curves={'原始':original,**{k:x[ix,ch] for k,x in methods.items()}}
            draw_lines(axes[ch,col],curves)
            axes[ch,col].set_title(f'{channel} · {"左" if c==-1 else "右"} · 原始试次 #{trial_ids[ix]}')
            scale[ch].extend(curves.values())
    shared_row_limits(axes,scale)
    save_fig(fig,out/filename)
    return selected


def plot_heatmap(name,methods,cues,trial_ids,out,focus=False):
    stages=list(methods)
    all_abs=np.concatenate([np.abs(x[:,0]).ravel() for x in methods.values()])
    full=float(all_abs.max())
    limit=float(np.quantile(all_abs,.98)) if focus else full
    exceed=float(np.mean(all_abs>limit))*100
    fig,axes=plt.subplots(2,len(stages),figsize=(17,8),layout='constrained')
    for row,c in enumerate((-1,1)):
        ids=np.flatnonzero(cues==c)
        for col,stage in enumerate(stages):
            ax=axes[row,col]
            image=ax.imshow(methods[stage][ids,0,:],aspect='auto',origin='lower',interpolation='nearest',
                            cmap='RdBu_r',vmin=-limit,vmax=limit,extent=[TIMES[0],TIMES[-1],.5,len(ids)+.5])
            ticks=np.unique(np.linspace(0,len(ids)-1,5,dtype=int))
            ax.set_yticks(ticks+1,[str(i) for i in trial_ids[ids[ticks]]])
            ax.set_title(f'{"左" if c==-1 else "右"}刺激 · {stage} · n={len(ids)}')
            ax.set_xlabel('相对提示时间（ms）')
            ax.set_ylabel('原始试次编号')
    fig.colorbar(image,ax=axes,label='Fz 原始电位单位')
    subtitle=(f'细节视图：全组四阶段共同98%绝对值分位 ±{limit:.1f}；{exceed:.2f}%像素截色' if focus
              else f'完整色域：全组四阶段共用 ±{limit:.1f}，无截色')
    fig.suptitle(name+' · Fz 试次时间热力图\n'+subtitle)
    save_fig(fig,out/('Fz热力图_细节视图.png' if focus else 'Fz热力图_完整色域.png'))


def plot_spatial(name,methods,cues,out):
    fig,axes=make_axes(name+' · 左右与额区差分','左列为左减右；右列为F3减F4；同图内四阶段同尺度；差分可能包含残余眼动')
    scale={k:[] for k in range(3)}
    for ch,channel in enumerate(CHANNELS):
        signals={s:x[cues==-1,ch].mean(0)-x[cues==1,ch].mean(0) for s,x in methods.items()}
        draw_lines(axes[ch,0],signals)
        axes[ch,0].set_title(channel+' · 左减右')
        scale[ch].extend(signals.values())
    for row,c in enumerate((-1,1)):
        signals={s:x[cues==c,1].mean(0)-x[cues==c,2].mean(0) for s,x in methods.items()}
        draw_lines(axes[row,1],signals)
        axes[row,1].set_title(('左' if c==-1 else '右')+'刺激 · F3减F4')
        scale[row].extend(signals.values())
    fig.delaxes(axes[2,1])
    # 差分类型不同，不强制左右列同一幅度；每图方法共享其自身坐标。
    save_fig(fig,out/'左右与F3-F4差分.png')


def plot_fits(name,methods,cues,out):
    fig,axes=make_axes(name+' · V5 三通道分方向响应曲线拟合',
                       '三次平滑样条；灰色为残差；窗内局部正峰只是描述量，不确认生理P300')
    records=[]
    for col,c in enumerate((-1,1)):
        n=int((cues==c).sum())
        for ch,channel in enumerate(CHANNELS):
            erp=methods['V5'][cues==c,ch].mean(0)
            fitted,fit=spline_fit(erp)
            ax=axes[ch,col]
            ax.plot(TIMES,erp,color=COLORS['V5'],label='V5 ERP')
            mask=(TIMES>=50)&(TIMES<=750)
            ax.plot(TIMES[mask],fitted[mask],color='#32373c',ls='--',label='样条拟合')
            ax.plot(TIMES[mask],(erp-fitted)[mask],color='#9a938d',lw=.8,label='残差')
            if np.isfinite(fit['local_peak_latency_ms']):
                ax.scatter([fit['local_peak_latency_ms']],[fit['local_positive_peak']],s=14,color='#32373c',zorder=3)
            ax.legend(fontsize=8,loc='upper left')
            ax.set_title(f'{channel} · {"左" if c==-1 else "右"} · n={n} · 窗内拟合RMSE={fit["fit_RMSE_250_500"]:.2f}\n'+fit['peak_status'],fontsize=10)
            records.append(dict(cue=c,channel=channel,n_trials=n,**fit))
    save_fig(fig,out/'分方向样条拟合与残差.png')
    save_csv(records,out/'分方向样条拟合参数.csv')
    return records


def stage_summary_markdown(data,dataset):
    subset=pd.DataFrame(data)
    subset=subset[subset.dataset==dataset]
    result=[]
    for metric in ('MAE','corr','SNR_proxy_dB','baseline_RMS','P300_positive_mean_error','P300_AUC_error_unit_ms'):
        rows=subset[subset.metric==metric].set_index('stage')
        if len(rows)==4:
            result.append('| '+metric+' | '+' | '.join(f'{rows.loc[s,"mean"]:.3f}' for s in STAGES)+' | '+str(int(rows.loc['V5','paired_rows']))+' |')
    return result


def write_report(out,datasets,summary,spatial,benchmark,fitrows):
    summary=pd.DataFrame(summary)
    spatial=pd.DataFrame(spatial)
    benchmark=pd.DataFrame(benchmark)
    fit=pd.DataFrame(fitrows)
    lines=['# 第一问：脑电预处理、伪影校正与有效视觉响应拟合（终稿实验报告）','',
    '## 赛题对应与结果定位','',
    '针对项目一与项目二、Fz/F3/F4原始记录，完成视觉提示事件提取、固定质量筛查、连续信号预处理、折外校正、左右方向分层响应估计及曲线拟合。刺激方向从VisCue读取，仅用于已知条件下的离线分析；本结果不是未知刺激解码器。',
    '', '核心结论：V5在四组数据上均降低了相对预处理波形的代理MAE，并给出可复核的平滑响应曲线；然而代理参考不是无伪影真值，左右差分幅度在部分组仍显著缩小。故不能宣称所有视觉形状特征已被完整保留。',
    '', '## 数据与数学定义','',
    '- 四组输入各100个提示事件；只用原始Fz、F3、F4与VisCue，绝不使用机器处理后的FzDecon/F3Decon/F4Decon作为输入。采样率256 Hz；试次窗−250至796.875 ms，分析窗250–500 ms。幅度单位沿用“原始电位单位”，不假定µV。',
    '- 固定预处理：连续信号60 Hz陷波、0.1–30 Hz零相位带通、刺激前250 ms中位数基线校正。提高高通截止频率会改变慢ERP的幅值和潜伏期，因此未通过抬高截止频率强行消除项目二的慢变化。',
    '- 严重坏试次规则沿用V3：原始跨通道采样绝对值≥999的总点数≥15，或滤波后峰峰值>1800，或相邻点跳变>600；这些是数据量纲下的工程阈值。没有证据表明“15点”是连续硬件饱和，指标解释按实际计算。',
    '- 五折外层分层划分、固定随机种子42。每折仅用训练试次估计方向条件模板、残差尺度和选型；当前折测试试次不进入自己的模板。被剔除试次不进入任何ERP、拟合或指标。',
    '- V3为原双分量校正；V4为前版训练内半合成选型；V5增加训练折刺激前趋势诊断。设第i试次通道c的刺激前最小二乘斜率为b_ic，训练折所有b_ic的中位数为m。m<0.03原始单位/ms时采用V3；0.03≤m<0.075时采用V4并从输出减去0.10·b_ic·t；m≥0.075时采用V3并减去0.15·b_ic·t，最后重新以刺激前中位数校正。所用阈值是在这四组数据的开发分析后冻结；不能把外层折结果解释为全新受试者上的独立验证。',
    '- 刺激前斜率和刺激后慢变化存在相关，但不能仅据此把后者认定为伪影。V5的趋势项以刺激前斜率外推，在先验上不使用该试次刺激后的形状拟合趋势；它仍可能改变真实慢电位，因此与V3/V4逐项并报。',
    '- 代理参考为当前外折训练集中同方向低伪影半数试次的逐点中位数；每个评价试次对应训练模板按条件平均用于指标。它与校正模型相关，不是独立无噪声真值。V3/V4/V5和预处理在相同试次、相同参考和相同窗内比较，六个方向×通道行等权；缺失指标使用四阶段共同有效行配对，不补零。',
    '- 曲线拟合采用三次平滑样条，平滑强度按ERP局部差分噪声尺度设定，输出50–750 ms和250–500 ms残差RMSE。只有250–500 ms内的局部正峰才报告峰时刻；正面积可在无局部峰时报告。它仅是额区正向响应表征，不能称已经在中央/顶区观察到典型P300。',
    '', '## 试次与训练参考对账','',
    '| 数据组 | 事件 | 边界剔除 | 质量剔除 | 可用并折外评价 | 左 / 右 | 每折训练参考数 |',
    '|---|---:|---:|---:|---:|---|---|']
    for ds,d in datasets.items():
        a=d['audit'];c=d['cues']
        refs=d['reference_counts']
        lines.append(f'| {d["name"]} | {len(a)} | {int((a.role=="边界剔除").sum())} | {int((a.role=="质量剔除").sum())} | {len(c)} | {int((c==-1).sum())} / {int((c==1).sum())} | '+', '.join(map(str,refs))+' |')
    lines += ['', '训练参考在不同外折可以重复出现，上表不能将每折参考数简单相加。各组“完整事件与试次审计.csv”“逐折训练参考评价清单.csv”“可复核波形.npz”保留原始试次编号和折号。',
             '', '## 四阶段同口径指标','',
             '表中“V3”指按共同训练评分和参考流程重算的V3校正，数值不得与旧版不同评价口径CSV直接相减。MAE及相关针对代理参考；SNR为ERP功率/试次残差功率的代理值，残差也含真实试次差异。',
             '', '| 数据组 | 指标 | 预处理 | V3 | V4 | V5 | 共同有效行数/6 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for ds,d in datasets.items():
        for row in stage_summary_markdown(summary,ds):
            lines.append('| '+d['name']+' | '+row[2:])
    lines += ['', '### 视觉形状与空间差分','',
              '左减右、F3减F4均用相同有效试次计算。差分范数比以预处理差分为分母；接近1只说明幅度更接近处理前，不能区分保留神经信息与保留方向相关眼动。',
              '', '| 数据组 | V3左右差分比 | V4左右差分比 | V5左右差分比 | V5差分波形相关均值 |',
              '|---|---:|---:|---:|---:|']
    for ds,d in datasets.items():
        g=spatial[(spatial.dataset==ds)&(spatial.quantity=='left_minus_right_ERP')]
        vals=[g[g.stage==s].retention_ratio.mean() for s in ('V3','V4','V5')]
        corr=g[g.stage=='V5'].waveform_correlation.mean()
        lines.append(f'| {d["name"]} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {corr:.3f} |')
    lines += ['', 'F3/F4空间差分及每通道左右差分的完整数值见“左右刺激与额区空间差异指标.csv”。即使总体平均值较好，也不能掩盖某通道差分幅度或形状明显下降。',
              '', '### 正峰缺失、样条拟合与不确定性','',
              '逐方向逐通道CSV列出每阶段`positive_peak_valid`、`reference_positive_peak_valid`与潜伏期有效性。无正峰不赋予零潜伏期。样条拟合全部可计算，但“可拟合”不等于“有可信生理P300”。',
              '', '| 数据组 | V5有合格窗内局部正峰 / 6 | V5正峰检出行 / 6 | 四阶段峰误差共同配对行 / 6 |',
              '|---|---:|---:|---:|']
    for ds,d in datasets.items():
        f=fit[fit.dataset==ds]
        m=d['metrics'];v=m[m.stage=='V5']
        paired=summary[(summary.dataset==ds)&(summary.metric=='amp_error')&(summary.stage=='V5')]
        lines.append(f'| {d["name"]} | {int((f.peak_status=="窗内局部正峰，非生理确认").sum())} | {int(v.positive_peak_valid.sum())} | {int(paired.paired_rows.iloc[0]) if len(paired) else 0} |')
    lines += ['', '区间文件对每组每方向每通道进行400次试次重采样，固定处理后的波形与参考，提供95%逐指标区间和V5−基线配对差的区间。它仅量化本组试次有限样本波动，不能推断人群疗效或诊断能力。',
              '', '## 半合成闭环验证','',
              '在四组各外折测试试次上人工加入随机时刻、宽度、极性的眨眼样、扫视样和运动样扰动；污染前的预处理实测波形是可计算恢复目标，仍可能含原有伪影。固定三个测试随机种子和0/0.5/1/2倍注入；0倍只检验无新污染时算法改动背景的程度。每组每幅度、每算法另存原始单位RMSE、按背景RMS归一化RMSE、左右差分恢复误差和正向均值误差。',
              '', '| 数据组 | 幅度 | 未校正归一化RMSE | V3 | V4 | V5 |',
              '|---|---:|---:|---:|---:|---:|']
    for ds,d in datasets.items():
        for level in (0.,1.,2.):
            g=benchmark[(benchmark.dataset==ds)&(benchmark.level==level)]
            values=[g[g.stage==s].normalized_RMSE.mean() for s in STAGES]
            lines.append(f'| {d["name"]} | {level:g} | '+' | '.join(f'{x:.3f}' for x in values)+' |')
    lines += ['', '不能将这些半合成结果推广为真实无噪声脑电恢复精度；没有专门的眼电或独立真值。各试次重复注入也不能当作独立受试者。',
              '', '## 图表检查与解释','',
              '- 同方向、同通道四阶段共享试次；左右面板同一通道共享纵轴。颜色/线型固定。置信带来自试次重采样，注明为逐点区间。',
              '- 典型波形同时给50%与75%伪影评分试次，不把75%试次称普通典型；标注原始编号。两图都使用完整幅度，不截断原始大波形。',
              '- 热力图同时给完整共用色域与共用98%分位细节视图；细节图显式写出截色像素比例。不得凭细节图单独宣称大波形消失。',
              '- 跨项目图按左、右方向分别比较，各方向写明样本数；拟合图显示残差和窗内拟合RMSE。',
              '', '## 结论边界和可复现性','',
              'V5是四组数据上经过开发调试的折外实验结果：操作层面测试试次未进入其折的模板或策略判断，但开发者已查看全部四组数据，故这些数字是开发集结果，不应称为全新受试者独立验证。三额区通道没有眼电通道，无法从这份数据单独证明前额共同缓慢变化是眼电还是神经慢电位。',
              '', '要把结果用于未知刺激分类，需要在完全独立数据上重新建立不使用测试Cue的校正和分类流程；当前曲线是已知条件下的视觉响应描述。数模论文可用本结果讨论可观测信号和方法取舍，不能声称神经源唯一识别、临床诊断准确率或人群泛化。',
              '', '运行 `python EEG_P300_artifact_correction_v5.py --output eeg_v5_results`，读取各组“可复核波形.npz”和运行清单JSON重算。源程序、四份.mat文件以及数值结果的SHA256均保存。',
              '', '## 方法出处','',
              '- [Tanner等：不适当高通滤波可能使认知ERP产生人为效应](https://pmc.ncbi.nlm.nih.gov/articles/PMC4506207/)。这里据此保留0.1 Hz基线滤波并把更强滤波仅作为未采用的备选。',
              '- [Luck等：伪影校正与剔除的ERP评价框架](https://pmc.ncbi.nlm.nih.gov/articles/11021170/)。这里据此同时报告噪声代理量、响应特征与局限，而非只凭一种误差证明去噪。']
    (out/'第一问终稿实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot_cross_task(out,datasets):
    folder=out/'跨项目对比';folder.mkdir(exist_ok=True)
    for subject in 'AB':
        fig,axes=make_axes('受试者'+subject+' · V5按方向跨项目ERP',
                           '同通道左右共享纵轴；项目一/二仅描述同受试者条件差异，不作为跨人群效应')
        scale={ch:[] for ch in range(3)}
        for col,c in enumerate((-1,1)):
            for ch,channel in enumerate(CHANNELS):
                ax=axes[ch,col]
                for task,color,style in ((1,'#007f89','-'),(2,'#c87927','--')):
                    d=datasets[f'VisualCog{subject}_Task-{task}']
                    y=d['stages']['V5'][d['cues']==c,ch].mean(0)
                    ax.plot(TIMES,y,color=color,ls=style,label=f'项目{task} · n={(d["cues"]==c).sum()}')
                    scale[ch].append(y)
                ax.legend(fontsize=8)
                ax.set_title(channel+' · '+('左' if c==-1 else '右')+'刺激')
        shared_row_limits(axes,scale)
        save_fig(fig,folder/f'受试者{subject}_左右分方向跨项目.png')


def plot_benchmark(out,synthetic):
    folder=out/'半合成验证'
    frame=pd.DataFrame(synthetic)
    fig,axes=plt.subplots(2,2,figsize=(12.5,8),layout='constrained',sharex=True)
    for ax,(ds,g) in zip(axes.flat,frame.groupby('dataset',sort=False)):
        for stage in STAGES:
            s=g[g.stage==stage].groupby('level').normalized_RMSE.mean()
            ax.plot(s.index,s.values,color=COLORS[stage],ls=LINES[stage],marker='o',label=stage)
        ax.set_title(ds+' · 背景样本与外折固定')
        ax.set_xlabel('人工注入幅度倍数（0表示未注入）')
        ax.set_ylabel('恢复RMSE / 背景RMS')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8,ncol=2)
    fig.suptitle('四组各自归一化：已知注入扰动恢复与零注入背景改动\n污染前为实测预处理背景，可能已含伪影；试次重用不表示独立受试者')
    save_fig(fig,folder/'四组多幅度半合成验证.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'eeg_v5_results')
    parser.add_argument('--no-plots',action='store_true',help='生成数值副本供重复性核验')
    args=parser.parse_args()
    out=args.output.resolve()
    if out==ROOT or any(out.is_relative_to(p) for p in (ROOT/'data',ROOT/'eeg_v2_results',ROOT/'eeg_v3_results',ROOT/'eeg_v4_results')):
        raise ValueError('输出路径不得覆盖原始数据与旧结果')
    out.mkdir(parents=True,exist_ok=True)
    summary_dir=out/'汇总与说明';summary_dir.mkdir(exist_ok=True)
    synth_dir=out/'半合成验证';synth_dir.mkdir(exist_ok=True)
    manifest={'inputs':{},'source':{p.name:sha(p) for p in (Path(__file__),ROOT/'EEG_P300_artifact_correction_v3.py',ROOT/'EEG_P300_artifact_correction_v4.py')},
              'python':platform.python_version(),'numpy':np.__version__,'fold_seed':42,'test_seeds':[2027,2039,2053],
              'unit':'原始电位单位','pre_slope_strategy':{'low':.03,'high':.075,'medium_alpha':.10,'high_alpha':.15}}
    datasets={};summary=[];spatial=[];synthetic=[];fits=[];strategies=[];slope_diagnostics=[]
    for subject in 'AB':
        for task in (1,2):
            ds=f'VisualCog{subject}_Task-{task}'
            name=f'受试者{subject}_项目'+('一' if task==1 else '二')
            input_path=ROOT/'data'/(ds+'.mat')
            manifest['inputs'][input_path.name]=sha(input_path)
            dest=out/name;dest.mkdir(exist_ok=True)
            raw,x,cues,scores,audit,sensitivity=old.read_dataset(input_path)
            trial_ids=audit.loc[audit.role=='折外评价','trial_id'].to_numpy(int)
            n=len(x);v3=np.empty_like(x);v4=np.empty_like(x);v5=np.empty_like(x);refs_trial=np.empty_like(x)
            folds=np.zeros(n,dtype=int);reference_counts=[];fold_roles=[]
            cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
            for fold,(train,test) in enumerate(cv.split(x,cues),1):
                train_scores=core.detect_bad_trials(raw[train],x[train])[1]
                model=core.build_clean_reference_model(x[train],cues[train],np.zeros(len(train),bool),train_scores)
                v4_choice,_=old.select_candidate(x[train],cues[train],100+fold)
                method,alpha,median_slope=strategy_from_training(x[train])
                for j in test:
                    refs_trial[j]=model[0][cues[j]]
                    fold_roles.append(dict(fold=fold,trial_id=int(trial_ids[j]),role='折外评价'))
                reference_ids=set(trial_ids[train[model[3]]])
                assert not reference_ids.intersection(trial_ids[test])
                reference_counts.append(len(reference_ids))
                for j in train:
                    fold_roles.append(dict(fold=fold,trial_id=int(trial_ids[j]),role='训练参考' if trial_ids[j] in reference_ids else '训练非参考'))
                v3[test]=apply_v3(x[test],cues[test],model)
                v4[test]=apply_v4(x[test],cues[test],model,v4_choice)
                selected=v3[test] if method=='V3' else v4[test]
                v5[test]=drift_correct(selected,x[test],alpha)
                folds[test]=fold
                strategies.append(dict(dataset=ds,fold=fold,train_n=len(train),test_n=len(test),
                                       train_median_pre_slope=median_slope,V4_candidate=v4_choice,
                                       V5_base=method,trend_strength=alpha,reference_count=len(reference_ids)))
                # 同试次、同注入、同训练折；V5先处理污染后的试次，再由其刺激前斜率求趋势项。
                for seed in (2027,2039,2053):
                    for level in (0.,.5,1.,2.):
                        contaminated=old.inject(x[test],cues[test],seed+fold,level)
                        s3=apply_v3(contaminated,cues[test],model)
                        s4=apply_v4(contaminated,cues[test],model,v4_choice)
                        s5=drift_correct(s3 if method=='V3' else s4,contaminated,alpha)
                        for stage,recovered in zip(STAGES,(contaminated,s3,s4,s5)):
                            result=old.known_errors(x[test],recovered,cues[test])
                            synthetic.append(dict(dataset=ds,fold=fold,seed=seed,level=level,stage=stage,
                                                  n_trials=len(test),V5_base=method,trend_strength=alpha,**result))
            assert (folds>0).all()
            audit.loc[audit.role=='折外评价','fold']=folds
            save_csv(audit,dest/'完整事件与试次审计.csv')
            save_csv(fold_roles,dest/'逐折训练参考评价清单.csv')
            save_csv(sensitivity,dest/'坏试次阈值敏感性.csv')
            refs={c:refs_trial[cues==c].mean(axis=0) for c in (-1,1)}
            stages=dict(zip(STAGES,(x,v3,v4,v5)))
            metrics=extended_metrics(x,stages,cues,refs)
            save_csv(metrics,dest/'逐方向逐通道评价指标.csv')
            summary.extend(summarize_metrics(metrics,ds))
            save_csv(bootstrap_metrics(stages,cues,refs),dest/'试次重采样区间.csv')
            for stage,y in stages.items():
                spatial.extend([dict(stage=stage,**z) for z in core.evaluate_spatial_metrics(x,y,cues,ds)])
                for c in (-1,1):
                    for ch,channel in enumerate(CHANNELS):
                        early=y[cues==c,ch][:,(TIMES>=0)&(TIMES<200)].mean()
                        late=y[cues==c,ch][:,(TIMES>=600)&(TIMES<800)].mean()
                        slope_diagnostics.append(dict(dataset=ds,stage=stage,cue=c,channel=channel,
                                                      n_trials=int((cues==c).sum()),late_minus_early=late-early,
                                                      mean_pre_slope=float(prestim_slopes(y)[cues==c,ch].mean())))
            np.savez_compressed(dest/'可复核波形.npz',raw=raw,before=x,v3=v3,v4=v4,v5=v5,cues=cues,
                                trial_ids=trial_ids,folds=folds,reference_per_trial=refs_trial,times_ms=TIMES)
            dataset=dict(name=name,audit=audit,cues=cues,stages=stages,metrics=metrics,reference_counts=reference_counts)
            datasets[ds]=dataset
            if not args.no_plots:
                plot_condition(name,stages,cues,refs,dest)
                plot_condition(name,stages,cues,refs,dest,zoom=True)
                examples=[]
                examples+=plot_examples(name,raw,stages,cues,scores,trial_ids,dest,.50,'中位伪影试次_全幅对照.png')
                examples+=plot_examples(name,raw,stages,cues,scores,trial_ids,dest,.75,'较高伪影试次_全幅对照.png')
                save_csv(examples,dest/'固定示例试次编号.csv')
                plot_heatmap(name,stages,cues,trial_ids,dest,focus=False)
                plot_heatmap(name,stages,cues,trial_ids,dest,focus=True)
                plot_spatial(name,stages,cues,dest)
                fits.extend([dict(dataset=ds,**z) for z in plot_fits(name,stages,cues,dest)])
            else:
                local_fits=[]
                for c in (-1,1):
                    for ch,channel in enumerate(CHANNELS):
                        _,fit=spline_fit(stages['V5'][cues==c,ch].mean(axis=0))
                        local_fits.append(dict(cue=c,channel=channel,n_trials=int((cues==c).sum()),**fit))
                save_csv(local_fits,dest/'分方向样条拟合参数.csv')
                fits.extend([dict(dataset=ds,**z) for z in local_fits])
            print(f'{name}：{len(audit)}事件，{n}可用；V5各折策略 '+', '.join(f'{r["V5_base"]}+{r["trend_strength"]}' for r in strategies[-5:]),flush=True)
    save_csv(summary,summary_dir/'四阶段共同配对指标汇总.csv')
    save_csv(spatial,summary_dir/'左右刺激与额区空间差异指标.csv')
    save_csv(strategies,summary_dir/'逐折V5策略与参考数量.csv')
    save_csv(slope_diagnostics,summary_dir/'刺激前斜率与刺激后慢变化.csv')
    save_csv(synthetic,synth_dir/'多场景逐折配对验证.csv')
    bench=pd.DataFrame(synthetic)
    save_csv(bench.groupby(['dataset','level','stage'],sort=False)[['RMSE','normalized_RMSE','contrast_RMSE','positive_mean_error','latency_error_ms']].mean().reset_index(),
             synth_dir/'按数据组和幅度汇总.csv')
    if fits:save_csv(fits,summary_dir/'分方向样条拟合参数汇总.csv')
    if not args.no_plots:
        plot_cross_task(out,datasets)
        plot_benchmark(out,synthetic)
    write_report(out,datasets,summary,spatial,synthetic,fits)
    manifest['csv_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*.csv'))}
    (summary_dir/'运行清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('完成：'+str(out),flush=True)


if __name__=='__main__':main()
