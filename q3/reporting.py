"""Source-backed Chinese report and standalone, reproducible scientific figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .data import TIME,FS,KEYS,SEED
from .model import Candidate,visual_response,slow_states

COLORS={'observed':'#242424','M0':'#3565A0','M1':'#B48A32','M2':'#D96532','M3':'#737C45',
        'ordinary':'#3565A0','cognitive':'#D96532','median':'#737C45','error':'#AD5882','timeout':'#9A9A9A'}
CH=('Fz','F3','F4')
STAGE={'encoding':'提示编码期','maintenance':'保持准备期','retrieval':'目标后整合期'}


def configure():
    names={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Noto Sans CJK SC','Microsoft YaHei','SimHei','Noto Sans CJK JP'] if f in names),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#242424',
        'text.color':'#242424','axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white',
        'axes.prop_cycle':plt.cycler(color=list(COLORS.values()))})


def save(fig,out,name,caption):
    engine=fig.get_layout_engine()
    if engine is not None:
        left,bottom,width,_=engine.get()['rect']
        # Constrained-layout rect is (left,bottom,width,height), not (l,b,r,t).
        engine.set(rect=(left,bottom,width,.91-bottom),h_pad=.07,w_pad=.04)
        if fig._suptitle is not None:fig._suptitle.set_in_layout(False)
    note=fig.text(.02,.012,caption,fontsize=9,va='bottom',color='#555555')
    extra=([fig._suptitle] if fig._suptitle is not None else [])+list(fig.legends)+[note]
    fig.savefig(out/f'{name}.png',dpi=150,bbox_inches='tight',bbox_extra_artists=extra)
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight',bbox_extra_artists=extra)
    plt.close(fig)


def bootstrap_wave(values,blocks,seed=SEED):
    finite=np.isfinite(values);sums=[];counts=[]
    for b in np.unique(blocks):
        ii=blocks==b;sums.append(np.nansum(values[ii],axis=0));counts.append(finite[ii].sum(0))
    sums=np.array(sums);counts=np.array(counts);shape=sums.shape[1:]
    avg=sums.sum(0)/np.maximum(counts.sum(0),1)
    rng=np.random.default_rng(seed);draws=rng.multinomial(len(sums),np.ones(len(sums))/len(sums),1000)
    bs=(draws@sums.reshape(len(sums),-1))/np.maximum(draws@counts.reshape(len(sums),-1),1)
    lo,hi=np.quantile(bs,[.025,.975],axis=0).reshape(2,*shape)
    count=finite.sum(0);avg[count<10]=np.nan;lo[count<10]=np.nan;hi[count<10]=np.nan
    return avg,lo,hi


def figure_events(out,figs):
    df=pd.read_csv(out/'event_audit.csv');fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,key in zip(axes.flat,KEYS):
        d=df[df.dataset==key];ids=d.trial_id.to_numpy();target=d.target_s.to_numpy();response=d.response_s.to_numpy()
        ax.hlines(ids,0,response-.1,color='#E0E0E0',lw=.7)
        ax.scatter(np.zeros(len(d)),ids,s=8,color=COLORS['M0'],label='提示')
        ax.scatter(target,ids,s=10,color=COLORS['M1'],label='候选目标出现')
        ax.scatter(response,ids,s=14,facecolors='none' if key.endswith('1') else COLORS['M2'],
                   edgecolors=COLORS['M2'],label='候选应答' if key.endswith('1') else '独立点击')
        ax.set(title=f'{key} · 原始 100 次 / 主分析保留 {int(d.retained.sum())} 次',xlabel='相对提示时间（秒）',ylabel='原始试次编号',xlim=(-.15,4.5))
        ax.legend(fontsize=8,loc='upper left');ax.grid(axis='x',alpha=.15)
    fig.suptitle('第三问事件审计：目标标记与应答标记分开处理',fontsize=16)
    fig.get_layout_engine().set(rect=(0,.06,1,.94))
    save(fig,figs,'01_事件时间轴','来源：四份 MAT，共 400 次提示。空心点为 Task-1 平台终点代理；正确性和超时状态均未确证。')


def figure_model(figs):
    fig=plt.figure(figsize=(12,8));gs=fig.add_gridspec(3,3,height_ratios=[.9,1,1],hspace=.7,wspace=.3)
    ax=fig.add_subplot(gs[0,:]);ax.axis('off')
    labels=['提示形状 / 共同目标输入','LGN → E/I 视觉响应 v','记忆保持 m ↔ 认知整合 p','三通道有效混合观测']
    for j,label in enumerate(labels):
        x=.02+j*.25
        ax.text(x+.105,.6,label,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=.65',fc='#F3F4F4',ec='#777777'),transform=ax.transAxes)
        if j<3:ax.annotate('',xy=(x+.245,.6),xytext=(x+.218,.6),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.05,'记忆状态具有功能解释；有效混合矩阵不等于个体解剖导联',ha='center',transform=ax.transAxes,fontsize=10)
    target=568/FS;c=Candidate('M2',2.,.3,.05,.1,1.)
    for row,cue in enumerate((-1,1)):
        v=visual_response(cue,52,568);m,p=slow_states(v,[c],target)
        for col,(name,z) in enumerate([('视觉响应 v',v),('记忆状态 m',m[0]),('认知状态 p',p[0])]):
            a=fig.add_subplot(gs[row+1,col])
            for j,(label,style,color) in enumerate(zip(('左形状偏好','右形状偏好','共同'),('-','--',':'),(COLORS['M0'],COLORS['M2'],'#777777'))):
                a.plot(TIME,z[:,j],ls=style,color=color,label=label,lw=1.6)
            a.axvline(target,color='#999999',lw=.8,ls=':');a.set(xlim=(-.1,4.2),title=f'{"左" if cue==-1 else "右"}向提示 · {name}',xlabel='相对提示时间（秒）',ylabel='模型内部任意单位')
            if row==0 and col==0:a.legend(fontsize=8)
    fig.suptitle('宏观认知模型及其内部活动示例',fontsize=16);fig.subplots_adjust(bottom=.12,top=.9)
    save(fig,figs,'02_宏观模型与内部活动','固定示例参数 τm=2 s、τp=0.3 s、δ=50 ms、κ=0.1、β=1。全部曲线为仿真，非直接测得的海马活动。')


def figure_eeg(out,figs):
    fig,axes=plt.subplots(4,3,figsize=(13,12),sharex=True,layout='constrained')
    contrast,axs=plt.subplots(4,3,figsize=(13,12),sharex=True,layout='constrained')
    for row,key in enumerate(KEYS):
        with np.load(out/f'{key}_waves.npz') as z:
            x=z['x'];blocks=z['blocks'];cues=z['cues'];time=z['time'];valid=np.isfinite(x)
            avg,lo,hi=bootstrap_wave(x,blocks)
            means={m:np.nanmean(np.where(valid,z[m],np.nan),axis=0) for m in ('M0','M2','M3')}
            for col in range(3):
                ax=axes[row,col];ax.fill_between(time,lo[:,col],hi[:,col],color='#DADADA',alpha=.65)
                ax.plot(time,avg[:,col],color=COLORS['observed'],label='实测',lw=1.4)
                for m,style in [('M0','--'),('M2','-'),('M3',':')]:
                    vv=means[m][:,col].copy();vv[valid[:,:,col].sum(0)<10]=np.nan
                    ax.plot(time,vv,color=COLORS[m],ls=style,label=m,lw=1.3)
                ax.axvline(.8,color='#BBBBBB',ls=':',lw=.7);ax.axvline(np.median(z['target']),color='#BBBBBB',ls=':',lw=.7)
                ax.set(title=f'{key} · {CH[col]} · n={len(x)}',xlim=(-.1,4.2),ylabel='原始电位单位');ax.grid(alpha=.12)
                if row==0 and col==0:ax.legend(ncol=4,fontsize=8)
                if row==3:ax.set_xlabel('相对提示时间（秒）')
            l=x[cues==-1];r=x[cues==1];al,ll,hl=bootstrap_wave(l,blocks[cues==-1]);ar,lr,hr=bootstrap_wave(r,blocks[cues==1])
            # Difference bands from block-resampled differences (paired recording blocks).
            rng=np.random.default_rng(SEED);diffs=[]
            for _ in range(500):
                idx=np.concatenate([np.flatnonzero(blocks==b) for b in rng.choice(np.unique(blocks),5)])
                a=x[idx];label=cues[idx]
                with np.errstate(invalid='ignore'):diffs.append(np.nanmean(a[label==1],axis=0)-np.nanmean(a[label==-1],axis=0))
            lower,upper=np.nanquantile(diffs,[.025,.975],axis=0)
            predicted=np.where(valid,z['M2'],np.nan)
            pdiff=np.nanmean(predicted[cues==1],axis=0)-np.nanmean(predicted[cues==-1],axis=0)
            good=(np.isfinite(l).sum(0)>=10)&(np.isfinite(r).sum(0)>=10)
            for col in range(3):
                ax=axs[row,col];observed=(ar-al)[:,col];observed[~good[:,col]]=np.nan
                pc=pdiff[:,col].copy();pc[~good[:,col]]=np.nan
                lower[~good]=np.nan;upper[~good]=np.nan
                ax.fill_between(time,lower[:,col],upper[:,col],color='#DADADA',alpha=.65)
                ax.plot(time,observed,color=COLORS['observed'],label='实测右减左');ax.plot(time,pc,color=COLORS['M2'],label='M2 留出预测')
                ax.axhline(0,color='#777777',ls=':',lw=.8);ax.set(title=f'{key} · {CH[col]}',xlim=(-.1,4.2),ylabel='右减左（原始单位）')
                if row==0 and col==0:ax.legend(fontsize=8)
                if row==3:ax.set_xlabel('相对提示时间（秒）')
    fig.suptitle('条件 EEG 的外层留出预测',fontsize=16);fig.get_layout_engine().set(rect=(0,.06,1,.95))
    save(fig,figs,'03_三阶段EEG留出预测','灰带：连续块重采样逐点 95% 区间。每点仅平均尚在认知窗中的试次，n<10 不绘制；尾部存在反应时选择效应。')
    contrast.suptitle('方向差分检验：实测右减左与 M2 留出预测',fontsize=16);contrast.get_layout_engine().set(rect=(0,.06,1,.95))
    save(contrast,figs,'04_左右差分留出预测','灰带：按连续块成对重采样的逐点 95% 区间，非同时置信带。任一方向每点 n<10 不绘制。')


def figure_comparison(out,figs):
    df=pd.read_csv(out/'cognitive_intervals.csv');fig,axes=plt.subplots(1,3,figsize=(14,6),layout='constrained')
    for ax,stage in zip(axes,STAGE):
        for offset,control in zip((-.2,0,.2),('M0','M1','M3')):
            d=df[(df.stage==stage)&(df.control==control)].set_index('dataset').loc[list(KEYS)]
            ax.errorbar(d.S_cog,np.arange(4)+offset,xerr=np.maximum(0,np.vstack([d.S_cog-d.ci_low,d.ci_high-d.S_cog])),
                        fmt='o',capsize=3,color=COLORS[control],label=f'M2 对 {control}')
        ax.axvline(0,color='#333333',ls='--',lw=1);ax.set(yticks=np.arange(4),yticklabels=KEYS,title=STAGE[stage],xlabel='S_cog：正值表示 M2 误差更低')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    axes[0].legend(fontsize=9);fig.suptitle('新增记忆环节是否改善真实 EEG 预测',fontsize=16)
    fig.get_layout_engine().set(rect=(0,.1,1,.92))
    nboot=int(df.n_bootstraps.iloc[0])
    save(fig,figs,'05_认知增量与对照',f'每组五个连续时间块等权；误差线为 {nboot:,} 次块重采样 95% 区间。区间宽度反映当前记录内不确定性。')


def figure_behavior(out,figs):
    df=pd.read_csv(out/'behavior_predictions.csv');ci=pd.read_csv(out/'behavior_intervals.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    for ax,key in zip(axes[0],('A2','B2')):
        d=df[df.dataset==key]
        ax.scatter(d.rt_s,d.ordinary,s=18,alpha=.5,color=COLORS['ordinary'],marker='x',label='普通 EEG 前缀')
        ax.scatter(d.rt_s,d.cognitive,s=22,alpha=.65,color=COLORS['cognitive'],label='认知状态前缀')
        lo=min(d[['rt_s','ordinary','cognitive']].min())-.03;hi=max(d[['rt_s','ordinary','cognitive']].max())+.03
        ax.plot([lo,hi],[lo,hi],ls=':',color='#555555');ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='实测目标至点击间隔（秒）',ylabel='外层留出预测（秒）',title=f'{key} · 固定前缀 · n={len(d)}')
        ax.legend(fontsize=8)
    for ax,control in zip(axes[1],('median','ordinary')):
        d=ci[ci.contrast=='cognitive_vs_'+control].set_index('dataset').loc[['A2','B2','pooled']]
        ax.errorbar(d.MAE_gain*1000,np.arange(3),xerr=np.maximum(0,np.vstack([d.MAE_gain-d.ci_low,d.ci_high-d.MAE_gain]))*1000,
                    fmt='o',capsize=4,color=COLORS['cognitive'])
        ax.axvline(0,color='#555555',ls=':');ax.set(yticks=np.arange(3),yticklabels=['A2','B2','合并'],xlabel='MAE 改善（毫秒；正值更好）',title='认知模型对'+('训练中位数' if control=='median' else '普通 EEG 前缀'))
        ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    fig.suptitle('反应时验证：预测只使用目标后 600 ms 以内的 EEG',fontsize=15)
    fig.get_layout_engine().set(rect=(0,.08,1,.94))
    save(fig,figs,'06_反应时预测与增量','行为质量筛选和特征均使用固定前缀；嵌套拟合排除测试块。下图为记录内分层块重采样 95% 区间。')


def figure_decision(out,figs):
    df=pd.read_csv(out/'decision_summary.csv');fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    names=['常规证据','记忆衰减','噪声增加','低证据'];left=np.zeros(4)
    for field,label,color in [('correct_rate','正确',COLORS['M0']),('error_rate','错误',COLORS['error']),('timeout_rate','截止前未答',COLORS['timeout'])]:
        axes[0].barh(names,df[field],left=left,label=label,color=color);left+=df[field].to_numpy()
    axes[0].set(xlim=(0,1),xlabel='模拟试次比例',title='给定真实目标映射的模拟结果')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.06,.085),ncol=3,fontsize=9)
    with np.load(out/'decision_paths.npz') as z:
        time=z['nominal_time'];path=z['nominal_trajectories'];choices=z['nominal_choice']
        colors={1:COLORS['M0'],-1:COLORS['error'],0:COLORS['timeout']}
        for p,c in zip(path,choices):axes[1].plot(time,p,color=colors[int(c)],lw=1,alpha=.65)
    for b in (-1,1):axes[1].axhline(b,color='#555555',ls='--')
    axes[1].set(xlabel='目标出现后的决策时间（秒）',ylabel='积累证据 z',title='常规场景的代表性轨迹',ylim=(-1.35,1.35))
    fig.suptitle('错误与超时的机制仿真',fontsize=16);fig.get_layout_engine().set(rect=(0,.2,1,.7))
    save(fig,figs,'07_正确错误与超时仿真',f'每场景 {int(df.n.iloc[0]):,} 次模拟；边界 ±1，截止 2 秒。全部比例是仿真值，不能替代缺失的实测正确率或超时率。')


def figure_recovery(out,figs):
    rows=json.loads((out/'recovery.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    levels=[r['noise_level'] for r in rows]
    axes[0].bar([str(v) for v in levels],[r['NRMSE'] for r in rows],color=COLORS['M2'])
    axes[0].set(xlabel='真实残差背景 / 已知信号 RMS',ylabel='留出信号恢复 NRMSE',title='已知生成信号的恢复误差')
    axes[1].bar([str(v) for v in levels],[r['near_equivalent_candidates'] for r in rows],color=COLORS['M0'])
    axes[1].set(xlabel='真实残差背景 / 已知信号 RMS',ylabel='近似等价候选数量',title='不同参数可能产生近似观测')
    fig.suptitle('半合成验证：波形恢复与参数恢复分别检查',fontsize=15)
    fig.get_layout_engine().set(rect=(0,.12,1,.92))
    save(fig,figs,'08_半合成恢复与可辨识性','背景来自现有 EEG 残差；不视为无噪声脑源。近似等价阈值：最小误差 + max(其 1%, 1e-5)。')


def figure_sensitivity(out,figs):
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    main=pd.read_csv(out/'trial_metrics.csv');values=[]
    for j,key in enumerate(KEYS):
        other=pd.read_csv(out/f'{key}_endpoint_sensitivity.csv')
        for gap in (.05,.1,.15,.2):
            d=main[main.dataset==key] if gap==.1 else other[other.gap_s==gap]
            # Only late period matters for the endpoint test; equal block weighting.
            m=d[d.stage=='retrieval'].groupby(['block','model']).MSE.mean().unstack()
            value=1-m.M2.mean()/m.M3.mean();values.append(dict(dataset=key,gap=gap,S=value))
        vv=[r for r in values if r['dataset']==key]
        axes[0].plot([r['gap']*1000 for r in vv],[r['S'] for r in vv],marker=['o','s','^','D'][j],color=list(COLORS.values())[j+1],label=key)
    axes[0].axhline(0,color='#555555',ls=':');axes[0].set(xlabel='应答前截断距离（毫秒）',ylabel='目标后 S_cog：M2 对 M3',title='冻结主模型后的截断敏感性');axes[0].legend(fontsize=8,ncol=2)
    for j,key in enumerate(KEYS):
        vals=[]
        for name,file in [('因果',out/'cognitive_intervals.csv'),('零相位',out/'sensitivity'/f'{key}_zero_phase_intervals.csv')]:
            d=pd.read_csv(file);d=d[(d.dataset==key)&(d.stage=='retrieval')&(d.control=='M3')]
            vals.append(float(d.S_cog.iloc[0]))
        axes[1].plot([0,1],vals,marker=['o','s','^','D'][j],color=list(COLORS.values())[j+1],label=key)
    axes[1].axhline(0,color='#555555',ls=':');axes[1].set(xticks=[0,1],xticklabels=['因果主分析','零相位离线对照'],ylabel='目标后 S_cog：M2 对 M3',title='滤波方式敏感性（分别重拟合）')
    fig.suptitle('关键处理假设的敏感性检验',fontsize=15);fig.get_layout_engine().set(rect=(0,.12,1,.92))
    save(fig,figs,'09_截断与滤波敏感性','端点敏感性按与主分析共有的有效试次评分，试次集合仍可能随截断变化；零相位结果不用于实时或无泄漏主结论。')


def md_table(df,formats=None):
    formats=formats or {};columns=list(df.columns)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in df.iterrows():
        cells=[]
        for c in columns:
            v=r[c];cells.append(formats[c].format(v) if c in formats and pd.notna(v) else str(v))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def write_report(out):
    audit=pd.read_csv(out/'event_audit.csv');summary=pd.read_csv(out/'eeg_summary.csv');ci=pd.read_csv(out/'cognitive_intervals.csv')
    beh=pd.read_csv(out/'behavior_metrics.csv');perm=pd.read_csv(out/'behavior_permutation.csv');bci=pd.read_csv(out/'behavior_intervals.csv')
    decision=pd.read_csv(out/'decision_summary.csv');recovery=json.loads((out/'recovery.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    late=ci[(ci.stage=='retrieval')&(ci.control=='M3')]
    positive=int((late.S_cog>0).sum());strong=int((late.ci_low>0).sum())
    pooled=beh[(beh.dataset=='pooled')&(beh.model=='cognitive')].iloc[0]
    sig=perm[(perm.dataset=='pooled')&(perm.contrast=='cognitive_vs_ordinary')].iloc[0]
    gain=bci[(bci.dataset=='pooled')&(bci.contrast=='cognitive_vs_ordinary')].iloc[0]
    supported=strong==4 and gain.ci_low>0 and sig.p_Holm_6<.05
    conclusion=('完整模型满足本轮预设的内部增量证据标准，但仍不是脑源定位或独立受试者确认。' if supported else
                '宏观模型的构建和验证流程已完成；本轮结果未满足“跨四组一致的记忆机制增量且行为增量成立”的实测支持标准。不能写成已经发现或恢复海马认知信号。')
    counts=audit.groupby('dataset').agg(提示次数=('trial_id','size'),主分析保留=('retained','sum'),候选反应时中位数秒=('rt_s','median')).reset_index()
    table=ci[ci.control.isin(['M0','M1','M3'])][['dataset','stage','control','S_cog','ci_low','ci_high']].copy();table.stage=table.stage.map(STAGE)
    absolute=summary[summary.model=='M2'][['dataset','stage','n_trials','RMSE','NRMSE','S_train_baseline']].copy()
    absolute.stage=absolute.stage.map(STAGE)
    contrasts=pd.concat([pd.read_csv(out/f'{key}_contrasts.csv') for key in KEYS],ignore_index=True)
    contrast_table=contrasts[contrasts.model=='M2'].groupby(['dataset','stage'])[['MSE','zero_MSE']].mean().reset_index()
    contrast_table['S_delta']=1-contrast_table.MSE/contrast_table.zero_MSE;contrast_table.stage=contrast_table.stage.map(STAGE)
    behavior_table=beh[['dataset','model','n','MAE','R2','S_vs_train_median','spearman']]
    selected=[]
    for key in KEYS:
        rows=json.loads((out/f'{key}_choices.json').read_text())
        for r in rows:
            if r['model']=='M2':
                p=r['parameters'];boundary=[name for name,ends in [('tau_m',(.5,4.)),('tau_p',(.1,.6)),('delay',(0.,.1)),('feedback',(0.,.3))] if p[name] in ends]
                if key.endswith('2') and p['beta'] in (.5,2.):boundary.append('beta')
                selected.append(dict(dataset=key,block=r['held_block'],**p,lambda_ridge=r['selection']['lambda_ridge'],edf=r['meta']['edf'],
                    min_singular=min(r['meta']['design_singular_values']),irls_last_change=r['meta']['irls_changes'][-1],
                    grid_boundary_parameters=','.join(boundary)))
    save_csv=pd.DataFrame(selected).to_csv
    save_csv(out/'selected_parameters.csv',index=False,encoding='utf-8-sig')
    boundary_count=sum(bool(r['grid_boundary_parameters']) for r in selected)
    low_rank=sum(r['min_singular']<1e-3 for r in selected)
    not_converged=sum(r['irls_last_change']>=1e-6 for r in selected)
    sensitivity_rows=[]
    for key in KEYS:
        file=out/'sensitivity'/f'{key}_zero_phase_intervals.csv'
        d=pd.read_csv(file);r=d[(d.stage=='retrieval')&(d.control=='M3')].iloc[0]
        sensitivity_rows.append(dict(dataset=key,analysis='零相位重拟合',stage='目标后',S_cog=r.S_cog,ci_low=r.ci_low,ci_high=r.ci_high))
        if key.endswith('1'):
            d=pd.read_csv(out/'sensitivity'/f'{key}_pretarget_intervals.csv')
            r=d[(d.stage=='maintenance')&(d.control=='M3')].iloc[0]
            sensitivity_rows.append(dict(dataset=key,analysis='仅目标前重拟合',stage='保持期',S_cog=r.S_cog,ci_low=r.ci_low,ci_high=r.ci_high))
    text=f'''# 第三问 视觉输入与记忆认知整合的宏观动力学模型

## 结论与题目对应

{conclusion}

目标后阶段，M2 相对一般慢趋势 M3 的留出误差点估计在四组中的 **{positive} 组**较低，其中块重采样区间完全高于零的有 **{strong} 组**。项目二合并认知前缀反应时预测 MAE 为 **{pooled.MAE:.4f} 秒**，相对普通 EEG 前缀的 MAE 改善为 **{gain.MAE_gain*1000:.2f} ms**（95% 区间 **{gain.ci_low*1000:.2f}～{gain.ci_high*1000:.2f} ms**），六项 Holm 校正置换 p={sig.p_Holm_6:.4f}。

题目要求的长认知窗、Q2 视觉形成路径、多来源记忆—认知网络、真实 EEG 验证以及错误／未答机制均已实现。正确性和截止时间不可由当前 MAT 确认，所以错误和超时仅在有已知目标映射的仿真中验证，不虚构实测标签。

## 数据与事件审计

输入为原题四份 MAT、256 Hz 原始 Fz/F3/F4，幅度使用原始记录单位。每份 100 次提示，总计 400 次；未知正确性和超时字段留空并记录原因。

{md_table(counts,{'候选反应时中位数秒':'{:.4f}'})}

Task-2 的 ±2 是独立点击，恰与 ±1 平台终点同采样点；平台起点作为候选目标出现。Task-1 没有独立点击，平台终点只作应答代理。B1 第 49、87 次提示与平台符号不一致，不直接判为答错。点击左右位置不等于提示三角朝向。无日志不能确认目标真值、超时规则或所有标记语义。

主窗为提示前 250 ms 至应答前 100 ms；编码期 0–0.8 s，保持期 0.8 s 至候选目标，整合期从候选目标至主窗终点。不对试次做时间拉伸。五个连续块各自处理，块两端保守剔除 24 s 缓冲；主分析用 60 Hz/Q30 陷波与四阶 0.1–30 Hz **单向因果**滤波、提示前中位数基线。模型观测施加相同因果滤波。

长窗质量控制沿用 Q1 的幅度和跳变工程阈值；原始饱和比例阈值为 15/(3×269)，滤后峰峰值大于 1800 或单点跳变大于 600 则剔除。饱和按比例处理，避免简单把长窗总点数与短窗 15 点混同。工程阈值不代表硬件饱和真值。行为分析另按固定前缀作独立质量筛查，不依赖后续波形。

![事件时间轴](figures/01_事件时间轴.png)

## 模型方程与参数解释

Q2 形状编码从题图提取左／右镜像特征，三组代表左形状偏好、右形状偏好和共同输入，不代表半球。视觉层复用 Q2 的 LGN、三级 E/I 及双级突触动力学，取最后视觉层三个等效响应 v。局部时间尺度倍率 1、递归耦合 0.1 固定；目标出现时加入 200 ms 共同输入，持续时间是工程假设，不使用最终点击符号或反应时决定输入。

记忆与认知的等效活动偏差为：

$$\\tau_m\\dot m_j=-m_j+v_j+\\kappa p_j(t-\\delta),\\qquad
\\tau_p\\dot p_j=-p_j+v_j+\\beta_kG_T(t)m_j(t-\\delta).$$

$$\\widehat{{x}}(t)=L_vv(t)+L_mm(t)+L_pp(t).$$

初始慢变量为零；G_T 在候选目标出现后开启。主网格 τm∈{{0.5,1,2,4}} s，τp∈{{0.1,0.3,0.6}} s，δ∈{{0,0.05,0.1}} s，κ∈{{0,0.1,0.3}}；β1=1，β2∈{{0.5,1,2}}。满足 κβ<1 的小增益充分条件，零输入延迟反馈稳定。慢状态用 256 Hz 指数 Euler 加分数延迟插值，测试比较 512 Hz 解。

混合矩阵在同一记录的左右条件间共享。每个训练集内按源设计 RMS 固定尺度，按训练 EEG 均方根归一化通道；损失按试次与非空阶段等权，Huber 阈值 1.5，岭惩罚候选 {{0.001,0.01,0.1,1,10}}。所有候选先作训练内嵌套岭筛查，前两组候选／惩罚组合再用 Huber IRLS 重新做内层留块选型及最终拟合（最多 7 次迭代，变化小于 10⁻⁶ 结束）。这一筛查是固定计算规则，不以外层成绩调整。

20 个 M2 外折中，至少一个动态参数位于候选网格边界的有 {boundary_count} 折，标准化设计最小奇异值低于 10⁻³ 的有 {low_rank} 折；IRLS 在 7 次上限处仍未低于 10⁻⁶ 变化阈值的有 {not_converged} 折（实际变化记录在参数表）。有效自由度采用同一正则化的初始二次拟合迹，作为复杂度诊断，不声称是稳健估计器的精确自由度。边界选型和近共线不能写成精确生理参数测量。

主模型只输入提示方向、提示时长、任务和候选目标出现时刻，在固定 5 s 模拟时域生成整条波形；应答时刻只决定评分掩码。每个外层留出块之外再做内层选型；参数、混合矩阵、源缩放、通道尺度都不读取测试 EEG。参数选择与实际有效自由度见 `selected_parameters.csv`，逐折训练试次和完整系数见 `*_choices.json`。

M0 只有视觉项；M1 增加普通认知低通但无记忆；M2 使用完整记忆网络；M3 使用六列共同／提示符号调制的平滑趋势，并作同样正则化选型。另有 κ=0 和 Lm=0 的完整重拟合消融。由于三个额区传感器、多来源高度相关，记忆增益、混合系数及时间常数不能默认被唯一辨识。

![模型内部活动](figures/02_宏观模型与内部活动.png)

## EEG 留出验证

所有方法比较同一记录内同一试次和同一时间掩码。`eeg_summary.csv` 的 MSE 对试次、通道等权；RMSE 为其平方根。NRMSE 使用各折训练通道尺度。S_train_baseline=1−MSE/MSE_baseline，基线是训练试次对应阶段的平均电位，不是使用测试均值计算的拟合 R²。方向差分 S_delta 另以零方向差为参照。

{md_table(absolute,{'RMSE':'{:.3f}','NRMSE':'{:.3f}','S_train_baseline':'{:.4f}'})}

下面 S_cog=1−MSE_M2/MSE_control 先在每个块内平均通道与试次，再对五个块等权。区间按五个连续块重采样，不能解释为独立受试者总体置信区间；与逐试次等权总表的数值可能略有不同。

{md_table(table,{'S_cog':'{:.4f}','ci_low':'{:.4f}','ci_high':'{:.4f}'})}

![EEG 留出预测](figures/03_三阶段EEG留出预测.png)

![方向差分](figures/04_左右差分留出预测.png)

M2 方向差分的五折等权结果如下。S_delta>0 才优于预测零方向差；该分数不能解释为未知方向分类准确率。

{md_table(contrast_table[['dataset','stage','S_delta']],{'S_delta':'{:.4f}'})}

![模型增量](figures/05_认知增量与对照.png)

波形图尾部只包含尚未达到认知截点的较慢试次，因此存在反应时选择效应；不能把均值尾部的改变直接解释为所有试次的认知演化。阴影为逐点区间，非同时置信带。

## 固定前缀反应时验证

Task-2 只读取候选目标后 600 ms 以内的 EEG。按训练模型冻结的混合矩阵，把前缀观测投影为带单位先验的九个正则化源幅度，提取记忆状态、认知状态及其约 50 ms 变化率，共九维。普通 EEG 基线为编码、保持和目标后前缀三个窗口 × 三通道的九维均值。

行为内层每次也重新拟合仅使用其训练块的 EEG 参数和混合矩阵；并非先在完整外层训练集上提特征再让内层验证混入模型。行为岭回归预测 log(RT)，内层用 log(RT) MSE 选惩罚，报告秒尺度预测指标；预测秒值是对数预测的指数，不宣称条件均值无偏。对照为外层训练 RT 中位数及普通 EEG 前缀。目标至点击间隔包括未建模的运动等非决策耗时，因此不等同于纯认知持续时间。

{md_table(behavior_table,{'MAE':'{:.4f}','R2':'{:.4f}','S_vs_train_median':'{:.4f}','spearman':'{:.4f}'})}

行为 R²=1−Σ误差²/Σ(RT−全体该组均值)² 是汇总描述分数；S_vs_train_median 另给训练基线参照分数。记录内、块内置换 RT 共 {manifest['permutations']} 次，每次重新拟合监督回归并重新内层选型；与 RT 无关的严格嵌套 EEG 特征变换可缓存复用。A2、B2、合并 × 两个对照共六项 p 值作 Holm 校正。

{md_table(perm[['dataset','contrast','MAE_gain','p_raw','p_Holm_6']],{'MAE_gain':'{:.5f}','p_raw':'{:.4f}','p_Holm_6':'{:.4f}'})}

![行为预测](figures/06_反应时预测与增量.png)

## 稳健性、可辨识性与错误应答仿真

应答前 50、150、200 ms 的敏感性使用冻结的主分析外层模型，在与各新窗口共有的合格试次上重新评分，不借结果重选参数。零相位对照对所有候选重新拟合，仅作离线敏感性；Task-1 另用目标前窗口重拟合 M0–M3，避免结论完全依赖候选应答终点。详细表和逐折信息见 `sensitivity/`。

{md_table(pd.DataFrame(sensitivity_rows),{'S_cog':'{:.4f}','ci_low':'{:.4f}','ci_high':'{:.4f}'})}

![处理敏感性](figures/09_截断与滤波敏感性.png)

半合成实验用已知 M2 参数与混合矩阵生成信号，分别加入 0、0.5、1 倍信号 RMS 的真实 EEG 残差背景，再按训练块选型、留出块检查干净生成信号恢复。真实残差不等于干净真值；已知的是人工生成部分。三种噪声下精确动态参数恢复分别为 {[r['exact_dynamic_recovery'] for r in recovery]}，选中参数与近似等价候选见 `recovery.json` 与 `recovery_profiles.csv`。恢复波形不等于恢复唯一脑源参数。

![半合成恢复](figures/08_半合成恢复与可辨识性.png)

错误／未答机制采用 dz=γwᵀp dt+σdW，边界 ±1、截止 2 s，首次越过正边界为正确，负边界为错误，未越界为未答。只有仿真给定真实映射，所有场景参数预先指定且未拟合实测错误率。此处输出是决策时间，不添加未估计的运动时延。

{md_table(decision[['scenario','n','correct_rate','error_rate','timeout_rate','median_decision_time_s']],{'correct_rate':'{:.3f}','error_rate':'{:.3f}','timeout_rate':'{:.3f}','median_decision_time_s':'{:.3f}'})}

![决策机制仿真](figures/07_正确错误与超时仿真.png)

## 可提交结论与证据边界

{conclusion}

本工作给出一套能连接视觉输入、记忆保持／提取、额区整合与头皮电位的可运行宏观候选模型，并对其真实数据预测和行为关联进行了分层验证。参数具有功能性解释，LGN、海马和额区标签不构成实测源定位；无 EOG 和真实头模型不能排除眼动或其他慢活动。四份记录已用于多轮开发，所有 p 值与区间均为该批数据中的内部探索证据。无临床诊断或跨人群推广结论。

## 复现与参考

运行：`OPENBLAS_NUM_THREADS=1 .venv/bin/python -B q3/main.py --output q3_result`。分阶段命令、独立复核、字段解释见 `../q3/README.md`。输入、源码、运行配置和依赖版本见 `manifest.json`；逐试次误差、波形和行为预测均可独立重算。

- Wilson HR, Cowan JD. 1972. [Excitatory and inhibitory interactions in localized populations of model neurons](https://pubmed.ncbi.nlm.nih.gov/4332108/).
- Daume J et al. 2024. [Control of working memory by phase–amplitude coupling of human hippocampal neurons](https://www.nature.com/articles/s41586-024-07309-z). 本工作借用记忆与控制交互的动机，不声称从三通道 30 Hz 以下 EEG 验证该文的 theta–gamma 耦合。
- Ratcliff R, McKoon G. 2008. [The diffusion decision model](https://pmc.ncbi.nlm.nih.gov/articles/PMC2474742/).
'''
    if manifest['quick']:text='> 本目录是快速试跑，不用于正式论文结论。\n\n'+text
    (out/'第三问_建模与验证报告.md').write_text(text,encoding='utf8')
    (out/'README.md').write_text('''# 第三问结果索引

先阅读 [建模与验证报告](第三问_建模与验证报告.md)。本目录包含模型构建结果和实测证据判定，二者不等价。

- `event_audit.csv`：400 次事件、标记可靠性、质量与排除原因。
- `*_waves.npz`、`*_choices.json`：逐试次实测和外层预测、训练来源及参数。
- `trial_metrics.csv`、`eeg_summary.csv`、`cognitive_intervals.csv`：EEG 误差及增量区间。
- `behavior_*`：固定前缀行为审计、特征、预测、指标、置换和区间。
- `sensitivity/`、`*_endpoint_sensitivity.csv`：重拟合或冻结评分的敏感性。
- `recovery.json`、`recovery_profiles.csv`：半合成参数恢复与非唯一性。
- `decision_*`：仅仿真的正确、错误和未答，不能当作实验标签。
- `figures/`：PNG 与矢量 PDF；各图问题及证据边界见 `图表说明.md`。
- `manifest.json`、`独立复核.json`：输入源码哈希、运行状态及独立指标核验。
''',encoding='utf8')
    chart_rows=[('01_事件时间轴','事件时刻是否被混同','400 个试次；明确独立点击与代理终点'),
      ('02_宏观模型与内部活动','模型是否形成视觉、保持与提取活动','固定参数仿真，非实测脑区活动'),
      ('03_三阶段EEG留出预测','模型能否预测完整认知波形','外层留出均值与逐点区间'),
      ('04_左右差分留出预测','方向差分能否转移到留出块','右减左；条件样本数不足处不绘制'),
      ('05_认知增量与对照','记忆环节是否优于基础与慢趋势','S_cog 与块重采样区间'),
      ('06_反应时预测与增量','认知状态是否含额外行为信息','固定前缀，独立质量筛选，嵌套留块'),
      ('07_正确错误与超时仿真','机制是否能够产生三种结局','模拟目标真值与截止时间；非实测错误率'),
      ('08_半合成恢复与可辨识性','波形与参数是否都能恢复','已知信号加真实残差背景'),
      ('09_截断与滤波敏感性','结论是否依赖预处理选择','冻结端点评分与独立重拟合分别说明')]
    note='# 图表说明\n\n统一使用白底、深灰文字；蓝色视觉/普通基线、橙色完整模型、橄榄色慢趋势，线型和点型提供非颜色区分。来源均为本目录保存的数值结果，输出采用 Matplotlib 静态 PNG 与 PDF。\n\n'
    for name,question,limit in chart_rows:note+=f'## {name}\n\n问题：{question}。解释范围：{limit}。\n\n![{name}](figures/{name}.png)\n\n'
    (out/'图表说明.md').write_text(note,encoding='utf8')


def render_report(out):
    import warnings
    configure();figs=Path(out)/'figures';figs.mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Mean of empty slice')
        warnings.filterwarnings('ignore',message='All-NaN slice encountered')
        figure_events(out,figs);figure_model(figs);figure_eeg(out,figs);figure_comparison(out,figs)
        figure_behavior(out,figs);figure_decision(out,figs);figure_recovery(out,figs);figure_sensitivity(out,figs)
    write_report(out)
