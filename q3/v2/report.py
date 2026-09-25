"""Chinese V2 report with explicit comparison settings and inspected figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from q3.data import KEYS,TIME,save_csv
from q3.reporting import configure,save as base_save,STAGE
from matplotlib.colors import LinearSegmentedColormap

BLUE='#3565A0';ORANGE='#D96532';GRAY='#777777'


def save(fig,out,name,caption):
    # Reserve a separate footer band; captions must not overlap x-axis labels.
    engine=fig.get_layout_engine()
    if engine is not None:engine.set(rect=(0.,.13,1.,.78))
    base_save(fig,out,name,caption)


def table(frame,decimals=4):
    columns=list(frame.columns)
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for values in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:.{decimals}f}' if isinstance(v,(float,np.floating)) else str(v) for v in values)+' |')
    return '\n'.join(rows)


def effects(ax,frame,contrasts,title):
    for j,(control,label,color,marker) in enumerate(contrasts):
        part=frame[(frame.control==control)&(frame.model=='N2')].set_index('dataset').reindex(KEYS)
        y=np.arange(4)+(j-(len(contrasts)-1)/2)*.18
        value=100*part.S.to_numpy();lo=100*part.ci_low.to_numpy();hi=100*part.ci_high.to_numpy()
        ax.errorbar(value,y,xerr=np.array([np.maximum(value-lo,0),np.maximum(hi-value,0)]),
            fmt=marker,color=color,capsize=3,label=label,ms=5)
    ax.axvline(0,c='#555555',ls=':',lw=1)
    ax.set(yticks=range(4),yticklabels=KEYS,title=title,xlabel='留出 MSE 降幅（%；正值更好）')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.legend(fontsize=8)


def render(out):
    configure();figs=out/'figures';figs.mkdir(exist_ok=True)
    d=pd.read_csv(out/'diagnostics.csv');iv=pd.read_csv(out/'intervals.csv')
    s=pd.read_csv(out/'summary.csv');ci=pd.read_csv(out/'context_intervals.csv')
    behavior=pd.read_csv(out/'behavior_metrics.csv');bi=pd.read_csv(out/'behavior_intervals.csv')
    bc=pd.read_csv(out/'behavior_v1_comparison.csv');recovery=pd.read_csv(out/'recovery.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    fig.suptitle('长认知窗的低频活动与通道关系')
    axes[0].bar(d.dataset,d.power_le_1Hz*100,color=BLUE)
    axes[0].set(ylim=(0,100),ylabel='1 Hz 以下功率比例（%）',title='提示后 0–3 秒，Welch 功率估计')
    for i,v in enumerate(d.power_le_1Hz):axes[0].text(i,100*v+1,f'{100*v:.1f}%',ha='center')
    corr=d[['corr_Fz_F3','corr_Fz_F4','corr_F3_F4']].to_numpy()
    im=axes[1].imshow(corr,vmin=-1,vmax=1,cmap=LinearSegmentedColormap.from_list('blue_gray_orange',[BLUE,'#F2F2F2',ORANGE]),aspect='auto')
    axes[1].set(xticks=range(3),xticklabels=['Fz–F3','Fz–F4','F3–F4'],yticks=range(4),yticklabels=KEYS,title='有效试次拼接后的通道相关')
    for i in range(4):
        for j in range(3):axes[1].text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr[i,j])>.6 else '#222222')
    fig.colorbar(im,ax=axes[1],label='相关系数')
    save(fig,figs,'01_慢变化诊断',f'诊断使用完整 0–3 秒窗的 {int(d.n_diagnostic_trials.sum())}/268 次试次。低频占比高不等于全部为伪影，不据此删除慢波。')

    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    fig.suptitle('仅输入刺激事件：新版与原版及认知对照的留出比较')
    for ax,stage in zip(axes,('encoding','maintenance','retrieval')):
        effects(ax,iv[(iv.setting=='stimulus')&(iv.stage==stage)],
            [('V1_M2','N2 对原版 M2',BLUE,'o'),('N1','N2 对无记忆 N1',GRAY,'s'),('N3','N2 对慢趋势 N3',ORANGE,'D')],STAGE[stage])
    save(fig,figs,'02_新旧模型与认知增量','同一 EEG、试次、有效时间与五个外层时间块；误差棒为 2000 次块重采样 95% 区间。内部探索验证。')

    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    fig.suptitle('提示前背景辅助：预测改善与记忆增量分别检验')
    effects(axes[0],ci[ci.stage=='retrieval'],[('stimulus_N2','背景辅助 N2 对事件输入 N2',BLUE,'o')],'背景信息的额外预测价值')
    effects(axes[1],iv[(iv.setting=='past')&(iv.stage=='retrieval')],
        [('B0','N2 对纯背景 B0',GRAY,'s'),('N1','N2 对同背景无记忆 N1',BLUE,'o'),('N3','N2 对同背景慢趋势 N3',ORANGE,'D')],'相同背景信息下的认知增量')
    save(fig,figs,'03_背景与认知分离','背景只读取提示之前 EEG；直接预测同一原始评分信号。背景带来的改善不能归因于海马或记忆来源。')

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fig.suptitle('Fz 汇总波形：观测与外层预测')
    for ax,key in zip(axes.flat,KEYS):
        a=np.load(out/f'{key}_stimulus_waves.npz');b=np.load(out/f'{key}_past_waves.npz')
        mask=np.isfinite(a['x'][:,:,0]);count=mask.sum(0)
        for yy,label,color,style in [(a['x'][:,:,0],'观测','#222222','-'),(a['V1_M2'][:,:,0],'原版 M2',GRAY,':'),
            (a['N2'][:,:,0],'事件输入 N2',BLUE,'--'),(b['N2'][:,:,0],'背景辅助 N2',ORANGE,'-.')]:
            avg=np.where(mask,yy,0).sum(0)/np.maximum(count,1);avg[count<10]=np.nan
            ax.plot(TIME,avg,label=label,c=color,ls=style,lw=1.3)
        ax.axvline(np.median(a['target']),c='#AAAAAA',ls=':',lw=1)
        ax.set(xlim=(-.25,4),xlabel='相对提示时间（秒）',ylabel='原始记录单位',title=f'{key} · n={len(a["ids"])}')
        if key=='A1':ax.legend(fontsize=8)
    save(fig,figs,'04_外层波形','Fz 汇总仅作可视化，定量评价使用全部三个通道。末尾只包含应答较慢的试次，少于 10 次处不绘制。')

    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    fig.suptitle('固定前缀反应时验证：新版的误差及增量')
    frame=behavior[behavior.dataset.isin(['A2','B2'])]
    for j,(model,label,color) in enumerate([('median','训练中位数',GRAY),('ordinary','普通 EEG 前缀',BLUE),('cognitive','新版认知前缀',ORANGE)]):
        a=frame[frame.model==model].set_index('dataset').reindex(['A2','B2'])
        axes[0].bar(np.arange(2)+(j-1)*.23,a.MAE*1000,width=.21,label=label,color=color)
    axes[0].set(xticks=[0,1],xticklabels=['A2','B2'],ylabel='MAE（毫秒）',title='外层留出误差');axes[0].legend(fontsize=8);axes[0].set_ylim(0,frame.MAE.max()*1400)
    for j,(contrast,label,color) in enumerate([('cognitive_vs_ordinary','对普通 EEG',ORANGE),('cognitive_vs_median','对训练中位数',GRAY)]):
        a=bi[bi.contrast==contrast].set_index('dataset').reindex(['A2','B2','pooled'])
        v=a.MAE_gain.to_numpy()*1000
        axes[1].errorbar(v,np.arange(3)+(j-.5)*.17,xerr=np.array([v-a.ci_low.to_numpy()*1000,a.ci_high.to_numpy()*1000-v]),fmt='o',c=color,label=label,capsize=3)
    axes[1].axvline(0,c='#555555',ls=':');axes[1].set(yticks=range(3),yticklabels=['A2','B2','合并'],xlabel='MAE 改善（毫秒；正值更好）',title='2000 次块重采样区间');axes[1].invert_yaxis();axes[1].legend(fontsize=8)
    save(fig,figs,'05_反应时验证','139 次独立前缀质量筛查试次；仅用目标后 600 ms 内 EEG。1999 次记录内、块内置换均重新监督选型。')

    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    fig.suptitle('机制消融与半合成恢复')
    effects(axes[0],iv[(iv.setting=='stimulus')&(iv.stage=='retrieval')],
        [('N2_no_projection','N2 对无直接记忆投影',BLUE,'o'),('N2_no_retrieval','N2 对关闭提取门',ORANGE,'D')],'目标后阶段的记忆环节增量')
    axes[1].plot(recovery.noise_ratio,recovery.NRMSE_clean,'o-',c=BLUE)
    axes[1].set(xlabel='残差背景强度 / 干净源 RMS',ylabel='留出干净波形 NRMSE',title='已知生成模型的恢复检查',ylim=(0,max(.1,recovery.NRMSE_clean.max()*1.3)))
    for row in recovery.itertuples():axes[1].annotate('动态参数恢复' if row.exact_parameters else '动态参数未完全恢复',(row.noise_ratio,row.NRMSE_clean),xytext=(4 if row.noise_ratio==0 else (-4 if row.noise_ratio==1 else 0),12),textcoords='offset points',ha='left' if row.noise_ratio==0 else ('right' if row.noise_ratio==1 else 'center'),fontsize=8)
    save(fig,figs,'06_消融与恢复','所有消融均重新训练。半合成固定前四块训练、第五块验证；恢复人工源不等于从实测 EEG 定位海马。')

    selected=[]
    for path in sorted(out.glob('*_choices.json')):
        if 'stimulus' not in path.name and 'past' not in path.name:continue
        for row in json.loads(path.read_text()):
            selected.append(dict(dataset=row['dataset'],setting=row['setting'],held_block=row['held_block'],requested_model=row['model'],
                **row['parameters'],lambda_ridge=row['selection']['lambda_ridge'],edf=row['meta']['edf'],
                last_IRLS_change=row['meta']['irls_changes'][-1]))
    save_csv(out/'selected_parameters.csv',selected)
    # In the table preserve the requested model label for constrained ablations.
    main=iv[(iv.setting=='stimulus')&(iv.stage=='retrieval')]
    oldmain=main[main.control=='V1_M2'];positive=int((oldmain.S>0).sum());supported=int((oldmain.ci_low>0).sum())
    gain=bi[(bi.dataset=='pooled')&(bi.contrast=='cognitive_vs_ordinary')].iloc[0]
    rt=behavior[(behavior.dataset=='pooled')&(behavior.model=='cognitive')].iloc[0]
    past=ci[(ci.stage=='retrieval')&(ci.control=='stimulus_N2')]
    source_sel=pd.DataFrame(selected)
    story=f'''# 第三问 V2 优化与验证报告

## 优化结论

已完成一轮有依据的结构优化，原 `q3_result` 全部保留。新版与原版使用同一 268 个有效 EEG 试次、同一信号、评分时点和五块划分；行为另用 139 个固定前缀试次。模型与观测改动不会通过更强滤波或删除难预测试次人为降低误差。

仅输入刺激事件时，目标后阶段新版 N2 相对原版 M2 的 MSE 点估计在 **{positive}/4** 组下降，块重采样 95% 区间完全高于零的有 **{supported}/4** 组。新版认知前缀反应时 MAE 为 **{rt.MAE:.4f} 秒**，相对普通 EEG 前缀改善 **{gain.MAE_gain*1000:.2f} ms**，区间 **[{gain.ci_low*1000:.2f}, {gain.ci_high*1000:.2f}] ms**。

这些比较区分三个问题：新版是否比原版预测更好、背景能否解释误差、记忆模块是否优于相同信息下的普通慢响应。前两项的改善不能代替第三项，也不能证明海马来源已恢复。原题要求建立模型并用数据验证，并未要求四组显著改善或反应时预测必须成功；后者是额外支持证据。

提示前背景辅助使四组目标后阶段的留出 MSE 分别下降 **{'、'.join(f'{v*100:.1f}%' for v in past.set_index('dataset').reindex(KEYS).S)}**（相对于新版仅输入事件的 N2；具体区间见下文）。这是输入信息增加后对背景的预测收益，不能表述为记忆机制本身提高了这些比例。相同背景条件下，N2 没有稳定超过 N3；行为优势的区间也包含零。

## 诊断与修改依据

{table(d[['dataset','n_trials','n_diagnostic_trials','power_le_1Hz','common_energy_fraction','corr_Fz_F3','corr_Fz_F4','corr_F3_F4']])}

诊断仅纳入具有完整 0–3 秒评分窗的 265 次试次（A1/A2/B1/B2 为 64/68/62/71），并非改动主分析的 268 次队列。1 Hz 以下功率占比来自提示后 0–3 秒 Welch 估计；窗长和频率分辨率限制了解释。这些慢变化可能包括背景、伪影与认知成分，不能全部当作伪影删除。A1/B1 的额中与额侧相关为负，也不支持对所有记录统一减去三通道平均值。

事件定义沿用原审计：平台起点只是候选目标时刻；Task-1 无独立点击，Task-2 才有 ±2 标记。B1 第 49/87 次符号不一致不自动判错，正确性与超时保持未知。现有资料不足以重新确证这些语义，因此未为提高分数而改写标记。

![诊断](figures/01_慢变化诊断.png)

## 改进后的宏观模型

由 Q2 的形状编码、LGN、E/I 群体和突触响应分别形成提示驱动 vᶜ 与共同目标驱动 vᵀ。共同提示成分与朝向差分分开投影；目标只采用共同输入，不使用点击方向或应答时长。与原版相比，目标活动不再写入需要保持的旧提示记忆。

$$\\tau_m\\dot m=-m+v^c,\\qquad
\\tau_p\\dot p^c=-p^c+v^c+G_T(t)m(t-\\delta),\\qquad
\\tau_p\\dot p^T=-p^T+v^T.$$

$$\\widehat x=L_v[v^c,v^T]+L_mm+L_p[p^c,p^T].$$

共同/差分提示各一个记忆状态，目标有独立整合状态；不同列的有效观测系数由训练数据估计。δ 固定 50 ms、反馈固定为零、提取增益固定为 1，避免与混合幅度同时自由变化。这是简化假设，不是测出的生理常数。τm 取 0.5/1/2/4 秒，τp 取 0.1/0.3/0.6 秒。

目标输入持续时间 0.1/0.2/0.4 秒在训练内选型，只表示未知显示过程的敏感性候选。方向成分相对共同成分的岭惩罚倍率为 1/10，允许弱方向成分更强收缩。所有模拟神经信号施加与实际 EEG 相同的因果观测滤波；保持原 V1 评分目标。

N0 为视觉模型；N1 添加普通低通而没有记忆；N2 为完整模型；N3 为一般慢趋势。无直接记忆投影、关闭提取门、固定目标时长 200 ms、固定等强度惩罚各自重新训练。内层先对完整候选网格作岭筛查，前两组用 Huber IRLS 重拟合选型。外层测试块不参与缩放或参数估计。

## 真实 EEG 的新旧对照

以下为目标后阶段，S=1−MSE_N2/MSE_对照，五个块等权，正值更好。95% 区间来自 2000 次时间块重采样；未作多重区间校正，不能等同于独立受试者总体置信区间。

{table(main[main.control.isin(['V1_M2','N0','N1','N3'])][['dataset','control','S','ci_low','ci_high']])}

![新旧比较](figures/02_新旧模型与认知增量.png)

仅输入事件的绝对预测分数如下。S_train_baseline 使用训练集各阶段平均电位作为基线；正值才优于该基线，不能仅凭新旧相对降幅判断预测充分。

{table(s[(s.setting=='stimulus')&(s.model=='N2')][['dataset','stage','RMSE','NRMSE','S_train_baseline']])}

固定目标时长和等惩罚对照用于检查新增自由度是否值得保留，不按外层结果再挑选一个“最终赢家”。完整消融、参数及来源系数见 `intervals.csv`、`selected_parameters.csv`、各组 `*_choices.json`；方向条件及右减左差分误差见 `*_contrasts.csv`。方向已知输入下的波形预测不等于未知朝向分类。

## 提示前背景辅助预测

该附加设定显式允许使用提示前 EEG，因此与仅输入事件的预测分开报告。读取每次提示之前最多 8 秒的因果滤波数据，在 0.25/1/4 秒尺度估计历史斜率 d；外推项采用 dτ(1−exp(−t/τ))，所有系数仍仅由训练块估计。背景项直接定义在已滤波观测域，不解释为神经源；不使用提示后的实际 EEG 更新背景。

{table(past[['dataset','S','ci_low','ci_high']])}

上述表只衡量背景信息的预测价值。纯背景 B0、视觉 N0、无记忆 N1、完整 N2 和慢趋势 N3 都获得相同背景输入，记忆证据必须来自它们之间的比较。

{table(iv[(iv.setting=='past')&(iv.model=='N2')&(iv.stage=='retrieval')&(iv.control.isin(['B0','N1','N3']))][['dataset','control','S','ci_low','ci_high']])}

![背景与认知](figures/03_背景与认知分离.png)

![外层波形](figures/04_外层波形.png)

图中观测是跨试次汇总；尾部仅含应答较慢的试次，不把均值尾部变化直接解释为所有试次的神经演化。

## 行为验证与机制消融

行为保留原固定前缀队列，最后纳入目标后 597.65625 ms 的采样点。每个行为内层重新用其训练块前缀 EEG 选择 N2 动力学与读出矩阵；从前缀估计少量源幅度，构造记忆水平、认知水平、变化率及保持量变化。监督岭回归预测 log(RT)，与相同前缀的普通九维 EEG 均值特征和训练 RT 中位数比较。没有使用完整应答窗长度或应答锁定坐标。

{table(behavior[['dataset','model','n','MAE','R2','S_vs_train_median','spearman']])}

{table(pd.read_csv(out/'behavior_permutation.csv')[['dataset','contrast','MAE_gain','p_raw','p_Holm_6']])}

相对原版行为模型的配对比较：

{table(bc)}

![行为](figures/05_反应时验证.png)

{table(recovery)}

半合成实验固定前四块训练、第五块验证。真实 EEG 残差只作有结构的噪声背景，已知真值仅是人工生成部分；三种噪声强度不是三名受试者。动态参数准确恢复与预测波形误差分别报告，不将其中一项替代另一项。

![消融与恢复](figures/06_消融与恢复.png)

错误与未及时应答的证据积累机制保持原模型设计，见 [原版决策仿真](../q3_result/第三问_建模与验证报告.md)。这些仍是已知目标映射下的机制仿真，新版没有补造实测正确率或超时率，也没有把原版仿真伪称为新版拟合结果。

## 论文表述与边界

本轮优化修正了提示保持与新目标输入混用的问题，收缩了难辨识自由参数，并通过相同观测下的对照区分背景预测与认知机制增量。应按上述分组表描述改善范围，不将局部改善写成全面成功。只有三路额区 EEG、两名受试者；记忆状态具有功能解释，不等于实际恢复海马信号。

这是在已反复查看的数据上的内部探索优化。即使某个区间完全高于零，也需要新试次或新受试者验证，不能作为疾病诊断依据。没有因为原版表现差而降低检查标准，也不要求把真实负结果改成正结果。

## 复现与文件

代码及事先固定的工程方案见 [V2 复现说明](../q3/v2/README.md)。入口为 `OPENBLAS_NUM_THREADS=1 .venv/bin/python -B q3/v2/run.py`。独立审计入口为 `q3/v2/audit.py`。

`*_waves.npz` 保存观测和外层预测；`*_choices.json` 保存训练试次、选型与系数；`manifest.json` 保存来源哈希。旧结果未覆盖。六组新图同时提供 PNG 与矢量 PDF。
'''
    (out/'第三问_V2优化报告.md').write_text(story,encoding='utf8')
    (out/'README.md').write_text('# 第三问 V2 结果\n\n主报告：[第三问_V2优化报告.md](第三问_V2优化报告.md)。\n\n事件输入与背景辅助预测分开；各比较保持同一实际 EEG。完整来源与阶段见 manifest.json；数值审计见 独立复核.json。\n',encoding='utf8')
