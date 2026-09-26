"""Generate figures and a self-contained Chinese methods/results report."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import TIME,KEYS

BLUE='#315B8A';ORANGE='#D67843';RED='#AC4F55';GRAY='#788692'


def table(df,digits=4):
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.{digits}f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def save(fig,path,caption=None,bottom=.12):
    fig.subplots_adjust(bottom=bottom,top=.88,wspace=.3,hspace=.44)
    png=path.with_suffix('.png');temp_png=path.parent/'.__plot_tmp.png'
    fig.savefig(temp_png,dpi=170,facecolor='white');png.unlink(missing_ok=True);temp_png.replace(png)
    pdf=path.with_suffix('.pdf');temp_pdf=path.parent/'.__plot_tmp.pdf'
    fig.savefig(temp_pdf,facecolor='white');pdf.unlink(missing_ok=True);temp_pdf.replace(pdf);plt.close(fig)


def render(out):
    plt.rcParams.update({'font.sans-serif':['Noto Sans CJK JP','Noto Sans CJK SC','Microsoft YaHei','SimHei','DejaVu Sans'],
                        'axes.unicode_minus':False,'font.size':10,
                        'figure.facecolor':'white','axes.facecolor':'white',
                        'axes.edgecolor':'#333333','axes.linewidth':.9,
                        'axes.spines.top':True,'axes.spines.right':True,
                        'axes.axisbelow':True,'axes.grid':True,
                        'grid.color':'#D9DEE3','grid.linestyle':'--',
                        'grid.linewidth':.7,'grid.alpha':.75,
                        'legend.frameon':True})
    figs=out/'figures';figs.mkdir(exist_ok=True)
    events=pd.read_csv(out/'event_audit.csv');summary=pd.read_csv(out/'summary.csv')
    dec=pd.read_csv(out/'decision_summary.csv');curves=pd.read_csv(out/'decision_cumulative.csv')
    sens=pd.read_csv(out/'window_sensitivity.csv');states=np.load(out/'cognitive_states.npz')
    paths=np.load(out/'decision_paths.npz');surv=pd.read_csv(out/'observed_response_survival.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5.2));fig.suptitle('图1  应答标记与认知信号截窗',fontsize=15)
    for j,(status,label,color) in enumerate([('explicit_click','独立点击',BLUE),('platform_end_proxy','平台末端代理',ORANGE),('unobserved','无明确标记',GRAY)]):
        counts=[int(((events.dataset==k)&(events.response_status==status)).sum()) for k in KEYS]
        axes[0].bar(np.arange(4)+(j-1)*.22,counts,width=.21,label=label,color=color)
    axes[0].set(xticks=range(4),xticklabels=KEYS,ylabel='事件数',ylim=(0,120),title='应答标记类型');axes[0].legend(fontsize=9)
    ax=axes[1];ax.set(xlim=(-.2,2.5),ylim=(-.7,2.7),yticks=[2,1,0],yticklabels=['明确点击','项目1代理','标记缺失'],xlabel='相对目标出现时间（s）',title='认知信号截窗规则')
    for y,end,c in [(2,1.3,BLUE),(1,1.3,ORANGE),(0,2.,GRAY)]:ax.plot([0,end],[y,y],c=c,lw=8,solid_capstyle='butt')
    for y in [2,1]:
        ax.plot([1.4,1.4],[y-.22,y+.22],c='#333333');ax.text(1.45,y,'应答前\n100 ms',fontsize=9,va='center')
    ax.axvline(0,c='#999999',ls=':');ax.text(2.05,0,'观察上限\n2 s',fontsize=9,va='center')
    save(fig,figs/'01_事件与截窗','实测标记审计与截窗规则；标记缺失不等同于未应答。')

    fig,axes=plt.subplots(2,2,figsize=(12,8));fig.suptitle('图2  认知模型预测与实测脑电',fontsize=15)
    for ax,key in zip(axes.flat,KEYS):
        z=np.load(out/f'{key}_stimulus_waves.npz');past=np.load(out/f'{key}_past_waves.npz');mask=np.isfinite(z['x'][:,:,0]);n=mask.sum(0)
        for a,label,color in [(z['x'],'实测脑电','#333333'),(z['N2'],'认知模型预测',BLUE),(past['N2'],'背景辅助预测',ORANGE)]:
            mean=np.where(mask,a[:,:,0],0).sum(0)/np.maximum(n,1);mean[n<10]=np.nan
            ax.plot(TIME,mean,label=label,c=color,lw=1.3)
        ax.set(title=f'{key}组（{len(z["ids"])}次）',xlim=(-.25,4.3),xlabel='相对提示时间（s）',ylabel='Fz电位（原始单位）')
        ax.axvline(np.median(z['target']),c='#999999',ls=':')
    axes[0,0].legend(fontsize=8)
    save(fig,figs/'02_EEG继承与重建','同一主队列的冻结参数预测；虚线标出目标出现时刻。')

    fig,axes=plt.subplots(1,3,figsize=(14,5.4));fig.suptitle('图3  记忆—认知—决策状态',fontsize=15)
    for name,label,color in [('nominal','基准',BLUE),('weak_memory','短记忆',ORANGE),('retrieval_off','无提取',GRAY)]:
        axes[0].plot(TIME,states[name+'_m'][:,1],label=label,c=color)
        axes[1].plot(TIME,states[name+'_p'][:,1],label=label,c=color)
        axes[2].plot(paths[name+'_time'],paths[name+'_drift'],label=label,c=color)
    for ax,title in zip(axes,['记忆状态 $m$','认知状态 $p$','决策累积量']):
        ax.set(title=title,ylabel='模型状态（无量纲）',xlabel='相对提示时间（s）' if ax!=axes[2] else '相对目标时间（s）');ax.legend(fontsize=8)
    for ax in axes[:2]:ax.set_xlim(0,4.5);ax.axvline(568/256,c='#999999',ls=':')
    save(fig,figs/'03_统一认知与决策','记忆与认知状态经同一方程驱动决策；曲线均为模型仿真。')

    fig,axes=plt.subplots(1,2,figsize=(13,5.8));fig.suptitle('图4  认知模型的三类决策结果',fontsize=15)
    bottom=np.zeros(len(dec))
    for col,label,color in [('correct_rate','正确',BLUE),('error_rate','错误',RED),('unresolved_rate','未越界（删失）',GRAY)]:
        axes[0].bar(np.arange(len(dec)),dec[col],bottom=bottom,color=color,label=label);bottom+=dec[col]
    axes[0].set(xticks=np.arange(len(dec)),xticklabels=dec.label,ylim=(0,1.05),ylabel='轨迹比例',title='正确、错误与未决比例')
    axes[0].tick_params(axis='x',rotation=20);axes[0].legend(fontsize=8,loc='upper left',ncol=1)
    for name,label,color in [('nominal','基准',BLUE),('weak_memory','短保持',ORANGE),('high_noise','高噪声',RED)]:
        part=curves[curves.scenario==name]
        axes[1].plot(part.time_s,part.unresolved_survival,label=label,c=color)
    axes[1].set(xlabel='相对目标时间（s）',ylabel='未决比例',ylim=(0,1.05),title='未决比例随时间变化');axes[1].legend()
    save(fig,figs/'04_正确错误与删失')

    fig,axes=plt.subplots(1,2,figsize=(12,5.4));fig.suptitle('图5  观察窗敏感性与实测点击时间',fontsize=15)
    for key,color in zip(KEYS,[BLUE,ORANGE,GRAY,RED]):
        s=sens[(sens.dataset==key)&(sens.stage=='retrieval')&(sens.control=='N3')]
        axes[0].plot(s.horizon_s,100*s.S,'o-',c=color,label=key)
    axes[0].axhline(0,c='#888888',ls=':');axes[0].set(xlabel='目标后观察上限（s）',ylabel='相对慢趋势的均方误差改善（%）',title='观察窗敏感性');axes[0].legend(ncol=2)
    for key,color in [('A2',BLUE),('B2',ORANGE)]:
        s=surv[(surv.dataset==key)&(surv.horizon_s==2)]
        axes[1].step(np.r_[0,s.time_s],np.r_[1,s.survival],where='post',label=key,c=color)
    axes[1].set(xlabel='目标出现至点击（s）',ylabel='尚未点击比例',title='项目2实测点击时间',ylim=(0,1.05));axes[1].legend()
    save(fig,figs/'05_观察窗敏感性','观察上限不是实验超时阈值；点击时间包含认知与运动过程。')

    counts=events.groupby(['dataset','response_status']).size().unstack(fill_value=0).reset_index()
    counts['EEG保留']=[int(events[(events.dataset==k)&events.retained].shape[0]) for k in counts.dataset]
    selected=pd.read_csv(out/'selected_parameter_decisions.csv')
    selected_table=selected.groupby('dataset')[['correct_rate','error_rate','unresolved_rate']].agg(['min','max'])
    selected_table.columns=['_'.join(c) for c in selected_table.columns];selected_table=selected_table.reset_index()
    decision_table=dec[['label','n','correct_rate','error_rate','unresolved_rate',
                        'censor_aware_median_s','restricted_mean_decision_time_s']].copy()
    decision_table=decision_table.rename(columns={'censor_aware_median_s':'删失中位数_s',
                                                   'restricted_mean_decision_time_s':'受限均值_s'})
    decision_table['删失中位数_s']=decision_table['删失中位数_s'].apply(lambda x:'观察期内未达到' if pd.isna(x) else f'{x:.4f}')
    inherited=pd.read_csv(out/'reference/intervals.csv')
    iv=inherited[(inherited.setting=='stimulus')&(inherited.stage=='retrieval')&(inherited.control=='N3')][['dataset','S','ci_low','ci_high']]
    report=f'''# 第三问 V3 认知模型与不确定应答处理

V3补齐了没有明确应答时的有限EEG观察窗，以及由同一V2认知状态驱动的正确、错误和未决行为仿真。神经模型与已拟合头皮读出保持V2定义，268次主EEG队列保持一致；新增内容不改变旧版预测成绩，也不将仿真结果写成实测行为准确率。

## 题目要求与事件证据

所有四份MAT的原始Fz/F3/F4、VisCue和TgtAct重新审计。±1平台起点仍仅是假设的目标出现，Task-1平台末端仍是应答代理；Task-2的±2为独立点击。正确目标映射和实验超时阈值没有提供，400次真实correctness和timeout保持未知。

{table(counts)}

![事件与截窗](figures/01_事件与截窗.png)

有已知应答或代理时，EEG终点为应答前100 ms与行政观察上限的较早者。无明确应答时，只要提示和候选目标可识别，就允许在质量检查后保留至目标后2 s的EEG；记录长度、下一提示和24 s块保护仍限制可用范围。这个上限沿用既有决策示例的工程时间尺度，未经实验日志确证，1.5/2/2.5 s均提供敏感性结果。

错误、正确和未知正误使用同一截窗和质量规则，不能因为答错就删除认知过程。若完整记录中存在晚于人为观察上限的点击，则在该上限的时间分析中属于行政右删失；它不等同于实验超时。只有缺失标记而无法判断是否真的未应答的记录，标为“应答未知/可能标记缺失”，仅允许EEG分析，不自动进入有效生存删失统计。Task-1代理也不进入真实点击生存曲线。

当前200次Task-2均有独立点击，200次Task-1均仅有代理，未发现可确认的真实无应答案例。synthetic_marker_stress仅删除已有试次的点击标记来检验程序，真实EEG中可能仍有实际动作活动，因此这些压力样本不进入实测模型结论。额外的合成干净EEG单元测试验证了无应答标记并不会自动导致剔除。

## 统一的视觉 记忆 认知与观测模型

继承Q2形状编码及LGN/E-I/突触动力学，采用V2的提示共同/差分和独立目标通路。局部视觉参数仍固定，头皮读出仍来自V2外层训练块，不是重新定位脑源。

$$\\tau_m\\dot m=-m+v^c,\\quad
\\tau_p\\dot p^c=-p^c+v^c+G_T(t)m(t-\\delta),\\quad
\\tau_p\\dot p^T=-p^T+v^T.$$

$$\\widehat x=L_v[v^c,v^T]+L_mm+L_p[p^c,p^T],\\qquad\\delta=50\\text{{ ms}}.$$

反馈固定为零，记忆仅接收提示。所有神经源经过与实际EEG相同的因果观测滤波。V3重建所有刺激输入/背景辅助外层模型的预测，并核验保存参数排除了对应测试块；没有重新执行V2参数搜索，也没有重新执行139次前缀行为的1999次置换。必要的冻结参数、预测基准及对照表保存在reference，原V2来源清单保存在q3/reference_manifest.json；未使用的旧版报告和图表已清理。

![EEG继承](figures/02_EEG继承与重建.png)

仅输入刺激时，目标后N2相对普通慢趋势N3的原有配对证据仍为：

{table(iv)}

四组区间均跨零。V3的贡献是处理流程和模型连接完整性，不是宣称取得新EEG预测提升。背景辅助与事件输入的信息集合不同，其收益仍不能归因于记忆机制。

## 从同一认知状态到决策

取未经过头皮滤波的V2认知差分状态pΔ，形成方向对齐证据e(t)=d·pΔ(t)/a，其中d为提示方向，a由固定参考状态给出。仿真中的正确目标侧s*由模拟任务布局给定，左右提示与目标左右布局进行平衡组合；真实MAT的点击方向不用于构造s*。

$$dz=s^*\\gamma e(t)dt+\\sigma dW_t,\\quad z(0)=0,\\quad b=1.$$

首次触及±b确定选择。选择等于仿真给定s*时为正确，否则为错误；截止D前未触界的潜在决策时间右删失，保存观察时长D但不把D或0伪装成反应时。z是行为读出，不增加为头皮源，也不反馈修改EEG。没有估计运动耗时，因此这里是模型决策时间，不能直接等同实际点击间隔。

这个任务布局到动作的映射是仿真假设，并未从真实目标画面或EEG中恢复；目标视觉通路仍是共同输入，没有声称完整解决逐次目标匹配或what/where识别。

![认知决策连接](figures/03_统一认知与决策.png)

参考τm=2 s、τp=0.3 s、目标输入0.2 s，γ=1.5、σ=0.6、D=2 s。各场景只按预定规则改变指定参数。每场景6000条轨迹，给出的Wilson区间仅表示蒙特卡洛抽样误差，不是受试者总体区间。不同场景使用共同随机数。

{table(decision_table)}

删失中位数取累计触界比例首次达到50%的时间；若截止时超过一半仍未决，则报告“观察期内未达到”，不把已应答子集的中位数代替总体中位数。受限均值为E[min(T,D)]的样本估计，包含所有未决轨迹；它是观察上限内的统计量，不是假定每名未决者在D时作出了决定。CSV另保留已触界样本的条件中位数并明确其范围。

![仿真行为](figures/04_正确错误与删失.png)

另将四组各五折由EEG选择的V2参数直接接入同一决策层，每折1000次仿真；以下是五折输出范围，不是实测正确率、跨人群区间或行为拟合：

{table(selected_table)}

真实独立点击的描述性生存曲线与上述决策仿真分开保存；它只统计任何点击的时间，不能从中推断正确率、纯认知时长或真实超时。

## 窗口敏感性与验证

1.5/2/2.5 s上限均使用V2冻结参数，并在各上限与主分析共有的合格试次上评分，不根据外层成绩选最佳上限。区间来自2000次时间块重采样，属于两名受试者既有记录的内部探索。窗口变化改变了评分目标，不能当作不同模型公平优劣排名。

![窗口敏感性](figures/05_观察窗敏感性.png)

测试覆盖错误标签不改变窗口、缺失点击不等于超时、无标记EEG可保留、代理不能作为真实生存事件、缺失目标不伪造、因果处理、删失计算、镜像提示证据一致性、决策首次触界与三类结局守恒，以及保存V2预测重建。独立审计从保存波形重算全部误差、仿真结局和累计概率，并核对源码、原始MAT及继承文件哈希。数值测试不能验证未提供的事件语义。

## 可用于论文的结论

本文在第二问视觉形成机制上建立提示保持、目标触发提取与认知整合的功能性宏观模型，并通过头皮观测映射与真实EEG进行对比。对错误应答不作先验排除；对明确应答、应答代理和应答不可观测分别设置截窗与统计规则。由同一认知状态驱动的随机累积层能够产生正确、错误及有限观察期内未决的行为，提供了题目所提示异常应答的可计算表示。

真实EEG尚未稳定支持记忆模块优于一般慢趋势；正确目标映射、超时规则和独立海马观测缺失，故模拟行为不能作为实测准确率或生理源定位证据。V3完成的是统一模型和处理规则，不是补造未被观测的数据。

复现：在项目根目录运行 `.venv/bin/python -B q3/main.py`。代码见 ../q3，原始数据继续从 ../data 只读加载。全部新结果保存在本目录；来源、冻结继承与新计算在manifest及独立复核文件中记录。
'''
    (out/'第三问_最终报告.md').write_text(report,encoding='utf8')
    (out/'README.md').write_text('# 第三问最终结果\n\n先读[完整报告](第三问_最终报告.md)。\n\n- event_audit.csv：400次真实事件与观察规则。\n- *_waves.npz、trial_metrics.csv：V2冻结参数重新生成的外层EEG预测与误差。\n- window_sensitivity*.csv：行政观察上限敏感性。\n- decision_*：直接使用V2认知状态的新决策仿真；不是实测行为。\n- selected_parameter_decisions.csv：V2各外折EEG参数驱动的仿真。\n- synthetic_marker_stress*：标记缺失的软件压力测试，不进入真实统计。\n- reference/：18份必要冻结参数、预测基准和对照表。\n- [优化报告](优化_机制验证报告.md)：新增拟合、参数恢复和目标匹配对照。\n- 入口：`q3/main.py`；默认生成主结果和优化对照，`--stage reconstruct`仅重建主结果。\n- 独立复核.json、测试验收.txt、visual_qa.json：验证记录。\n',encoding='utf8')
