"""Rebuild all research figures and the Chinese report from saved numeric evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import NAMES, TIMES, B, FIT_MASK, save_json, save_csv

BLUE='#2864A0';ORANGE='#C67627';GRAY='#6F747A'
plt.rcParams.update({'font.sans-serif':['Noto Sans CJK SC','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'axes.labelcolor':'#24272B','text.color':'#24272B','axes.titlecolor':'#24272B'})


def table(df,cols=None):
    if cols:df=df[cols]
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.4f}' if np.isfinite(x) else 'NA'
        return str(x)
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |',
        '| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])


def make_figures(out):
    out=Path(out);dest=out/'figures';dest.mkdir(exist_ok=True);chartmap=[]
    def save(fig,name,question,source):
        fig.savefig(dest/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white')
        fig.savefig(dest/f'{name}.pdf',bbox_inches='tight',facecolor='white')
        plt.close(fig);chartmap.append(dict(figure=name,question=question,source=source,
                renderer='matplotlib static',palette='blue orange and neutral; solid/dashed distinction'))
    if (out/'mechanism/spatial_encoding.npz').exists():
        s=np.load(out/'mechanism/spatial_encoding.npz')
        fig,ax=plt.subplots(2,4,figsize=(14,6),layout='constrained')
        titles=['左向三角','右向三角','缺失尖角','排列打乱']
        for i,title in enumerate(titles):
            ax[0,i].imshow(s['images'][i],cmap='gray',vmin=0,vmax=1);ax[0,i].set_title(title);ax[0,i].axis('off')
            ax[1,i].bar([0,1,2],s['inputs'][i],color=[BLUE,ORANGE,GRAY]);ax[1,i].set_xticks([0,1,2],['左选择','右选择','共同'])
            ax[1,i].set_ylim(0,1.15);ax[1,i].set_title(f"η = {s['eta'][i]:+.3f}");ax[1,i].set_ylabel('归一化响应')
        fig.suptitle('示意刺激的局部空间组合响应\n模板由示意三角建立；不代表实测皮层位置',fontsize=15)
        save(fig,'01_空间组合选择性','镜像和局部结构是否改变选择性响应','mechanism/spatial_encoding.npz')
        d=np.load(out/'mechanism/neural_forward.npz');fig,ax=plt.subplots(3,3,figsize=(14,10),layout='constrained')
        for j,name in enumerate(['左选择群体','右选择群体','共同群体']):
            for i,(color,style,label) in enumerate([(BLUE,'-','左刺激'),(ORANGE,'--','右刺激')]):
                ax[0,j].plot(d['t']*1000,d['state'][i,j],color=color,ls=style,label=label)
                ax[1,j].plot(d['t']*1000,d['dipoles'][i,j],color=color,ls=style)
                ax[2,j].plot(d['t']*1000,d['eeg'][i,j],color=color,ls=style)
            ax[0,j].set_title(name);ax[2,j].set_xlabel('相对刺激时间 / ms')
            ax[2,j].set_title(['Fz','F3','F4'][j])
        for a in ax.flat:a.axvline(0,color=GRAY,lw=.7);a.grid(alpha=.15)
        for i,l in enumerate(['LGN 驱动 / 示意单位','突触偶极 / 示意单位','头皮观测 / 示意单位']):ax[i,0].set_ylabel(l)
        ax[0,0].legend();fig.suptitle('LGN → 群体动态与突触电流 → 固定空间投影\n参数与导联几何均为示意假设；与实测拟合参数分开',fontsize=15)
        save(fig,'02_神经群体前向响应','各机制阶段如何形成电位','mechanism/neural_forward.npz')
    for key,name in NAMES.items():
        if not (out/'fits'/key/'before/waveforms.npz').exists():continue
        before=np.load(out/'fits'/key/'before/waveforms.npz');v7=np.load(out/'fits'/key/'v7/waveforms.npz')
        fig,axes=plt.subplots(3,3,figsize=(17,10),layout='constrained')
        for ch,cname in enumerate(['Fz','F3','F4']):
            for col,data in enumerate([before,v7]):
                ax=axes[ch,col]
                for d,(color,label) in enumerate([(BLUE,'左'),(ORANGE,'右')]):
                    ax.plot(TIMES*1000,data['observed'][d,ch],color=color,label=label+' ERP')
                    ax.plot(TIMES*1000,data['predicted'][d,ch],color=color,ls='--',label=label+'模型')
                ax.set_title(cname+' · '+['预处理','V7'][col]);ax.set_ylabel('原始电位单位')
            lo=min(axes[ch,0].get_ylim()[0],axes[ch,1].get_ylim()[0]);hi=max(axes[ch,0].get_ylim()[1],axes[ch,1].get_ylim()[1])
            axes[ch,0].set_ylim(lo,hi);axes[ch,1].set_ylim(lo,hi)
            for d,(color,label) in enumerate([(BLUE,'左'),(ORANGE,'右')]):
                axes[ch,2].plot(TIMES*1000,v7['residual'][d,ch],color=color,ls='-' if d==0 else '--',label=label+'残差')
            axes[ch,2].set_title(cname+' · V7 残差');axes[ch,2].axhline(0,color=GRAY,lw=.8)
        for ax in axes.flat:
            ax.axvline(0,color=GRAY,lw=.7);ax.axvspan(50,750,color=GRAY,alpha=.04);ax.set_xlabel('相对刺激时间 / ms');ax.grid(alpha=.15)
        axes[0,0].legend(ncol=2,fontsize=9);axes[0,2].legend()
        fig.suptitle(f'{name} 条件 ERP 与机制约化拟合\n同数据描述性拟合；灰底为 50–750 ms 拟合范围',fontsize=15)
        save(fig,f'03_{key}_条件拟合','模型解释各通道条件均值的程度',f'fits/{key}/before and v7/waveforms.npz')
    for key,name in NAMES.items():
        paths=list((out/'validation/blocked'/key).glob('fold_*.npz'))
        if not paths:continue
        obs=[];pred=[]
        for p in sorted(paths):
            v=np.load(p);obs.append(B.T@(v['obs'][1]-v['obs'][0]));pred.append(B.T@(v['prediction_M1'][1]-v['prediction_M1'][0]))
        obs=np.stack(obs);pred=np.stack(pred)
        fig,ax=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
        for j,label in enumerate(['共同','中线相对两侧','两侧差分']):
            ax[j].fill_between(TIMES*1000,obs[:,j].min(0),obs[:,j].max(0),color=BLUE,alpha=.12,label='五折观测范围')
            ax[j].plot(TIMES*1000,obs[:,j].mean(0),color=BLUE,label='留出观测均值')
            ax[j].plot(TIMES*1000,pred[:,j].mean(0),color=ORANGE,ls='--',label='训练模型预测均值')
            ax[j].set_title(label);ax[j].set_xlabel('相对刺激时间 / ms');ax[j].set_ylabel('右减左 / 原始电位单位');ax[j].axhline(0,color=GRAY,lw=.7)
        ax[0].legend(fontsize=8);fig.suptitle(f'{name} 时间留出方向差分\n五折等权汇总；阴影为折间最小–最大范围，不是置信区间',fontsize=14)
        save(fig,f'04_{key}_留出空间差分','训练差分能否复现在留出时段',f'validation/blocked/{key}/fold_*.npz')
    summary_path=out/'validation/classification_summary.csv'
    if summary_path.exists():
        s=pd.read_csv(summary_path);models=['mechanism_selected','window_mean_9','common_only','lateral_only','prestim_past_only','trial_order','artifact_quality']
        labels=['机制特征','固定窗均值','仅共同模态','仅两侧差分','刺激前独立处理','试次顺序','伪影量']
        fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
        for ax,(key,name) in zip(axes.flat,NAMES.items()):
            g=s[s.dataset==key].set_index('model').loc[models]
            for i,row in enumerate(g.itertuples()):
                ax.plot([row.BA_low,row.BA_high],[i,i],color=BLUE,lw=2)
                ax.scatter(row.BA,i,color=BLUE,marker='o' if i==0 else 's',facecolors=BLUE if i==0 else 'white',zorder=3)
                ax.text(1.03,i,f'{row.BA:.3f}',va='center',fontsize=9)
            ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,1.12);ax.set_xticks([0,.25,.5,.75,1])
            ax.axvline(.5,color=GRAY,ls='--');ax.set_title(name);ax.set_xlabel('平衡准确率 BA')
        fig.suptitle(f'时间留块验证与对照\n区间：固定折外模型的 {int(s.n_boot.iloc[0]):,} 次时间块重采样；机会参考 0.5',fontsize=15)
        save(fig,'05_识别与负对照','机制特征与透明基准相比是否有益','validation/classification_summary.csv')
        fig,axes=plt.subplots(1,4,figsize=(14,4.3),layout='constrained')
        for ax,(key,name) in zip(axes,NAMES.items()):
            row=s[(s.dataset==key)&(s.model=='mechanism_selected')].iloc[0]
            cm=np.array([[row.TN,row.FP],[row.FN,row.TP]],int);ax.imshow(cm,cmap='Blues',vmin=0)
            for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else '#24272B',fontsize=14)
            ax.set_xticks([0,1],['左','右']);ax.set_yticks([0,1],['左','右']);ax.set_xlabel('预测方向');ax.set_ylabel('真实方向');ax.set_title(name)
        fig.suptitle('机制特征折外混淆矩阵 · 数值为试次数',fontsize=14)
        save(fig,'06_混淆矩阵','左右召回是否均衡','validation/classification_summary.csv')
    pp=out/'validation/permutation_tests.csv'
    if pp.exists():
        pvals=pd.read_csv(pp);fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,(key,name) in zip(axes.flat,NAMES.items()):
            vals=pd.read_csv(out/'validation/permutations'/f'{key}.csv');row=pvals[pvals.dataset==key].iloc[0]
            ax.hist(vals.BA,bins=20,color=BLUE,alpha=.75);ax.axvline(row.observed_BA,color=ORANGE,lw=2,ls='--',label='真实标签 BA')
            ax.set_title(f'{name}\n置换 p={row.p_raw:.4f}；Holm p={row.p_holm:.4f}')
            ax.set_xlabel('每次完整重拟合的 BA');ax.set_ylabel('置换次数');ax.legend()
        fig.suptitle('块内标签置换检验\n每次重做监督时间核拟合、训练内选型和分类训练',fontsize=14)
        save(fig,'07_置换检验','观测识别效果是否超过块内置换参考','validation/permutations/*.csv')
    effects=[]
    for key in NAMES:
        p=out/'features'/f'{key}_oof_nine_features.csv'
        if not p.exists():continue
        f=pd.read_csv(p)
        for k in range(1,10):
            right=f.loc[f.cue==1,f'z_{k}'];left=f.loc[f.cue==-1,f'z_{k}']
            pool=np.sqrt(((len(right)-1)*right.var()+(len(left)-1)*left.var())/(len(f)-2))
            effect=(right.mean()-left.mean())/pool if pool>1e-12 else np.nan
            rng=np.random.default_rng(20260924+k);vals=[]
            for _ in range(1000):
                sample=pd.concat([f[f.fold==b] for b in rng.choice(f.fold.unique(),len(f.fold.unique()))])
                a=sample.loc[sample.cue==1,f'z_{k}'];b=sample.loc[sample.cue==-1,f'z_{k}']
                sd=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(sample)-2))
                vals.append((a.mean()-b.mean())/sd if sd>1e-12 else np.nan)
            lo,hi=np.nanquantile(vals,[.025,.975])
            effects.append(dict(dataset=key,feature=k,standardized_effect=effect,low=lo,high=hi,
                interval='fixed_fold_specific_nine_kernel_models_exploratory'))
    if effects:
        eff=pd.DataFrame(effects);save_csv(effects,out/'features/effect_sizes.csv')
        fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
        labels=[m+'·'+k for m in ['共同','中线','两侧'] for k in ['快','中','慢']]
        for ax,(key,name) in zip(axes.flat,NAMES.items()):
            g=eff[eff.dataset==key]
            for j,r in enumerate(g.itertuples()):ax.plot([r.low,r.high],[j,j],color=BLUE);ax.scatter(r.standardized_effect,j,color=BLUE,s=20)
            ax.set_yticks(range(9),labels);ax.invert_yaxis();ax.axvline(0,color=GRAY,ls='--');ax.set_title(name);ax.set_xlabel('右减左 / 合并组内标准差')
        fig.suptitle('三核九维候选特征的探索性效应\n每折训练标准化；条件区间固定已训练核，特征坐标随折而变',fontsize=14)
        save(fig,'08_九维特征效应','方向差异位于哪些空间时间成分','features/effect_sizes.csv')
    p=out/'validation/supplemental_metrics.csv'
    if p.exists():
        s=pd.read_csv(p);fig,ax=plt.subplots(figsize=(11,9),layout='constrained')
        labels=[f'{r.source}→{r.target}  {r.validation}' for r in s.itertuples()]
        ax.scatter(s.BA,np.arange(len(s)),color=BLUE);ax.set_yticks(np.arange(len(s)),labels);ax.invert_yaxis();ax.set_xlim(0,1)
        ax.axvline(.5,color=GRAY,ls='--');ax.set_xlabel('平衡准确率 BA');ax.set_title('前向、缓冲敏感性、校正敏感性与跨组检验\n单次划分或配对折外结果；两名受试者不支持人群泛化结论')
        save(fig,'09_补充验证','模型在更严格场景中的表现','validation/supplemental_metrics.csv')
    p=out/'synthetic/scenario_metrics.csv'
    if p.exists():
        s=pd.read_csv(p);fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        ax.barh(s.scenario,s.BA,color=BLUE);ax.axvline(.5,color=GRAY,ls='--');ax.set_xlim(0,1);ax.set_xlabel('平衡准确率 BA')
        ax.set_title('假设内与模型外合成检验\n眼动场景没有神经方向项；其可分类性不能作为神经形状编码证据')
        save(fig,'10_合成检验','方向相关伪影是否也会被特征识别','synthetic/scenario_metrics.csv')
    save_json(chartmap,dest/'chart_map.json')
    return chartmap


def write_report(out):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_text());quick=manifest['config']['quick']
    parts=['# 第二问机制建模与内部验证实验报告',
        '\n本报告由保存的数值结果自动生成。计算流程将已知条件的波形解释与未知方向识别分开；全部差分采用右减左。',
        '**运行模式：'+('快速调试，不能作为完整验证终稿。' if quick else '正式完整配置。')+'**',
        '\n## 输入与隔离',
        '四组原始数据均使用 Fz/F3/F4，256 Hz。V7 仅用于条件描述；未知刺激识别重新读取原始 EEG。每组按原始事件划分五段，段间不共享滤波样本，主分析使用 24 秒缓冲。固定质量阈值沿用第一问，数值单位为原始电位单位。']
    counts=[]
    for key in NAMES:
        p=out/'audit'/f'{key}_events_guard24.csv'
        if p.exists():
            a=pd.read_csv(p);g=a[a.retained]
            counts.append(dict(数据组=key,总事件=len(a),可用=len(g),左=int((g.cue==-1).sum()),右=int((g.cue==1).sum())))
    if counts:parts.append(table(pd.DataFrame(counts)))
    parts.append('数值阶段状态：'+ '；'.join(f'{k}={v["status"]}' for k,v in manifest['stages'].items() if k!='report'))
    parts.extend(['\n## 机制模型与求解',
        '空间仿真采用示意三角的局部边缘组合响应，随后驱动稳定 Wilson–Cowan 群体，经突触滤波和固定几何偶极投影生成示意电位。假设参数与真实 EEG 估计参数分别存放。局部空间组合思想参考 [Azzopardi 与 Petkov 2014](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2014.00080/full)。',
        '实测模型为 x̂=B(A+dD)H。空间基 B 正交，不能解释为三个已定位脑源。有限脉冲和级联因果核先经同样的零相位观测滤波，再作刺激前中位数基线校正和加权归一化。τG 固定为20 ms，并检查10/40 ms。',
        '内层采用固定有界网格上的变量投影，选择核数和正则强度；外层最终训练采用三个不同初值的有界连续细化。近似平局定义为最小验证误差的1%以内，依次偏向两核和较强正则。每次置换执行相同流程。内层网格是计算近似，不声称找到连续参数的全局最优。',
        '三个时间窗按长度归一化并等权，左右条件等权。投影特征采用同一窗口加权和0.1岭正则；标准化、自动收缩LDA及等类别先验全部只在训练中建立。三核九维候选单独保存；选到两核时主模型实际使用六维，不用零填充冒充九维。'])
    p=out/'fits/parameters.csv'
    if p.exists():
        parts.append('\n## 已知条件的描述性拟合')
        parts.append(table(pd.read_csv(p),['dataset','stage','rank','delta_ms','tau_E_ms','tau_I_ms','tau_S_ms','condition','bound_hits','contrast_RMSE']))
        channel_scores=pd.read_csv(out/'fits/channel_metrics.csv')
        negative=int((channel_scores.R2<0).sum())
        parts.append(f'全部 {len(channel_scores)} 个方向×通道描述拟合中有 {negative} 个 R²<0，说明模型对部分波形的解释仍不足，不能概括为整体拟合良好。')
        parts.append('这里的误差来自同组条件均值拟合，不是留出预测。边界命中和较大条件数提示有效时间参数可能不唯一；系数符号不能直接解释为兴奋或抑制。逐通道R²、残差和局部正峰缺失情况见 fits/channel_metrics.csv。')
    p=out/'validation/classification_summary.csv'
    if p.exists():
        s=pd.read_csv(p);g=s[s.model=='mechanism_selected']
        parts.extend(['\n## 时间留出与单试次识别',table(g,['dataset','n','BA','BA_low','BA_high','AUC','recall_left','recall_right'])])
        base=s[s.model=='window_mean_9'].set_index('dataset')
        comparisons=[]
        for r in g.itertuples():comparisons.append(f'{r.dataset}：机制 BA={r.BA:.3f}，固定窗均值 BA={base.loc[r.dataset,"BA"]:.3f}。')
        parts.append(' '.join(comparisons))
        parts.append('BA区间为固定已训练折外模型的时间块重采样，只有五个时间块，不覆盖开发选择、模型重训或跨受试者不确定性。AUC使用合并的折外LDA概率，跨折概率尺度仍可能不同。')
        waves=pd.concat([pd.read_csv(p) for p in (out/'validation/blocked').glob('*_waveform.csv')])
        ws=waves.groupby(['dataset','model'])[['contrast_RMSE','S_delta','weighted_MSE']].mean().reset_index()
        parts.extend(['\n下表为各折指标的等权描述汇总；SΔ 分母是对应测试块的差分能量。',table(ws)])
        parts.append('SΔ>0表示优于预测无方向差，≤0表示未获得这项收益；差分能量很小时须优先看绝对RMSE。M0不含方向项，M2为训练ERP模板。')
        controls=s[s.model.isin(['prestim_past_only','prestim_zero_phase','trial_order','artifact_quality'])]
        parts.extend(['\n## 负对照与补充检验',table(controls,['dataset','model','BA','AUC'])])
        parts.append('刺激前、试次顺序和伪影量对照用于暴露非特异信息，不是单独的神经机制检验。刺激前独立处理只使用刺激前24.25秒数据；末端滤波仍有边界效应，但不会读取刺激后的样本。非显著负对照也不能排除眼动。')
    p=out/'validation/permutation_tests.csv'
    if p.exists():
        s=pd.read_csv(p);parts.extend(['\n## 完整监督流程标签置换',table(s)])
        significant=s.loc[s.p_holm<.05,'dataset'].tolist()
        parts.append('Holm 校正后 p<0.05 的数据组：'+('、'.join(significant) if significant else '无')+'。这个检验支持的最多是相应记录中的方向相关信息，不能区分神经活动与方向相关眼动。')
        parts.append('块内置换保持每块左右数量，p=(1+置换BA≥观测BA的次数)/(B+1)。成立条件是块内标签近似可交换；较长时段中的漂移或更细时间依赖仍是限制。其他消融和跨组结果为探索性，不进行择优替换主检验。')
    else:parts.append('\n**完整置换尚未完成；当前不得作显著性结论。**')
    p=out/'validation/supplemental_metrics.csv'
    if p.exists():parts.append(table(pd.read_csv(p),['source','target','validation','n','BA','AUC']))
    p=out/'validation/shared_project_parameters.csv'
    if p.exists():
        s=pd.read_csv(p).groupby(['subject','model'])[['weighted_MSE','contrast_RMSE']].mean().reset_index()
        parts.extend(['\n## 项目共享参数与参数稳定性',table(s),
            '共享模型的时间参数跨项目共用，幅度独立；内层验证也不读取测试块。只依据留出误差比较复杂度，不能从慢核直接推断记忆或海马活动。'])
    boot=[]
    for key in NAMES:
        p=out/'stability'/key/'full_refit_bootstrap.json'
        if not p.exists():continue
        b=json.loads(p.read_text());boot.append(dict(dataset=key,n_refits=len(b),rank2=sum(r['rank']==2 for r in b),
            rank3=sum(r['rank']==3 for r in b),condition_median=np.median([r['condition'] for r in b]),
            bound_hit_fraction=np.mean([bool(r['bound_hits']) for r in b])))
    if boot:parts.extend([table(pd.DataFrame(boot)),
        '重采样在每个原时间块和方向内有放回抽取试次，并完整重做内层选型与参数拟合；保留原块用于内层隔离。它量化条件于五个记录块的试次波动，不等于人群参数置信区间。参数剖面、初值轨迹、边界及正则敏感性均保存在 stability。'])
    p=out/'synthetic/scenario_metrics.csv'
    if p.exists():parts.extend(['\n## 合成闭环与模型外风险',table(pd.read_csv(p)),
        '精确投影闭环检查程序代数实现，完整拟合检查参数和波形恢复；正则会带来偏差，时间常数不保证唯一恢复。眼动场景明确把真实神经方向系数设为零，用来演示高识别率也可能来自伪影。'])
    parts.extend(['\n## 结论边界',
        '应分别判断平均波形拟合、跨时间差分预测和单试次识别，不将其中任何一个成功自动推广为另两个成功。数据只有两名受试者、三个额区电极，第一问和第二问都在现有开发数据上完成；无法唯一确定脑源、排除方向相关眼动、证实海马来源或给出临床诊断精度。处理是离线零相位流程，不是实时因果解码器。',
        '\n## 复核与文件入口',
        '运行命令、依赖和阶段说明见运行说明.md；公式与缺失值口径见指标字典.md。manifest.json记录代码、输入哈希和阶段状态；所有图表从CSV/NPZ重建。checkpoints中保留每次置换、每次重采样的独立结果。'])
    if (out/'validation/permutation_tests.csv').exists() and (out/'validation/classification_summary.csv').exists():
        pv=pd.read_csv(out/'validation/permutation_tests.csv')
        scores=pd.read_csv(out/'validation/classification_summary.csv')
        main=scores[scores.model=='mechanism_selected']
        score_text='、'.join(f'{r.dataset} {r.BA:.1%}' for r in main.itertuples())
        conclusion=('四组主检验经Holm校正后均未达到0.05；当前模型尚未获得稳定区分左右刺激的证据。'
                    if not (pv.p_holm<.05).any() else '部分记录通过Holm校正检验，证据范围仍限于对应记录的方向相关信息。')
        parts.insert(2,f'核心结果：严格时间留块识别的平衡准确率为{score_text}。{conclusion}拟合、留出差分和识别指标分别报告，结果未按期望准确率筛选。')
    figures=sorted((out/'figures').glob('*.png'))
    parts.append('\n## 图表索引\n\n'+'\n'.join(f'- [{p.stem}](figures/{p.name})' for p in figures))
    (out/'第二问实验报告.md').write_text('\n\n'.join(parts)+'\n',encoding='utf8')
