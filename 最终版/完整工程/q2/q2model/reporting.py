"""Paper figures and evidence-conditioned Chinese manuscript from saved results."""
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .data import TIMES, WEIGHTS, NAMES

BLUE='#2667a5';RED='#c44e52';GREY='#687782';GREEN='#28816b'
CHANNELS=('Fz','F3','F4')
FEATURE_NAMES={'amplitude':'均值振幅','peak_latency':'峰值特征','spatial':'侧化特征',
               'projection6':'六维机制投影','direction3':'三维方向投影','mechanism':'机制特征','covariance':'协方差基线','past_only':'刺激前',
               'previous_cue':'前次标签','post_given_past':'刺激后增量'}
MODEL_NAMES={'full':'完整模型','no_shape':'取消形状','no_recurrence':'取消跨群复发','symmetric_readout':'对称观测',
             'gamma':'六时间核','step':'平滑阶跃','ramp':'慢斜坡','zero':'零方向差'}


def setup():
    available={f.name for f in font_manager.fontManager.ttflist}
    candidates=['Microsoft YaHei','Noto Sans CJK SC','SimHei','Arial Unicode MS']
    chosen=next((f for f in candidates if f in available),None)
    if chosen is None: raise RuntimeError('Install a Chinese font before rendering manuscript figures')
    plt.rcParams.update({'font.family':chosen,'axes.unicode_minus':False,'font.size':9,
                         'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240,
                         'figure.facecolor':'white','axes.facecolor':'white',
                         'axes.edgecolor':'#333333','axes.linewidth':.9,
                         'axes.spines.top':True,'axes.spines.right':True,
                         'axes.axisbelow':True,'axes.grid':True,
                         'grid.color':'#D9DEE3','grid.linestyle':'--',
                         'grid.linewidth':.7,'grid.alpha':.75,
                         'legend.frameon':True})


def save(fig,out,name):
    fig.savefig(out/f'Q2_{name}.png',bbox_inches='tight',facecolor='white')
    fig.savefig(out/f'Q2_{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def decorate(ax,ylabel=True):
    ax.axvline(0,color=GREY,lw=.6);ax.axhline(0,color=GREY,lw=.5,alpha=.5)
    ax.axvspan(250,500,color='#e5d6a6',alpha=.24)
    ax.set_xlim(-250,800);ax.set_xlabel('相对提示时间（ms）')
    if ylabel: ax.set_ylabel('电位（原始单位）')


def render_all(out):
    setup();out=Path(out)
    w=np.load(out/'Q2_机制波形.npz');data=np.load(out/'Q2_分析数据.npz');shape=np.load(out/'Q2_形状输入.npz')
    summary=pd.read_csv(out/'Q2_机制汇总.csv');metric=pd.read_csv(out/'Q2_表4_特征判别.csv')
    ms=1000*w['times_s']
    # Figure 1: actual encoded schematic and explicit model architecture.
    fig=plt.figure(figsize=(12,6.1),layout='constrained');gs=fig.add_gridspec(2,4,height_ratios=[1,1.3])
    labels=['左三角','右三角','缺失尖角（模拟）','位置打乱（模拟）']
    for i,label in enumerate(labels):
        ax=fig.add_subplot(gs[0,i]);im=shape['images'][i];ys,xs=np.nonzero(shape['images'][:2].sum(0))
        ax.imshow(im[max(0,ys.min()-5):ys.max()+6,max(0,xs.min()-5):xs.max()+6],cmap='Blues',vmin=0,vmax=1)
        ax.set_title(label)
        ax.axis('off')
    ax=fig.add_subplot(gs[1,:]);ax.axis('off');ax.grid(False);ax.set_xlim(0,1);ax.set_ylim(0,1)
    boxes=[('边缘与空间组合',.07),('LGN\n三路中继',.25),('视觉皮层三级响应\nE–I神经群',.52),('突触源与\n头皮观测',.78),('Fz\nF3 / F4',.94)]
    for text,x in boxes:
        ax.text(x,.62,text,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.6',fc='#eef3f7',ec=BLUE))
    for x1,x2 in ((.14,.2),(.3,.36),(.68,.72),(.84,.9)):
        ax.annotate('',xy=(x2,.62),xytext=(x1,.62),arrowprops=dict(arrowstyle='->',color=GREY,lw=1.5))
    fig.suptitle('图1  左右三角形状编码与脑电形成路径',fontsize=13)
    save(fig,out,'图1_机制与形状编码')
    for key in NAMES:
        suffix='' if key=='A1' else f'_补图_{key}'
        # Figure 2: all stages, all E/I populations and source dynamics.
        t=w[f'{key}_neural_t']*1000;s=w[f'{key}_neural_state'];q=w[f'{key}_neural_q']
        fig,axes=plt.subplots(3,3,figsize=(12,8),sharex=True,layout='constrained')
        for stage,name in enumerate(('早期视觉','形状整合','额区响应')):
            # Overlaying both mirrored preference populations hides the effect
            # through exact superposition. Compare the SAME population instead.
            for group,style,gname in [(0,'-','左形状偏好群')]:
                for cond,color in enumerate((BLUE,RED)):
                    axes[stage,0].plot(t,s[cond,3+stage*3+group],color=color,ls=style,lw=1)
                    axes[stage,1].plot(t,s[cond,12+stage*3+group],color=color,ls=style,lw=1)
                    axes[stage,2].plot(t,q[cond,stage,group],color=color,ls=style,lw=1,
                                       label=f'{("左三角","右三角")[cond]} / {gname}')
            for col,ct in enumerate(('兴奋群 E','抑制群 I','突触等效源 q')):
                axes[stage,col].set_title(name+' '+ct);axes[stage,col].set_xlim(-100,1100)
                axes[stage,col].axvspan(0,203.125,color=GREY,alpha=.1)
                axes[stage,col].set_xlabel('相对提示时间（ms）');axes[stage,col].set_ylabel('模型响应（无量纲）')
        axes[0,2].legend(fontsize=6,ncol=2)
        fig.suptitle(f'图2  {key}组皮层形状选择响应',fontsize=12)
        save(fig,out,'图2_皮层内部响应'+suffix)
        # Dedicated LGN inset supplement keeps main figure readable.
        fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
        for cond,color in enumerate((BLUE,RED)):
            for g,style in enumerate(('-','--',':')):
                ax.plot(t,s[cond,g],color=color,ls=style,label=f'{("左三角","右三角")[cond]} 通路{g+1}')
        ax.set(xlim=(-100,600),xlabel='相对提示时间（ms）',ylabel='模型响应（无量纲）',title=f'{key}组 LGN中继响应');ax.legend(ncol=3)
        save(fig,out,f'附图_LGN_{key}')
        fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True,layout='constrained')
        pairs=[(w[f'{key}_strict_all_observed'],w[f'{key}_strict_all_predicted'],'同样本描述拟合'),
               (w[f'{key}_full_observed'].mean(0),w[f'{key}_full_predicted'].mean(0),'五个留出块等权汇总')]
        for row,(obs,pred,title) in enumerate(pairs):
            for ch in range(3):
                ax=axes[row,ch]
                for cond,color in enumerate((BLUE,RED)):
                    ax.plot(ms,obs[cond,ch],color=color,label=f'{("左三角","右三角")[cond]} 实测')
                    ax.plot(ms,pred[cond,ch],color=color,ls='--',label=f'{("左三角","右三角")[cond]} 预测')
                decorate(ax);ax.set_title(f'{title} {CHANNELS[ch]}')
        axes[0,0].legend(ncol=2);fig.suptitle(f'图3  {key}组左右三角ERP拟合与留出预测',fontsize=12)
        save(fig,out,'图3_ERP拟合与留出'+suffix)
        # Q1 dual-track supplementary figure.
        fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
        for row,stage in enumerate(('q1_before','q1_v7')):
            for ch in range(3):
                ax=axes[row,ch]
                for cond,color in enumerate((BLUE,RED)):
                    ax.plot(ms,w[f'{key}_{stage}_observed'][cond,ch],color=color)
                    ax.plot(ms,w[f'{key}_{stage}_predicted'][cond,ch],color=color,ls='--')
                decorate(ax);ax.set_title(f'{"降噪前" if stage=="q1_before" else "降噪后"} {CHANNELS[ch]}')
        fig.suptitle(f'{key}组问题一与问题二输入衔接')
        save(fig,out,f'附图_Q1衔接_{key}')
    # Figure 4: all recordings and all channels, not selected successes.
    fig,axes=plt.subplots(4,3,figsize=(12,10),sharex=True,layout='constrained')
    for r,key in enumerate(NAMES):
        obs=w[f'{key}_full_observed'].mean(0);pred=w[f'{key}_full_predicted'].mean(0)
        ci=w[f'{key}_delta_ci'];score=summary[(summary.dataset==key)&(summary.model=='full')].iloc[0]
        for ch in range(3):
            ax=axes[r,ch];ax.fill_between(ms,ci[0,ch],ci[1,ch],color=BLUE,alpha=.17,label='实测点态95%区间')
            ax.plot(ms,obs[1,ch]-obs[0,ch],color=BLUE,label='实测：右−左')
            ax.plot(ms,pred[1,ch]-pred[0,ch],color=RED,ls='--',label='预测：右−左')
            decorate(ax);ax.set_title(f'{key}组 {CHANNELS[ch]}'+(f'（SΔ={score.S_delta:+.3f}）' if ch==0 else ''))
    axes[0,0].legend(fontsize=7);fig.suptitle('图4  左右三角ERP差异波的留出预测',fontsize=12)
    save(fig,out,'图4_左右差异波')
    # Figure 5: descriptive LI, unnormalized lateral difference, and trial distributions.
    fig,axes=plt.subplots(4,3,figsize=(12,10),layout='constrained');lirows=[]
    for row,key in enumerate(NAMES):
        obs=w[f'{key}_full_observed'].mean(0);pred=w[f'{key}_full_predicted'].mean(0)
        x=data[f'{key}_x'];y=data[f'{key}_y'];mask=(TIMES>=.25)&(TIMES<.5)
        amp=x[...,mask].mean(-1);floor=max(np.quantile(abs(amp[:,1])+abs(amp[:,2]),.1),1e-8)
        trial_li=(amp[:,2]-amp[:,1])/np.maximum(abs(amp[:,1])+abs(amp[:,2]),floor)
        for cond,color in enumerate((BLUE,RED)):
            for waves,style,label in ((obs,'-','实测'),(pred,'--','预测')):
                axes[row,0].plot(ms,waves[cond,2]-waves[cond,1],color=color,ls=style,label=f'{("左","右")[cond]} {label}')
                a=waves[cond];li=(a[2]-a[1])/np.maximum(abs(a[1])+abs(a[2]),floor)
                axes[row,1].plot(ms,li,color=color,ls=style)
            chosen=y==(-1 if cond==0 else 1)
            lirows.append(dict(dataset=key,direction='left' if cond==0 else 'right',n=int(chosen.sum()),
                               LI_mean=trial_li[chosen].mean(),LI_SD=trial_li[chosen].std(ddof=1),
                               LI_floor=floor,amplitude_F4_minus_F3=(amp[chosen,2]-amp[chosen,1]).mean(),
                               role='descriptive_all_data'))
        decorate(axes[row,0]);decorate(axes[row,1],False);axes[row,1].set_ylabel('侧化指数')
        axes[row,0].set_title(f'{key}组 F4−F3');axes[row,1].set_title(f'{key}组 侧化指数')
        axes[row,2].boxplot([trial_li[y==-1],trial_li[y==1]],tick_labels=['左','右'],showfliers=False)
        axes[row,2].set_title(f'{key}组 P300窗侧化指数');axes[row,2].axhline(0,color=GREY,lw=.5)
    axes[0,0].legend(ncol=2);fig.suptitle('图5  额区侧化特征的实测与预测',fontsize=12)
    save(fig,out,'图5_空间侧化')
    pd.DataFrame(lirows).to_csv(out/'Q2_侧化统计.csv',index=False,encoding='utf-8-sig')
    # Figure 6: both re-fitted ablations and fixed interventions.
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    model_order=list(MODEL_NAMES)
    for j,key in enumerate(NAMES):
        vals=summary[summary.dataset==key].set_index('model').loc[model_order]
        offset=(j-1.5)*.18
        axes[0,0].bar(np.arange(len(model_order))+offset,vals.S_delta,width=.18,label=key)
    axes[0,0].set_xticks(np.arange(len(model_order)),[MODEL_NAMES[m] for m in model_order],rotation=30,ha='right')
    axes[0,0].set_title('各模型留出预测增益');axes[0,0].set_ylabel('差异波预测增益 $S_\\Delta$');axes[0,0].axhline(0,color=GREY,lw=.8);axes[0,0].legend(ncol=4)
    fixed=pd.read_csv(out/'Q2_固定参数消融.csv')
    agg=fixed.groupby(['dataset','intervention'])[['delta_MSE','full_delta_MSE']].mean()
    for j,ab in enumerate(('no_shape','no_recurrence','symmetric_readout')):
        val=agg.xs(ab,level=1);relative=(val.delta_MSE-val.full_delta_MSE)/val.full_delta_MSE
        axes[0,1].bar(np.arange(4)+(j-1)*.24,relative,width=.24,label=MODEL_NAMES[ab])
    axes[0,1].set_xticks(range(4),NAMES);axes[0,1].set_title('机制消融的误差变化');axes[0,1].set_ylabel('差异波MSE相对变化');axes[0,1].axhline(0,color=GREY,lw=.8);axes[0,1].legend(fontsize=7)
    sensitivity=pd.read_csv(out/'Q2_参数敏感性.csv')
    for j,p in enumerate(('time_scale','recurrence','F4_readout')):
        s=sensitivity[sensitivity.parameter==p].groupby('factor').S_delta.mean()
        axes[1,0].plot(s.index,s.values,marker='o',label=p)
    axes[1,0].set(title='参数敏感性',xlabel='参数倍数',ylabel='差异波预测增益 $S_\\Delta$');axes[1,0].legend()
    vals=summary[summary.model=='full'];axes[1,1].errorbar(np.arange(4),vals.S_delta,
        yerr=np.array([np.maximum(vals.S_delta-vals.S_low,0),np.maximum(vals.S_high-vals.S_delta,0)]),fmt='o',color=BLUE,capsize=4)
    axes[1,1].set_xticks(range(4),vals.dataset);axes[1,1].axhline(0,color=GREY,lw=.8)
    axes[1,1].set_title('完整模型的95%重采样区间');axes[1,1].set_ylabel('差异波预测增益 $S_\\Delta$')
    fig.suptitle('图6  模型消融与参数敏感性',fontsize=12)
    save(fig,out,'图6_消融与敏感性')
    # Figure 7: training-fitted PCA + out-of-fold diagnostics and actual null.
    pca=pd.read_csv(out/'Q2_训练参考PCA.csv');null=pd.read_csv(out/'Q2_置换分布.csv')
    for key in NAMES:
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');points=pca[pca.dataset==key]
        for direction,color in ((-1,BLUE),(1,RED)):
            for role,marker in (('train','.'),('test','^')):
                p=points[(points.cue==direction)&(points.role==role)]
                axes[0,0].scatter(p.PC1,p.PC2,c=color,marker=marker,s=22 if role=='train' else 45,
                                   alpha=.45 if role=='train' else .95,label=f'{("左三角" if direction==-1 else "右三角")} {("训练" if role=="train" else "测试")}')
        axes[0,0].legend(ncol=2,fontsize=7);axes[0,0].set(title='主成分投影（第5块留出）',xlabel='主成分1',ylabel='主成分2')
        m=metric[metric.dataset==key].set_index('features').loc[list(FEATURE_NAMES)]
        axes[0,1].errorbar(np.arange(len(m)),m.BA,yerr=[np.maximum(m.BA-m.BA_low,0),np.maximum(m.BA_high-m.BA,0)],fmt='o',capsize=3)
        axes[0,1].axhline(.5,color=GREY,ls='--');axes[0,1].set_ylim(0,1)
        axes[0,1].set_xticks(range(len(m)),[FEATURE_NAMES[x] for x in m.index],rotation=35,ha='right')
        axes[0,1].set(title='各特征的留出判别',ylabel='平衡准确率')
        row=m.loc['mechanism'];cm=np.array([[row.TN,row.FP],[row.FN,row.TP]])
        axes[1,0].imshow(cm,cmap='Blues');axes[1,0].set_xticks([0,1],['预测左','预测右']);axes[1,0].set_yticks([0,1],['实际左','实际右'])
        for i in range(2):
            for j in range(2): axes[1,0].text(j,i,str(int(cm[i,j])),ha='center',va='center',fontsize=15,
                                            color='white' if cm[i,j]>(cm.max()+cm.min())/2 else 'black')
        axes[1,0].grid(False);axes[1,0].set_title('机制特征混淆矩阵')
        n=null[(null.dataset==key)&(null.features=='mechanism')&(null.kind=='block_permutation')]
        axes[1,1].hist(n.BA,bins=20,color=GREY,alpha=.65);axes[1,1].axvline(row.BA,color=RED,lw=2)
        axes[1,1].set(title=f'块内置换检验（校正$p$={row.block_permutation_maxT_p:.3f}）',xlabel='置换平衡准确率',ylabel='频数')
        fig.suptitle(f'图7  {key}组左右三角判别特征',fontsize=12)
        save(fig,out,'图7_特征与判别'+('' if key=='A1' else f'_补图_{key}'))
    # Pooled summary, still retaining per-recording main figures and metrics.
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for j,key in enumerate((*NAMES,'pooled')):
        sub=metric[(metric.dataset==key)&metric.features.isin(['amplitude','mechanism','projection6','direction3','past_only'])]
        ax.plot(range(len(sub)),sub.BA,marker='o',label=key)
    ax.set_xticks(range(len(sub)),[FEATURE_NAMES[s] for s in sub.features]);ax.axhline(.5,color=GREY,ls='--')
    ax.set(ylabel='平衡准确率',title='左右三角留出判别汇总');ax.legend(ncol=5)
    save(fig,out,'附图_判别汇总')
    w.close();data.close();shape.close()


def md_table(frame,columns):
    f=frame[columns].copy()
    def cell(x):
        if isinstance(x,(float,np.floating)):return '—' if not np.isfinite(x) else f'{x:.4f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                      ['| '+' | '.join(cell(x) for x in row)+' |' for row in f.itertuples(index=False,name=None)])


def write_paper(out):
    out=Path(out);manifest=json.loads((out/'Q2_运行清单.json').read_text(encoding='utf8'))
    summary=pd.read_csv(out/'Q2_机制汇总.csv');metric=pd.read_csv(out/'Q2_表4_特征判别.csv')
    full=summary[summary.model=='full'];pooled=metric[metric.dataset=='pooled'].set_index('features')
    row=pooled.loc['mechanism'];res=pooled.loc['post_given_past'];n=sum(manifest['counts'].values())
    count=int(sum(full.S_delta>0));sig=row.block_permutation_maxT_p<.05
    gain=pd.read_csv(out/'Q2_判别增量区间.csv')
    incremental=gain[(gain.dataset=='pooled')&(gain.features=='mechanism_minus_spatial')].iloc[0]
    support='部分支持' if count else '尚未支持'
    decoder_support='部分支持' if sig and row.BA_low>.5 else '尚未支持'
    conclusions=[
        ('形状产生不同输入','模型内支持','图1及形状响应表','镜像输入交换形状选择群响应；并非实测皮层放电。'),
        ('群体活动形成头皮观测','构建完成','图1–3及参数表','同一观测矩阵作用于两种输入；矩阵不是个体解剖导联。'),
        ('重现实测方向差异',support,'图3–4及机制汇总',f'四组中{count}组留出 SΔ>0；具体幅度与区间必须逐组报告。'),
        ('关键机制贡献','模型内支持；生理归因未确认','图6、固定消融及重拟合对照','形状取消使差异消失是结构性检验；真实来源需优于替代解释。'),
        ('稳定特征区分能力',decoder_support,'图7及表4',f'机制特征 BA={row.BA:.3f}，AUC={row.AUC:.3f}，七组合校正 p={row.block_permutation_maxT_p:.4f}。')]
    pd.DataFrame(conclusions,columns=['命题','判定','证据','可写结论']).to_csv(out/'Q2_结论证据对照.csv',index=False,encoding='utf-8-sig')
    lead=(f'本文构建图像驱动的三级兴奋—抑制群体模型，将三角形边缘的空间组合、LGN中继、皮层动态和额区头皮观测连接起来。'
          f'四组原始记录经独立时间块处理后保留{n}次试验；方向差分预测在{count}/4组优于零差基线。'
          f'23维机制增强特征的合并留出平衡准确率为{row.BA:.1%}（时间块重采样95%区间{row.BA_low:.1%}–{row.BA_high:.1%}），'
          f'AUC为{row.AUC:.3f}，块内置换七特征组合maxT校正p={row.block_permutation_maxT_p:.4f}。'
          '这些结果评估的是候选机制的解释与预测能力，不构成真实神经源的唯一识别。')
    equation=r'''
\[
f_j(I)=\exp\left[\frac1K\sum_{k=1}^{K}\log\left(\max_{\delta\in\mathcal N}r_{\theta_{jk}}(p_{jk}+\delta;I)+\epsilon\right)\right],\quad j\in\{L,R\}.
\]
\[
\tau_g\dot g=-g+3f(I)\mathbf1_{0\le t<T_s},\quad
\tau_{E,s}\dot E_s=-E_s+\sigma(E_s-I_s+\rho WE_s+d_s-2),\quad
\tau_{I,s}\dot I_s=-I_s+\sigma(E_s-I_s-2),
\]
其中 \(d_1=g\)、\(d_s=3(E_{s-1}-E_{s-1,0})\)，\(W\) 的对角元为0、非对角元为1/2；所有方向共享参数。
\[
\tau_{e,s}\dot u_{e,s}=E_s-E_{s,0}-u_{e,s},\quad
\tau_{e,s}\dot v_{e,s}=u_{e,s}-v_{e,s},\quad q_s=v_{e,s}-0.7v_{i,s},\qquad \hat y_d=L\tilde q_d.
\]
抑制突触使用同构方程。\(\tilde q\) 是在256 Hz采样率下施加与EEG一致的观测滤波和基线处理的源；神经状态本身保持因果，零相位观测滤波可产生刺激前延展。
\[
\hat L=\arg\min_L\left\{\frac12\|C_y-LC_q\|_W^2+\frac12\|D_y-LD_q\|_W^2+\lambda\|L\|_F^2\right\},\quad
C_y=(y_R+y_L)/2,\ D_y=y_R-y_L.
\]
上式范数按时间窗加权；各源列在共同与差分联合设计中归一化，避免尺度改变正则含义。
\[
S_\Delta=1-\frac{\operatorname{MSE}(D^{obs},D^{pred})}{\operatorname{MSE}(D^{obs},0)},\qquad
R^2=1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}.
\]
\[
LI=\frac{A_{F4}-A_{F3}}{\max(|A_{F4}|+|A_{F3}|,\epsilon_{train})},\qquad
\phi=[A_{1:3},P_{1:3},T_{1:3},M_{1:3},LI,A_{F4}-A_{F3},b_{1:6},e_{1:3}]\in\mathbb R^{23}.
\]
'''
    sections=[('# 第二问 图像驱动的脑电形成模型与视觉响应表示',lead),
      ('## 数据与验证设计',f'数据为两名受试者、两个项目的四份记录，采样率256 Hz，实际EEG通道顺序为Fz/F3/F4。左、右指三角朝向，不推定左右视野。'
       f'严格主分析各组保留数量为{manifest["counts"]}。每份记录按原始事件划分五个连续块，块内单独滤波，边界保护24 s，'
       '截取−250至约800 ms，使用刺激前中位数作基线。三窗50–250、250–500、500–750 ms在机制损失中等权。'
       '外层测试块不参与模型选择，内层再留块选择时间尺度、复发强度和岭正则。'
       '内层选择JSON中的损失省略了所有候选共有的目标能量常数，因而可能为负；这些数值只用于候选排序，不是负的均方误差。'
       'Q1 before/v7仅用于已知条件描述性衔接；V7方向知情模板不进入主判别。幅值使用原始记录单位，未假定已标定为微伏。'),
      ('## 形状编码与群体模型',
       '采用COSFIRE思想的简化边缘空间组合编码[1]，从题目示意图提取三角并镜像，建立两个形状偏好模板与共同注视输入。'
       '缺角与位置打乱是计算机反事实刺激，不是新增实验。三级群体动力学采用Wilson–Cowan型E/I方程[2]，'
       '分别表示早期视觉、形状整合和额区响应，每级三个群体。LGN状态表示中继信号在预设空间通路上的投影，'
       '不声称LGN细胞本身完成三角形识别；形状偏好由后续皮层输入的空间组合权重表示。采用突触等效源而非直接把放电率当成EEG；'
       '源与头皮观测的关系参考EEG计算模型综述[3]。所有常数、拓扑和阶段名称均为候选模型假设。\n'+equation),
      ('## 机制拟合与实测差异',md_table(full,['dataset','S_delta','S_low','S_high','delta_RMSE','R2','Corr'])+'\n\n'
       'SΔ的零基线对应左右无差异预测，与去均值定义的R²不同。总ERP的良好拟合不能替代方向差分的留出检验。'
       '表内区间重采样已有五个时间块，属于当前记录的描述性范围，不是五项独立研究或人群置信区间。'
       '完整模型、共同响应、六时间核、平滑阶跃和慢斜坡使用同样划分；其可拟合参数数目不同，比较解释为留出预测比较，不能称为严格等自由度机制鉴别。'),
      ('## 消融与参数敏感性',
       '固定参数消融保持训练得到的观测矩阵与源归一化尺度，分别取消形状选择性、复发连接或头皮左右投影差异。'
       '其中取消复发指令同级跨群体耦合系数ρ=0，局部E/I反馈仍保留。重新拟合消融则在同一内外层划分中重新选择其余参数，检验剩余结构是否足以补偿。'
       '取消形状选择后左右模拟相同、对称观测后F3/F4一致，是实现必须满足的结构性约束；并不因此证明实测差异必由此机制产生。'
       '对时间尺度、复发强度及F4观测行做±20%扰动，报告留出误差变化和静息稳定性。复发强度选为0时，乘法扰动保持0，不能解释为已证明低敏感性。'),
      ('## 特征表示及有效性',
       '基础特征从250–500 ms提取三通道均值、有效正峰幅度和潜伏期。正峰要求为窗内局部极大值、幅度大于0且显著度至少为刺激前标准差的一半；'
       '无峰记缺失，训练中位数补缺并保留缺失标记。侧化指数使用训练分母10%分位数稳定。'
       '机制特征把未知试次投影到训练拟合的共同和方向差分模板，保存各通道两项投影系数及残差RMS，共23维。'
       '新增六维投影和三维方向投影固定对照，均仅由训练折建立模板；它们与原五组特征一起接受maxT校正，未按外层成绩挑选表示。它不需要未知试次的真实方向。模型时间窗特征不等于已确认的P300来源。\n\n'+
       md_table(metric[metric.dataset=='pooled'],['features','BA','BA_low','BA_high','AUC','F1','block_permutation_p','block_permutation_maxT_p'])+
       f'\n\n加入机制相对空间特征的BA差为{row.BA-pooled.loc["spatial"].BA:+.1%}，成对时间块重采样区间为'
       f'[{incremental.BA_low:+.1%},{incremental.BA_high:+.1%}]。扣除刺激前可线性预测部分后，BA为{res.BA:.1%}、AUC为{res.AUC:.3f}。'
       'Precision、Recall和F1以右刺激为正类。PCA只用固定前四块拟合，不能据散点分离程度作显著性判断。'),
      ('## 统计检验与迁移',
       f'执行{manifest["permutations"]}次“记录×时间块”内标签置换；每次重新选择机制参数、拟合模板与分类器。'
       f'七项特征组合的BA采用maxT校正，并补充{manifest["circular_shifts"]}次记录内非零循环移位。'
       '循环移位后若某留出块只剩一类，按预先规定的可估计条件重新抽取，故该检验为条件随机化敏感性分析。'
       '判别区间对已经生成的折外预测进行时间块重采样，没有在每次重采样中重新训练；它不包含完整的模型训练不确定性。'
       'maxT覆盖本次七个特征组合，不覆盖此前反复开发过的全部模型，也不自动校正跨数据组选择。'
       '另提供同项目A→B与B→A迁移，不使用测试受试者的均值或尺度拟合标准化。仅两名受试者，迁移结果不能外推为人群泛化。\n\n'+
       md_table(pd.read_csv(out/'Q2_跨受试者迁移.csv').query("features=='mechanism'"),['train','test','BA','AUC','F1'])),
      ('## 五项命题的证据判定',md_table(pd.DataFrame(conclusions,columns=['命题','判定','证据','可写结论']),['命题','判定','证据','可写结论'])+
       '\n\n三路额区EEG、未知真实头模型和缺少同步眼位/EOG，使视觉源、眼动和任务准备的贡献不能唯一分解。'
       '本研究的贡献是可计算、可消融、可验证的候选形成模型与明确的试次特征；证据不足的命题保留其不确定性。'),
      ('## 复现',
       '完整运行：`python -B q2/main.py`。重绘及生成论文文字：`python -B q2/main.py --redraw`。'
       '快速检查使用`--quick`，不可将其低置换次数结果替代正式分析。全部新文件直接位于q2_result根目录。'
       'Q2_运行清单.json保存版本与哈希，Q2_结果复核.json保存从逐试次输出重算指标的核验结果。'),
      ('## 参考文献',
       '[1] Azzopardi G, Petkov N. Ventral-stream-like shape representation: from pixel intensity values to trainable object-selective COSFIRE models. Front Comput Neurosci, 2014, 8:80. https://doi.org/10.3389/fncom.2014.00080\n\n'
       '[2] Wilson HR, Cowan JD. Excitatory and inhibitory interactions in localized populations of model neurons. Biophys J, 1972, 12:1–24. https://doi.org/10.1016/S0006-3495(72)86068-5\n\n'
       '[3] Glomb K, et al. Computational Models in Electroencephalography. Brain Topogr, 2022, 35:142–161. https://doi.org/10.1007/s10548-021-00828-2')]
    text='\n\n'.join(h+'\n\n'+body for h,body in sections)
    if manifest['quick']: text='> 当前为快速检查结果，正式论文应先执行默认完整运行。\n\n'+text
    captions=[
      ('图1_机制与形状编码','图1 图像驱动的脑电形成路径。上排为题图三角及构造的反事实形状，数值为模型输入响应；下排为共享动力学和观测关系。'),
      ('图2_皮层内部响应','图2 A1描述标定模型中，同一个左形状偏好群在左右刺激下的三级E/I及突触响应。右形状偏好群呈镜像互换；完整九群体状态存入NPZ，曲线均为模型内部量。'),
      ('图3_ERP拟合与留出','图3 A1左右ERP。上排为同样本描述拟合，下排为五个测试块预测与实测均值的等权汇总；实线为实测、虚线为模型。'),
      ('图4_左右差异波','图4 四组右减左差分留出预测。阴影为按块并在块内按方向重采样的实测点态95%区间，不是同时置信带或模型预测区间。'),
      ('图5_空间侧化','图5 F4−F3和稳定化LI。时变曲线比较实测与模型，箱线图为试次窗口LI；图中分母稳定项采用描述性全数据值，分类使用训练折值。'),
      ('图6_消融与敏感性','图6 重拟合消融、固定参数干预、参数扰动与差分预测收益。保留负收益，干预效应只作为候选模型内证据。'),
      ('图7_特征与判别','图7 A1训练参考PCA、留出BA、机制特征混淆矩阵和置换分布。PCA训练与测试点使用不同标记，显著性来自完整验证流程。')]
    parameter_table=pd.read_csv(out/'Q2_表1_模型参数.csv')
    feature_table=pd.read_csv(out/'Q2_表3_特征定义.csv')
    erp_table=pd.read_csv(out/'Q2_表2_ERP拟合与预测.csv')
    erp_compact=erp_table.query("model=='full' and stage=='strict_oof'").groupby(
        ['dataset','condition','channel'],as_index=False)[['RMSE','R2','Corr']].mean()
    pd.DataFrame(erp_compact).to_csv(out/'Q2_表2_正文汇总.csv',index=False,encoding='utf-8-sig')
    matrices=[]
    with np.load(out/'Q2_机制波形.npz') as stored:
        for key in NAMES:
            for fold in range(-1,5):
                coef=stored[f'{key}_full_coef' if fold==-1 else f'{key}_coef_{fold}']
                for source in range(9):
                    for channel in range(3):
                        matrices.append(dict(dataset=key,held_block=fold,stage=source//3+1,
                            population=['left_preference','right_preference','common'][source%3],
                            channel=CHANNELS[channel],normalized_source_coefficient=coef[source,channel],
                            role='all_data_descriptive' if fold==-1 else 'outer_training_only'))
    pd.DataFrame(matrices).to_csv(out/'Q2_有效观测矩阵.csv',index=False,encoding='utf-8-sig')
    text+='\n\n## 四类论文表\n\n'
    text+='### 表1 模型参数与来源\n\n'+md_table(parameter_table,list(parameter_table.columns))
    text+='\n\n### 表2 完整模型逐方向逐通道留出结果\n\n各指标为五个留出块的算术平均；RMSE、R²、相关的平均值与先合并波形后计算不同。\n\n'+md_table(erp_compact,list(erp_compact.columns))
    text+='\n\n### 表3 特征定义\n\n'+md_table(feature_table,list(feature_table.columns))
    text+='\n\n### 表4 特征判别\n\n合并比较见上文；逐组完整指标、校正p值和区间见[表4 CSV](Q2_表4_特征判别.csv)。'
    text+='\n\n## 正文图与图注\n\n'+'\n\n'.join(f'![{cap}](Q2_{name}.png)\n\n{cap}' for name,cap in captions)
    (out/'Q2_论文正文.md').write_text(text,encoding='utf8')
    (out/'Q2_图注与使用说明.md').write_text('\n\n'.join(cap for _,cap in captions)+
       '\n\nA1固定为正文展示组；A2/B1/B2采用相同补图规格。表1–4对应同名CSV；Q1衔接图为描述性附图。',encoding='utf8')
    # Self-contained Chinese LaTeX manuscript, with table handling independent of pandoc.
    def esc(s):
        return str(s).replace('\\','\\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
    latex=[r'\documentclass[UTF8,a4paper,11pt]{ctexart}',r'\usepackage[margin=2cm]{geometry}',
           r'\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,hyperref}',r'\begin{document}',
           r'\title{第二问 图像驱动的脑电形成模型与视觉响应表示}\author{}\date{}\maketitle',esc(lead)]
    for heading,body in sections[1:-1]:
        latex.append(r'\section{'+esc(heading.replace('## ',''))+'}')
        if heading=='## 形状编码与群体模型':
            intro=body.split('\\[')[0];latex.append(esc(intro))
            latex.append(equation.replace('\\(','$').replace('\\)','$').replace('%',r'\%'))
        else:
            # Emit narrative paragraphs and compact numeric tables separately.
            plain='\n'.join(line for line in body.splitlines() if not line.startswith('|'))
            latex.append(esc(plain).replace('`',''))
    latex.append(r'\section{核心结果表}')
    for df,cols,title in [(full,['dataset','S_delta','S_low','S_high'],'机制留出差分收益'),
                          (metric[metric.dataset=='pooled'],['features','BA','AUC','block_permutation_maxT_p'],'合并特征判别')]:
        latex.extend([r'\begin{longtable}{'+'l'*len(cols)+'}',r'\caption{'+title+r'}\\\toprule',
                      ' & '.join(esc(c) for c in cols)+r'\\\midrule'])
        for vals in df[cols].itertuples(index=False,name=None):
            latex.append(' & '.join(esc(f'{v:.4f}' if isinstance(v,float) and np.isfinite(v) else ('—' if isinstance(v,float) else v)) for v in vals)+r'\\')
        latex.append(r'\bottomrule\end{longtable}')
    # Include all four requested table classes with wrapping columns.
    for title,df,widths in [('模型参数与来源',parameter_table[['parameter','value','source']],['.26','.43','.22']),
                           ('逐条件逐通道留出指标 五块算术平均',erp_compact,['.12','.12','.12','.16','.16','.16']),
                           ('特征定义',feature_table[['feature','dimension','meaning']],['.30','.10','.51'])]:
        latex.extend([r'\small\begin{longtable}{'+''.join('p{'+v+r'\linewidth}' for v in widths)+'}',
                      r'\caption{'+title+r'}\\\toprule',
                      ' & '.join(esc(c) for c in df.columns)+r'\\\midrule\endfirsthead',
                      ' & '.join(esc(c) for c in df.columns)+r'\\\midrule\endhead'])
        for vals in df.itertuples(index=False,name=None):
            latex.append(' & '.join(esc(f'{v:.4f}' if isinstance(v,float) and np.isfinite(v) else v) for v in vals)+r'\\')
        latex.append(r'\bottomrule\end{longtable}\normalsize')
    for name,cap in captions:
        latex.extend([r'\begin{figure}[p]\centering',r'\includegraphics[width=\linewidth,height=.8\textheight,keepaspectratio]{Q2_'+name+'.pdf}',
                      r'\caption{'+esc(cap)+r'}\end{figure}'])
    latex.extend([r'\nocite{azzopardi2014,wilson1972,glomb2022}',r'\bibliographystyle{plain}\bibliography{Q2_参考文献}',r'\end{document}'])
    (out/'Q2_论文正文.tex').write_text('\n\n'.join(latex),encoding='utf8')
    bib='''@article{azzopardi2014, author={Azzopardi, George and Petkov, Nicolai}, title={Ventral-stream-like shape representation: from pixel intensity values to trainable object-selective COSFIRE models}, journal={Frontiers in Computational Neuroscience}, year={2014}, volume={8}, pages={80}, doi={10.3389/fncom.2014.00080}}
@article{wilson1972, author={Wilson, Hugh R. and Cowan, Jack D.}, title={Excitatory and inhibitory interactions in localized populations of model neurons}, journal={Biophysical Journal}, year={1972}, volume={12}, pages={1--24}, doi={10.1016/S0006-3495(72)86068-5}}
@article{glomb2022, author={Glomb, Katharina and Cabral, Joana and Cattani, Anna and Mazzoni, Alberto and Raj, Ashish and Franceschiello, Benedetta}, title={Computational Models in Electroencephalography}, journal={Brain Topography}, year={2022}, volume={35}, pages={142--161}, doi={10.1007/s10548-021-00828-2}}
'''
    (out/'Q2_参考文献.bib').write_text(bib,encoding='utf8')
    files=sorted(p.name for p in out.glob('Q2_*') if p.is_file())
    (out/'Q2_结果索引.md').write_text('# 第二问结果索引\n\n先读[论文正文](Q2_论文正文.md)及[结论证据对照](Q2_结论证据对照.csv)。\n\n'+
       '\n'.join(f'- [{f}]({f})' for f in files),encoding='utf8')
