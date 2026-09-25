#!/usr/bin/env python3
"""Audit numerical outputs and render the paired Q2 denoising comparison."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/q2-fold-report-mpl')
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from q2model.data import sha, save_csv, save_json, TIMES, NAMES
from q2model.model import Fit
from q2model.validation import metrics
from q2model.fold_denoising import q1_module

ROOT = Path(__file__).resolve().parents[1]
NAMES_SHORT = {'A1':'A项目一', 'A2':'A项目二', 'B1':'B项目一', 'B2':'B项目二'}
COLORS = {'baseline':'#2563A6', 'denoised':'#D97732'}
LABELS = {'baseline':'基础预处理', 'denoised':'训练折内新版去噪'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'Q2_fold_denoising_results')
    args = parser.parse_args(); out = args.output.resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['status'] != 'comparison_complete':
        raise ValueError('Numerical comparison is incomplete')
    for field, base in [('source', ROOT), ('inputs', ROOT), ('output_hashes', out)]:
        for name, expected in manifest[field].items():
            if sha(base/name) != expected:
                raise ValueError('Hash mismatch: '+name)
    pred = pd.read_csv(out/'predictions.csv')
    summary = pd.read_csv(out/'classification_summary.csv')
    changes = pd.read_csv(out/'paired_changes.csv')
    waves = pd.read_csv(out/'waveform_metrics.csv')
    feature = pd.read_csv(out/'feature_stability.csv')
    checks = []
    for row in summary.itertuples():
        g = pred[(pred.dataset == row.dataset)&(pred.branch == row.branch)&(pred.model == row.model)]
        assert g.trial_id.nunique() == len(g)
        result = metrics(g.truth.to_numpy(), g.prediction.to_numpy(), g.score.to_numpy())
        for m in ('BA', 'AUC', 'n', 'TN', 'FP', 'FN', 'TP'):
            assert abs(result[m]-getattr(row, m)) < 1e-12
    assembled = {}
    for key in manifest['config']['datasets']:
        bs = pred.query('dataset == @key and branch == "baseline" and model == "mechanism_selected"').sort_values('trial_id')
        dn = pred.query('dataset == @key and branch == "denoised" and model == "mechanism_selected"').sort_values('trial_id')
        np.testing.assert_array_equal(bs[['trial_id','fold','truth']], dn[['trial_id','fold','truth']])
        data = []
        for block in sorted(bs.fold.unique()):
            for branch in ('baseline', 'denoised'):
                path = out/'folds'/key/branch/f'fold_{block}'
                record = json.loads(path.with_suffix('.json').read_text())
                w = dict(np.load(path.with_suffix('.npz')))
                train, test = set(record['train_ids']), set(record['test_ids'])
                assert not train & test
                assert all((i-1)//20 == block for i in test)
                assert all((i-1)//20 != block for i in train)
                for r in record['fit']['diagnostics']['inner_validation']:
                    a,b = set(r['train_ids']),set(r['validation_ids'])
                    assert not a & b and a | b == train and not (a | b) & test
                for r in record['inner_denoisers']:
                    d = r['denoiser']; a=set(d['train_ids']); b=set(r['validation_ids'])
                    assert not a & b and a | b == train and not a & test
                    assert set(d['reference_ids']) <= a
                    assert not set(d['selector_train_ids']) & set(d['selector_validation_ids'])
                    assert set(d['selector_train_ids']) | set(d['selector_validation_ids']) == a
                if record['denoiser']:
                    d = record['denoiser']
                    assert set(d['train_ids']) == train and set(d['reference_ids']) <= train
                f = record['fit']
                fit = Fit(f['rank'],f['lambda_A'],w['theta'],w['A'],w['D'],w['h'],f['diagnostics'])
                np.testing.assert_allclose(fit.features(w['after']),w['features'],atol=1e-12)
                z = (w['features']-w['classifier_mean'])/w['classifier_scale']
                logit = (z@w['classifier_coef'].T+w['classifier_intercept']).ravel()
                probability = 1/(1+np.exp(-logit))
                np.testing.assert_allclose(probability,w['score'],atol=1e-12)
                saved = pred.query('dataset == @key and branch == @branch and fold == @block and model == "mechanism_selected"')
                np.testing.assert_array_equal(saved.trial_id,w['test_ids'])
                np.testing.assert_allclose(saved.score,w['score'],atol=1e-12)
                checks.append(dict(dataset=key,branch=branch,fold=int(block),isolation=True,prediction_recomputed=True))
                if branch == 'denoised':data.append(w)
        arrays = {name:np.concatenate([w[name] for w in data]) for name in ('test_ids','truth','before','after')}
        order = np.argsort(arrays['test_ids'])
        assembled[key] = {name:value[order] for name,value in arrays.items()}
    spatial=[]; policies=[]
    for key,w in assembled.items():
        spatial.extend(q1_module().evaluate_spatial_metrics(w['before'],w['after'],w['truth'],key))
    for path in (out/'folds').glob('*/denoised/fold_*.json'):
        r=json.loads(path.read_text())
        policies.append(dict(dataset=r['dataset'],fold=r['fold'],candidate=r['denoiser']['candidate'],
                             rank=r['fit']['rank'],bound_hits=len(r['fit']['diagnostics']['bound_hits'])))
    save_csv(spatial,out/'spatial_descriptive.csv');save_csv(sorted(policies,key=lambda r:(r['dataset'],r['fold'])),out/'selected_policies.csv')
    comparison=[]
    for key in manifest['config']['datasets']:
        s=summary.query('dataset == @key and model == "mechanism_selected"').set_index('branch')
        d=changes.query('dataset == @key and model == "mechanism_selected"').iloc[0]
        comparison.append(dict(dataset=key,n=int(s.loc['baseline','n']),
                               baseline_BA=s.loc['baseline','BA'],denoised_BA=s.loc['denoised','BA'],
                               BA_change_pp=100*d.BA_change,BA_low_pp=100*d.BA_change_low,BA_high_pp=100*d.BA_change_high,
                               baseline_AUC=s.loc['baseline','AUC'],denoised_AUC=s.loc['denoised','AUC']))
    save_csv(comparison,out/'comparison_table.csv')
    # Chart contract: static scientific comparison; same four cohorts and paired
    # time blocks; blue/orange plus square/circle encodings, chance/zero anchors.
    plt.rcParams.update({'font.sans-serif':['Noto Sans CJK SC','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11})
    keys=list(assembled); ys=np.arange(len(keys))
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,metric,title in zip(axes[:2],('BA','AUC'),('平衡准确率 BA','ROC AUC')):
        for branch,offset,marker in [('baseline',-.12,'s'),('denoised',.12,'o')]:
            s=summary.query('model == "mechanism_selected" and branch == @branch').set_index('dataset').loc[keys]
            ax.errorbar(s[metric],ys+offset,xerr=np.maximum(0,np.vstack([s[metric]-s[metric+'_low'],s[metric+'_high']-s[metric]])),
                        fmt=marker,color=COLORS[branch],label=LABELS[branch],capsize=3,ms=6)
        ax.axvline(.5,color='#888',ls=':',lw=1);ax.set_xlim(0,1);ax.set_yticks(ys,[NAMES_SHORT[k] for k in keys]);ax.invert_yaxis()
        ax.set_title(title);ax.grid(axis='x',alpha=.15)
    ax=axes[2];d=changes.query('model == "mechanism_selected"').set_index('dataset').loc[keys]
    for y,(key,row) in zip(ys,d.iterrows()):
        ax.plot([100*row.BA_change_low,100*row.BA_change_high],[y,y],color=COLORS['denoised'])
        ax.plot(100*row.BA_change,y,'o',color=COLORS['denoised'])
    ax.axvline(0,color='#777',ls=':');ax.set_yticks(ys,[NAMES_SHORT[k] for k in keys]);ax.invert_yaxis()
    ax.set_title('新版 − 基础预处理');ax.set_xlabel('BA差值（百分点）');ax.grid(axis='x',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=10,loc='outside lower center',ncol=2)
    fig.suptitle('第二问：训练折内去噪与基础预处理\n机制特征 + 相同LDA分类器；95%区间为固定模型的时间块配对重采样',fontsize=13)
    fig.savefig(out/'classification_comparison.png',dpi=160);fig.savefig(out/'classification_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    feature_summary=feature.groupby(['dataset','branch']).same_sign.mean()
    for j,(target,title) in enumerate([('common_baseline','共同目标：基础预处理测试ERP'),('branch_output','各自目标：本分支测试ERP')]):
        ax=axes[j]
        for branch,offset,marker in [('baseline',-.1,'s'),('denoised',.1,'o')]:
            s=waves.query('model == "mechanism" and target == @target and branch == @branch').groupby('dataset').contrast_RMSE.mean().reindex(keys)
            ax.plot(s,ys+offset,marker,color=COLORS[branch],label=LABELS[branch])
        limit=waves.query('model == "mechanism" and target == @target').groupby(['dataset','branch']).contrast_RMSE.mean().max()
        ax.set_yticks(ys,[NAMES_SHORT[k] for k in keys]);ax.invert_yaxis();ax.set_xlim(0,limit*1.12)
        ax.set_title(title,fontsize=11);ax.set_xlabel('左右差分预测RMSE（原始电位单位）');ax.grid(axis='x',alpha=.15)
    for branch,offset,marker in [('baseline',-.1,'s'),('denoised',.1,'o')]:
        vals=[feature_summary.loc[(k,branch)] for k in keys]
        axes[2].plot(vals,ys+offset,marker,color=COLORS[branch],label=LABELS[branch])
    axes[2].set_yticks(ys,[NAMES_SHORT[k] for k in keys]);axes[2].invert_yaxis();axes[2].set_xlim(0,1)
    axes[2].set_title('九个固定窗特征的方向稳定性',fontsize=11)
    axes[2].set_xlabel('训练/测试左右效应同号比例');axes[2].grid(axis='x',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=10,loc='outside lower center',ncol=2)
    fig.suptitle('第二问：波形预测与特征稳定性\n每组5折等权；同号比例为5折×9特征的描述量，不是分类准确率或信息保留率',fontsize=13)
    fig.savefig(out/'waveform_and_features.png',dpi=160);fig.savefig(out/'waveform_and_features.pdf');plt.close(fig)
    improvements=sum(r['BA_change_pp']>1e-10 for r in comparison)
    equals=sum(abs(r['BA_change_pp'])<=1e-10 for r in comparison)
    conclusion=(f'机制特征分类的平衡准确率在{len(comparison)}组中有{improvements}组提高、{equals}组持平、'
                f'{len(comparison)-improvements-equals}组下降。'
                '本次对照未支持将新版去噪替代基础预处理作为第二问主分析；保留新版为敏感性对照。')
    report=['# 第二问训练折内去噪对照实验', '', conclusion, '',
            '已完成四组相同试次、相同时间块上的基础预处理与新版去噪对照。新版模板、尺度、均值修正和候选选择均仅来自当前训练集；内层模型选型重新拟合去噪。测试标签只用于最终评分和描述。', '',
            '**方法与范围**', '',
            '输入从原始MAT的Fz/F3/F4重新构建，保留5个独立时间块、60 Hz陷波、0.1–30 Hz带通、基线校正和24秒边界缓冲。两分支共享试次剔除结果。没有读入第一问已经处理的NPZ。', '',
            '当前新版的候选选择仍使用训练方向标签，并以训练集内部的受控污染恢复损失选择候选，故称为“训练标签辅助选型、推断不使用标签”。本实验保持原候选评价和最终15%/25%混合规则，没有为提高本次分数额外优化。', '',
            '机制模型采用原有2/3核、三个正则强度、完整4点网格及多初值连续细化；同一规则在每个内层留块中选型。分类器固定为训练内标准化的收缩LDA。另保存固定九维时间窗特征对照。', '',
            '**机制特征分类结果**', '',
            '|数据组|共同试次数|基础BA|去噪BA|BA差值及95%区间（百分点）|基础AUC|去噪AUC|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparison:
        report.append(f'|{NAMES_SHORT[r["dataset"]]}|{r["n"]}|{r["baseline_BA"]:.3f}|{r["denoised_BA"]:.3f}|{r["BA_change_pp"]:+.2f} [{r["BA_low_pp"]:+.2f}, {r["BA_high_pp"]:+.2f}]|{r["baseline_AUC"]:.3f}|{r["denoised_AUC"]:.3f}|')
    report.extend(['', 'BA为左右召回率的平均，AUC为合并折外预测的ROC面积。95%区间对相同时间块成对重采样，固定已拟合模型；仅有5个块，不覆盖训练重估、开发调参和跨受试者不确定性。A1的BA差值区间为[0,0]是因为两分支给出完全相同的类别预测，并不表示未来两模型效果必然相同。不能把区间或总体AUC当成独立受试者泛化证据。', '',
                   '**波形与特征**', '',
                   '|数据组|共同目标差分RMSE：基础→去噪|各自目标差分RMSE：基础→去噪|固定窗方向同号比例：基础→去噪|',
                   '|---|---:|---:|---:|'])
    for key in keys:
        v=waves.query('dataset == @key and model == "mechanism"').groupby(['target','branch']).contrast_RMSE.mean()
        report.append(f'|{NAMES_SHORT[key]}|{v.loc[("common_baseline","baseline")]:.2f} → {v.loc[("common_baseline","denoised")]:.2f}|{v.loc[("branch_output","baseline")]:.2f} → {v.loc[("branch_output","denoised")]:.2f}|{feature_summary.loc[(key,"baseline")]:.3f} → {feature_summary.loc[(key,"denoised")]:.3f}|')
    report.extend(['', '共同目标是同一批基础预处理测试ERP，仍包含伪影；各自目标则随去噪发生变化，较小误差可能仅代表拟合了更平滑的对象。两者均不是神经真值恢复精度。特征稳定性采用三空间模态×三个固定时间窗的标准化左右效应，统计训练与测试同号比例；没有把微小效应剔除，也没有以此选择分支。', '',
                   '**可追溯性与局限**', '',
                   '- `folds/`逐折保存去噪参数、训练/测试编号、内部选型编号、机制模型参数、测试波形、特征及分类器参数；只应按对应训练折使用，不能把所有折外输出再拼成“独立”训练数据进行另一套划分。',
                   '- `audit/`保存独立滤波区间及剔除审计；`verification.json`记录编号隔离、源文件和输出哈希、独立指标复算、由保存分类器参数重建概率的检查。',
                   '- `predictions.csv`可重算BA、AUC和混淆矩阵；`paired_changes.csv`保存两方法配对差值区间。',
                   '- 本次完成嵌套时间分块对照与固定模型配对区间；没有为新版重跑999次全流程标签置换、200次完整参数重估、跨受试者和前向预测验证。旧结果中的p值不能移用到新版，不能据此宣称显著高于随机或证明真实神经机制。',
                   '- 当前机制核显式包含基础线性滤波，但没有建立新版非线性去噪的神经观测模型。因此去噪后拟合参数只能解释为有效响应参数，不能直接映射为神经源时间常数；逐折边界命中和优化诊断保存在JSON，不能凭拟合改善声称生理机制得到验证。',
                   '- 所有选择均来自已查看过的四组开发数据。即使划分操作严格，也不是新的独立受试者验证。', '',
                   '**复现命令**', '',
                   '```bash', 'python3 Q2/run_q2_fold_denoising.py --output Q2_fold_denoising_results',
                   'python3 Q2/report_fold_denoising.py --output Q2_fold_denoising_results',
                   'python3 -m unittest discover -s tests -p test_q2_fold_denoising.py -v', '```', '',
                   '已有结果目录拒绝无意覆盖；`--resume`核验相同源文件、输入和配置后重算，`--quick`仅供另目录调试。'])
    (out/'第二问去噪对照报告.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    verification=dict(status='passed',folds=checks,fold_count=len(checks),
                      source_and_output_hashes=True,same_test_trials=True,
                      classification_recomputed=True,saved_model_probability_recomputed=True,
                      reporting_source_sha256=sha(Path(__file__)),visual_QA='pending')
    save_json(verification,out/'verification.json')
    print(pd.DataFrame(comparison).to_string(index=False))
    print('Numerical audit passed. Inspect both PNGs before final handoff.')


if __name__ == '__main__':main()
