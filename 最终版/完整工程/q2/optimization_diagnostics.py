"""Compare current Q2 outputs with the immutable pre-optimization archive."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pandas as pd
from q2.q2model.validation import metrics
from q2.q2model.reporting import md_table
ROOT=Path(__file__).resolve().parents[1]


def run():
    out=ROOT/'q2_result'
    with ZipFile(ROOT/'优化计划/优化前基线.zip') as z:
        old=pd.read_csv(BytesIO(z.read('q2_result/Q2_机制逐折指标.csv')))
        old_scores=pd.read_csv(BytesIO(z.read('q2_result/Q2_逐试次留出预测.csv')))
    rows=[]
    for (key,block,feature),g in old_scores.groupby(['dataset','block','features']):
        rows.append(dict(dataset=key,block=block,features=feature,**metrics(g.cue.to_numpy(),g.score.to_numpy())))
    pd.DataFrame(rows).to_csv(out/'Q2_优化前分块判别.csv',index=False)
    old.to_csv(out/'Q2_优化前机制诊断.csv',index=False)
    model_comp=old.groupby('model')[['delta_MSE','RMSE']].mean().reset_index()
    new=pd.read_csv(out/'Q2_表4_特征判别.csv');shrink=pd.read_csv(out/'Q2_方向收缩对照.csv')
    comparison=[];rng=np.random.default_rng(20260926)
    for key,g in shrink.groupby('dataset'):
        a=g.delta_MSE.to_numpy();b=g.original_delta_MSE.to_numpy();ix=rng.integers(0,5,(2000,5));samples=1-a[ix].mean(1)/b[ix].mean(1)
        comparison.append(dict(dataset=key,relative_to_full_gain=1-a.mean()/b.mean(),ci_low=np.quantile(samples,.025),ci_high=np.quantile(samples,.975),
            mean_factor=g.factor.mean(),S_delta=1-g.delta_MSE.mean()/g.zero_MSE.mean()))
    comparison=pd.DataFrame(comparison);comparison.to_csv(out/'Q2_方向收缩汇总.csv',index=False)
    pooled=new[new.dataset=='pooled']
    text='# 第二问优化诊断与结果\n\n优化前结果从优化计划/优化前基线.zip读取，未根据新分数改变候选集合。\n\n'
    text+='## 原模型的问题\n\n强正则与跨时间块不稳定同时存在，不能只归咎于分类器。原 full 的20折中有'+str(int((old[old.model=='full'].ridge==100).sum()))+'折选择最大岭惩罚100。普通斜坡的平均整体RMSE更低，但方向误差不一定更小；波形与方向须分别评价。\n\n'+md_table(model_comp,list(model_comp))
    text+='\n\n## 固定低维表示\n\n模板、标准化和LDA全部在训练折构造，六维和三维表示没有利用测试标签。七组主特征共同进行1999次完整重拟合块内置换及maxT校正，另有199次循环移位。\n\n'+md_table(pooled,['features','BA','BA_low','BA_high','AUC','block_permutation_maxT_p'])
    text+='\n\n不依据外层最高分自动替换原主模型；若校正证据不足，低维表示仍是候选。合并BA按试次汇总，区间为保存折外预测的块重采样。\n\n'
    text+='## 方向收缩\n\n共同响应保持原值。方向差分乘0/0.25/0.5/1，强度在训练块内选择，每个内层模板也排除其验证块。这是预测正则对照，不能当作新神经源证据；没有将其用于筛选分类器。\n\n'+md_table(comparison,list(comparison))
    (out/'Q2_优化报告.md').write_text(text,encoding='utf8')


if __name__=='__main__':run()
