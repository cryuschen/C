#!/usr/bin/env python3
"""Build the diagnostic narrative and canonical portable-report input."""
from pathlib import Path
import datetime
import json
import numpy as np
import pandas as pd
from q2model.data import save_json,save_csv,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Q2_result/diagnosis'


def main():
    d=pd.read_csv(OUT/'descriptive_counterfactuals.csv');v=d[d.stage=='v7'].set_index(['dataset','variant'])
    w=pd.read_csv(OUT/'heldout_waveforms.csv');wt=w[w.split=='test'].groupby(['dataset','variant']).weighted_MSE.mean()
    budget=pd.read_csv(OUT/'prediction_error_decomposition.csv').groupby('dataset')[['total_error','span_approximation_error','coefficient_transfer_error']].mean()
    summary=[]
    for k in ['A1','A2','B1','B2']:
        base=v.loc[(k,'original'),'weighted_MSE']
        summary.append(dict(dataset=k,original_fit_error=base,
            no_ridge_fit_reduction=1-v.loc[(k,'same_kernel_no_ridge'),'weighted_MSE']/base,
            affine_fit_reduction=1-v.loc[(k,'same_kernel_plus_affine'),'weighted_MSE']/base,
            no_ridge_test_change=wt.loc[(k,'same_kernel_no_ridge')]/wt.loc[(k,'original')]-1,
            affine_test_change=wt.loc[(k,'same_kernel_plus_affine')]/wt.loc[(k,'original')]-1,
            coefficient_transfer_share=budget.loc[k,'coefficient_transfer_error']/budget.loc[k,'total_error'],
            span_approximation_share=budget.loc[k,'span_approximation_error']/budget.loc[k,'total_error']))
    save_csv(summary,OUT/'diagnostic_summary.csv')
    c=pd.read_csv(OUT/'counterfactual_classification.csv')
    title='Q2 Model Failure Diagnosis'
    sections=[
    ('summary','## 诊断结论\n当前失败包含两个层次：**机制核与慢变化的形状不匹配，以及训练时估计的幅度和方向系数不能稳定预测其他时间块。后者是留出误差的主要组成。**\n\n尚未发现能够解释整体失败的线性求解或预测复现错误。现在最值得修改的是背景项与试次稳健估计，并让时间参数更谨慎地参与解释；直接扩大参数范围、去掉正则或加复杂分类器缺乏现有证据支持。','summary_source'),
    ('definition','## 比较口径\n诊断读取原始 Fz/F3/F4 独立分段处理后的四组数据，保留原 24 秒缓冲、原试次和五折时间划分。A1/A2/B1/B2 分别保留 67/70/69/69 个试次。已知条件的描述性实验另在同一组 V7 ERP 上比较。\n\n波形损失沿用 50–250、250–500、500–750 ms 等窗口权重及左右等权，幅度为原始电位单位。训练拟合改善与测试误差变化分开报告；测试变化为负表示改善。所有新对照均为开发数据上的探索性诊断，没有新的显著性结论，也没有替换原正式主模型。','primary'),
    ('implementation','## 实现检查：没有发现足以解释失败的代数错误\n用独立的正规方程重新求解原核的 A/D，描述性预测最大差异为 4.3×10⁻¹³，20 个外折预测最大差异为 2.3×10⁻¹²。原 16 项验收测试已通过，其中包括第一问回归、测试标签不变性和置换重训检查。\n\n这支持将当前重点放在模型假设和统计稳定性上，但不等于证明全部预处理与所有代码绝无问题。三维正交空间变换本身可逆，保留全部三个模态时不会因为坐标旋转而丢失通道信息。',None),
    ('fit','## 拟合不足：强收缩与受限核形状共同作用\n项目一的 A1、B1 描述拟合选择了 λA=1、λD=10。固定时间核只去掉收缩，训练误差分别下降 74.7% 和 70.3%；但五折留出整体波形误差分别增加 4.3% 和 12.7%。强收缩牺牲了平均曲线拟合，同时也在抑制不稳定系数。不能直接将其认定为应该删除的错误。\n\n加入截距与线性趋势后，四组描述拟合误差下降 73.3%–97.9%。甚至只有截距和趋势的简单模型也能很好逼近项目一。这说明低维慢变化占据了很多波形结构，原有限脉冲核被迫承担背景拟合任务；它不能证明这些慢变化全部是伪影。\n\n这里“去收缩”和“加趋势”的训练内收益不能相加；二者解释的误差有重叠。特征投影并不直接使用拟合 D，因此单改 D 的惩罚、保持核不动，也不会直接改善分类器。','descriptive'),
    ('transfer','## 主要瓶颈：已有核内的系数不能稳定迁移\n将每个测试 ERP 投影到训练核张成的空间，仅用于事后诊断，可以将测试误差正交分为“该空间无法表达的误差”与“空间内系数没有预测对的误差”。四组后者占总测试误差约 93.7%、90.2%、85.0%、90.7%。分解闭合误差小于 3×10⁻⁹。\n\n这个带测试标签的理想投影不是可用预测器，也没有进入特征学习或选型。该比例包含正则偏差、有限试次均值噪声和潜在时间变化，不能全部称为真正的生理非平稳性。它说明：仅增加时间核表达能力，无法解释当前大部分测试误差。','decomposition'),
    ('stability','## 方向差异与块均值不稳定\n每组五个时间块形成 10 对比较。在 F3−F4 模态中，A1/A2/B1 各有 4 对、B2 有 6 对方向 ERP 的加权余弦相似度小于零，即整体方向出现反转。\n\n按块内左右样本方差估计的差分均值噪声能量，与观测差分能量之比的中位数分别约 0.66、1.08、1.27、2.28。该计算依赖近似独立试次假设，是均值不确定性的提示，不是神经信号 SNR 或伪影占比。\n\nB2 第 56 个原始事件对所在测试块左条件 ERP 的影响尤其大：仅删除它，均值改变的加权 RMS 为 112.9，而原条件 ERP 的 RMS 为 53.9。这里只做影响诊断，未据此剔除试次或改写主结果。小样本、极端幅度和背景波动都值得关注；没有 EOG，不能确定具体伪影来源。',None),
    ('changes','## 已试修改：训练改善不等于识别改善\n固定原训练核加入趋势，整体测试误差仅变化约 −2.5% 至 +2.7%。去掉投影正则、刺激前趋势外推、全窗口去趋势、只用早窗等对照，也没有在四组中一致提高识别率。全窗口去趋势在 B2 反而明显下降，不能将慢项直接视为应当删除的噪声。\n\n保持原阶数而放宽时间参数范围，可以改善部分训练拟合，但参数仍可顶到新边界，或引起核共线性。A2 的 V7 对照所有连续优化初值均未收敛，已明确记录失败并从比较图中留空；没有把保留原参数的回退结果冒充成功。','classification'),
    ('recommendation','## 修改模型的优先顺序\n**第一优先：在试次层面显式区分事件响应与背景。** 保留空间选择性→群体动态→头皮投影的机制主线，为观测模型增加低维背景项：Xᵢ(t)=B[A+dᵢD]H(t;θ)+Cᵢq(t)+εᵢ(t)。q 可先取截距、斜率或更受限的慢基；Cᵢ 的估计与未知方向无关，并受训练内确定的收缩约束。不能任意允许背景与响应互相抵消。\n\n**第二优先：稳健地估计共享响应和方向项。** 对单试次残差采用 Huber 或 Student-t 型稳健损失，检查高影响试次的权重。λA、λD 分别在训练内选择；核拟合考虑共享项和差分项的噪声尺度，防止大幅共同背景主导时间参数。上述方法是待检验候选，不是已经证实的修复。\n\n**第三优先：再检验慢核与延迟范围。** 内层选型与最终训练使用一致的求解精度，检查边界、条件数和背景项的可辨识性；只有背景处理后仍有可预测残差，才增加慢状态或方向延迟。当前不建议直接上复杂神经网络，也不建议仅因训练 R² 变好就采用更宽时间范围。\n\n新版本应预先固定少量候选，沿用样本隔离、训练内选型和完全监督重训置换。选择规则可包括留出波形、方向差分和特征稳定性，但必须事先定义。现有数据已用于多轮开发，后续内部验证不能替代新受试者或新记录的外部验证。',None),
    ('decision','## 当前判断与交付\n原结果保留为冻结基线。此次已经完成失败原因定位和受控对照代码，尚未把推荐的新观测模型实现成正式替代版本。下一版应先验证“背景项＋稳健试次拟合”是否改善跨块稳定性，再决定是否重跑完整 999 次置换；不能承诺一定能达到较高识别准确率。\n\n可复核脚本为 Q2/diagnose_q2.py；表格、图、优化失败记录和独立检查均保存在 Q2_result/diagnosis。HTML 中的图表是这些结果的固定快照。',None)]
    sources=[
        dict(id='primary',label='冻结正式结果及数据审计',path='Q2_result/第二问实验报告.md'),
        dict(id='summary_source',label='诊断汇总',path='Q2_result/diagnosis/diagnostic_summary.csv'),
        dict(id='descriptive',label='同一 ERP 的拟合对照',path='Q2_result/diagnosis/descriptive_counterfactuals.csv'),
        dict(id='decomposition',label='正交预测误差分解',path='Q2_result/diagnosis/prediction_error_decomposition.csv'),
        dict(id='classification',label='相同试次的探索性折外预测',path='Q2_result/diagnosis/counterfactual_classification.csv'),
        dict(id='heldout',label='训练与留出波形对照',path='Q2_result/diagnosis/heldout_waveforms.csv'),
        dict(id='influence',label='逐试次影响诊断',path='Q2_result/diagnosis/trial_influence.csv'),
        dict(id='similarity',label='时间块方向差异一致性',path='Q2_result/diagnosis/block_direction_similarity.csv')]
    # Independently reproduce the chart aggregates in SQLite; retain the actual
    # executed aggregation query, upstream filenames and scientific generator.
    import sqlite3
    connection=sqlite3.connect(':memory:')
    for table,file in [('descriptive_counterfactuals','descriptive_counterfactuals.csv'),('heldout_waveforms','heldout_waveforms.csv'),('prediction_error_decomposition','prediction_error_decomposition.csv'),('counterfactual_classification','counterfactual_classification.csv')]:
        pd.read_csv(OUT/file).to_sql(table,connection,index=False)
    query='''WITH d AS (
      SELECT dataset,
        MAX(CASE WHEN variant='original' THEN weighted_MSE END) AS original_fit_error,
        MAX(CASE WHEN variant='same_kernel_no_ridge' THEN weighted_MSE END) AS nr,
        MAX(CASE WHEN variant='same_kernel_plus_affine' THEN weighted_MSE END) AS af
      FROM descriptive_counterfactuals WHERE stage='v7' GROUP BY dataset),
    w AS (
      SELECT dataset,
        AVG(CASE WHEN variant='original' THEN weighted_MSE END) AS original_test,
        AVG(CASE WHEN variant='same_kernel_no_ridge' THEN weighted_MSE END) AS nr,
        AVG(CASE WHEN variant='same_kernel_plus_affine' THEN weighted_MSE END) AS af
      FROM heldout_waveforms WHERE split='test' GROUP BY dataset),
    b AS (
      SELECT dataset, SUM(total_error) AS total,
        SUM(coefficient_transfer_error) AS transfer, SUM(span_approximation_error) AS approximation
      FROM prediction_error_decomposition GROUP BY dataset)
    SELECT d.dataset,d.original_fit_error,
      1-d.nr/d.original_fit_error AS no_ridge_fit_reduction,
      1-d.af/d.original_fit_error AS affine_fit_reduction,
      w.nr/w.original_test-1 AS no_ridge_test_change,
      w.af/w.original_test-1 AS affine_test_change,
      b.transfer/b.total AS coefficient_transfer_share,
      b.approximation/b.total AS span_approximation_share
    FROM d JOIN w USING(dataset) JOIN b USING(dataset) ORDER BY d.dataset'''
    checked=pd.read_sql_query(query,connection)
    np.testing.assert_allclose(checked.drop(columns='dataset'),pd.DataFrame(summary).drop(columns='dataset'),atol=1e-12)
    summary=checked.to_dict(orient='records')
    class_query='SELECT * FROM counterfactual_classification ORDER BY dataset,variant'
    c=pd.read_sql_query(class_query,connection)
    for s in sources:
        if s['id']=='summary_source':s['query']={'language':'sql','engine':'SQLite','sql':query,'description':'实际运行的独立汇总；输入 CSV 由 Q2/diagnose_q2.py 生成并原样载入同名表，完整路径均在 Q2_result/diagnosis 下。','tables_used':['descriptive_counterfactuals','heldout_waveforms','prediction_error_decomposition']}
        if s['id']=='classification':s['query']={'language':'sql','engine':'SQLite','sql':class_query,'description':'逐试次预测先由 Q2/diagnose_q2.py 计算 BA/AUC，保存在同名 CSV，再原样载入 SQLite。','tables_used':['counterfactual_classification']}
    (OUT/'report_aggregation.sql').write_text(query+';\n\n'+class_query+';\n')
    charts=[dict(id='fit_chart',title='描述性拟合误差下降比例',subtitle='同一 V7 条件 ERP；这里只比较训练内拟合',type='bar',dataset='summary',sourceId='summary_source',xField='dataset',series=[dict(field='no_ridge_fit_reduction',label='同核去收缩'),dict(field='affine_fit_reduction',label='同核加趋势')],valueFormat='percent',layout='full'),
        dict(id='transfer_chart',title='测试误差中系数迁移误差的占比',subtitle='测试 ERP 的事后理想投影仅用于诊断，不是预测器',type='bar',dataset='summary',sourceId='summary_source',xField='dataset',series=[dict(field='coefficient_transfer_share',label='系数迁移误差占比')],valueFormat='percent',layout='full')]
    tables=[dict(id='changes_table',title='测试误差的相对变化',dataset='summary',sourceId='summary_source',defaultSort={'field':'dataset','direction':'asc'},columns=[dict(field='dataset',label='数据组',type='text'),dict(field='no_ridge_test_change',label='去收缩：测试误差变化',format='percent'),dict(field='affine_test_change',label='加趋势：测试误差变化',format='percent')]),
        dict(id='class_table',title='分类诊断对照：BA 与 AUC',dataset='classification',sourceId='classification',defaultSort={'field':'dataset','direction':'asc'},columns=[dict(field='dataset',label='数据组',type='text'),dict(field='variant',label='对照方法',type='text'),dict(field='n',label='试次',format='number'),dict(field='BA',label='平衡准确率',format='percent'),dict(field='AUC',label='AUC',format='number')])]
    fit_rows=[dict(dataset=r['dataset'],variant=label,reduction=r[field],original_fit_error=r['original_fit_error']) for r in summary for field,label in [('no_ridge_fit_reduction','同核去收缩'),('affine_fit_reduction','同核加趋势')]]
    for chart in charts:
        chart.pop('xField');chart.pop('series')
        chart['encodings']={'x':{'field':'dataset','type':'nominal','label':'数据组'},'y':{'field':'coefficient_transfer_share','type':'quantitative','format':'percent','label':'系数迁移误差占比'}}
    charts[0]['dataset']='fit_comparison'
    charts[0]['encodings'].update(y={'field':'reduction','type':'quantitative','format':'percent','label':'训练误差下降'},color={'field':'variant','type':'nominal'})
    blocks=[dict(id='title',type='markdown',body='# '+title)]
    for key,body,source in sections:
        b=dict(id=key,type='markdown',body=body)
        if source:b['sourceId']=source
        blocks.append(b)
        if key=='fit':blocks.append(dict(id='fit_plot',type='chart',chartId='fit_chart'))
        if key=='transfer':blocks.append(dict(id='transfer_plot',type='chart',chartId='transfer_chart'))
        if key=='changes':blocks.extend([dict(id='changes_values',type='table',tableId='changes_table'),dict(id='class_values',type='table',tableId='class_table')])
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    artifact=dict(surface='report',manifest=dict(version=1,surface='report',title=title,generatedAt=stamp,blocks=blocks,charts=charts,tables=tables,sources=sources),snapshot=dict(version=1,generatedAt=stamp,status='ready',datasets={'summary':summary,'fit_comparison':fit_rows,'classification':json.loads(c.to_json(orient='records'))}),sources=sources)
    save_json(artifact,OUT/'artifact.json')
    # Markdown is a readable source companion; HTML is the portable report surface.
    md='# 第二问失败原因诊断与修改判断\n\n'+'\n\n'.join(s[1] for s in sections)
    md+='\n\n## 可复核图表\n\n'+''.join(f'- [{f.stem}]({f.name})\n' for f in sorted(OUT.glob('*.png')))
    (OUT/'失败原因诊断与修改建议.md').write_text(md.rstrip()+'\n')
    print('Report source ready:',OUT/'artifact.json')


if __name__=='__main__':main()
