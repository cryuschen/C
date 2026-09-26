#!/usr/bin/env python3
"""Reproduce the bounded optimization in the existing source/result directories."""
from pathlib import Path
import argparse,subprocess,sys,os,json,hashlib,tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"c-optimization-mpl"))


def execute(args,log):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(Path(tempfile.gettempdir())/'c-optimization-mpl'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    print('Running:', ' '.join(map(str,args)),flush=True)
    with (ROOT/log).open('w') as f:
        subprocess.run([sys.executable,'-B',*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def q1_replay():
    """Recompute original V7, verify arrays/CSVs, retain already consumed Q2 inputs."""
    original=ROOT/'eeg_v7_results';maximum=0.;n_arrays=0;n_csv=0
    before=json.loads((original/'汇总与说明/运行清单.json').read_text())
    with tempfile.TemporaryDirectory(prefix='c-q1-replay-') as directory:
        out=Path(directory)
        execute(['EEG_P300_artifact_correction_v7.py','--no-plots','--output',out],'eeg_v7_results/独立验证_V7复算日志.txt')
        for p in out.rglob('*.npz'):
            with np.load(p) as new,np.load(original/p.relative_to(out)) as old:
                assert set(new.files)==set(old.files)
                for name in new.files:
                    a,b=new[name],old[name]
                    if np.issubdtype(a.dtype,np.number):
                        np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-9,equal_nan=True)
                        if np.isfinite(a).any():maximum=max(maximum,float(np.nanmax(abs(a-b))))
                    else:np.testing.assert_array_equal(a,b)
                    n_arrays+=1
        for p in out.rglob('*.csv'):
            other=original/p.relative_to(out)
            pd.testing.assert_frame_equal(pd.read_csv(p),pd.read_csv(other),check_exact=False,rtol=1e-8,atol=1e-8)
            n_csv+=1
        newmanifest=json.loads((out/'汇总与说明/运行清单.json').read_text())
    before.setdefault('original_statistical_source',before['source'].copy())
    before['source']=newmanifest['source']
    before['validated_replay']={'arrays':n_arrays,'csv':n_csv,'max_absolute_array_difference':maximum,
        'original_binary_inputs_retained':True,'reason':'Q2 consumed the original NPZ bytes; replay verifies identical numerical method without replacing those files'}
    before['supplementary_manifest']='独立验证_清单.json'
    (original/'汇总与说明/运行清单.json').write_text(json.dumps(before,ensure_ascii=False,indent=2))
    save_report=original/'第一问终稿实验报告.md';s=save_report.read_text();marker='\n## 新增独立验证\n'
    if marker in s:s=s.split(marker)[0]
    s+=marker+'\n[独立验证报告](独立验证_报告.md)补充275次独立时间块队列、未参与选型的污染家族、简单收缩与保守门控；与本报告371次队列分开。原V7数值已用当前源码隔离复算，保存的NPZ未替换，以保留Q2输入来源。\n'
    save_report.write_text(s)
    print('Original V7 numerical replay verified:',n_arrays,n_csv,maximum,flush=True)


def integration():
    from zipfile import ZipFile
    from io import BytesIO
    from q2.q2model.reporting import md_table
    q1=ROOT/'eeg_v7_results';q2=ROOT/'q2_result';q3=ROOT/'q3_result'
    q2audit=pd.read_csv(q2/'Q2_事件审计.csv');q3audit=pd.read_csv(q3/'event_audit.csv');rows=[]
    names={'A1':'受试者A_项目一','A2':'受试者A_项目二','B1':'受试者B_项目一','B2':'受试者B_项目二'}
    for key,folder in names.items():
        a=pd.read_csv(q1/folder/'完整事件与试次审计.csv').set_index('trial_id')
        b=q2audit[q2audit.dataset==key].set_index('trial_id');c=q3audit[q3audit.dataset==key].set_index('trial_id')
        for trial in range(1,101):
            rows.append(dict(dataset=key,trial_id=trial,Q1_included=a.loc[trial,'role']=='折外评价',Q1_reason=a.loc[trial,'role'],
                Q2_included=bool(b.loc[trial,'retained']),Q2_reason=b.loc[trial,'reason'],
                Q3_included=bool(c.loc[trial,'retained']),Q3_reason=c.loc[trial,'reason'],response_status=c.loc[trial,'response_status'],
                correctness='unknown',timeout='unknown'))
    pd.DataFrame(rows).to_csv(ROOT/'优化计划/三题试次对账.csv',index=False)
    d=pd.read_csv(q2/'Q2_表4_特征判别.csv');d=d[d.dataset=='pooled']
    first=pd.read_csv(q1/'独立验证_汇总.csv')
    third=pd.read_csv(q3/'优化_新拟合区间.csv')
    rec=pd.read_csv(q3/'优化_恢复网格.csv').groupby('noise')[['exact_tau','exact_dynamics','NRMSE']].mean().reset_index()
    text='# 三题优化结果与复现\n\n在原源码和现有结果目录内完成。本轮候选范围记录于本轮实验配置.json；优化前的代码、表格、报告及清单保存在优化前基线.zip，Git基线见配置。所有统计仍属于已反复查看的两名受试者内部研究。\n\n'
    text+='## 第二题\n\n'+md_table(d[d.features.isin(['amplitude','mechanism','projection6','direction3'])],['features','BA','AUC','block_permutation_maxT_p'])
    text+='\n\n新增表示没有建立显著判别优势；方向收缩在三组变差，未替换主模型。新主检验校正范围从五组扩展为七组，故旧特征的校正p值可能变化，原始分数保持可对账。详见../q2_result/Q2_优化报告.md。\n\n'
    text+='## 第一题\n\n'+md_table(first[first.level.isin([0,.5])],['origin','family','level','method','normalized_RMSE','contrast_RMSE'])
    text+='\n\n零污染非零误差代表背景改动。详细逐组/逐块/种子结果、选择和真值见../eeg_v7_results/独立验证_报告.md。保守门控为并列候选，不能把半合成收益直接命名为真实视觉神经信息恢复。\n\n'
    text+='## 第三题\n\n扩大网格没有稳定改善原V2 N2，故保留原EEG主结果；新拟合作为敏感性对照。新拟合N2相对N3的目标后增量：\n\n'+md_table(third[third.stage=='retrieval'],list(third.columns))
    text+='\n\n条件合成恢复（27场景）：\n\n'+md_table(rec,list(rec.columns))
    text+='\n\n新拟合、冻结参数截窗检验和仿真分别保存。不能将扩展网格后的测试最佳时间常数当作训练选择；剖面不是参数置信区间。目标匹配使用模拟Task-2图像布局，实测400次正误/超时仍未知。详见../q3_result/优化_机制验证报告.md。\n\n'
    text+='## 三问衔接\n\nQ1原V7为已知条件离线校正；Q2独立时间块用于未知提示方向检验；Q3因果滤波与应答前长窗用于认知模型。三题主样本371/275/268，逐试次原因见三题试次对账.csv。Q1左减右、F3减F4与Q2右减左、F4减F3需要反号才能比较。幅度全部为原始记录单位。\n\n'
    text+='## 一条命令复现\n\n` .venv/bin/python -B run_optimization.py `\n\n可加`--stage q2/q1/q3/report/verify`单独运行。q2与q3阶段会更新对应现有结果，q1会保存新独立验证并隔离复算旧V7数值。原始MAT从未修改。新增N2/N3拟合从原始MAT训练；冻结截窗对照仍依赖V2保存参数。第三题main默认同时生成冻结重建、事件验证和优化对照。\n'
    (ROOT/'优化计划/本轮优化结果.md').write_text(text,encoding='utf8')


def verify():
    execute(['-m','unittest','discover','-s','tests','-v'],'优化计划/Q1测试.txt')
    execute(['-m','unittest','discover','-s','q2/tests','-v'],'优化计划/Q2测试.txt')
    execute(['-m','unittest','discover','-s','q3/tests','-v'],'优化计划/Q3测试.txt')
    from q2.q2_pipeline import verify_outputs,finalize_manifest
    verify_outputs(ROOT/'q2_result');finalize_manifest(ROOT/'q2_result',redraw=True)
    from q3.audit import audit
    audit(ROOT/'q3_result')
    from audit_optimization import audit as audit_new
    audit_new()
    sources=[ROOT/'audit_optimization.py',ROOT/'run_optimization.py',*ROOT.glob('q2/**/*.py'),*ROOT.glob('Q1/*.py'),*ROOT.glob('q3/**/*.py'),ROOT/'q3/fit.py']
    raw=[*ROOT.glob('data/*.mat')]
    checks={'status':'verified','source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'input_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw}}
    (ROOT/'优化计划/本轮验收.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','q2','q1','q3','report','verify'],default='all');a=p.parse_args()
    if a.stage in ('all','q2'):
        execute(['q2/main.py'],'q2_result/Q2_优化运行日志.txt')
        from q2.optimization_diagnostics import run
        run()
    if a.stage in ('all','q1'):
        execute(['EEG_P300_artifact_correction_v7.py','--independent'],'eeg_v7_results/独立验证_运行日志.txt');q1_replay()
    if a.stage in ('all','q3'):
        execute(['q3/main.py'],'q3_result/运行日志.txt')
    if a.stage in ('all','report'):integration()
    if a.stage in ('all','verify'):verify()


if __name__=='__main__':main()
