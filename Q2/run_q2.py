#!/usr/bin/env python3
"""Run the complete Q2 pipeline. Default: full, 999 permutations + 200 refits/group."""
import os
for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[var]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/eeg-q2-mpl')
from pathlib import Path
import argparse
import json
import sys
import hashlib
import platform
import time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import scipy
import sklearn
from q2model.data import (NAMES,FS,TIMES,filters,independent_data,sha,save_json,save_csv)
from q2model.mechanism import run_mechanism
from q2model.experiments import descriptive,shared_project_test,synthetic_tests
from q2model.validation import (nested_cv,block_intervals,supplemental,stability,run_permutations,holm)
from q2model.report import make_figures,write_report

ROOT=Path(__file__).resolve().parents[1]
STAGES=['audit','mechanism','fit','validate','supplemental','synthetic','stability','permutation','report']


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'Q2_result')
    p.add_argument('--seed',type=int,default=20260924)
    p.add_argument('--n-jobs',type=int,default=min(8,os.cpu_count() or 1))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=['all']+STAGES,default='all')
    p.add_argument('--quick',action='store_true',help='Debug only: 3 permutations, 3 full refits, 50 CI draws')
    return p


def documentation(out,args):
    cmd=f'{sys.executable} {Path(__file__).resolve()} --output {out}'
    (out/'运行说明.md').write_text(f'''# 第二问运行与复核

正式完整运行：

```bash
{cmd} --n-jobs {args.n_jobs}
```

断点恢复：在原命令后添加 `--resume`。已有目录拒绝无恢复参数的覆盖。代码、输入或数值配置变化时拒绝复用检查点，应使用新的输出目录。`--stage` 可单独运行阶段，运行报告会明确显示未完成检验。

快速调试必须另选目录并添加 `--quick`，只有3次置换和3次参数重采样，不能标为正式完整验证。正式配置每组999次置换、200次完整重估和2000次折外预测区间重采样。进程级并行，每个工作进程的BLAS限制为单线程。

依赖沿用项目 requirements-v7.txt，另使用其scikit-learn依赖中的joblib。建议使用已验证的Python环境；版本见manifest.json。

阶段：audit（隔离及试次）、mechanism（示意仿真）、fit（V7/预处理条件描述）、validate（严格分块识别）、supplemental（前向/跨组/共享参数）、synthetic（合成）、stability（敏感性及200次重估）、permutation（999次完整置换）、report（图表与报告）。

公共接口：`train_model(training_dataset)` 只接收训练数据；模型 `features(eeg)` 和分类预测不接收测试标签。Dataset保存原始试次编号、块、提示宽度及质量量。每个fold的JSON包含训练/测试编号和所有内层编号。预测CSV可独立重算BA/AUC/混淆矩阵。

图表均由已有CSV/NPZ重新构建，PNG用于预览、PDF为矢量输出。推断模型来自原始数据独立分段处理，不能用V7数组替换。真实数据没有无噪声真值，仿真导联和神经群体参数是示意假设。
''',encoding='utf8')
    (out/'指标字典.md').write_text('''# 第二问指标与数值口径

- 条件ERP：同方向保留试次的算术均值。Q2差分统一为右减左。
- 三模态：共同=(Fz+F3+F4)/√3；中线=(2Fz−F3−F4)/√6；两侧=(F3−F4)/√2。属于观测坐标，不是脑源。
- 训练损失：两方向等权，50–250、250–500、500–750ms每窗等权；区间左闭右开，最后包含750ms。
- 核：采样指数核与alpha核按离散单位面积归一化；连续卷积的dt已吸收到离散权重。延迟采用相邻采样点线性分配，未滤波核因果。零相位观测滤波后的核允许刺激前扩散，不能反推负生理延迟。
- 模型系数：A为条件共享，D为半个右减左方向项。核作加权范数归一化，系数单位为原始电位单位；不是实际突触连接。
- 内层为有界网格搜索，每个时间维度正式4个点；外层最终拟合从3个不同优良网格点连续优化。网格与优化数值近似不代表全局最优。
- RMSE：50–750ms观测与预测差的均方根。R²=1−SSE/Σ(y−均值y)²，按方向/通道报告；可为负，常量目标记NA。
- S_delta=1−Σ(测试差分−训练预测差分)²/Σ测试差分²。正值才优于零差分；分母≤1e−10记NA，同时保留绝对误差与分母。
- 特征：三空间模态对训练确定的时间核进行加权岭投影，λf=0.1。固定窗均值为九维透明基准。缺失正峰潜伏期记NA，不补零。
- BA=(左召回+右召回)/2；机会参考0.5。AUC的正类为+1右。真实类别只有一类时BA/AUC记NA。
- 混淆矩阵：行真实、列预测，顺序均为左/右。预测为右的概率≥0.5时判右。
- 置换p=(1+置换BA≥真实BA的次数)/(置换次数+1)；四个预先指定主检验用Holm校正。每次完整重训监督步骤。
- 折外区间：按5个时间块抽样，固定已训练模型；不覆盖重训、开发或人群泛化不确定性。
- 参数重采样：各时间块内按左右分别抽样并完整重估；不同核数的参数不可直接混成一个分位区间。
- 核条件数：加权核设计矩阵的2范数条件数。边界命中定义为距工程范围边界≤0.5ms，另检查有序约束。
- 九维效应：折外、按各折训练集标准化的候选三核系数之右减左均值除以合并组内标准差；为探索性，核坐标在折间不完全相同。
- 离线滤波边界：使用组合双向滤波冲激响应的相对L1尾部≤0.001确定24s主缓冲；另做32s敏感性分析。分段原始样本不交叉。
''',encoding='utf8')


def main():
    args=parser().parse_args();out=args.output.resolve()
    if args.n_jobs<1:raise ValueError('--n-jobs must be positive')
    numeric_files=[Path(__file__).resolve()]+sorted((ROOT/'Q2/q2model').glob('*.py'))
    source={str(p.relative_to(ROOT)):sha(p) for p in numeric_files}
    inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'data').glob('*.mat'))}
    for p in sorted((ROOT/'eeg_v7_results').glob('受试者*/可复核波形.npz')):inputs[str(p.relative_to(ROOT))]=sha(p)
    for p in [ROOT/'EEG_P300_artifact_correction_v7.py',ROOT/'服务于脑机接口与精神性疾病诊断的脑电图计算模型.docx']:
        inputs[str(p.relative_to(ROOT))]=sha(p)
    config=dict(seed=args.seed,quick=args.quick,permutations=3 if args.quick else 999,
                refit_bootstrap=3 if args.quick else 200,interval_bootstrap=50 if args.quick else 2000,
                guard_seconds=24.,tau_G_ms=20.,kernel_grid_points=3 if args.quick else 4)
    signature=hashlib.sha256(json.dumps(dict(source=source,inputs=inputs,config=config),sort_keys=True).encode()).hexdigest()
    path=out/'manifest.json'
    if path.exists():
        if not args.resume:raise RuntimeError('Output exists; use --resume or another output directory')
        manifest=json.loads(path.read_text())
        if manifest['signature']!=signature:raise RuntimeError('Code/input/config changed: checkpoints cannot be reused')
    else:
        out.mkdir(parents=True,exist_ok=True)
        manifest=dict(signature=signature,source=source,inputs=inputs,config=config,stages={},
            created_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
            numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
            unit='原始电位单位',direction='right_minus_left',status='running')
    save_json(manifest,path);documentation(out,args)
    for folder in ('audit','mechanism','fits','features','validation','stability','synthetic','figures','checkpoints'):
        (out/folder).mkdir(exist_ok=True)
    datasets={}
    def get_datasets():
        if not datasets:
            for key in NAMES:datasets[key]=independent_data(ROOT,key,24,out)
        return datasets
    stages=STAGES if args.stage=='all' else [args.stage]
    for stage in stages:
        if args.resume and manifest['stages'].get(stage,{}).get('status')=='complete' and stage!='report':
            print(f'SKIP {stage}: verified signature',flush=True);continue
        start=time.monotonic();print(f'STAGE {stage}',flush=True)
        manifest['stages'][stage]=dict(status='running');save_json(manifest,path)
        try:
            if stage=='audit':
                ds=get_datasets()
                from scipy.signal import filtfilt,sosfiltfilt
                n=FS*240+1;imp=np.zeros(n);imp[n//2]=1;b,a,sos=filters()
                h=sosfiltfilt(sos,filtfilt(b,a,imp));dist=np.abs(np.arange(n)-n//2)
                tails={str(s):float(np.abs(h)[dist>s*FS].sum()/np.abs(h).sum()) for s in (16,24,32)}
                save_json(dict(relative_L1_tails=tails,counts={k:len(d.x) for k,d in ds.items()}),out/'audit/filter_support.json')
            elif stage=='mechanism':run_mechanism(out)
            elif stage=='fit':descriptive(ROOT,out,args.quick)
            elif stage=='validate':
                summary=[]
                for key,ds in get_datasets().items():
                    rows,_,_=nested_cv(ds,args.quick,True,out)
                    summary.extend(block_intervals(rows,config['interval_bootstrap'],args.seed))
                    print(f'VALIDATED {key}',flush=True)
                save_csv(summary,out/'validation/classification_summary.csv')
            elif stage=='supplemental':
                supplemental(get_datasets(),ROOT,out,args.quick)
                shared_project_test(get_datasets(),out,args.quick)
            elif stage=='synthetic':synthetic_tests(out,args.quick,args.seed)
            elif stage=='stability':
                for ds in get_datasets().values():stability(ds,out,args.quick,config['refit_bootstrap'],args.n_jobs,args.seed)
            elif stage=='permutation':
                summary=pd.read_csv(out/'validation/classification_summary.csv');pvalues=[]
                for key,ds in get_datasets().items():
                    observed=float(summary[(summary.dataset==key)&(summary.model=='mechanism_selected')].BA.iloc[0])
                    pvalues.append(run_permutations(ds,observed,config['permutations'],args.seed,args.quick,out,args.n_jobs))
                corrected=holm([r['p_raw'] for r in pvalues])
                for row,p in zip(pvalues,corrected):row['p_holm']=p
                save_csv(pvalues,out/'validation/permutation_tests.csv')
            elif stage=='report':make_figures(out);write_report(out)
        except Exception as exc:
            manifest['stages'][stage]=dict(status='failed',error=repr(exc),seconds=time.monotonic()-start)
            manifest['status']='failed';save_json(manifest,path);raise
        manifest['stages'][stage]=dict(status='complete',seconds=time.monotonic()-start)
        save_json(manifest,path)
    manifest['status']='complete' if all(manifest['stages'].get(s,{}).get('status')=='complete' for s in STAGES) else 'partial'
    manifest['output_hashes']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p!=path and not p.name.endswith('.tmp')}
    save_json(manifest,path)
    print(f'DONE {manifest["status"]} {out}',flush=True)


if __name__=='__main__':main()
