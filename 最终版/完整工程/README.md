# 视觉认知脑电三题建模与验证

项目使用题目提供的两名受试者、两个任务、四份原始MAT，建立伪影校正、视觉形成和认知动态模型。所有分析仅使用Fz/F3/F4原始EEG；设备Decon不作为建模输入。结果支持范围须与统计检验一起阅读，模拟行为不能当作真实正确率。

## 当前入口

| 内容 | 代码 | 结果 |
|---|---|---|
| 三题优化与验收 | `run_optimization.py` | `优化计划/本轮优化结果.md` |
| Q1原V7及独立验证 | `EEG_P300_artifact_correction_v7.py` | `eeg_v7_results/` |
| Q1固定图4/5 | `Q1/main.py` | `Q1/figures_4_5_v7/` |
| Q2形成与判别 | `q2/main.py` | `q2_result/` 中Q2_文件 |
| Q3冻结重建与行为仿真 | `q3/main.py` | `q3_result/` |
| Q3新拟合与机制检验 | `q3/main.py --stage optimize` | `q3_result/` 中优化_文件 |

大小写以实际目录为准：`q2`、`q3`，第一题绘图目录为`Q1`。

## 运行

Python 3.11+，安装`requirements.txt`并提供Noto Sans CJK SC等中文字体。在项目根目录运行：

```bash
.venv/bin/python -B run_optimization.py
```

Windows将解释器替换为`.venv\Scripts\python.exe`，脚本和包名仍保持小写。分阶段：

```bash
.venv/bin/python -B run_optimization.py --stage q2
.venv/bin/python -B run_optimization.py --stage q1
.venv/bin/python -B run_optimization.py --stage q3
.venv/bin/python -B run_optimization.py --stage report
.venv/bin/python -B run_optimization.py --stage verify
```

完整流程会更新原结果目录的对应文件。优化前源码、CSV、文档与清单保存在`优化计划/优化前基线.zip`；原始MAT保持只读。大体量历史NPZ可由配置中的Git提交取得，基线ZIP没有复制这些波形文件。

Q2执行1999次块内置换、199次循环移位与1000次重采样。Q1独立验证区分半合成和完全合成目标，并隔离复算旧V7、保留Q2已使用的NPZ字节。Q3先重建V2结果，再进行新的训练与参数恢复；新训练从原始MAT计算，冻结截窗对照仍需V2参数文件。

## 解释边界

新增低维特征未建立稳定识别优势，方向收缩未替代原机制主模型。Q1保守门控侧重降低背景失真，不能承诺全部污染强度下更优。Q3记忆状态是功能模型变量，参数恢复和仿真不能单独证明海马来源。400个真实事件的正确与实验超时状态保持未知。

逐试次队列差异、最新数值和验证范围见`优化计划/本轮优化结果.md`、`优化计划/三题试次对账.csv`及各题报告。
