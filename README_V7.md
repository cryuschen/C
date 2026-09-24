# 第一问终版（V7）

先阅读 [第一问终稿实验报告](eeg_v7_results/第一问终稿实验报告.md)，再查看 `eeg_v7_results/` 中四组图表、逐方向逐通道指标、半合成验证和运行清单。44 张结果图均由 `EEG_P300_artifact_correction_v7.py` 独立生成；运行时需安装 `requirements-v7.txt` 所列第三方库，并将四份原始 `.mat` 数据放在同目录的 `data/` 中。

V7 分别分析项目一、项目二的 Fz/F3/F4 原始通道，采用固定预处理、坏试次筛查、五折训练选型、折外波形校正、分左右方向 ERP 与样条拟合。选到“保守双分量”候选时，对试次改动量采用 25% 比例并保护 10% 的训练方向均值；其余候选采用 15% 比例。比例是在查看全部四组开发结果后确定的，不是独立受试者验证出的最优值。

```bash
cd /home/cryus/code-project/C
/home/cryus/code-project/ML/.venv/bin/python EEG_P300_artifact_correction_v7.py --output eeg_v7_results
/home/cryus/code-project/ML/.venv/bin/python -m unittest discover -s tests -v
```

其他环境使用 Python 3.11+ 并安装 `requirements-v7.txt`。只重算数值可使用 `--no-plots --output /tmp/eeg-v7-recheck`。主要复核文件是每组的 `可复核波形.npz`、`完整事件与试次审计.csv`、`逐折训练参考评价清单.csv`，以及汇总目录的 `运行清单.json`。指标公式与解释见 [指标字典_V7.md](指标字典_V7.md)。

结论只适用于这四组开发数据：V7 相对预处理的代理参考 MAE 降幅为 5.4%–10.0%，按方向分层的试次重采样组均值区间均高于零。SNR 代理量的四组点估计略升，但增量区间都跨零；左右差分幅度保留比为 0.81–0.95，却不能证明保留的是神经特征。按原始事件顺序前后对半，A 项目一和 B 项目二的三通道左右差分相关为负。半合成轻扰动与零注入场景显示 V7 可能改动背景，不能保证每种强度均优于未校正。详细数值见 `汇总与说明/四组核心指标重采样区间.csv` 和 `汇总与说明/左右差分前后时段稳定性.csv`。代理参考并非无噪声真值，三个额区通道也无法区分眼动和真实慢响应，结果不能称为独立受试者泛化、已确认的神经源恢复、未知刺激解码或临床诊断准确率。
