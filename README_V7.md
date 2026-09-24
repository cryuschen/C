# 第一问终版（V7）

提交时先阅读 [第一问终稿实验报告](eeg_v7_results/第一问终稿实验报告.md)，再查看 `eeg_v7_results/` 中四组图表、逐方向逐通道指标、半合成验证和运行清单。完整结果由 `EEG_P300_artifact_correction_v7.py` 生成，依赖仓库中的 V3、V4 模块与 `requirements-v4.txt`；原始四份 `.mat` 数据保存在 `data/`。

V7 在项目一、项目二中分别分析 Fz/F3/F4 原始通道，采用固定预处理和坏试次筛查、五折训练选型、折外波形校正、分左右方向 ERP 与样条拟合。V7 在训练折选到较弱的“双分量”候选时，对试次改动量采用 25% 比例并保护 10% 的训练方向均值；其余候选采用 V6 的 15% 比例。所有分支只使用当前折训练试次估计参数，但这些固定比例是在查看全部四组开发结果后确定的。

```bash
cd /home/cryus/code-project/C
/home/cryus/code-project/ML/.venv/bin/python EEG_P300_artifact_correction_v7.py --output eeg_v7_results
/home/cryus/code-project/ML/.venv/bin/python -m unittest discover -s tests -v
```

其他环境使用 Python 3.11+ 并安装 `requirements-v4.txt`。重算数值、跳过绘图可使用 `--no-plots --output /tmp/eeg-v7-recheck`。主要的可复核文件是每组的 `可复核波形.npz`、`完整事件与试次审计.csv`、`逐折训练参考评价清单.csv`，以及汇总目录的 `运行清单.json`。指标的公式与解释见 [指标字典_V7.md](指标字典_V7.md)。

图文 PDF 由 `build_first_question_pdf.py` 从 V7 结果重建；另需安装 `requirements-report.txt` 中的 ReportLab 和 Pillow，并有可用的 Noto Sans SC 字体。

结论只适用于这四组开发数据：V7 相对预处理的代理参考 MAE 降幅为 5.4%–10.0%，按方向分层的试次重采样组均值区间均高于零。SNR 代理量的四组点估计略升，但增量区间都跨零；左右差分幅度保留比为 0.81–0.95，却不能证明保留的是神经特征。按原始事件顺序前后对半，A 项目一和 B 项目二的三通道左右差分相关为负。1 倍和 2 倍半合成注入恢复误差较 V6 略低，但 0.5 倍轻扰动与零注入仍有背景改动。详细数值见 `汇总与说明/四组核心指标重采样区间.csv` 和 `汇总与说明/左右差分前后时段稳定性.csv`。代理参考并非无噪声真值，三个额区通道也无法区分眼动和真实慢响应，结果不能称为独立受试者泛化、已确认的神经源恢复、未知刺激解码或临床诊断准确率。
