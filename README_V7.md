# 第一问终版（V7）

提交时先阅读 [第一问终稿实验报告](eeg_v7_results/第一问终稿实验报告.md)，再查看 `eeg_v7_results/` 中四组图表、逐方向逐通道指标、半合成验证和运行清单。完整结果由 `EEG_P300_artifact_correction_v7.py` 生成，依赖仓库中的 V3、V4 模块与 `requirements-v4.txt`；原始四份 `.mat` 数据保存在 `data/`。

V7 在项目一、项目二中分别分析 Fz/F3/F4 原始通道，采用固定预处理和坏试次筛查、五折训练选型、折外波形校正、分左右方向 ERP 与样条拟合。V7 在训练折选到较弱的“双分量”候选时，对试次改动量采用 25% 比例并保护 10% 的训练方向均值；其余候选采用 V6 的 15% 比例。所有分支只使用当前折训练试次估计参数，但这些固定比例是在查看全部四组开发结果后确定的。

```bash
cd /home/cryus/code-project/C
/home/cryus/code-project/ML/.venv/bin/python EEG_P300_artifact_correction_v7.py --output eeg_v7_results
/home/cryus/code-project/ML/.venv/bin/python -m unittest discover -s tests -v
```

其他环境使用 Python 3.11+ 并安装 `requirements-v4.txt`。重算数值、跳过绘图可使用 `--no-plots --output /tmp/eeg-v7-recheck`。主要的可复核文件是每组的 `可复核波形.npz`、`完整事件与试次审计.csv`、`逐折训练参考评价清单.csv`，以及汇总目录的 `运行清单.json`。指标的公式与解释见 [指标字典_V7.md](指标字典_V7.md)。

结论只适用于这四组开发数据：V7 相对预处理降低了代理参考 MAE、提高了 SNR 代理量，并保留较多处理前左右差分；在本设计的 1 倍和 2 倍半合成注入中，恢复误差较 V6 略低。0.5 倍轻扰动与零注入仍有背景改动，项目二的缓慢漂移和真实视觉慢响应难以仅凭三个额区通道区分。代理参考并非无噪声真值，结果不能称为独立受试者泛化、已确认的神经源恢复、未知刺激解码或临床诊断准确率。
