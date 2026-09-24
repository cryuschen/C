# 第一问 V6 结果

推荐从 [第一问终稿实验报告](eeg_v6_results/第一问终稿实验报告.md) 开始阅读，再看四个受试者/项目目录的同口径指标、分方向 ERP、空间差分、完整与细节热力图、半合成验证。V6 源码为 `EEG_P300_artifact_correction_v6.py`，依赖 `EEG_P300_artifact_correction_v3.py`、`EEG_P300_artifact_correction_v4.py` 和 `requirements-v4.txt`。四份原始 MAT 数据位于 `data/`。

本机完整重算：

```bash
cd /home/cryus/code-project/C
/home/cryus/code-project/ML/.venv/bin/python EEG_P300_artifact_correction_v6.py --output eeg_v6_results
/home/cryus/code-project/ML/.venv/bin/python -m unittest discover -s tests -v
```

其他机器使用 Python 3.11+，先按 `requirements-v4.txt` 安装依赖。无绘图复算可使用 `--no-plots --output /tmp/eeg-v6-recheck`，数值 CSV 应与完整输出一致。

V6 将外折 V4 校正量缩至 15%。它相对预处理降低四组代理 MAE，保留较多左右差分形状，并在四组 1 倍、2 倍半合成扰动下降低恢复误差；A 项目一 0.5 倍弱扰动及部分代理 SNR 仍有退步。0.15 是看过本组数据与半合成场景后选的开发参数，外折不等于独立调参验证。缺少眼电和无伪影真值，因此结果是可提交的第一问**实验论证材料**，不能解读为已证明真实神经源恢复或临床诊断效果。

`eeg_v5_results` 保留激进趋势校正的反例，不是推荐提交结果；它在弱噪声恢复与左右差分方面有明显失败。原 V3 中文整理和 V4 基线均未由 V6 脚本覆盖。
