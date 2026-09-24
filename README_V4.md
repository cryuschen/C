# 第一问 V4 运行与查看

入口：`EEG_P300_artifact_correction_v4.py`。保留 V3 文件作为滤波、原算法和指标函数依赖，不运行其主程序。

本机已验证的完整运行命令：

```bash
cd /home/cryus/code-project/C
/home/cryus/code-project/ML/.venv/bin/python EEG_P300_artifact_correction_v4.py --output eeg_v4_results
```

其他环境使用 Python 3.11+ 和 `requirements-v4.txt`，在已安装依赖的环境内运行同一入口。当前实现使用 `numpy.trapezoid`，要求 NumPy ≥2.0。

结果优先查看 `eeg_v4_results/汇总与说明/第四版结果说明.md`。四个受试者/项目目录各有6张图、分方向评价、拟合参数、审计与波形。`指标字典_V4.md`解释计算公式、分母与限制。

如需验证重复性，可运行：

```bash
/home/cryus/code-project/ML/.venv/bin/python EEG_P300_artifact_correction_v4.py --no-plots --output /tmp/eeg-v4-recheck
/home/cryus/code-project/ML/.venv/bin/python -m unittest discover -s tests -v
```

`--no-plots`仅计算数值，不生成拟合参数/图片。V3原结果不覆盖，工作区原有未提交修改不重置。V4是保守校正与可解释评价版本，并非所有指标均优于V3，报告列出了改善和退步。
