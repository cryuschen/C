# 第一问图 4、图 5

本目录只保留这 18 张图片的绘图入口、固定输入和图片。运行：

```bash
cd /home/cryus/code-project/C
.venv/bin/python Q1/main.py
```

若使用其他 Python 环境，需安装 NumPy 和 Matplotlib。

运行后，图片统一写入 [`figures_4_5_v7/`](figures_4_5_v7/)。图 4 有一张三通道总图和三张单通道图；图 5 有受试者 A、B 各一张总图，以及每人两个项目、每项目三个通道的单图。脚本只生成这 18 张 PNG，不创建其他结果目录。

`main.py` 仅读取同目录的 `figure_data_v7.npz`。该文件合并了原版 V7 的四组 `可复核波形.npz` 中绘图所需的 Raw、V7、提示方向、试次编号和时间轴，以及图 4 固定试次的设备 Decon 波形。数据从 Git 中已提交的 V7 结果提取；原始 MAT 与 V7 脚本的 SHA256、源结果哈希和提交号保存在压缩文件的 `provenance_json` 中。图 4 的 Decon 仅用于展示对照，没有参与 V7 校正。

这个入口重绘**固定的 V7 图像**，不重新执行原始 EEG 滤波、V7 策略搜索或去噪。若需从四份原始 MAT 重算完整算法结果，仍使用项目根目录的 `EEG_P300_artifact_correction_v7.py`。V7 校正已知视觉提示方向，因此这些图也不能用于证明未知方向识别能力。

## 原代码新增独立验证

根目录运行`.venv/bin/python -B EEG_P300_artifact_correction_v7.py --independent`，结果保存在`eeg_v7_results/`的独立验证_文件中；包括275次独立时间块队列、完全合成真值、两种留出污染族、简单收缩和训练门控。此处`main.py`仍只重绘原V7固定图片。
