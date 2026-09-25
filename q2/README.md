# 第二问 图像驱动的形成模型与论文证据链

当前主入口采用三级 E/I 群体模型与严格时间块验证。旧版脚本及结果保留，当前入口不再调用。

在项目根目录运行：

```powershell
.venv/Scripts/python.exe -B -X utf8 Q2/main.py
.venv/Scripts/python.exe -B -X utf8 -m unittest discover -s Q2/tests -v
```

其他环境使用 Python 3.11+，安装本目录 `requirements.txt`。绘图需要 Microsoft YaHei、Noto Sans CJK SC 或 SimHei。

所有新结果直接写入已有 `q2_result` 根目录，采用 `Q2_` 前缀，不创建文件夹，不修改 Q1 或历史结果。`--output` 必须指定已存在目录。先读 `Q2_论文正文.md`、`Q2_结论证据对照.csv` 和 `Q2_结果索引.md`。

- 严格主分析：原始 Fz/F3/F4，独立连续块预处理，24 秒边界保护，A1/A2/B1/B2 保留 67/70/69/69 次。
- Q1 before/v7 仅用于描述性衔接。现成 V7 方向知情模板不进入未知方向判别。
- 三级 E/I 网络由镜像形状输入驱动，左右条件共享参数及有效观测矩阵；在外层训练块内再留块选型。
- 标准化、峰值补缺、侧化稳定项、机制模板、PCA 和 LDA 仅在训练折拟合。
- 默认 1999 次块内置换、199 次循环移位、1000 次块重采样。maxT 只校正本次五组特征，不能覆盖历史探索。

```powershell
# 快速检查，会覆盖本轮同名文件；正式论文请随后完整运行
.venv/Scripts/python.exe -B -X utf8 Q2/main.py --quick
# 从已保存数值重绘、生成论文并核验
.venv/Scripts/python.exe -B -X utf8 Q2/main.py --redraw
# 只计算，随后可 --redraw
.venv/Scripts/python.exe -B -X utf8 Q2/main.py --no-plots
```

逐试次特征组的列按构造顺序编号。23 维机制组为：三均值、三峰幅、三潜伏期、三缺失标记、LI、F4−F3、六投影系数、三残差 RMS。机制 NPZ 和内层选择 JSON 提供预测波形与参数来源。LaTeX 源在 `q2_result` 内用 XeLaTeX、BibTeX、XeLaTeX 两次编译；分析不依赖 TeX 安装。

SΔ 与标准 R² 不同；模型内部量不是实测神经活动，形状偏好不是半球，观测矩阵不是个体解剖导联。消融只能单独支持模型内部依赖。报告保留负结果，不把同样本拟合、PCA 图或弱判别写成确定机制。
