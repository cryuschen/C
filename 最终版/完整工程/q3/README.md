# 第三题最终代码

本目录是第三题唯一代码目录，结果统一写入 `../q3_result/`。

- `main.py`：统一入口；默认完成冻结参数重建、决策仿真、优化对照、绘图和数值审计。
- `common.py`、`events.py`：数据读取、因果滤波、试次质量与应答截窗。
- `model.py`、`fit.py`：视觉—记忆—认知状态与训练块内拟合。拟合器从旧版提取合并，数值方法不变。
- `decision.py`：条件决策和目标匹配仿真。
- `optimization.py`：新增拟合、参数剖面、恢复实验及敏感性对照。
- `reference.py`、`reference_manifest.json`：冻结参考的读取与不可变哈希核对。
- `report.py`、`audit.py`、`tests/`：报告、独立数值复核与测试。

在完整工程目录中使用已配置的Python环境运行：

```bash
python q3/main.py
python q3/main.py --stage reconstruct
python q3/main.py --stage optimize
python q3/main.py --stage render
python q3/main.py --stage verify
```

也可从外层调用 `python3 最终版/main.py --stage q3`，自动寻找本机虚拟环境。

所有数据依赖均在完整工程内：`data/` 的四份原始MAT、题目DOCX中的形状图像、`q2/q2model/` 的视觉编码模块，以及 `q3_result/reference/` 的18份冻结参数、波形和对照表。主结果重建不会重新搜索冻结参数；优化对照会执行新的训练块内拟合。`--output`支持新目录并复制必要参考，不能覆盖原始数据、源码或冻结参考。

冻结参考不能删除，它用于保持原主结果与重新计算的比较独立。无需 `q3_v3`、`q3_result_v2`、`q3_result_v3` 或旧版q3代码。原始V2来源记录保存在reference_manifest.json，历史路径仅作为来源记载，不作为运行依赖。

阅读结果请打开[第三问最终报告](../q3_result/第三问_最终报告.md)及[优化机制验证](../q3_result/优化_机制验证报告.md)。实测正确与超时标签仍未知，仿真不能作为生理机制证明。
