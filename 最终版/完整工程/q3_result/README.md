# 第三问最终结果

先读[完整报告](第三问_最终报告.md)。

- event_audit.csv：400次真实事件与观察规则。
- *_waves.npz、trial_metrics.csv：V2冻结参数重新生成的外层EEG预测与误差。
- window_sensitivity*.csv：行政观察上限敏感性。
- decision_*：直接使用V2认知状态的新决策仿真；不是实测行为。
- selected_parameter_decisions.csv：V2各外折EEG参数驱动的仿真。
- synthetic_marker_stress*：标记缺失的软件压力测试，不进入真实统计。
- reference/：18份必要冻结参数、预测基准和对照表。
- [优化报告](优化_机制验证报告.md)：新增拟合、参数恢复和目标匹配对照。
- 入口：`q3/main.py`；默认生成主结果和优化对照，`--stage reconstruct`仅重建主结果。
- 独立复核.json、测试验收.txt、visual_qa.json：验证记录。
