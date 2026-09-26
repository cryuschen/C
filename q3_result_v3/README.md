# 第三问 V3 结果

先读[完整报告](第三问_V3完整报告.md)。

- event_audit.csv：400次真实事件与观察规则。
- *_waves.npz、trial_metrics.csv：V2冻结参数重新生成的外层EEG预测与误差。
- window_sensitivity*.csv：行政观察上限敏感性。
- decision_*：直接使用V2认知状态的新决策仿真；不是实测行为。
- selected_parameter_decisions.csv：V2各外折EEG参数驱动的仿真。
- synthetic_marker_stress*：标记缺失的软件压力测试，不进入真实统计。
- inherited_v2/：旧证据快照，没有重跑旧置换或重新拟合。
- 独立复核.json、测试验收.txt、visual_qa.json：验证记录。
