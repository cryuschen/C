# 第三问结果索引

先阅读 [建模与验证报告](第三问_建模与验证报告.md)。本目录包含模型构建结果和实测证据判定，二者不等价。

- `event_audit.csv`：400 次事件、标记可靠性、质量与排除原因。
- `*_waves.npz`、`*_choices.json`：逐试次实测和外层预测、训练来源及参数。
- `trial_metrics.csv`、`eeg_summary.csv`、`cognitive_intervals.csv`：EEG 误差及增量区间。
- `behavior_*`：固定前缀行为审计、特征、预测、指标、置换和区间。
- `sensitivity/`、`*_endpoint_sensitivity.csv`：重拟合或冻结评分的敏感性。
- `recovery.json`、`recovery_profiles.csv`：半合成参数恢复与非唯一性。
- `decision_*`：仅仿真的正确、错误和未答，不能当作实验标签。
- `figures/`：PNG 与矢量 PDF；各图问题及证据边界见 `图表说明.md`。
- `manifest.json`、`独立复核.json`：输入源码哈希、运行状态及独立指标核验。
