# 第二问 V3 结果

本目录只保留第二问 V3 的报告和核验输出；[运行代码](../q2/README.md)独立放在 `q2/`。原题 DOCX 和四份 MAT 仍在项目根目录与 `data/`，属于只读输入。

从[完整复核与可提交结论](docs/第二问V3完整复核与可提交结论.md)开始阅读。方法细节见[模型方程](docs/模型方程与输入.md)、[机制报告](docs/mechanism_v3_report.md)、[解码报告](docs/decoder_v3_erp_results.md)和[原始数据审计](docs/数据审计.md)；[验收口径](docs/第二问V3验收口径.md)记录了判定标准。

`results/audit/` 是事件和刺激前对照；`results/mechanism/` 是图像驱动模型的时间块留出预测；`results/mechanism_incremental/` 是事后探索性增量对照；`results/decoder/` 是逐试次特征、预测、置换分布与图。已有输出保留完整的试次级结果和运行清单。

这些结果支持**可运行的候选机制与候选特征**，尚不能证实实测左右三角 EEG 差异的真实生理成因，也没有验证出稳定的刺激特异判别表示。
