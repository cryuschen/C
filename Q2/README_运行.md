# 第二问计算实现

统一入口为 `run_q2.py`，原建模方案和讨论文档保留在本目录。新增 `q2model` 包实现实际求解，输出集中于项目根目录的 `Q2_result`。

```bash
/home/cryus/code-project/ML/.venv/bin/python Q2/run_q2.py --n-jobs 8
/home/cryus/code-project/ML/.venv/bin/python Q2/run_q2.py --n-jobs 8 --resume
/home/cryus/code-project/ML/.venv/bin/python Q2/audit_q2_results.py
/home/cryus/code-project/ML/.venv/bin/python -m unittest discover -s tests -v
```

调试使用不同的输出目录，例如 `--quick --output /tmp/q2-debug`。正式默认每组999次完整监督流程置换、200次完整参数重估、2000次折外预测区间重采样。

结果入口为 `Q2_result/第二问实验报告.md`，并附 `运行说明.md`、`指标字典.md`、`manifest.json`。运行模式与各阶段状态以清单为准。快速调试或部分阶段执行不是完整验证。

原始通道与V7描述数据通过试次编号对应。模型估计、内层选型和标准化只使用训练数据；测试特征接口不需要方向。已有V7采用条件标签校正，所以仅用于已知条件的描述拟合，不用于未知方向识别。

时间核的内层选型采用固定有界网格上的变量投影，最终训练由不同初值继续做连续优化。候选核可缓存，标签相关拟合结果不能在置换间复用。所有失败和边界情况保存在诊断记录中。
