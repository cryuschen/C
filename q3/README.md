# 第三问 宏观认知模型与完整验证

从题目四份原始 MAT 重建长认知窗，连接 Q2 图像驱动的 LGN/E-I 视觉模型、慢记忆与认知状态、有效头皮投影，以及独立的模拟决策过程。正式结果写入 `q3_result`；不修改 Q1、Q2 或原始数据。

## 运行

在项目根目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B q3/main.py
.venv/bin/python -B -m unittest discover -s q3/tests -v
.venv/bin/python -B q3/audit_results.py
```

其他 Python 3.11+ 环境安装 `q3/requirements.txt`。数值计算单线程，减少笔记本负载。绘图自动使用 Noto Sans CJK SC 等已安装中文字体。

默认执行 `eeg → sensitivity → behavior → simulation → render`。可用 `--stage` 单独运行一个阶段；重新运行会覆盖该输出目录中第三问的同名文件。`--output` 可指定新目录，使用同一目录续跑必须保持 quick、置换数和重采样设置一致。完整网格的实际计算时间取决于设备。

```bash
# 快速端到端检查，强制使用独立目录，不覆盖正式结果
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B q3/main.py --quick --output /tmp/q3-smoke
# 只重绘，不重新拟合
.venv/bin/python -B q3/main.py --stage render
```

默认 1999 次行为置换、2000 次时间块重采样；`--permutations`、`--bootstraps` 可改。快速版本缩小候选集合，仅用于检查程序，不能作为正式结果。

## 代码与接口

| 模块 | 接口和职责 |
|---|---|
| `data.py` | `audit_events(key, data)` 解析全部事件；`build_trials` 返回变长认知窗及审计；`prefix_quality_trials` 独立按固定前缀筛查 |
| `model.py` | `visual_response(cue,duration_samples,target_samples)` 不接收应答；`slow_states` 实现延迟慢网络；`predict` 仅白名单读取刺激输入 |
| `fit.py` | `Bank` 保存精确岭充分统计量；`select` 执行全候选训练内筛查和前两组稳健重拟合；`evaluate` 生成五折预测 |
| `behavior.py` | 前缀 EEG 参数在行为内层重新拟合；`predict_all_labels` 对真实及置换标签重新选监督回归参数 |
| `validation.py` | 时间块区间、方向差分、半合成恢复和仅仿真的三类应答 |
| `reporting.py` | 从保存数值生成中文报告、图注、PNG 与 PDF |
| `audit_results.py` | 独立重算误差、行为指标、置换 p、样本数和来源哈希 |

`Trials.x` 为试次×固定时间网格×三通道，允许分析窗以外为 NaN；不做时轴拉伸。模型始终生成提示前 250 ms 至提示后 5 s 的固定网格。`response_sample` 只用于评分掩码，Task-1 标注 `platform_end_proxy`，Task-2 为 `explicit_click`。`correctness`、`timeout` 保留空值和对应原因。

## 方法约定与边界

- 图像编码与 E/I 方程复用当前 Q2；题图只是示意，未提供真实视角或逐次目标画面。
- 目标 ±1 平台起点的语义是工作假设；共同目标输入固定 200 ms。点击方向不进入视觉驱动。
- 主分析为因果滤波；零相位只作离线敏感性。模拟观测使用对应滤波。
- 每份记录五个连续块；外层测试块从所有内层选型、缩放、混合矩阵和行为回归中排除。
- EEG 筛查是全网格嵌套岭回归，随后固定前两组候选／惩罚做 Huber IRLS 验证及重拟合；不宣称对全网格逐一求得全局稳健最优解。
- 主 EEG 质量筛选可使用其完整认知窗；行为队列独立，只筛查固定前缀。后者不借用完整窗质量标记。
- 行为特征严格使用目标后 **600 ms 以内**：最后纳入第 153 个采样间隔（597.65625 ms），独占切片末端为 601.5625 ms。独占端点本身不进入特征。所有实际点击都晚于该窗口。
- 端点敏感性冻结主拟合后重新评分；零相位和 Task-1 目标前对照完全重拟合。评分队列可能随质量/边界变化，审计文件记录各自数量。
- 参数和模型内部状态只具候选功能解释；海马活动、疾病诊断和真实错误率不是已测得结果。
- 正式报告保存负结果，并区分“构建完成”和“获得实测支持”。

## 可复核输出

`*_waves.npz` 保存每个试次的实际值、各模型外层预测和掩码；`*_choices.json` 保存训练试次、内层损失、参数、读出系数、有效自由度、奇异值和 IRLS 收敛过程。状态选择触及候选边界、低奇异值和恢复失败应作为不确定性报告。

最终图像需逐张查看；程序的数值审计不代替视觉检查。`独立复核.json` 是独立重算结果，`manifest.json` 保存完成阶段及输入源码 SHA256。
