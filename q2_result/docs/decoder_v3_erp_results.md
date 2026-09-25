# 第二问 V3：固定时序 ERP 表示的左右判别复核

## 方法与预先固定的表示

使用原始 Fz、F3、F4 和 VisCue（左 `−1`、右 `+1`），沿用五个连续时间块、24 秒滤波边界缓冲与固定质量排除。四组分别保留 A1=67、A2=70、B1=69、B2=69 个试次。刺激前负对照取 `−250–0 ms`，来自**在提示开始前截止并单独滤波**的片段；刺激后窗口取 `50–750 ms`。

对任一未知试次的三通道 EEG `X_i(t)`，先投影到预设的共同、中线、侧化三个正交空间模态 `BᵀX_i(t)`，再对各模态的刺激后时间序列取前六个正交离散余弦系数：

\[
\phi_i=\operatorname{vec}\!\left\{\operatorname{DCT}_{0:5}\left[B^\top X_i(t),\ t\in[50,750)\,\mathrm{ms}\right]\right\}\in\mathbb R^{18}.
\]

严格刺激前特征 `u_i∈R¹⁸` 同法由单独滤波的过去片段计算。前六项在 700 ms 刺激后窗口表征约 `0–3.6 Hz` 的平滑 ERP 时程；`6` 项、协方差 `80%` 向单位阵收缩、以及下述岭系数均在查看本次结果前固定，没有依据留出块表现择优。刺激前的六项跨度更短、对应频带更宽，因此两者特征维数相同，但频率覆盖并不完全相同。此方法保留时间波形信息，区别于此前三个宽时间窗均值及三电极协方差表示；它是带收缩的 ERP 均值差判别，**不是 xDAWN**。这种低样本收缩判别的统计动机可参见 [Blankertz 等人的单试次 ERP 分类研究](https://www.sciencedirect.com/science/article/abs/pii/S1053811910009067)。

每份记录单独留出一个连续时间块，用其余四块拟合全部中心、尺度、刺激前到刺激后的线性预测及左右 ERP 模板。对训练折标准化后的刺激后特征 `p` 与刺激前特征 `u`，拟合不使用方向标签的 `A=(UᵀU+nI)⁻¹UᵀP`，定义去除过去线性可预测部分的表示 `r_i=p_i−u_iA`。对 `u`、`p`、`r` 分别以训练折左右均值差 `μ_R−μ_L` 和 `80%` 收缩的类内协方差构造判别向量，未知试次按分数正负预测右或左。每次置换都在每个时间块内部打乱方向，重新拟合全部**有监督**模板与分类器；三种表示的合并准确率再做 `maxT` 校正。共 `1999` 次置换。四份 MAT 已在此前反复开发，本次 p 值仅是内部探索证据。

## 留出结果

BA 为平衡准确率；AUC 基于连续判别分数。`p_family` 对合并的三种表示做 `maxT`，分组行对四组 × 三种表示共十二项做 `maxT`。

| 数据 | 严格刺激前 BA | 刺激后 ERP BA | 刺激后扣除过去 BA | 扣除后 AUC | 扣除后原始 p | 扣除后 p_family |
|---|---:|---:|---:|---:|---:|---:|
| A1，67 次 | 58.4% | 48.1% | 49.0% | 0.423 | 0.583 | 0.9995 |
| A2，70 次 | 43.1% | 51.0% | 49.7% | 0.473 | 0.5175 | 0.9995 |
| B1，69 次 | 53.7% | 53.7% | 55.2% | 0.597 | 0.232 | 0.9305 |
| B2，69 次 | 54.5% | 51.8% | 57.6% | 0.546 | 0.182 | 0.8245 |
| **合并，275 次** | **52.4%** | **51.3%** | **53.2%** | **0.525** | **0.2015** | **0.4040** |

合并扣除过去的表示相对严格刺激前仅高 **0.74 个百分点**；5000 次成对时间块 bootstrap 的差值 95% 区间为 **−6.77 至 +8.35 个百分点**。该差值在原有块内置换零分布中的描述性单侧 `p=0.469`，这检验无方向关联零假设，**不是严格的条件独立检验**。扣除过去的表示本身的合并 BA 95% 时间块 bootstrap 区间为 **47.2%–59.6%**。各数据集效果方向与大小均不稳定，全部校正 p 值未达 `0.05`。

因此，本次固定的 18 维时序方法**没有证明刺激后 EEG 提供超出刺激前状态的可靠左右方向判别信息**。它给出了未知试次可直接计算的候选特征和可复核的折外分数，但不能称作已经验证的“左、右三角形视觉响应特异表示”。这项分类结果也不能验证 LGN→皮层→头皮 EEG 的真实神经形成路径。

## 可复核文件

- [运行脚本](../../q2/decoder_v3_erp.py)、[成对增量审计](../../q2/decoder_v3_incremental_audit.py)、[数据流测试](../../q2/tests/test_q2_decoder_v3_erp.py)。
- [每组与合并完整指标](../results/decoder/metrics.csv)、[逐试次预测](../results/decoder/oof_predictions.csv)、[逐试次 18 维折外特征](../results/decoder/oof_features.csv)。
- [全部置换零分布](../results/decoder/permutation_null.csv)、[成对增量数值](../results/decoder/incremental_audit.json)、[输入哈希与参数清单](../results/decoder/manifest.json)、[实际检查过的结果图](../results/decoder/decoder_v3_erp.png)。

运行命令：

```bash
MPLCONFIGDIR=/tmp/q2_decoder_v3_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/cryus/code-project/ML/.venv/bin/python q2/decoder_v3_erp.py --output /tmp/q2_decoder_v3_repro --permutations 1999 --bootstraps 2000
OPENBLAS_NUM_THREADS=1 /home/cryus/code-project/ML/.venv/bin/python q2/decoder_v3_incremental_audit.py /tmp/q2_decoder_v3_repro --bootstraps 5000
/home/cryus/code-project/ML/.venv/bin/python -m unittest q2/tests/test_q2_decoder_v3_erp.py -v
```

`--output` 必须是空目录或新目录；上述参数配置的完整流程已成功结束，四项数据流测试通过。
