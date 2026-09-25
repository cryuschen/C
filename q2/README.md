# 第二问 V3 代码

本目录仅保留 V3 的原始数据审计、图像驱动机制留出检验、探索性增量对照和 ERP 判别代码。四份 MAT 位于项目根目录 `data/`；原题 DOCX 位于项目根目录。输出与报告统一位于 [`q2_result`](../q2_result/README.md)。

运行环境包含 `requirements.txt` 中的科学计算库。在项目根目录执行：

```bash
python q2/audit_sequence.py
python q2/mechanism_v3_transfer.py --output /tmp/q2-mechanism-v3-new
python q2/mechanism_v3_incremental.py --base /tmp/q2-mechanism-v3-new --output /tmp/q2-incremental-v3-new
python q2/decoder_v3_erp.py --output /tmp/q2-decoder-v3-new --permutations 1999 --bootstraps 2000
python q2/decoder_v3_incremental_audit.py /tmp/q2-decoder-v3-new --bootstraps 5000
```

三个 `--output` 目标须为空。默认目标分别为 `q2_result/results/mechanism`、`mechanism_incremental` 和 `decoder`；复算已有结果时请先选新的输出目录，核对后再替换。为减少 CPU 线程竞争，可设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。

检查数据流：

```bash
python -m unittest discover -s q2/tests -p 'test_q2_decoder_v3_erp.py' -v
python -c "import sys; sys.path.insert(0, 'q2/tests'); import test_q2_mechanism_v3_transfer as t; t.test_shape_driven_neural_basis_and_ridge_limit(); t.test_held_block_labels_do_not_change_its_predictions()"
```

结果的统计含义与局限以[最终复核](../q2_result/docs/第二问V3完整复核与可提交结论.md)为准。
