对比基准为 PyTorch FP32（GPU）；精度在 200 条验证样本上计算，其中同模态最近邻 Top-1 一致率表示：分别排除样本自身后，PyTorch 与 TensorRT 在同模态样本中检索到同一个最近邻的比例；延迟为 batch=1 的 p50（warmup 50 / measure 300，CUDA event 计时）。表中 `[batch=1]` 表示列表里 `batch` 字段等于 1 的元素；字段级来源与补充验证信息见下方折叠 provenance。

| Model | Backend | Precision | Accuracy (cos_sim_min / same-modal NN Top-1 agreement) | Latency (batch=1, p50, ms) | Speedup vs PyTorch FP32 |
|---|---|---|---|---|---|
| PC | PyTorch | FP32 | reference | 1.577 | 1.0x |
| PC | TensorRT | FP32 (TF32 off) | 1.000000 / 100.0% | 0.397 | 4.0x |
| PC | TensorRT | FP32 (TF32 on) | 1.000000 / 100.0% | 0.353 | 4.5x |
| PC | TensorRT | FP16 mixed | 0.999998 / 99.5% | 0.273 | 5.8x |
| PC | TensorRT | INT8 (implicit calibration) | 0.999994 / 100.0% | 0.292 | 5.4x |
| HSI | PyTorch | FP32 | reference | 1.820 | 1.0x |
| HSI | TensorRT | FP32 (TF32 off) | 1.000000 / 100.0% | 0.420 | 4.3x |
| HSI | TensorRT | FP32 (TF32 on) | 0.999997 / 100.0% | 0.353 | 5.2x |
| HSI | TensorRT | FP16 mixed | 0.999847 / 99.0% | 0.248 | 7.3x |
| HSI | TensorRT | INT8 (implicit calibration) | 0.991568 / 98.0% | 0.263 | 6.9x |
| HSI | TensorRT | INT8 (QDQ) | 0.754749 / 91.0% | 0.301 | 6.0x |

PC 的 INT8 QDQ 路径已放弃，说明见「技术要点」第 3 条。
