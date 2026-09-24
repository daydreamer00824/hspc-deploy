对比基准为 PyTorch FP32（GPU）；精度在 200 条验证样本上计算，其中同模态最近邻 Top-1 一致率表示：分别排除样本自身后，PyTorch 与 TensorRT 在同模态样本中检索到同一个最近邻的比例；延迟为 batch=1 的 p50（warmup 50 / measure 300，CUDA event 计时）。表中 `[batch=1]` 表示列表里 `batch` 字段等于 1 的元素；每个单元格的来源见文末脚注。

| Model | Backend | Precision | Accuracy (cos_sim_min / same-modal NN Top-1 agreement) | Latency (batch=1, p50, ms) | Speedup vs PyTorch FP32 |
|---|---|---|---|---|---|
| PC | PyTorch | FP32 | reference | 1.577[^pc-pytorch_fp32_gpu-lat] | 1.0x[^pc-pytorch_fp32_gpu-spd] |
| PC | TensorRT | FP32 (TF32 off) | 1.000000 / 100.0%[^pc-trt_fp32_notf32-acc] | 0.397[^pc-trt_fp32_notf32-lat] | 4.0x[^pc-trt_fp32_notf32-spd] |
| PC | TensorRT | FP32 (TF32 on) | 1.000000 / 100.0%[^pc-trt_fp32_tf32-acc] | 0.353[^pc-trt_fp32_tf32-lat] | 4.5x[^pc-trt_fp32_tf32-spd] |
| PC | TensorRT | FP16 mixed | 0.999998 / 99.5%[^pc-trt_fp16-acc] | 0.273[^pc-trt_fp16-lat] | 5.8x[^pc-trt_fp16-spd] |
| PC | TensorRT | INT8 (implicit calibration) | 0.999994 / 100.0%[^pc-trt_int8_implicit-acc] | 0.292[^pc-trt_int8_implicit-lat] | 5.4x[^pc-trt_int8_implicit-spd] |
| HSI | PyTorch | FP32 | reference | 1.820[^hsi-pytorch_fp32_gpu-lat] | 1.0x[^hsi-pytorch_fp32_gpu-spd] |
| HSI | TensorRT | FP32 (TF32 off) | 1.000000 / 100.0%[^hsi-trt_fp32_notf32-acc] | 0.420[^hsi-trt_fp32_notf32-lat] | 4.3x[^hsi-trt_fp32_notf32-spd] |
| HSI | TensorRT | FP32 (TF32 on) | 0.999997 / 100.0%[^hsi-trt_fp32_tf32-acc] | 0.353[^hsi-trt_fp32_tf32-lat] | 5.2x[^hsi-trt_fp32_tf32-spd] |
| HSI | TensorRT | FP16 mixed | 0.999847 / 99.0%[^hsi-trt_fp16-acc] | 0.248[^hsi-trt_fp16-lat] | 7.3x[^hsi-trt_fp16-spd] |
| HSI | TensorRT | INT8 (implicit calibration) | 0.991568 / 98.0%[^hsi-trt_int8_implicit-acc] | 0.263[^hsi-trt_int8_implicit-lat] | 6.9x[^hsi-trt_int8_implicit-spd] |
| HSI | TensorRT | INT8 (QDQ) | 0.754749 / 91.0%[^hsi-trt_int8_qdq-acc] | 0.301[^hsi-trt_int8_qdq-lat] | 6.0x[^hsi-trt_int8_qdq-spd] |

PC 的 INT8 QDQ 路径已放弃，说明见「技术要点」第 3 条。

[^pc-pytorch_fp32_gpu-lat]: `results/stage3_benchmark.json` → `results.pc.pytorch_fp32_gpu[batch=1].p50_ms`
[^pc-pytorch_fp32_gpu-spd]: `results/stage3_benchmark.json` → `results.pc.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.pc.pytorch_fp32_gpu[batch=1].p50_ms`（由这两个字段相除得出）
[^pc-trt_fp32_notf32-acc]: `results/stage2_trt_accuracy.json` → `results.pc.fp32_notf32.cos_sim_min`, `results.pc.fp32_notf32.top1_agreement`
[^pc-trt_fp32_notf32-lat]: `results/stage3_benchmark.json` → `results.pc.trt_fp32_notf32[batch=1].p50_ms`
[^pc-trt_fp32_notf32-spd]: `results/stage3_benchmark.json` → `results.pc.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.pc.trt_fp32_notf32[batch=1].p50_ms`（由这两个字段相除得出）
[^pc-trt_fp32_tf32-acc]: `results/stage2_trt_accuracy.json` → `results.pc.fp32_tf32.cos_sim_min`, `results.pc.fp32_tf32.top1_agreement`
[^pc-trt_fp32_tf32-lat]: `results/stage3_benchmark.json` → `results.pc.trt_fp32_tf32[batch=1].p50_ms`
[^pc-trt_fp32_tf32-spd]: `results/stage3_benchmark.json` → `results.pc.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.pc.trt_fp32_tf32[batch=1].p50_ms`（由这两个字段相除得出）
[^pc-trt_fp16-acc]: `results/stage2_trt_accuracy.json` → `results.pc.fp16.cos_sim_min`, `results.pc.fp16.top1_agreement`；同配置另有 5 次独立构建的稳定性验证，cos_sim_min 为 0.999995–0.999998（`results/stage7_mixed_precision.json` → `pc.stability.cos_mins`）
[^pc-trt_fp16-lat]: `results/stage3_benchmark.json` → `results.pc.trt_fp16[batch=1].p50_ms`
[^pc-trt_fp16-spd]: `results/stage3_benchmark.json` → `results.pc.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.pc.trt_fp16[batch=1].p50_ms`（由这两个字段相除得出）
[^pc-trt_int8_implicit-acc]: `results/stage2_trt_accuracy.json` → `results.pc.int8_implicit.cos_sim_min`, `results.pc.int8_implicit.top1_agreement`
[^pc-trt_int8_implicit-lat]: `results/stage3_benchmark.json` → `results.pc.trt_int8_implicit[batch=1].p50_ms`
[^pc-trt_int8_implicit-spd]: `results/stage3_benchmark.json` → `results.pc.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.pc.trt_int8_implicit[batch=1].p50_ms`（由这两个字段相除得出）
[^hsi-pytorch_fp32_gpu-lat]: `results/stage3_benchmark.json` → `results.hsi.pytorch_fp32_gpu[batch=1].p50_ms`
[^hsi-pytorch_fp32_gpu-spd]: `results/stage3_benchmark.json` → `results.hsi.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.hsi.pytorch_fp32_gpu[batch=1].p50_ms`（由这两个字段相除得出）
[^hsi-trt_fp32_notf32-acc]: `results/stage2_trt_accuracy.json` → `results.hsi.fp32_notf32.cos_sim_min`, `results.hsi.fp32_notf32.top1_agreement`
[^hsi-trt_fp32_notf32-lat]: `results/stage3_benchmark.json` → `results.hsi.trt_fp32_notf32[batch=1].p50_ms`
[^hsi-trt_fp32_notf32-spd]: `results/stage3_benchmark.json` → `results.hsi.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.hsi.trt_fp32_notf32[batch=1].p50_ms`（由这两个字段相除得出）
[^hsi-trt_fp32_tf32-acc]: `results/stage2_trt_accuracy.json` → `results.hsi.fp32_tf32.cos_sim_min`, `results.hsi.fp32_tf32.top1_agreement`
[^hsi-trt_fp32_tf32-lat]: `results/stage3_benchmark.json` → `results.hsi.trt_fp32_tf32[batch=1].p50_ms`
[^hsi-trt_fp32_tf32-spd]: `results/stage3_benchmark.json` → `results.hsi.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.hsi.trt_fp32_tf32[batch=1].p50_ms`（由这两个字段相除得出）
[^hsi-trt_fp16-acc]: `results/stage2_trt_accuracy.json` → `results.hsi.fp16.cos_sim_min`, `results.hsi.fp16.top1_agreement`；同配置另有 5 次独立构建的稳定性验证，cos_sim_min 为 0.999784–0.999801（`results/stage7_mixed_precision.json` → `hsi.stability.cos_mins`）
[^hsi-trt_fp16-lat]: `results/stage3_benchmark.json` → `results.hsi.trt_fp16[batch=1].p50_ms`
[^hsi-trt_fp16-spd]: `results/stage3_benchmark.json` → `results.hsi.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.hsi.trt_fp16[batch=1].p50_ms`（由这两个字段相除得出）
[^hsi-trt_int8_implicit-acc]: `results/stage2_trt_accuracy.json` → `results.hsi.int8_implicit.cos_sim_min`, `results.hsi.int8_implicit.top1_agreement`
[^hsi-trt_int8_implicit-lat]: `results/stage3_benchmark.json` → `results.hsi.trt_int8_implicit[batch=1].p50_ms`
[^hsi-trt_int8_implicit-spd]: `results/stage3_benchmark.json` → `results.hsi.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.hsi.trt_int8_implicit[batch=1].p50_ms`（由这两个字段相除得出）
[^hsi-trt_int8_qdq-acc]: `results/stage2_trt_accuracy.json` → `results.hsi.int8_qdq.cos_sim_min`, `results.hsi.int8_qdq.top1_agreement`。该路径构建稳定、但精度本身偏低：同一 QDQ 图经 TensorRT 独立重建 3 次，cos_sim_min 为 0.7474–0.7547；不经 TensorRT、直接用 ONNX Runtime 运行该 QDQ 模型，cos_sim_min 为 0.7858–0.8582，同样偏低（`results/hsi_qdq_check.json` → `trt_rebuilds[*].cos_sim_min`、`ort_direct.*.cos_sim_min`）。因此这是 QDQ 量化误差本身较大，不是技术要点第 3 条中 PC 路径那种重建后输出数值异常的问题
[^hsi-trt_int8_qdq-lat]: `results/stage3_benchmark.json` → `results.hsi.trt_int8_qdq[batch=1].p50_ms`
[^hsi-trt_int8_qdq-spd]: `results/stage3_benchmark.json` → `results.hsi.pytorch_fp32_gpu[batch=1].p50_ms` ÷ `results.hsi.trt_int8_qdq[batch=1].p50_ms`（由这两个字段相除得出）
