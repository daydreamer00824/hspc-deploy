"""Benchmark: PyTorch FP32(GPU) / ORT-CUDA / TRT(FP32-noTF32, FP32-TF32, FP16, INT8) 延迟与吞吐。

batch in {1, 8, 32, 64}；CUDA event 计时；warmup 50 / measure 300。
batch=1 是端侧最关心的延迟点；batch=64 反映吞吐上限。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hspc_common as hc
from trt_runner import TrtRunner

ENGINE_DIR = hc.ROOT / "engines"
ONNX_DIR = hc.ROOT / "onnx"
OUT = hc.ROOT / "results" / "stage3_benchmark.json"

BATCHES = [1, 8, 32, 64]
WARMUP = 50
MEASURE = 300

# scene engine 的 profile 是 min1/opt2048/max2048，额外测一个 batch=2048 反映整景推理吞吐；
# BATCHES 里的 1/8/32/64 仍在 scene engine 的合法 shape 范围内（min<=x<=max），一并测出来
# 方便跟 fp16(batch64 profile) 直接比较同一 batch 下的延迟差异。
EXTRA_BATCHES = {"fp16_scene": [2048]}

TRT_MODES = {
    # fp16 = 现行正式交付（选择性混合精度，batch profile 1/8/64）
    # fp16_legacy = 阶段5之前的旧文件，已知不可复现，仅作对照，不要用于新部署
    # fp16_scene = 混合精度 + 整景大 batch profile（1/2048/2048），整景批量推理默认
    "pc": ["fp32_notf32", "fp32_tf32", "fp16", "fp16_legacy", "fp16_scene", "int8_implicit"],
    "hsi": ["fp32_notf32", "fp32_tf32", "fp16", "fp16_legacy", "fp16_scene", "int8_implicit", "int8_qdq"],
}


def latency_stats(times_ms: np.ndarray, batch: int) -> dict:
    return {
        "batch": batch,
        "mean_ms": float(times_ms.mean()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "throughput_samples_per_s": float(batch * 1000.0 / times_ms.mean()),
    }


def bench_torch(model, which: str, batch: int) -> dict:
    x = hc.dummy_input(which, batch).cuda()
    with torch.no_grad(), hc.math_sdpa():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        # 与 bench_trt 相同：逐次记录 event，末尾统一同步，避免同步气泡污染测量
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE)]
        for i in range(MEASURE):
            starts[i].record()
            model(x)
            ends[i].record()
        torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return latency_stats(np.array(times), batch)


def bench_ort_cuda(onnx_path: Path, which: str, batch: int) -> dict:
    sess = ort.InferenceSession(str(onnx_path), providers=[("CUDAExecutionProvider", {"use_tf32": "0"})])
    x = hc.dummy_input(which, batch).numpy()
    # 注意：ORT 的 sess.run 是阻塞式 API，每次调用必然包含同步与 H2D/D2H，
    # 无法像 TRT/PyTorch 那样消除同步气泡，因此 ORT 一行与 TRT 行不完全同口径（偏保守）。
    times = []
    for _ in range(WARMUP):
        sess.run(None, {hc.INPUT_NAME: x})
    for _ in range(MEASURE):
        t0 = time.perf_counter()
        sess.run(None, {hc.INPUT_NAME: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    return latency_stats(np.array(times), batch)


def bench_trt(plan_path: Path, which: str, batch: int) -> dict:
    runner = TrtRunner(plan_path)
    x = hc.dummy_input(which, batch).numpy()
    in_dtype = runner.engine.get_tensor_dtype(hc.INPUT_NAME)
    from trt_runner import _TRT_TO_TORCH
    xt = torch.from_numpy(x).to(_TRT_TO_TORCH[in_dtype]).cuda().contiguous()
    out_shape = None
    runner.context.set_input_shape(hc.INPUT_NAME, x.shape)
    out_shape = tuple(runner.context.get_tensor_shape(hc.OUTPUT_NAME))
    out_dtype = _TRT_TO_TORCH[runner.engine.get_tensor_dtype(hc.OUTPUT_NAME)]
    yt = torch.empty(out_shape, dtype=out_dtype, device="cuda")
    runner.context.set_tensor_address(hc.INPUT_NAME, xt.data_ptr())
    runner.context.set_tensor_address(hc.OUTPUT_NAME, yt.data_ptr())
    stream = runner.stream
    for _ in range(WARMUP):
        with torch.cuda.stream(stream):
            runner.context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # 关键：event 逐次记录但不逐次同步，末尾统一 sync 再读取。
    # 每轮 synchronize 会让 GPU 在两次 launch 间空转，把 launch 延迟和同步开销算进测量值
    # （并导致 GPU 时钟掉档），实测使结果比 trtexec 高 20~75%。此处与 trtexec 的计时方式对齐。
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE)]
    with torch.cuda.stream(stream):
        for i in range(MEASURE):
            starts[i].record(stream)
            runner.context.execute_async_v3(stream.cuda_stream)
            ends[i].record(stream)
    stream.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return latency_stats(np.array(times), batch)


def engine_stats(which: str, mode: str) -> dict:
    p = ENGINE_DIR / f"{which}_{mode}.plan"
    return {"size_kb": p.stat().st_size / 1024}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["pc", "hsi"])
    args = ap.parse_args()

    hc.setup_determinism()
    results = {}
    for which in args.which:
        results[which] = {}
        model = hc.get_model(which, "cuda")

        results[which]["pytorch_fp32_gpu"] = [bench_torch(model, which, b) for b in BATCHES]
        print(f"[{which}] pytorch_fp32_gpu done")

        results[which]["ort_cuda_fp32"] = [
            bench_ort_cuda(ONNX_DIR / f"{which}_encoder.onnx", which, b) for b in BATCHES
        ]
        print(f"[{which}] ort_cuda_fp32 done")

        for mode in TRT_MODES[which]:
            plan = ENGINE_DIR / f"{which}_{mode}.plan"
            if not plan.exists():
                print(f"[{which}] engine {mode} missing, skip")
                continue
            batches = BATCHES + EXTRA_BATCHES.get(mode, [])
            stats = [bench_trt(plan, which, b) for b in batches]
            for s in stats:
                s.update(engine_stats(which, mode))
            results[which][f"trt_{mode}"] = stats
            print(f"[{which}] trt_{mode} done: batch1 mean={stats[0]['mean_ms']:.3f}ms "
                  f"batch{stats[-1]['batch']} mean={stats[-1]['mean_ms']:.3f}ms")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "batches": BATCHES, "warmup": WARMUP, "measure": MEASURE, "results": results,
    }, indent=2))
    print(f"\nwritten -> {OUT.relative_to(hc.ROOT)}")


if __name__ == "__main__":
    main()
