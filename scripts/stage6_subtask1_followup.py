"""阶段6 追加诊断：
1. deploy profile 强制 FP16（precision=HALF + PREFER_PRECISION_CONSTRAINTS）构建 + 精度 + batch 64/1024 GPU时间对比
2. (c) 的量化：已有4次default构建的 cos_min/top1/batch1&64延迟；
   缓解A：同一 timing cache 连续构建3次default；缓解B：default+强制HALF约束构建3次
3. GPU负载：构建期间用 nvidia-smi dmon 采样，报告负载水平

不改模型结构/权重/超参数；所有 engine 只写到 /tmp/hspc_stage6_scratch，不进 engines/。
运行环境：conda modelopt，PYTHONPATH=/usr/lib/python3.10/dist-packages。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_trt as bt  # noqa: E402
import hspc_common as hc  # noqa: E402
from trt_runner import TrtRunner, _TRT_TO_TORCH  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ROOT = hc.ROOT
ONNX_DIR = ROOT / "onnx"
SCRATCH = Path("/tmp/hspc_stage6_scratch/engines")
SCRATCH.mkdir(parents=True, exist_ok=True)
REF_HSI = np.load(ROOT / "results/ref/hsi_fp32_cpu.npy")
VAL_X = hc.load_samples("hsi", "val")


def eval_acc(plan_path: Path, max_batch: int) -> dict:
    runner = TrtRunner(plan_path)
    y = runner.infer_batched(VAL_X, max_batch=max_batch)
    return hc.accuracy_report(REF_HSI, y)


def bench(plan_path: Path, batch: int, warmup=50, measure=300) -> dict:
    """阶段3 口径：逐次记录 CUDA event，末尾统一同步。"""
    runner = TrtRunner(plan_path)
    x = hc.dummy_input("hsi", batch).numpy()
    in_dtype = runner.engine.get_tensor_dtype(hc.INPUT_NAME)
    xt = torch.from_numpy(x).to(_TRT_TO_TORCH[in_dtype]).cuda().contiguous()
    runner.context.set_input_shape(hc.INPUT_NAME, x.shape)
    out_shape = tuple(runner.context.get_tensor_shape(hc.OUTPUT_NAME))
    out_dtype = _TRT_TO_TORCH[runner.engine.get_tensor_dtype(hc.OUTPUT_NAME)]
    yt = torch.empty(out_shape, dtype=out_dtype, device="cuda")
    runner.context.set_tensor_address(hc.INPUT_NAME, xt.data_ptr())
    runner.context.set_tensor_address(hc.OUTPUT_NAME, yt.data_ptr())
    stream = runner.stream
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            runner.context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    with torch.cuda.stream(stream):
        for i in range(measure):
            starts[i].record(stream)
            runner.context.execute_async_v3(stream.cuda_stream)
            ends[i].record(stream)
    stream.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return {"batch": batch, "mean_ms": float(times.mean()), "p50_ms": float(np.percentile(times, 50))}


def build_hsi(profile_min, profile_opt, profile_max, out_path: Path,
              force_half: bool = False, timing_cache: "trt.ITimingCache | None" = None) -> tuple[dict, "trt.ITimingCache | None"]:
    onnx_path = ONNX_DIR / "hsi_encoder.onnx"
    builder = trt.Builder(TRT_LOGGER)
    network = bt.build_network_from_onnx(builder, onnx_path)
    config = builder.create_builder_config()

    dims = bt.INPUT_SHAPES["hsi"]
    profile = builder.create_optimization_profile()
    profile.set_shape(hc.INPUT_NAME, min=(profile_min, *dims), opt=(profile_opt, *dims), max=(profile_max, *dims))
    config.add_optimization_profile(profile)

    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.FP16)

    if timing_cache is not None:
        config.set_timing_cache(timing_cache, ignore_mismatch=False)

    if force_half:
        # 只强制"真正的算术层"（白名单 LayerType）且全部输出本来就是 FLOAT/HALF 的层。
        # 跳过 CAST/FILL/SHAPE/CONSTANT/GATHER/SLICE/SHUFFLE/CONCATENATION 等——这些层即使
        # 输出 dtype 报告为 FLOAT，内部也可能已经被 ONNX 解析器锁定了 toType/outputType，
        # 再用 set_output_type 覆盖会触发 TRT 内部一致性断言，导致整个 engine 构建失败
        # （第一版粗暴地对所有层都设 precision，先后撞上 IShapeLayer int64、
        # IConstantLayer int64、IFillLayer toType 冲突、ICastLayer outputType 冲突四类错误）。
        ARITH_TYPES = {
            trt.LayerType.CONVOLUTION, trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE,
            trt.LayerType.ACTIVATION, trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX,
            trt.LayerType.SCALE, trt.LayerType.POOLING, trt.LayerType.EINSUM,
            trt.LayerType.REDUCE, trt.LayerType.UNARY,
        }
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        n_forced, n_skipped = 0, 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            outputs = [layer.get_output(o) for o in range(layer.num_outputs)]
            all_float = len(outputs) > 0 and all(
                t.dtype in (trt.DataType.FLOAT, trt.DataType.HALF) for t in outputs)
            if layer.type not in ARITH_TYPES or not all_float:
                n_skipped += 1
                continue
            try:
                layer.precision = trt.DataType.HALF
                for o, t in enumerate(outputs):
                    layer.set_output_type(o, trt.DataType.HALF)
                n_forced += 1
            except Exception:
                n_skipped += 1

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    dt = time.time() - t0
    used_cache = config.get_timing_cache()
    if serialized is None:
        return {"build_failed": True, "force_half": force_half}, used_cache
    out_path.write_bytes(bytes(serialized))
    info = {
        "profile": [profile_min, profile_opt, profile_max], "force_half": force_half,
        "build_sec": dt, "size_kb": out_path.stat().st_size / 1024, "out_path": str(out_path),
    }
    if force_half:
        info["n_layers_precision_forced"] = n_forced
        info["n_layers_precision_skip"] = n_skipped
    return info, used_cache


def gpu_snapshot() -> dict:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.current.sm",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    parts = out.stdout.strip().split(",")
    return {"util_pct": parts[0].strip(), "mem_used_mib": parts[1].strip(),
            "temp_c": parts[2].strip(), "sm_clock_mhz": parts[3].strip()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["A", "B1", "B2", "B3"])
    ap.add_argument("--merge-into", default=None, help="已有 json，合并输出而不是覆盖")
    args = ap.parse_args()

    report = {}
    if args.merge_into:
        report = json.loads(Path(args.merge_into).read_text())

    if "A" in args.parts:
        print("=== Part A: deploy profile 强制 FP16 诊断构建 ===")
        print("GPU snapshot before build:", gpu_snapshot())
        forced_path = SCRATCH / "hsi_fp16_deploy_forced_half.plan"
        finfo, _ = build_hsi(1, 1024, 4096, forced_path, force_half=True)
        print("GPU snapshot right after build:", gpu_snapshot())
        print("build info:", finfo)
        if not finfo.get("build_failed"):
            facc = eval_acc(forced_path, max_batch=64)
            finfo["accuracy"] = {"cos_sim_min": facc["cos_sim_min"], "cos_sim_mean": facc["cos_sim_mean"],
                                  "max_abs_err": facc["max_abs_err"], "top1_agreement": facc["top1_agreement"]}
            print("accuracy:", finfo["accuracy"])
            forced_bench_64 = bench(forced_path, 64)
            forced_bench_1024 = bench(forced_path, 1024)
            auto_bench_64 = bench(ROOT / "engines/hsi_fp16_deploy.plan", 64)
            auto_bench_1024 = bench(ROOT / "engines/hsi_fp16_deploy.plan", 1024)
            finfo["benchmark"] = {
                "forced_half_batch64": forced_bench_64, "forced_half_batch1024": forced_bench_1024,
                "trt_auto_fp32like_deploy_batch64": auto_bench_64,
                "trt_auto_fp32like_deploy_batch1024": auto_bench_1024,
            }
            print("benchmark:", finfo["benchmark"])
        report["part_a_forced_fp16_deploy"] = finfo
        print("GPU snapshot after:", gpu_snapshot())

    if "B1" in args.parts:
        print("\n=== Part B-1: 已有4次default构建的量化（cos_min/top1/batch1&64延迟） ===")
        default_builds = [
            ("plain_rep1", SCRATCH / "hsi_fp16_default_plain_rep1.plan"),
            ("plain_rep2", SCRATCH / "hsi_fp16_default_plain_rep2.plan"),
            ("detailed_rep1", SCRATCH / "hsi_fp16_default_detailed_rep1.plan"),
            ("detailed_rep2", SCRATCH / "hsi_fp16_default_detailed_rep2.plan"),
        ]
        existing_default_quant = []
        for tag, p in default_builds:
            if not p.exists():
                existing_default_quant.append({"tag": tag, "missing": True})
                continue
            acc = eval_acc(p, max_batch=64)
            b1 = bench(p, 1)
            b64 = bench(p, 64)
            entry = {"tag": tag, "size_kb": p.stat().st_size / 1024, "cos_sim_min": acc["cos_sim_min"],
                      "top1_agreement": acc["top1_agreement"], "max_abs_err": acc["max_abs_err"],
                      "batch1_p50_ms": b1["p50_ms"], "batch64_p50_ms": b64["p50_ms"]}
            existing_default_quant.append(entry)
            print("[b1]", entry)
        report["part_b1_existing_default_builds_quantified"] = existing_default_quant

    if "B2" in args.parts:
        print("\n=== Part B-2: 缓解方案A —— 同一 timing cache 连续构建default 3次 ===")
        cache_runs = []
        shared_cache = None
        for rep in range(1, 4):
            print(f"GPU snapshot before sharedcache rep{rep}:", gpu_snapshot())
            p = SCRATCH / f"hsi_fp16_default_sharedcache_rep{rep}.plan"
            info, shared_cache = build_hsi(1, 8, 64, p, force_half=False, timing_cache=shared_cache)
            if not info.get("build_failed"):
                acc = eval_acc(p, max_batch=64)
                info["cos_sim_min"] = acc["cos_sim_min"]
                info["max_abs_err"] = acc["max_abs_err"]
            info["gpu_before"] = gpu_snapshot()
            cache_runs.append(info)
            print(f"[sharedcache rep{rep}] size_kb={info.get('size_kb')} max_abs={info.get('max_abs_err')}")
        report["part_b2_mitigation_a_shared_timing_cache"] = cache_runs

    if "B3" in args.parts:
        print("\n=== Part B-3: 缓解方案B —— default + 强制HALF约束 构建3次 ===")
        forced_default_runs = []
        for rep in range(1, 4):
            print(f"GPU snapshot before forced_half rep{rep}:", gpu_snapshot())
            p = SCRATCH / f"hsi_fp16_default_forced_half_rep{rep}.plan"
            info, _ = build_hsi(1, 8, 64, p, force_half=True)
            if not info.get("build_failed"):
                acc = eval_acc(p, max_batch=64)
                info["cos_sim_min"] = acc["cos_sim_min"]
                info["max_abs_err"] = acc["max_abs_err"]
                info["top1_agreement"] = acc["top1_agreement"]
            forced_default_runs.append(info)
            print(f"[forced_half rep{rep}] build_failed={info.get('build_failed')} "
                  f"size_kb={info.get('size_kb')} max_abs={info.get('max_abs_err')}")
        report["part_b3_mitigation_b_forced_half_default"] = forced_default_runs

    out_path = SCRATCH / "stage6_subtask1_followup.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
