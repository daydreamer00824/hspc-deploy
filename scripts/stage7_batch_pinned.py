"""阶段7 步骤2：{max_batch64, 大batch} x {pageable, pinned} 受控对比 + CUDA Graph 决策。

设计要点：
- 大profile engine 用步骤1的 mixed 策略构建（避免阶段5/6发现的 auto 模式大profile精度坍缩问题，
  已在 stage7_mixed_precision.json 的 deploy_profile_accuracy_check 里验证过）
- 正确性对照基准用新的 engines/{hsi,pc}_fp16.plan（batch64, pageable），不用 legacy 文件
- 同一进程内正向(A→D)+反向(D→A)两轮，只改一个变量做2x2（控制变量）
- CUDA event 记录GPU时间、wall-clock记录CPU端"enqueue+这次调用总耗时"，
  CPU开销 = wall - gpu_event；只有CPU开销超过HSI阶段总耗时20%才建议做CUDA Graph
- 记录各batch档位(64 vs 大batch)的 device_memory_size

engine 来源：
- 64: engines/hsi_fp16.plan / engines/pc_fp16.plan（步骤1产出，min1/opt8/max64, mixed精度）
- 大batch: /tmp scratch 的 hsi_deploy_mixed.plan / pc_deploy_mixed.plan
  （步骤1产出，min1/opt1024/max4096, 同样的 mixed 精度策略，已验证 cos_min 0.9998/0.999995）

运行环境：conda modelopt + PYTHONPATH=/usr/lib/python3.10/dist-packages（用 hsi_grid_patches
做速度和显存实验，不需要 gdal/pyproj/laspy，所以不用切到 hspc-preprocess 环境）。
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_scene import LiteTrtRunner, _CUDART  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("HSPC_SCRATCH", "/tmp/hspc_stage7_scratch")) / "stage7_engines"
ENGINES = {
    "hsi": {"batch64": ROOT / "engines/hsi_fp16.plan", "large": SCRATCH / "hsi_deploy_mixed.plan",
            "input_dims": (342, 3, 3), "max_large": 4096},
    "pc": {"batch64": ROOT / "engines/pc_fp16.plan", "large": SCRATCH / "pc_deploy_mixed.plan",
           "input_dims": (15, 3), "max_large": 4096},
}
REPEAT = 10
CUDA_GRAPH_CPU_OVERHEAD_THRESHOLD = 0.20  # CPU开销超过HSI阶段总耗时的这个比例才建议做CUDA Graph

# ---------------- CUDA event ctypes 封装（复用 deploy_scene._CUDART 同一个 libcudart handle） ----------------
_CUDART.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_CUDART.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_CUDART.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
_CUDART.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
_CUDART.cudaEventDestroy.argtypes = [ctypes.c_void_p]


class CudaEvent:
    def __init__(self):
        e = ctypes.c_void_p()
        _CUDART.cudaEventCreate(ctypes.byref(e))
        self.e = e

    def record(self, stream_ptr: int):
        _CUDART.cudaEventRecord(self.e, ctypes.c_void_p(stream_ptr))

    def synchronize(self):
        _CUDART.cudaEventSynchronize(self.e)

    @staticmethod
    def elapsed_ms(start: "CudaEvent", end: "CudaEvent") -> float:
        ms = ctypes.c_float()
        _CUDART.cudaEventElapsedTime(ctypes.byref(ms), start.e, end.e)
        return float(ms.value)

    def destroy(self):
        _CUDART.cudaEventDestroy(self.e)


def timed_infer_all(runner: LiteTrtRunner, x: np.ndarray):
    """跑一次 infer_all，同时用 CUDA event 测每次 execute_async_v3 的 GPU 时间累计。
    返回 (输出, wall_clock_sec, gpu_event_sum_ms, n_calls)。
    """
    max_batch = runner.max_batch
    n_calls = (len(x) + max_batch - 1) // max_batch
    starts = [CudaEvent() for _ in range(n_calls)]
    ends = [CudaEvent() for _ in range(n_calls)]
    out = np.empty((len(x), runner.output_dim), dtype=np.float32)

    t0 = time.perf_counter()
    for i, off in enumerate(range(0, len(x), max_batch)):
        batch = x[off:off + max_batch]
        n = len(batch)
        runner.h_in[:n] = batch
        runner.d_in.resize((n, *runner.input_dims))
        runner.d_out.resize((n, runner.output_dim))
        runner.context.set_input_shape(runner.input_name, (n, *runner.input_dims))
        runner.d_in.copy_from(runner.h_in[:n], runner.stream)
        runner.context.set_tensor_address(runner.input_name, runner.d_in.ptr)
        runner.context.set_tensor_address(runner.output_name, runner.d_out.ptr)
        starts[i].record(runner.stream.ptr)
        runner.context.execute_async_v3(runner.stream.ptr)
        ends[i].record(runner.stream.ptr)
        runner.d_out.copy_to(runner.h_out[:n], runner.stream)
        runner.stream.synchronize()
        out[off:off + n] = runner.h_out[:n]
    wall = time.perf_counter() - t0

    gpu_ms_sum = sum(CudaEvent.elapsed_ms(s, e) for s, e in zip(starts, ends))
    for s, e in zip(starts, ends):
        s.destroy(); e.destroy()
    return out, wall, gpu_ms_sum, n_calls


def run_config(which: str, tier: str, pinned: bool, x: np.ndarray) -> dict:
    info = ENGINES[which]
    plan = info["batch64"] if tier == "batch64" else info["large"]
    max_batch = 64 if tier == "batch64" else info["max_large"]
    runner = LiteTrtRunner(plan, input_dims=info["input_dims"], max_batch=max_batch, pinned=pinned)

    _ = timed_infer_all(runner, x)  # warmup
    outs, walls, gpu_sums, n_calls = [], [], [], None
    for _ in range(REPEAT):
        out, wall, gpu_ms, n_calls = timed_infer_all(runner, x)
        outs.append(out); walls.append(wall); gpu_sums.append(gpu_ms)

    result = {
        "tier": tier, "pinned": pinned, "max_batch": max_batch, "n_calls": n_calls,
        "device_memory_size_bytes": runner.device_memory_size,
        "wall_sec_median": float(np.median(walls)), "wall_sec_all": walls,
        "gpu_event_ms_sum_median": float(np.median(gpu_sums)), "gpu_event_ms_sum_all": gpu_sums,
        "cpu_overhead_sec_median": float(np.median(walls) - np.median(gpu_sums) / 1000),
        "output_identical_across_reps": bool(all(np.array_equal(outs[0], o) for o in outs[1:])),
        "_output_sample": outs[0],
    }
    if runner.pinned:
        runner.close()
    return result


def main():
    data = np.load(ROOT / "results/e2e_scene_preprocessed.npz")
    hsi_valid_mask = data["valid_mask"]
    hsi_all_patches = data["hsi_grid_patches"]
    # 只用有效像元的patch（阶段5的做法），跟部署链路口径一致
    vr, vc = np.where(hsi_valid_mask)
    rows_full, cols_full = int(data["hsi_rows"]), int(data["hsi_cols"])
    valid_idx = (vr * cols_full + vc)
    hsi_x = hsi_all_patches[valid_idx]
    pc_x = data["point_offsets"]

    report = {}
    for which, x in [("hsi", hsi_x), ("pc", pc_x)]:
        print(f"\n{'='*20} {which} (n={len(x)}) {'='*20}")
        configs = [("batch64", False), ("batch64", True), ("large", False), ("large", True)]
        forward, reverse = {}, {}
        for tier, pinned in configs:
            tag = f"{tier}_{'pinned' if pinned else 'pageable'}"
            r = run_config(which, tier, pinned, x)
            forward[tag] = r
            print(f"  [fwd] {tag}: wall_median={r['wall_sec_median']*1000:.2f}ms "
                  f"gpu_event_sum={r['gpu_event_ms_sum_median']:.2f}ms "
                  f"cpu_overhead={r['cpu_overhead_sec_median']*1000:.2f}ms "
                  f"mem={r['device_memory_size_bytes']/1e6:.2f}MB")
        for tier, pinned in reversed(configs):
            tag = f"{tier}_{'pinned' if pinned else 'pageable'}"
            r = run_config(which, tier, pinned, x)
            reverse[tag] = r
            print(f"  [rev] {tag}: wall_median={r['wall_sec_median']*1000:.2f}ms "
                  f"gpu_event_sum={r['gpu_event_ms_sum_median']:.2f}ms")

        # 正确性：batch64_pageable(fwd) 作为基准
        baseline = forward["batch64_pageable"]["_output_sample"]
        correctness = {}
        for tag, r in forward.items():
            same_engine = tag.startswith("batch64")
            out = r["_output_sample"]
            if same_engine:
                correctness[tag] = {"bit_exact_vs_baseline": bool(np.array_equal(out, baseline))}
            else:
                # 不同 profile 的 engine 大概率 tactic 不同、不会逐位一致，退而比较 cos_sim
                # （两个 engine 对同一批 patch 按相同顺序输出，逐行对应）
                a = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
                b = baseline / (np.linalg.norm(baseline, axis=1, keepdims=True) + 1e-12)
                cos_sim = np.sum(a * b, axis=1)
                correctness[tag] = {
                    "bit_exact_vs_baseline": bool(np.array_equal(out, baseline)),
                    "cos_sim_mean": float(cos_sim.mean()), "cos_sim_min": float(cos_sim.min()),
                }

        # 正反两轮方向一致性检查
        direction_consistent = all(
            abs(forward[tag]["wall_sec_median"] - reverse[tag]["wall_sec_median"]) / forward[tag]["wall_sec_median"] < 0.5
            for tag in forward
        )

        # CUDA Graph 决策：用当前部署链路里HSI阶段的实际耗时做分母（batch64_pageable这组的wall时间，
        # 即阶段5 deploy_scene.py 实际在用的配置）
        hsi_stage_wall = forward["batch64_pageable"]["wall_sec_median"]
        cpu_overhead = forward["batch64_pageable"]["cpu_overhead_sec_median"]
        cpu_overhead_ratio = cpu_overhead / hsi_stage_wall if hsi_stage_wall else None

        for cfg in forward.values():
            cfg.pop("_output_sample")
        for cfg in reverse.values():
            cfg.pop("_output_sample")

        report[which] = {
            "n_samples": len(x), "forward": forward, "reverse": reverse,
            "correctness_vs_batch64_pageable_baseline": correctness,
            "direction_consistent_fwd_vs_rev": direction_consistent,
            "cuda_graph_decision": {
                "cpu_overhead_sec": cpu_overhead, "stage_wall_sec": hsi_stage_wall,
                "cpu_overhead_ratio": cpu_overhead_ratio,
                "threshold": CUDA_GRAPH_CPU_OVERHEAD_THRESHOLD,
                "recommend_cuda_graph": (cpu_overhead_ratio or 0) > CUDA_GRAPH_CPU_OVERHEAD_THRESHOLD,
            },
        }

    out_path = ROOT / "results/stage7_batch_pinned.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
