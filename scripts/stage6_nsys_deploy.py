"""阶段6：deploy_scene.py 的真实时间分解。

环境限制（已核实）：本机 nsys 2024.5.1（WSL2）执行 `nsys status -e` 报告
"Timestamp counter supported: No"——WSL2 这套驱动不支持 GPU 时间戳计数器，CUPTI 无法记录
GPU 侧 kernel/memcpy 的时间线（用 hsi_fp16.plan 做过验证：`nsys stats --report cuda_gpu_kern_sum`
直接 SKIPPED，sqlite 里没有 CUPTI_ACTIVITY_KIND_KERNEL / MEMCPY 表，只有 CUPTI_ACTIVITY_KIND_RUNTIME，
即只能拿到 CPU 端发起 CUDA Runtime API 调用（cudaMemcpyAsync/cudaStreamSynchronize等）的耗时，
不是 GPU 实际执行时间）。因此本次不用 nsys 做 GPU kernel 级拆解，而是：
1. 用 monkeypatch 给 LiteTrtRunner.infer() 的每个子步骤（写 h_in / resize / set_input_shape /
   H2D copy_from / execute_async_v3 / D2H copy_to / synchronize）套 perf_counter，直接测量
   CPU 视角下各步骤的墙钟耗时（这是"H2D→exec→D2H 是否串行"这个问题能拿到的最直接证据：
   如果真的串行，各步骤耗时总和应约等于整次 infer() 的总耗时）
2. 同时在 nsys 下跑一遍，用 CUPTI_ACTIVITY_KIND_RUNTIME 里 cudaStreamSynchronize 的耗时
   交叉验证——如果 perf_counter 测的"synchronize 步骤耗时"和 nsys 记录的 cudaStreamSynchronize
   API 耗时数量级一致，说明每次 infer() 确实在 synchronize() 这一步等待了 GPU，间接印证串行执行

不改 deploy_scene.py 本身（只在本文件里 monkeypatch），不改模型结构/权重/超参数。
运行环境：conda hspc-preprocess + PYTHONPATH=/usr/lib/python3.10/dist-packages
（tensorrt 系统 deb 绑定），加上 PROJ_DATA/PROJ_LIB/GDAL_DATA。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import deploy_scene as ds  # noqa: E402

STEP_TIMES: dict[str, list] = {"hsi": [], "pc": []}


def make_instrumented_infer(orig_infer, which_tag: str):
    def infer(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        assert n <= self.max_batch, f"batch {n} > max_batch {self.max_batch}"
        t0 = time.perf_counter()
        self.h_in[:n] = x
        t1 = time.perf_counter()
        self.d_in.resize((n, *self.input_dims))
        self.d_out.resize((n, self.output_dim))
        t2 = time.perf_counter()
        self.context.set_input_shape(self.input_name, (n, *self.input_dims))
        t3 = time.perf_counter()
        self.d_in.copy_from(self.h_in[:n], self.stream)
        t4 = time.perf_counter()
        self.context.set_tensor_address(self.input_name, self.d_in.ptr)
        self.context.set_tensor_address(self.output_name, self.d_out.ptr)
        ok = self.context.execute_async_v3(self.stream.ptr)
        t5 = time.perf_counter()
        if not ok:
            raise RuntimeError("execute_async_v3 failed")
        self.d_out.copy_to(self.h_out[:n], self.stream)
        t6 = time.perf_counter()
        self.stream.synchronize()
        t7 = time.perf_counter()
        STEP_TIMES[which_tag].append({
            "n": n,
            "write_h_in_s": t1 - t0, "resize_s": t2 - t1, "set_input_shape_s": t3 - t2,
            "h2d_copy_from_enqueue_s": t4 - t3, "execute_async_v3_enqueue_s": t5 - t4,
            "d2h_copy_to_enqueue_s": t6 - t5, "stream_synchronize_s": t7 - t6,
            "total_s": t7 - t0,
        })
        return self.h_out[:n].astype(np.float32, copy=True)
    return infer


def main():
    # 只 patch infer()（infer_all 循环调用它，天然对齐 hsi/pc 两类 runner）。
    # 用 input_dims 区分 hsi(342,3,3) vs pc(15,3)，patch 时按实例覆盖，不动类定义本身。
    orig_infer = ds.LiteTrtRunner.infer
    orig_init = ds.LiteTrtRunner.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        tag = "hsi" if len(self.input_dims) == 3 else "pc"
        self.infer = make_instrumented_infer(orig_infer, tag).__get__(self, ds.LiteTrtRunner)

    ds.LiteTrtRunner.__init__ = patched_init

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    args = ap.parse_args()
    sys.argv = ["deploy_scene.py", "--precision", args.precision]

    ds.main()

    out = ROOT / "results" / "logs" / f"stage6_nsys_step_times_{args.precision}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tag, reps in STEP_TIMES.items():
        if not reps:
            continue
        keys = [k for k in reps[0] if k.endswith("_s")]
        summary[tag] = {
            "n_infer_calls": len(reps),
            "median_per_call_ms": {k: float(np.median([r[k] for r in reps])) * 1000 for k in keys},
            "sum_over_all_calls_ms": {k: float(np.sum([r[k] for r in reps])) * 1000 for k in keys},
        }
    out.write_text(json.dumps({"summary": summary, "raw": STEP_TIMES}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
