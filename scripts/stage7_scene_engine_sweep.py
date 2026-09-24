"""阶段7：scene engine（整景推理用的大batch engine）档位扫描。

决定：大batch engine 提升为正式交付物，先扫 max∈{1024,2048,4096}，
opt=max（整景推理每次调用都尽量填满到 max，opt 设成实际最常见的调用规模），同一进程内测
HSI 整景推理耗时和 device_memory_size，选速度接近最优、显存最小的档位。PC 同步扫描。

精度策略沿用步骤1选定的 mixed：HSI 的 stem 组、PC 的 softmax 组保留 FLOAT，其余 HALF。
运行环境：conda modelopt + PYTHONPATH=/usr/lib/python3.10/dist-packages。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_trt as bt  # noqa: E402
from deploy_scene import LiteTrtRunner  # noqa: E402
from stage7_batch_pinned import timed_infer_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("HSPC_SCRATCH", "/tmp/hspc_stage7_scratch")) / "stage7_engines"
FP32_GROUPS = {"hsi": ["stem"], "pc": ["softmax"]}
INPUT_DIMS = {"hsi": (342, 3, 3), "pc": (15, 3)}
TIERS = [1024, 2048, 4096]
REPEAT = 10
NEAR_OPTIMAL_TOL = 0.05  # 与最快档位差距在5%以内视为"接近最优"


def build_tier(which: str, max_batch: int) -> Path:
    name = f"scene{max_batch}"
    bt.PROFILES[name] = {"min": 1, "opt": max_batch, "max": max_batch}
    out = SCRATCH / f"{which}_fp16_{name}.plan"
    info: dict = {}
    bt.build_engine(which, "fp16", profile_name=name, precision_policy="mixed",
                    fp32_groups=FP32_GROUPS[which], out_path=out, policy_info=info)
    return out


def measure(which: str, plan: Path, max_batch: int, x: np.ndarray) -> dict:
    runner = LiteTrtRunner(plan, input_dims=INPUT_DIMS[which], max_batch=max_batch)
    timed_infer_all(runner, x)  # warmup
    timed_infer_all(runner, x)
    walls, gpus = [], []
    for _ in range(REPEAT):
        _, wall, gpu_ms, n_calls = timed_infer_all(runner, x)
        walls.append(wall); gpus.append(gpu_ms)
    return {
        "max_batch": max_batch, "n_calls": n_calls,
        "device_memory_size_bytes": int(runner.device_memory_size),
        "wall_ms_median": float(np.median(walls) * 1000), "wall_ms_min": float(np.min(walls) * 1000),
        "gpu_event_ms_median": float(np.median(gpus)),
        "plan": str(plan), "plan_size_kb": plan.stat().st_size / 1024,
    }


def main():
    data = np.load(ROOT / "results/e2e_scene_preprocessed.npz")
    vr, vc = np.where(data["valid_mask"])
    hsi_x = data["hsi_grid_patches"][vr * int(data["hsi_cols"]) + vc]
    xs = {"hsi": hsi_x, "pc": data["point_offsets"]}

    report = {}
    for which in ["hsi", "pc"]:
        plans = {t: build_tier(which, t) for t in TIERS}
        fwd = {t: measure(which, plans[t], t, xs[which]) for t in TIERS}
        rev = {t: measure(which, plans[t], t, xs[which]) for t in reversed(TIERS)}
        rows = []
        for t in TIERS:
            wall = float(np.median([fwd[t]["wall_ms_median"], rev[t]["wall_ms_median"]]))
            rows.append({**fwd[t], "wall_ms_median_fwd": fwd[t]["wall_ms_median"],
                         "wall_ms_median_rev": rev[t]["wall_ms_median"], "wall_ms_combined": wall})
            print(f"  {which} max={t}: fwd={fwd[t]['wall_ms_median']:.2f}ms rev={rev[t]['wall_ms_median']:.2f}ms "
                  f"gpu={fwd[t]['gpu_event_ms_median']:.2f}ms mem={fwd[t]['device_memory_size_bytes']/1e6:.1f}MB "
                  f"calls={fwd[t]['n_calls']}")
        best = min(r["wall_ms_combined"] for r in rows)
        near = [r for r in rows if r["wall_ms_combined"] <= best * (1 + NEAR_OPTIMAL_TOL)]
        chosen = min(near, key=lambda r: r["device_memory_size_bytes"])
        report[which] = {"n_samples": len(xs[which]), "tiers": rows,
                         "rule": f"wall 与最快档位相差 ≤{NEAR_OPTIMAL_TOL:.0%} 的档位中取 device_memory_size 最小者",
                         "chosen_max_batch": chosen["max_batch"]}
        print(f"  => {which} 选定 max={chosen['max_batch']}")

    (ROOT / "results/stage7_scene_engine_sweep.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("written results/stage7_scene_engine_sweep.json")


if __name__ == "__main__":
    main()
