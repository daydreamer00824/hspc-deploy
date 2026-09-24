"""阶段7 步骤1：选择性混合精度构建，找 cos_min>=0.999 的最小 FP32 层组集合。

背景：阶段6发现 fp16(auto) 模式下 TRT 自选精度不稳定（同配置多次构建，max_abs 在
1e-4~0.3 间跳，偶有整体退回FP32），且强制纯 HALF 过不了阶段2的 cos_min>=0.999 门槛
（HSI 实测 cos_min≈0.990，200条里4条排序改变，等价于阶段2 int8_implicit 的水平）。
本脚本用 OBEY_PRECISION_CONSTRAINTS 做敏感度搜索，找一个稳定、可控、达标的中间方案。

流程：
1. 全 HALF + 单组 FP32，逐组测精度贡献
2. 对贡献最大的几组做组合，取满足门槛的最小 FP32 集合
3. 抽查 1~2 个候选各构建 2 次，确认 OBEY 约束下构建结果是否稳定（跟auto模式的不稳定
   形成对照）——若不稳定，中止并输出报告
4. 选定方案后，记录相对纯HALF的 trtexec 速度（batch 64/1024）。若比纯HALF慢超过30%，
   先输出"精度-速度"取舍表，人工确认后再选定
5. 选定方案对每个模型连续构建5次，验证精度/文件大小/速度的稳定性

不改模型结构/权重/超参数。运行环境：conda modelopt + PYTHONPATH=/usr/lib/python3.10/dist-packages。
engine 全部先写到 /tmp 的 scratch 目录，选定方案后再由 promote_to_engines() 写入 engines/。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_trt as bt  # noqa: E402
import hspc_common as hc  # noqa: E402
from trt_runner import TrtRunner  # noqa: E402

ROOT = hc.ROOT
SCRATCH = Path(os.environ.get("HSPC_SCRATCH", "/tmp/hspc_stage7_scratch")) / "stage7_engines"
SCRATCH.mkdir(parents=True, exist_ok=True)
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
ACC_THRESHOLD = {"cos_sim_min": 0.999}
SPEED_REGRESSION_LIMIT = 0.30  # 相对纯HALF慢超过30%则中止，输出取舍表后人工选定


def build_variant(which: str, fp32_groups: list[str], tag: str, profile: str = "default") -> tuple[Path, dict]:
    out_path = SCRATCH / f"{which}_{tag}.plan"
    policy_info: dict = {}
    onnx_path = bt.ONNX_DIR / f"{which}_encoder.onnx"
    builder = trt.Builder(bt.TRT_LOGGER)
    network = bt.build_network_from_onnx(builder, onnx_path)
    config = builder.create_builder_config()
    config.add_optimization_profile(bt.make_profile(builder, which, profile))
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.FP16)
    # 注意：fp32_groups=[] 是"全部算术层强制HALF"这一合法配置，不是"不加约束"，
    # 所以这里始终 set OBEY_PRECISION_CONSTRAINTS，不能写成 `if fp32_groups:`
    # （空列表在 Python 里是 falsy，写成那样会让 all_half 误退化成 auto 模式——已实测踩过一次坑）
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    policy_info.update(bt.apply_mixed_policy(network, fp32_groups))
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    dt = time.time() - t0
    if serialized is None:
        return None, {"build_failed": True, "fp32_groups": fp32_groups}
    out_path.write_bytes(bytes(serialized))
    policy_info.update({"build_sec": dt, "size_kb": out_path.stat().st_size / 1024, "fp32_groups": fp32_groups})
    return out_path, policy_info


def eval_accuracy(which: str, plan_path: Path) -> dict:
    ref = np.load(ROOT / f"results/ref/{which}_fp32_cpu.npy")
    x = hc.load_samples(which, "val")
    runner = TrtRunner(plan_path)
    y = runner.infer_batched(x, max_batch=64)
    return hc.accuracy_report(ref, y)


def trtexec_p50(plan_path: Path, batch: int) -> float | None:
    dims = bt.INPUT_SHAPES["hsi" if "hsi" in plan_path.name else "pc"]
    shape = f"input:{batch}x" + "x".join(str(d) for d in dims)
    cmd = [TRTEXEC, f"--loadEngine={plan_path}", f"--shapes={shape}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    for line in r.stdout.splitlines():
        if "GPU Compute Time: min" in line:
            for part in line.split(","):
                if "median" in part:
                    return float(part.split("=")[1].strip().replace("ms", ""))
    return None


def sensitivity_search(which: str) -> dict:
    print(f"\n=== {which}: 逐组敏感度（全HALF + 单组FP32） ===")
    result = {"groups": {}}

    plan, info = build_variant(which, [], "all_half")
    acc = eval_accuracy(which, plan) if plan else None
    result["all_half"] = {"build": info, "accuracy": acc}
    print(f"  all_half: cos_min={acc['cos_sim_min']:.6f} max_abs={acc['max_abs_err']:.3e}")

    for group in bt.LAYER_GROUPS:
        plan, info = build_variant(which, [group], f"g_{group}")
        if plan is None:
            result["groups"][group] = {"build": info}
            print(f"  +{group}: build FAILED")
            continue
        acc = eval_accuracy(which, plan)
        result["groups"][group] = {"build": info, "accuracy": acc}
        print(f"  +{group}: cos_min={acc['cos_sim_min']:.6f} max_abs={acc['max_abs_err']:.3e} "
              f"n_fp32={info['n_fp32']}")
    return result


def greedy_search(which: str, sens: dict) -> dict:
    """按单组贡献（cos_min 提升量）从大到小排序，贪心累加，直到达标或用完所有组。"""
    base_cos = sens["all_half"]["accuracy"]["cos_sim_min"]
    ranked = sorted(
        bt.LAYER_GROUPS,
        key=lambda g: sens["groups"][g].get("accuracy", {}).get("cos_sim_min", base_cos),
        reverse=True,
    )
    trace = []
    chosen: list[str] = []
    for g in ranked:
        if sens["groups"][g].get("accuracy") is None:
            continue
        chosen.append(g)
        plan, info = build_variant(which, chosen, f"combo_{len(chosen)}")
        if plan is None:
            trace.append({"fp32_groups": list(chosen), "build_failed": True})
            chosen.pop()
            continue
        acc = eval_accuracy(which, plan)
        trace.append({"fp32_groups": list(chosen), "build": info, "accuracy": acc})
        print(f"  组合 {chosen}: cos_min={acc['cos_sim_min']:.6f} n_fp32={info['n_fp32']}")
        if acc["cos_sim_min"] >= ACC_THRESHOLD["cos_sim_min"]:
            return {"trace": trace, "selected": list(chosen), "plan_path": str(plan), "accuracy": acc}
    return {"trace": trace, "selected": None}


def stability_check(which: str, fp32_groups: list[str], n: int = 5) -> dict:
    accs, sizes, plans = [], [], []
    for i in range(n):
        plan, info = build_variant(which, fp32_groups, f"stability_{i}")
        acc = eval_accuracy(which, plan)
        accs.append(acc); sizes.append(info["size_kb"]); plans.append(str(plan))
        print(f"  第{i+1}/{n}次: cos_min={acc['cos_sim_min']:.6f} max_abs={acc['max_abs_err']:.3e} size={info['size_kb']:.0f}KB")
    cos_mins = [a["cos_sim_min"] for a in accs]
    return {
        "n": n, "cos_mins": cos_mins, "sizes_kb": sizes,
        "all_pass": all(c >= ACC_THRESHOLD["cos_sim_min"] for c in cos_mins),
        "cos_min_std": float(np.std(cos_mins)), "cos_min_range": [min(cos_mins), max(cos_mins)],
        "accuracies": accs, "plan_paths": plans,
    }


def speed_tradeoff(which: str, mixed_plan: Path) -> dict:
    all_half_plan, _ = build_variant(which, [], "speed_all_half")
    out = {}
    for batch in [64, 1024]:
        half_p50 = trtexec_p50(all_half_plan, batch)
        mixed_p50 = trtexec_p50(mixed_plan, batch)
        out[f"batch{batch}"] = {
            "all_half_p50_ms": half_p50, "mixed_p50_ms": mixed_p50,
            "slowdown_pct": (mixed_p50 / half_p50 - 1) * 100 if half_p50 else None,
        }
    return out


def main():
    report = {}
    for which in ["hsi", "pc"]:
        print(f"\n{'='*20} {which} {'='*20}")
        sens = sensitivity_search(which)
        search = greedy_search(which, sens)
        entry = {"sensitivity": sens, "search": search}

        if search["selected"] is None:
            entry["status"] = "no_solution_found"
            print(f"  !! {which}: 穷举所有层组仍未达到 cos_min>=0.999，中止")
            report[which] = entry
            continue

        fp32_groups = search["selected"]
        plan_path = Path(search["plan_path"])

        print(f"\n--- {which}: 抽查稳定性（选定方案独立重建2次） ---")
        recheck = [eval_accuracy(which, build_variant(which, fp32_groups, f"recheck_{i}")[0])
                   for i in range(2)]
        entry["obey_stability_precheck"] = {
            "cos_mins": [r["cos_sim_min"] for r in recheck],
            "consistent": max(r["cos_sim_min"] for r in recheck) - min(r["cos_sim_min"] for r in recheck) < 1e-4,
        }
        print(f"  两次重建 cos_min: {[round(r['cos_sim_min'], 6) for r in recheck]}")
        if not entry["obey_stability_precheck"]["consistent"]:
            entry["status"] = "unstable_precheck_failed"
            print(f"  !! {which}: OBEY 约束下选定方案仍不稳定，中止")
            report[which] = entry
            continue

        print(f"\n--- {which}: 精度-速度取舍（selected={fp32_groups}） ---")
        entry["speed_tradeoff"] = speed_tradeoff(which, plan_path)
        max_slowdown = max(
            v["slowdown_pct"] for v in entry["speed_tradeoff"].values() if v["slowdown_pct"] is not None
        )
        entry["max_slowdown_vs_all_half_pct"] = max_slowdown

        if max_slowdown > SPEED_REGRESSION_LIMIT * 100:
            entry["status"] = "speed_regression_needs_decision"
            print(f"  !! {which}: 比纯HALF慢 {max_slowdown:.1f}% (>30%)，中止并输出取舍表")
            report[which] = entry
            continue

        print(f"\n--- {which}: 稳定性验证（连续构建5次） ---")
        entry["stability"] = stability_check(which, fp32_groups, n=5)
        entry["status"] = "ready_to_promote" if entry["stability"]["all_pass"] else "stability_check_failed"
        report[which] = entry

    out = ROOT / "results/stage7_mixed_precision.json"
    # accuracy dict 里的 numpy 标量转 float，避免 json 报错
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    out.write_text(json.dumps(clean(report), ensure_ascii=False, indent=2))
    print(f"\nwritten {out}")

    for which, entry in report.items():
        print(f"\n{which}: status={entry['status']}")


if __name__ == "__main__":
    main()
