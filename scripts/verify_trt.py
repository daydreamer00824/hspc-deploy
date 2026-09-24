"""TensorRT engines vs PyTorch FP32 参考输出精度验证（后200条验证集，与校准集不重叠）。

构建顺序即诊断顺序：fp32_notf32 是闸门，未通过则不应继续验证/使用 fp16/int8 结果。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hspc_common as hc
from trt_runner import TrtRunner

REF = hc.ROOT / "results" / "ref"
ENGINE_DIR = hc.ROOT / "engines"
OUT = hc.ROOT / "results" / "stage2_trt_accuracy.json"

GATE_MODE = "fp32_notf32"
THRESH = {
    "fp32_notf32": {"cos_sim_min": 0.9999, "max_abs_err": 1.5e-4},
    "fp32_tf32": None,     # 仅参照，不设通过/失败阈值
    "fp16": {"cos_sim_min": 0.999, "max_abs_err": None},
    "int8_implicit": None,  # 仅报告，不设硬阈值
    "int8_qdq": None,
}

# PC 的 int8_qdq 已放弃交付：TensorRT 10.13 对
# onnx/pc_qdq.onnx 的 INT8 tactic 自动调优存在构建不确定性，4 次独立构建 3 次产出
# 数值垃圾（cos_mean≈-0.09，接近随机正交），且无构建报错/NaN 可供检测；
# OBEY_PRECISION_CONSTRAINTS 复现为稳定出错，定位在 ModelOpt 生成的 QDQ 图对
# 位置编码分支的精度标注有问题。engines/pc_int8_qdq.plan 已删除，不再生成。
# HSI 的 int8_qdq 本身稳定（3 次重建数值一致），但精度明显弱于 int8_implicit，作为
# 对照保留（详细诊断过程未随仓库公开）。
MODEL_MODES = {
    "pc": ["fp32_notf32", "fp32_tf32", "fp16", "int8_implicit"],
    "hsi": ["fp32_notf32", "fp32_tf32", "fp16", "int8_implicit", "int8_qdq"],
}


def verify_one(which: str, mode: str) -> dict | None:
    plan = ENGINE_DIR / f"{which}_{mode}.plan"
    if not plan.exists():
        print(f"[{which}/{mode}] engine not found, skip")
        return None
    ref = np.load(REF / f"{which}_fp32_cpu.npy")
    x = hc.load_samples(which, "val")
    runner = TrtRunner(plan)
    y = runner.infer_batched(x, max_batch=64)
    rep = hc.accuracy_report(ref, y)
    hc.print_report(f"{which}/{mode}", rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=None,
                     help="不传则按 MODEL_MODES 为每个模型用各自的默认模式列表")
    args = ap.parse_args()

    results = {}
    gate_pass = True
    for which in ("pc", "hsi"):
        results[which] = {}
        modes = args.modes or MODEL_MODES[which]
        for mode in modes:
            rep = verify_one(which, mode)
            if rep is None:
                continue
            results[which][mode] = rep
            th = THRESH.get(mode)
            if th:
                ok = True
                if th.get("cos_sim_min") is not None:
                    ok &= rep["cos_sim_min"] >= th["cos_sim_min"]
                if th.get("max_abs_err") is not None:
                    ok &= rep["max_abs_err"] <= th["max_abs_err"]
                results[which][mode]["_pass"] = ok
                if mode == GATE_MODE and not ok:
                    gate_pass = False
                    print(f"  !! GATE FAIL [{which}/{mode}]: cos_min={rep['cos_sim_min']:.6f}, "
                          f"max_abs={rep['max_abs_err']:.3e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"gate_mode": GATE_MODE, "gate_pass": gate_pass,
                                "thresholds": THRESH, "results": results}, indent=2))
    print(f"\nwritten -> {OUT.relative_to(hc.ROOT)}")
    print(f"GATE ({GATE_MODE}): {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
