"""阶段6 前置诊断：2x2x2 对照实验，隔离"DETAILED verbosity" 和 "profile 大小" 两个变量。

背景：用当前代码 DETAILED 重建 hsi/default(1/8/64) 后精度接近 FP32，但这个结果和
真实的 engines/hsi_fp16.plan（用 14:11:54 修改前的旧版 build_trt.py、非 DETAILED 构建）对比，
混入了"代码版本"和"构建随机性"两个变量，不能把差异全部归因于 DETAILED。

本脚本用当前代码、全部现场重建，做 {default(1/8/64), deploy(1/1024/4096)} ×
{DETAILED, 非DETAILED} 2x2，每格构建2次（共8个engine），只用 fp16/hsi，只报告大小和精度
（cos_min/max_abs_err，对照后200条真实验证集的 PyTorch FP32 基准）。不做判断性改写，
输出留给使用者读数字。

不改模型结构/权重/超参数；engine 全部写到 /tmp/hspc_stage6_scratch，不进 engines/。
运行环境：conda modelopt，PYTHONPATH=/usr/lib/python3.10/dist-packages。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_trt as bt  # noqa: E402
import hspc_common as hc  # noqa: E402
from trt_runner import TrtRunner  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ROOT = hc.ROOT
ONNX_DIR = ROOT / "onnx"
SCRATCH = Path("/tmp/hspc_stage6_scratch/engines")
SCRATCH.mkdir(parents=True, exist_ok=True)


def build_hsi_fp16(profile_min, profile_opt, profile_max, out_path: Path, detailed: bool) -> dict:
    """逐行对照 build_trt.py 当前版本 build_engine() 的 fp16 分支：
    create_builder_config -> add_optimization_profile -> clear TF32 -> set FP16 ->
    [可选 profiling_verbosity=DETAILED] -> build_serialized_network。
    与 build_trt.py 的唯一差别就是 DETAILED 这一行，其余逐字相同。
    """
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
    if detailed:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    dt = time.time() - t0
    if serialized is None:
        return {"build_failed": True}
    out_path.write_bytes(bytes(serialized))

    ref = np.load(ROOT / "results/ref/hsi_fp32_cpu.npy")
    x = hc.load_samples("hsi", "val")
    runner = TrtRunner(out_path)
    y = runner.infer_batched(x, max_batch=min(64, profile_max))
    acc = hc.accuracy_report(ref, y)

    return {
        "profile": [profile_min, profile_opt, profile_max], "detailed": detailed,
        "build_sec": dt, "size_kb": out_path.stat().st_size / 1024,
        "cos_sim_min": acc["cos_sim_min"], "cos_sim_mean": acc["cos_sim_mean"],
        "max_abs_err": acc["max_abs_err"], "top1_agreement": acc["top1_agreement"],
    }


def main():
    combos = [
        ("default", 1, 8, 64, False),
        ("default", 1, 8, 64, True),
        ("deploy", 1, 1024, 4096, False),
        ("deploy", 1, 1024, 4096, True),
    ]
    results = []
    for tag, mn, opt, mx, detailed in combos:
        for rep in (1, 2):
            out_path = SCRATCH / f"hsi_fp16_{tag}_{'detailed' if detailed else 'plain'}_rep{rep}.plan"
            info = build_hsi_fp16(mn, opt, mx, out_path, detailed)
            info.update({"tag": tag, "rep": rep})
            results.append(info)
            print(f"[{tag} detailed={detailed} rep={rep}] size_kb={info.get('size_kb'):.1f} "
                  f"cos_min={info.get('cos_sim_min')} max_abs={info.get('max_abs_err')}")

    # 参照组：真实存在的 engines/*.plan（不重建），仅供并排对比，不参与2x2判断
    reference = {}
    for name, plan_rel in [
        ("real_hsi_fp16_default_old_code", "engines/hsi_fp16.plan"),
        ("real_hsi_fp16_deploy_new_code", "engines/hsi_fp16_deploy.plan"),
        ("real_hsi_fp32_notf32_gate", "engines/hsi_fp32_notf32.plan"),
    ]:
        plan = ROOT / plan_rel
        if plan.exists():
            ref = np.load(ROOT / "results/ref/hsi_fp32_cpu.npy")
            x = hc.load_samples("hsi", "val")
            runner = TrtRunner(plan)
            y = runner.infer_batched(x, max_batch=64)
            acc = hc.accuracy_report(ref, y)
            reference[name] = {"size_kb": plan.stat().st_size / 1024, "cos_sim_min": acc["cos_sim_min"],
                                "max_abs_err": acc["max_abs_err"]}
            print(f"[reference {name}] size_kb={reference[name]['size_kb']:.1f} "
                  f"cos_min={reference[name]['cos_sim_min']} max_abs={reference[name]['max_abs_err']}")

    out = {"combos_2x2x2": results, "reference_existing_plans": reference}
    out_path = SCRATCH / "confound_check.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
