"""阶段6：核查 *_deploy.plan（大 profile）的 FP16 flag 是否生效。

不改模型结构/权重/超参数。所有重建的 engine 只写到 scratch 目录，不进 engines/。
运行环境：conda modelopt（torch + tensorrt），PYTHONPATH=/usr/lib/python3.10/dist-packages。

判定三条证据线：
1. builder flag 读回（config.get_flag(FP16)）——纯代码层面，排查 (a)
2. EngineInspector.get_engine_information(JSON) 统计各层输出精度（Float/Half/Int8）——排查 (a)/(b)
3. hc.accuracy_report 用真实验证集（后200条）对比 PyTorch FP32 基准——精度层面的直接证据

如果发现 default profile 的 fp16 engine 里已经有大量层是 Float 输出（不是 deploy 独有），
则说明"精度接近FP32"未必是 deploy profile 特有现象，需要单独报告。

追加：对 hsi 做 profile 二分（1/64/4096、1/1024/1024、1/256/1024），定位是 opt 还是 max 触发的。
追加：如果 deploy engine 精度确认劣化，尝试 OBEY_PRECISION_CONSTRAINTS + 逐层 precision=HALF 强制 FP16，
      看 TRT 是否仍然选择 Float 实现（说明有硬约束以外的原因，比如没有可用的 FP16 tactic）。
"""
from __future__ import annotations

import argparse
import copy
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

SCRATCH = Path(sys.argv[sys.argv.index("--scratch") + 1]) if "--scratch" in sys.argv else None


def build_fp16(which: str, profile_min: int, profile_opt: int, profile_max: int,
                out_path: Path, force_precision_constraints: bool = False,
                detailed: bool = False) -> dict:
    """照抄 build_trt.build_engine 的 fp16 分支（不加 DETAILED，见下方"DETAILED confound"发现）。
    可选加逐层 precision 约束用于诊断分支。返回构建元信息，不做精度/性能评估。
    """
    onnx_path = ONNX_DIR / f"{which}_encoder.onnx"
    builder = trt.Builder(TRT_LOGGER)
    network = bt.build_network_from_onnx(builder, onnx_path)
    config = builder.create_builder_config()

    dims = bt.INPUT_SHAPES[which]
    profile = builder.create_optimization_profile()
    profile.set_shape(hc.INPUT_NAME, min=(profile_min, *dims), opt=(profile_opt, *dims), max=(profile_max, *dims))
    config.add_optimization_profile(profile)

    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.FP16)
    if detailed:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    if force_precision_constraints:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            try:
                layer.precision = trt.DataType.HALF
                for o in range(layer.num_outputs):
                    if layer.get_output(o).dtype in (trt.DataType.FLOAT, trt.DataType.HALF):
                        layer.set_output_type(o, trt.DataType.HALF)
            except Exception:
                pass  # 部分层（如 shape 层）不支持设精度，跳过

    flag_readback = {
        "FP16": bool(config.get_flag(trt.BuilderFlag.FP16)),
        "TF32": bool(config.get_flag(trt.BuilderFlag.TF32)),
        "OBEY_PRECISION_CONSTRAINTS": bool(config.get_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)),
    }

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    dt = time.time() - t0
    if serialized is None:
        return {"which": which, "profile": [profile_min, profile_opt, profile_max],
                "flag_readback": flag_readback, "build_failed": True}
    out_path.write_bytes(bytes(serialized))
    return {
        "which": which, "profile": [profile_min, profile_opt, profile_max],
        "out_path": str(out_path), "flag_readback": flag_readback,
        "build_sec": dt, "size_kb": out_path.stat().st_size / 1024,
        "force_precision_constraints": force_precision_constraints,
    }


def inspect_precision(plan_path: Path, which: str, profile_batch: int) -> dict:
    """加载 engine，用 EngineInspector 在给定 batch 下统计各层输出精度分布。

    注意：若 engine 构建时 profiling_verbosity 不是 DETAILED，get_engine_information 只返回
    层名字符串列表（无 Outputs/TacticName 字段），此时本函数只报告层数和名字样本。
    """
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    context = engine.create_execution_context()
    dims = bt.INPUT_SHAPES[which]
    context.set_input_shape(hc.INPUT_NAME, (profile_batch, *dims))
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context
    info_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    info = json.loads(info_str)
    layers = info.get("Layers", [])
    if layers and isinstance(layers[0], str):
        return {
            "plan": plan_path.name, "n_layers": len(layers),
            "detail_level": "names_only (engine 未用 DETAILED verbosity 构建，无法获取逐层 dtype)",
            "layer_names_sample": layers[:10],
        }
    dtype_counts: dict[str, int] = {}
    gemm_like = []
    for layer in layers:
        outputs = layer.get("Outputs", [])
        for o in outputs:
            dt_ = o.get("Format/Datatype", o.get("Desc", "unknown"))
            dtype_counts[dt_] = dtype_counts.get(dt_, 0) + 1
        name = layer.get("Name", "")
        if any(k in name for k in ["MatMul", "Gemm", "self_attn", "Conv", "mha", "gemm", "stem"]):
            gemm_like.append({"name": name, "tactic": layer.get("TacticName", layer.get("TacticValue", "")),
                               "outputs": [o.get("Format/Datatype") for o in outputs]})
    return {
        "plan": plan_path.name, "n_layers": len(layers), "detail_level": "detailed",
        "dtype_output_counts": dtype_counts, "gemm_like_sample": gemm_like[:15],
    }


def eval_accuracy(plan_path: Path, which: str) -> dict:
    ref = np.load(ROOT / "results/ref" / f"{which}_fp32_cpu.npy")
    x = hc.load_samples(which, "val")
    runner = TrtRunner(plan_path)
    y = runner.infer_batched(x, max_batch=64)
    return hc.accuracy_report(ref, y)


def bench_batches(plan_path: Path, which: str, batches: list[int], warmup=50, measure=300) -> list[dict]:
    """复用阶段3的计时口径（逐次记录 CUDA event，末尾统一同步），只用于本次条件分支的复测。"""
    import torch
    from trt_runner import _TRT_TO_TORCH

    out = []
    runner = TrtRunner(plan_path)
    for batch in batches:
        x = hc.dummy_input(which, batch).numpy()
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
        out.append({
            "batch": batch, "mean_ms": float(times.mean()), "p50_ms": float(np.percentile(times, 50)),
            "us_per_sample_p50": float(np.percentile(times, 50) * 1000 / batch),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    report: dict = {"builds": [], "inspections": [], "accuracy": {}, "bisection": [], "forced_fp16": None,
                     "benchmark_conditional": None, "detailed_verbosity_confound": None}

    # --- 0. DETAILED verbosity confound 探测：用 DETAILED 重建 hsi/default(1/8/64)，
    #     对照真实的 engines/hsi_fp16.plan（未用 DETAILED 构建）。
    #     如果两者精度特征不同，说明 profiling_verbosity=DETAILED 本身会改变 kernel/精度选择，
    #     不能用它来无损地检查真实 deploy engine——必须先确认这一点，再决定后续证据链怎么用。
    confound_path = scratch / "hsi_fp16_default_DETAILED_confound_probe.plan"
    confound_info = build_fp16("hsi", 1, 8, 64, confound_path, detailed=True)
    confound_acc = eval_accuracy(confound_path, "hsi") if not confound_info.get("build_failed") else None
    real_default_acc = eval_accuracy(ROOT / "engines/hsi_fp16.plan", "hsi")
    confound_detected = bool(confound_acc and confound_acc["max_abs_err"] < real_default_acc["max_abs_err"] / 5)
    report["detailed_verbosity_confound"] = {
        "purpose": "验证 profiling_verbosity=DETAILED 是否会改变 TRT 的 kernel/精度选择（confound），"
                   "如果会，就不能用 DETAILED 重建 + EngineInspector 来无损诊断真实的 *_deploy.plan",
        "detailed_rebuild_of_default_profile": {
            "size_kb": confound_info.get("size_kb"), "accuracy": confound_acc,
        },
        "real_engines_hsi_fp16_plan_no_detailed": {
            "size_kb": (ROOT / "engines/hsi_fp16.plan").stat().st_size / 1024, "accuracy": real_default_acc,
        },
        "confound_confirmed": confound_detected,
    }
    print(f"[confound_probe] DETAILED-rebuilt default profile: size={confound_info.get('size_kb')}KB "
          f"max_abs={confound_acc['max_abs_err'] if confound_acc else None}; "
          f"real default plan: size={report['detailed_verbosity_confound']['real_engines_hsi_fp16_plan_no_detailed']['size_kb']}KB "
          f"max_abs={real_default_acc['max_abs_err']}; confound_confirmed={confound_detected}")
    if confound_detected:
        insp_confound = inspect_precision(confound_path, "hsi", 64)
        report["detailed_verbosity_confound"]["inspection_of_confounded_build"] = insp_confound
        print(f"[confound_probe] DETAILED build 的 stem/gemm 层样本: "
              f"{[g['outputs'] for g in insp_confound.get('gemm_like_sample', [])[:3]]}")

    # --- 1. 构建矩阵：default vs deploy（hsi + pc），不加 DETAILED（避免上面的 confound） ---
    matrix = [
        ("hsi", 1, 8, 64, "default"),
        ("hsi", 1, 1024, 4096, "deploy"),
        ("pc", 1, 8, 64, "default"),
        ("pc", 1, 1024, 4096, "deploy"),
    ]
    built = {}
    for which, mn, opt, mx, tag in matrix:
        out_path = scratch / f"{which}_fp16_{tag}_rebuild.plan"
        info = build_fp16(which, mn, opt, mx, out_path)
        info["tag"] = tag
        report["builds"].append(info)
        built[(which, tag)] = out_path
        print(f"[build] {which}/{tag} profile={mn}/{opt}/{mx} flag_readback={info['flag_readback']} "
              f"size_kb={info.get('size_kb')}")

    # --- 2. 对照构建：直接调用 build_trt.build_engine（证明手写重建和原脚本行为一致） ---
    orig_engine_dir = bt.ENGINE_DIR
    try:
        bt.ENGINE_DIR = scratch
        control_path = bt.build_engine("hsi", "fp16", profile_name="deploy")
    finally:
        bt.ENGINE_DIR = orig_engine_dir
    report["control_build_via_build_trt"] = {
        "path": str(control_path), "size_kb": control_path.stat().st_size / 1024,
        "matches_manual_rebuild_size_kb": abs(control_path.stat().st_size -
                                               built[("hsi", "deploy")].stat().st_size) < 1024,
        "matches_existing_engines_hsi_fp16_deploy_plan_size_kb":
            abs(control_path.stat().st_size - (ROOT / "engines/hsi_fp16_deploy.plan").stat().st_size) < 1024,
    }

    # --- 3. EngineInspector：现有 engines/*_fp16.plan 和 *_fp16_deploy.plan（原样，无需重建，
    #     但这些不是 DETAILED 构建的，只能拿到层名，不能拿到逐层 dtype——见上面的 confound 结论） ---
    for which in ["hsi", "pc"]:
        for tag, batch in [("default", 64), ("deploy", 1024)]:
            plan = ROOT / "engines" / f"{which}_fp16{'_deploy' if tag=='deploy' else ''}.plan"
            if plan.exists():
                insp = inspect_precision(plan, which, batch)
                insp["tag"] = tag
                report["inspections"].append(insp)
                print(f"[inspect] {which}/{tag}: detail_level={insp['detail_level']} n_layers={insp['n_layers']}")

    # --- 4. 精度：现有 engines/*_fp16.plan vs *_fp16_deploy.plan，用真实验证集（主证据） ---
    for which in ["hsi", "pc"]:
        report["accuracy"][which] = {}
        for tag in ["default", "deploy"]:
            plan = ROOT / "engines" / f"{which}_fp16{'_deploy' if tag=='deploy' else ''}.plan"
            if plan.exists():
                acc = eval_accuracy(plan, which)
                report["accuracy"][which][tag] = acc
                print(f"[accuracy] {which}/{tag}: cos_min={acc['cos_sim_min']:.6f} max_abs={acc['max_abs_err']:.3e}")

    # --- 5. 二分定位（hsi）：1/64/4096, 1/1024/1024, 1/256/1024 ---
    bisect_points = [(1, 64, 4096), (1, 1024, 1024), (1, 256, 1024)]
    for mn, opt, mx in bisect_points:
        out_path = scratch / f"hsi_fp16_bisect_{mn}_{opt}_{mx}.plan"
        info = build_fp16_detailed("hsi", mn, opt, mx, out_path)
        if not info.get("build_failed"):
            acc = eval_accuracy(out_path, "hsi")
            info["accuracy_cos_min"] = acc["cos_sim_min"]
            info["accuracy_max_abs_err"] = acc["max_abs_err"]
        report["bisection"].append(info)
        print(f"[bisect] hsi {mn}/{opt}/{mx}: cos_min={info.get('accuracy_cos_min')} "
              f"max_abs={info.get('accuracy_max_abs_err')}")

    # --- 6. 条件分支：若 deploy 精度确认劣化（cos_min 明显高于 default 且 max_abs 明显小于 default 的 FP16 量级），
    #        尝试强制逐层 FP16 + OBEY_PRECISION_CONSTRAINTS，看能否恢复 FP16 精度特征 ---
    deploy_acc = report["accuracy"].get("hsi", {}).get("deploy")
    default_acc = report["accuracy"].get("hsi", {}).get("default")
    degraded = (deploy_acc is not None and default_acc is not None and
                deploy_acc["max_abs_err"] < default_acc["max_abs_err"] / 5)
    report["degraded_precision_detected"] = bool(degraded)
    if degraded:
        forced_path = scratch / "hsi_fp16_deploy_forced.plan"
        finfo = build_fp16_detailed("hsi", 1, 1024, 4096, forced_path, force_precision_constraints=True)
        if not finfo.get("build_failed"):
            facc = eval_accuracy(forced_path, "hsi")
            finfo["accuracy"] = facc
            finsp = inspect_precision(forced_path, "hsi", 1024)
            finfo["inspection"] = finsp
        report["forced_fp16"] = finfo
        print(f"[forced_fp16] build_failed={finfo.get('build_failed')} accuracy={finfo.get('accuracy')}")

        # --- 7. 条件分支：若拿到了真正 FP16 的大 profile engine，复测 launch-bound 假设 ---
        if not finfo.get("build_failed") and finfo.get("accuracy", {}).get("max_abs_err", 1.0) < \
                default_acc["max_abs_err"] * 5:
            bench = bench_batches(forced_path, "hsi", [64, 256, 1024, 4096])
            bench_default = bench_batches(ROOT / "engines/hsi_fp16.plan", "hsi", [64])
            report["benchmark_conditional"] = {
                "forced_fp16_deploy": bench, "existing_default_fp16_batch64": bench_default,
                "note": "对照阶段3 warmup50/measure300 口径；仅用于判断 launch-bound 假设，不改变部署推荐",
            }
            print(f"[benchmark_conditional] {report['benchmark_conditional']}")

    out_json = scratch / "stage6_subtask1_engine_precision.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"written {out_json}")


if __name__ == "__main__":
    main()
