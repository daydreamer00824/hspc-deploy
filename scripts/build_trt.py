"""TensorRT engine 构建：FP32(TF32关闭) / FP32(TF32默认) / FP16 / INT8(隐式量化, TRT原生calibrator)。

INT8 显式 QDQ 路径见 scripts/ptq_modelopt.py + 本文件的 build_from_onnx(int8=False,...)复用。
构建顺序即诊断顺序：FP32(noTF32) 先建先验，是转换正确性闸门。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hspc_common as hc

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
ONNX_DIR = hc.ROOT / "onnx"
ENGINE_DIR = hc.ROOT / "engines"
LOG_DIR = hc.ROOT / "results" / "logs"
ENGINE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = {
    # 阶段2/3 的验证与基准口径（engines/{which}_{mode}.plan），保持不变
    "default": {"min": 1, "opt": 8, "max": 64},
    # 阶段5 的尝试（auto 精度 + 大 profile，已废弃）：大 profile 下 TRT 自选精度退回近 FP32，
    # 反而更慢；产物 engines/*_deploy.plan 仅作对照记录
    "deploy": {"min": 1, "opt": 1024, "max": 4096},
    # 阶段7 正式整景 engine（需配合 --precision-policy mixed）：max 由档位扫描选定
    # （results/stage7_scene_engine_sweep.json：2048 与 4096 速度差 <5%，显存约一半）
    "scene": {"min": 1, "opt": 2048, "max": 2048},
}
INPUT_SHAPES = {"pc": (15, 3), "hsi": (342, 3, 3)}

# ---------------------------------------------------------------------------
# 选择性混合精度（阶段7）：fp16 模式下 TRT 自选精度不稳定（阶段6：同配置多次构建
# max_abs 在 1e-4~0.3 间跳，偶尔整体退回 FP32），且纯 HALF 过不了 cos_min>=0.999 门槛。
# mixed 策略用 OBEY_PRECISION_CONSTRAINTS 把全部算术层钉死：fp32_groups 内的层为 FLOAT，
# 其余为 HALF。层按 ONNX 节点名归组（ONNX 由本仓库导出、名字固定），不依赖 TRT 内部
# 融合后的 myelin 层名，换 TRT 版本（如 Jetson）也能复用同一份组定义。
# ---------------------------------------------------------------------------
ARITH_TYPES = {
    trt.LayerType.CONVOLUTION, trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ELEMENTWISE,
    trt.LayerType.ACTIVATION, trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX,
    trt.LayerType.SCALE, trt.LayerType.POOLING, trt.LayerType.EINSUM,
    trt.LayerType.REDUCE, trt.LayerType.UNARY,
}
LAYER_GROUPS = ["layernorm", "softmax", "attn_qkv", "attn_qk", "attn_out", "residual",
                "ffn_linear", "gelu", "out_head", "stem", "posenc", "input_proj"]

_ATTN_QKV = {"MatMul", "Add"}
_ATTN_QK = {"Sqrt", "Sqrt_1", "Sqrt_2", "Div_1", "Mul_1", "Mul_2", "MatMul_1"}
_ATTN_OUT = {"MatMul_2", "Gemm"}


def classify_layer(name: str) -> str | None:
    """按 ONNX 节点名归组；返回 None 表示无名层（由调用方并入前一个算术层的组）。"""
    if name.startswith("(Unnamed Layer"):
        return None
    if "LayerNormalization" in name:
        return "layernorm"
    if "/out_head/" in name:
        return "out_head"
    if "/stem/" in name:
        return "stem"
    if "/pos_enc" in name or name == "/Add" or name.startswith("ONNXTRT_ShapeElementWise"):
        return "posenc"
    if "/input_proj/" in name:
        return "input_proj"
    if "/self_attn/" in name:
        op = name.rsplit("/", 1)[-1]
        if op == "Softmax":
            return "softmax"
        if op in _ATTN_QKV:
            return "attn_qkv"
        if op in _ATTN_QK:
            return "attn_qk"
        if op in _ATTN_OUT:
            return "attn_out"
    if "/linear1/" in name or "/linear2/" in name:
        return "ffn_linear"
    if "/transformer/layers." in name:
        op = name.rsplit("/", 1)[-1]
        if op in ("Add", "Add_2"):
            return "residual"
        if op in ("Div", "Erf", "Add_1", "Mul", "Mul_1"):
            return "gelu"
    raise ValueError(f"未归组的算术层：{name}")


def apply_mixed_policy(network, fp32_groups) -> dict:
    """把算术层（输出为浮点）按组设 FLOAT / HALF。无名层（Gemm 的 bias add）跟随前一个算术层。"""
    fp32_groups = set(fp32_groups)
    unknown = fp32_groups - set(LAYER_GROUPS)
    if unknown:
        raise ValueError(f"未知的层组：{sorted(unknown)}")
    prev_group = None
    fp32_layers, n_half = [], 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        outputs = [layer.get_output(o) for o in range(layer.num_outputs)]
        if layer.type not in ARITH_TYPES or not outputs or not all(
                t.dtype in (trt.DataType.FLOAT, trt.DataType.HALF) for t in outputs):
            continue
        group = classify_layer(layer.name) or prev_group
        prev_group = group
        dtype = trt.DataType.FLOAT if group in fp32_groups else trt.DataType.HALF
        layer.precision = dtype
        for o in range(layer.num_outputs):
            layer.set_output_type(o, dtype)
        if dtype == trt.DataType.FLOAT:
            fp32_layers.append(layer.name)
        else:
            n_half += 1
    return {"fp32_groups": sorted(fp32_groups), "fp32_layers": fp32_layers,
            "n_fp32": len(fp32_layers), "n_half": n_half}


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """TRT 原生隐式量化 calibrator（TRT10 已弃用但仍受支持的路径），用前400条真实样本。"""

    def __init__(self, which: str, cache_path: Path, batch_size: int = 32):
        super().__init__()
        import torch
        self.which = which
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.data = hc.load_samples(which, "calib")  # 前400条，仅供校准
        self.n = len(self.data)
        self.idx = 0
        self._torch = torch
        self._dev_buf = None

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx >= self.n:
            return None
        batch = self.data[self.idx:self.idx + self.batch_size]
        self.idx += len(batch)
        t = self._torch.from_numpy(batch).float().cuda().contiguous()
        self._dev_buf = t  # 保持引用，防止被垃圾回收
        return [t.data_ptr()]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)


def build_network_from_onnx(builder: trt.Builder, onnx_path: Path):
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")
    return network


def make_profile(builder: trt.Builder, which: str, profile_name: str = "default") -> trt.IOptimizationProfile:
    PROFILE = PROFILES[profile_name]
    profile = builder.create_optimization_profile()
    dims = INPUT_SHAPES[which]
    profile.set_shape(
        hc.INPUT_NAME,
        min=(PROFILE["min"], *dims),
        opt=(PROFILE["opt"], *dims),
        max=(PROFILE["max"], *dims),
    )
    return profile


def build_engine(which: str, mode: str, onnx_path: Path | None = None, profile_name: str = "default",
                 precision_policy: str = "auto", fp32_groups=(), out_path: Path | None = None,
                 policy_info: dict | None = None) -> Path:
    """mode: fp32_notf32 | fp32_tf32 | fp16 | int8_implicit | int8_qdq

    precision_policy（仅 mode=fp16 有效）：auto = TRT 自选精度（阶段2~6 的行为）；
    mixed = OBEY_PRECISION_CONSTRAINTS，fp32_groups 内的层 FLOAT、其余算术层 HALF。
    policy_info 若传入 dict，会被填入 apply_mixed_policy 的结果（FP32 层列表等）。
    """
    if onnx_path is None:
        src_name = f"{which}_qdq.onnx" if mode == "int8_qdq" else f"{which}_encoder.onnx"
        onnx_path = ONNX_DIR / src_name
    if mode == "int8_qdq" and not onnx_path.exists():
        raise FileNotFoundError(f"{onnx_path} 不存在，先运行 scripts/ptq_modelopt.py")
    if precision_policy not in ("auto", "mixed"):
        raise ValueError(precision_policy)
    if precision_policy == "mixed" and mode != "fp16":
        raise ValueError("mixed 精度策略只用于 fp16 模式")
    if out_path is None:
        suffix = "" if profile_name == "default" else f"_{profile_name}"
        out_path = ENGINE_DIR / f"{which}_{mode}{suffix}.plan"

    builder = trt.Builder(TRT_LOGGER)
    network = build_network_from_onnx(builder, onnx_path)
    config = builder.create_builder_config()
    config.add_optimization_profile(make_profile(builder, which, profile_name))

    # TF32 默认是开启的，除 fp32_tf32 模式外一律显式关闭
    if mode != "fp32_tf32":
        config.clear_flag(trt.BuilderFlag.TF32)
    else:
        config.set_flag(trt.BuilderFlag.TF32)

    calibrator = None
    if mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        if precision_policy == "mixed":
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            info = apply_mixed_policy(network, fp32_groups)
            if policy_info is not None:
                policy_info.update(info)
    elif mode == "int8_implicit":
        config.set_flag(trt.BuilderFlag.FP16)  # 允许 INT8 不支持的层退回 FP16
        config.set_flag(trt.BuilderFlag.INT8)
        cache_path = ENGINE_DIR / f"{which}_int8_implicit.calib"
        calibrator = EntropyCalibrator(which, cache_path)
        config.int8_calibrator = calibrator
    elif mode == "int8_qdq":
        # 显式量化：QDQ 节点已带 scale/zero-point，无需 calibrator；仅需允许 INT8/FP16 kernel
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)

    print(f"[{which}/{mode}/{precision_policy}{list(fp32_groups) if precision_policy == 'mixed' else ''}] "
          f"building from {onnx_path.name} -> {out_path.name} ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    dt = time.time() - t0
    if serialized is None:
        # 备用诊断路径：用 trtexec 复现同一构建拿完整日志，例如
        #   /usr/src/tensorrt/bin/trtexec --onnx=<onnx> --int8 --fp16 --verbose \
        #     > results/logs/build_<which>_<mode>.log 2>&1
        raise RuntimeError(
            f"build failed for {which}/{mode} (onnx={onnx_path.name})；"
            f"用 trtexec 复现该构建以获取完整日志，输出到 {LOG_DIR}")
    out_path.write_bytes(bytes(serialized))
    print(f"[{which}/{mode}] done in {dt:.1f}s, size={out_path.stat().st_size/1024:.1f} KB")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["pc", "hsi"])
    ap.add_argument("--modes", nargs="+",
                     default=["fp32_notf32", "fp32_tf32", "fp16", "int8_implicit"])
    ap.add_argument("--profile", choices=list(PROFILES), default="default")
    ap.add_argument("--precision-policy", choices=["auto", "mixed"], default="auto")
    ap.add_argument("--fp32-groups", nargs="*", default=[], choices=LAYER_GROUPS)
    args = ap.parse_args()

    for which in args.which:
        for mode in args.modes:
            build_engine(which, mode, profile_name=args.profile,
                         precision_policy=args.precision_policy, fp32_groups=args.fp32_groups)


if __name__ == "__main__":
    main()
