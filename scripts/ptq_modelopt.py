"""nvidia-modelopt PTQ：前400条真实样本校准 -> 显式 QDQ ONNX（TRT10推荐的INT8路径）。

输出 onnx/{pc,hsi}_qdq.onnx，交给 build_trt.py 的 build_from_onnx 用普通(非calibrator)
路径构建 int8_qdq engine（QDQ节点自带量化信息，构建时不需要再传 calibrator）。
"""
import sys
from pathlib import Path

import numpy as np
from modelopt.onnx.quantization import quantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hspc_common as hc

ONNX_DIR = hc.ROOT / "onnx"


def ptq_one(which: str):
    src = ONNX_DIR / f"{which}_encoder.onnx"
    dst = ONNX_DIR / f"{which}_qdq.onnx"
    calib = hc.load_samples(which, "calib")  # 前400条，仅供校准
    print(f"[{which}] PTQ calibrating with {calib.shape} real samples -> {dst.name}")
    quantize(
        onnx_path=str(src),
        quantize_mode="int8",
        calibration_data={hc.INPUT_NAME: calib},
        calibration_method="entropy",
        output_path=str(dst),
        high_precision_dtype="fp16",  # 非量化算子回退 FP16（与 int8_implicit 路径对齐可比）
        calibration_eps=["cuda:0", "cpu"],
    )
    print(f"[{which}] QDQ model saved -> {dst.relative_to(hc.ROOT)} "
          f"({dst.stat().st_size/1024:.1f} KB)")


def main():
    for which in ("pc", "hsi"):
        ptq_one(which)


if __name__ == "__main__":
    main()
