"""PyTorch -> ONNX 导出（legacy TorchScript 导出器，dynamic batch）。

torch 2.13 的 torch.onnx.export 默认 dynamo=True（走新的 FX 导出器），
这里显式 dynamo=False 走 legacy 路径，以保证 dynamic_axes 语义可控、图结构可预期。
"""
import sys
from collections import Counter
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hspc_common as hc

OUT = hc.ROOT / "onnx"
OUT.mkdir(parents=True, exist_ok=True)

OPSET = 17


def op_histogram(model_path: Path) -> Counter:
    m = onnx.load(str(model_path))
    return Counter(n.op_type for n in m.graph.node)


def export_one(which: str):
    cfg = hc.load_config()
    m = hc.get_model(which, "cpu")
    dummy = hc.dummy_input(which, batch=2)  # batch=2 避免导出器把 batch 维特化成常量1
    out_path = OUT / f"{which}_encoder.onnx"

    with hc.math_sdpa():
        torch.onnx.export(
            m,
            (dummy,),
            str(out_path),
            input_names=[hc.INPUT_NAME],
            output_names=[hc.OUTPUT_NAME],
            dynamic_axes={hc.INPUT_NAME: {0: "batch"}, hc.OUTPUT_NAME: {0: "batch"}},
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    inferred = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(inferred, str(out_path))

    hist = op_histogram(out_path)
    print(f"[{which}] exported -> {out_path.relative_to(hc.ROOT)} "
          f"({out_path.stat().st_size/1024:.1f} KB)")
    print(f"  op histogram ({sum(hist.values())} nodes): "
          + ", ".join(f"{k}={v}" for k, v in sorted(hist.items())))
    return out_path


def main():
    hc.setup_determinism()
    for which in ("pc", "hsi"):
        export_one(which)


if __name__ == "__main__":
    main()
