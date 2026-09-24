"""TensorRT engine 加载与推理封装（execute_async_v3 + torch cuda tensor 作 binding）。

不依赖 pycuda/cuda-python：显存分配、H2D/D2H 拷贝全部通过 torch cuda tensor 完成，
用其 .data_ptr() 作为 TRT 的 device 地址。
"""
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hspc_common as hc

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


def load_engine(plan_path: Path) -> trt.ICudaEngine:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize engine: {plan_path}")
    return engine


class TrtRunner:
    """单输入单输出、dynamic batch 的 engine 推理器。"""

    def __init__(self, plan_path: Path, input_name=hc.INPUT_NAME, output_name=hc.OUTPUT_NAME):
        self.engine = load_engine(Path(plan_path))
        self.context = self.engine.create_execution_context()
        self.input_name = input_name
        self.output_name = output_name
        self.stream = torch.cuda.Stream()

    def infer(self, x: np.ndarray) -> np.ndarray:
        batch = x.shape[0]
        in_dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(self.input_name)]
        out_dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(self.output_name)]

        self.context.set_input_shape(self.input_name, x.shape)
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        assert out_shape[0] == batch, f"engine output batch {out_shape} != input batch {batch}"

        x_t = torch.from_numpy(x).to(in_dtype).cuda().contiguous()
        y_t = torch.empty(out_shape, dtype=out_dtype, device="cuda")

        self.context.set_tensor_address(self.input_name, x_t.data_ptr())
        self.context.set_tensor_address(self.output_name, y_t.data_ptr())

        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("execute_async_v3 failed")
        return y_t.float().cpu().numpy()

    def infer_batched(self, x: np.ndarray, max_batch: int) -> np.ndarray:
        """输入样本数超过 profile max 时按 max_batch 分块调用 infer。"""
        out = []
        for i in range(0, len(x), max_batch):
            out.append(self.infer(x[i:i + max_batch]))
        return np.concatenate(out, axis=0)
