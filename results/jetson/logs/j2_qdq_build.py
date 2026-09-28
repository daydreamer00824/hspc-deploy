"""int8_qdq 模式构建 3 次/模型（build_trt.build_engine 原样调用）；仅把 profiling_verbosity 设为 DETAILED 以便逐层核查精度，不影响 tactic 选择。"""
import sys; sys.path.insert(0,"scripts")
import tensorrt as trt, build_trt as b
from pathlib import Path
_orig = trt.Builder.create_builder_config
def _cfg(self):
    c = _orig(self); c.profiling_verbosity = trt.ProfilingVerbosity.DETAILED; return c
trt.Builder.create_builder_config = _cfg
S = Path("scratch/stage7_engines")
for i in range(3):
    for w in ("hsi", "pc"):
        b.build_engine(w, "int8_qdq", out_path=S / f"{w}_qdq_{i}.plan")
open("scratch/QDQ_FINISHED", "w").write("done")
