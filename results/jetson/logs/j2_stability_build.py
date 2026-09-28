import sys; sys.path.insert(0,"scripts")
import build_trt as b
from pathlib import Path
S=Path("scratch/stage7_engines")
for i in (1,2):
    for w in ("hsi","pc"):
        b.build_engine(w,"fp16",out_path=S/f"{w}_stability_{i}.plan")
