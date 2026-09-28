import sys; sys.path.insert(0,"scripts")
import build_trt as b
from pathlib import Path
S=Path("scratch/stage7_engines")
for i in range(3):
    b.build_engine("hsi","fp16",precision_policy="mixed",fp32_groups=["stem"],out_path=S/f"hsi_mixed_{i}.plan")
