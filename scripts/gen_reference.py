"""生成 PyTorch FP32 参考输出（CPU 与 CUDA 各一份，分离框架差异与硬件差异）。

用途：results/ref/{pc,hsi}_fp32_{cpu,cuda}.npy 是全流程唯一数值基准。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hspc_common as hc

OUT = hc.ROOT / "results" / "ref"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    hc.setup_determinism()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"devices: {devices}")

    for which in ("pc", "hsi"):
        x_val = hc.load_samples(which, "val")     # 后200条，精度验证专用
        x_calib = hc.load_samples(which, "calib")  # 前400条，仅供 INT8 校准脚本读取校验一致性
        print(f"[{which}] val={x_val.shape} calib={x_calib.shape}")
        for dev in devices:
            m = hc.get_model(which, dev)
            y = hc.torch_forward(m, x_val, device=dev)
            fname = OUT / f"{which}_fp32_{dev}.npy"
            np.save(fname, y)
            print(f"  saved {fname.relative_to(hc.ROOT)} {y.shape}")
            if dev != "cpu":
                y_cpu = np.load(OUT / f"{which}_fp32_cpu.npy")
                rep = hc.accuracy_report(y_cpu, y)
                hc.print_report(f"{which} cpu-vs-{dev}", rep)


if __name__ == "__main__":
    main()
