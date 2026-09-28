"""场景字符串到 F 盘原始数据路径的映射：deploy_scene.py、e2e_stage_a_preprocess.py、deploy_jetson.py、
cpp_guardrail.py 共用同一份定义，避免不同脚本对同一个 scene 字符串解析出不同的文件。

刻意保持零依赖（不 import torch/tensorrt/laspy），因为 e2e_stage_a_preprocess.py 运行在没有这些库的
conda hspc-preprocess 环境里。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 原始数据根目录，默认 <repo>/data，可用环境变量 HSPC_DATA_ROOT 覆盖
DATA_ROOT = Path(os.environ.get("HSPC_DATA_ROOT", ROOT / "data"))
DEFAULT_HSI_PATH = DATA_ROOT / "hsi_spatial_spectral_resampled_common_342/24data/hsi/10.6/1_spec342.dat"
DEFAULT_LAS_PATH = DATA_ROOT / "lai_icp_registered_resampled_hsi/24data/10.6/rice_las/1_rice_icp.las"


def scene_paths(scene: str) -> tuple[Path, Path]:
    """scene 形如 "25data/8.4/1"（批次/日期/样地号），返回 (HSI 路径, LAS 路径)。"""
    batch, date, sid = scene.split("/")
    return (DATA_ROOT / f"hsi_spatial_spectral_resampled_common_342/{batch}/hsi/{date}/{sid}_spec342.dat",
            DATA_ROOT / f"lai_icp_registered_resampled_hsi/{batch}/{date}/rice_las/{sid}_rice_icp.las")
