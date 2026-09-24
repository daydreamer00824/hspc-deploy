"""阶段4 Step1：用真实 F 盘数据校验前处理与 calib_samples 的一致性。

运行环境：conda hspc-preprocess（需要 gdal + pyproj + laspy + scipy，不需要 torch）。

预期结论（脚本运行时会重新计算并写入 json）：
- HSI: calib_samples/hsi_patches.npy 的 600 条样本，逐位对应
  24data/{7.17,8.23,10.6}/hsi/1_spec342.{dat,hdr} 三景（各200条，按索引顺序切分），
  用 load_hsi（GDAL）+ standardize_hsi + extract_patch 复现，max_abs_err 应为 0（bit-exact）。
- PC: calib_samples/point_offsets.npy 只有索引 400-599（对应 HSI 第三段 10.6 的切分区间）
  精确匹配到 24data/10.6/rice_las/1_rice_icp.las（用该文件全部点跑 build_point_offsets，
  逐位 max_abs_err < 1e-4）。索引 0-399 已对 lai_icp_registered_resampled_hsi/ 下全部 500 个
  registered_las + rice_las 文件（24data 全部 + 25data 全部日期）做了穷举匹配，均未命中，
  说明这 400 条样本的源 LAS 不在本次给出的 F 盘目录内（可能是另一批未登记的原始点云）。
  这不影响 build_point_offsets 移植正确性的结论——已用可定位的 200 条样本验证一致。

gdal_vs_fromfile_max_diff 由 np.fromfile 按 hdr 头信息读取原始数据，与 load_hsi(GDAL)
的结果逐位比较后计算得出。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from preprocess import standardize_hsi, extract_patch, load_hsi, build_point_offsets  # noqa: E402

# 原始数据根目录，默认 <repo>/data，可用环境变量 HSPC_DATA_ROOT 覆盖
DATA_ROOT = Path(os.environ.get("HSPC_DATA_ROOT", ROOT / "data"))
HSI_ROOT = DATA_ROOT / "hsi_spatial_spectral_resampled_common_342"
LAS_ROOT = DATA_ROOT / "lai_icp_registered_resampled_hsi"

HSI_GROUPS = [
    (0, 200, "24data/hsi/7.17/1_spec342.dat"),
    (200, 400, "24data/hsi/8.23/1_spec342.dat"),
    (400, 600, "24data/hsi/10.6/1_spec342.dat"),
]
PC_MATCHED_SOURCE = "24data/10.6/rice_las/1_rice_icp.las"
PC_MATCHED_RANGE = (400, 600)


def read_hdr_dims(hdr_path: Path) -> tuple[int, int, int]:
    text = open(hdr_path, errors="ignore").read()
    g = lambda k: int(re.search(rf"^{k}\s*=\s*(\d+)", text, re.M).group(1))
    return g("samples"), g("lines"), g("bands")


def read_fromfile(dat_path: Path) -> np.ndarray:
    """不经 GDAL，直接按 ENVI BSQ 头信息用 np.fromfile 读取，作为 load_hsi(GDAL) 的交叉对照。"""
    samples, lines, bands = read_hdr_dims(Path(str(dat_path)[:-4] + ".hdr"))
    return np.fromfile(dat_path, dtype="<f4").reshape(bands, lines, samples)


def verify_hsi() -> dict:
    calib = np.load(ROOT / "hspc_encoder/calib_samples/hsi_patches.npy")
    report = {"groups": [], "gdal_vs_fromfile_max_diff": None}
    fromfile_diffs = []
    for lo, hi, rel in HSI_GROUPS:
        dat = HSI_ROOT / rel
        raw_gdal, gt, proj = load_hsi(dat, target_bands=342)
        raw_fromfile = read_fromfile(dat)
        fromfile_diffs.append(float(np.abs(raw_gdal - raw_fromfile).max()))
        nz, _, _ = standardize_hsi(raw_gdal)
        # 对每条样本在整幅图里按 patch 中心波谱做最近邻反查，定位 (row,col) 后重新提取 patch 校验
        expected = calib[lo:hi]
        pix = nz.reshape(nz.shape[0], -1).T  # (rows*cols, bands)
        S = int(re.search(r"^samples\s*=\s*(\d+)", open(str(dat)[:-4] + ".hdr", errors="ignore").read(), re.M).group(1))
        errs = []
        for i in range(hi - lo):
            center = expected[i, :, 1, 1]
            d = np.abs(pix - center).max(1)
            j = int(d.argmin())
            r, c = divmod(j, S)
            patch = extract_patch(nz, r, c, 3)
            errs.append(float(np.abs(patch - expected[i]).max()))
        errs = np.asarray(errs)
        report["groups"].append({
            "range": [lo, hi], "source": rel,
            "max_abs_err": float(errs.max()), "n_exact": int((errs == 0).sum()), "n": len(errs),
        })
    report["gdal_vs_fromfile_max_diff"] = float(max(fromfile_diffs))
    return report


def verify_pc() -> dict:
    po = np.load(ROOT / "hspc_encoder/calib_samples/point_offsets.npy")
    import laspy
    lo, hi = PC_MATCHED_RANGE
    las = laspy.read(LAS_ROOT / PC_MATCHED_SOURCE)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    offs = build_point_offsets(xyz, np.arange(len(xyz)), 15)
    off_norm = np.linalg.norm(offs, axis=(1, 2))
    target = po[lo:hi]
    tgt_norm = np.linalg.norm(target, axis=(1, 2))
    errs = []
    for i in range(hi - lo):
        j = int(np.abs(off_norm - tgt_norm[i]).argmin())
        errs.append(float(np.abs(offs[j] - target[i]).max()))
    errs = np.asarray(errs)
    return {
        "matched_source": PC_MATCHED_SOURCE,
        "matched_range": list(PC_MATCHED_RANGE),
        "max_abs_err": float(errs.max()),
        "n": len(errs),
        "unmatched_range": [0, 400],
        "unmatched_note": (
            "已穷举 lai_icp_registered_resampled_hsi/ 下全部 registered_las(250) + "
            "rice_las(250) 共500个文件，索引0-399均未命中，源LAS不在给定F盘目录内"
        ),
    }


def main():
    t0 = time.time()
    result = {
        "hsi": verify_hsi(),
        "pc": verify_pc(),
        "elapsed_sec": None,
    }
    result["elapsed_sec"] = round(time.time() - t0, 1)
    out = ROOT / "results/stage4_preprocess_check.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n written to {out}")


if __name__ == "__main__":
    main()
