"""阶段4 Step2 Stage A：场景级前处理（不含模型推理）。

运行环境：conda hspc-preprocess（gdal + pyproj + laspy + scipy，无 torch/tensorrt）。
场景固定为 24data/10.6/1（未参与 INT8 校准，是现有精度验证集来源景）。

产出 results/e2e_scene_preprocessed.npz，供 Stage B（modelopt 环境）做模型推理与 score_window 匹配。

参数来源：原科研工程的 config 契约文件 repair_contract_v2.json、spatial_mapping_contract_v1.json
（只读这两个 json）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from preprocess import (  # noqa: E402
    build_hsi_patches,
    build_point_offsets,
    canonical_in_bounds_mask,
    load_hsi,
    project_xyz_to_canonical_pixels,
    standardize_hsi,
    valid_vegetation_mask,
)
from matching import VARIANTS  # noqa: E402

# 原始数据根目录，默认 <repo>/data，可用环境变量 HSPC_DATA_ROOT 覆盖
DATA_ROOT = Path(os.environ.get("HSPC_DATA_ROOT", ROOT / "data"))
HSI_PATH = DATA_ROOT / "hsi_spatial_spectral_resampled_common_342/24data/hsi/10.6/1_spec342.dat"
LAS_PATH = DATA_ROOT / "lai_icp_registered_resampled_hsi/24data/10.6/rice_las/1_rice_icp.las"

CONTRACT = {
    "target_bands": 342,
    "point_cloud_crs": "EPSG:32651",
    "red_band_index": 143,
    "nir_band_index": 262,
    "ndvi_threshold": 0.2,
    "nodata_epsilon": 1e-5,
    "patch_size": 3,
    "point_neighbors": 15,
    "search_radius": 5,
    "inference_variants": {
        name: {"checkpoint": "global_only", "spatial_weight": w} for name, w in VARIANTS.items()
    },
}
SPATIAL_CONTRACT = {
    "source_crs": "EPSG:32651",
    "target_crs": "EPSG:4326",
}
SAMPLES_PER_SCENE = 1000
SEED = 20260617


def main():
    import laspy

    print(f"[1/6] load_hsi: {HSI_PATH}")
    raw, gt, projection = load_hsi(HSI_PATH, CONTRACT["target_bands"])
    bands, rows, cols = raw.shape
    print(f"  shape={raw.shape} gt={gt}")

    print("[2/6] standardize + full patch grid")
    normalized, _, _ = standardize_hsi(raw)
    rowcols_all = [(r, c) for r in range(rows) for c in range(cols)]
    all_patches = build_hsi_patches(normalized, rowcols_all, CONTRACT["patch_size"])
    print(f"  patches: {all_patches.shape}")

    print("[3/6] valid_vegetation_mask")
    valid_mask = valid_vegetation_mask(
        raw, CONTRACT["red_band_index"], CONTRACT["nir_band_index"],
        CONTRACT["ndvi_threshold"], CONTRACT["nodata_epsilon"],
    )
    print(f"  valid pixels: {int(valid_mask.sum())}/{valid_mask.size}")

    print(f"[4/6] load LAS: {LAS_PATH}")
    las = laspy.read(LAS_PATH)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    print(f"  points: {len(xyz)}")

    print("[5/6] project xyz -> canonical pixels")
    # target_crs 直接用 GDAL 读出的每景 WKT 投影；spatial_mapping_contract_v1.json 的
    # target_crs="EPSG:4326"，project_xyz_to_canonical_pixels 内部会断言两者等价（ENVI头已确认为WGS-84经纬度）
    ref_rows, ref_cols = project_xyz_to_canonical_pixels(
        xyz[:, 0], xyz[:, 1], CONTRACT["point_cloud_crs"], projection, gt, SPATIAL_CONTRACT,
    )
    in_bounds = canonical_in_bounds_mask(ref_rows, ref_cols, rows, cols)
    ref_valid = np.zeros(len(xyz), dtype=bool)
    valid_idx = np.where(in_bounds)[0]
    ref_valid[valid_idx] = valid_mask[ref_rows[valid_idx], ref_cols[valid_idx]]
    eligible = np.arange(len(xyz), dtype=np.int64)[ref_valid]
    print(f"  in_bounds={int(in_bounds.sum())} eligible(valid)={len(eligible)}")

    rng = np.random.default_rng(SEED)
    if eligible.size > SAMPLES_PER_SCENE:
        eligible = np.sort(rng.choice(eligible, SAMPLES_PER_SCENE, replace=False))
    print(f"  sampled eligible: {len(eligible)}")

    print("[6/6] build_point_offsets for eligible points")
    offsets = build_point_offsets(xyz, eligible, CONTRACT["point_neighbors"])
    er, ec = ref_rows[eligible], ref_cols[eligible]
    truth = raw[:, er, ec].T.astype(np.float32)

    out = ROOT / "results/e2e_scene_preprocessed.npz"
    np.savez_compressed(
        out,
        hsi_grid_patches=all_patches.astype(np.float32),
        hsi_raw=raw.astype(np.float32),
        hsi_rows=rows, hsi_cols=cols,
        valid_mask=valid_mask,
        point_offsets=offsets.astype(np.float32),
        ref_rows=er, ref_cols=ec,
        truth_spectra=truth,
        eligible_point_indices=eligible,
        n_las_points=len(xyz),
    )
    meta = {
        "hsi_path": str(HSI_PATH), "las_path": str(LAS_PATH),
        "hsi_shape": [bands, rows, cols],
        "n_las_points": int(len(xyz)),
        "n_eligible_total_before_sample": int(ref_valid.sum()),
        "n_sampled": int(len(eligible)),
        "n_grid_patches": int(all_patches.shape[0]),
    }
    (ROOT / "results/e2e_scene_preprocessed_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
