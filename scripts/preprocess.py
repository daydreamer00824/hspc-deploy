"""前处理：原始数据 -> encoder 输入张量。

来源说明：
- `build_point_offsets` = 从 hspc_encoder/inference.py 逐行原样移植
- `standardize_hsi` / `extract_patch` / `load_hsi` / `valid_vegetation_mask`
  = 从原始科研工程的 hsi.py 逐行原样移植
  （已与原文件逐行对照，逻辑一致）
- `project_xyz_to_canonical_pixels` / `canonical_in_bounds_mask`
  = 从同一工程 geometry.py 逐行原样移植
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# PC 前处理：原样移植自 hspc_encoder/inference.py
# ---------------------------------------------------------------------------

def build_point_offsets(xyz: np.ndarray, point_indices, k: int = 15, balanced_tree: bool = True) -> np.ndarray:
    """(N,3) 点云 + 待查询点索引 -> (len(point_indices), k, 3) 邻域坐标偏移。

    balanced_tree=False（阶段7步骤4新增，默认仍是 True 不改变已有调用方行为）：cKDTree 建树用
    sliding-midpoint 规则代替默认的 balanced 规则，在24data/10.6/1这个场景（159897点查询1000个
    点的15近邻）上实测快约1.6~1.7x，且用真实eligible点验证过 k近邻结果（含最终的offsets输出）
    逐位相同——kNN 找到的近邻集合不依赖树的具体划分方式，是精确搜索。

    但有两点保留意见：
    1. 当查询点到多个候选近邻的距离恰好相等（tie）时，不同建树方式下 cKDTree 返回的近邻顺序
       理论上可能不同——本项目验证时该场景没有触发过这种 tie（逐位比对全部通过），不代表任意
       点云分布下都不会发生。`build_point_offsets` 的近邻顺序会影响 `output` 里 15 个偏移量的
       排列，如果换到新场景后逐位比对不通过，先检查是否有并列距离的近邻
    2. 逐位验证只覆盖了 24data/10.6/1 这一景，换场景/换点云密度分布后建议重新验证

    所以默认关闭（balanced_tree=True，原行为），只有 scripts/deploy_scene.py 显式传
    balanced_tree=False 来用。
    """
    tree = cKDTree(xyz, balanced_tree=balanced_tree)
    query_k = min(k + 1, len(xyz))
    _, neighbors = tree.query(xyz[point_indices], k=query_k)
    neighbors = np.atleast_2d(neighbors).astype(np.int64)
    output = np.empty((len(point_indices), k, 3), dtype=np.float32)
    for j, point_index in enumerate(point_indices):
        current = neighbors[j]
        current = current[current != point_index]
        if current.size == 0:
            current = np.asarray([point_index])
        if current.size < k:
            current = np.pad(current, (0, k - current.size), mode="edge")
        output[j] = (xyz[current[:k]] - xyz[point_index]).astype(np.float32)
    return output


# ---------------------------------------------------------------------------
# HSI 前处理：原样移植自原科研工程的 hsi.py
# ---------------------------------------------------------------------------

def load_hsi(path, target_bands: int = 342):
    """读取高光谱影像文件 -> (raw, geotransform, projection)。

    依赖 GDAL（`osgeo.gdal`），当前环境未安装，见 README「前处理」章节。
    """
    from pathlib import Path
    from osgeo import gdal

    path = Path(path)
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Cannot open HSI: {path}")
    raw = dataset.ReadAsArray()
    if raw.ndim == 2:
        raw = raw[None, :, :]
    elif raw.shape[0] <= 100 and raw.shape[-1] > 100:
        raw = np.moveaxis(raw, -1, 0)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[0] != target_bands:
        raise ValueError(f"Expected exactly {target_bands} bands, got {raw.shape[0]}: {path}")
    return raw, dataset.GetGeoTransform(), dataset.GetProjection()


def valid_vegetation_mask(raw, red_index=143, nir_index=262, ndvi_threshold=0.2, nodata_epsilon=1e-5):
    """NDVI + nodata 过滤，得到可采样的植被像元掩膜。"""
    red = raw[red_index].astype(np.float32)
    nir = raw[nir_index].astype(np.float32)
    denom = nir + red
    ndvi = np.divide(nir - red, denom, out=np.zeros_like(denom), where=np.abs(denom) > 1e-12)
    return (ndvi > ndvi_threshold) & (np.sum(raw, axis=0) > nodata_epsilon)


def standardize_hsi(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bands, rows, cols) -> 按波段 z-score 标准化。

    注意：统计量对整幅影像的全部像素计算，不做 valid_mask 过滤。
    """
    mean = raw.mean(axis=(1, 2), keepdims=True)
    std = raw.std(axis=(1, 2), keepdims=True) + 1e-8
    return ((raw - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def extract_patch(cube: np.ndarray, row: int, col: int, patch_size: int = 3) -> np.ndarray:
    """从 (bands, rows, cols) 取以 (row,col) 为中心的 patch_size×patch_size 窗口。

    边界处理：零值填充（不是坐标钳位；两者在边界样本上的结果不同）。
    """
    bands, rows, cols = cube.shape
    half = patch_size // 2
    r0, r1 = max(0, row - half), min(rows, row + half + 1)
    c0, c1 = max(0, col - half), min(cols, col + half + 1)
    patch = cube[:, r0:r1, c0:c1]
    pad = (
        (0, 0),
        (max(0, half - row), max(0, row + half + 1 - rows)),
        (max(0, half - col), max(0, col + half + 1 - cols)),
    )
    if patch.shape[1:] != (patch_size, patch_size):
        patch = np.pad(patch, pad, mode="constant", constant_values=0)
    if patch.shape != (bands, patch_size, patch_size):
        raise AssertionError(f"Patch shape mismatch: {patch.shape}")
    return np.asarray(patch, dtype=np.float32)


def build_hsi_patches(normalized: np.ndarray, rowcols, patch_size: int = 3) -> np.ndarray:
    """批量提取 -> (N, bands, patch, patch)，可直接喂 HSI encoder。"""
    return np.stack([extract_patch(normalized, r, c, patch_size) for r, c in rowcols])


def iter_hsi_patch_batches(normalized: np.ndarray, rowcols, batch: int, patch_size: int = 3):
    """`build_hsi_patches` 的向量化版本（部署推理优化尝试）：一次零填充整幅影像，
    用 `sliding_window_view` 取零拷贝滑窗视图，fancy-index 出 patch。
    结果与 `build_hsi_patches`（Python 循环 + `extract_patch`）逐位一致。

    已实测无稳定收益，deploy_scene.py 未采用：在真实场景规模（9645~15840个patch）上，
    跟 build_hsi_patches 的 Python 循环相比，15840个时快约1.9x（82ms vs 153ms），但9645个
    （即只对有效像元推理时的实际规模）反而更慢（91ms vs 81ms）——根因是 sliding_window_view
    产生的多维滑窗视图存在跨band的大跨度访存，fancy indexing 本质仍是逐元素 gather，
    numpy 在这种访存模式下赚不到向量化的便宜。保留此函数供以后需要全图（非仅有效像元）
    批量提取时参考，不要默认认为它比循环快。
    只支持奇数 patch_size（本项目固定 patch_size=3，符合要求）。

    用法：`for batch_patches in iter_hsi_patch_batches(normalized, rowcols, 64): ...`
    """
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size 必须为奇数，got {patch_size}")
    half = patch_size // 2
    padded = np.pad(normalized, ((0, 0), (half, half), (half, half)), mode="constant", constant_values=0)
    # windows: (bands, rows, cols, patch, patch)，stride trick 零拷贝视图
    windows = np.lib.stride_tricks.sliding_window_view(padded, (patch_size, patch_size), axis=(1, 2))
    rc = np.asarray(rowcols)
    for i in range(0, len(rc), batch):
        r, c = rc[i:i + batch, 0], rc[i:i + batch, 1]
        out = windows[:, r, c]  # fancy index -> (bands, n, patch, patch)
        yield np.ascontiguousarray(np.moveaxis(out, 1, 0), dtype=np.float32)  # (n, bands, patch, patch)


# ---------------------------------------------------------------------------
# 几何：原样移植自原科研工程的 geometry.py
# ---------------------------------------------------------------------------

def validate_geotransform(gt):
    values = np.asarray(gt, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError(f"GeoTransform must contain six finite values: {gt}")
    matrix = np.asarray([[values[1], values[2]], [values[4], values[5]]], dtype=np.float64)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-18:
        raise ValueError("Non-invertible GeoTransform")
    return tuple(float(value) for value in values)


def map_to_fractional_pixel(gt, x, y):
    gt = validate_geotransform(gt)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    matrix = np.asarray([[gt[1], gt[2]], [gt[4], gt[5]]], dtype=np.float64)
    inv = np.linalg.inv(matrix)
    delta = np.stack([x - gt[0], y - gt[3]], axis=0)
    col_row = inv @ delta.reshape(2, -1)
    cols = col_row[0].reshape(x.shape)
    rows = col_row[1].reshape(y.shape)
    return rows, cols


def map_to_canonical_pixel(gt, x, y):
    rows_f, cols_f = map_to_fractional_pixel(gt, x, y)
    if not np.all(np.isfinite(rows_f)) or not np.all(np.isfinite(cols_f)):
        raise ValueError("Non-finite fractional pixel coordinate")
    return np.floor(rows_f).astype(np.int64), np.floor(cols_f).astype(np.int64)


def project_xyz_to_canonical_pixels(x, y, source_crs, target_crs, gt, spatial_contract):
    """LAS XY -> HSI canonical (row, col)。依赖 pyproj（当前环境未安装）。"""
    from pyproj import CRS, Transformer

    source = CRS.from_user_input(source_crs)
    target = CRS.from_user_input(target_crs)
    expected_source = CRS.from_user_input(spatial_contract["source_crs"])
    expected_target = CRS.from_user_input(spatial_contract["target_crs"])
    if not source.equals(expected_source):
        raise AssertionError(f"Unexpected source CRS: actual={source}, expected={expected_source}")
    if not target.equals(expected_target):
        raise AssertionError(f"Unexpected target CRS: actual={target}, expected={expected_target}")
    transformer = Transformer.from_crs(source, target, always_xy=True)
    hsi_x, hsi_y = transformer.transform(x, y)
    return map_to_canonical_pixel(gt, hsi_x, hsi_y)


def canonical_in_bounds_mask(rows, cols, image_rows: int, image_cols: int):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.shape != cols.shape:
        raise ValueError("Canonical row/col shapes differ")
    return (rows >= 0) & (rows < image_rows) & (cols >= 0) & (cols < image_cols)


# ---------------------------------------------------------------------------
# 校验：拿真实数据确认前处理与 calib_samples 一致
# ---------------------------------------------------------------------------

def verify_hsi_preprocess(raw: np.ndarray, rowcols, expected_patches: np.ndarray,
                          patch_size: int = 3, tol: float = 1e-4) -> dict:
    """用一景原始 HSI + 已知正确的 patch 输出，端到端校验前处理链路。"""
    normalized, _, _ = standardize_hsi(raw)
    got = build_hsi_patches(normalized, rowcols, patch_size)
    err = np.abs(got - expected_patches)
    ok = bool(err.max() <= tol)
    return {
        "match": ok,
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "n": int(len(got)),
        "verdict": "与原实现一致，可用于部署" if ok else "不一致，需要复查",
    }
