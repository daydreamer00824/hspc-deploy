"""部署推理优化（阶段5起）：单进程、不依赖 torch 的场景级推理链路。

背景：阶段4的端到端验收用的是验证/基准工具（trt_runner.py 依赖 torch、score_window 用
torch 张量逐点循环），实测有两处可优化的真实开销：
1. TrtRunner 每次调用都重新分配 CUDA 显存 + 创建新的 torch tensor，阶段4实测端到端比
   纯kernel慢 2.0~2.2倍（见 results/stage4_e2e.json 的 runner_overhead_vs_stage3_kernel）
2. torch 版 score_window_batch 的逐点循环里有大量 .clone()/张量创建/CPU开销，
   实测2个variant合计369ms；本文件改用 numpy 翻译同一套逻辑，实测127ms（2.9x）

本文件只依赖 numpy/scipy/gdal/pyproj/laspy/tensorrt/polygraphy，不依赖 torch。
运行环境：conda hspc-preprocess + PYTHONPATH=/usr/lib/python3.10/dist-packages（给 tensorrt
用系统 deb 绑定）。

已验证无稳定收益、因此没有采用的方案（详见 scripts/preprocess.py:iter_hsi_patch_batches
的 docstring）：
- sliding_window_view 向量化 patch 提取（在 9645~15840 个 patch 的实际规模下没有稳定优势）
- 固定窗口批量 einsum 的 score_window（对内部点浪费算力，比循环慢 5 倍）

关于大 batch profile：阶段5曾把"更大 profile 在 batch=64 下反而慢2倍"解释为 GPU 已经跑满，
该解释不成立——真正原因是当时的 fp16 用的是 TRT `auto` 精度模式，在大 profile 下会不稳定地
退回接近 FP32 的 tactic（阶段6用 nsys + 受控重建证实）。改用
`build_trt.py --precision-policy mixed`（选择性混合精度，见 results/stage7_mixed_precision.json）后，大 batch
profile 的 engine（`engines/*_fp16_scene.plan`，min1/opt2048/max2048）比 batch64 的
`engines/*_fp16.plan` 实测快 2.1~2.5x（`results/stage7_batch_pinned.json`、
`results/stage7_scene_engine_sweep.json`），已提升为整景推理的默认 engine（`--engine-tier scene`）。
batch64 版本保留给低延迟单点/小批场景用（`--engine-tier batch64`）。

`LiteTrtRunner(pinned=True)`：pinned host 内存已实现，但当前调用方式下没有稳定收益
（推断，未做进一步隔离实验确认根因）。现在的写法是先把数据准备在普通 numpy 数组里，
再整体拷进 pinned buffer（`self.h_in[:n] = x`），host 端的拷贝次数并没有减少，只是把"拷进
pageable buffer"换成了"拷进pinned buffer"，而 pinned 的收益本该来自"CPU 准备下一批的同时
GPU 处理当前批"的真正重叠，这需要数据直接生成在 pinned buffer 里（省掉一次拷贝）并配合
双缓冲流水线（本阶段不做）。这个结论不适用于 Jetson：Jetson 是 CPU/GPU
统一内存架构，pinned/pageable 的机制和这里测的 dGPU+PCIe 场景完全不同。

CUDA Graph：阶段7测过 CPU 开销占比（wall减去CUDA event测的GPU时间）在30%~54%之间，
数字上过了"20%才考虑"的门槛，但这个测量口径本身有问题——"墙钟减GPU时间"里包含了
host端数据准备（h_in拷贝、resize、set_input_shape），这些不是 CUDA Graph 能消除的
launch开销。且大batch后HSI阶段已降到约50ms，Graph 的理论收益上限本来就小。列为待定项，
本次不实现。

不改模型结构/权重/超参数。只对 valid_vegetation_mask 判定为有效的像元做 HSI 推理
（score_window 本来就把无效像元的分数置为 -inf，不影响匹配结果，本景只有约61%像元有效，
省下约40%的patch数）。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from polygraphy import cuda

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from preprocess import (  # noqa: E402
    build_hsi_patches,
    canonical_in_bounds_mask,
    load_hsi,
    map_to_canonical_pixel,
    standardize_hsi,
    valid_vegetation_mask,
)
from matching import VARIANTS, combine_score  # noqa: E402

# 原始数据根目录，默认 <repo>/data，可用环境变量 HSPC_DATA_ROOT 覆盖
DATA_ROOT = Path(os.environ.get("HSPC_DATA_ROOT", ROOT / "data"))
HSI_PATH = DATA_ROOT / "hsi_spatial_spectral_resampled_common_342/24data/hsi/10.6/1_spec342.dat"
LAS_PATH = DATA_ROOT / "lai_icp_registered_resampled_hsi/24data/10.6/rice_las/1_rice_icp.las"
ENGINE_DIR = ROOT / "engines"

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
}
SPATIAL_CONTRACT = {"source_crs": "EPSG:32651", "target_crs": "EPSG:4326"}
SAMPLES_PER_SCENE = 1000
SEED = 20260617
ENGINE_MAX_BATCH = 64  # engines/*.plan（非 _deploy）的 profile max，未改动

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
_TRT_TO_NP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT8: np.int8,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL: np.bool_,
}

_CUDART = cuda.wrapper().handle  # 复用 polygraphy 已加载的同一个 libcudart.so，避免多份 CUDA 上下文/ABI 不一致
_CUDART.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
_CUDART.cudaFreeHost.argtypes = [ctypes.c_void_p]


class PinnedArray:
    """阶段7 步骤2：用 cudaHostAlloc 分配的页锁定（pinned）host 内存，包成 numpy 数组视图。
    对 pageable 内存做 async H2D/D2H 时，CUDA 驱动内部会退化成同步拷贝（阶段6已用nsys证实），
    pinned 内存能让拷贝真正异步、和计算重叠。用完必须调用 free()，否则泄漏页锁定内存。
    """

    def __init__(self, shape: tuple, dtype):
        self.nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        ptr = ctypes.c_void_p()
        err = _CUDART.cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(self.nbytes), ctypes.c_uint(0))
        if err != 0:
            raise RuntimeError(f"cudaHostAlloc failed, code={err}")
        self._ptr = ptr
        buf = (ctypes.c_byte * self.nbytes).from_address(ptr.value)
        self.array = np.frombuffer(buf, dtype=dtype).reshape(shape)

    @property
    def ptr(self) -> int:
        return self._ptr.value

    def free(self):
        if self._ptr:
            _CUDART.cudaFreeHost(self._ptr)
            self._ptr = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class LiteTrtRunner:
    """不依赖 torch 的 TRT engine 推理器。显存/host buffer 按 max_batch 只分配一次并复用，
    H2D→execute→D2H 在同一个 stream 上异步下发，尾部统一 synchronize。

    对照组 trt_runner.TrtRunner 每次 infer() 都用 torch 重新分配 CUDA tensor，
    这是阶段4实测出的 2.0~2.2x runner 开销的主因。

    pinned=True（阶段7步骤2新增）：host buffer 改用页锁定内存（PinnedArray），
    让 H2D/D2H 真正异步。pageable（默认）是阶段5的原始行为，保留作对照。
    """

    def __init__(self, plan_path: Path, input_dims: tuple, output_dim: int = 1024,
                 max_batch: int = ENGINE_MAX_BATCH, input_name="input", output_name="output",
                 pinned: bool = False):
        self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(Path(plan_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {plan_path}")
        self.context = self.engine.create_execution_context()
        self.input_name, self.output_name = input_name, output_name
        self.max_batch = max_batch
        self.stream = cuda.Stream()
        self.pinned = pinned

        self.in_dtype = _TRT_TO_NP[self.engine.get_tensor_dtype(input_name)]
        self.out_dtype = _TRT_TO_NP[self.engine.get_tensor_dtype(output_name)]
        self.input_dims = input_dims
        self.output_dim = output_dim
        # TRT10 用 device_memory_size_v2（有 profile 相关的动态显存）；旧属性名做兜底
        self.device_memory_size = getattr(self.engine, "device_memory_size_v2", None)
        if self.device_memory_size is None:
            self.device_memory_size = self.engine.device_memory_size

        self.d_in = cuda.DeviceArray(shape=(max_batch, *input_dims), dtype=self.in_dtype)
        self.d_out = cuda.DeviceArray(shape=(max_batch, output_dim), dtype=self.out_dtype)
        if pinned:
            self._h_in_buf = PinnedArray((max_batch, *input_dims), self.in_dtype)
            self._h_out_buf = PinnedArray((max_batch, output_dim), self.out_dtype)
            self.h_in = self._h_in_buf.array
            self.h_out = self._h_out_buf.array
        else:
            self.h_in = np.empty((max_batch, *input_dims), dtype=self.in_dtype)
            self.h_out = np.empty((max_batch, output_dim), dtype=self.out_dtype)

    def close(self):
        if self.pinned:
            self._h_in_buf.free()
            self._h_out_buf.free()

    def infer(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        assert n <= self.max_batch, f"batch {n} > max_batch {self.max_batch}"
        self.h_in[:n] = x
        self.d_in.resize((n, *self.input_dims))
        self.d_out.resize((n, self.output_dim))
        self.context.set_input_shape(self.input_name, (n, *self.input_dims))
        self.d_in.copy_from(self.h_in[:n], self.stream)
        self.context.set_tensor_address(self.input_name, self.d_in.ptr)
        self.context.set_tensor_address(self.output_name, self.d_out.ptr)
        ok = self.context.execute_async_v3(self.stream.ptr)
        if not ok:
            raise RuntimeError("execute_async_v3 failed")
        self.d_out.copy_to(self.h_out[:n], self.stream)
        self.stream.synchronize()
        return self.h_out[:n].astype(np.float32, copy=True)

    def infer_all(self, x: np.ndarray) -> np.ndarray:
        """按 max_batch 分块跑完整个数组，buffer 全程复用。"""
        out = np.empty((len(x), self.output_dim), dtype=np.float32)
        for i in range(0, len(x), self.max_batch):
            out[i:i + self.max_batch] = self.infer(x[i:i + self.max_batch])
        return out


def numpy_score_window(point_features, feature_grid, valid_mask, ref_rows, ref_cols,
                        search_radius, spatial_weight):
    """e2e_stage_b_infer.py:score_window_batch 的 numpy 翻译版（逻辑逐条对应，只换掉 torch
    操作）。逐点循环，argmax 取第一个最大值（跟 hspc_encoder/inference.py:score_window 一致）。
    """
    rows, cols, dim = feature_grid.shape
    grid_n = feature_grid / (np.linalg.norm(feature_grid, axis=-1, keepdims=True) + 1e-12)
    pf_n = point_features / (np.linalg.norm(point_features, axis=-1, keepdims=True) + 1e-12)

    n = len(ref_rows)
    out_rows = np.full(n, -1, dtype=np.int64)
    out_cols = np.full(n, -1, dtype=np.int64)
    out_cos = np.full(n, np.nan, dtype=np.float32)
    out_disp = np.full(n, np.nan, dtype=np.float32)
    out_none = 0

    for i in range(n):
        ref_row, ref_col = int(ref_rows[i]), int(ref_cols[i])
        r0, r1 = max(0, ref_row - search_radius), min(rows, ref_row + search_radius + 1)
        c0, c1 = max(0, ref_col - search_radius), min(cols, ref_col + search_radius + 1)
        if r0 >= r1 or c0 >= c1:
            out_none += 1
            continue
        mask = valid_mask[r0:r1, c0:c1].reshape(-1)
        if not mask.any():
            out_none += 1
            continue
        window = grid_n[r0:r1, c0:c1].reshape(-1, dim)
        cosine = window @ pf_n[i]
        cosine = np.where(mask, cosine, -np.inf).astype(np.float32)
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        distance = np.sqrt((rr.astype(np.float32) - ref_row) ** 2 + (cc.astype(np.float32) - ref_col) ** 2).reshape(-1)
        final_score = combine_score(cosine, distance, spatial_weight, search_radius)
        final_score = np.where(mask, final_score, -np.inf).astype(np.float32)
        if not np.isfinite(final_score).any():
            out_none += 1
            continue
        best = int(np.argmax(final_score))
        width = c1 - c0
        mr = r0 + best // width
        mc = c0 + best % width
        out_rows[i], out_cols[i] = mr, mc
        out_cos[i] = float(cosine[best])
        out_disp[i] = float(np.hypot(mr - ref_row, mc - ref_col))
    return {"rows": out_rows, "cols": out_cols, "cosine": out_cos, "pixel_displacement": out_disp, "n_none": out_none}


def numpy_score_window_multi(point_features, feature_grid, valid_mask, ref_rows, ref_cols,
                              search_radius, spatial_weights: dict):
    """阶段7步骤4：`numpy_score_window` 的多variant合并版。多个 variant 共享同一个窗口的
    cosine/distance（它们跟 spatial_weight 无关，只有 final_score 的组合方式不同），只需要
    一次窗口 reshape + 一次矩阵乘法，而不是每个 variant 都重算一遍。同时把 `np.meshgrid` 换成
    行列各自的一维数组广播（结果逐位相同，省掉 meshgrid/broadcast_arrays 的调用开销）。

    经验证：在真实场景数据上跟"对每个 variant 分别调用 numpy_score_window"逐位相同
    （rows/cols/cosine/pixel_displacement 全部 array_equal），2 个 variant 合计耗时从
    191ms 降到 84ms（2.3x），main() 已改用这个版本；`numpy_score_window` 单variant版本
    保留供 scripts/stage6_ablation.py 等其他脚本引用。
    """
    rows, cols, dim = feature_grid.shape
    grid_n = feature_grid / (np.linalg.norm(feature_grid, axis=-1, keepdims=True) + 1e-12)
    pf_n = point_features / (np.linalg.norm(point_features, axis=-1, keepdims=True) + 1e-12)
    n = len(ref_rows)
    results = {name: {
        "rows": np.full(n, -1, dtype=np.int64), "cols": np.full(n, -1, dtype=np.int64),
        "cosine": np.full(n, np.nan, dtype=np.float32), "pixel_displacement": np.full(n, np.nan, dtype=np.float32),
        "n_none": 0,
    } for name in spatial_weights}

    for i in range(n):
        ref_row, ref_col = int(ref_rows[i]), int(ref_cols[i])
        r0, r1 = max(0, ref_row - search_radius), min(rows, ref_row + search_radius + 1)
        c0, c1 = max(0, ref_col - search_radius), min(cols, ref_col + search_radius + 1)
        if r0 >= r1 or c0 >= c1:
            for name in spatial_weights:
                results[name]["n_none"] += 1
            continue
        mask = valid_mask[r0:r1, c0:c1].reshape(-1)
        if not mask.any():
            for name in spatial_weights:
                results[name]["n_none"] += 1
            continue
        window = grid_n[r0:r1, c0:c1].reshape(-1, dim)
        cosine = window @ pf_n[i]
        cosine = np.where(mask, cosine, -np.inf).astype(np.float32)
        rr = (np.arange(r0, r1, dtype=np.float32) - ref_row)[:, None]
        cc = (np.arange(c0, c1, dtype=np.float32) - ref_col)[None, :]
        distance = np.sqrt(rr ** 2 + cc ** 2).reshape(-1)
        width = c1 - c0
        for name, sw in spatial_weights.items():
            final_score = combine_score(cosine, distance, sw, search_radius)
            final_score = np.where(mask, final_score, -np.inf).astype(np.float32)
            if not np.isfinite(final_score).any():
                results[name]["n_none"] += 1
                continue
            best = int(np.argmax(final_score))
            mr, mc = r0 + best // width, c0 + best % width
            results[name]["rows"][i], results[name]["cols"][i] = mr, mc
            results[name]["cosine"][i] = float(cosine[best])
            results[name]["pixel_displacement"][i] = float(np.hypot(mr - ref_row, mc - ref_col))
    return results


WARMUP = 2
REPEAT = 10


def timed(fn, warmup=WARMUP, repeat=REPEAT):
    """warmup 次 + 重复 repeat 次，返回 (最后一次的返回值, 中位数耗时秒, 各次耗时列表)。"""
    for _ in range(warmup):
        fn()
    elapsed = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        elapsed.append(time.perf_counter() - t0)
    return result, float(np.median(elapsed)), elapsed


def run_las_pipeline(las_path: Path, point_cloud_crs: str, target_proj_wkt: str, gt,
                      spatial_contract: dict, valid_mask: np.ndarray, rows: int, cols: int,
                      k_neighbors: int, samples_per_scene: int, seed: int, balanced_tree: bool = False):
    """LAS 全流程，细分计时（阶段7步骤4在 stage7_cpu_profile.py 里验证过跟
    preprocess.py:project_xyz_to_canonical_pixels + build_point_offsets 逐位相同后，
    移到这里作为正式实现）。

    crs_transformer_init 每次调用都重新构造 Transformer（不做跨调用缓存）——这跟生产环境
    "一个进程处理一景"的语义一致；如果以后要做多景批处理，跨景复用同一个 Transformer 对象能
    省掉这部分时间（阶段7步骤4实测：进程内构造一次后，同进程后续构造耗时趋近于0，说明主要
    开销是 PROJ 数据的首次加载，不是每次构造本身贵）。
    """
    from pyproj import CRS, Transformer
    from scipy.spatial import cKDTree
    import laspy

    t = {}
    t0 = time.perf_counter()
    las = laspy.read(las_path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    t["laspy_read_io"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    source = CRS.from_user_input(point_cloud_crs)
    target = CRS.from_user_input(target_proj_wkt)
    expected_source = CRS.from_user_input(spatial_contract["source_crs"])
    expected_target = CRS.from_user_input(spatial_contract["target_crs"])
    assert source.equals(expected_source) and target.equals(expected_target), "CRS 契约不匹配"
    transformer = Transformer.from_crs(source, target, always_xy=True)
    t["crs_transformer_init"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hsi_x, hsi_y = transformer.transform(xyz[:, 0], xyz[:, 1])
    ref_rows_full, ref_cols_full = map_to_canonical_pixel(gt, hsi_x, hsi_y)
    t["projection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    in_bounds = canonical_in_bounds_mask(ref_rows_full, ref_cols_full, rows, cols)
    ref_valid = np.zeros(len(xyz), dtype=bool)
    idx = np.where(in_bounds)[0]
    ref_valid[idx] = valid_mask[ref_rows_full[idx], ref_cols_full[idx]]
    eligible = np.arange(len(xyz), dtype=np.int64)[ref_valid]
    t["mask_filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    if eligible.size > samples_per_scene:
        eligible = np.sort(rng.choice(eligible, samples_per_scene, replace=False))
    t["rng_sample"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree = cKDTree(xyz, balanced_tree=balanced_tree)  # 阶段7优化A（默认False），见 preprocess.py:build_point_offsets
    t["ckdtree_build"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    query_k = min(k_neighbors + 1, len(xyz))
    _, neighbors = tree.query(xyz[eligible], k=query_k)
    neighbors = np.atleast_2d(neighbors).astype(np.int64)
    offsets = np.empty((len(eligible), k_neighbors, 3), dtype=np.float32)
    for j, point_index in enumerate(eligible):
        current = neighbors[j]
        current = current[current != point_index]
        if current.size == 0:
            current = np.asarray([point_index])
        if current.size < k_neighbors:
            current = np.pad(current, (0, k_neighbors - current.size), mode="edge")
        offsets[j] = (xyz[current[:k_neighbors]] - xyz[point_index]).astype(np.float32)
    t["ckdtree_query_offsets"] = time.perf_counter() - t0

    ref_rows, ref_cols = ref_rows_full[eligible], ref_cols_full[eligible]
    return offsets, eligible, ref_rows, ref_cols, t


SEGMENT_ORDER = [
    "load_hsi_io", "standardize_and_mask", "patch_build",
    "laspy_read_io", "crs_transformer_init", "projection", "mask_filter", "rng_sample",
    "ckdtree_build", "ckdtree_query_offsets",
    "hsi_infer", "pc_infer", "feature_grid_assemble", "match",
]
IO_SEGMENTS = {"load_hsi_io", "laspy_read_io"}


def run_scene_once(hsi_runner: LiteTrtRunner, pc_runner: LiteTrtRunner):
    """整景流程跑一次，返回 (输出, 各段耗时dict)。engine 由调用方传入（不计入这里的计时，
    加载在 main() 里单独计时一次）。"""
    t = {}

    t0 = time.perf_counter()
    raw, gt, proj = load_hsi(HSI_PATH, CONTRACT["target_bands"])
    t["load_hsi_io"] = time.perf_counter() - t0
    bands, rows, cols = raw.shape

    t0 = time.perf_counter()
    normalized, _, _ = standardize_hsi(raw)
    valid_mask = valid_vegetation_mask(raw, CONTRACT["red_band_index"], CONTRACT["nir_band_index"],
                                        CONTRACT["ndvi_threshold"], CONTRACT["nodata_epsilon"])
    vr, vc = np.where(valid_mask)
    valid_rowcols = list(zip(vr.tolist(), vc.tolist()))
    t["standardize_and_mask"] = time.perf_counter() - t0
    n_valid = len(valid_rowcols)

    t0 = time.perf_counter()
    valid_patches = build_hsi_patches(normalized, valid_rowcols, CONTRACT["patch_size"])
    t["patch_build"] = time.perf_counter() - t0

    offsets, eligible, ref_rows, ref_cols, las_t = run_las_pipeline(
        LAS_PATH, CONTRACT["point_cloud_crs"], proj, gt, SPATIAL_CONTRACT,
        valid_mask, rows, cols, CONTRACT["point_neighbors"], SAMPLES_PER_SCENE, SEED)
    t.update(las_t)

    t0 = time.perf_counter()
    hsi_feats = hsi_runner.infer_all(valid_patches)
    t["hsi_infer"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pc_feats = pc_runner.infer_all(offsets)
    t["pc_infer"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # 只对有效像元算了特征，拼回 (rows, cols, dim) 网格；无效像元保持 0
    # （score_window 会用 valid_mask 屏蔽掉，不会被访问到有效数值之外的用途）
    feature_grid = np.zeros((rows, cols, hsi_feats.shape[1]), dtype=np.float32)
    feature_grid[vr, vc] = hsi_feats
    t["feature_grid_assemble"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    match_result = numpy_score_window_multi(pc_feats, feature_grid, valid_mask, ref_rows, ref_cols,
                                             CONTRACT["search_radius"], VARIANTS)
    t["match"] = time.perf_counter() - t0

    outputs = {
        "n_grid_patches_full": int(rows * cols), "n_grid_patches_valid_only": int(n_valid),
        "n_points": int(len(eligible)), "hsi_feats": hsi_feats, "pc_feats": pc_feats,
        "match_result": match_result,
    }
    return outputs, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--engine-tier", choices=["scene", "batch64"], default="scene",
                     help="scene: engines/*_fp16_scene.plan（min1/opt2048/max2048，整景批量推理默认，"
                          "阶段7实测比batch64快2.1~2.5x）；batch64: engines/*_fp16.plan（低延迟单点/"
                          "小批场景）。int8精度目前只有batch64一档（int8无速度/精度收益，非推荐路径）。")
    args = ap.parse_args()
    mode = {"fp16": "fp16", "int8": "int8_implicit"}[args.precision]
    tier = args.engine_tier if args.precision == "fp16" else "batch64"
    suffix = "_scene" if tier == "scene" else ""
    engine_max_batch = 2048 if tier == "scene" else ENGINE_MAX_BATCH

    t0 = time.perf_counter()
    hsi_runner = LiteTrtRunner(ENGINE_DIR / f"hsi_{mode}{suffix}.plan", input_dims=(342, 3, 3),
                               max_batch=engine_max_batch)
    pc_runner = LiteTrtRunner(ENGINE_DIR / f"pc_{mode}{suffix}.plan", input_dims=(15, 3),
                              max_batch=engine_max_batch)
    engine_load_sec = time.perf_counter() - t0

    # 冷启动：进程内第一次跑整景（含 PROJ 首次初始化）。页缓存状态不受控制（未清缓存），
    # 注意：这不是"干净"的冷启动，只是"这个新进程第一次跑"。
    cold_outputs, cold_timing = run_scene_once(hsi_runner, pc_runner)

    # warmup + repeat（热态）
    for _ in range(WARMUP):
        run_scene_once(hsi_runner, pc_runner)
    warm_runs = [run_scene_once(hsi_runner, pc_runner) for _ in range(REPEAT)]
    warm_timings = [t for _, t in warm_runs]

    warm_stats = {}
    for seg in SEGMENT_ORDER:
        vals = [t[seg] for t in warm_timings]
        warm_stats[seg] = {"median_sec": float(np.median(vals)), "min_sec": float(np.min(vals)), "all_sec": vals}

    total_with_io = sum(warm_stats[s]["median_sec"] for s in SEGMENT_ORDER)
    total_without_io = sum(warm_stats[s]["median_sec"] for s in SEGMENT_ORDER if s not in IO_SEGMENTS)

    # 确定性：热态 REPEAT 次里，特征与匹配结果逐位相同
    last_outputs = warm_runs[-1][0]
    determinism = {
        "hsi_feats_identical_across_reps": bool(all(
            np.array_equal(r[0]["hsi_feats"], last_outputs["hsi_feats"]) for r in warm_runs)),
        "pc_feats_identical_across_reps": bool(all(
            np.array_equal(r[0]["pc_feats"], last_outputs["pc_feats"]) for r in warm_runs)),
        "match_rowcols_identical_across_reps": {
            vname: bool(all(
                np.array_equal(r[0]["match_result"][vname]["rows"], last_outputs["match_result"][vname]["rows"])
                and np.array_equal(r[0]["match_result"][vname]["cols"], last_outputs["match_result"][vname]["cols"])
                for r in warm_runs))
            for vname in VARIANTS
        },
    }

    fp32_match_path = ROOT / "results/stage4_fp32_match.npz"
    fp32_ref = np.load(fp32_match_path) if fp32_match_path.exists() else None
    variants_report = {}
    for vname in VARIANTS:
        m = last_outputs["match_result"][vname]
        entry = {"n_none": int(m["n_none"]), "pixel_displacement_mean": float(np.nanmean(m["pixel_displacement"]))}
        if fp32_ref is not None:
            key = vname.replace("+", "_")
            bm_rows, bm_cols = fp32_ref[f"{key}_rows"], fp32_ref[f"{key}_cols"]
            entry["pixel_match_agreement_vs_pytorch_fp32"] = float(
                np.mean((m["rows"] == bm_rows) & (m["cols"] == bm_cols)))
        variants_report[vname] = entry

    report = {
        "scene": "24data/10.6/1", "precision": args.precision, "engine_tier": tier,
        "n_grid_patches_full": last_outputs["n_grid_patches_full"],
        "n_grid_patches_valid_only": last_outputs["n_grid_patches_valid_only"],
        "n_points": last_outputs["n_points"],
        "engine_load_sec": engine_load_sec,
        "cold_start": {
            "note": "进程内第一次调用 run_scene_once（含PROJ首次初始化）；操作系统页缓存状态未清空、不受控制，"
                    "不是严格意义的\"干净冷启动\"，只是本进程的第一次运行。",
            "timing_sec": cold_timing,
            "total_with_io_sec": sum(cold_timing.values()),
            "total_without_io_sec": sum(v for k, v in cold_timing.items() if k not in IO_SEGMENTS),
        },
        "warm": {
            "note": f"warmup={WARMUP} + repeat={REPEAT}，同一进程，报告每段 median/min。",
            "segments": warm_stats,
            "total_with_io_median_sec": total_with_io,
            "total_without_io_median_sec": total_without_io,
        },
        "determinism": determinism,
        "variants": variants_report,
    }

    out = ROOT / "results/stage5_deploy_opt.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "warm"} | {
        "warm": {"note": report["warm"]["note"],
                 "total_with_io_median_sec": total_with_io, "total_without_io_median_sec": total_without_io,
                 "segments_median_ms": {s: round(warm_stats[s]["median_sec"] * 1000, 2) for s in SEGMENT_ORDER}}
    }, ensure_ascii=False, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
