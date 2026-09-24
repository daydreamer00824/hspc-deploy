"""阶段7 步骤4：CPU 段 profiling（LAS 处理 + 匹配 + patch 构建），热态分段计时 + cProfile。

背景：用了 scene engine 之后，HSI/PC 推理已经压到 ~54ms/~5ms，整景瓶颈转移到 CPU 段
（估算：匹配~174ms、LAS处理~102-231ms不等——单次测量噪声大、patch构建~60-80ms）。
本脚本用 warmup2+repeat10 的热态测量替代这些单次估算，并对最大的几段做 cProfile 定位热点。

硬约束：任何优化都要跟现有实现（scripts/preprocess.py、scripts/deploy_scene.py 里的
numpy_score_window）逐位比对，不合格不采用。只用 numpy/scipy 等现有依赖，不引入 numba/
Cython/C++扩展——如果分析出某项优化需要新依赖，只报告方案和预估收益，不实现（Jetson上
加依赖有成本，需另行评估）。

运行环境：conda hspc-preprocess + PYTHONPATH=/usr/lib/python3.10/dist-packages
（laspy/pyproj/gdal + tensorrt/polygraphy 都要用，deploy_scene.py 同款环境）。
"""
from __future__ import annotations

import cProfile
import json
import pstats
import io as pyio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import (  # noqa: E402
    build_hsi_patches, build_point_offsets, canonical_in_bounds_mask, load_hsi,
    project_xyz_to_canonical_pixels, standardize_hsi, valid_vegetation_mask,
)
from deploy_scene import (  # noqa: E402
    CONTRACT, HSI_PATH, LAS_PATH, SAMPLES_PER_SCENE, SEED, SPATIAL_CONTRACT, VARIANTS,
    ENGINE_DIR, LiteTrtRunner, numpy_score_window, numpy_score_window_multi,
    run_las_pipeline as _run_las_pipeline_prod,
)

ROOT = Path(__file__).resolve().parents[1]
REPEAT = 10
WARMUP = 2


def median_min(vals):
    return {"median_ms": float(np.median(vals) * 1000), "min_ms": float(np.min(vals) * 1000)}


def timed_las_pipeline(raw, gt, proj, valid_mask, rows, cols, transformer_cache: dict):
    """薄封装：调用 deploy_scene.run_las_pipeline（阶段7步骤5起的正式实现，字段名与这里
    历史上用的略有出入，只做改名，不重复实现）。transformer_cache 参数保留但不再使用——
    正式实现里 crs_transformer_init 每次都重新构造，跟生产单景语义一致（阶段7步骤5决定）。
    """
    output, eligible, ref_rows, ref_cols, t_prod = _run_las_pipeline_prod(
        LAS_PATH, CONTRACT["point_cloud_crs"], proj, gt, SPATIAL_CONTRACT,
        valid_mask, rows, cols, CONTRACT["point_neighbors"], SAMPLES_PER_SCENE, SEED)
    t = {
        "laspy_read_io": t_prod["laspy_read_io"],
        "crs_transformer_init_this_call": t_prod["crs_transformer_init"],
        "coordinate_projection": t_prod["projection"],
        "in_bounds_and_veg_mask_filter": t_prod["mask_filter"],
        "rng_sample": t_prod["rng_sample"],
        "ckdtree_build": t_prod["ckdtree_build"],
        "ckdtree_query": t_prod["ckdtree_query_offsets"],  # 正式实现把query+offsets循环合并计时
        "offsets_python_loop": 0.0,
    }
    return output, eligible, ref_rows, ref_cols, t


def verify_las_pipeline_matches_reference(raw, gt, proj, valid_mask, rows, cols):
    """确认上面拆开计时的版本跟 preprocess.py 原函数逐位相同（不是重新实现了一套逻辑）。"""
    offsets, eligible, ref_rows, ref_cols, _ = timed_las_pipeline(raw, gt, proj, valid_mask, rows, cols, {})

    import laspy
    las = laspy.read(LAS_PATH)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    ref_rows_full, ref_cols_full = project_xyz_to_canonical_pixels(
        xyz[:, 0], xyz[:, 1], CONTRACT["point_cloud_crs"], proj, gt, SPATIAL_CONTRACT)
    in_bounds = canonical_in_bounds_mask(ref_rows_full, ref_cols_full, rows, cols)
    ref_valid = np.zeros(len(xyz), dtype=bool)
    idx = np.where(in_bounds)[0]
    ref_valid[idx] = valid_mask[ref_rows_full[idx], ref_cols_full[idx]]
    eligible_ref = np.arange(len(xyz), dtype=np.int64)[ref_valid]
    rng = np.random.default_rng(SEED)
    if eligible_ref.size > SAMPLES_PER_SCENE:
        eligible_ref = np.sort(rng.choice(eligible_ref, SAMPLES_PER_SCENE, replace=False))
    offsets_ref = build_point_offsets(xyz, eligible_ref, CONTRACT["point_neighbors"])

    return {
        "eligible_identical": bool(np.array_equal(eligible, eligible_ref)),
        "offsets_identical": bool(np.array_equal(offsets, offsets_ref)),
        "ref_rows_identical": bool(np.array_equal(ref_rows, ref_rows_full[eligible_ref])),
        "ref_cols_identical": bool(np.array_equal(ref_cols, ref_cols_full[eligible_ref])),
    }


def profile_section(fn, label):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    s = pyio.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    return s.getvalue()


def main():
    raw, gt, proj = load_hsi(HSI_PATH, CONTRACT["target_bands"])
    bands, rows, cols = raw.shape
    normalized, _, _ = standardize_hsi(raw)
    valid_mask = valid_vegetation_mask(raw, CONTRACT["red_band_index"], CONTRACT["nir_band_index"],
                                        CONTRACT["ndvi_threshold"], CONTRACT["nodata_epsilon"])
    vr, vc = np.where(valid_mask)
    valid_rowcols = list(zip(vr.tolist(), vc.tolist()))

    print("=== 正确性核对：拆分计时版 LAS 流程 vs preprocess.py 原函数 ===")
    correctness = verify_las_pipeline_matches_reference(raw, gt, proj, valid_mask, rows, cols)
    print(json.dumps(correctness, indent=2))
    assert all(correctness.values()), "拆分计时版本跟原实现不一致，停止"

    # ---------------- 分段热态计时（含 Transformer 复用） ----------------
    print("\n=== LAS pipeline 分段计时（warmup2+repeat10，Transformer 跨rep复用）===")
    transformer_cache = {}
    all_t = []
    for i in range(WARMUP + REPEAT):
        offsets, eligible, ref_rows, ref_cols, t = timed_las_pipeline(
            raw, gt, proj, valid_mask, rows, cols, transformer_cache)
        if i >= WARMUP:
            all_t.append(t)
    las_stats = {k: median_min([t[k] for t in all_t]) for k in all_t[0]}
    for k, v in las_stats.items():
        print(f"  {k}: median={v['median_ms']:.2f}ms min={v['min_ms']:.2f}ms")
    las_total_median = sum(v["median_ms"] for v in las_stats.values())
    print(f"  LAS各段之和(median): {las_total_median:.2f}ms")

    # patch 构建（有效像元）
    print("\n=== patch 构建计时 ===")
    _ = build_hsi_patches(normalized, valid_rowcols, CONTRACT["patch_size"])  # warmup
    patch_times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        valid_patches = build_hsi_patches(normalized, valid_rowcols, CONTRACT["patch_size"])
        patch_times.append(time.perf_counter() - t0)
    patch_stats = median_min(patch_times)
    print(f"  build_hsi_patches: median={patch_stats['median_ms']:.2f}ms min={patch_stats['min_ms']:.2f}ms")

    # GPU 推理（scene engine，仅记录，非本次优化对象）
    hsi_runner = LiteTrtRunner(ENGINE_DIR / "hsi_fp16_scene.plan", input_dims=(342, 3, 3), max_batch=2048)
    pc_runner = LiteTrtRunner(ENGINE_DIR / "pc_fp16_scene.plan", input_dims=(15, 3), max_batch=2048)
    hsi_runner.infer_all(valid_patches)
    pc_runner.infer_all(offsets)
    hsi_feats = hsi_runner.infer_all(valid_patches)
    pc_feats = pc_runner.infer_all(offsets)
    feature_grid = np.zeros((rows, cols, hsi_feats.shape[1]), dtype=np.float32)
    feature_grid[vr, vc] = hsi_feats

    # ---------------- 优化前后对比（A/B，同一进程，各自warmup+repeat）----------------
    def _time_once(fn):
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    import laspy
    las = laspy.read(LAS_PATH)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

    print("\n=== 优化A: cKDTree balanced_tree=True(旧) vs False(新) ===")

    def old_offsets():
        return build_point_offsets(xyz, eligible, CONTRACT["point_neighbors"], balanced_tree=True)

    def new_offsets():
        return build_point_offsets(xyz, eligible, CONTRACT["point_neighbors"], balanced_tree=False)

    old_offsets(); new_offsets()
    old_o, new_o = old_offsets(), new_offsets()
    print(f"  逐位相同: {bool(np.array_equal(old_o, new_o))}")
    old_bt = median_min([_time_once(old_offsets) for _ in range(REPEAT)])
    new_bt = median_min([_time_once(new_offsets) for _ in range(REPEAT)])
    print(f"  balanced=True: median={old_bt['median_ms']:.2f}ms  balanced=False: median={new_bt['median_ms']:.2f}ms")
    optimization_a = {"identical": bool(np.array_equal(old_o, new_o)), "before_ms": old_bt, "after_ms": new_bt}

    print("\n=== 优化B: numpy_score_window(逐variant调用) vs numpy_score_window_multi(合并) ===")

    def old_match():
        return {v: numpy_score_window(pc_feats, feature_grid, valid_mask, ref_rows, ref_cols,
                                       CONTRACT["search_radius"], sw) for v, sw in VARIANTS.items()}

    def new_match():
        return numpy_score_window_multi(pc_feats, feature_grid, valid_mask, ref_rows, ref_cols,
                                         CONTRACT["search_radius"], VARIANTS)

    old_match(); new_match()
    old_r, new_r = old_match(), new_match()
    identical = all(
        np.array_equal(old_r[v]["rows"], new_r[v]["rows"]) and np.array_equal(old_r[v]["cols"], new_r[v]["cols"])
        and np.array_equal(old_r[v]["cosine"], new_r[v]["cosine"], equal_nan=True)
        for v in VARIANTS
    )
    print(f"  逐位相同(rows/cols/cosine): {identical}")
    old_match_t = median_min([_time_once(old_match) for _ in range(REPEAT)])
    new_match_t = median_min([_time_once(new_match) for _ in range(REPEAT)])
    print(f"  旧(2次分调用): median={old_match_t['median_ms']:.2f}ms  新(合并): median={new_match_t['median_ms']:.2f}ms")
    optimization_b = {"identical": identical, "before_ms": old_match_t, "after_ms": new_match_t}

    match_total_median = new_match_t["median_ms"]

    # ---------------- 对账 ----------------
    core_sum = las_total_median + patch_stats["median_ms"] + match_total_median
    print(f"\n=== 对账：LAS({las_total_median:.1f}) + patch({patch_stats['median_ms']:.1f}) + "
          f"match({match_total_median:.1f}) = {core_sum:.1f}ms（不含GPU推理/IO/engine加载）")

    # ---------------- cProfile：最大的几段（用优化后的实现）----------------
    print("\n=== cProfile: 匹配（最大段，合并版）===")
    match_profile = profile_section(
        lambda: numpy_score_window_multi(pc_feats, feature_grid, valid_mask, ref_rows, ref_cols,
                                          CONTRACT["search_radius"], VARIANTS),
        "match")
    print(match_profile[:3000])

    print("\n=== cProfile: LAS pipeline（offsets循环，第二大段） ===")
    las_profile = profile_section(
        lambda: timed_las_pipeline(raw, gt, proj, valid_mask, rows, cols, transformer_cache), "las")
    print(las_profile[:3000])

    print("\n=== cProfile: patch 构建（第三大段） ===")
    patch_profile = profile_section(
        lambda: build_hsi_patches(normalized, valid_rowcols, CONTRACT["patch_size"]), "patch")
    print(patch_profile[:3000])

    report = {
        "correctness_vs_reference": correctness,
        "las_pipeline_sections_ms": las_stats,
        "las_pipeline_total_median_ms": las_total_median,
        "patch_build_ms": patch_stats,
        "match_total_median_ms": match_total_median,
        "core_sum_ms": core_sum,
        "optimization_a_ckdtree_balanced_tree": optimization_a,
        "optimization_b_merged_score_window": optimization_b,
        "cprofile_match": match_profile,
        "cprofile_las_pipeline": las_profile,
        "cprofile_patch_build": patch_profile,
    }
    (ROOT / "results/stage7_cpu_profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nwritten results/stage7_cpu_profile.json")


if __name__ == "__main__":
    main()
