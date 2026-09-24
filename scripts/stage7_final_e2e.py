"""阶段7 步骤5②：阶段4原始链路 vs 最终版，同一进程、同一口径、累加式 S0→S7 对比。

运行环境：conda modelopt + PYTHONPATH=/usr/lib/python3.10/dist-packages，另外 pip 装了
laspy + pyproj（不装GDAL）。HSI 读取用 np.fromfile（`verify_preprocess.py:read_fromfile`，
已验证与GDAL逐位相同），gt/投影WKT 从 `results/e2e_scene_geo.json`（hspc-preprocess环境
用GDAL导出一次）读取。

累加链（每步只改一个变量，每个配置都跑完整整景的全部分段）：
  S0 阶段4原始：legacy engine(*_fp16_legacy.plan) + TrtRunner(batch64) + 全图15840个patch
     + torch score_window_batch（每个variant单独调用）+ cKDTree默认(balanced_tree=True)
  S1 engine 换成 mixed 版 batch64 (engines/*_fp16.plan)
  S2 只算有效像元 (9645个patch)
  S3 runner 换成 LiteTrtRunner（去掉torch依赖）
  S4 engine 换成 scene engine (*_fp16_scene.plan, max_batch=2048)
  S5 匹配换成 numpy_score_window（每个variant单独调用）
  S6 匹配换成 numpy_score_window_multi（合并variant）
  S7 cKDTree 换成 balanced_tree=False —— 即最终版，等于 deploy_scene.py 当前默认行为

护栏：S7 的 eligible/offsets/ref_rows/ref_cols/patches 必须跟 hspc-preprocess 环境产出的
results/e2e_scene_preprocessed.npz 逐位相同，不一致则中止（pip版pyproj的PROJ数据库
可能跟conda版不同）。

口径：每配置 warmup2 + repeat10；正向 S0→S7 一轮、反向 S7→S0 一轮，同一进程。
不改模型结构/权重/超参数；不重建任何 engine。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hspc_common as hc  # noqa: E402
from preprocess import build_hsi_patches, standardize_hsi, valid_vegetation_mask  # noqa: E402
from verify_preprocess import read_fromfile  # noqa: E402
from trt_runner import TrtRunner  # noqa: E402
from e2e_stage_b_infer import score_window_batch  # noqa: E402
from deploy_scene import (  # noqa: E402
    CONTRACT, HSI_PATH, LAS_PATH, SAMPLES_PER_SCENE, SEED, SPATIAL_CONTRACT, VARIANTS,
    ENGINE_DIR, LiteTrtRunner, numpy_score_window, numpy_score_window_multi, run_las_pipeline,
)

ROOT = hc.ROOT
WARMUP, REPEAT = 2, 10

CONFIGS = [
    {"name": "S0_stage4_original", "patches": "full", "engine": "legacy", "runner": "TrtRunner",
     "max_batch": 64, "match": "torch_separate", "balanced_tree": True},
    {"name": "S1_mixed_precision_batch64", "patches": "full", "engine": "mixed_batch64", "runner": "TrtRunner",
     "max_batch": 64, "match": "torch_separate", "balanced_tree": True},
    {"name": "S2_valid_pixels_only", "patches": "valid", "engine": "mixed_batch64", "runner": "TrtRunner",
     "max_batch": 64, "match": "torch_separate", "balanced_tree": True},
    {"name": "S3_lite_runner_no_torch", "patches": "valid", "engine": "mixed_batch64", "runner": "LiteTrtRunner",
     "max_batch": 64, "match": "torch_separate", "balanced_tree": True},
    {"name": "S4_scene_engine", "patches": "valid", "engine": "mixed_scene", "runner": "LiteTrtRunner",
     "max_batch": 2048, "match": "torch_separate", "balanced_tree": True},
    {"name": "S5_numpy_match_separate", "patches": "valid", "engine": "mixed_scene", "runner": "LiteTrtRunner",
     "max_batch": 2048, "match": "numpy_separate", "balanced_tree": True},
    {"name": "S6_numpy_match_merged", "patches": "valid", "engine": "mixed_scene", "runner": "LiteTrtRunner",
     "max_batch": 2048, "match": "numpy_merged", "balanced_tree": True},
    {"name": "S7_final_ckdtree_unbalanced", "patches": "valid", "engine": "mixed_scene", "runner": "LiteTrtRunner",
     "max_batch": 2048, "match": "numpy_merged", "balanced_tree": False},
]

ENGINE_PATHS = {
    "legacy": (ENGINE_DIR / "hsi_fp16_legacy.plan", ENGINE_DIR / "pc_fp16_legacy.plan"),
    "mixed_batch64": (ENGINE_DIR / "hsi_fp16.plan", ENGINE_DIR / "pc_fp16.plan"),
    "mixed_scene": (ENGINE_DIR / "hsi_fp16_scene.plan", ENGINE_DIR / "pc_fp16_scene.plan"),
}

SEGMENT_ORDER = [
    "load_hsi_io", "standardize_and_mask", "patch_build",
    "laspy_read_io", "crs_transformer_init", "projection", "mask_filter", "rng_sample",
    "ckdtree_build", "ckdtree_query_offsets",
    "hsi_infer", "pc_infer", "feature_grid_assemble", "match",
]
IO_SEGMENTS = {"load_hsi_io", "laspy_read_io"}


def load_engines(cfg):
    hsi_path, pc_path = ENGINE_PATHS[cfg["engine"]]
    if cfg["runner"] == "TrtRunner":
        return TrtRunner(hsi_path), TrtRunner(pc_path)
    return (LiteTrtRunner(hsi_path, input_dims=(342, 3, 3), max_batch=cfg["max_batch"]),
            LiteTrtRunner(pc_path, input_dims=(15, 3), max_batch=cfg["max_batch"]))


def run_config_once(cfg, hsi_runner, pc_runner, gt, proj_wkt):
    t = {}
    t0 = time.perf_counter()
    raw = read_fromfile(HSI_PATH)
    t["load_hsi_io"] = time.perf_counter() - t0
    bands, rows, cols = raw.shape

    t0 = time.perf_counter()
    normalized, _, _ = standardize_hsi(raw)
    valid_mask = valid_vegetation_mask(raw, CONTRACT["red_band_index"], CONTRACT["nir_band_index"],
                                        CONTRACT["ndvi_threshold"], CONTRACT["nodata_epsilon"])
    vr, vc = np.where(valid_mask)
    t["standardize_and_mask"] = time.perf_counter() - t0

    if cfg["patches"] == "full":
        rowcols = [(r, c) for r in range(rows) for c in range(cols)]
    else:
        rowcols = list(zip(vr.tolist(), vc.tolist()))
    t0 = time.perf_counter()
    patches = build_hsi_patches(normalized, rowcols, CONTRACT["patch_size"])
    t["patch_build"] = time.perf_counter() - t0

    offsets, eligible, ref_rows, ref_cols, las_t = run_las_pipeline(
        LAS_PATH, CONTRACT["point_cloud_crs"], proj_wkt, gt, SPATIAL_CONTRACT,
        valid_mask, rows, cols, CONTRACT["point_neighbors"], SAMPLES_PER_SCENE, SEED,
        balanced_tree=cfg["balanced_tree"])
    t.update(las_t)

    if cfg["runner"] == "TrtRunner":
        t0 = time.perf_counter()
        hsi_feats = hsi_runner.infer_batched(patches, max_batch=cfg["max_batch"])
        t["hsi_infer"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pc_feats = pc_runner.infer_batched(offsets, max_batch=cfg["max_batch"])
        t["pc_infer"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        hsi_feats = hsi_runner.infer_all(patches)
        t["hsi_infer"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pc_feats = pc_runner.infer_all(offsets)
        t["pc_infer"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cfg["patches"] == "full":
        feature_grid = hsi_feats.reshape(rows, cols, -1).astype(np.float32)
    else:
        feature_grid = np.zeros((rows, cols, hsi_feats.shape[1]), dtype=np.float32)
        feature_grid[vr, vc] = hsi_feats
    t["feature_grid_assemble"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if cfg["match"] == "torch_separate":
        feature_grid_t = torch.from_numpy(feature_grid).float()
        pf_t = torch.from_numpy(pc_feats).float()
        match_result = {v: score_window_batch(pf_t, feature_grid_t, valid_mask, ref_rows, ref_cols,
                                               CONTRACT["search_radius"], sw) for v, sw in VARIANTS.items()}
    elif cfg["match"] == "numpy_separate":
        match_result = {v: numpy_score_window(pc_feats, feature_grid, valid_mask, ref_rows, ref_cols,
                                               CONTRACT["search_radius"], sw) for v, sw in VARIANTS.items()}
    else:
        match_result = numpy_score_window_multi(pc_feats, feature_grid, valid_mask, ref_rows, ref_cols,
                                                 CONTRACT["search_radius"], VARIANTS)
    t["match"] = time.perf_counter() - t0

    outputs = {
        "hsi_feats": hsi_feats, "pc_feats": pc_feats, "match_result": match_result,
        "eligible": eligible, "offsets": offsets, "ref_rows": ref_rows, "ref_cols": ref_cols,
        "patches": patches, "n_valid_rowcols": len(rowcols),
    }
    return outputs, t


def guardrail_check(cfg, gt, proj_wkt):
    """S7 应跟 hspc-preprocess 环境(GDAL+conda pyproj)产出的缓存 npz 逐位相同。"""
    npz_path = ROOT / "results/e2e_scene_preprocessed.npz"
    if not npz_path.exists():
        return {"skipped": True, "reason": "e2e_scene_preprocessed.npz 不存在"}
    ref = np.load(npz_path)
    hsi_runner, pc_runner = load_engines(cfg)
    out, _ = run_config_once(cfg, hsi_runner, pc_runner, gt, proj_wkt)
    result = {
        "eligible_identical": bool(np.array_equal(out["eligible"], ref["eligible_point_indices"])),
        "offsets_identical": bool(np.array_equal(out["offsets"], ref["point_offsets"])),
        "ref_rows_identical": bool(np.array_equal(out["ref_rows"], ref["ref_rows"])),
        "ref_cols_identical": bool(np.array_equal(out["ref_cols"], ref["ref_cols"])),
    }
    # patches: ref npz 存的是全图 hsi_grid_patches，S7 是 valid-only，按 eligible 对应位置比较不现实
    # （ref 是按 (row,col) 索引的全图顺序，S7 是 valid_rowcols 顺序）——改为用 vr,vc 顺序重新核对
    valid_mask_ref = ref["valid_mask"]
    vr, vc = np.where(valid_mask_ref)
    rows_ref, cols_ref = int(ref["hsi_rows"]), int(ref["hsi_cols"])
    ref_patches_valid = ref["hsi_grid_patches"][vr * cols_ref + vc]
    result["patches_identical"] = bool(np.array_equal(out["patches"], ref_patches_valid))
    return result


def timed_config(cfg, gt, proj_wkt):
    hsi_runner, pc_runner = load_engines(cfg)
    for _ in range(WARMUP):
        run_config_once(cfg, hsi_runner, pc_runner, gt, proj_wkt)
    runs = [run_config_once(cfg, hsi_runner, pc_runner, gt, proj_wkt) for _ in range(REPEAT)]
    timings = [t for _, t in runs]
    stats = {}
    for seg in SEGMENT_ORDER:
        vals = [t[seg] for t in timings]
        stats[seg] = {"median_ms": float(np.median(vals) * 1000), "min_ms": float(np.min(vals) * 1000)}
    total_with_io = sum(stats[s]["median_ms"] for s in SEGMENT_ORDER)
    total_without_io = sum(stats[s]["median_ms"] for s in SEGMENT_ORDER if s not in IO_SEGMENTS)
    last = runs[-1][0]
    fp32_ref = np.load(ROOT / "results/stage4_fp32_match.npz")
    agreement = {}
    for vname in VARIANTS:
        key = vname.replace("+", "_")
        m = last["match_result"][vname]
        agreement[vname] = float(np.mean((m["rows"] == fp32_ref[f"{key}_rows"]) & (m["cols"] == fp32_ref[f"{key}_cols"])))
    return {
        "segments_ms": stats, "total_with_io_median_ms": total_with_io,
        "total_without_io_median_ms": total_without_io,
        "agreement_vs_pytorch_fp32": agreement,
    }


def main():
    geo = json.loads((ROOT / "results/e2e_scene_geo.json").read_text())
    gt, proj_wkt = geo["gt"], geo["proj_wkt"]

    print("=== 护栏：S7 vs hspc-preprocess(GDAL+conda pyproj) 缓存 ===")
    guard = guardrail_check(CONFIGS[-1], gt, proj_wkt)
    print(json.dumps(guard, indent=2))
    if not guard.get("skipped") and not all(guard.values()):
        print("!! 护栏未通过，停止")
        (ROOT / "results/stage7_final_e2e.json").write_text(
            json.dumps({"guardrail": guard, "status": "FAILED_STOPPED"}, ensure_ascii=False, indent=2))
        return

    print("\n=== 冷启动：进程内每个配置第一次运行（S0, S7）===")
    cold = {}
    for cfg in [CONFIGS[0], CONFIGS[-1]]:
        hsi_runner, pc_runner = load_engines(cfg)
        _, t = run_config_once(cfg, hsi_runner, pc_runner, gt, proj_wkt)
        cold[cfg["name"]] = {
            "timing_ms": {k: round(v * 1000, 2) for k, v in t.items()},
            "total_with_io_ms": sum(t.values()) * 1000,
            "total_without_io_ms": sum(v for k, v in t.items() if k not in IO_SEGMENTS) * 1000,
        }
        print(f"  {cfg['name']}: total_with_io={cold[cfg['name']]['total_with_io_ms']:.1f}ms "
              f"total_without_io={cold[cfg['name']]['total_without_io_ms']:.1f}ms")

    print("\n=== 正向 S0->S7 ===")
    fwd = {}
    for cfg in CONFIGS:
        r = timed_config(cfg, gt, proj_wkt)
        fwd[cfg["name"]] = r
        print(f"  {cfg['name']}: with_io={r['total_with_io_median_ms']:.1f}ms "
              f"without_io={r['total_without_io_median_ms']:.1f}ms "
              f"agree={r['agreement_vs_pytorch_fp32']}")

    print("\n=== 反向 S7->S0 ===")
    rev = {}
    for cfg in reversed(CONFIGS):
        r = timed_config(cfg, gt, proj_wkt)
        rev[cfg["name"]] = r
        print(f"  {cfg['name']}: with_io={r['total_with_io_median_ms']:.1f}ms "
              f"without_io={r['total_without_io_median_ms']:.1f}ms")

    direction_consistent = all(
        abs(fwd[c["name"]]["total_without_io_median_ms"] - rev[c["name"]]["total_without_io_median_ms"])
        / fwd[c["name"]]["total_without_io_median_ms"] < 0.5
        for c in CONFIGS
    )

    # 各步贡献（相邻两步差值 / S0->S7 总节省，均用不含IO的口径；正向数据）
    names = [c["name"] for c in CONFIGS]
    contrib = {}
    total_saved = fwd[names[0]]["total_without_io_median_ms"] - fwd[names[-1]]["total_without_io_median_ms"]
    for i in range(1, len(names)):
        delta = fwd[names[i - 1]]["total_without_io_median_ms"] - fwd[names[i]]["total_without_io_median_ms"]
        contrib[f"{names[i-1]}->{names[i]}"] = {
            "delta_ms": delta,
            "pct_of_total_saved": (delta / total_saved * 100) if total_saved else None,
        }

    report = {
        "guardrail": guard,
        "cold_start": cold,
        "forward": fwd, "reverse": rev,
        "direction_consistent": direction_consistent,
        "overall_speedup_without_io": fwd[names[0]]["total_without_io_median_ms"] / fwd[names[-1]]["total_without_io_median_ms"],
        "overall_speedup_with_io": fwd[names[0]]["total_with_io_median_ms"] / fwd[names[-1]]["total_with_io_median_ms"],
        "contribution_breakdown_forward_without_io": contrib,
        "contribution_note": "贡献拆分依赖累加顺序，各步之间可能存在交互效应（比如只算有效像元在"
                              "batch64下省得更多，因为batch64调用次数更敏感于patch数量）；S0用legacy"
                              "engine，一致率跟S1-S7不是同一个基准（legacy是阶段2/3的原始fp16engine，"
                              "不是mixed精度），仅供参考不代表S0本身有精度问题",
        "status": "OK",
    }
    (ROOT / "results/stage7_final_e2e.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n整景加速比(不含IO): {report['overall_speedup_without_io']:.2f}x")
    print(f"整景加速比(含IO): {report['overall_speedup_with_io']:.2f}x")
    print(f"方向一致: {direction_consistent}")
    print("written results/stage7_final_e2e.json")


if __name__ == "__main__":
    main()
