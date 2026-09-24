"""阶段6 消融重做：拆解阶段5 中 1.55x 的来源，同一进程/同一 warmup&repeat 口径。

配置（累加式）：
  A 阶段4原始：patch构建=全图15840，HSI/PC推理=TrtRunner，匹配=torch score_window_batch
  B +只算有效像元：patch构建=9645，其余同A
  C +匹配改numpy：同B，匹配=numpy_score_window（deploy_scene.py）
  D +LiteTrtRunner（=最终版）：HSI/PC推理=LiteTrtRunner，匹配=numpy（同C）

只用仓库现有的 engines/hsi_fp16.plan、engines/pc_fp16.plan（不用本阶段任何重建的
engine，避免构建差异混入性能对比）。standardize+mask 四个配置共用，只测一次。
每个分段 warmup 2 + repeat 10，报告 median/min/全部rep。跑正向 A→D 和反向 D→A 两轮，
控制顺序效应。每配置的匹配结果与 results/stage4_fp32_match.npz（阶段4 PyTorch FP32 基准）
比对一致率，作为正确性护栏。

不改模型结构/权重/超参数；不重建任何 engine。
运行环境：conda modelopt（torch+tensorrt+polygraphy），PYTHONPATH=/usr/lib/python3.10/dist-packages。
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
from trt_runner import TrtRunner  # noqa: E402
from e2e_stage_b_infer import score_window_batch  # noqa: E402
from deploy_scene import LiteTrtRunner, numpy_score_window  # noqa: E402
from matching import VARIANTS  # noqa: E402

ROOT = hc.ROOT
ENGINE_DIR = ROOT / "engines"
SEARCH_RADIUS = 5
WARMUP, REPEAT = 2, 10


def timed(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    elapsed, result = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        elapsed.append(time.perf_counter() - t0)
    return result, float(np.median(elapsed)), float(np.min(elapsed)), elapsed


def match_agreement(match_result, fp32_ref):
    out = {}
    for vname in VARIANTS:
        key = vname.replace("+", "_")
        m = match_result[vname]
        bm_rows, bm_cols = fp32_ref[f"{key}_rows"], fp32_ref[f"{key}_cols"]
        out[vname] = float(np.mean((m["rows"] == bm_rows) & (m["cols"] == bm_cols)))
    return out


def run_config(tag, normalized, valid_mask, raw, rows, cols, all_rowcols, valid_rowcols,
                vr, vc, offsets, ref_rows, ref_cols, hsi_runner_trt, pc_runner_trt,
                hsi_runner_lite, pc_runner_lite, fp32_ref):
    """tag in {A,B,C,D}。返回本配置各分段计时 + 正确性。"""
    rowcols = all_rowcols if tag == "A" else valid_rowcols
    n_patches = len(rowcols)

    # --- patch 构建 ---
    def do_patch():
        return build_hsi_patches(normalized, rowcols, 3)
    patches, patch_med, patch_min, patch_reps = timed(do_patch)

    # --- HSI 推理 ---
    if tag == "D":
        def do_hsi():
            return hsi_runner_lite.infer_all(patches)
    else:
        def do_hsi():
            return hsi_runner_trt.infer_batched(patches, max_batch=64)
    hsi_feats, hsi_med, hsi_min, hsi_reps = timed(do_hsi)

    # --- PC 推理 ---
    if tag == "D":
        def do_pc():
            return pc_runner_lite.infer_all(offsets)
    else:
        def do_pc():
            return pc_runner_trt.infer_batched(offsets, max_batch=64)
    pc_feats, pc_med, pc_min, pc_reps = timed(do_pc)

    # --- 特征网格组装（reshape 或 scatter） ---
    dim = hsi_feats.shape[1]
    if tag == "A":
        def do_grid():
            return hsi_feats.reshape(rows, cols, dim).astype(np.float32, copy=True)
    else:
        def do_grid():
            g = np.zeros((rows, cols, dim), dtype=np.float32)
            g[vr, vc] = hsi_feats
            return g
    feature_grid_np, grid_med, grid_min, grid_reps = timed(do_grid, warmup=1, repeat=5)

    # --- 匹配 ---
    if tag in ("A", "B"):
        feature_grid_t = torch.from_numpy(feature_grid_np).float()
        pf_t = torch.from_numpy(pc_feats).float()

        def do_match():
            return {v: score_window_batch(pf_t, feature_grid_t, valid_mask, ref_rows, ref_cols,
                                           SEARCH_RADIUS, sw) for v, sw in VARIANTS.items()}
    else:
        def do_match():
            return {v: numpy_score_window(pc_feats, feature_grid_np, valid_mask, ref_rows, ref_cols,
                                           SEARCH_RADIUS, sw) for v, sw in VARIANTS.items()}
    match_result, match_med, match_min, match_reps = timed(do_match)

    agreement = match_agreement(match_result, fp32_ref)

    return {
        "tag": tag, "n_patches": n_patches,
        "patch_build": {"median_s": patch_med, "min_s": patch_min, "all_reps_s": patch_reps},
        "hsi_infer": {"median_s": hsi_med, "min_s": hsi_min, "all_reps_s": hsi_reps,
                      "us_per_patch_median": hsi_med * 1e6 / n_patches},
        "pc_infer": {"median_s": pc_med, "min_s": pc_min, "all_reps_s": pc_reps},
        "feature_grid_assembly": {"median_s": grid_med, "min_s": grid_min},
        "match_both_variants": {"median_s": match_med, "min_s": match_min, "all_reps_s": match_reps},
        "total_median_s": patch_med + hsi_med + pc_med + grid_med + match_med,
        "match_agreement_vs_stage4_pytorch_fp32": agreement,
    }


def main():
    hc.setup_determinism()
    data = np.load(ROOT / "results/e2e_scene_preprocessed.npz")
    hsi_raw = data["hsi_raw"]
    rows, cols = int(data["hsi_rows"]), int(data["hsi_cols"])
    valid_mask_npz = data["valid_mask"]
    offsets = data["point_offsets"]
    ref_rows, ref_cols = data["ref_rows"], data["ref_cols"]

    print("[setup] standardize + mask（4个配置共用，只测一次）")
    t0 = time.perf_counter()
    normalized, _, _ = standardize_hsi(hsi_raw)
    standardize_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    valid_mask = valid_vegetation_mask(hsi_raw, 143, 262, 0.2, 1e-5)
    mask_sec = time.perf_counter() - t0
    assert np.array_equal(valid_mask, valid_mask_npz), "valid_mask 与 npz 里 Stage A 产出不一致！"
    vr, vc = np.where(valid_mask)
    valid_rowcols = list(zip(vr.tolist(), vc.tolist()))
    all_rowcols = [(r, c) for r in range(rows) for c in range(cols)]
    print(f"  standardize={standardize_sec*1000:.1f}ms mask={mask_sec*1000:.1f}ms "
          f"n_valid={len(valid_rowcols)}/{rows*cols}")

    fp32_ref = np.load(ROOT / "results/stage4_fp32_match.npz")

    print("[setup] 加载 engines/hsi_fp16.plan / pc_fp16.plan（TrtRunner + LiteTrtRunner 各一份）")
    hsi_runner_trt = TrtRunner(ENGINE_DIR / "hsi_fp16.plan")
    pc_runner_trt = TrtRunner(ENGINE_DIR / "pc_fp16.plan")
    hsi_runner_lite = LiteTrtRunner(ENGINE_DIR / "hsi_fp16.plan", input_dims=(342, 3, 3), max_batch=64)
    pc_runner_lite = LiteTrtRunner(ENGINE_DIR / "pc_fp16.plan", input_dims=(15, 3), max_batch=64)

    common_kwargs = dict(
        normalized=normalized, valid_mask=valid_mask, raw=hsi_raw, rows=rows, cols=cols,
        all_rowcols=all_rowcols, valid_rowcols=valid_rowcols, vr=vr, vc=vc, offsets=offsets,
        ref_rows=ref_rows, ref_cols=ref_cols, hsi_runner_trt=hsi_runner_trt, pc_runner_trt=pc_runner_trt,
        hsi_runner_lite=hsi_runner_lite, pc_runner_lite=pc_runner_lite, fp32_ref=fp32_ref,
    )

    report = {"setup": {"standardize_sec": standardize_sec, "mask_sec": mask_sec,
                         "n_valid_pixels": len(valid_rowcols), "n_total_pixels": rows * cols},
              "forward_A_to_D": {}, "reverse_D_to_A": {}}

    print("\n=== 正向 A -> B -> C -> D ===")
    for tag in ["A", "B", "C", "D"]:
        res = run_config(tag, **common_kwargs)
        report["forward_A_to_D"][tag] = res
        print(f"[{tag}] n_patches={res['n_patches']} patch={res['patch_build']['median_s']*1000:.1f}ms "
              f"hsi={res['hsi_infer']['median_s']*1000:.1f}ms({res['hsi_infer']['us_per_patch_median']:.2f}us/p) "
              f"pc={res['pc_infer']['median_s']*1000:.1f}ms grid={res['feature_grid_assembly']['median_s']*1000:.1f}ms "
              f"match={res['match_both_variants']['median_s']*1000:.1f}ms total={res['total_median_s']*1000:.1f}ms "
              f"agree={res['match_agreement_vs_stage4_pytorch_fp32']}")

    print("\n=== 反向 D -> C -> B -> A（控制顺序效应） ===")
    for tag in ["D", "C", "B", "A"]:
        res = run_config(tag, **common_kwargs)
        report["reverse_D_to_A"][tag] = res
        print(f"[{tag}] n_patches={res['n_patches']} patch={res['patch_build']['median_s']*1000:.1f}ms "
              f"hsi={res['hsi_infer']['median_s']*1000:.1f}ms({res['hsi_infer']['us_per_patch_median']:.2f}us/p) "
              f"pc={res['pc_infer']['median_s']*1000:.1f}ms grid={res['feature_grid_assembly']['median_s']*1000:.1f}ms "
              f"match={res['match_both_variants']['median_s']*1000:.1f}ms total={res['total_median_s']*1000:.1f}ms "
              f"agree={res['match_agreement_vs_stage4_pytorch_fp32']}")

    out = ROOT / "results/followup_stage6_ablation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
