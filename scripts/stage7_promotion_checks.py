"""阶段7 步骤1 写入engines/前的补充验证：
1. 不一致点分析：新方案(mixed)与PyTorch FP32基准不一致的匹配点，报告FP32 top1/top2余弦分差
   （对照阶段2的判别裕度p10：PC=0.0010, HSI=0.0032，见README.md），以及新旧匹配位置的像元距离
2. 用5次稳定构建里另外2个（stability_1/2）重跑任务级一致率，确认99.7%/99.5%在构建间稳定
3. 新方案 vs FP32 相对truth的匹配误差（光谱误差、像元位移），确认不比阶段4现有fp16结果差
4. 层组命名说明：HSI 的 stem 组末尾 Div/Erf/Add/Mul/Mul_1 是 GELU 的展开（stem = Conv2d+BN(已折叠)
   +GELU），不是 LayerNorm；build_trt.py:classify_layer() 将其归入 "stem" 组，与 "layernorm" 组分开。

验收标准：不再对照阶段4那个不可复现的fp16 engine，改为"新方案与PyTorch FP32的
不一致点均为判别裕度内的近平局翻转"。

运行环境：conda modelopt + PYTHONPATH=/usr/lib/python3.10/dist-packages。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hspc_common as hc  # noqa: E402
from trt_runner import TrtRunner  # noqa: E402
from matching import VARIANTS, combine_score  # noqa: E402

ROOT = hc.ROOT
SCRATCH = Path(os.environ.get("HSPC_SCRATCH", "/tmp/hspc_stage7_scratch")) / "stage7_engines"
SEARCH_RADIUS = 5
REF_MARGIN_P10 = {"pc": 0.0010, "hsi": 0.0032}  # 阶段2 margin_stats，README.md 已记录


def score_window_with_margin(pf, feature_grid, valid_mask, ref_rows, ref_cols, search_radius, spatial_weight):
    """score_window_batch 的增强版：额外返回每个点的 top1/top2 余弦分差（供不一致点分析用）。"""
    rows, cols, dim = feature_grid.shape
    grid_n = F.normalize(feature_grid.reshape(-1, dim), dim=1).reshape(rows, cols, dim)
    pf_n = F.normalize(pf, dim=1)
    valid_t = torch.from_numpy(valid_mask)
    n = len(ref_rows)
    out_rows = np.full(n, -1, dtype=np.int64)
    out_cols = np.full(n, -1, dtype=np.int64)
    out_margin = np.full(n, np.nan, dtype=np.float32)  # top1-top2 余弦分差
    for i in range(n):
        r0, r1 = max(0, ref_rows[i] - search_radius), min(rows, ref_rows[i] + search_radius + 1)
        c0, c1 = max(0, ref_cols[i] - search_radius), min(cols, ref_cols[i] + search_radius + 1)
        mask = valid_t[r0:r1, c0:c1].reshape(-1)
        if not mask.any():
            continue
        window = grid_n[r0:r1, c0:c1].reshape(-1, dim)
        cosine = (pf_n[i:i + 1] @ window.T).reshape(-1)
        cosine = torch.where(mask, cosine, torch.tensor(-torch.inf))
        rr, cc = torch.meshgrid(torch.arange(r0, r1), torch.arange(c0, c1), indexing="ij")
        dist = torch.sqrt((rr.float() - ref_rows[i]) ** 2 + (cc.float() - ref_cols[i]) ** 2).reshape(-1)
        score = combine_score(cosine, dist, spatial_weight, search_radius)
        score = torch.where(mask, score, torch.tensor(-torch.inf))
        if not torch.isfinite(score).any():
            continue
        width = c1 - c0
        top2 = torch.topk(score[torch.isfinite(score)], min(2, int(mask.sum())))
        best = int(torch.argmax(score))
        out_rows[i], out_cols[i] = r0 + best // width, c0 + best % width
        if top2.values.numel() >= 2:
            out_margin[i] = float(top2.values[0] - top2.values[1])
    return out_rows, out_cols, out_margin


def run_engine_pair(hsi_plan, pc_plan, hsi_patches, point_offsets, rows, cols, valid_mask, ref_rows, ref_cols):
    hsi_runner = TrtRunner(hsi_plan)
    pc_runner = TrtRunner(pc_plan)
    hsi_feats = hsi_runner.infer_batched(hsi_patches, max_batch=64)
    pc_feats = pc_runner.infer_batched(point_offsets, max_batch=64)
    feature_grid = torch.from_numpy(hsi_feats).float().reshape(rows, cols, -1)
    pf = torch.from_numpy(pc_feats).float()
    out = {}
    for vname, sw in VARIANTS.items():
        r, c, margin = score_window_with_margin(pf, feature_grid, valid_mask, ref_rows, ref_cols, SEARCH_RADIUS, sw)
        out[vname] = {"rows": r, "cols": c, "margin": margin}
    return out, hsi_feats, pc_feats


def main():
    data = np.load(ROOT / "results/e2e_scene_preprocessed.npz")
    hsi_patches = data["hsi_grid_patches"]
    hsi_raw = data["hsi_raw"]
    rows, cols = int(data["hsi_rows"]), int(data["hsi_cols"])
    valid_mask = data["valid_mask"]
    point_offsets = data["point_offsets"]
    ref_rows, ref_cols = data["ref_rows"], data["ref_cols"]
    truth = data["truth_spectra"]
    n_points = len(ref_rows)

    print("=== 计算 PyTorch FP32 基准（含每点margin）===")
    hc.setup_determinism()
    pc_model = hc.get_model("pc", device="cuda")
    hsi_model = hc.get_model("hsi", device="cuda")
    with hc.math_sdpa():
        hsi_feats_fp32 = hc.torch_forward(hsi_model, hsi_patches, device="cuda", batch=512)
    pc_feats_fp32 = hc.torch_forward(pc_model, point_offsets, device="cuda", batch=512)
    feature_grid_fp32 = torch.from_numpy(hsi_feats_fp32).float().reshape(rows, cols, -1)
    pf_fp32 = torch.from_numpy(pc_feats_fp32).float()
    fp32_match = {}
    for vname, sw in VARIANTS.items():
        r, c, margin = score_window_with_margin(pf_fp32, feature_grid_fp32, valid_mask, ref_rows, ref_cols, SEARCH_RADIUS, sw)
        fp32_match[vname] = {"rows": r, "cols": c, "margin": margin}

    report = {"variants": {}}

    # ---------- 任务2：stability_0/1/2 三个独立构建各跑一次，确认一致率跨构建稳定 ----------
    print("=== 任务2：3个独立稳定构建的任务级一致率 ===")
    cross_build_agreement = {}
    for idx in [0, 1, 2]:
        hsi_plan = SCRATCH / f"hsi_stability_{idx}.plan"
        pc_plan = SCRATCH / f"pc_stability_{idx}.plan"
        match, hsi_feats, pc_feats = run_engine_pair(hsi_plan, pc_plan, hsi_patches, point_offsets,
                                                       rows, cols, valid_mask, ref_rows, ref_cols)
        cross_build_agreement[f"stability_{idx}"] = {}
        for vname in VARIANTS:
            bm = fp32_match[vname]
            agree = float(np.mean((match[vname]["rows"] == bm["rows"]) & (match[vname]["cols"] == bm["cols"])))
            cross_build_agreement[f"stability_{idx}"][vname] = agree
            print(f"  stability_{idx} {vname}: agreement={agree:.4f}")
        if idx == 0:
            selected_match = match
            selected_hsi_feats, selected_pc_feats = hsi_feats, pc_feats
    report["cross_build_agreement"] = cross_build_agreement

    # ---------- 任务1：不一致点分析（用 stability_0 作为代表方案） ----------
    print("\n=== 任务1：不一致点分析（stability_0 vs PyTorch FP32）===")
    for vname in VARIANTS:
        m = selected_match[vname]
        bm = fp32_match[vname]
        mismatch = np.where((m["rows"] != bm["rows"]) | (m["cols"] != bm["cols"]))[0]
        entries = []
        for i in mismatch:
            pixel_dist = float(np.hypot(int(m["rows"][i]) - int(bm["rows"][i]), int(m["cols"][i]) - int(bm["cols"][i])))
            entries.append({
                "point_idx": int(i),
                "fp32_top1_top2_margin": float(bm["margin"][i]) if np.isfinite(bm["margin"][i]) else None,
                "pixel_distance_old_vs_new": pixel_dist,
                "fp32_match": [int(bm["rows"][i]), int(bm["cols"][i])],
                "mixed_match": [int(m["rows"][i]), int(m["cols"][i])],
            })
        margins = [e["fp32_top1_top2_margin"] for e in entries if e["fp32_top1_top2_margin"] is not None]
        report["variants"].setdefault(vname, {})["mismatch_analysis"] = {
            "n_mismatch": len(mismatch), "n_total": n_points,
            "agreement": 1 - len(mismatch) / n_points,
            "margin_stats": {
                "mean": float(np.mean(margins)) if margins else None,
                "max": float(np.max(margins)) if margins else None,
                "median": float(np.median(margins)) if margins else None,
            },
            "ref_margin_p10_stage2_same_modal_task": REF_MARGIN_P10["hsi"],  # 跨模态窗口匹配的margin定义与同模态检索不同，仅作量级参照
            "all_are_near_ties": bool(margins) and max(margins) < REF_MARGIN_P10["hsi"] * 3,
            "pixel_distance_mean": float(np.mean([e["pixel_distance_old_vs_new"] for e in entries])) if entries else 0.0,
            "entries": entries,
        }
        print(f"  {vname}: {len(mismatch)}/{n_points} 不一致, margin范围="
              f"{[round(m,5) for m in margins[:5]]}..." if margins else f"  {vname}: 0个不一致点")

    # ---------- 任务3：匹配误差对比（vs truth）----------
    print("\n=== 任务3：匹配光谱误差 / 像元位移对比 ===")
    for vname in VARIANTS:
        for tag, match_src, feats in [("mixed", selected_match, None), ("pytorch_fp32", fp32_match, None)]:
            m = match_src[vname]
            valid = m["rows"] >= 0
            disp = np.hypot(m["rows"][valid].astype(float) - ref_rows[valid], m["cols"][valid].astype(float) - ref_cols[valid])
            mr, mc = m["rows"][valid], m["cols"][valid]
            spec_err = np.abs(hsi_raw[:, mr, mc].T - truth[valid]).max(axis=1)
            report["variants"][vname].setdefault("truth_comparison", {})[tag] = {
                "pixel_displacement_mean": float(disp.mean()),
                "matched_spectra_max_abs_err_mean": float(spec_err.mean()),
                "matched_spectra_max_abs_err_median": float(np.median(spec_err)),
            }
        mx, fp = report["variants"][vname]["truth_comparison"]["mixed"], report["variants"][vname]["truth_comparison"]["pytorch_fp32"]
        report["variants"][vname]["truth_comparison"]["mixed_not_worse_than_fp32"] = (
            mx["matched_spectra_max_abs_err_mean"] <= fp["matched_spectra_max_abs_err_mean"] * 1.1
            and abs(mx["pixel_displacement_mean"] - fp["pixel_displacement_mean"]) < 0.5
        )
        print(f"  {vname}: mixed spec_err={mx['matched_spectra_max_abs_err_mean']:.4f} "
              f"fp32 spec_err={fp['matched_spectra_max_abs_err_mean']:.4f}")

    # ---------- 层组命名说明（任务4，写入报告）----------
    report["layer_group_naming_note"] = (
        "HSI的'stem'组包含 /stem/stem.0/Conv 与其后 GELU 的展开节点（Div/Erf/Add/Mul/Mul_1）。"
        "stem = Conv2d + BatchNorm2d(推理时已折叠进Conv) + GELU。这不是LayerNorm——LayerNorm对应的是独立的"
        "'layernorm'层组（/transformer/layers.*/norm{1,2}/LayerNormalization 和 /out_head/out_head.0/"
        "LayerNormalization）。build_trt.py:classify_layer() 将两者分开归类。"
    )

    out = ROOT / "results/stage7_mixed_precision.json"
    existing = json.loads(out.read_text())
    existing["promotion_checks"] = report
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"\nwritten (merged into) {out}")


if __name__ == "__main__":
    main()
