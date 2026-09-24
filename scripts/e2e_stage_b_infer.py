"""阶段4 Step2 Stage B：场景级模型推理 + score_window 匹配 + 三后端对比。

运行环境：conda modelopt + PYTHONPATH=/usr/lib/python3.10/dist-packages（tensorrt）。
读取 Stage A 产出的 results/e2e_scene_preprocessed.npz，逐后端跑：
PyTorch FP32（GPU，作为基准）/ TRT FP16 / TRT INT8(implicit)。
匹配逻辑对应 hspc_encoder/inference.py 的 score_window（仅只读引用，未运行该文件）。

计时方法：
- 模型/权重/engine 的加载单独计时为 load_sec，不计入推理耗时（否则 CUDA 初始化、engine 反序列化开销会混入）
- 每个后端先 warmup 一次完整前向，再重复 REPEAT 次，每次前后 torch.cuda.synchronize()，
  用 time.perf_counter() 记录墙钟时间（含 H2D/D2H），报告中位数（用于对比）和最小值（参考）
- PyTorch 侧用 batch=512（沿用 hc.torch_forward），TRT 侧受 engine profile 限制 max_batch=64，
  两者调用粒度不同，是端到端口径下的正常差异，不做归一化
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "hspc_encoder"))
import hspc_common as hc  # noqa: E402
from trt_runner import TrtRunner  # noqa: E402
from matching import VARIANTS, combine_score  # noqa: E402

SEARCH_RADIUS = 5
ENGINE_DIR = ROOT / "engines"
DEVICE = "cuda"
REPEAT = 5


def score_window_batch(point_features, feature_grid, valid_mask, ref_rows, ref_cols, search_radius, spatial_weight):
    """对 inference.py:score_window 的逐点循环版本（保持逻辑一致，仅做批量外层循环）。"""
    rows, cols, _ = feature_grid.shape
    grid_n = F.normalize(feature_grid.reshape(-1, feature_grid.shape[-1]), dim=1)
    pf_n = F.normalize(point_features, dim=1)
    valid_t = torch.from_numpy(valid_mask)
    rr_full, cc_full = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")

    out_rows = np.full(len(ref_rows), -1, dtype=np.int64)
    out_cols = np.full(len(ref_rows), -1, dtype=np.int64)
    out_cos = np.full(len(ref_rows), np.nan, dtype=np.float32)
    out_disp = np.full(len(ref_rows), np.nan, dtype=np.float32)
    out_none = 0

    for i in range(len(ref_rows)):
        ref_row, ref_col = int(ref_rows[i]), int(ref_cols[i])
        r0, r1 = max(0, ref_row - search_radius), min(rows, ref_row + search_radius + 1)
        c0, c1 = max(0, ref_col - search_radius), min(cols, ref_col + search_radius + 1)
        if r0 >= r1 or c0 >= c1:
            out_none += 1
            continue
        mask = valid_t[r0:r1, c0:c1].reshape(-1)
        if not mask.any():
            out_none += 1
            continue
        window = grid_n.reshape(rows, cols, -1)[r0:r1, c0:c1].reshape(-1, grid_n.shape[-1])
        cosine = (pf_n[i:i + 1] @ window.T).reshape(-1)
        cosine = cosine.clone()
        cosine[~mask] = -torch.inf
        rr = rr_full[r0:r1, c0:c1].reshape(-1).float()
        cc = cc_full[r0:r1, c0:c1].reshape(-1).float()
        distance = torch.sqrt((rr - ref_row) ** 2 + (cc - ref_col) ** 2)
        final_score = combine_score(cosine, distance, spatial_weight, search_radius)
        final_score = final_score.clone()
        final_score[~mask] = -torch.inf
        if not torch.isfinite(final_score).any():
            out_none += 1
            continue
        best = int(torch.argmax(final_score))
        width = c1 - c0
        mr = r0 + best // width
        mc = c0 + best % width
        out_rows[i], out_cols[i] = mr, mc
        out_cos[i] = float(cosine[best])
        out_disp[i] = float(np.hypot(mr - ref_row, mc - ref_col))
    return {"rows": out_rows, "cols": out_cols, "cosine": out_cos, "pixel_displacement": out_disp, "n_none": out_none}


def load_backend(name):
    """加载模型/engine，单独计时，不计入推理耗时。返回 (hsi_forward_fn, pc_forward_fn, load_sec)。"""
    t0 = time.perf_counter()
    if name == "pytorch_fp32":
        hc.setup_determinism()
        pc = hc.get_model("pc", device=DEVICE)
        hsi = hc.get_model("hsi", device=DEVICE)

        def hsi_fwd(x):
            with hc.math_sdpa():
                return hc.torch_forward(hsi, x, device=DEVICE, batch=512)

        def pc_fwd(x):
            return hc.torch_forward(pc, x, device=DEVICE, batch=512)
    else:
        prefix = {"trt_fp16": "fp16", "trt_int8": "int8_implicit"}[name]
        hsi_runner = TrtRunner(ENGINE_DIR / f"hsi_{prefix}.plan")
        pc_runner = TrtRunner(ENGINE_DIR / f"pc_{prefix}.plan")
        hsi_fwd = lambda x: hsi_runner.infer_batched(x, max_batch=64)  # noqa: E731
        pc_fwd = lambda x: pc_runner.infer_batched(x, max_batch=64)  # noqa: E731
    torch.cuda.synchronize()
    load_sec = time.perf_counter() - t0
    return hsi_fwd, pc_fwd, load_sec


def timed_repeat(fn, x, repeat=REPEAT):
    """warmup一次 + 重复repeat次，每次前后同步。返回 (outputs_per_rep, elapsed_per_rep_sec)。"""
    _ = fn(x)  # warmup
    torch.cuda.synchronize()
    outs, elapsed = [], []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(x)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - t0)
        outs.append(out)
    return outs, elapsed


def run_backend(name, hsi_grid_patches, point_offsets, rows, cols, valid_mask, ref_rows, ref_cols, truth):
    hsi_fwd, pc_fwd, load_sec = load_backend(name)

    hsi_outs, hsi_elapsed = timed_repeat(hsi_fwd, hsi_grid_patches)
    pc_outs, pc_elapsed = timed_repeat(pc_fwd, point_offsets)

    hsi_feats = hsi_outs[0]
    pc_feats = pc_outs[0]
    determinism = {
        "hsi_feats_identical_across_reps": bool(all(np.array_equal(hsi_feats, o) for o in hsi_outs[1:])),
        "pc_feats_identical_across_reps": bool(all(np.array_equal(pc_feats, o) for o in pc_outs[1:])),
    }

    feature_grid = torch.from_numpy(hsi_feats).float().reshape(rows, cols, -1)
    pf = torch.from_numpy(pc_feats).float()

    matching = {}
    matching_reps = {}  # 用于 PyTorch FP32 的 rows/cols 确定性复核
    for vname, sw in VARIANTS.items():
        t2 = time.perf_counter()
        m = score_window_batch(pf, feature_grid, valid_mask, ref_rows, ref_cols, SEARCH_RADIUS, sw)
        match_time = time.perf_counter() - t2
        matching[vname] = {
            "rows": m["rows"], "cols": m["cols"], "cosine": m["cosine"],
            "pixel_displacement": m["pixel_displacement"], "n_none": m["n_none"],
            "match_time_sec": match_time,
        }
        if name == "pytorch_fp32":
            reps = []
            for feats_rep in hsi_outs[1:] + [hsi_feats]:
                fg_rep = torch.from_numpy(feats_rep).float().reshape(rows, cols, -1)
                m_rep = score_window_batch(pf, fg_rep, valid_mask, ref_rows, ref_cols, SEARCH_RADIUS, sw)
                reps.append(m_rep)
            matching_reps[vname] = all(
                np.array_equal(r["rows"], m["rows"]) and np.array_equal(r["cols"], m["cols"]) for r in reps
            )

    if name == "pytorch_fp32":
        determinism["match_rowcols_identical_across_reps"] = {k: bool(v) for k, v in matching_reps.items()}

    return {
        "backend": name,
        "load_sec": load_sec,
        "timing_sec": {
            "hsi_infer_median": float(np.median(hsi_elapsed)),
            "hsi_infer_min": float(np.min(hsi_elapsed)),
            "hsi_infer_all_reps": hsi_elapsed,
            "pc_infer_median": float(np.median(pc_elapsed)),
            "pc_infer_min": float(np.min(pc_elapsed)),
            "pc_infer_all_reps": pc_elapsed,
        },
        "determinism": determinism,
        "matching": matching,
        "hsi_feats": hsi_feats,
        "pc_feats": pc_feats,
    }


def compute_runner_overhead(hsi_median_sec, n_patches):
    """跟 results/stage3_benchmark.json 的 batch=64 kernel-only p50 对比，算出真实开销倍数。"""
    stage3_path = ROOT / "results/stage3_benchmark.json"
    if not stage3_path.exists():
        return None
    stage3 = json.loads(stage3_path.read_text())
    out = {}
    for key, mode in [("trt_fp16", "trt_fp16"), ("trt_int8", "trt_int8_implicit")]:
        records = stage3.get("results", {}).get("hsi", {}).get(mode, [])
        rec64 = next((r for r in records if r["batch"] == 64), None)
        if rec64 is None:
            continue
        kernel_us_per_patch = rec64["p50_ms"] * 1000 / 64
        e2e_us_per_patch = hsi_median_sec[key] * 1e6 / n_patches
        out[key] = {
            "stage3_batch64_p50_ms": rec64["p50_ms"],
            "kernel_only_us_per_patch": kernel_us_per_patch,
            "e2e_us_per_patch": e2e_us_per_patch,
            "overhead_ratio": e2e_us_per_patch / kernel_us_per_patch,
        }
    return out


def main():
    data = np.load(ROOT / "results/e2e_scene_preprocessed.npz")
    hsi_grid_patches = data["hsi_grid_patches"]
    hsi_raw = data["hsi_raw"]  # (bands, rows, cols)
    rows, cols = int(data["hsi_rows"]), int(data["hsi_cols"])
    valid_mask = data["valid_mask"]
    point_offsets = data["point_offsets"]
    ref_rows, ref_cols = data["ref_rows"], data["ref_cols"]
    truth = data["truth_spectra"]
    n_points = len(ref_rows)
    n_patches = hsi_grid_patches.shape[0]
    print(f"scene: grid={rows}x{cols} patches={n_patches} points={n_points}")

    backends = {}
    for name in ["pytorch_fp32", "trt_fp16", "trt_int8"]:
        print(f"=== {name} ===")
        backends[name] = run_backend(name, hsi_grid_patches, point_offsets, rows, cols, valid_mask, ref_rows, ref_cols, truth)
        b = backends[name]
        print(f"  load_sec: {b['load_sec']:.2f}s")
        print(f"  hsi_infer median={b['timing_sec']['hsi_infer_median']*1000:.1f}ms min={b['timing_sec']['hsi_infer_min']*1000:.1f}ms")
        print(f"  pc_infer  median={b['timing_sec']['pc_infer_median']*1000:.1f}ms min={b['timing_sec']['pc_infer_min']*1000:.1f}ms")
        print(f"  determinism: {b['determinism']}")

    base = backends["pytorch_fp32"]

    # 供 scripts/deploy_scene.py（阶段5，不依赖torch的部署链路）比对用的 FP32 基准匹配结果
    fp32_match = {}
    for vname in VARIANTS:
        key = vname.replace("+", "_")
        fp32_match[f"{key}_rows"] = base["matching"][vname]["rows"]
        fp32_match[f"{key}_cols"] = base["matching"][vname]["cols"]
    np.savez(ROOT / "results/stage4_fp32_match.npz", **fp32_match)

    report = {
        "scene": "24data/10.6/1", "n_grid_patches": int(n_patches), "n_points": int(n_points),
        "timing_method": (
            f"warmup 1 + repeat {REPEAT}，每次前后 cuda.synchronize()，perf_counter 墙钟计时；"
            "加载(load_sec)不计入推理时间；报告中位数(median)与最小值(min)"
        ),
        "backends": {},
    }
    for name, res in backends.items():
        entry = {
            "load_sec": res["load_sec"],
            "timing_sec": {k: v for k, v in res["timing_sec"].items() if not k.endswith("_all_reps")},
            "timing_sec_all_reps": {
                "hsi_infer": res["timing_sec"]["hsi_infer_all_reps"],
                "pc_infer": res["timing_sec"]["pc_infer_all_reps"],
            },
            "determinism": res["determinism"],
            "variants": {},
        }
        if name != "pytorch_fp32":
            entry["speedup_vs_pytorch_fp32"] = {
                "hsi_infer_median": base["timing_sec"]["hsi_infer_median"] / res["timing_sec"]["hsi_infer_median"],
                "pc_infer_median": base["timing_sec"]["pc_infer_median"] / res["timing_sec"]["pc_infer_median"],
            }
        for vname in VARIANTS:
            m = res["matching"][vname]
            bm = base["matching"][vname]
            row_agree = np.mean((m["rows"] == bm["rows"]) & (m["cols"] == bm["cols"]))
            disp_diff = np.abs(m["pixel_displacement"] - bm["pixel_displacement"])
            disp_diff = disp_diff[np.isfinite(disp_diff)]
            valid_match = m["rows"] >= 0
            matched_spectra_err = np.full(n_points, np.nan, dtype=np.float32)
            if valid_match.any():
                mr, mc = m["rows"][valid_match], m["cols"][valid_match]
                matched_spectra = hsi_raw[:, mr, mc].T  # (n_valid, bands)
                err = np.abs(matched_spectra - truth[valid_match]).max(axis=1)
                matched_spectra_err[valid_match] = err
            entry["variants"][vname] = {
                "n_none": int(m["n_none"]),
                "match_time_sec": m["match_time_sec"],
                "pixel_match_agreement_vs_fp32": float(row_agree) if name != "pytorch_fp32" else 1.0,
                "pixel_displacement_mean": float(np.nanmean(m["pixel_displacement"])),
                "pixel_displacement_diff_vs_fp32_mean": float(disp_diff.mean()) if name != "pytorch_fp32" and len(disp_diff) else 0.0,
                "cosine_mean": float(np.nanmean(m["cosine"])),
                "matched_spectra_max_abs_err_vs_truth_mean": float(np.nanmean(matched_spectra_err)),
                "matched_spectra_max_abs_err_vs_truth_median": float(np.nanmedian(matched_spectra_err)),
            }
        if name != "pytorch_fp32":
            entry["hsi_embedding_accuracy"] = hc.accuracy_report(base["hsi_feats"], res["hsi_feats"])
            entry["pc_embedding_accuracy"] = hc.accuracy_report(base["pc_feats"], res["pc_feats"])
        report["backends"][name] = entry

    hsi_median_sec = {k: backends[k]["timing_sec"]["hsi_infer_median"] for k in ["trt_fp16", "trt_int8"]}
    report["runner_overhead_vs_stage3_kernel"] = compute_runner_overhead(hsi_median_sec, n_patches)

    out = ROOT / "results/stage4_e2e.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
