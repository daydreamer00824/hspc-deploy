"""C++ 部署程序的逐位护栏（板上运行）。C++ 导出的数组与 Python 参考链路逐数组比较。

  cpp_guardrail.py latency-refs <dir>    导出验证集前 8 条样本（float32 二进制）与 Python 引擎的 batch 1/8 输出，供 hspc_latency 使用
  cpp_guardrail.py latency-check <dir>   比较 hspc_latency 的 --dump-dir 输出与 Python 输出（eager / graph 各一份）
  cpp_guardrail.py prep-check <scene> <dir>  比较 hspc_prep_check 导出的 CPU 前处理数组与 Python 链路（掩膜、均值/标准差、点云坐标、
                                             投影、采样、kNN 邻域偏移），scene 形如 24data/10.6/1；需要 conda 环境（gdal/pyproj/laspy/scipy）
  cpp_guardrail.py deploy-check <scene> <dir>  比较 hspc_deploy --dump-dir 导出的整景链路中间结果（patch、HSI/PC 特征）和最终匹配行列与 Python 链路
                                             （preprocess.py + LiteTrtRunner + deploy_jetson.Pipeline 最终配置 E）；conda 环境（含 torch）
  cpp_guardrail.py matchbench-py <scene>  Python（torch）版"拼特征网格 + 窗口匹配"在 GPU 上的耗时（CUDA event，含最后取回结果），与 C++ 融合 kernel 对照
  cpp_guardrail.py knn-ties <scene> [out.json]  统计该景 1000 个采样点的 kNN 里距离并列的情况，以及"并列时改按下标取舍"会改变多少点的邻域（说明为什么 C++ 直接用 SciPy 的 kd 树内核）
  cpp_guardrail.py window-cover <scene> <dir> [out.json]  匹配窗口（11x11）覆盖了多少有效像元：只对被覆盖的像元做 HSI 推理能省多少（读 hspc_deploy --dump-dir 的导出）
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")


def latency_refs(d: Path):
    import torch
    import hspc_common as hc
    from jetson_scene_opt import ENGINE_DIR, GraphRunner
    hc.setup_determinism()
    d.mkdir(parents=True, exist_ok=True)
    for which in ("pc", "hsi"):
        val = hc.load_samples(which, "val")[:8].astype(np.float32)
        val.tofile(d / f"{which}_samples.f32")
        plan = ENGINE_DIR / f"{which}_fp16.plan"
        for b in (1, 8):
            r = GraphRunner(plan, which, b, use_graph=False)
            with torch.cuda.stream(r.stream):
                r.x.copy_(torch.from_numpy(val[:b].copy()).cuda())
                r.run()
            r.stream.synchronize()
            r.y.float().cpu().numpy().astype(np.float32).tofile(d / f"{which}_b{b}_python.f32")
            del r
    print("refs written ->", d)


def latency_check(d: Path):
    res = {}
    for which in ("pc", "hsi"):
        for b in (1, 8):
            ref = np.fromfile(d / f"{which}_b{b}_python.f32", dtype=np.float32)
            for mode in ("eager", "graph"):
                out = np.fromfile(d / f"{which}_b{b}_{mode}.f32", dtype=np.float32)
                res[f"{which}_b{b}_{mode}"] = bool(ref.shape == out.shape and np.array_equal(ref, out))
    print(json.dumps(res, indent=2))
    sys.exit(0 if all(res.values()) else 1)


def _loader(d: Path):
    """按 (name, dtype) 从导出目录读一个数组；prep_check/deploy_check 共用，避免各自定义一份同名闭包。"""
    def ld(name, dt):
        return np.fromfile(d / name, dtype=dt)
    return ld


def _scene_features(scene: str):
    """deploy_check 和 matchbench_py 共用的部分：读 HSI → 掩膜 → 标准化 → patch → LAS 段 → 两个引擎推理。
    返回的 dict 里的 ds/dj 模块对象和 LiteTrtRunner 已经在返回前释放（del + empty_cache），
    调用方如果还要用 dj.Pipeline 之类的，自己重新 import。"""
    import torch

    import deploy_jetson as dj
    import deploy_scene as ds
    from preprocess import build_hsi_patches, load_hsi, standardize_hsi, valid_vegetation_mask

    hsi_path, las_path = ds.scene_paths(scene)
    C = ds.CONTRACT
    raw, gt, wkt = load_hsi(hsi_path, C["target_bands"])
    mask = valid_vegetation_mask(raw, C["red_band_index"], C["nir_band_index"], C["ndvi_threshold"], C["nodata_epsilon"])
    normalized, _, _ = standardize_hsi(raw)
    vr, vc = np.where(mask)
    patches = build_hsi_patches(normalized, list(zip(vr.tolist(), vc.tolist())), C["patch_size"])
    offsets, eligible, ref_rows, ref_cols, _ = ds.run_las_pipeline(
        las_path, C["point_cloud_crs"], wkt, gt, ds.SPATIAL_CONTRACT, mask, raw.shape[1], raw.shape[2], C["point_neighbors"],
        ds.SAMPLES_PER_SCENE, ds.SEED)
    hsi = ds.LiteTrtRunner(ROOT / "engines/hsi_fp16_scene1024.plan", dj.INPUT_DIMS["hsi"], max_batch=1024)
    pc = ds.LiteTrtRunner(ROOT / "engines/pc_fp16_scene1024.plan", dj.INPUT_DIMS["pc"], max_batch=1024)
    y_hsi, y_pc = hsi.infer_all(patches), pc.infer_all(offsets)
    del hsi, pc
    torch.cuda.empty_cache()
    return {"hsi_path": hsi_path, "las_path": las_path, "raw": raw, "gt": gt, "wkt": wkt, "mask": mask, "vr": vr, "vc": vc,
            "patches": patches, "offsets": offsets, "eligible": eligible, "ref_rows": ref_rows, "ref_cols": ref_cols,
            "y_hsi": y_hsi, "y_pc": y_pc, "C": C}


def prep_check(scene: str, d: Path):
    import laspy
    from pyproj import Transformer

    import deploy_scene as ds
    from preprocess import load_hsi, map_to_fractional_pixel, standardize_hsi, valid_vegetation_mask

    hsi_path, las_path = ds.scene_paths(scene)
    raw, gt, wkt = load_hsi(hsi_path, ds.CONTRACT["target_bands"])
    bands, rows, cols = raw.shape
    C = ds.CONTRACT
    mask = valid_vegetation_mask(raw, C["red_band_index"], C["nir_band_index"], C["ndvi_threshold"], C["nodata_epsilon"])
    _, mean, std = standardize_hsi(raw)
    las = laspy.read(las_path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    tr = Transformer.from_crs(C["point_cloud_crs"], wkt, always_xy=True)
    hx, hy = tr.transform(xyz[:, 0], xyz[:, 1])
    rows_f, cols_f = map_to_fractional_pixel(gt, hx, hy)
    offsets, eligible, ref_rows, ref_cols, _ = ds.run_las_pipeline(
        las_path, C["point_cloud_crs"], wkt, gt, ds.SPATIAL_CONTRACT, mask, rows, cols, C["point_neighbors"],
        ds.SAMPLES_PER_SCENE, ds.SEED)
    full_rows, full_cols = np.floor(rows_f).astype(np.int64), np.floor(cols_f).astype(np.int64)

    ld = _loader(d)

    # 每个导出数组只读一次：mean/std/offsets/hsi_x 各在 res 和 info 里都要用到
    ld_mean, ld_std = ld("mean.f32", np.float32), ld("std.f32", np.float32)
    ld_offsets = ld("offsets.f32", np.float32).reshape(offsets.shape)
    ld_hsi_x, ld_hsi_y = ld("hsi_x.f64", np.float64), ld("hsi_y.f64", np.float64)

    res = {"shape": [int(x) for x in (d / "shape.txt").read_text().split()] == [bands, rows, cols],
           "gt": bool(np.array_equal(ld("gt.f64", np.float64), np.asarray(gt, dtype=np.float64))),
           "wkt": (d / "wkt.txt").read_text() == wkt,
           "raw_cube": bool(np.array_equal(ld("raw.f32", np.float32), raw.reshape(-1))),
           "valid_mask": bool(np.array_equal(ld("mask.u8", np.uint8).astype(bool), mask.reshape(-1))),
           "band_mean": bool(np.array_equal(ld_mean, mean.reshape(-1))),
           "band_std": bool(np.array_equal(ld_std, std.reshape(-1))),
           "xyz": bool(np.array_equal(ld("xyz.f64", np.float64), xyz.reshape(-1))),
           "projected_xy_bitwise": bool(np.array_equal(ld_hsi_x, hx) and np.array_equal(ld_hsi_y, hy)),
           "full_rows": bool(np.array_equal(ld("full_rows.i64", np.int64), full_rows)),
           "full_cols": bool(np.array_equal(ld("full_cols.i64", np.int64), full_cols)),
           "sampled_eligible": bool(np.array_equal(ld("eligible.i64", np.int64), eligible)),
           "ref_rows": bool(np.array_equal(ld("ref_rows.i64", np.int64), ref_rows)),
           "ref_cols": bool(np.array_equal(ld("ref_cols.i64", np.int64), ref_cols)),
           "point_offsets": bool(np.array_equal(ld_offsets.reshape(-1), offsets.reshape(-1)))}
    info = {}
    for nm, a, b in (("band_mean", ld_mean, mean.reshape(-1)), ("band_std", ld_std, std.reshape(-1))):
        info[nm + "_n_diff"] = int((a != b).sum())
        info[nm + "_max_abs"] = float(np.abs(a.astype(np.float64) - b).max())
    row_diff = np.where((ld_offsets != offsets).reshape(len(offsets), -1).any(axis=1))[0]
    info["offsets_rows_differ"] = int(len(row_diff))
    # 只是邻域顺序不同（集合相同）的点数：对每个点把 15 个偏移按字典序排序后比较
    def canon(a):
        return np.sort(a.view([("x", np.float32), ("y", np.float32), ("z", np.float32)]).reshape(a.shape[0], -1), axis=1, order=("x", "y", "z"))
    if len(row_diff):
        info["offsets_rows_differ_only_in_order"] = int(sum(np.array_equal(canon(ld_offsets[i:i + 1]), canon(offsets[i:i + 1])) for i in row_diff))
    info["projected_x_max_abs"] = float(np.abs(ld_hsi_x - hx).max())
    info["n_eligible_sampled"] = int(len(eligible))
    print(json.dumps({"scene": scene, "identical": res, "info": info}, indent=2))
    sys.exit(0 if all(res.values()) else 1)


def deploy_check(scene: str, d: Path):
    import deploy_jetson as dj
    from matching import VARIANTS

    sf = _scene_features(scene)
    mask, patches, offsets, y_hsi, y_pc = sf["mask"], sf["patches"], sf["offsets"], sf["y_hsi"], sf["y_pc"]
    pipe = dj.Pipeline(ROOT / "engines/hsi_fp16_scene1024.plan", ROOT / "engines/pc_fp16_scene1024.plan", 1024)
    e_out, _ = pipe.run(sf["hsi_path"], sf["las_path"], overlap_las=True, las_workers=-1, las_parallel=True)

    ld = _loader(d)

    n_valid, n_pts, rows, cols = [int(x) for x in (d / "meta.txt").read_text().split()]
    res = {"n_valid_and_points": (n_valid, n_pts) == (len(sf["vr"]), len(offsets)),
           "valid_mask": bool(np.array_equal(ld("mask.u8", np.uint8).astype(bool), mask.reshape(-1))),
           "patches": bool(np.array_equal(ld("patches.f32", np.float32).reshape(patches.shape), patches)),
           "point_offsets": bool(np.array_equal(ld("offsets.f32", np.float32).reshape(offsets.shape), offsets)),
           "ref_rows": bool(np.array_equal(ld("ref_rows.i32", np.int32), sf["ref_rows"])),
           "hsi_features": bool(np.array_equal(ld("hsi_feat.f32", np.float32).reshape(y_hsi.shape), y_hsi)),
           "pc_features": bool(np.array_equal(ld("pc_feat.f32", np.float32).reshape(y_pc.shape), y_pc))}
    info = {"n_valid": n_valid, "n_points": n_pts}
    for vi, v in enumerate(VARIANTS):
        cr, cc = ld(f"match_rows_{vi}.i32", np.int32), ld(f"match_cols_{vi}.i32", np.int32)
        pr, pcc = np.asarray(e_out["match"][v]["rows"]), np.asarray(e_out["match"][v]["cols"])
        res[f"match_rowcols_{v}"] = bool(np.array_equal(cr, pr) and np.array_equal(cc, pcc))
        diff = np.where((cr != pr) | (cc != pcc))[0]
        info[f"match_{v}_n_diff_points"] = int(len(diff))
        if len(diff):
            info[f"match_{v}_diff_points"] = diff.tolist()[:20]
    print(json.dumps({"scene": scene, "identical": res, "info": info}, indent=2))
    sys.exit(0 if all(res.values()) else 1)


def matchbench_py(scene: str):
    """输入固定为该景的真实特征。范围：整幅特征网格的构建（zeros + 按有效像元写入）+ gpu_score_window_multi（含特征归一化、窗口 gather、bmm、
    两个 variant 的打分/argmax 与结果取回）——与 C++ 的 gpu_normalize_match_download（特征归一化 + 融合匹配 kernel + 结果取回）同一范围。"""
    import torch

    import deploy_jetson as dj
    from matching import VARIANTS

    sf = _scene_features(scene)
    mask, ref_rows, ref_cols, C = sf["mask"], sf["ref_rows"], sf["ref_cols"], sf["C"]
    vr_t, vc_t = torch.from_numpy(sf["vr"]).cuda(), torch.from_numpy(sf["vc"]).cuda()
    yh, yp = torch.from_numpy(sf["y_hsi"]).cuda(), torch.from_numpy(sf["y_pc"]).cuda()
    rows, cols = sf["raw"].shape[1:]
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def once():
        torch.cuda.synchronize()
        e0.record()
        grid = torch.zeros((rows, cols, 1024), dtype=torch.float32, device="cuda")
        grid[vr_t, vc_t] = yh
        dj.gpu_score_window_multi(yp, grid, mask, ref_rows, ref_cols, C["search_radius"], VARIANTS, chunk=1000)
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1)

    for _ in range(5):
        once()
    ts = [once() for _ in range(30)]
    print(json.dumps({"scene": scene, "scope": "grid build + gpu_score_window_multi (chunk=1000)", "repeat": 30,
                      "median_ms": float(np.median(ts)), "min_ms": float(np.min(ts))}))


def knn_ties(scene: str, out: Path | None):
    import laspy
    from scipy.spatial import cKDTree

    import deploy_scene as ds
    from preprocess import load_hsi, valid_vegetation_mask

    hsi_path, las_path = ds.scene_paths(scene)
    C = ds.CONTRACT
    raw, gt, wkt = load_hsi(hsi_path, C["target_bands"])
    mask = valid_vegetation_mask(raw, C["red_band_index"], C["nir_band_index"], C["ndvi_threshold"], C["nodata_epsilon"])
    _, eligible, _, _, _ = ds.run_las_pipeline(las_path, C["point_cloud_crs"], wkt, gt, ds.SPATIAL_CONTRACT, mask, raw.shape[1], raw.shape[2],
                                               C["point_neighbors"], ds.SAMPLES_PER_SCENE, ds.SEED)
    las = laspy.read(las_path)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    k = C["point_neighbors"] + 1
    tree = cKDTree(xyz, balanced_tree=False)
    d, nb = tree.query(xyz[eligible], k=k)
    d17, _ = tree.query(xyz[eligible], k=k + 1)
    set_changed = order_changed = 0
    for j, pi in enumerate(eligible):  # 暴力法：按 (距离², 下标) 排序，即"并列时取下标小的"
        diff = xyz - xyz[pi]
        d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
        canon = np.lexsort((np.arange(len(xyz)), d2))[:k]
        set_changed += set(canon.tolist()) != set(nb[j].tolist())
        order_changed += canon.tolist() != nb[j].tolist()
    res = {"scene": scene, "n_points": int(len(eligible)), "n_cloud_points": int(len(xyz)),
           "points_with_tie_inside_16_nearest": int((d[:, 1:] == d[:, :-1]).any(axis=1).sum()),
           "points_with_tie_at_15th_16th": int((d[:, k - 1] == d[:, k - 2]).sum()),
           "points_with_tie_at_16th_17th_cut": int((d17[:, k] == d17[:, k - 1]).sum()),
           "points_where_index_tiebreak_changes_neighbor_set": int(set_changed),
           "points_where_index_tiebreak_changes_neighbor_order": int(order_changed)}
    print(json.dumps(res))
    if out:
        out.write_text(json.dumps(res, indent=2))


def window_cover(scene: str, d: Path, out: Path | None):
    n_valid, n_pts, rows, cols = [int(x) for x in (d / "meta.txt").read_text().split()]
    mask = np.fromfile(d / "mask.u8", dtype=np.uint8).reshape(rows, cols).astype(bool)
    rr, cc = np.fromfile(d / "ref_rows.i32", dtype=np.int32), np.fromfile(d / "ref_cols.i32", dtype=np.int32)
    cover = np.zeros((rows, cols), bool)
    for r, c in zip(rr, cc):
        cover[max(0, r - 5):r + 6, max(0, c - 5):c + 6] = True
    needed = int((cover & mask).sum())
    res = {"scene": scene, "valid_pixels": n_valid, "valid_pixels_covered_by_match_windows": needed, "fraction": needed / n_valid}
    print(json.dumps(res))
    if out:
        out.write_text(json.dumps(res, indent=2))


ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "window-cover":
        window_cover(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]) if len(sys.argv) > 4 else None)
        sys.exit(0)
    if cmd == "knn-ties":
        knn_ties(sys.argv[2], Path(sys.argv[3]) if len(sys.argv) > 3 else None)
        sys.exit(0)
    if cmd == "matchbench-py":
        matchbench_py(sys.argv[2])
        sys.exit(0)
    if cmd == "deploy-check":
        deploy_check(sys.argv[2], Path(sys.argv[3]))
        sys.exit(0)
    if cmd == "prep-check":
        prep_check(sys.argv[2], Path(sys.argv[3]))
    else:
        {"latency-refs": latency_refs, "latency-check": latency_check}[cmd](Path(sys.argv[2]))
