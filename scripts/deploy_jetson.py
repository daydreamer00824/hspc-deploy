"""Jetson 部署入口：从原始文件（HSI + LAS）到跨模态匹配结果的完整整景链路。

对应 `scripts/jetson_scene_opt.py` 里累加对比的最终配置（数据链路留在 GPU 上），再加上文件读取与 LAS 段：

  load_hsi（GDAL）→ 标准化 + 有效像元掩膜 → GPU 上 gather HSI patch → TensorRT HSI（scene 引擎，异步）
  → LAS 读取 / 投影 / kNN 构造点云邻域（CPU；--overlap-las 时与 GPU 上的 HSI 推理重叠）
  → TensorRT PC（第二个 stream）→ GPU 上拼特征网格 + 跨模态窗口匹配 → 取回 1000 个点的匹配行列

与 x86 的 `deploy_scene.py`（不依赖 torch、CPU 构建 patch、NumPy 匹配）相比，这里依赖 torch 做设备内存与 GPU 计算，
面向 CPU/GPU 共享内存的 Jetson。计时方式同 `deploy_scene.py`：进程内首次整景单独报告（冷启动），之后 warmup 2 次 +
重复 10 次，报告各分段中位数；页缓存不清（文件 IO 是"文件缓存热"的数字）。

默认参数即最终配置：LAS 段与 GPU 推理重叠、kd 树构建与投影并行、kNN 查询用全部核。对照实验（板上 4 景、正反两轮）：
A（x86 最终链路 deploy_scene.py 原样）→ B（本链路顺序执行）→ C（+重叠）→ D（+多核查询）→ E（+并行建树），
每一步匹配结果逐位相同，见 results/jetson/e2e_full_summary.json、e2e_las_summary.json。

  python scripts/deploy_jetson.py --scene 24data/10.6/1                       # 最终配置 E
  python scripts/deploy_jetson.py --no-overlap-las --las-workers 1 --no-las-parallel   # 对照组 B

运行环境（Jetson）：conda `hspc-jetson`（gdal/pyproj/laspy/scipy + Jetson 版 torch + TensorRT 绑定 + polygraphy），
需要 `HSPC_DATA_ROOT` 指向原始数据根目录、`PROJ_DATA`/`GDAL_DATA` 指向环境的 share 目录。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from deploy_scene import (  # noqa: E402
    CONTRACT, SAMPLES_PER_SCENE, SEED, SPATIAL_CONTRACT, LiteTrtRunner, run_las_pipeline, scene_paths,
)
from matching import VARIANTS, combine_score  # noqa: E402
from preprocess import load_hsi, valid_vegetation_mask  # noqa: E402

ENGINE_DIR = ROOT / "engines"
INPUT_DIMS = {"hsi": (342, 3, 3), "pc": (15, 3)}
WARMUP, REPEAT = 2, 10
DEV = "cuda"
_TORCH_DT = {np.dtype(np.float32): torch.float32, np.dtype(np.float16): torch.float16}


# ---------------------------------------------------------------- GPU 算子
def normalize_on_gpu(raw: np.ndarray, stream: torch.cuda.Stream):
    """与 preprocess.standardize_hsi 逐位相同：均值/标准差的归约仍用 NumPy（归约顺序与 GPU 不同，结果不能保证逐位一致），
    只把逐元素的 (raw - mean) / std 放到 GPU 上（IEEE 逐元素减法/除法结果确定）。"""
    mean = raw.mean(axis=(1, 2), keepdims=True)
    std = raw.std(axis=(1, 2), keepdims=True) + 1e-8
    with torch.cuda.stream(stream):
        return ((torch.from_numpy(raw).to(DEV) - torch.from_numpy(mean).to(DEV))
                / torch.from_numpy(std.astype(np.float32)).to(DEV))


def gather_patches(normalized_t: torch.Tensor, vr_t: torch.Tensor, vc_t: torch.Tensor, patch_size: int = 3) -> torch.Tensor:
    """(bands, rows, cols) 设备张量 → (N, bands, p, p)。四周补 p//2 圈零，与 preprocess.extract_patch 的零填充一致，纯拷贝。"""
    half = patch_size // 2
    cube = torch.nn.functional.pad(normalized_t, (half, half, half, half))
    ar = torch.arange(patch_size, device=normalized_t.device)
    patches = cube[:, vr_t[:, None, None] + ar[None, :, None], vc_t[:, None, None] + ar[None, None, :]]
    return patches.permute(1, 0, 2, 3).contiguous()


def gpu_score_window_multi(point_features, feature_grid, valid_mask, ref_rows, ref_cols, search_radius,
                           spatial_weights: dict, chunk: int = 256, single_transfer: bool = False) -> dict:
    """deploy_scene.numpy_score_window_multi 的 GPU 版：网格四周补 search_radius 圈零并把掩膜补 False，这样每个点的窗口都是固定的
    (2r+1)x(2r+1)，可以按 chunk 个点一批 gather + bmm；窗口外/无效像元的分数置 -inf，行优先 argmax 取第一个最大值，
    与逐点循环的裁剪窗口取法等价。feature_grid / point_features / valid_mask 可以是 numpy 数组或 cuda 张量。
    """
    dev, r = DEV, search_radius
    rows, cols, dim = feature_grid.shape
    g = feature_grid if torch.is_tensor(feature_grid) else torch.from_numpy(feature_grid).to(dev)
    g = g / (torch.linalg.norm(g, dim=-1, keepdim=True) + 1e-12)
    gp = torch.nn.functional.pad(g, (0, 0, r, r, r, r))
    vm = valid_mask if torch.is_tensor(valid_mask) else torch.from_numpy(valid_mask).to(dev)
    mp = torch.nn.functional.pad(vm, (r, r, r, r), value=False)
    pf = point_features if torch.is_tensor(point_features) else torch.from_numpy(point_features).to(dev)
    pf = pf / (torch.linalg.norm(pf, dim=-1, keepdim=True) + 1e-12)
    offs = torch.arange(-r, r + 1, device=dev)
    w = 2 * r + 1
    distance = torch.sqrt(offs.float()[:, None] ** 2 + offs.float()[None, :] ** 2).reshape(-1)
    rr_all = torch.from_numpy(np.asarray(ref_rows)).to(dev)
    cc_all = torch.from_numpy(np.asarray(ref_cols)).to(dev)
    n = len(ref_rows)
    res = {v: {"rows": torch.full((n,), -1, dtype=torch.int64, device=dev),
               "cols": torch.full((n,), -1, dtype=torch.int64, device=dev),
               "cosine": torch.full((n,), float("nan"), device=dev),
               "pixel_displacement": torch.full((n,), float("nan"), device=dev)} for v in spatial_weights}
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        rr = rr_all[sl][:, None] + offs[None, :] + r          # 补边后的行坐标 (m, w)
        cc = cc_all[sl][:, None] + offs[None, :] + r
        win = gp[rr[:, :, None], cc[:, None, :]].reshape(rr.shape[0], w * w, dim)
        mask = mp[rr[:, :, None], cc[:, None, :]].reshape(rr.shape[0], w * w)
        cosine = torch.bmm(win, pf[sl][:, :, None]).squeeze(-1)
        cosine = torch.where(mask, cosine, torch.tensor(float("-inf"), device=dev))
        for v, sw in spatial_weights.items():
            score = torch.where(mask, combine_score(cosine, distance[None, :], sw, search_radius),
                                torch.tensor(float("-inf"), device=dev))
            ok = torch.isfinite(score).any(dim=1)
            best = torch.argmax(score, dim=1)
            mr = rr_all[sl] + offs[best // w]
            mc = cc_all[sl] + offs[best % w]
            R = res[v]
            R["rows"][sl] = torch.where(ok, mr, R["rows"][sl])
            R["cols"][sl] = torch.where(ok, mc, R["cols"][sl])
            R["cosine"][sl] = torch.where(ok, cosine.gather(1, best[:, None]).squeeze(1), R["cosine"][sl])
            R["pixel_displacement"][sl] = torch.where(
                ok, torch.hypot((mr - rr_all[sl]).float(), (mc - cc_all[sl]).float()), R["pixel_displacement"][sl])
    out = {}
    if single_transfer:  # 所有结果拼成一个张量，一次 D2H（逐位不变，只减少同步次数）
        keys = ("rows", "cols", "cosine", "pixel_displacement")
        packed = torch.stack([R[k].double() for R in res.values() for k in keys]).cpu().numpy()
        for vi, v in enumerate(res):
            out[v] = {k: packed[vi * len(keys) + ki].astype(np.int64 if k in ("rows", "cols") else np.float32)
                      for ki, k in enumerate(keys)}
            out[v]["n_none"] = int((out[v]["rows"] < 0).sum())
        return out
    for v, R in res.items():
        out[v] = {k: x.cpu().numpy() for k, x in R.items()}
        out[v]["n_none"] = int((out[v]["rows"] < 0).sum())
    return out


# ---------------------------------------------------------------- 整景链路
class Pipeline:
    def __init__(self, hsi_plan: Path, pc_plan: Path, max_batch: int, match_chunk: int = 1000):
        self.hsi = LiteTrtRunner(hsi_plan, INPUT_DIMS["hsi"], max_batch=max_batch)
        self.pc = LiteTrtRunner(pc_plan, INPUT_DIMS["pc"], max_batch=max_batch)
        assert np.dtype(self.hsi.in_dtype) == np.float32 and np.dtype(self.pc.in_dtype) == np.float32
        self.sa, self.sb = torch.cuda.Stream(), torch.cuda.Stream()  # 主流水线 / PC 推理
        self.match_chunk = match_chunk

    def run(self, hsi_path: Path, las_path: Path, overlap_las: bool, las_workers: int = 1, las_parallel: bool = False):
        t = {}

        def mark(name, t0):
            t[name] = time.perf_counter() - t0
            return time.perf_counter()

        t0 = time.perf_counter()
        raw, gt, proj = load_hsi(hsi_path, CONTRACT["target_bands"])
        t0 = mark("load_hsi_io", t0)
        bands, rows, cols = raw.shape
        valid_mask = valid_vegetation_mask(raw, CONTRACT["red_band_index"], CONTRACT["nir_band_index"],
                                           CONTRACT["ndvi_threshold"], CONTRACT["nodata_epsilon"])
        vr, vc = np.where(valid_mask)
        n = len(vr)
        norm_t = normalize_on_gpu(raw, self.sa)
        t0 = mark("standardize_and_mask", t0)

        with torch.cuda.stream(self.sa):
            vr_t, vc_t = torch.from_numpy(vr).to(DEV), torch.from_numpy(vc).to(DEV)
            patches = gather_patches(norm_t, vr_t, vc_t, CONTRACT["patch_size"])
            y = torch.empty((n, 1024), dtype=_TORCH_DT[np.dtype(self.hsi.out_dtype)], device=DEV)
            self.hsi.infer_ptrs(patches.data_ptr(), y.data_ptr(), n, self.sa.cuda_stream)
        # 发射后即可释放：这些块只会被同一 stream（sa）上后续的分配复用，而那些操作排在 HSI 推理之后，读写不会冲突。
        # 及时释放能把整景显存峰值压低约 140MB（Jetson 上 CPU/GPU 共用 8GB）。
        del patches, norm_t
        if not overlap_las:  # 顺序执行：等 HSI 推理完成再做 LAS
            self.sa.synchronize()
        t0 = mark("hsi_patch_and_infer" if not overlap_las else "hsi_patch_and_launch", t0)

        offsets, eligible, ref_rows, ref_cols, las_t = run_las_pipeline(
            las_path, CONTRACT["point_cloud_crs"], proj, gt, SPATIAL_CONTRACT, valid_mask, rows, cols,
            CONTRACT["point_neighbors"], SAMPLES_PER_SCENE, SEED,
            query_workers=las_workers, parallel_tree_build=las_parallel)
        t.update(las_t)
        t0 = time.perf_counter()

        n_pc = len(offsets)
        with torch.cuda.stream(self.sb):
            px = torch.from_numpy(offsets).to(DEV)
            py = torch.empty((n_pc, 1024), dtype=_TORCH_DT[np.dtype(self.pc.out_dtype)], device=DEV)
            self.pc.infer_ptrs(px.data_ptr(), py.data_ptr(), n_pc, self.sb.cuda_stream)
        if overlap_las:
            self.sa.synchronize()  # HSI 推理大部分已在 LAS 期间完成，这里等它收尾
            t0 = mark("hsi_wait", t0)
        with torch.cuda.stream(self.sa):
            self.sa.wait_stream(self.sb)
            grid = torch.zeros((rows, cols, 1024), dtype=torch.float32, device=DEV)
            grid[vr_t, vc_t] = y.float()
            match = gpu_score_window_multi(py.float(), grid, valid_mask, ref_rows, ref_cols, CONTRACT["search_radius"],
                                           VARIANTS, chunk=self.match_chunk)
        t0 = mark("pc_infer_assemble_match", t0)
        del px
        out = {"n_valid": int(n), "n_points": int(n_pc), "match": {v: {"rows": m["rows"], "cols": m["cols"]}
                                                                   for v, m in match.items()},
               "offsets": offsets, "ref_rows": ref_rows, "ref_cols": ref_cols, "valid_mask": valid_mask}
        return out, t


IO_SEGMENTS = {"load_hsi_io", "laspy_read_io"}


def peak_rss_kb() -> int:
    """进程常驻内存峰值（/proc/self/status 的 VmHWM）。Jetson 上 CUDA 分配走 NvMap，不一定计入，整板内存另见 tegrastats。"""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="24data/10.6/1", help="批次/日期/样地号")
    ap.add_argument("--hsi-engine", default="engines/hsi_fp16_scene1024.plan")
    ap.add_argument("--pc-engine", default="engines/pc_fp16_scene1024.plan")
    ap.add_argument("--max-batch", type=int, default=1024)
    # 默认即最终配置（results/jetson/e2e_las_summary.json 的 E）；以下开关只用于对照实验
    ap.add_argument("--no-overlap-las", dest="overlap_las", action="store_false",
                    help="关闭：HSI 推理异步发射后立刻做 LAS 段、不等 GPU（对照组 B）")
    ap.add_argument("--las-workers", type=int, default=-1, help="kNN 查询线程数（默认 -1 = 全部核；对照组 C 为 1）")
    ap.add_argument("--no-las-parallel", dest="las_parallel", action="store_false",
                    help="关闭：kd 树构建放后台线程、与投影并行（对照组 C/D）")
    ap.add_argument("--check-cache", default=None,
                    help="护栏：给出该景的缓存 npz 时，核对本链路的前处理产物与缓存逐位相同")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--repeat", type=int, default=REPEAT)
    ap.add_argument("--seconds", type=float, default=0, help=">0：整景连续跑够这么多秒（能效测量用），代替固定次数")
    ap.add_argument("--cold-only", action="store_true", help="只跑进程内首次整景（冷启动测量用）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.warmup < 0 or a.repeat < 1 or a.seconds < 0:
        ap.error("need --warmup >= 0, --repeat >= 1, --seconds >= 0")  # argparse 以退出码 2 结束
    hsi_path, las_path = scene_paths(a.scene)

    t0 = time.perf_counter()
    pipe = Pipeline(ROOT / a.hsi_engine, ROOT / a.pc_engine, a.max_batch)
    torch.cuda.synchronize()
    engine_load_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    cold_out, cold_t = pipe.run(hsi_path, las_path, a.overlap_las, a.las_workers, a.las_parallel)
    torch.cuda.synchronize()
    cold_total = time.perf_counter() - t0
    if a.cold_only:
        out = Path(a.out) if a.out else ROOT / "results/jetson/cold_only.json"
        out.write_text(json.dumps({"scene": a.scene, "engine_load_sec": engine_load_sec, "cold_start": {"total_sec": cold_total},
                                   "peak_rss_kb": peak_rss_kb()}))
        return
    for _ in range(a.warmup):
        pipe.run(hsi_path, las_path, a.overlap_las, a.las_workers, a.las_parallel)
    # 只保留"参照结果"（第一次热身后）和"最后一次结果"，逐次比较、不攒下整个数组——这样 --seconds 长时间模式
    # （能效测量，可能几十次）也能对"每一次"都做确定性检查，而不是只在结尾比较剩下的一两次。
    totals, seg_lists = [], {}
    ref_match, last, determinism = None, None, {v: True for v in VARIANTS}
    wall0, loop0 = time.time(), time.perf_counter()
    i = 0
    while i == 0 or ((time.perf_counter() - loop0 < a.seconds) if a.seconds > 0 else (i < a.repeat)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, t = pipe.run(hsi_path, las_path, a.overlap_las, a.las_workers, a.las_parallel)
        torch.cuda.synchronize()
        totals.append(time.perf_counter() - t0)
        for k, v in t.items():
            seg_lists.setdefault(k, []).append(v)
        base = cold_out["match"] if ref_match is None else ref_match
        for v in VARIANTS:
            determinism[v] = determinism[v] and bool(np.array_equal(base[v]["rows"], out["match"][v]["rows"]) and
                                                     np.array_equal(base[v]["cols"], out["match"][v]["cols"]))
        if ref_match is None:
            ref_match = out["match"]
        last = out
        i += 1
    wall1 = time.time()
    seg_ms = {k: float(np.median(v) * 1000) for k, v in seg_lists.items()}
    io_ms = sum(v for k, v in seg_ms.items() if k in IO_SEGMENTS)
    report = {
        "scene": a.scene, "overlap_las": a.overlap_las, "las_workers": a.las_workers, "las_parallel": a.las_parallel, "hsi_engine": a.hsi_engine, "pc_engine": a.pc_engine,
        "max_batch": a.max_batch, "n_valid_pixels": last["n_valid"], "n_points": last["n_points"],
        "warmup": a.warmup, "repeat": a.repeat, "engine_load_sec": engine_load_sec,
        "cold_start": {"note": "进程内首次整景（含 CUDA/PROJ 初始化）；页缓存未清", "total_sec": cold_total,
                       "segments_ms": {k: v * 1000 for k, v in cold_t.items()}},
        "warm": {"n_timed": len(totals), "t_start_epoch": wall0, "t_end_epoch": wall1,
                 "total_ms_median": float(np.median(totals) * 1000), "total_ms_all": [x * 1000 for x in totals],
                 "segments_ms_median": seg_ms, "io_ms_median_sum": io_ms,
                 "total_without_io_ms_median": float(np.median(totals) * 1000) - io_ms},
        "determinism_match_rowcols_identical_across_reps": determinism, "peak_rss_kb": peak_rss_kb(),
        "match_rowcols": {v: {"rows": m["rows"].tolist(), "cols": m["cols"].tolist()} for v, m in last["match"].items()},
    }
    if a.check_cache:
        d = np.load(a.check_cache)
        report["guardrail_vs_cache"] = {
            "valid_mask": bool(np.array_equal(last["valid_mask"], d["valid_mask"])),
            "point_offsets": bool(np.array_equal(last["offsets"], d["point_offsets"])),
            "ref_rows": bool(np.array_equal(last["ref_rows"], d["ref_rows"])),
            "ref_cols": bool(np.array_equal(last["ref_cols"], d["ref_cols"])),
        }
    out = Path(a.out) if a.out else ROOT / f"results/jetson/e2e_full_{a.scene.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("match_rowcols",)}, ensure_ascii=False, indent=2))
    print("written ->", out)


if __name__ == "__main__":
    main()
