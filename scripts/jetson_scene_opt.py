"""Jetson 整景推理加速优化（O0～O6）：单进程、单变量累加式对比。

子命令：
  profile <配置>   跑一个配置：warmup 后只对 1 次完整整景开启 cudaProfiler 区间，供 nsys 采时间线（O0）
  sweep            scene 大 profile 引擎档位扫描（O2）：构建、特征精度、整景 HSI/PC 推理耗时、显存占用
  compare          累加式配置 S0→Sk 正反各一轮的整景耗时对比 + 精度护栏（O6）

范围：从缓存的 hsi_raw 起（标准化、掩膜、patch 构建、推理、拼网格、匹配），不含文件 IO 和 LAS 段
（板上没有 GDAL/laspy/pyproj）。点云特征直接用缓存的 point_offsets、ref_rows/ref_cols。

配置（每个只比上一个多改一处）：
  S0 全网格 15840 patch + torch TrtRunner + batch64 引擎 + torch 逐点匹配（=J4 的写法）
  S1 只推理有效像元
  S2 匹配换成 numpy 合并版（numpy_score_window_multi）
  S3 runner 换成 LiteTrtRunner（buffer 复用）
  S4 引擎换成 scene 大 profile（--tier 指定）

所有耗时都是墙钟，含 CPU 前处理段，warmup 2 次 + 重复 10 次取中位数。
运行环境（Jetson）：/usr/bin/python3，PYTHONPATH=pylibs:scripts（pylibs 里只有 polygraphy）。
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_trt as bt  # noqa: E402
import hspc_common as hc  # noqa: E402
from deploy_jetson import gpu_score_window_multi  # noqa: E402  最终部署入口里的 GPU 匹配，实验与部署用同一份代码
from deploy_scene import CONTRACT, LiteTrtRunner, PinnedArray, numpy_score_window_multi  # noqa: E402
from e2e_stage_b_infer import score_window_batch  # noqa: E402
from matching import VARIANTS  # noqa: E402
from preprocess import build_hsi_patches, standardize_hsi, valid_vegetation_mask  # noqa: E402
from stage7_promotion_checks import REF_MARGIN_P10, SEARCH_RADIUS, score_window_with_margin  # noqa: E402
from trt_runner import TrtRunner  # noqa: E402

ENGINE_DIR = ROOT / "engines"
RESULTS = ROOT / "results"
INPUT_DIMS = {"hsi": (342, 3, 3), "pc": (15, 3)}
WARMUP, REPEAT = 2, 10
NEAR_TIE_FACTOR = 3

CONFIGS = {
    "S0": dict(valid_only=False, runner="torch", engine="b64", match="torch"),
    "S1": dict(valid_only=True, runner="torch", engine="b64", match="torch"),
    "S2": dict(valid_only=True, runner="torch", engine="b64", match="numpy"),
    "S3": dict(valid_only=True, runner="lite", engine="b64", match="numpy"),
    "S4": dict(valid_only=True, runner="lite", engine="scene", match="numpy"),
    "S5": dict(valid_only=True, runner="lite", engine="scene", match="numpy", zero_copy=True),  # S4 + 统一内存零拷贝
    "S6": dict(valid_only=True, runner="lite", engine="scene", match="gpu", zero_copy=True),  # S5 + GPU 批量匹配
    # S7+ 走 run_scene_gpu：整条数据链路留在 GPU
    "S7": dict(valid_only=True, runner="lite", engine="scene", match="gpu", gpu_patch=True),  # S6 + GPU 上提取 patch，直接作为 TRT 输入
    "S8": dict(valid_only=True, runner="lite", engine="scene", match="gpu", gpu_patch=True, gpu_resident=True),  # + 特征留在 GPU（拼网格/匹配不经 host）
    "S9": dict(valid_only=True, runner="lite", engine="scene", match="gpu", gpu_patch=True, gpu_resident=True,
               pc_async=True),  # + PC 推理放到第二个 stream，与 CPU 标准化重叠
    "S10": dict(valid_only=True, runner="lite", engine="scene", match="gpu", gpu_patch=True, gpu_resident=True,
                pc_async=True, match_chunk=1000),  # + 匹配不分块（1000 点一次 gather+bmm，少下发，结果逐位相同）
    "S11": dict(valid_only=True, runner="lite", engine="scene", match="gpu", gpu_patch=True, gpu_resident=True,
                pc_async=True, match_chunk=1000, gpu_prep=True),  # + 标准化与掩膜在 GPU 上算（不再逐位相同，采纳条件见 PROGRESS）
    "S11b": dict(valid_only=True, runner="lite", engine="scene", match="gpu", gpu_patch=True, gpu_resident=True,
                 pc_async=True, match_chunk=1000, gpu_prep="hybrid"),  # S10 + 归约在 CPU（NumPy，逐位不变）、逐元素标准化在 GPU
    "S12": dict(valid_only=True, runner="lite", engine="scene_tf32", match="gpu", gpu_patch=True, gpu_resident=True,
                pc_async=True, match_chunk=1000, gpu_prep=True),  # + HSI 引擎的 FP32 层（stem）允许 TF32
    "S10T": dict(valid_only=True, runner="lite", engine="scene_tf32", match="gpu", gpu_patch=True, gpu_resident=True,
                 pc_async=True, match_chunk=1000),  # S10 + TF32 stem（S11 不采纳时用它单独评估 TF32）
}


# ---------------------------------------------------------------- GPU 上的标准化与掩膜（S11，匹配函数已移到 deploy_jetson.py）
def gpu_standardize_and_mask(raw_t: torch.Tensor):
    """preprocess.standardize_hsi + valid_vegetation_mask 的 GPU 版（逻辑逐条对应）。归约顺序与 NumPy 不同，
    标准化结果不保证逐位相同；掩膜是逐元素比较，护栏要求逐位相同。"""
    mean = raw_t.mean(dim=(1, 2), keepdim=True)
    std = raw_t.std(dim=(1, 2), keepdim=True, correction=0) + 1e-8
    normalized = (raw_t - mean) / std
    red, nir = raw_t[CONTRACT["red_band_index"]], raw_t[CONTRACT["nir_band_index"]]
    denom = nir + red
    ndvi = torch.where(denom.abs() > 1e-12, (nir - red) / denom, torch.zeros_like(denom))
    mask = (ndvi > CONTRACT["ndvi_threshold"]) & (raw_t.sum(dim=0) > CONTRACT["nodata_epsilon"])
    return normalized, mask


# ---------------------------------------------------------------- 数据与分段计时
SCENE_NPZ = RESULTS / "e2e_scene_preprocessed.npz"  # --scene-npz 可换成 results/scenes/<景>_preprocessed.npz


def load_scene() -> dict:
    d = np.load(SCENE_NPZ)
    keys = ("hsi_raw", "valid_mask", "point_offsets", "ref_rows", "ref_cols", "hsi_grid_patches")
    s = {k: d[k] for k in keys}
    s["rows"], s["cols"] = int(d["hsi_rows"]), int(d["hsi_cols"])
    return s


class Seg:
    """分段计时；同时打 NVTX 标记，nsys 时间线上能看到每段。"""

    def __init__(self):
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, name):
        torch.cuda.nvtx.range_push(name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = time.perf_counter() - t0
            torch.cuda.nvtx.range_pop()


def plan_path(which: str, engine: str, tier: int | None) -> Path:
    if engine == "b64":
        return ENGINE_DIR / f"{which}_fp16.plan"
    if engine == "scene_tf32" and which == "hsi":  # PC 引擎没有 FLOAT 层，TF32 与否无区别，沿用 scene 引擎
        return ENGINE_DIR / f"hsi_fp16_scene{tier}_tf32.plan"
    return ENGINE_DIR / f"{which}_fp16_scene{tier}.plan"


class Runners:
    """按 (runner 类型, 引擎) 缓存，进程内复用。"""

    def __init__(self, tier: int | None):
        self.tier, self._c = tier, {}
        self.stream_a, self.stream_b = torch.cuda.Stream(), torch.cuda.Stream()  # S7+ 用：主流水线 / PC 并发

    def get(self, which: str, cfg: dict):
        key = (which, cfg["runner"], cfg["engine"])
        if key not in self._c:
            p = plan_path(which, cfg["engine"], self.tier)
            mb = 64 if cfg["engine"] == "b64" else self.tier
            self._c[key] = (TrtRunner(p) if cfg["runner"] == "torch"
                            else LiteTrtRunner(p, INPUT_DIMS[which], max_batch=mb))
        return self._c[key]

    def mapped(self, name: str, shape: tuple, dtype) -> PinnedArray:
        """零拷贝用的映射 host 内存，按名字缓存；需要更大时重新分配。"""
        cur = self._c.get(("mapped", name))
        if cur is None or cur.array.shape[0] < shape[0]:
            if cur is not None:
                cur.free()
            cur = self._c[("mapped", name)] = PinnedArray(shape, dtype, mapped=True)
        return cur

    def infer(self, which: str, cfg: dict, x: np.ndarray) -> np.ndarray:
        r = self.get(which, cfg)
        if cfg["runner"] == "torch":
            return r.infer_batched(x, max_batch=64 if cfg["engine"] == "b64" else self.tier)
        return r.infer_all(x)


def run_scene(cfg: dict, s: dict, runners: Runners, keep: bool = False):
    """整景跑一次，返回 (输出, 各段耗时秒)。"""
    seg = Seg()
    rows, cols = s["rows"], s["cols"]
    with seg("prep"):  # 标准化 + 有效像元掩膜
        normalized, _, _ = standardize_hsi(s["hsi_raw"])
        valid_mask = valid_vegetation_mask(s["hsi_raw"], CONTRACT["red_band_index"], CONTRACT["nir_band_index"],
                                           CONTRACT["ndvi_threshold"], CONTRACT["nodata_epsilon"])
        vr, vc = np.where(valid_mask)
    zero_copy = cfg.get("zero_copy", False)
    with seg("patch_build"):
        if cfg["valid_only"]:
            rc = list(zip(vr.tolist(), vc.tolist()))
        else:
            rc = [(r, c) for r in range(rows) for c in range(cols)]
        if zero_copy:  # patch 直接写进映射内存，GPU 不再需要 H2D
            hr = runners.get("hsi", cfg)
            x_map = runners.mapped("hsi_in", (len(rc), *INPUT_DIMS["hsi"]), hr.in_dtype)
            patches = build_hsi_patches(normalized, rc, CONTRACT["patch_size"], out=x_map.array[:len(rc)])
        else:
            patches = build_hsi_patches(normalized, rc, CONTRACT["patch_size"])
    with seg("hsi_infer"):
        if zero_copy:
            y_map = runners.mapped("hsi_out", (len(rc), 1024), hr.out_dtype)
            hr.infer_all_zero_copy(x_map, len(rc), y_map)
            hsi_feats = y_map.array[:len(rc)]
        else:
            hsi_feats = runners.infer("hsi", cfg, patches)
    with seg("pc_infer"):
        if zero_copy:
            pr = runners.get("pc", cfg)
            px = runners.mapped("pc_in", s["point_offsets"].shape, pr.in_dtype)
            px.array[:] = s["point_offsets"]
            py = runners.mapped("pc_out", (len(s["point_offsets"]), 1024), pr.out_dtype)
            pr.infer_all_zero_copy(px, len(s["point_offsets"]), py)
            pc_feats = py.array[:len(s["point_offsets"])].astype(np.float32)
        else:
            pc_feats = runners.infer("pc", cfg, s["point_offsets"])
    with seg("assemble"):
        if cfg["valid_only"]:
            grid = np.zeros((rows, cols, hsi_feats.shape[1]), dtype=np.float32)
            grid[vr, vc] = hsi_feats
            feats_valid = hsi_feats.astype(np.float32) if keep else None  # 仅护栏用，不计入耗时
        else:
            grid = hsi_feats.reshape(rows, cols, -1)
            feats_valid = (hsi_feats.reshape(rows * cols, -1)[np.flatnonzero(valid_mask.ravel())] if keep else None)
        if cfg["match"] == "torch":
            grid_t, pf_t = torch.from_numpy(grid).float(), torch.from_numpy(pc_feats).float()
    with seg("match"):
        if cfg["match"] == "torch":
            match = {v: score_window_batch(pf_t, grid_t, valid_mask, s["ref_rows"], s["ref_cols"],
                                           SEARCH_RADIUS, sw) for v, sw in VARIANTS.items()}
        elif cfg["match"] == "gpu":
            match = gpu_score_window_multi(pc_feats, grid, valid_mask, s["ref_rows"], s["ref_cols"],
                                           SEARCH_RADIUS, VARIANTS)
        else:
            match = numpy_score_window_multi(pc_feats, grid, valid_mask, s["ref_rows"], s["ref_cols"],
                                             SEARCH_RADIUS, VARIANTS)
    out = {"match": {v: {"rows": m["rows"], "cols": m["cols"]} for v, m in match.items()},
           "hsi_feats_valid": feats_valid, "pc_feats": pc_feats.astype(np.float32) if keep else None,
           "n_patches": len(patches)}
    return out, seg.t


_TORCH_DT = {np.dtype(np.float32): torch.float32, np.dtype(np.float16): torch.float16}


def run_scene_gpu(cfg: dict, s: dict, runners: Runners, keep: bool = False):
    """S7+：整条数据链路留在 GPU。CPU 只做标准化 + 掩膜；标准化立方体上传后在 GPU 上 gather 出 patch（补零方式与
    preprocess.extract_patch 相同），直接作为 TRT 输入地址；resident 时特征也留在设备上拼网格、匹配，最后只拿回
    1000 个点的结果。"""
    seg = Seg()
    dev = "cuda"
    sa, sb = runners.stream_a, runners.stream_b
    rows, cols = s["rows"], s["cols"]
    hr, pr = runners.get("hsi", cfg), runners.get("pc", cfg)
    n_pc = len(s["point_offsets"])
    resident, pc_async = cfg.get("gpu_resident", False), cfg.get("pc_async", False)
    pc_dt = _TORCH_DT[np.dtype(pr.out_dtype)]
    assert np.dtype(hr.in_dtype) == np.float32, "GPU gather 产出 float32 patch，引擎输入 dtype 必须一致"
    assert not cfg.get("gpu_prep") or cfg.get("gpu_resident"), "gpu_prep 只支持 gpu_resident 配置"

    def launch_pc(stream):
        with torch.cuda.stream(stream):
            px = torch.from_numpy(s["point_offsets"]).to(dev)
            py = torch.empty((n_pc, 1024), dtype=pc_dt, device=dev)
            pr.infer_ptrs(px.data_ptr(), py.data_ptr(), n_pc, stream.cuda_stream)
        return px, py  # px 必须活到推理结束

    if pc_async:  # PC 不依赖 HSI，先在第二个 stream 上发射，与后面的 CPU 标准化重叠
        with seg("pc_infer"):
            px, py = launch_pc(sb)
    gpu_prep = cfg.get("gpu_prep", False)
    with seg("prep"):
        if gpu_prep == "hybrid":  # S11b：均值/标准差/掩膜的归约仍用 NumPy（逐位不变），只把逐元素 (raw-mean)/std 放到 GPU
            raw = s["hsi_raw"]
            mean = raw.mean(axis=(1, 2), keepdims=True)
            std = raw.std(axis=(1, 2), keepdims=True) + 1e-8
            valid_mask = valid_vegetation_mask(raw, CONTRACT["red_band_index"], CONTRACT["nir_band_index"],
                                               CONTRACT["ndvi_threshold"], CONTRACT["nodata_epsilon"])
            vr, vc = np.where(valid_mask)
            n = len(vr)
            with torch.cuda.stream(sa):
                norm_t = ((torch.from_numpy(raw).to(dev) - torch.from_numpy(mean).to(dev))
                          / torch.from_numpy(std.astype(np.float32)).to(dev))
                vr_t, vc_t = torch.from_numpy(vr).to(dev), torch.from_numpy(vc).to(dev)
        elif gpu_prep:  # S11：原始立方体上传一次，标准化与掩膜都在 GPU 上算
            with torch.cuda.stream(sa):
                raw_t = torch.from_numpy(s["hsi_raw"]).to(dev)
                norm_t, mask_t = gpu_standardize_and_mask(raw_t)
                vrc = torch.nonzero(mask_t)  # 行优先，与 np.where 同序
                vr_t, vc_t = vrc[:, 0].contiguous(), vrc[:, 1].contiguous()
            n = int(vr_t.shape[0])  # 需要 n 分配输出，这里同步一次
            valid_mask = mask_t
        else:
            normalized, _, _ = standardize_hsi(s["hsi_raw"])
            valid_mask = valid_vegetation_mask(s["hsi_raw"], CONTRACT["red_band_index"], CONTRACT["nir_band_index"],
                                               CONTRACT["ndvi_threshold"], CONTRACT["nodata_epsilon"])
            vr, vc = np.where(valid_mask)
            n = len(vr)
    with seg("patch_build"):  # 上传立方体 + GPU gather（同步以便归因到这一段）
        with torch.cuda.stream(sa):
            if gpu_prep:  # S11 / S11b：标准化结果已在 GPU 上
                cube = torch.nn.functional.pad(norm_t, (1, 1, 1, 1))
            else:
                cube = torch.nn.functional.pad(torch.from_numpy(normalized).to(dev), (1, 1, 1, 1))  # 补零 1 圈
                vr_t, vc_t = torch.from_numpy(vr).to(dev), torch.from_numpy(vc).to(dev)
            ar = torch.arange(3, device=dev)
            patches = cube[:, vr_t[:, None, None] + ar[None, :, None], vc_t[:, None, None] + ar[None, None, :]]
            patches = patches.permute(1, 0, 2, 3).contiguous()  # (N, 342, 3, 3)
        sa.synchronize()
    with seg("hsi_infer"):
        if resident:
            y = torch.empty((n, 1024), dtype=_TORCH_DT[np.dtype(hr.out_dtype)], device=dev)
            with torch.cuda.stream(sa):
                hr.infer_ptrs(patches.data_ptr(), y.data_ptr(), n, sa.cuda_stream)
            sa.synchronize()
            # 推理完就释放 patch（119MB）和补边立方体：Jetson 上 CPU/GPU 共用 8GB，不及时释放会和匹配段的窗口 gather 叠加，
            # 在空闲内存偏低时触发 NvMap 分配失败（见 PROGRESS）。只影响显存占用，不影响任何计算结果。
            del patches, cube
        else:  # S7：输出仍写到映射 host 内存（与 S6 相同），只把输入换成设备张量
            y_map = runners.mapped("hsi_out", (n, 1024), hr.out_dtype)
            with torch.cuda.stream(sa):
                hr.infer_ptrs(patches.data_ptr(), y_map.dev_ptr, n, sa.cuda_stream)
            sa.synchronize()
    if not pc_async:
        with seg("pc_infer"):
            if resident:
                px, py = launch_pc(sa)
                sa.synchronize()
            else:  # S7：与 S6 相同的映射内存 PC 路径
                pxm = runners.mapped("pc_in", s["point_offsets"].shape, pr.in_dtype)
                pxm.array[:] = s["point_offsets"]
                pym = runners.mapped("pc_out", (n_pc, 1024), pr.out_dtype)
                pr.infer_all_zero_copy(pxm, n_pc, pym)
                pc_feats = pym.array[:n_pc].astype(np.float32)
    with seg("assemble"):
        if resident:
            with torch.cuda.stream(sa):
                grid = torch.zeros((rows, cols, 1024), dtype=torch.float32, device=dev)
                grid[vr_t, vc_t] = y.float()
                if pc_async:
                    sa.wait_stream(sb)  # PC 推理已在第二个 stream 上跑完/排队，主流水线等它
                pf = py.float()
        else:
            hsi_feats = y_map.array[:n]
            grid = np.zeros((rows, cols, 1024), dtype=np.float32)
            grid[vr, vc] = hsi_feats
            pf = pc_feats
    with seg("match"):
        with torch.cuda.stream(sa):
            match = gpu_score_window_multi(pf, grid, valid_mask, s["ref_rows"], s["ref_cols"], SEARCH_RADIUS, VARIANTS,
                                           chunk=cfg.get("match_chunk", 256))
    out = {"match": {v: {"rows": m["rows"], "cols": m["cols"]} for v, m in match.items()},
           "hsi_feats_valid": None, "pc_feats": None, "n_patches": n}
    if keep:
        out["valid_mask"] = valid_mask.cpu().numpy() if torch.is_tensor(valid_mask) else valid_mask
        out["hsi_feats_valid"] = (y.float().cpu().numpy() if resident else y_map.array[:n].astype(np.float32))
        out["pc_feats"] = (py.float().cpu().numpy() if resident else pc_feats)
    return out, seg.t


def run_any(cfg: dict, s: dict, runners: Runners, keep: bool = False):
    return (run_scene_gpu if cfg.get("gpu_patch") else run_scene)(cfg, s, runners, keep)


def timed_config(cfg: dict, s: dict, runners: Runners):
    """warmup + repeat，返回 (首次计时的输出, 每段中位数 ms, 总耗时中位数 ms, 各次总耗时 ms)。"""
    first = None
    for i in range(WARMUP):  # 最后一次 warmup 保留特征拷贝，供护栏用（拷贝不进计时）
        out, _ = run_any(cfg, s, runners, keep=(i == WARMUP - 1))
        first = out
    segs, totals = [], []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, t = run_any(cfg, s, runners)
        torch.cuda.synchronize()
        totals.append((time.perf_counter() - t0) * 1000)
        segs.append(t)
    med = {k: float(np.median([t[k] for t in segs]) * 1000) for k in segs[0]}
    return first, med, float(np.median(totals)), totals


# ---------------------------------------------------------------- 精度护栏
def fp32_reference(s: dict) -> dict:
    """板上 PyTorch FP32（GPU）基准：有效像元特征、点特征、匹配行列及 top1-top2 分差。"""
    hc.setup_determinism()
    valid_mask = s["valid_mask"]
    vr, vc = np.where(valid_mask)
    normalized, _, _ = standardize_hsi(s["hsi_raw"])
    patches = build_hsi_patches(normalized, list(zip(vr.tolist(), vc.tolist())), 3)
    hsi_feats = hc.torch_forward(hc.get_model("hsi", "cuda"), patches, device="cuda", batch=512)
    pc_feats = hc.torch_forward(hc.get_model("pc", "cuda"), s["point_offsets"], device="cuda", batch=512)
    grid = np.zeros((s["rows"], s["cols"], 1024), dtype=np.float32)
    grid[vr, vc] = hsi_feats
    grid_t, pf_t = torch.from_numpy(grid), torch.from_numpy(pc_feats).float()
    ref = {"hsi_feats_valid": hsi_feats, "pc_feats": pc_feats, "match": {}}
    for v, sw in VARIANTS.items():
        r, c, margin = score_window_with_margin(pf_t, grid_t, valid_mask, s["ref_rows"], s["ref_cols"],
                                                SEARCH_RADIUS, sw)
        ref["match"][v] = {"rows": r, "cols": c, "margin": margin}
    return ref


def check_vs_fp32(out: dict, ref: dict) -> dict:
    rep = {"hsi_feat": hc.accuracy_report(ref["hsi_feats_valid"], out["hsi_feats_valid"]),
           "pc_feat": hc.accuracy_report(ref["pc_feats"], out["pc_feats"]), "match": {}}
    n = len(next(iter(ref["match"].values()))["rows"])
    for v, m in out["match"].items():
        b = ref["match"][v]
        bad = np.where((m["rows"] != b["rows"]) | (m["cols"] != b["cols"]))[0]
        margins = [float(b["margin"][i]) for i in bad if np.isfinite(b["margin"][i])]
        rep["match"][v] = {"agreement": 1 - len(bad) / n, "n_mismatch": int(len(bad)),
                           "max_fp32_margin_of_mismatch": max(margins) if margins else None,
                           "all_near_ties": (max(margins) < REF_MARGIN_P10["hsi"] * NEAR_TIE_FACTOR) if margins else True}
    return rep


def same_as(prev: dict, out: dict) -> dict:
    """与上一个配置比较：特征最大绝对差、匹配行列是否逐位相同。"""
    return {"hsi_feat_max_abs_diff": float(np.abs(prev["hsi_feats_valid"] - out["hsi_feats_valid"]).max()),
            "pc_feat_max_abs_diff": float(np.abs(prev["pc_feats"] - out["pc_feats"]).max()),
            "match_rowcol_identical": {v: bool(np.array_equal(prev["match"][v]["rows"], out["match"][v]["rows"])
                                               and np.array_equal(prev["match"][v]["cols"], out["match"][v]["cols"]))
                                       for v in out["match"]}}


def frontend_guardrail(s: dict) -> dict:
    """重建的全网格 patch / 掩膜必须与缓存逐位相同，否则整景对比没有意义。"""
    normalized, _, _ = standardize_hsi(s["hsi_raw"])
    rc = [(r, c) for r in range(s["rows"]) for c in range(s["cols"])]
    patches = build_hsi_patches(normalized, rc, 3)
    mask = valid_vegetation_mask(s["hsi_raw"], CONTRACT["red_band_index"], CONTRACT["nir_band_index"],
                                 CONTRACT["ndvi_threshold"], CONTRACT["nodata_epsilon"])
    g = {"patches_identical_to_cache": bool(np.array_equal(patches, s["hsi_grid_patches"])),
         "valid_mask_identical_to_cache": bool(np.array_equal(mask, s["valid_mask"]))}
    if not all(g.values()):
        raise SystemExit(f"前处理护栏失败：{g}")
    return g


# ---------------------------------------------------------------- 子命令
def cmd_profile(a):
    s = load_scene()
    cfg = CONFIGS[a.config]
    runners = Runners(a.tier)
    for _ in range(WARMUP + 1):
        run_any(cfg, s, runners)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"scene_{a.config}")
    out, t = run_any(cfg, s, runners)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()
    print(json.dumps({"config": a.config, "segments_ms": {k: v * 1000 for k, v in t.items()},
                      "n_patches": out["n_patches"]}, indent=2))


def cmd_compare(a):
    s = load_scene()
    guard = frontend_guardrail(s)
    ref = fp32_reference(s)
    names = a.configs
    runners = Runners(a.tier)
    result = {"tier": a.tier, "warmup": WARMUP, "repeat": REPEAT, "frontend_guardrail": guard, "rounds": {},
              "accuracy_vs_fp32": {}, "vs_previous_config": {}}
    prev_out = None
    for rname, order in (("forward", names), ("reverse", names[::-1])):
        result["rounds"][rname] = {}
        for n in order:
            out, med, total, totals = timed_config(CONFIGS[n], s, runners)
            result["rounds"][rname][n] = {"total_ms_median": total, "total_ms_all": totals, "segments_ms_median": med,
                                          "n_patches": out["n_patches"]}
            print(f"[{rname}] {n}: total {total:.1f} ms  " + "  ".join(f"{k}={v:.1f}" for k, v in med.items()), flush=True)
            if rname == "forward":
                result["accuracy_vs_fp32"][n] = check_vs_fp32(out, ref)
                if out.get("valid_mask") is not None:  # 掩膜护栏：任何配置算出的有效像元掩膜都必须与缓存逐位相同
                    result.setdefault("valid_mask_identical_to_cache", {})[n] = bool(np.array_equal(out["valid_mask"], s["valid_mask"]))
                if prev_out is not None:
                    result["vs_previous_config"][n] = same_as(prev_out, out)
                if not (a.vs_first and prev_out is not None):  # --vs-first：每个配置都与列表第一个配置比较
                    prev_out = out
    base = names[0]
    result["speedup_total"] = {r: {n: result["rounds"][r][base]["total_ms_median"] / result["rounds"][r][n]["total_ms_median"]
                                   for n in names} for r in result["rounds"]}
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["speedup_total"], indent=2))
    print("written ->", a.out)


def cmd_sweep(a):
    from stage7_scene_engine_sweep import measure  # 复用 measure：LiteTrtRunner + CUDA event/墙钟计时

    s = load_scene()
    ref = fp32_reference(s)
    normalized, _, _ = standardize_hsi(s["hsi_raw"])
    vr, vc = np.where(s["valid_mask"])
    hsi_x = build_hsi_patches(normalized, list(zip(vr.tolist(), vc.tolist())), 3)
    xs = {"hsi": hsi_x, "pc": s["point_offsets"]}
    val = {w: (hc.load_samples(w, "val"), np.load(RESULTS / "ref" / f"{w}_fp32_cpu.npy")) for w in ("hsi", "pc")}
    groups = {"hsi": ["stem"], "pc": []}  # PC 先走 auto；只有精度/稳定性不达标才改 mixed softmax
    result = {"tiers": a.tiers, "groups": groups, "repeat_builds": a.repeat_builds, "engines": {}}
    out_path = Path(a.out)
    for which in a.which:
        result["engines"][which] = {}
        for tier in a.tiers:
            name = f"scene{tier}"
            bt.PROFILES[name] = {"min": 1, "opt": tier, "max": tier}
            policy = "mixed" if groups[which] else "auto"
            builds = []
            sfx = "_tf32" if a.allow_tf32 else ""
            for i in range(a.repeat_builds):
                out = ENGINE_DIR / (f"{which}_fp16_scene{tier}{sfx}.plan" if i == 0 else f"{which}_fp16_scene{tier}{sfx}_b{i}.plan")
                try:
                    bt.build_engine(which, "fp16", profile_name=name, precision_policy=policy,
                                    fp32_groups=groups[which], out_path=out, policy_info={}, allow_tf32=a.allow_tf32)
                except Exception as e:  # 构建失败（如共享内存不足）：记录错误，继续后面的档位
                    result["engines"][which][str(tier)] = {"build_error": str(e)[:300], "build_index": i}
                    print(f"{which} tier {tier}: BUILD FAILED ({str(e)[:120]})", flush=True)
                    builds = None
                    break
                builds.append(out)
            if builds is None:
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float))
                continue
            r = measure(which, builds[0], tier, xs[which])
            outs = []
            for b in builds:
                runner = LiteTrtRunner(b, INPUT_DIMS[which], max_batch=min(tier, 64))  # 精度用 val 集 200 条
                outs.append(runner.infer_all(val[which][0]))
                del runner
            r["val_accuracy"] = [hc.accuracy_report(val[which][1], y) for y in outs]
            r["build_pairwise_max_abs"] = {f"{i}-{j}": float(np.abs(outs[i] - outs[j]).max())
                                           for i in range(len(outs)) for j in range(i + 1, len(outs))}
            r["precision_policy"] = policy
            result["engines"][which][str(tier)] = r
            print(f"{which} tier {tier}: wall {r['wall_ms_median']:.1f} ms  gpu {r['gpu_event_ms_median']:.1f} ms  "
                  f"devmem {r['device_memory_size_bytes'] / 2**20:.0f} MiB  "
                  f"cos_min {[round(x['cos_sim_min'], 6) for x in r['val_accuracy']]}  "
                  f"pairwise {r['build_pairwise_max_abs']}", flush=True)
            gc.collect()
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    print("written ->", a.out)


# ---------------------------------------------------------------- 单样本低延迟：CUDA Graph（O4）
class GraphRunner:
    """固定 batch 的 TRT 推理：输入/输出用固定的设备张量，可选 CUDA Graph（捕获一次、replay 复用）。
    用于单样本低延迟路径，与整景吞吐路径分开报告。eager 与 graph 走同一批 kernel，输出应逐位相同。"""

    def __init__(self, plan: Path, which: str, batch: int, use_graph: bool):
        from trt_runner import _TRT_TO_TORCH, load_engine
        self.engine = load_engine(plan)
        self.ctx = self.engine.create_execution_context()
        self.x = torch.zeros((batch, *INPUT_DIMS[which]), dtype=_TRT_TO_TORCH[self.engine.get_tensor_dtype("input")], device="cuda")
        self.ctx.set_input_shape("input", tuple(self.x.shape))
        self.y = torch.empty(tuple(self.ctx.get_tensor_shape("output")),
                             dtype=_TRT_TO_TORCH[self.engine.get_tensor_dtype("output")], device="cuda")
        self.ctx.set_tensor_address("input", self.x.data_ptr())
        self.ctx.set_tensor_address("output", self.y.data_ptr())
        self.stream = torch.cuda.Stream()
        self.graph = None
        with torch.cuda.stream(self.stream):  # 捕获前必须先 eager 跑一次（TRT 惰性初始化）
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if use_graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.ctx.execute_async_v3(self.stream.cuda_stream)

    def run(self):
        if self.graph is not None:
            with torch.cuda.stream(self.stream):
                self.graph.replay()
        else:
            self.ctx.execute_async_v3(self.stream.cuda_stream)


def cmd_latency(a):
    """batch 1/8 的单样本延迟：eager vs CUDA Graph。三个口径：event（只计 GPU 段，同 J3）、wall（下发+同步）、
    request（含 pinned 主机内存的 H2D 输入拷贝和 D2H 输出拷贝，最接近真实请求）。"""
    hc.setup_determinism()
    result = {"warmup": 50, "measure": a.iters, "engines": {}}
    for which in ("pc", "hsi"):
        plan = ENGINE_DIR / f"{which}_fp16.plan"
        val = hc.load_samples(which, "val")
        for b in a.batches:
            key, outs = f"{which}_b{b}", {}
            result["engines"][key] = {}
            for mode in ("eager", "graph"):
                r = GraphRunner(plan, which, b, use_graph=(mode == "graph"))
                host_in = torch.from_numpy(val[:b].copy()).pin_memory()
                host_out = torch.empty(r.y.shape, dtype=r.y.dtype).pin_memory()
                with torch.cuda.stream(r.stream):
                    r.x.copy_(host_in, non_blocking=True)
                    r.run()
                r.stream.synchronize()
                outs[mode] = r.y.float().cpu().numpy().copy()

                def request():
                    with torch.cuda.stream(r.stream):
                        r.x.copy_(host_in, non_blocking=True)
                        r.run()
                        host_out.copy_(r.y, non_blocking=True)
                    r.stream.synchronize()

                for _ in range(50):
                    request()
                ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.iters)]
                with torch.cuda.stream(r.stream):
                    for s_, e_ in ev:
                        s_.record(r.stream); r.run(); e_.record(r.stream)
                r.stream.synchronize()
                event_ms = np.array([s_.elapsed_time(e_) for s_, e_ in ev])
                wall, req = [], []
                for _ in range(a.iters):
                    t0 = time.perf_counter(); r.run(); r.stream.synchronize(); wall.append((time.perf_counter() - t0) * 1000)
                for _ in range(a.iters):
                    t0 = time.perf_counter(); request(); req.append((time.perf_counter() - t0) * 1000)
                st = lambda v: {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)), "p99_ms": float(np.percentile(v, 99))}  # noqa: E731
                result["engines"][key][mode] = {"event": st(event_ms), "wall": st(np.array(wall)), "request": st(np.array(req))}
                del r
            result["engines"][key]["graph_output_identical_to_eager"] = bool(np.array_equal(outs["eager"], outs["graph"]))
            e, g = result["engines"][key]["eager"], result["engines"][key]["graph"]
            result["engines"][key]["speedup_median"] = {k: e[k]["median_ms"] / g[k]["median_ms"] for k in ("event", "wall", "request")}
            print(key, {k: (round(e[k]["median_ms"], 3), round(g[k]["median_ms"], 3)) for k in ("event", "wall", "request")},
                  "identical:", result["engines"][key]["graph_output_identical_to_eager"], flush=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("written ->", a.out)


# ---------------------------------------------------------------- 引擎构建优化级别（HSI 已是 GPU 瓶颈）
def cmd_optlevel(a):
    """HSI scene 引擎：builder_optimization_level 3（默认）vs 5，同进程对照测整景 HSI 推理耗时，测 L3 - L5 - L3 防漂移。"""
    from stage7_scene_engine_sweep import measure

    import tensorrt as trt
    s = load_scene()
    normalized, _, _ = standardize_hsi(s["hsi_raw"])
    vr, vc = np.where(s["valid_mask"])
    x = build_hsi_patches(normalized, list(zip(vr.tolist(), vc.tolist())), 3)
    val_x, val_ref = hc.load_samples("hsi", "val"), np.load(RESULTS / "ref" / "hsi_fp32_cpu.npy")
    tier = a.tier
    l3 = plan_path("hsi", "scene", tier)
    l5 = ENGINE_DIR / f"hsi_fp16_scene{tier}_opt{a.level}.plan"
    orig = trt.Builder.create_builder_config

    def cfg_with_level(self):
        c = orig(self); c.builder_optimization_level = a.level; return c

    bt.PROFILES[f"scene{tier}"] = {"min": 1, "opt": tier, "max": tier}
    result = {"tier": tier, "level": a.level, "l3_first": measure("hsi", l3, tier, x)}
    print("L3 first:", result["l3_first"]["wall_ms_median"], result["l3_first"]["gpu_event_ms_median"], flush=True)
    trt.Builder.create_builder_config = cfg_with_level
    t0 = time.time()
    try:
        bt.build_engine("hsi", "fp16", profile_name=f"scene{tier}", precision_policy="mixed", fp32_groups=["stem"], out_path=l5)
    finally:
        trt.Builder.create_builder_config = orig
    result["build_seconds"] = time.time() - t0
    result["lN"] = measure("hsi", l5, tier, x)
    runner = LiteTrtRunner(l5, INPUT_DIMS["hsi"], max_batch=min(tier, 64))
    result["lN_val_accuracy"] = hc.accuracy_report(val_ref, runner.infer_all(val_x))
    del runner
    result["l3_second"] = measure("hsi", l3, tier, x)
    print(f"L{a.level}:", result["lN"]["wall_ms_median"], result["lN"]["gpu_event_ms_median"],
          "build", round(result["build_seconds"]), "s; L3 second:", result["l3_second"]["gpu_event_ms_median"], flush=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float))


# ---------------------------------------------------------------- 能效与冷启动
def cmd_energy(a):
    """按阶段连续跑（空闲基线 / 各配置 >= a.seconds 秒 / 空闲基线），记录每个阶段的起止时间戳和整景次数，
    与 tegrastats 日志（VDD_IN 整板输入功率）事后对齐，见 energy-report。"""
    s = load_scene()
    runners = Runners(a.tier)
    phases = []

    def idle(name):
        t0 = time.time(); time.sleep(a.idle_seconds)
        phases.append({"name": name, "t_start": t0, "t_end": time.time(), "n_scenes": 0})

    idle("idle_before")
    for name in a.configs:
        cfg = CONFIGS[name]
        for _ in range(WARMUP):
            run_any(cfg, s, runners)
        torch.cuda.synchronize()
        t0, n = time.time(), 0
        while time.time() - t0 < a.seconds:
            run_any(cfg, s, runners); torch.cuda.synchronize(); n += 1
        phases.append({"name": name, "t_start": t0, "t_end": time.time(), "n_scenes": n})
        print(name, "scenes", n, flush=True)
    idle("idle_after")
    Path(a.out).write_text(json.dumps({"phases": phases}, indent=2))


def cmd_energy_report(a):
    from tegrastats_util import mean_or_die, parse_tegrastats
    # 本命令按约定在板上直接运行（见 docs/jetson.md 复现命令），tegrastats 的本地时间和 t_start/t_end
    # 用的 time.time() 是同一台机器，不需要跨时区换算，用运行本脚本这台机器的本地时区解析即可。
    samples = parse_tegrastats(Path(a.tegrastats))
    phases = json.loads(Path(a.phases).read_text())["phases"]
    rep = {}
    for ph in phases:
        # tegrastats 时间戳精度 1 秒：首尾各去掉 1 秒边界
        w = [r for r in samples if ph["t_start"] + 1 <= r["t"] <= ph["t_end"] - 1]
        pw = [r["power_mw"] for r in w if r["power_mw"] is not None]
        gr = [r["gr3d_pct"] for r in w if r["gr3d_pct"] is not None]
        vdd_in_mean_mw = mean_or_die(pw, f"energy-report phase {ph['name']} ({ph['t_start']}~{ph['t_end']})")
        rep[ph["name"]] = {"duration_s": ph["t_end"] - ph["t_start"], "n_scenes": ph["n_scenes"], "n_samples": len(w),
                           "vdd_in_mean_mw": vdd_in_mean_mw, "vdd_in_max_mw": max(pw),
                           "gr3d_mean_pct": float(np.mean(gr)) if gr else None}
    idle_mw = float(np.mean([rep["idle_before"]["vdd_in_mean_mw"], rep["idle_after"]["vdd_in_mean_mw"]]))
    for name, r in rep.items():
        if r["n_scenes"]:
            per = r["duration_s"] / r["n_scenes"]
            r["scene_ms"] = per * 1000
            r["energy_per_scene_j_total_board"] = r["vdd_in_mean_mw"] / 1000 * per
            r["energy_per_scene_j_above_idle"] = (r["vdd_in_mean_mw"] - idle_mw) / 1000 * per
            r["scenes_per_s_per_w_total_board"] = (1 / per) / (r["vdd_in_mean_mw"] / 1000)
    rep["_idle_mean_mw"] = idle_mw
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def cmd_coldstart(a):
    """新进程冷启动：CUDA 初始化、载入场景数据、引擎反序列化+建 context+分配 buffer、连续 4 次整景。
    （页缓存不清：不用 sudo，所以是"进程冷、文件缓存热"，如实标注。）"""
    t = {}
    t0 = time.perf_counter(); torch.zeros(1).cuda(); torch.cuda.synchronize(); t["cuda_init_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); s = load_scene(); t["load_scene_s"] = time.perf_counter() - t0
    cfg = CONFIGS[a.config]
    runners = Runners(a.tier)
    t0 = time.perf_counter(); runners.get("hsi", cfg); runners.get("pc", cfg); torch.cuda.synchronize()
    t["engines_load_s"] = time.perf_counter() - t0
    runs = []
    for _ in range(4):
        t0 = time.perf_counter(); run_any(cfg, s, runners); torch.cuda.synchronize(); runs.append(time.perf_counter() - t0)
    t["scene_runs_s"] = runs
    t["config"] = a.config
    Path(a.out).write_text(json.dumps(t, indent=2))
    print(json.dumps(t, indent=2))


def cmd_matchbench(a):
    """匹配段微基准：固定真实特征（S8 链路算出，驻留 GPU），只改 chunk / single_transfer，逐位比较行列与 cosine。"""
    s = load_scene()
    runners = Runners(a.tier)
    cfg = CONFIGS["S8"]
    out, _ = run_scene_gpu(cfg, s, runners, keep=True)
    vr, vc = np.where(s["valid_mask"])
    grid = torch.zeros((s["rows"], s["cols"], 1024), device="cuda")
    grid[torch.from_numpy(vr).cuda(), torch.from_numpy(vc).cuda()] = torch.from_numpy(out["hsi_feats_valid"]).cuda()
    pf = torch.from_numpy(out["pc_feats"]).cuda()
    args = (pf, grid, s["valid_mask"], s["ref_rows"], s["ref_cols"], SEARCH_RADIUS, VARIANTS)
    base = gpu_score_window_multi(*args)
    result = {"warmup": 3, "repeat": a.repeat, "variants": {}}
    for chunk in a.chunks:
        for st in (False, True):
            f = lambda: gpu_score_window_multi(*args, chunk=chunk, single_transfer=st)  # noqa: E731
            for _ in range(3):
                f()
            ts = []
            for _ in range(a.repeat):
                torch.cuda.synchronize(); t0 = time.perf_counter(); r = f(); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            same = all(np.array_equal(r[v][k], base[v][k]) for v in base for k in ("rows", "cols", "cosine", "pixel_displacement"))
            key = f"chunk{chunk}_single{int(st)}"
            result["variants"][key] = {"median_ms": float(np.median(ts)), "min_ms": float(np.min(ts)), "identical_to_default": bool(same),
                                       "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
            torch.cuda.reset_peak_memory_stats()
            print(key, round(float(np.median(ts)), 2), "ms identical:", same, flush=True)
    Path(a.out).write_text(json.dumps(result, indent=2))


def main():
    global SCENE_NPZ
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-npz", help="场景缓存（默认 results/e2e_scene_preprocessed.npz）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile"); p.add_argument("config", choices=CONFIGS); p.add_argument("--tier", type=int)
    p.set_defaults(fn=cmd_profile)
    p = sub.add_parser("sweep"); p.add_argument("--tiers", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--repeat-builds", type=int, default=1)
    p.add_argument("--allow-tf32", action="store_true"); p.add_argument("--which", nargs="+", default=["hsi", "pc"])
    p.add_argument("--out", default=str(RESULTS / "opt_o2_sweep.json")); p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("compare"); p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    p.add_argument("--tier", type=int); p.add_argument("--out", default=str(RESULTS / "opt_o6_compare.json"))
    p.add_argument("--vs-first", action="store_true", help="vs_previous_config 改为与第一个配置比较")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("matchbench"); p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--chunks", type=int, nargs="+", default=[128, 256, 512, 1000]); p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--out", default=str(RESULTS / "opt_matchbench.json")); p.set_defaults(fn=cmd_matchbench)
    p = sub.add_parser("latency"); p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument("--iters", type=int, default=300); p.add_argument("--out", default=str(RESULTS / "opt_o4_latency.json"))
    p.set_defaults(fn=cmd_latency)
    p = sub.add_parser("optlevel"); p.add_argument("--tier", type=int, default=1024); p.add_argument("--level", type=int, default=5)
    p.add_argument("--out", default=str(RESULTS / "opt_optlevel.json")); p.set_defaults(fn=cmd_optlevel)
    p = sub.add_parser("energy"); p.add_argument("--configs", nargs="+", default=["S0", "S10"], choices=CONFIGS)
    p.add_argument("--tier", type=int, default=1024); p.add_argument("--seconds", type=float, default=25)
    p.add_argument("--idle-seconds", type=float, default=20); p.add_argument("--out", default=str(RESULTS / "opt_energy_phases.json"))
    p.set_defaults(fn=cmd_energy)
    p = sub.add_parser("energy-report"); p.add_argument("--tegrastats", required=True); p.add_argument("--phases", required=True)
    p.add_argument("--out", default=str(RESULTS / "opt_energy.json")); p.set_defaults(fn=cmd_energy_report)
    p = sub.add_parser("coldstart"); p.add_argument("config", choices=CONFIGS); p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--out", default=str(RESULTS / "opt_coldstart.json")); p.set_defaults(fn=cmd_coldstart)
    a = ap.parse_args()
    if a.scene_npz:
        SCENE_NPZ = Path(a.scene_npz)
    a.fn(a)


if __name__ == "__main__":
    main()
