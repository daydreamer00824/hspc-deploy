"""C++ 部署程序 vs Python 最终配置的完整端到端汇总（本机运行，读板上拉回的 results/jetson/cpp_e2e/*.json）。

护栏：每个 K 配置、每一轮的匹配行列都必须与同一景的 Python P（正轮）逐位相同；
输出 results/jetson/cpp_e2e_summary.json。用法：cpp_e2e_summarize.py [输入目录] [输出文件]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching import VARIANTS  # noqa: E402

SCENES = ["24data/10.6/1", "24data/10.6/10", "25data/8.4/1", "25data/10.22/1"]
CFGS = ["P", "K0", "K1", "K2", "K3", "K4", "K5"]
D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/jetson/cpp_e2e")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results/jetson/cpp_e2e_summary.json")


def rowcols(r, cfg):
    """统一成 {variant: (rows, cols)}；C++ 的 variant0/1 按 VARIANTS 顺序对应。"""
    m = r["match_rowcols"]
    if cfg == "P":
        return {v: (m[v]["rows"], m[v]["cols"]) for v in VARIANTS}
    return {v: (m[f"variant{i}"]["rows"], m[f"variant{i}"]["cols"]) for i, v in enumerate(VARIANTS)}


summary = {"note": "P=Python 最终配置 E（deploy_jetson.py 默认）；K0=C++ 按 E 的调度原样移植；K1=+HSI 掩膜/统计量多线程；K2=+投影多线程；"
                   "K3=+HSI 文件整块读；K4=+跨景复用 PROJ 变换对象；K5=+LAS 段与 HSI 读取同时开始。各自独立进程，正轮 P→K0→…→K5，反轮 K5→…→K0→P；"
                   "每个 JSON 内 warmup 2 + 重复 10 次取中位数。total 为整景墙钟中位数（含文件 IO，页缓存热）。", "scenes": {}}
for sc in SCENES:
    tag = sc.replace("/", "_")
    p_fwd = json.loads((D / f"P_{tag}_fwd.json").read_text())
    ref = rowcols(p_fwd, "P")
    ent = {}
    for c in CFGS:
        ent[c] = {}
        for rnd in ("fwd", "rev"):
            r = p_fwd if (c, rnd) == ("P", "fwd") else json.loads((D / f"{c}_{tag}_{rnd}.json").read_text())
            rc = rowcols(r, c)
            same = {v: bool(np.array_equal(rc[v][0], ref[v][0]) and np.array_equal(rc[v][1], ref[v][1])) for v in VARIANTS}
            w = r["warm"]
            segs = w["segments_ms_median"]
            ent[c][rnd] = {"total_ms": w["total_ms_median"], "total_without_io_ms": w["total_without_io_ms_median"], "segments_ms": segs,
                           "cold_start_sec": r["cold_start"]["total_sec"], "engine_load_sec": r["engine_load_sec"],
                           "peak_rss_kb": r.get("peak_rss_kb"), "deterministic": r["determinism_match_rowcols_identical_across_reps"],
                           "match_rowcols_identical_to_P": same}
            assert all(same.values()) and ent[c][rnd]["deterministic"] is not False, (sc, c, rnd, same)
    summary["scenes"][sc] = ent
    print(sc, " ".join(f"{c}={np.mean([ent[c][r]['total_ms'] for r in ('fwd', 'rev')]):.1f}" for c in CFGS))
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("written ->", OUT)
