"""P5 汇总（板上运行，系统 Python）：完整端到端 A/B/C × 4 景 × 正反两轮。

护栏：每个 JSON 里的匹配行列都必须与"缓存输入 + 最终配置 S11b"（同一进程现算）逐位相同；B/C 的前处理产物与缓存逐位相同。
输出 results/e2e_full_summary.json（拉回本机后放 results/jetson/）。
用法：p5_summarize.py [配置字母（默认 ABC）] [输出文件名]；如 `p5_summarize.py C2DE e2e_las_summary.json`（C2 记为一个配置名）。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import jetson_scene_opt as j  # noqa: E402

SCENES = ["24data/10.6/1", "24data/10.6/10", "25data/8.4/1", "25data/10.22/1"]
CFGS = (["C2", "D", "E"] if len(sys.argv) > 1 and sys.argv[1] == "C2DE" else list(sys.argv[1] if len(sys.argv) > 1 else "ABC"))
OUT = Path("results") / (sys.argv[2] if len(sys.argv) > 2 else "e2e_full_summary.json")
D = Path("results/jetson_e2e")
summary = {"note": "A=deploy_scene.py（x86 最终链路原样）；B=deploy_jetson.py 顺序执行；C=B+LAS 与 GPU 推理重叠。"
                   "各自独立进程，顺序 A→B→C（fwd）再 C→B→A（rev）；每个 JSON 内 warmup 2 + 重复 10 次取中位数。", "scenes": {}}
for sc in SCENES:
    tag = sc.replace("/", "_")
    j.SCENE_NPZ = Path(f"results/scenes/{tag}_preprocessed.npz")
    s = j.load_scene()
    ref, _ = j.run_any(j.CONFIGS["S11b"], s, j.Runners(1024), keep=True)
    ent = {}
    for c in CFGS:
        ent[c] = {}
        for rnd in ("fwd", "rev"):
            r = json.loads((D / f"e2e_full_{c}_{tag}_{rnd}.json").read_text())
            same = {v: bool(np.array_equal(r["match_rowcols"][v]["rows"], ref["match"][v]["rows"]) and
                            np.array_equal(r["match_rowcols"][v]["cols"], ref["match"][v]["cols"])) for v in ref["match"]}
            if c == "A":
                w = r["warm"]
                segs = {k: v["median_sec"] * 1000 for k, v in w["segments"].items()}
                total, total_wo_io = w["total_with_io_median_sec"] * 1000, w["total_without_io_median_sec"] * 1000
                cold = r["cold_start"]["total_with_io_sec"]
                guard = None
                note = "A 的总计为各段中位数之和（deploy_scene.py 原有口径）"
            else:
                w = r["warm"]
                segs, total, total_wo_io = w["segments_ms_median"], w["total_ms_median"], w["total_without_io_ms_median"]
                cold = r["cold_start"]["total_sec"]
                guard = r.get("guardrail_vs_cache")
                note = "B/C 的总计为整景墙钟中位数"
            ent[c][rnd] = {"total_ms": total, "total_without_io_ms": total_wo_io, "segments_ms": segs,
                           "cold_start_sec": cold, "engine_load_sec": r["engine_load_sec"],
                           "match_rowcols_identical_to_cached_S11b": same, "preprocess_guardrail_vs_cache": guard,
                           "total_definition": note}
            print(tag, c, rnd, f"total {total:.1f} ms (w/o IO {total_wo_io:.1f})", "rowcols==S11b", same, "guard", guard, flush=True)
    summary["scenes"][sc] = ent
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("written ->", OUT)
