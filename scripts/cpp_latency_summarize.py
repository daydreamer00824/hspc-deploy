"""单样本延迟汇总：Python（GraphRunner）vs C++（hspc_latency），同一场次 Py→C++→C++→Py 交替（本机运行，读 results/jetson/cpp_latency/）。

对应 scripts/cpp_latency_run.sh。输出 results/jetson/cpp_latency_summary.json。用法：cpp_latency_summarize.py [输入目录] [输出文件]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tegrastats_util import board_utc_offset, tj_max  # noqa: E402

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/jetson/cpp_latency")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results/jetson/cpp_latency_summary.json")
KEYS = ["pc_b1", "pc_b8", "hsi_b1", "hsi_b8"]
res = {"note": "口径同 opt_o4_latency.json：event=只计 GPU 段；wall=下发+同步；request=页锁定内存 H2D + 推理 + D2H + 同步。引擎 *_fp16.plan（batch64 profile），"
               "warmup 50 + 测 300 次取中位数；同一场次 Py→C++→C++→Py 各一轮（fwd/rev），表中为两轮均值。", "models": {}}
# 每个文件只读一次：py_{rnd}.json 含全部 4 个 key，cpp_{rnd}_{pc,hsi}.json 各含同前缀的 2 个 key（b1/b8）
_py_cache = {rnd: json.loads((D / f"py_{rnd}.json").read_text())["engines"] for rnd in ("fwd", "rev")}
_cpp_cache = {(rnd, prefix): json.loads((D / f"cpp_{rnd}_{prefix}.json").read_text())["engines"]
             for rnd in ("fwd", "rev") for prefix in ("pc", "hsi")}
for k in KEYS:
    ent = {}
    for rnd in ("fwd", "rev"):
        py = _py_cache[rnd][k]
        cpp = _cpp_cache[(rnd, k.split("_")[0])][k]
        assert py["graph_output_identical_to_eager"] and cpp["graph_output_identical_to_eager"], (k, rnd)
        ent[rnd] = {"python": py, "cpp": cpp}
    avg = {}
    for lang in ("python", "cpp"):
        avg[lang] = {mode: {q: float(np.mean([ent[r][lang][mode][q]["median_ms"] for r in ("fwd", "rev")])) for q in ("event", "wall", "request")}
                     for mode in ("eager", "graph")}
    avg["cpp_vs_python_speedup"] = {mode: {q: avg["python"][mode][q] / avg["cpp"][mode][q] for q in ("event", "wall", "request")}
                                    for mode in ("eager", "graph")}
    res["models"][k] = {"mean_of_rounds": avg, "rounds": ent}
res["tj_max_c"] = tj_max(D / "tegrastats.log", board_utc_offset(D))
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
for k in KEYS:
    a = res["models"][k]["mean_of_rounds"]
    print(k, "request ms  py eager/graph %.3f/%.3f  c++ eager/graph %.3f/%.3f" % (a["python"]["eager"]["request"], a["python"]["graph"]["request"],
                                                                               a["cpp"]["eager"]["request"], a["cpp"]["graph"]["request"]),
          " event graph py/c++ %.3f/%.3f" % (a["python"]["graph"]["event"], a["cpp"]["graph"]["event"]))
print("tj max", res["tj_max_c"])
