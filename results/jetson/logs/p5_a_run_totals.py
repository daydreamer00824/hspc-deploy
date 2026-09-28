"""修正完整端到端 A（deploy_scene.py）与 B～E（deploy_jetson.py）总计的统计口径不一致问题
（外部审查指出：A 原为"各段中位数之和"，B～E 为"10 次整景墙钟的中位数"，二者相除得到的加速比
混用了两种统计量）。本机运行，只读已拉回的板上原始记录（results/jetson/e2e_full/*.json，
未入库、含逐点匹配行列，不受影响，只用其中的计时数据），不重跑、不产生新测量。

做法：A 也改用"10 次整景耗时的中位数"——A 链路完全顺序执行，每次整景耗时 = 该次各分段耗时之和
（deploy_scene.py 的 warm.segments[*].all_sec[i]，10 次各自求和取中位数），与 B～E 同口径；
deploy_scene.py 自身报告的"各段中位数之和"保留在新增的 total_ms_sum_of_segment_medians 字段。
B～E 的 total_ms 本身已经是墙钟中位数（warm.total_ms_all 的中位数），不改数值，只补
total_statistic 字段并修正 total_definition 的措辞。

同时修正 e2e_las_summary.json 顶层 note：该文件实际配置是 C2/D/E（第二批复测 C + D + E），
之前误写成了 A/B/C 的说明。

用法：python results/jetson/logs/p5_a_run_totals.py
读取并原地更新 results/jetson/e2e_full_summary.json（A/B/C）与 results/jetson/e2e_las_summary.json（C2/D/E）。
可重复运行：所有数值都从原始记录重新推导，不依赖上一次的输出。
"""
import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _load_segment_constants() -> tuple[list[str], set[str]]:
    """从 scripts/deploy_scene.py 的源码里解析 SEGMENT_ORDER / IO_SEGMENTS 这两个模块级常量，
    不 import 该文件（它依赖 tensorrt/polygraphy/私有 matching 模块，本机汇总脚本不需要装这些）。"""
    tree = ast.parse((ROOT / "scripts/deploy_scene.py").read_text())
    found: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("SEGMENT_ORDER", "IO_SEGMENTS"):
                found[name] = ast.literal_eval(node.value)
    assert "SEGMENT_ORDER" in found and "IO_SEGMENTS" in found, "scripts/deploy_scene.py 里找不到 SEGMENT_ORDER / IO_SEGMENTS"
    return found["SEGMENT_ORDER"], found["IO_SEGMENTS"]


SEGMENT_ORDER, IO_SEGMENTS = _load_segment_constants()
SCENES = ["24data/10.6/1", "24data/10.6/10", "25data/8.4/1", "25data/10.22/1"]
RAW_DIR = ROOT / "results/jetson/e2e_full"
FULL_JSON = ROOT / "results/jetson/e2e_full_summary.json"
LAS_JSON = ROOT / "results/jetson/e2e_las_summary.json"

A_DEFINITION = (
    "A（deploy_scene.py）顺序执行；每次整景耗时 = 该次各分段耗时之和（不含分段之间约 <1ms 未计时的"
    "胶水代码），10 次取中位数，与 B～E 同口径。deploy_scene.py 自身报告的“各段中位数之和”保留在 "
    "total_ms_sum_of_segment_medians / total_without_io_ms_sum_of_segment_medians，两者通常相差不到 2ms。"
)
BE_DEFINITION = "整景墙钟中位数：每次整景整体计时（含文件读取，页缓存热），10 次取中位数。"

FULL_NOTE_SUFFIX = "（A 的 total_ms 与 B/C 同口径，均为 10 次整景墙钟耗时的中位数；A 自身各段中位数之和另见 total_ms_sum_of_segment_medians。）"
LAS_NOTE = (
    "C2=第二批复测的 C（B + LAS 与 GPU 推理重叠，用于确认批次间无漂移）；D=C2 + kNN 查询多核；"
    "E=D + kd 树构建与投影并行（最终配置，deploy_jetson.py 默认）。各自独立进程，每景 C2→D→E（fwd）"
    "再 E→D→C2（rev）；每个 JSON 内 warmup 2 + 重复 10 次取中位数，total_ms 为整景墙钟中位数。"
)


def recompute_a(tag: str, rnd: str) -> dict:
    raw = json.loads((RAW_DIR / f"e2e_full_A_{tag}_{rnd}.json").read_text())
    segs = raw["warm"]["segments"]
    assert set(segs) == set(SEGMENT_ORDER), f"A {tag} {rnd}：分段集合与 deploy_scene.SEGMENT_ORDER 不一致"
    n = len(segs["match"]["all_sec"])
    assert all(len(v["all_sec"]) == n for v in segs.values()), f"A {tag} {rnd}：各段重复次数不一致"

    per_run_total = [sum(segs[s]["all_sec"][i] for s in segs) for i in range(n)]
    per_run_total_wo_io = [sum(segs[s]["all_sec"][i] for s in segs if s not in IO_SEGMENTS) for i in range(n)]
    total_ms = float(np.median(per_run_total)) * 1000
    total_wo_io_ms = float(np.median(per_run_total_wo_io)) * 1000
    sum_of_medians_ms = sum(segs[s]["median_sec"] for s in segs) * 1000
    sum_of_medians_wo_io_ms = sum(segs[s]["median_sec"] for s in segs if s not in IO_SEGMENTS) * 1000

    # 与原始 JSON 自带的 total_with_io_median_sec（deploy_scene.py 当时写出的口径）交叉核对，
    # 确认我们独立重算出的"各段中位数之和"与它一致——这只是校验，不作为最终采用值。
    raw_reported = raw["warm"]["total_with_io_median_sec"] * 1000
    diff = abs(sum_of_medians_ms - raw_reported)
    assert diff < 0.05, f"A {tag} {rnd}：重算的各段中位数之和 {sum_of_medians_ms:.3f} 与原始记录 {raw_reported:.3f} 不一致（差 {diff:.3f}ms）"

    print(f"  A {tag} {rnd}: median_of_run_totals={total_ms:.1f} sum_of_segment_medians={sum_of_medians_ms:.1f} "
          f"diff={total_ms - sum_of_medians_ms:+.1f}ms range=[{min(per_run_total)*1000:.1f}, {max(per_run_total)*1000:.1f}]")
    return {
        "total_ms": total_ms,
        "total_without_io_ms": total_wo_io_ms,
        "total_ms_sum_of_segment_medians": sum_of_medians_ms,
        "total_without_io_ms_sum_of_segment_medians": sum_of_medians_wo_io_ms,
        "total_statistic": "median_of_run_totals",
        "total_definition": A_DEFINITION,
    }


def recompute_be(cfg: str, tag: str, rnd: str) -> dict:
    raw = json.loads((RAW_DIR / f"e2e_full_{cfg}_{tag}_{rnd}.json").read_text())
    w = raw["warm"]
    all_ms = w["total_ms_all"]
    median_ms = float(np.median(all_ms))
    diff = abs(median_ms - w["total_ms_median"])
    assert diff < 1e-6, f"{cfg} {tag} {rnd}：重算中位数 {median_ms} 与原始记录 {w['total_ms_median']} 不一致"
    print(f"  {cfg} {tag} {rnd}: median_of_run_totals={median_ms:.1f} (n={len(all_ms)}, "
          f"range=[{min(all_ms):.1f}, {max(all_ms):.1f}])")
    return {"total_statistic": "median_of_run_totals", "total_definition": BE_DEFINITION}


def update_file(path: Path, cfgs: list[str], note_fix) -> None:
    summary = json.loads(path.read_text())
    for sc in SCENES:
        tag = sc.replace("/", "_")
        for cfg in cfgs:
            for rnd in ("fwd", "rev"):
                entry = summary["scenes"][sc][cfg][rnd]
                patch = recompute_a(tag, rnd) if cfg == "A" else recompute_be(cfg, tag, rnd)
                entry.update(patch)
    summary["note"] = note_fix(summary["note"])
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("written ->", path)


def _append_once(note: str, suffix: str) -> str:
    return note if suffix in note else note + suffix


if __name__ == "__main__":
    print("== e2e_full_summary.json (A/B/C) ==")
    update_file(FULL_JSON, ["A", "B", "C"], lambda note: _append_once(note, FULL_NOTE_SUFFIX))
    print("== e2e_las_summary.json (C2/D/E) ==")
    update_file(LAS_JSON, ["C2", "D", "E"], lambda note: LAS_NOTE)
