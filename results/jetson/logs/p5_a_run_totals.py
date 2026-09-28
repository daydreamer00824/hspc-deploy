"""修正完整端到端 A（deploy_scene.py）与 B～E（deploy_jetson.py）总计的统计口径问题
（外部审查两轮指出：第一轮，A 原为"各段中位数之和"，B～E 为"10 次整景墙钟的中位数"，
二者相除混用了两种聚合方式；第二轮，即使把 A 也改成"10 次运行取中位数"，A 与 B～E 各自
一次运行里计时覆盖的范围仍不同——A 是该次运行各分段耗时之和，B～E 是该次运行的整景外层
墙钟）。本机运行，只读已拉回的板上原始记录（results/jetson/e2e_full/*.json，未入库、含
逐点匹配行列，不受影响，只用其中的计时数据），不重跑、不产生新测量。

做法：
1. 聚合方式统一：A 的 total_ms 改成"10 次整景耗时的中位数"——每次运行 = 该次各分段耗时之和
   （deploy_scene.py 的 warm.segments[*].all_sec[i]），10 次各自求和后取中位数，与 B～E 的
   "10 次运行取中位数"聚合方式相同。deploy_scene.py 自身报告的"各段中位数之和"保留在新增的
   total_ms_sum_of_segment_medians 字段，只作来源追溯，不作为比较用的值。
2. 计时范围仍不同：A 的分段计时全部在主线程上顺序执行、互不重叠（run_scene_once /
   run_las_pipeline 的写法），不包含分段之间未计时的胶水代码，所以每次运行的"分段之和"
   小于等于该次运行的真实整景墙钟；中位数对逐项不等式单调，所以 A 的 total_ms 小于等于
   A 真实整景墙钟的中位数。当时没有记录 A 的外层墙钟，这个差值无法恢复。
   B～E 的 total_ms 本身就是整景墙钟（total_ms_all 的中位数），不改数值。
   因此用 A.total_ms / E.total_ms 算出的倍数只能是真实加速比的**保守下界**，不是精确值；
   本文件给每条记录标注 timing_boundary（A："per_run_segment_sum_lower_bound"，
   B/C/C2/D/E："whole_scene_wall_clock"），下游脚本（make_readme_assets.py）据此把 A→E
   的展示强制为下界、向下截断到两位小数，不做四舍五入。

同时修正 e2e_las_summary.json 顶层 note：该文件实际配置是 C2/D/E（第二批复测 C + D + E），
之前误写成了 A/B/C 的说明。

用法：python results/jetson/logs/p5_a_run_totals.py
读取并原地更新 results/jetson/e2e_full_summary.json（A/B/C）与 results/jetson/e2e_las_summary.json（C2/D/E）。
可重复运行：所有数值都从原始记录重新推导，不依赖上一次的输出；两个顶层 note 每次都整体重写
（不是追加），所以连续运行两次的输出逐字节相同。
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
    "A（deploy_scene.py）顺序执行。total_ms 的计算：①每次运行把该次各正式分段的耗时相加；"
    "②对 10 次运行的和取中位数。各分段计时都在主线程上顺序发生、互不重叠，且不包含分段之间"
    "未被计时的胶水代码，所以每次运行的“分段之和”小于等于该次运行真实的整景墙钟；中位数对"
    "逐项不等式单调，因此 A 的 total_ms 小于等于 A 真实整景墙钟耗时的中位数。当时没有记录 A "
    "的外层整景墙钟，这个差值无法恢复。以 E 的整景墙钟为分母时，A.total_ms / E.total_ms 只能"
    "作为真实加速比的保守下界（timing_boundary=per_run_segment_sum_lower_bound），展示时要向下"
    "截断到两位小数，不能四舍五入。deploy_scene.py 自身报告的“各段中位数之和”（与上面①②的顺序"
    "相反：先取每段中位数再相加）保留在 total_ms_sum_of_segment_medians / "
    "total_without_io_ms_sum_of_segment_medians，只作来源追溯，不能当作比较用的正式值，也不能"
    "用它与 total_ms 的差值反推未计时胶水代码的耗时（两者是不同的聚合顺序，不是同一件事的两次测量）。"
)
BE_DEFINITION = (
    "整景墙钟中位数（timing_boundary=whole_scene_wall_clock）：每次整景整体计时"
    "（pipe.run 前后同步并计时，含文件读取，页缓存热），10 次取中位数。"
)

FULL_NOTE = (
    "A=deploy_scene.py（x86 最终链路原样）；B=deploy_jetson.py 顺序执行；C=B+LAS 与 GPU 推理重叠。"
    "各自独立进程，顺序 A→B→C（fwd）再 C→B→A（rev）；每个 JSON 内 warmup 2 + 重复 10 次取中位数。"
    "A/B/C 的 total_statistic 都是 median_of_run_totals（10 次运行取中位数），但 timing_boundary "
    "不同：B/C 是整景外层墙钟（whole_scene_wall_clock）；A 是每次运行各分段耗时之和"
    "（per_run_segment_sum_lower_bound），小于等于其真实整景墙钟。因此 A.total_ms / E.total_ms "
    "（E 见 e2e_las_summary.json）只能作为真实加速比的保守下界，展示时向下截断到两位小数，"
    "详见每条记录的 total_definition。"
)
LAS_NOTE = (
    "C2=第二批复测的 C（B + LAS 与 GPU 推理重叠，用于确认批次间无漂移）；D=C2 + kNN 查询多核；"
    "E=D + kd 树构建与投影并行（最终配置，deploy_jetson.py 默认）。各自独立进程，每景 C2→D→E（fwd）"
    "再 E→D→C2（rev）；每个 JSON 内 warmup 2 + 重复 10 次取中位数，total_ms 为整景墙钟中位数"
    "（timing_boundary=whole_scene_wall_clock）。"
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
        "timing_boundary": "per_run_segment_sum_lower_bound",
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
    return {"total_statistic": "median_of_run_totals", "timing_boundary": "whole_scene_wall_clock",
            "total_definition": BE_DEFINITION}


def update_file(path: Path, cfgs: list[str], note: str) -> None:
    summary = json.loads(path.read_text())
    for sc in SCENES:
        tag = sc.replace("/", "_")
        for cfg in cfgs:
            for rnd in ("fwd", "rev"):
                entry = summary["scenes"][sc][cfg][rnd]
                patch = recompute_a(tag, rnd) if cfg == "A" else recompute_be(cfg, tag, rnd)
                entry.update(patch)
    summary["note"] = note  # 每次整体重写，不追加，保证脚本重复运行时输出逐字节相同
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("written ->", path)


if __name__ == "__main__":
    print("== e2e_full_summary.json (A/B/C) ==")
    update_file(FULL_JSON, ["A", "B", "C"], FULL_NOTE)
    print("== e2e_las_summary.json (C2/D/E) ==")
    update_file(LAS_JSON, ["C2", "D", "E"], LAS_NOTE)
