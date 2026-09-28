"""生成 README 用的图表与结果总表，所有数字在运行时从 results/*.json 读取。

运行（需要 matplotlib）：
    python scripts/make_readme_assets.py

输出：
    assets/e2e_latency.png    整景端到端耗时对比（stage7_final_e2e.json）
    assets/cpu_matching.png   CPU 匹配段优化前后（stage7_cpu_profile.json）
    assets/results_table.md   结果总表（精度取 stage2_trt_accuracy.json，延迟/加速比取 stage3_benchmark.json）

生成前会检查关键数字与 README.md 里已有的写法一致，不一致则报错退出，不写任何输出。
运行结束打印 provenance 表：图表里出现的每个数字对应的 JSON 文件与字段路径。
"""
from __future__ import annotations

import json
import sys
from decimal import ROUND_FLOOR, Decimal, localcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ASSETS = ROOT / "assets"

F2 = "results/stage2_trt_accuracy.json"
F3 = "results/stage3_benchmark.json"
FE = "results/stage7_final_e2e.json"
FC = "results/stage7_cpu_profile.json"
FJ = "results/jetson/opt_final2_compare.json"
FJE = "results/jetson/opt_final2_energy.json"
FJB0 = "results/jetson/logs/nsys/o0_S0_gpu_busy.json"
FJB1 = "results/jetson/logs/nsys/final_S11b_gpu_busy.json"
FJM = "results/jetson/opt_final2_multiscene_{}.json"
FJF = "results/jetson/e2e_full_summary.json"
FJL = "results/jetson/e2e_las_summary.json"
FX8 = "results/x86_int8_implicit_verification.json"
FCE = "results/jetson/cpp_e2e_summary.json"
FCL = "results/jetson/cpp_latency_summary.json"
FCP = "results/jetson/cpp_power_summary.json"

# 配色：reference palette 的 categorical slot 1 / 2（light 模式，色值不改）
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_BEFORE, C_AFTER = "#2a78d6", "#eb6834"
C_SLOT3 = "#1baf7a"  # categorical slot 3（aqua）；浅色底上对比度 <3:1，按规范所有段都直接标数值

PROV: list[tuple] = []


def rec(where: str, shown: str, file: str, field: str, value, derived: str = "") -> float:
    PROV.append((where, shown, file, field, value, derived))
    return value


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def lower_bound_2dp(a: Decimal, e: Decimal) -> Decimal:
    """A/E 的保守下界，向下截断到两位小数——不用 round()、`:.2f`（会四舍五入）或
    floor(float*100)/100（float 乘法可能进位）。a、e 必须是从原始 JSON 文本用
    parse_float=Decimal 读出的 Decimal，除法与截断都在 ROUND_FLOOR 下进行，
    保证返回值不高于真实比值（下面用 assert 再核实一次）。"""
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.rounding = ROUND_FLOOR
        raw = a / e
        shown = raw.quantize(Decimal("0.01"), rounding=ROUND_FLOOR)
    assert shown <= raw, f"下界截断错误：{shown} > {raw}"
    return shown


def check_readme(readme: str, desc: str, needle: str) -> None:
    if needle not in readme:
        sys.exit(f"README 与 JSON 不一致（{desc}）：README 中找不到 {needle!r}")


# ------------------------------------------------------------------ 读数据
readme = (ROOT / "README.md").read_text(encoding="utf-8")
s2 = load("stage2_trt_accuracy.json")
s3 = load("stage3_benchmark.json")
e2e = load("stage7_final_e2e.json")
cpu = load("stage7_cpu_profile.json")

# ---- 图1：整景端到端（forward 方向，与 README 的 714.7ms → 331.7ms 同口径）
S0, S7 = "S0_stage4_original", "S7_final_ckdtree_unbalanced"
IO_SEGS = {"load_hsi_io", "laspy_read_io"}
fwd = e2e["forward"]
seg_keys = [k for k in fwd[S0]["segments_ms"] if k not in IO_SEGS]
tot0 = fwd[S0]["total_without_io_median_ms"]
tot7 = fwd[S7]["total_without_io_median_ms"]
speedup_e2e = e2e["overall_speedup_without_io"]
for s, tot in ((S0, tot0), (S7, tot7)):
    seg_sum = sum(fwd[s]["segments_ms"][k]["median_ms"] for k in seg_keys)
    if abs(seg_sum - tot) > 1e-6:
        sys.exit(f"{s}：分段中位数之和 {seg_sum} 与 total_without_io_median_ms {tot} 不一致")

# ---- 图2：匹配段
ob = cpu["optimization_b_merged_score_window"]
b_med, a_med = ob["before_ms"]["median_ms"], ob["after_ms"]["median_ms"]
b_min, a_min = ob["before_ms"]["min_ms"], ob["after_ms"]["min_ms"]
if ob["identical"] is not True:
    sys.exit("optimization_b_merged_score_window.identical 不是 true，无法标注 bit-identical")

# ---- 与 README 已有数字对账
for model, label in (("pc", "PC"), ("hsi", "HSI")):
    f16 = s2["results"][model]["fp16"]
    check_readme(readme, f"{label} FP16 精度",
                 f"{label} {f16['cos_sim_min']:.6f} / {f16['top1_agreement'] * 100:.1f}%")
    b1 = {e["batch"]: e for e in s3["results"][model]["pytorch_fp32_gpu"]}[1]["p50_ms"]
    t1 = {e["batch"]: e for e in s3["results"][model]["trt_fp16"]}[1]["p50_ms"]
    check_readme(readme, f"{label} 单样本加速比", f"{label} {b1 / t1:.1f}x")
check_readme(readme, "整景耗时与加速比", f"{tot0:.1f}ms → {tot7:.1f}ms（**{speedup_e2e:.2f}x**）")
check_readme(readme, "匹配段耗时与倍数", f"{b_med:.1f}ms → {a_med:.1f}ms（{b_med / a_med:.2f}x")
# ---- x86 隐式 INT8 核实：所有构建都没有 Int8 层
x8 = load("x86_int8_implicit_verification.json")
n8 = [b["layers_with_int8"] for m in ("pc", "hsi") for b in x8[m]["int8_implicit"]]
if any(n8):
    sys.exit(f"x86_int8_implicit_verification.json 里出现了含 Int8 的层：{n8}，README 的更正说明需要重写")
rec("Tech note 3 / Int8 layers in int8_implicit builds", "0", FX8, "<pc|hsi>.int8_implicit[*].layers_with_int8", max(n8))
check_readme(readme, "隐式 INT8 更正说明", "逐层统计全部为 0 个 Int8 层")

# ---- Jetson：累加对比、能效、GPU 忙碌率、多场景
jc = json.loads((ROOT / FJ).read_text(encoding="utf-8"))
jcf = jc["rounds"]["forward"]
J_CFGS = list(jcf)
J0, JN = J_CFGS[0], J_CFGS[-1]
j_tot0, j_totn = jcf[J0]["total_ms_median"], jcf[JN]["total_ms_median"]
j_sp = jc["speedup_total"]["forward"][JN]
je = json.loads((ROOT / FJE).read_text(encoding="utf-8"))
jb0 = json.loads((ROOT / FJB0).read_text(encoding="utf-8"))["scene_S0"]
jb1 = json.loads((ROOT / FJB1).read_text(encoding="utf-8"))[f"scene_{JN}"]
busy0 = jb0["gpu_busy_ms"] / jb0["wall_ms"] * 100
busy1 = jb1["gpu_busy_ms"] / jb1["wall_ms"] * 100
for n, a in jc["accuracy_vs_fp32"].items():
    if not all(v["all_near_ties"] for v in a["match"].values()):
        sys.exit(f"Jetson {n}：存在非近平局的不一致点，不能标注“精度不变”")
if not all(jc["frontend_guardrail"].values()):
    sys.exit("Jetson 前处理护栏未通过")
J_SCENES = ["24data_10.6_1", "24data_10.6_10", "25data_8.4_1", "25data_10.22_1"]
jm = {sc: json.loads((ROOT / FJM.format(sc)).read_text(encoding="utf-8")) for sc in J_SCENES}
j_ms_sp = [jm[sc]["speedup_total"]["forward"][JN] for sc in J_SCENES]
check_readme(readme, "Jetson 整景耗时与加速比", f"{j_tot0:.1f}ms → {j_totn:.1f}ms（**{j_sp:.2f}x**）")
check_readme(readme, "Jetson 多场景加速比范围", f"{min(j_ms_sp):.1f}～{max(j_ms_sp):.1f}x")
check_readme(readme, "Jetson GPU 忙碌率", f"{busy0:.1f}% → {busy1:.1f}%")
check_readme(readme, "Jetson 每景能耗",
             f"{je[J0]['energy_per_scene_j_total_board']:.1f}J → {je[JN]['energy_per_scene_j_total_board']:.1f}J")
if not all(v for sc in J_SCENES for v in jm[sc].get("valid_mask_identical_to_cache", {}).values()):
    sys.exit("Jetson 多场景：掩膜护栏未通过")

# ---- Jetson 完整端到端（含文件读取与 LAS）：A=x86 最终链路原样，B..E=Jetson 链路逐步优化；每个 JSON 都已核对匹配行列与缓存 S11b 逐位相同
jf = json.loads((ROOT / FJF).read_text(encoding="utf-8"))["scenes"]
jl = json.loads((ROOT / FJL).read_text(encoding="utf-8"))["scenes"]
for summ in (jf, jl):
    for sc, ent in summ.items():
        for c, rounds in ent.items():
            for rnd, r in rounds.items():
                if not all(r["match_rowcols_identical_to_cached_S11b"].values()):
                    sys.exit(f"完整端到端 {sc} {c} {rnd}：匹配行列与缓存 S11b 不一致")
                if r["preprocess_guardrail_vs_cache"] is not None and not all(r["preprocess_guardrail_vs_cache"].values()):
                    sys.exit(f"完整端到端 {sc} {c} {rnd}：前处理护栏未通过")
                # A（各段中位数之和）与 B~E（整景墙钟中位数）曾是两种不同的统计量，直接相除会混用聚合方式；
                # p5_a_run_totals.py 已把两边的聚合方式统一成"10 次运行取中位数"，这里要求每条记录显式声明，缺失就拒绝算加速比。
                if r.get("total_statistic") != "median_of_run_totals":
                    sys.exit(f"完整端到端 {sc} {c} {rnd}：total_statistic 不是 median_of_run_totals（{r.get('total_statistic')!r}），"
                             "拒绝计算加速比——先用 results/jetson/logs/p5_a_run_totals.py 统一聚合方式")
                # 聚合方式相同不代表计时覆盖的范围相同：A 目前是"每次运行各分段之和"，B~E 是"整景外层墙钟"，
                # 二者相除只能得到保守下界，不是精确 speedup。每条记录必须显式声明 timing_boundary，
                # 下面据此决定 A→E 的展示方式；B~E（作分母时必须是真实墙钟）不允许是下界口径。
                tb = r.get("timing_boundary")
                if c == "A":
                    if tb not in ("whole_scene_wall_clock", "per_run_segment_sum_lower_bound"):
                        sys.exit(f"完整端到端 {sc} {c} {rnd}：timing_boundary 不合法（{tb!r}）")
                else:
                    if tb != "whole_scene_wall_clock":
                        sys.exit(f"完整端到端 {sc} {c} {rnd}：timing_boundary 不是 whole_scene_wall_clock（{tb!r}），"
                                 "不能作为加速比的分母/分子")
F_SC = "24data/10.6/1"
FULL = {"A": jf[F_SC]["A"]["fwd"], "B": jf[F_SC]["B"]["fwd"], "C": jf[F_SC]["C"]["fwd"],
        "D": jl[F_SC]["D"]["fwd"], "E": jl[F_SC]["E"]["fwd"]}
full_sp = {sc: jf[sc]["A"]["fwd"]["total_ms"] / jl[sc]["E"]["fwd"]["total_ms"] for sc in jf}
a_lb = any(jf[sc]["A"][rd]["timing_boundary"] == "per_run_segment_sum_lower_bound"
           for sc in jf for rd in ("fwd", "rev"))

if a_lb:
    # 只用于计算 A/E 下界：从 JSON 原始文本按 Decimal 解析，避免与其余按 float 解析、用于画图的
    # jf/jl 混算；除法与截断全部在 ROUND_FLOOR 下进行，保证展示值不高于原始比值。
    jf_dec = json.loads((ROOT / FJF).read_text(encoding="utf-8"), parse_float=Decimal)["scenes"]
    jl_dec = json.loads((ROOT / FJL).read_text(encoding="utf-8"), parse_float=Decimal)["scenes"]
    lb_by_scene = {sc: lower_bound_2dp(jf_dec[sc]["A"]["fwd"]["total_ms"], jl_dec[sc]["E"]["fwd"]["total_ms"])
                   for sc in jf}
    for sc in jf:
        rec(f"FigJ3 / A→E lower bound / {sc}", f"≥{lb_by_scene[sc]}x", f"{FJF}, {FJL}",
            f"scenes.{sc}.A.fwd.total_ms / scenes.{sc}.E.fwd.total_ms（Decimal, ROUND_FLOOR, 截断到 2 位）",
            full_sp[sc], "原始比值（float，仅供核对，不是展示值）")
    lb_default = lb_by_scene[F_SC]
    lb_lo, lb_hi = min(lb_by_scene.values()), max(lb_by_scene.values())
    check_readme(readme, "Jetson 完整端到端（默认景）",
                 f"{FULL['A']['total_ms']:.1f}ms → {FULL['E']['total_ms']:.1f}ms（≥{lb_default}x，保守下界）")
    check_readme(readme, "Jetson 完整端到端 4 景范围", f"4 个场景的保守下界为 {lb_lo}～{lb_hi}x")

    # 防漏标 ①：旧的"精确默认景倍数"写法（四舍五入两位小数、无 ≥ 前缀）不得残留。
    # 括号+两位小数+右括号的组合是安全的判据：正确写法在左括号后紧跟"≥"，不会被误伤。
    for sc in jf:
        stale = f"（{full_sp[sc]:.2f}x）"
        if stale in readme:
            sys.exit(f"README 残留未标注下界的精确加速比写法：{stale!r}（场景 {sc}）")
    # 防漏标 ②：旧的"精确范围"写法（一位小数、无下界前缀）不得残留。
    stale_range = f"{min(full_sp.values()):.1f}～{max(full_sp.values()):.1f}x"
    if stale_range in readme:
        sys.exit(f"README 残留未标注下界的精确范围写法：{stale_range!r}")
    # 防漏标 ③：裸的每景四舍五入倍数不得出现——但如果四舍五入值恰好等于截断值（如 3.47），
    # 这个字符串就是正确下界文本"≥3.47x"的子串，禁止会误伤，所以跳过。
    for sc in jf:
        rounded = f"{full_sp[sc]:.2f}x"
        if rounded != f"{lb_by_scene[sc]}x" and rounded in readme:
            sys.exit(f"README 残留未标注下界的四舍五入倍数：{rounded!r}（场景 {sc}）")
    # 防漏标 ④：范围数字出现次数必须与带"保守下界为"前缀的出现次数一致，防止裸范围漏标下界说明。
    bare_range = f"{lb_lo}～{lb_hi}x"
    prefixed_range = f"保守下界为 {bare_range}"
    if readme.count(bare_range) != readme.count(prefixed_range):
        sys.exit(f"README 里 {bare_range!r} 出现次数与带“保守下界为”前缀的次数不一致，可能有未标注下界的裸范围")
else:
    # 将来如果 A 也有了真实整景墙钟（timing_boundary 全部是 whole_scene_wall_clock），
    # 恢复精确 speedup 的展示方式。
    check_readme(readme, "Jetson 完整端到端（默认景）",
                 f"{FULL['A']['total_ms']:.1f}ms → {FULL['E']['total_ms']:.1f}ms（{full_sp[F_SC]:.2f}x）")
    check_readme(readme, "Jetson 完整端到端 4 景范围", f"{min(full_sp.values()):.1f}～{max(full_sp.values()):.1f}x")

# ---- Jetson C++ 部署：P=Python 最终配置 E，K0..K5 为 C++ 的累加配置；每个 JSON 都已核对匹配行列与同景 P 逐位相同
ce = json.loads((ROOT / FCE).read_text(encoding="utf-8"))["scenes"]
for sc, ent in ce.items():
    for c, rounds in ent.items():
        for rnd, r in rounds.items():
            if not all(r["match_rowcols_identical_to_P"].values()) or r["deterministic"] is False:
                sys.exit(f"C++ 完整端到端 {sc} {c} {rnd}：匹配行列与 Python 版不一致或不确定")
CE_SC = "24data/10.6/1"
CE_CFGS = ["P", "K0", "K1", "K2", "K3", "K4", "K5"]
CE = {c: ce[CE_SC][c]["fwd"] for c in CE_CFGS}
ce_sp = {sc: ce[sc]["P"]["fwd"]["total_ms"] / ce[sc]["K4"]["fwd"]["total_ms"] for sc in ce}
check_readme(readme, "C++ 完整端到端（默认景）", f"{CE['P']['total_ms']:.1f}ms → {CE['K4']['total_ms']:.1f}ms（{ce_sp[CE_SC]:.2f}x）")
check_readme(readme, "C++ 完整端到端 4 景范围", f"{min(ce_sp.values()):.1f}～{max(ce_sp.values()):.1f}x")
cl = json.loads((ROOT / FCL).read_text(encoding="utf-8"))
cp = json.loads((ROOT / FCP).read_text(encoding="utf-8"))
CL_KEYS = ["pc_b1", "pc_b8", "hsi_b1", "hsi_b8"]
_lat = lambda k, lang, mode: cl["models"][k]["mean_of_rounds"][lang][mode]["request"]  # noqa: E731
check_readme(readme, "C++ 单样本请求延迟（PC/HSI, batch=1）",
             f"PC {_lat('pc_b1', 'python', 'graph'):.3f} → {_lat('pc_b1', 'cpp', 'graph'):.3f}ms、HSI {_lat('hsi_b1', 'python', 'graph'):.3f} → {_lat('hsi_b1', 'cpp', 'graph'):.3f}ms")
_c = cp["cold_start"]
check_readme(readme, "C++ 进程冷启动", f"{_c['P']['process_wall_s']['median']:.2f}s → {_c['K4']['process_wall_s']['median']:.2f}s")
check_readme(readme, "C++ 每景能耗", f"{cp['energy']['by_config']['P']['energy_per_scene_j_total_board']:.2f}J → {cp['energy']['by_config']['K4']['energy_per_scene_j_total_board']:.2f}J")


# ------------------------------------------------------------------ 绘图
def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


SEG_NAMES = {
    "standardize_and_mask": "Standardize + mask",
    "patch_build": "HSI patch build",
    "crs_transformer_init": "CRS transformer init",
    "projection": "LAS projection",
    "mask_filter": "Mask filter",
    "rng_sample": "RNG sampling",
    "ckdtree_build": "cKDTree build",
    "ckdtree_query_offsets": "cKDTree query",
    "hsi_infer": "HSI inference",
    "pc_infer": "PC inference",
    "feature_grid_assemble": "Feature grid assemble",
    "match": "Cross-modal matching",
}


def fig_e2e() -> None:
    rows = sorted(seg_keys, key=lambda k: fwd[S0]["segments_ms"][k]["median_ms"], reverse=True)
    fig = plt.figure(figsize=(11.5, 5.2), facecolor=SURFACE)
    ax_l = fig.add_axes([0.07, 0.11, 0.20, 0.67])
    ax_r = fig.add_axes([0.45, 0.11, 0.52, 0.67])

    # 左：总耗时
    v0 = rec("Fig1 left / Original", f"{tot0:.1f} ms", FE, f"forward.{S0}.total_without_io_median_ms", tot0)
    v7 = rec("Fig1 left / Optimized", f"{tot7:.1f} ms", FE, f"forward.{S7}.total_without_io_median_ms", tot7)
    sp = rec("Fig1 left / speedup", f"{speedup_e2e:.2f}x", FE, "overall_speedup_without_io", speedup_e2e)
    style_axes(ax_l)
    ax_l.bar([0, 1], [v0, v7], width=0.42, color=[C_BEFORE, C_AFTER], zorder=3)
    ymax = max(v0, v7) * 1.18
    ax_l.set_ylim(0, ymax)
    ax_l.set_xlim(-0.55, 1.55)
    ax_l.set_xticks([0, 1], ["Original", "Optimized"])
    ax_l.set_ylabel("Latency (ms)")
    ax_l.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    for x, v in ((0, v0), (1, v7)):
        ax_l.text(x, v + ymax * 0.015, f"{v:.1f} ms", ha="center", va="bottom", color=INK, fontsize=10)
    ax_l.text(1, ymax * 0.70, f"{sp:.2f}x", ha="center", va="center", color=INK, fontsize=17, weight="bold")
    ax_l.text(1, ymax * 0.63, "faster", ha="center", va="center", color=INK2, fontsize=9.5)

    # 右：分段
    style_axes(ax_r)
    xmax = max(fwd[s]["segments_ms"][k]["median_ms"] for s in (S0, S7) for k in seg_keys)
    for i, k in enumerate(rows):
        name = SEG_NAMES.get(k, k)
        m0 = fwd[S0]["segments_ms"][k]["median_ms"]
        m7 = fwd[S7]["segments_ms"][k]["median_ms"]
        rec(f"Fig1 right / {name} / Original", f"{m0:.1f}", FE, f"forward.{S0}.segments_ms.{k}.median_ms", m0)
        rec(f"Fig1 right / {name} / Optimized", f"{m7:.1f}", FE, f"forward.{S7}.segments_ms.{k}.median_ms", m7)
        ax_r.barh(i - 0.19, m0, height=0.32, color=C_BEFORE, zorder=3)
        ax_r.barh(i + 0.19, m7, height=0.32, color=C_AFTER, zorder=3)
        ax_r.text(max(m0, m7) + xmax * 0.015, i, f"{m0:.1f} → {m7:.1f}",
                  va="center", ha="left", color=INK2, fontsize=8.5)
    ax_r.set_yticks(range(len(rows)), [SEG_NAMES.get(k, k) for k in rows])
    ax_r.invert_yaxis()
    ax_r.set_xlim(0, xmax * 1.30)
    ax_r.set_xlabel("Median latency per segment (ms)")
    ax_r.grid(axis="x", color=GRID, linewidth=1, zorder=0)

    fig.text(0.07, 0.94, "Full-scene end-to-end latency (excluding file IO)",
             color=INK, fontsize=14, weight="bold", ha="left")
    fig.text(0.07, 0.895, "Original = baseline full-scene pipeline; Optimized = final pipeline; row label: original → optimized (ms)",
             color=INK2, fontsize=9.5, ha="left")
    fig.text(0.07, 0.855, "Within-pipeline segment medians; matching values are not directly comparable with the isolated benchmark below",
             color=INK2, fontsize=8.8, ha="left")
    fig.legend(handles=[Patch(color=C_BEFORE, label="Original pipeline"),
                        Patch(color=C_AFTER, label="Optimized pipeline")],
               loc="upper left", bbox_to_anchor=(0.065, 0.82), ncol=2, frameon=False,
               fontsize=9.5, labelcolor=INK2, handlelength=1.2)
    fig.savefig(ASSETS / "e2e_latency.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_cpu() -> None:
    base = "optimization_b_merged_score_window"
    bm = rec("Fig2 / Before bar", f"{b_med:.1f} ms", FC, f"{base}.before_ms.median_ms", b_med)
    am = rec("Fig2 / After bar", f"{a_med:.1f} ms", FC, f"{base}.after_ms.median_ms", a_med)
    bn = rec("Fig2 / Before min marker", "", FC, f"{base}.before_ms.min_ms", b_min)
    an = rec("Fig2 / After min marker", "", FC, f"{base}.after_ms.min_ms", a_min)
    rec("Fig2 / bit-identical note", "bit-identical output", FC, f"{base}.identical", ob["identical"])
    ratio = rec("Fig2 / speedup", f"{bm / am:.2f}x", FC, f"{base}.before_ms.median_ms / {base}.after_ms.median_ms",
                bm / am, derived="before.median_ms / after.median_ms")

    fig = plt.figure(figsize=(6.6, 4.8), facecolor=SURFACE)
    ax = fig.add_axes([0.13, 0.12, 0.83, 0.62])
    style_axes(ax)
    ax.bar([0, 1], [bm, am], width=0.42, color=[C_BEFORE, C_AFTER], zorder=3)
    ymax = bm * 1.2
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["Before\n(variants scored separately)", "After\n(merged window scoring)"])
    ax.set_ylabel("Latency (ms)")
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    for x, v in ((0, bm), (1, am)):
        ax.text(x, v + ymax * 0.015, f"{v:.1f} ms", ha="center", va="bottom", color=INK, fontsize=10)
    for x, v in ((0, bn), (1, an)):
        ax.plot([x], [v], marker="D", markersize=6, markerfacecolor=SURFACE, markeredgecolor=INK,
                markeredgewidth=1.4, linestyle="none", zorder=4)
    ax.text(0.5, ymax * 0.66, f"{ratio:.2f}x", ha="center", va="center", color=INK, fontsize=17, weight="bold")
    ax.text(0.5, ymax * 0.58, "faster", ha="center", va="center", color=INK2, fontsize=9.5)
    ax.text(0.5, ymax * 0.50, "bit-identical output", ha="center", va="center", color=INK2, fontsize=9)

    fig.text(0.05, 0.93, "CPU matching stage: before vs after", color=INK, fontsize=13, weight="bold", ha="left")
    fig.text(0.05, 0.885, "Fixed features; NumPy separate vs merged calls; 2 untimed pre-runs + 10 timed runs (median)",
             color=INK2, fontsize=8.8, ha="left")
    fig.legend(handles=[Patch(color=C_BEFORE, label="Before"), Patch(color=C_AFTER, label="After"),
                        Line2D([0], [0], marker="D", markersize=6, markerfacecolor=SURFACE,
                               markeredgecolor=INK, markeredgewidth=1.4, linestyle="none",
                               label="Fastest run (min)")],
               loc="upper left", bbox_to_anchor=(0.045, 0.865), ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK2, handlelength=1.2)
    fig.savefig(ASSETS / "cpu_matching.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


J_LABELS = {
    "S0": "S0  baseline (validation-tool pipeline)",
    "S1": "S1  + infer valid pixels only",
    "S2": "S2  + merged NumPy matching",
    "S3": "S3  + buffer-reusing TRT runner",
    "S4": "S4  + large-batch scene engine (1024)",
    "S5": "S5  + unified-memory zero-copy",
    "S6": "S6  + matching on GPU",
    "S7": "S7  + HSI patch gather on GPU",
    "S8": "S8  + features stay on GPU",
    "S9": "S9  + PC inference on 2nd stream",
    "S10": "S10 + single-shot matching",
    "S11b": "S11b + element-wise standardize on GPU",
}
J_GROUPS = [("CPU / data prep", ("prep", "patch_build", "assemble"), C_BEFORE),
            ("TensorRT inference", ("hsi_infer", "pc_infer"), C_AFTER),
            ("Cross-modal matching", ("match",), C_SLOT3)]


def fig_jetson_chain() -> None:
    fig = plt.figure(figsize=(11.5, 6.4), facecolor=SURFACE)
    ax = fig.add_axes([0.30, 0.09, 0.66, 0.70])
    style_axes(ax)
    xmax = max(jcf[c]["total_ms_median"] for c in J_CFGS)
    for i, c in enumerate(J_CFGS):
        seg = jcf[c]["segments_ms_median"]
        left = 0.0
        for gname, keys, color in J_GROUPS:
            v = sum(seg[k] for k in keys)
            rec(f"FigJ1 / {c} / {gname}", f"{v:.1f}", FJ, f"rounds.forward.{c}.segments_ms_median.{'+'.join(keys)}", v)
            ax.barh(i, v, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            left += v
        tot = rec(f"FigJ1 / {c} / total", f"{jcf[c]['total_ms_median']:.1f} ms", FJ,
                  f"rounds.forward.{c}.total_ms_median", jcf[c]["total_ms_median"])
        sp = rec(f"FigJ1 / {c} / speedup", f"{jc['speedup_total']['forward'][c]:.2f}x", FJ,
                 f"speedup_total.forward.{c}", jc["speedup_total"]["forward"][c])
        ax.text(max(left, tot) + xmax * 0.012, i, f"{tot:.0f} ms  ·  {sp:.1f}x", va="center", ha="left",
                color=INK if c in (J0, JN) else INK2, fontsize=9, weight="bold" if c in (J0, JN) else "normal")
    ax.set_yticks(range(len(J_CFGS)), [J_LABELS.get(c, c) for c in J_CFGS])
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.22)
    ax.set_xlabel("Whole-scene latency on Jetson Orin Nano Super (ms, median of 10 runs, forward round)")
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    for lab in ax.get_yticklabels():
        lab.set_color(INK); lab.set_fontsize(9)
    fig.text(0.02, 0.945, f"Jetson whole-scene pipeline: {j_tot0:.0f} ms → {j_totn:.0f} ms ({j_sp:.1f}x)",
             color=INK, fontsize=14, weight="bold", ha="left")
    fig.text(0.02, 0.905, "Cumulative single-variable steps in one process; each step bit-identical to the previous one "
             "except the engine switch (S4); all mismatches vs PyTorch FP32 are near-ties",
             color=INK2, fontsize=9, ha="left")
    fig.text(0.02, 0.873, "Scene 24data/10.6/1 (9,645 valid HSI patches, 1,000 points); from cached raw HSI, excluding file IO / LAS; "
             "stacked bars = segment medians, label = total median",
             color=INK2, fontsize=9, ha="left")
    fig.legend(handles=[Patch(color=c, label=g) for g, _, c in J_GROUPS], loc="upper left",
               bbox_to_anchor=(0.015, 0.855), ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.2)
    fig.savefig(ASSETS / "jetson_opt_chain.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_jetson_gpu_energy() -> None:
    b0 = rec("FigJ2 left / S0 GPU busy", f"{busy0:.1f}%", FJB0, "scene_S0.gpu_busy_ms / scene_S0.wall_ms", busy0,
             derived="nsys kernel+memcpy union / NVTX range wall")
    b1 = rec(f"FigJ2 left / {JN} GPU busy", f"{busy1:.1f}%", FJB1, f"scene_{JN}.gpu_busy_ms / scene_{JN}.wall_ms", busy1,
             derived="nsys kernel+memcpy union / NVTX range wall")
    e0 = rec("FigJ2 right / S0 J per scene", f"{je[J0]['energy_per_scene_j_total_board']:.1f} J", FJE,
             f"{J0}.energy_per_scene_j_total_board", je[J0]["energy_per_scene_j_total_board"])
    e1 = rec(f"FigJ2 right / {JN} J per scene", f"{je[JN]['energy_per_scene_j_total_board']:.1f} J", FJE,
             f"{JN}.energy_per_scene_j_total_board", je[JN]["energy_per_scene_j_total_board"])
    w0, w1 = je[J0]["vdd_in_mean_mw"] / 1000, je[JN]["vdd_in_mean_mw"] / 1000
    rec("FigJ2 right / S0 mean power", f"{w0:.1f} W", FJE, f"{J0}.vdd_in_mean_mw", je[J0]["vdd_in_mean_mw"])
    rec(f"FigJ2 right / {JN} mean power", f"{w1:.1f} W", FJE, f"{JN}.vdd_in_mean_mw", je[JN]["vdd_in_mean_mw"])

    fig = plt.figure(figsize=(10.5, 4.4), facecolor=SURFACE)
    for k, (vals, ylabel, fmt, sub, pos) in enumerate((
            ((b0, b1), "GPU busy during one scene (%)", "{:.1f}%", "nsys: kernel + memcpy time / wall time", [0.07, 0.12, 0.38, 0.66]),
            ((e0, e1), "Energy per scene, whole board (J)", "{:.1f} J", f"mean board power {w0:.1f} W → {w1:.1f} W (idle {je['_idle_mean_mw'] / 1000:.1f} W)",
             [0.58, 0.12, 0.38, 0.66]))):
        ax = fig.add_axes(pos)
        style_axes(ax)
        ax.bar([0, 1], vals, width=0.46, color=[C_BEFORE, C_AFTER], zorder=3)
        ymax = max(vals) * 1.2
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1], [f"{J0} baseline", f"{JN} optimized"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
        for x, v in enumerate(vals):
            ax.text(x, v + ymax * 0.015, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=10)
        ax.text(0.5, 1.04, sub, transform=ax.transAxes, ha="center", va="bottom", color=INK2, fontsize=8.8)
    fig.text(0.02, 0.93, "Jetson: the GPU is kept busy, and each scene costs less energy", color=INK, fontsize=13,
             weight="bold", ha="left")
    fig.savefig(ASSETS / "jetson_gpu_energy.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


F_LABELS = {"A": "A  x86 final pipeline, ported as-is", "B": "B  Jetson pipeline, sequential",
            "C": "C  + LAS overlapped with GPU inference", "D": "D  + multi-core kNN query",
            "E": "E  + kd-tree build parallel to projection"}
F_IO = ("load_hsi_io", "laspy_read_io")
F_LAS = ("crs_transformer_init", "projection", "mask_filter", "rng_sample", "ckdtree_build", "ckdtree_build_wait",
         "ckdtree_query_offsets")


def fig_jetson_full() -> None:
    fig = plt.figure(figsize=(11.5, 4.6), facecolor=SURFACE)
    ax = fig.add_axes([0.30, 0.13, 0.66, 0.64])
    style_axes(ax)
    cfgs = list(FULL)
    xmax = max(FULL[c]["total_ms"] for c in cfgs)
    groups = [("File IO (HSI + LAS read)", F_IO, C_BEFORE), ("LAS: projection + kNN", F_LAS, C_AFTER),
              ("HSI prep, inference, matching", None, C_SLOT3)]
    for i, c in enumerate(cfgs):
        seg = FULL[c]["segments_ms"]
        src = FJF if c in "ABC" else FJL
        left = 0.0
        for gname, keys, color in groups:
            v = sum(seg.get(k, 0.0) for k in keys) if keys else sum(val for k, val in seg.items() if k not in F_IO + F_LAS)
            rec(f"FigJ3 / {c} / {gname}", f"{v:.1f}", src, f"scenes.{F_SC}.{c}.fwd.segments_ms", v)
            ax.barh(i, v, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            left += v
        tot = rec(f"FigJ3 / {c} / total", f"{FULL[c]['total_ms']:.1f} ms", src, f"scenes.{F_SC}.{c}.fwd.total_ms",
                  FULL[c]["total_ms"])
        ax.text(max(left, tot) + xmax * 0.012, i, f"{tot:.0f} ms", va="center", ha="left",
                color=INK if c in ("A", "E") else INK2, fontsize=9, weight="bold" if c in ("A", "E") else "normal")
    ax.set_yticks(range(len(cfgs)), [F_LABELS[c] for c in cfgs])
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.15)
    ax.set_xlabel("End-to-end latency from raw files on Jetson Orin Nano Super (ms; bars = per-segment median, "
                  "label = median of 10 whole-scene runs)")
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    for lab in ax.get_yticklabels():
        lab.set_color(INK); lab.set_fontsize(9)
    if a_lb:
        title_sp = f"(≥{lb_default}x, conservative lower bound)"
        caption = ("Scene 24data/10.6/1; each config in its own process; all match results bit-identical to the cached-input pipeline; "
                   "file cache warm; overlapped stages show CPU-side time; segment bars are per-segment medians and may not sum exactly "
                   "to the printed total (median of whole-scene run totals). A's total is the sum of its (sequential, non-overlapping) "
                   "segment timers per run, which is <= A's true whole-scene wall clock; B-E use the true whole-scene wall clock. "
                   "So A/E is only a conservative lower bound on the true speedup, truncated (not rounded) to 2 decimals.")
    else:
        title_sp = f"({full_sp[F_SC]:.1f}x)"
        caption = ("Scene 24data/10.6/1; each config in its own process; all match results bit-identical to the cached-input pipeline; "
                   "file cache warm; overlapped stages show CPU-side time; segment bars are per-segment medians and may not sum exactly "
                   "to the printed total (median of whole-scene run totals)")
    fig.text(0.02, 0.93, f"Jetson, raw files → match results: {FULL['A']['total_ms']:.0f} ms → {FULL['E']['total_ms']:.0f} ms "
             f"{title_sp}", color=INK, fontsize=14, weight="bold", ha="left")
    fig.text(0.02, 0.87, caption, color=INK2, fontsize=9, ha="left")
    fig.legend(handles=[Patch(color=c, label=g) for g, _, c in groups], loc="upper left",
               bbox_to_anchor=(0.015, 0.855), ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.2)
    fig.savefig(ASSETS / "jetson_full_e2e.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


CE_LABELS = {"P": "P  Python, final config (E)", "K0": "K0  C++, same schedule as E", "K1": "K1  + multi-thread mask / stats",
             "K2": "K2  + multi-thread projection", "K3": "K3  + bulk read of the HSI file", "K4": "K4  + cached PROJ transformer",
             "K5": "K5  + LAS overlapped with HSI read (rejected)"}


def fig_cpp_e2e() -> None:
    fig = plt.figure(figsize=(11.5, 4.9), facecolor=SURFACE)
    ax = fig.add_axes([0.30, 0.12, 0.66, 0.62])
    style_axes(ax)
    xmax = max(CE[c]["total_ms"] for c in CE_CFGS)
    for i, c in enumerate(CE_CFGS):
        tot = rec(f"FigC1 / {c} / total", f"{CE[c]['total_ms']:.1f} ms", FCE, f"scenes.{CE_SC}.{c}.fwd.total_ms", CE[c]["total_ms"])
        color = C_BEFORE if c == "P" else ("#9b9a96" if c == "K5" else C_AFTER)
        ax.barh(i, tot, height=0.6, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        sp = CE["P"]["total_ms"] / tot
        ax.text(tot + xmax * 0.012, i, f"{tot:.0f} ms" + (f"  ·  {sp:.2f}x" if c != "P" else ""), va="center", ha="left",
                color=INK if c in ("P", "K4") else INK2, fontsize=9, weight="bold" if c in ("P", "K4") else "normal")
    gpu = rec("FigC1 / GPU chain (K4)", f"{CE['K4']['segments_ms']['gpu_hsi_upload_patch_infer']:.0f} ms", FCE,
              f"scenes.{CE_SC}.K4.fwd.segments_ms.gpu_hsi_upload_patch_infer", CE["K4"]["segments_ms"]["gpu_hsi_upload_patch_infer"])
    ax.axvline(gpu, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(gpu + xmax * 0.008, -0.62, f"GPU chain (HSI upload + patch + inference): {gpu:.0f} ms", ha="left", va="bottom",
            color=INK2, fontsize=8.5)
    ax.set_yticks(range(len(CE_CFGS)), [CE_LABELS[c] for c in CE_CFGS])
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.2)
    ax.set_ylim(len(CE_CFGS) - 0.5, -0.75)
    ax.set_xlabel("End-to-end latency from raw files on Jetson Orin Nano Super (ms, median of 10 runs, forward round)")
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    for lab in ax.get_yticklabels():
        lab.set_color(INK); lab.set_fontsize(9)
    fig.text(0.02, 0.93, f"Jetson, raw files → match results, C++ vs Python: {CE['P']['total_ms']:.0f} ms → {CE['K4']['total_ms']:.0f} ms "
             f"({ce_sp[CE_SC]:.2f}x)", color=INK, fontsize=14, weight="bold", ha="left")
    fig.text(0.02, 0.875, "Scene 24data/10.6/1; each config in its own process; match rows/cols bit-identical to the Python pipeline in every run "
             "(4 scenes, both rounds); file cache warm", color=INK2, fontsize=9, ha="left")
    fig.savefig(ASSETS / "jetson_cpp_e2e.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_cpp_latency() -> None:
    fig = plt.figure(figsize=(11.5, 4.6), facecolor=SURFACE)
    ax = fig.add_axes([0.07, 0.12, 0.90, 0.58])
    style_axes(ax)
    series = [("python", "eager", C_BEFORE, 0.45), ("python", "graph", C_BEFORE, 1.0), ("cpp", "eager", C_AFTER, 0.45), ("cpp", "graph", C_AFTER, 1.0)]
    w = 0.19
    ymax = max(_lat(k, "python", "eager") for k in CL_KEYS)
    for gi, k in enumerate(CL_KEYS):
        for si, (lang, mode, color, alpha) in enumerate(series):
            v = rec(f"FigC2 / {k} / {lang} {mode}", f"{_lat(k, lang, mode):.3f} ms", FCL,
                    f"models.{k}.mean_of_rounds.{lang}.{mode}.request.median_ms(两轮均值)", _lat(k, lang, mode))
            x = gi + (si - 1.5) * w
            ax.bar(x, v, width=w * 0.92, color=color, alpha=alpha, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(x, v + ymax * 0.015, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=8.5)
    ax.set_xticks(range(len(CL_KEYS)), [k.replace("_b", "  batch ").replace("pc", "PC").replace("hsi", "HSI") for k in CL_KEYS])
    ax.set_ylim(0, ymax * 1.16)
    ax.set_ylabel("Request latency (ms)")
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    fig.text(0.02, 0.93, "Single-sample request latency on Jetson: Python vs C++ (TensorRT FP16 engines)", color=INK, fontsize=14, weight="bold", ha="left")
    fig.text(0.02, 0.895, "Request = H2D + inference + D2H + sync (median). Same engines and inputs, outputs bit-identical across all four modes;"
             "\nGPU kernel time is the same, C++ removes the per-call dispatch overhead", color=INK2, fontsize=9, ha="left", va="top")
    fig.legend(handles=[Patch(color=C_BEFORE, alpha=0.45, label="Python eager"), Patch(color=C_BEFORE, label="Python + CUDA Graph"),
                        Patch(color=C_AFTER, alpha=0.45, label="C++ eager"), Patch(color=C_AFTER, label="C++ + CUDA Graph")],
               loc="upper left", bbox_to_anchor=(0.015, 0.825), ncol=4, frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.2)
    fig.savefig(ASSETS / "jetson_cpp_latency.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ 结果总表
ROWS = {
    "pc": [("PyTorch", "FP32", "pytorch_fp32_gpu", None),
           ("TensorRT", "FP32 (TF32 off)", "trt_fp32_notf32", "fp32_notf32"),
           ("TensorRT", "FP32 (TF32 on)", "trt_fp32_tf32", "fp32_tf32"),
           ("TensorRT", "FP16 mixed", "trt_fp16", "fp16")],
    "hsi": [("PyTorch", "FP32", "pytorch_fp32_gpu", None),
            ("TensorRT", "FP32 (TF32 off)", "trt_fp32_notf32", "fp32_notf32"),
            ("TensorRT", "FP32 (TF32 on)", "trt_fp32_tf32", "fp32_tf32"),
            ("TensorRT", "FP16 mixed", "trt_fp16", "fp16"),
            ("TensorRT", "INT8 (QDQ)", "trt_int8_qdq", "int8_qdq")],
}
NOT_MEASURED = "未测量"


def bench_p50(model: str, backend: str, batch: int = 1):
    for e in s3["results"].get(model, {}).get(backend, []):
        if e["batch"] == batch:
            return e["p50_ms"]
    return None


def build_table() -> str:
    ns = {s2["results"][m][mode]["n"] for m in ROWS for _, _, _, mode in ROWS[m] if mode}
    if len(ns) != 1:
        sys.exit(f"stage2 各模式的样本数 n 不一致：{ns}")
    n = ns.pop()
    warmup, measure = s3["warmup"], s3["measure"]
    rec("Table caption / n", str(n), F2, "results.<model>.<mode>.n", n)
    rec("Table caption / warmup", str(warmup), F3, "warmup", warmup)
    rec("Table caption / measure", str(measure), F3, "measure", measure)

    lines = [
        f"对比基准为 PyTorch FP32（GPU）；精度在 {n} 条验证样本上计算，其中同模态最近邻 Top-1 一致率表示："
        "分别排除样本自身后，PyTorch 与 TensorRT 在同模态样本中检索到同一个最近邻的比例；"
        "延迟为 batch=1 的 p50"
        f"（warmup {warmup} / measure {measure}，CUDA event 计时）。"
        "表中 `[batch=1]` 表示列表里 `batch` 字段等于 1 的元素；字段级来源与补充验证信息见下方折叠 provenance。",
        "",
        "| Model | Backend | Precision | Accuracy (cos_sim_min / same-modal NN Top-1 agreement) | Latency (batch=1, p50, ms) | Speedup vs PyTorch FP32 |",
        "|---|---|---|---|---|---|",
    ]
    for model, rows in ROWS.items():
        name = model.upper()
        for backend, precision, bkey, mode in rows:
            where = f"Table / {name} {backend} {precision}"
            # 精度
            if mode is None:
                acc = "reference"
            else:
                a = s2["results"].get(model, {}).get(mode)
                if a is None:
                    acc = NOT_MEASURED
                else:
                    cos = rec(f"{where} / cos_sim_min", f"{a['cos_sim_min']:.6f}", F2,
                              f"results.{model}.{mode}.cos_sim_min", a["cos_sim_min"])
                    top1 = rec(f"{where} / top1", f"{a['top1_agreement'] * 100:.1f}%", F2,
                               f"results.{model}.{mode}.top1_agreement", a["top1_agreement"])
                    acc = f"{cos:.6f} / {top1 * 100:.1f}%"
            # 延迟 / 加速比
            p50 = bench_p50(model, bkey)
            base = bench_p50(model, "pytorch_fp32_gpu")
            if p50 is None:
                lat = spd = NOT_MEASURED
            else:
                rec(f"{where} / p50", f"{p50:.3f}", F3, f"results.{model}.{bkey}[batch=1].p50_ms", p50)
                lat = f"{p50:.3f}"
                if base is None:
                    spd = NOT_MEASURED
                else:
                    sp = base / p50
                    rec(f"{where} / speedup", f"{sp:.1f}x", F3,
                        f"results.{model}.pytorch_fp32_gpu[batch=1].p50_ms / results.{model}.{bkey}[batch=1].p50_ms",
                        sp, derived="pytorch p50 / row p50")
                    spd = f"{sp:.1f}x"
            lines.append(f"| {name} | {backend} | {precision} | {acc} | {lat} | {spd} |")

    table_note = ("PC 的 INT8 QDQ 路径已放弃，说明见「技术要点」第 3 条。原表中的两行 “INT8 (implicit calibration)” 已删除："
                  "逐层核查显示这类引擎中没有任何 Int8 层，数值行为与 FP16(auto) 构建一致，不是 INT8 结果（见「技术要点」第 3 条）。")
    if any(NOT_MEASURED in line for line in lines):
        table_note = (f"“{NOT_MEASURED}”表示 `stage2_trt_accuracy.json` / "
                      f"`stage3_benchmark.json` 中没有对应数值。{table_note}")
    lines += ["", table_note]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    fig_e2e()
    fig_cpu()
    fig_jetson_chain()
    fig_jetson_gpu_energy()
    fig_jetson_full()
    fig_cpp_e2e()
    fig_cpp_latency()
    (ASSETS / "results_table.md").write_text(build_table(), encoding="utf-8")

    print("| Where | Shown | File | Field | Raw value | Derived |")
    print("|---|---|---|---|---|---|")
    for where, shown, file, field, value, derived in PROV:
        print(f"| {where} | {shown} | {file} | {field} | {value!r} | {derived} |")


if __name__ == "__main__":
    main()
