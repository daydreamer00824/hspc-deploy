"""生成 README 用的图表与结果总表，所有数字在运行时从 results/*.json 读取。

运行（需要 matplotlib）：
    python scripts/make_readme_assets.py

输出：
    assets/e2e_latency.png    整景端到端耗时对比（stage7_final_e2e.json）
    assets/cpu_matching.png   CPU 匹配段优化前后（stage7_cpu_profile.json）
    assets/results_table.md   结果总表 + 脚注（精度取 stage2_trt_accuracy.json，延迟/加速比取 stage3_benchmark.json；
                              脚注另引用 stage7_mixed_precision.json 的稳定性验证与 hsi_qdq_check.json 的 QDQ 核查）

生成前会检查关键数字与 README.md 里已有的写法一致，不一致则报错退出，不写任何输出。
运行结束打印 provenance 表：图表里出现的每个数字对应的 JSON 文件与字段路径。
"""
from __future__ import annotations

import json
import sys
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
F7 = "results/stage7_mixed_precision.json"
FQ = "results/hsi_qdq_check.json"

# 配色：reference palette 的 categorical slot 1 / 2（light 模式，色值不改）
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_BEFORE, C_AFTER = "#2a78d6", "#eb6834"

PROV: list[tuple] = []


def rec(where: str, shown: str, file: str, field: str, value, derived: str = "") -> float:
    PROV.append((where, shown, file, field, value, derived))
    return value


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def check_readme(readme: str, desc: str, needle: str) -> None:
    if needle not in readme:
        sys.exit(f"README 与 JSON 不一致（{desc}）：README 中找不到 {needle!r}")


# ------------------------------------------------------------------ 读数据
readme = (ROOT / "README.md").read_text(encoding="utf-8")
s2 = load("stage2_trt_accuracy.json")
s3 = load("stage3_benchmark.json")
e2e = load("stage7_final_e2e.json")
cpu = load("stage7_cpu_profile.json")
s7 = load("stage7_mixed_precision.json")
qc = load("hsi_qdq_check.json")

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
check_readme(readme, "PC INT8 cos_sim_min",
             f"cos_sim_min {s2['results']['pc']['int8_implicit']['cos_sim_min']:.6f}")


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


# ------------------------------------------------------------------ 结果总表
ROWS = {
    "pc": [("PyTorch", "FP32", "pytorch_fp32_gpu", None),
           ("TensorRT", "FP32 (TF32 off)", "trt_fp32_notf32", "fp32_notf32"),
           ("TensorRT", "FP32 (TF32 on)", "trt_fp32_tf32", "fp32_tf32"),
           ("TensorRT", "FP16 mixed", "trt_fp16", "fp16"),
           ("TensorRT", "INT8 (implicit calibration)", "trt_int8_implicit", "int8_implicit")],
    "hsi": [("PyTorch", "FP32", "pytorch_fp32_gpu", None),
            ("TensorRT", "FP32 (TF32 off)", "trt_fp32_notf32", "fp32_notf32"),
            ("TensorRT", "FP32 (TF32 on)", "trt_fp32_tf32", "fp32_tf32"),
            ("TensorRT", "FP16 mixed", "trt_fp16", "fp16"),
            ("TensorRT", "INT8 (implicit calibration)", "trt_int8_implicit", "int8_implicit"),
            ("TensorRT", "INT8 (QDQ)", "trt_int8_qdq", "int8_qdq")],
}
NOT_MEASURED = "未测量"


def bench_p50(model: str, backend: str, batch: int = 1):
    for e in s3["results"].get(model, {}).get(backend, []):
        if e["batch"] == batch:
            return e["p50_ms"]
    return None


def fp16_stability_note(model: str, where: str) -> str:
    """同配置多次独立构建的 cos_sim_min 范围（stage7 稳定性验证），补充说明正式 engine 之外的构建也稳定。"""
    st = s7[model]["stability"]
    lo, hi, n = min(st["cos_mins"]), max(st["cos_mins"]), st["n"]
    fld = f"{model}.stability.cos_mins"
    rec(f"{where} / stability n", str(n), F7, f"{model}.stability.n", n)
    rec(f"{where} / stability min", f"{lo:.6f}", F7, fld, lo, derived="min of list")
    rec(f"{where} / stability max", f"{hi:.6f}", F7, fld, hi, derived="max of list")
    return f"；同配置另有 {n} 次独立构建的稳定性验证，cos_sim_min 为 {lo:.6f}–{hi:.6f}（`{F7}` → `{fld}`）"


def qdq_note(where: str) -> str:
    """HSI QDQ 精度偏低的成因：重建稳定 + 不经 TensorRT 的 ORT 直接运行同样偏低 → 量化误差本身。"""
    rb = [r["cos_sim_min"] for r in qc["trt_rebuilds"]]
    ort = [v["cos_sim_min"] for v in qc["ort_direct"].values()]
    n = len(rb)
    rec(f"{where} / rebuild n", str(n), FQ, "trt_rebuilds", n, derived="len of list")
    rec(f"{where} / rebuild min", f"{min(rb):.4f}", FQ, "trt_rebuilds[*].cos_sim_min", min(rb), derived="min of list")
    rec(f"{where} / rebuild max", f"{max(rb):.4f}", FQ, "trt_rebuilds[*].cos_sim_min", max(rb), derived="max of list")
    rec(f"{where} / ORT min", f"{min(ort):.4f}", FQ, "ort_direct.*.cos_sim_min", min(ort), derived="min over providers")
    rec(f"{where} / ORT max", f"{max(ort):.4f}", FQ, "ort_direct.*.cos_sim_min", max(ort), derived="max over providers")
    return (f"。该路径构建稳定、但精度本身偏低：同一 QDQ 图经 TensorRT 独立重建 {n} 次，cos_sim_min 为 "
            f"{min(rb):.4f}–{max(rb):.4f}；不经 TensorRT、直接用 ONNX Runtime 运行该 QDQ 模型，cos_sim_min 为 "
            f"{min(ort):.4f}–{max(ort):.4f}，同样偏低（`{FQ}` → `trt_rebuilds[*].cos_sim_min`、`ort_direct.*.cos_sim_min`）。"
            "因此这是 QDQ 量化误差本身较大，不是技术要点第 3 条中 PC 路径那种重建后输出数值异常的问题")


def build_table() -> str:
    notes: dict[str, str] = {}

    def cell(text: str, fid: str, note: str) -> str:
        notes[fid] = note
        return f"{text}[^{fid}]"

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
        "表中 `[batch=1]` 表示列表里 `batch` 字段等于 1 的元素；每个单元格的来源见文末脚注。",
        "",
        "| Model | Backend | Precision | Accuracy (cos_sim_min / same-modal NN Top-1 agreement) | Latency (batch=1, p50, ms) | Speedup vs PyTorch FP32 |",
        "|---|---|---|---|---|---|",
    ]
    for model, rows in ROWS.items():
        name = model.upper()
        for backend, precision, bkey, mode in rows:
            rid = f"{model}-{bkey}"
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
                    note = (f"`{F2}` → `results.{model}.{mode}.cos_sim_min`, "
                            f"`results.{model}.{mode}.top1_agreement`")
                    if mode == "fp16":
                        note += fp16_stability_note(model, where)
                    if model == "hsi" and mode == "int8_qdq":
                        note += qdq_note(where)
                    acc = cell(f"{cos:.6f} / {top1 * 100:.1f}%", f"{rid}-acc", note)
            # 延迟 / 加速比
            p50 = bench_p50(model, bkey)
            base = bench_p50(model, "pytorch_fp32_gpu")
            if p50 is None:
                lat = spd = NOT_MEASURED
            else:
                rec(f"{where} / p50", f"{p50:.3f}", F3, f"results.{model}.{bkey}[batch=1].p50_ms", p50)
                lat = cell(f"{p50:.3f}", f"{rid}-lat", f"`{F3}` → `results.{model}.{bkey}[batch=1].p50_ms`")
                if base is None:
                    spd = NOT_MEASURED
                else:
                    sp = base / p50
                    rec(f"{where} / speedup", f"{sp:.1f}x", F3,
                        f"results.{model}.pytorch_fp32_gpu[batch=1].p50_ms / results.{model}.{bkey}[batch=1].p50_ms",
                        sp, derived="pytorch p50 / row p50")
                    spd = cell(f"{sp:.1f}x", f"{rid}-spd",
                               f"`{F3}` → `results.{model}.pytorch_fp32_gpu[batch=1].p50_ms` ÷ "
                               f"`results.{model}.{bkey}[batch=1].p50_ms`（由这两个字段相除得出）")
            lines.append(f"| {name} | {backend} | {precision} | {acc} | {lat} | {spd} |")

    table_note = "PC 的 INT8 QDQ 路径已放弃，说明见「技术要点」第 3 条。"
    if any(NOT_MEASURED in line for line in lines):
        table_note = (f"“{NOT_MEASURED}”表示 `stage2_trt_accuracy.json` / "
                      f"`stage3_benchmark.json` 中没有对应数值。{table_note}")
    lines += ["", table_note, ""]
    lines += [f"[^{fid}]: {text}" for fid, text in notes.items()]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    fig_e2e()
    fig_cpu()
    (ASSETS / "results_table.md").write_text(build_table(), encoding="utf-8")

    print("| Where | Shown | File | Field | Raw value | Derived |")
    print("|---|---|---|---|---|---|")
    for where, shown, file, field, value, derived in PROV:
        print(f"| {where} | {shown} | {file} | {field} | {value!r} | {derived} |")


if __name__ == "__main__":
    main()
