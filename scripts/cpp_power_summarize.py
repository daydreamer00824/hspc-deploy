"""进程级冷启动、整板内存、能效的汇总（本机运行，读板上拉回的 results/jetson/cpp_power/）。

对应 scripts/cpp_power_run.sh：cold（P/K4 各 5 次交替）与 energy（ABBA：P,K4,K4,P，各窗口 30s，中间夹空闲 15s）。
能效口径同 jetson_scene_opt.py 的 energy-report：整板输入功率 VDD_IN（tegrastats，1 秒一个采样，窗口首尾各去掉 1 秒），
每景能耗 = 窗口平均功率 × 每景耗时；另给"扣除空闲功率"的版本。用法：cpp_power_summarize.py [输入目录] [输出文件]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tegrastats_util import board_utc_offset, mean_or_die, parse_tegrastats  # noqa: E402

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/jetson/cpp_power")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "results/jetson/cpp_power_summary.json")
UTC_OFFSET = board_utc_offset(D)


def med(v):
    return float(np.median(v))


# ---------------------------------------------------------------- 冷启动 + 内存
tg_cold = parse_tegrastats(D / "tegrastats_cold.log", UTC_OFFSET)
if not tg_cold:
    sys.exit("tegrastats_cold.log 里没有可解析的样本")
t_idle = float((D / "cold_idle_start.txt").read_text())
base = mean_or_die([r["ram_mb"] for r in tg_cold if t_idle <= r["t"] < t_idle + 5], "cold idle baseline")
cold = {}
for line in (D / "cold_wall.txt").read_text().splitlines():
    c, i, t0, t1 = line.split()
    j = json.loads((D / f"cold_{c}_{i}.json").read_text())
    win = [r["ram_mb"] for r in tg_cold if float(t0) - 1 <= r["t"] <= float(t1) + 1]
    if not win:
        sys.exit(f"no tegrastats samples for cold window {c} {i} ({t0}~{t1})")
    e = cold.setdefault(c, {"process_wall_s": [], "engine_load_s": [], "first_scene_s": [], "peak_rss_mb": [], "board_ram_peak_delta_mb": []})
    e["process_wall_s"].append(float(t1) - float(t0))
    e["engine_load_s"].append(j["engine_load_sec"])
    e["first_scene_s"].append(j["cold_start"]["total_sec"])
    e["peak_rss_mb"].append(j["peak_rss_kb"] / 1024)
    e["board_ram_peak_delta_mb"].append(max(win) - base)
cold_summary = {c: {k: {"median": med(v), "min": float(min(v)), "max": float(max(v)), "all": v} for k, v in e.items()} for c, e in cold.items()}

# ---------------------------------------------------------------- 能效
tg_energy = parse_tegrastats(D / "tegrastats_energy.log", UTC_OFFSET)
if not tg_energy:
    sys.exit("tegrastats_energy.log 里没有可解析的样本")
phases, wins = [], []
for line in (D / "energy_phases.txt").read_text().splitlines():
    a = line.split()
    if a[0] == "idle":
        phases.append((float(a[1]), float(a[2])))
    else:
        # a[1] 是板上记录的路径（run_cfg 写的是相对 D 的文件名），本地汇总只关心文件名本身，
        # 不依赖板上 D 与本地 D 同名（D 可以用环境变量覆盖）
        j = json.loads((D / Path(a[1]).name).read_text())
        wins.append((a[0], j["warm"]["t_start_epoch"], j["warm"]["t_end_epoch"], j["warm"]["n_timed"], j["warm"]["total_ms_median"]))


def mean_power(t0, t1):
    w = [r["power_mw"] for r in tg_energy if t0 + 1 <= r["t"] <= t1 - 1 and r["power_mw"] is not None]
    return mean_or_die(w, f"energy window {t0}~{t1}"), len(w)


idle_mw = mean_or_die([mean_power(a, b)[0] for a, b in phases], "idle phases")
energy = {"_idle_mean_mw": idle_mw, "windows": []}
per_cfg = {}
for c, t0, t1, n, med_ms in wins:
    p, ns = mean_power(t0, t1)
    per = (t1 - t0) / n
    w = {"config": c, "duration_s": t1 - t0, "n_scenes": n, "n_samples": ns, "scene_ms_mean": per * 1000, "scene_ms_median": med_ms,
         "vdd_in_mean_mw": p, "energy_per_scene_j_total_board": p / 1000 * per, "energy_per_scene_j_above_idle": (p - idle_mw) / 1000 * per,
         "scenes_per_s_per_w_total_board": (1 / per) / (p / 1000)}
    energy["windows"].append(w)
    per_cfg.setdefault(c, []).append(w)
energy["by_config"] = {c: {k: float(np.mean([w[k] for w in ws])) for k in ("scene_ms_mean", "vdd_in_mean_mw", "energy_per_scene_j_total_board",
                                                                               "energy_per_scene_j_above_idle", "scenes_per_s_per_w_total_board")}
                       for c, ws in per_cfg.items()}
tj_all = [r["tj"] for r in tg_energy + tg_cold if r["tj"] is not None]
tj_max_c = max(tj_all) if tj_all else None  # 没有 tj 字段时（固件/tegrastats 版本差异）不要让 max([]) 把整份汇总一起崩掉
if tj_max_c is None:
    print("warning: no tj samples in tegrastats_cold.log / tegrastats_energy.log", file=sys.stderr)
summary = {"note": "P=Python 最终配置 E（deploy_jetson.py 默认）；K4=C++ 最终配置。场景 24data/10.6/1，页缓存热；风扇为系统默认闭环（nvfancontrol），时钟锁定 MAXN_SUPER。",
           "cold_start": {"board_ram_idle_baseline_mb": base, **cold_summary}, "energy": energy, "tj_max_c": tj_max_c}
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({"cold": {c: {k: round(v["median"], 3) for k, v in e.items()} for c, e in cold_summary.items()},
                  "energy_by_config": energy["by_config"], "idle_mw": idle_mw, "tj_max": tj_max_c}, indent=1))
