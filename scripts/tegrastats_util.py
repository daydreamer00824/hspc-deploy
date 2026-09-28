"""tegrastats 日志解析：cpp_power_summarize.py、cpp_latency_summarize.py、jetson_scene_opt.py 的
energy-report 共用，避免各自维护一份正则。"""
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Jetson 板的 UTC 偏移（板上 `date +%z` 确认为 +0800）。tegrastats 打印的是板上本地时间，没有时区信息；
# 汇总脚本所在主机与板子时区不同时，需要按这个偏移换算成 epoch，才能与其他地方记录的 UTC epoch
# （cold_wall.txt、energy_phases.txt 里的 time.time()/date +%s.%N）对齐窗口。
DEFAULT_BOARD_UTC_OFFSET = "+0800"


def parse_tegrastats(path: Path, utc_offset: str | None = None):
    """逐行解析成 {t, ram_mb, power_mw, tj, gr3d_pct} 的列表；缺失的字段记为 None。

    utc_offset：板子的 UTC 偏移（如 "+0800"）。传入时按这个时区把 tegrastats 的本地时间转成 epoch；
    不传则按运行本脚本这台机器的本地时区解析，仅在主机与板子时区相同时才对。
    """
    tzinfo = None
    if utc_offset is not None:
        sign = 1 if utc_offset[0] == "+" else -1
        hh, mm = int(utc_offset[1:3]), int(utc_offset[3:5])
        tzinfo = timezone(sign * timedelta(hours=hh, minutes=mm))
    rows = []
    for line in path.read_text().splitlines():
        m = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) ", line)
        ram = re.search(r"RAM (\d+)/(\d+)MB", line)
        pw = re.search(r"VDD_IN (\d+)mW", line)
        tj = re.search(r"tj@([\d.]+)C", line)
        gr3d = re.search(r"GR3D_FREQ (\d+)%", line)
        if m and ram:
            dt = datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S")
            if tzinfo is not None:
                dt = dt.replace(tzinfo=tzinfo)
            rows.append({"t": dt.timestamp(), "ram_mb": int(ram.group(1)),
                         "power_mw": int(pw.group(1)) if pw else None,
                         "tj": float(tj.group(1)) if tj else None,
                         "gr3d_pct": int(gr3d.group(1)) if gr3d else None})
    return rows


def tj_max(path: Path, utc_offset: str | None = None):
    """整段日志的最高温度；没有 tj 字段时返回 None 并警告，而不是让 max() 报错。"""
    vals = [r["tj"] for r in parse_tegrastats(path, utc_offset) if r["tj"] is not None]
    if not vals:
        print(f"warning: no tj samples in {path}", file=sys.stderr)
        return None
    return max(vals)


def board_utc_offset(d: Path) -> str:
    """读 d/board_utc_offset.txt（cpp_bench_common.sh 每轮运行写入）；不存在时退回
    DEFAULT_BOARD_UTC_OFFSET 并提示，而不是静默假设主机时区。"""
    f = d / "board_utc_offset.txt"
    if f.exists():
        return f.read_text().strip()
    print(f"warning: {f} not found, assuming board UTC offset {DEFAULT_BOARD_UTC_OFFSET}", file=sys.stderr)
    return DEFAULT_BOARD_UTC_OFFSET


def mean_or_die(vals, what):
    """np.mean([]) 只产出 NaN 和一条警告；这里改成直接退出并说明原因（多半是 tegrastats 没采到这段
    窗口，或时区偏移设置有误导致窗口对不上）。"""
    import numpy as np
    if not vals:
        sys.exit(f"no tegrastats samples for {what} — 检查 tegrastats 是否在这段时间内成功采样，"
                  f"或 board_utc_offset 是否设置正确")
    return float(np.mean(vals))
