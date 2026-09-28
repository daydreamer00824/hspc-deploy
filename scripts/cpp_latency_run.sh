#!/bin/bash
# 板上运行：Python（GraphRunner）与 C++（hspc_latency）单样本延迟，同一场次 Py→C++→C++→Py 交替，各 300 次。
# 输出到 results/cpp_latency/；同时记录 tegrastats（tj 温度、频率、风扇）。
# 单步失败也继续跑完其余步骤，一次性看到全部诊断信息；tegrastats 未能正常采集同样记为失败，
# 不会用缺失/过期的温度数据写出 FINISHED。
D=${D:-results/cpp_latency}
source "$(dirname "$(readlink -f "$0")")/cpp_bench_common.sh"
# 重跑前把同名的 py_*/cpp_*.json 和日志归档，不用 rm；board_utc_offset.txt 是 source 时刚为本次
# 写入的，从通配符里排除，否则会被立即归档走，导致结果目录里反而没有这个文件。
for f in "$D"/*; do
  [ "$(basename "$f")" = "board_utc_offset.txt" ] || archive_previous "$f"
done
start_tegrastats 1000 "$D/tegrastats.log"
py() {
  PYTHONPATH=pylibs:scripts /usr/bin/python3 scripts/jetson_scene_opt.py latency --out $D/py_$1.json > $D/py_$1.log 2>&1
  check_step $? $D/py_$1.json "py $1"
}
cpp() {
  for w in pc hsi; do
    ./cpp/build/hspc_latency --which $w --engine engines/${w}_fp16.plan \
      --samples scratch/cpp_lat/${w}_samples.f32 --out $D/cpp_${1}_$w.json 2> $D/cpp_${1}_$w.log
    check_step $? $D/cpp_${1}_$w.json "cpp $1 $w"
  done
}
py fwd; cpp fwd; cpp rev; py rev
finish
