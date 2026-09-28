#!/bin/bash
# 板上运行：进程级冷启动 + 整板内存 + 能效，Python（P = deploy_jetson.py 默认配置 E）与 C++（K4）交替对比。
#   cold：P/K4 各 5 次交替，测进程从启动到首个整景结果的墙钟、进程内引擎加载/首景耗时、进程 RSS 峰值；tegrastats(200ms) 记录整板 RAM。
#   energy：空闲 15s → 各配置连续跑 30s（ABBA 顺序：P,K4,K4,P，中间各插一段空闲）；VDD_IN 与每景耗时事后对齐（scripts/cpp_power_summarize.py）。
D=${D:-results/cpp_power}
source "$(dirname "$(readlink -f "$0")")/cpp_bench_common.sh"
SC=24data/10.6/1
# 每次重跑都从干净状态开始：这些文件要么按行追加、要么后面按时间戳/顺序跟其他文件配对，
# 留着上一次的内容会让汇总脚本把旧时间戳和这一次的 JSON 配在一起。归档而不是删除，旧的一轮完整保留。
archive_previous "$D/cold_wall.txt" "$D/cold_idle_start.txt" "$D/energy_phases.txt" "$D/tegrastats_cold.log" "$D/tegrastats_energy.log"

# ---- cold：5 轮交替
start_tegrastats 200 "$D/tegrastats_cold.log"
sleep 5; python3 -c "import time; print(time.time())" > $D/cold_idle_start.txt; sleep 5
for i in 1 2 3 4 5; do
  for c in P K4; do
    t0=$(date +%s.%N); run_cfg $c $SC $D/cold_${c}_$i.json --cold-only; t1=$(date +%s.%N)
    echo "$c $i $t0 $t1" >> $D/cold_wall.txt; sleep 3
  done
done
stop_tegrastats

# ---- energy：ABBA
start_tegrastats 1000 "$D/tegrastats_energy.log"
: > $D/energy_phases.txt
idle() { s=$(python3 -c "import time; print(time.time())"); sleep 15; e=$(python3 -c "import time; print(time.time())"); echo "idle $s $e" >> $D/energy_phases.txt; }
idle
for c in P K4 K4 P; do
  n=$(wc -l < $D/energy_phases.txt)
  run_cfg $c $SC $D/energy_${c}_$n.json --warmup 2 --seconds 30
  echo "$c $D/energy_${c}_$n.json" >> $D/energy_phases.txt
  idle
done
finish
