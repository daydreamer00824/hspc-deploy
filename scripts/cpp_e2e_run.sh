#!/bin/bash
# 板上运行：C++ 部署程序 K0～K5 与 Python 最终配置 P（deploy_jetson.py 默认 = E）的完整端到端对比。
# 4 景 × 正反两轮，每个配置各自独立进程：正轮 P→K0→…→K5，反轮 K5→…→K0→P。输出到 results/cpp_e2e/，同时记录 tegrastats。
D=${D:-results/cpp_e2e}
source "$(dirname "$(readlink -f "$0")")/cpp_bench_common.sh"
start_tegrastats 1000 "$D/tegrastats.log"
for sc in ${SCENES:-24data/10.6/1 24data/10.6/10 25data/8.4/1 25data/10.22/1}; do
  tag=${sc//\//_}
  for rnd in fwd rev; do
    if [ $rnd = fwd ]; then ORDER="P K0 K1 K2 K3 K4 K5"; else ORDER="K5 K4 K3 K2 K1 K0 P"; fi
    for c in $ORDER; do
      run_cfg $c $sc $D/${c}_${tag}_${rnd}.json
    done
  done
done
finish
