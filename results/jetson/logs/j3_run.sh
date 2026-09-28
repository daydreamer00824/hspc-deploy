cd ~/hspc_jetson
rm -f scratch/J3_FINISHED
tegrastats --interval 500 --logfile results/logs/j3_tegrastats.log &
TS=$!
/usr/bin/python3 scripts/benchmark.py > results/logs/j3_benchmark.log 2>&1
echo "benchmark rc=$?" >> results/logs/j3_benchmark.log
T=/usr/src/tensorrt/bin/trtexec
for spec in "pc_fp16:15x3" "hsi_fp16:342x3x3" "hsi_int8_qdq:342x3x3" "pc_fp32_notf32:15x3" "hsi_fp32_notf32:342x3x3"; do
  n=${spec%%:*}; s=${spec##*:}
  for b in 1 64; do
    $T --loadEngine=engines/$n.plan --shapes=input:${b}x$s --noDataTransfers --warmUp=200 --iterations=300 --avgRuns=300 > results/logs/j3_trtexec_${n}_b${b}.log 2>&1
  done
done
kill $TS
echo done > scratch/J3_FINISHED
