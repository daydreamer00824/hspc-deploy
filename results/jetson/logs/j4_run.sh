cd ~/hspc_jetson; rm -f scratch/J4_FINISHED
/usr/bin/python3 scripts/e2e_stage_b_infer.py trt_fp16 > results/logs/j4_e2e_stage_b.log 2>&1; echo "rc=$?" >> results/logs/j4_e2e_stage_b.log
HSPC_SCRATCH=$HOME/hspc_jetson/scratch/j4 /usr/bin/python3 scripts/stage7_promotion_checks.py > results/logs/j4_promotion_checks.log 2>&1; echo "rc=$?" >> results/logs/j4_promotion_checks.log
echo done > scratch/J4_FINISHED
