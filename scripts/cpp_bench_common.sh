# 板上 C++/Python 对比脚本共用：conda 环境变量、单次运行 + 失败记录 + 归档。被 cpp_e2e_run.sh /
# cpp_power_run.sh / cpp_latency_run.sh source（调用方先设好 D=结果目录，可用环境变量覆盖，再 source），不单独执行。
#
# 原则：重跑不删除任何东西。会被覆盖的旧文件（FINISHED/FAILED、旧 JSON/日志、旧 tegrastats 日志……）
# 一律 mv 进 "$D"_archive/replaced_at_<时间戳>_<随机后缀>/，附 sha256 清单，不用 rm。
cd ~/hspc_jetson
E=$HOME/miniforge3/envs/hspc-jetson
set -u
mkdir -p "$D"
FAIL=0
TS=""  # start_tegrastats() 之前不会被引用，这里只是让 set -u 下的误用报错清楚，而不是报"未绑定变量"
RUN_TS=$(date +%Y%m%dT%H%M%S)
ARCHIVE_DIR=""  # 第一次真正归档时才用 mktemp -d 创建：时间戳只到秒，同一秒启动两轮也不会共用目录、互相覆盖

# conda 环境变量只在真正需要的命令（run_cfg 里跑的 Python/hspc_deploy）上生效，不整段 export：
# cpp_latency_run.sh 用系统 Python 跑 Python 侧延迟测试，系统 torch 不能带 conda 的 LD_LIBRARY_PATH
# （会带入较新的 libstdc++，见 docs/jetson.md 的环境说明），必须两边环境互不污染。
HSPC_ENV=(LD_LIBRARY_PATH="$E/lib" PYTHONNOUSERSITE=1 PYTHONPATH=scripts PROJ_DATA="$E/share/proj" GDAL_DATA="$E/share/gdal" HSPC_DATA_ROOT="$HOME/hspc_jetson/data")

# archive_previous <文件...>：把存在的文件原样移进归档目录，并在 ARCHIVED.txt 里追加一行
# 原修改时间 + sha256 + 原路径 + 归档名。不存在的文件跳过；同名文件再次归档时加 .dupN 后缀，
# 不覆盖先归档的那份。mv 失败直接退出，避免没归档成功的旧记录被覆盖。
archive_previous() {
  local f
  for f in "$@"; do
    [ -e "$f" ] || continue
    if [ -z "$ARCHIVE_DIR" ]; then
      mkdir -p "${D}_archive" && ARCHIVE_DIR=$(mktemp -d "${D}_archive/replaced_at_${RUN_TS}_XXXXXX") \
        || { echo "cannot create archive dir under ${D}_archive" >&2; exit 1; }
    fi
    local name n=1
    name=$(basename "$f")
    while [ -e "$ARCHIVE_DIR/$name" ]; do name="$(basename "$f").dup$n"; n=$((n + 1)); done
    echo "$(date -r "$f" +%Y-%m-%dT%H:%M:%S) $(sha256sum "$f" | cut -d' ' -f1) $f $name" >> "$ARCHIVE_DIR/ARCHIVED.txt"
    mv -n "$f" "$ARCHIVE_DIR/$name" && [ ! -e "$f" ] \
      || { echo "failed to archive $f into $ARCHIVE_DIR, aborting before it gets overwritten" >&2; exit 1; }
  done
}

# 启动时就把旧的完成/失败标记挪进归档，避免中途异常退出时目录里留着一个误导性的旧 FINISHED。
archive_previous "$D/FINISHED" "$D/FAILED"

# 记下板子当前的 UTC 偏移，供本机汇总脚本（tegrastats_util.board_utc_offset）按板上时区解析
# tegrastats 的本地时间戳，主机与板子时区不一致时用于对齐功耗/内存的时间窗口。
archive_previous "$D/board_utc_offset.txt"
date +%z > "$D/board_utc_offset.txt"

# check_step <rc> <out文件> <名称>：返回码非 0 或输出文件为空时记入 FAILED、设 FAIL=1。
# run_cfg 和 cpp_latency_run.sh 共用，避免再写第三份判定逻辑。
check_step() {
  local rc=$1 out=$2 name=$3
  if [ "$rc" -ne 0 ] || [ ! -s "$out" ]; then
    echo "FAILED $name rc=$rc" | tee -a "$D/FAILED"
    FAIL=1
    return 1
  fi
  return 0
}

# run_cfg <cfg> <scene> <out.json> [额外参数...]：cfg=P 跑 Python 最终配置（scripts/deploy_jetson.py），
# 其余当作 hspc_deploy 的 --config。旧的同名输出先归档，失败由 check_step 判定。
run_cfg() {
  local c=$1 sc=$2 o=$3; shift 3
  archive_previous "$o" "${o%.json}.log"
  if [ "$c" = P ]; then
    env "${HSPC_ENV[@]}" "$E/bin/python" scripts/deploy_jetson.py --scene "$sc" --out "$o" "$@" > "${o%.json}.log" 2>&1
  else
    env "${HSPC_ENV[@]}" ./cpp/build/hspc_deploy --scene "$sc" --config "$c" --out "$o" "$@" > "${o%.json}.log" 2>&1
  fi
  local rc=$?
  echo "rc=$rc" >> "${o%.json}.log"
  check_step $rc "$o" "$c $o"
}

# 只有全部步骤都成功才写 FINISHED；否则非零退出，调用方脚本末尾调一次。先停 tegrastats 并检查它
# 有没有中途退出（会把 FAIL 置 1），再判定——顺序反过来会导致 FINISHED 和 FAILED 同时存在。
finish() {
  trap - EXIT
  stop_tegrastats
  if [ $FAIL -eq 0 ]; then echo FINISHED > "$D/FINISHED"
  else echo "one or more runs failed, see $D/FAILED" >&2; exit 1; fi
}

# start_tegrastats <interval_ms> <logfile>：不主动归档旧日志——cpp_e2e_run.sh 需要在同一个日志上跨次
# 追加（SCENES= 只重跑部分场景时保留其余场景的记录）；需要从空日志开始的调用方（如 cpp_power_run.sh）
# 自己先 archive_previous 该日志。启动后台进程，等最多 5 秒确认它真的在写（进程存活 + 日志变大）。
# 启动失败单独记入 FAILED，调用方仍往下走。启动后即挂上 EXIT trap，脚本中途被中断也会停掉它。
start_tegrastats() {
  local interval=$1 log=$2
  local before=0
  [ -f "$log" ] && before=$(wc -c < "$log")
  tegrastats --interval "$interval" --logfile "$log" &
  TS=$!
  trap stop_tegrastats EXIT
  local waited=0
  while [ $waited -lt 50 ]; do
    sleep 0.1; waited=$((waited + 1))
    kill -0 $TS 2>/dev/null || break
    local now=0
    [ -f "$log" ] && now=$(wc -c < "$log")
    if [ "$now" -gt "$before" ]; then return 0; fi
  done
  echo "FAILED tegrastats did not start writing $log" | tee -a "$D/FAILED"
  FAIL=1
}

# stop_tegrastats：先确认它没有中途退出（否则这段窗口的功率/温度数据不完整），再停止。没有在跑的
# （TS 为空）直接返回；停完清空 TS，避免 finish()/EXIT trap 重复调用时误判。
stop_tegrastats() {
  [ -n "$TS" ] || return 0
  if ! kill -0 $TS 2>/dev/null; then
    echo "FAILED tegrastats exited early (pid=$TS)" | tee -a "$D/FAILED"
    FAIL=1
  else
    kill $TS 2>/dev/null
  fi
  wait $TS 2>/dev/null
  TS=""
}
