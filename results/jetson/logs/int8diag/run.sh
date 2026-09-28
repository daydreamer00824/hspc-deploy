#!/usr/bin/env bash
set -euo pipefail

trtexec=/usr/src/tensorrt/bin/trtexec
output_dir=scratch/int8diag
mkdir -p "$output_dir"

# 重跑前归档已有的成功或失败产物，避免覆盖证据。
shopt -s nullglob
previous=()
for path in "$output_dir"/*; do
  [ "$path" = "$output_dir/archive" ] || previous+=("$path")
done
if [ "${#previous[@]}" -gt 0 ]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  archive_dir="$output_dir/archive/$stamp"
  suffix=0
  while [ -e "$archive_dir" ]; do
    suffix=$((suffix + 1))
    archive_dir="$output_dir/archive/${stamp}_$suffix"
  done
  mkdir -p "$archive_dir"
  mv -- "${previous[@]}" "$archive_dir/"
fi

trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "exit_code=%s\n" "$rc" > "$output_dir/FAILED"; fi' EXIT

for w in pc hsi; do
  if [ "$w" = pc ]; then shape=15x3; else shape=342x3x3; fi
  shape_args=("--minShapes=input:1x$shape" "--optShapes=input:8x$shape" "--maxShapes=input:64x$shape")
  "$trtexec" --onnx="onnx/${w}_encoder.onnx" "${shape_args[@]}" --noTF32 --fp16 --int8 \
    --calib="engines/${w}_int8_implicit.calib" --profilingVerbosity=detailed \
    --exportLayerInfo="$output_dir/${w}_int8_layers.json" --saveEngine="$output_dir/${w}_int8.plan" \
    --skipInference --verbose > "$output_dir/${w}_int8_build.log" 2>&1
  "$trtexec" --onnx="onnx/${w}_encoder.onnx" "${shape_args[@]}" --noTF32 --fp16 \
    --profilingVerbosity=detailed --exportLayerInfo="$output_dir/${w}_fp16_layers.json" \
    --saveEngine="$output_dir/${w}_fp16.plan" --skipInference > "$output_dir/${w}_fp16_build.log" 2>&1
done
printf 'done\n' > "$output_dir/FINISHED"
