#!/bin/bash
# Dispatch mlxcel-bench-engine to the pre-epic or the final build: first arg after flags "--which base|new".
args=(); which=new
while [ $# -gt 0 ]; do
  if [ "$1" = "--which" ]; then which=$2; shift 2; continue; fi
  args+=("$1"); shift
done
case $which in
  base) exec ~/Development/backend.ai/wt-epic-2166-base/target/release/mlxcel-bench-engine "${args[@]}";;
  new) export MLXCEL_CCCL_DIR=~/Development/backend.ai/wt-epic-2166-final/target/release/build/mlxcel-core-4c8821fbbcc3d8f4/out/build/include/cccl MLXCEL_CUTLASS_DIR=~/Development/backend.ai/wt-epic-2166-final/target/release/build/mlxcel-core-4c8821fbbcc3d8f4/out/build/include; exec ~/Development/backend.ai/wt-epic-2166-final/target/release/mlxcel-bench-engine "${args[@]}";;
esac
