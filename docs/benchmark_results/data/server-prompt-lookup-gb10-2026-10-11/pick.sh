#!/bin/bash
# Dispatch mlxcel-bench-engine to one of three builds: "--which base|prev|new".
#   base: 4c44e317 (pre-epic CxxGenerator CLI), prev: c2523e5f (engine CLI before #2229, penalty requests synchronous),
#   new: main after #2229 and #2255. base and prev carry a measurement-only --repetition-penalty patch.
args=(); which=new
while [ $# -gt 0 ]; do
  if [ "$1" = "--which" ]; then which=$2; shift 2; continue; fi
  args+=("$1"); shift
done
exec ${WT_PREFIX:-../wt-measure-}$which/target/release/mlxcel-bench-engine "${args[@]}"
