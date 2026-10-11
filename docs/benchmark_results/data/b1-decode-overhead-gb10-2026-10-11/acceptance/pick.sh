#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-base-2227
NEW_ROOT=/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2227
CORE_BUILD=mlxcel-core-4c8821fbbcc3d8f4

args=()
which=new
while (($#)); do
  if [[ $1 == --which ]]; then
    (($# >= 2)) || { echo 'pick.sh: --which requires base or new' >&2; exit 2; }
    which=$2
    shift 2
    continue
  fi
  args+=("$1")
  shift
done

case $which in
  base) root=$BASE_ROOT ;;
  new) root=$NEW_ROOT ;;
  *) echo "pick.sh: unsupported --which value: $which" >&2; exit 2 ;;
esac

include="$root/target/release/build/$CORE_BUILD/out/build/include"
bin="$root/target/release/mlxcel-bench-engine"
test -x "$bin"
test -d "$include/cccl"
export MLXCEL_CCCL_DIR="$include/cccl"
export MLXCEL_CUTLASS_DIR="$include"
exec "$bin" "${args[@]}"
