#!/bin/bash
# Wrapper that engine_bench_rounds.py runs as --bin. Arms pass only flags, so
# the two A/B switches travel as flags and become environment variables here:
#   --cublas-alt  preload the cuBLASLt in $MLXCEL_AB_CUBLAS_DIR instead of the
#                 one the binary links (libcublasLt.so.13 + libcublas.so.13)
#   --no-gemv     MLXCEL_CUDA_DISABLE_GEMV=1: M=1 dense matmuls go to cuBLASLt
# The binary comes from $MLXCEL_AB_BIN_DIR (default: <repo>/target/release).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
bin_dir=${MLXCEL_AB_BIN_DIR:-$repo/target/release}
args=()
for a in "$@"; do
  case "$a" in
    --cublas-alt)
      : "${MLXCEL_AB_CUBLAS_DIR:?set MLXCEL_AB_CUBLAS_DIR to the alternate cuBLAS lib dir}"
      export LD_PRELOAD="$MLXCEL_AB_CUBLAS_DIR/libcublasLt.so.13:$MLXCEL_AB_CUBLAS_DIR/libcublas.so.13"
      ;;
    --no-gemv) export MLXCEL_CUDA_DISABLE_GEMV=1 ;;
    *) args+=("$a") ;;
  esac
done
exec "$bin_dir/mlxcel-bench-engine" "${args[@]}"
