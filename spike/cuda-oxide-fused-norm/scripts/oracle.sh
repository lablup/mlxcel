#!/usr/bin/env bash
# Replay the fixtures through the MLX references (issue #2832).
#
# Builds and runs the `oxide_fused_norm_oracle` example of mlxcel-core with the
# application toolchain (1.97.1) and the CUDA feature, under gpu-lock. This is
# the only step that touches MLX; the standalone package never links it.
#
# Usage: scripts/oracle.sh [fixtures-dir] [out-dir]
set -euo pipefail

pkg_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(cd "$pkg_dir/../.." && pwd)
fixtures=${1:-$pkg_dir/run/fixtures}
out=${2:-$pkg_dir/run/oracle}

if ! command -v gpu-lock >/dev/null 2>&1; then
    echo "cuda-oxide-fused-norm: gpu-lock not found; refusing to run an unguarded GPU test" >&2
    exit 2
fi
[ -f "$fixtures/cases.tsv" ] || {
    echo "cuda-oxide-fused-norm: no fixtures at $fixtures; run scripts/run.sh first" >&2
    exit 2
}

cd "$repo_root"
# Build with the ordinary application environment (no cuda-oxide variables),
# then run the built example under the lock.
env -u CUDA_OXIDE_LLC -u LLVM_CONFIG_PATH -u LIBCLANG_PATH -u LLVM21_PREFIX \
    cargo build --profile test-fast -p mlxcel-core --features cuda --example oxide_fused_norm_oracle
target_dir=${CARGO_TARGET_DIR:-$repo_root/target}
rm -rf "$out"
MLX_ENABLE_TF32=0 gpu-lock run --tag issue-2832-oracle -- \
    "$target_dir/test-fast/examples/oxide_fused_norm_oracle" --fixtures "$fixtures" --out "$out" |
    tee "$pkg_dir/run/oracle.log"
