#!/usr/bin/env bash
# Build and run the lablup/mlxcel#2206 probes (bf16 GEMM residue on gfx11).
#
#   scripts/rocm_bf16_gemm_residue/run.sh [--all | --random]
#
# wmma_probe.hip isolates the instructions (v_wmma_f32_16x16x16_{bf16,f16},
# v_dot2_f32_bf16) against an f32 FMA baseline; gemm_repro.cpp runs the same
# configuration through hipBLASLt and rocBLAS without MLX. The argument goes to
# gemm_repro. Set ROCM_ARCH to build for a device other than gfx1151, and wrap
# the call in scripts/rocm_gpu_guard.sh on a shared host.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rocm="${ROCM_PATH:-/opt/rocm}"
arch="${ROCM_ARCH:-gfx1151}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

hipcc --offload-arch="$arch" -O2 "$here/wmma_probe.hip" -o "$out/wmma_probe"
hipcc --offload-arch="$arch" -O2 "$here/gemm_repro.cpp" -o "$out/gemm_repro" \
  -L"$rocm/lib" -lhipblaslt -lrocblas

export LD_LIBRARY_PATH="$rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "== instruction probe =="
"$out/wmma_probe"
echo "== library reproduction =="
"$out/gemm_repro" "$@"
