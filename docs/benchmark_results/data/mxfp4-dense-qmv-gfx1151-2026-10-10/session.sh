#!/usr/bin/env bash
# Driver for mxfp4-dense-qmv-gfx1151-2026-10-10 (issue #2246, profile first).
#
# One guarded profiled decode of gpt-oss-20b-MXFP4-Q4 (the only mxfp4
# group-size-32 checkpoint on the host), at the scripts/rocm_decode_profile.sh
# default pp512/tg128 greedy shape, on gfx1151. The *_kernel_stats.csv is
# rocprofv3's whole-process per-kernel summary and *_decode_kernels.csv is the
# decode-window per-kernel table cut out by scripts/rocm_decode_profile.py;
# the full kernel trace is not committed. See README.md for the finding.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

cargo build --release --features rocm --bin mlxcel-bench-decode

scripts/rocm_decode_profile.sh --out "$PWD/benchmarks/rocm_profiles/mxfp4-dense-qmv" \
  models/mlx/gpt-oss-20b-MXFP4-Q4
