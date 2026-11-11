#!/usr/bin/env bash
# Score the standalone kernel against both MLX references (issue #2832).
#
# Reads run/fixtures (standalone outputs) and run/oracle (MLX dumps), applies
# the frozen per-dtype budgets, and writes the machine-readable result with
# the artifact identity and host facts to results/compare-<arch>.json.
#
# Usage: scripts/compare.sh [sm_121]
set -euo pipefail

pkg_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$pkg_dir"
# shellcheck source=scripts/env.sh
. "$pkg_dir/scripts/env.sh"

target_arch=${1:-sm_121}
ptx=artifacts/fused_add_rms_norm.$target_arch.ptx
cubin=artifacts/fused_add_rms_norm.$target_arch.cubin
[ -f "$ptx" ] && [ -f "$cubin" ] || {
    echo "cuda-oxide-fused-norm: artifacts for $target_arch not found; run scripts/build.sh" >&2
    exit 2
}
[ -f run/oracle/meta.tsv ] || {
    echo "cuda-oxide-fused-norm: no oracle dump at run/oracle; run scripts/oracle.sh" >&2
    exit 2
}

sha() { sha256sum "$1" | cut -d' ' -f1; }
gpu=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
mkdir -p results
harness=target/release/cuda-oxide-fused-norm
[ -x "$harness" ] || {
    echo "cuda-oxide-fused-norm: $harness not built; run scripts/build.sh" >&2
    exit 2
}
# The comparison is pure host arithmetic over saved bits; no GPU is touched.
"$harness" compare \
    --fixtures run/fixtures \
    --oracle run/oracle \
    --out "results/compare-$target_arch.json" \
    --note "target_arch=$target_arch" \
    --note "ptx_sha256=$(sha "$ptx")" \
    --note "cubin_sha256=$(sha "$cubin")" \
    --note "gpu=$gpu" \
    --note "host=$(uname -srm)" \
    --note "cuda_toolkit=$(nvcc --version | tail -1 | sed 's/^ *//')" \
    --note "mlxcel_commit=$(git -C "$pkg_dir/../.." rev-parse HEAD 2>/dev/null || echo unknown)" \
    --note "run_date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --note "gpu_lock=$(command -v gpu-lock)"
