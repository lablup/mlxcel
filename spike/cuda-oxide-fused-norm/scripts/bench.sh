#!/usr/bin/env bash
# Standalone cold/warm benchmark under gpu-lock (issue #2832).
#
# Correctness is required for closure, speed is not; this records the kernel's
# own cost with synchronized event boundaries so #2834 can compare the same
# artifact standalone and through MLX. Results: results/bench-<arch>.json.
#
# Usage: scripts/bench.sh [sm_121] [iters]
set -euo pipefail

pkg_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$pkg_dir"
# shellcheck source=scripts/env.sh
. "$pkg_dir/scripts/env.sh"

target_arch=${1:-sm_121}
iters=${2:-200}
if ! command -v gpu-lock >/dev/null 2>&1; then
    echo "cuda-oxide-fused-norm: gpu-lock not found; refusing to run an unguarded GPU benchmark" >&2
    exit 2
fi
mkdir -p results run
harness=target/release/cuda-oxide-fused-norm
[ -x "$harness" ] || {
    echo "cuda-oxide-fused-norm: $harness not built; run scripts/build.sh" >&2
    exit 2
}
gpu-lock run --tag issue-2832-bench -- "$harness" bench \
    --out "results/bench-$target_arch.json" --iters "$iters" | tee "run/bench-$target_arch.log"
