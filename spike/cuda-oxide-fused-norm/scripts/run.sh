#!/usr/bin/env bash
# GPU execution of the case matrix under gpu-lock (issue #2832).
#
# Runs every case on device 0, checks each element against the host model,
# writes fixtures and device outputs under run/fixtures for the MLX oracle,
# then checks every launch's block and grid against the manifests.
#
# Usage: scripts/run.sh [sm_121]
set -euo pipefail

pkg_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$pkg_dir"
# shellcheck source=scripts/env.sh
. "$pkg_dir/scripts/env.sh"

target_arch=${1:-sm_121}
if ! command -v gpu-lock >/dev/null 2>&1; then
    echo "cuda-oxide-fused-norm: gpu-lock not found; refusing to run an unguarded GPU test" >&2
    exit 2
fi

mkdir -p run
log=run/run-$target_arch.log
rm -rf run/fixtures
harness=target/release/cuda-oxide-fused-norm
[ -x "$harness" ] || {
    echo "cuda-oxide-fused-norm: $harness not built; run scripts/build.sh" >&2
    exit 2
}
# The binary scripts/build.sh produced, so the launches are the artifact the
# manifests describe rather than a rebuild.
gpu-lock run --tag issue-2832 -- "$harness" run --out run/fixtures | tee "$log"

python3 scripts/oxide_artifact.py check-launch \
    --manifest-dir artifacts \
    --log "$log"
