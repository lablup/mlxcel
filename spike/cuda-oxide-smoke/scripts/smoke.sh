#!/usr/bin/env bash
# Standalone GPU smoke for the cuda-oxide package (issue #2831).
#
# Every GB10 GPU allocation goes through gpu-lock. A missing or busy lock is an
# error, never a reason to run unguarded. Compile-only results do not satisfy
# this gate; the harness must execute every case on the device. The captured
# output is then checked against the manifest's launch constraints.
#
# Usage: scripts/smoke.sh [sm_121]
set -euo pipefail

spike_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$spike_dir"
# shellcheck source=scripts/env.sh
. "$spike_dir/scripts/env.sh"

target_arch=${1:-sm_121}
if ! command -v gpu-lock >/dev/null 2>&1; then
    echo "cuda-oxide-smoke: gpu-lock not found; refusing to run an unguarded GPU test" >&2
    exit 2
fi

log=$(mktemp -t cuda-oxide-smoke-log.XXXXXX)
trap 'trash-put "$log" 2>/dev/null || true' EXIT

gpu-lock run --tag issue-2831 -- cargo oxide run --arch "$target_arch" | tee "$log"

python3 scripts/oxide_artifact.py check-launch \
    --manifest artifacts/vecadd.manifest.json \
    --log "$log"
