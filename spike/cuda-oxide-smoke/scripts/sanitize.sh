#!/usr/bin/env bash
# Compute Sanitizer run for the cuda-oxide package (issue #2831).
#
# Mirrors scripts/smoke.sh: every GPU allocation goes through gpu-lock, a
# missing lock is an error, and the captured output is checked against the
# manifest's launch constraints so a skipped or shortened run cannot pass.
#
# Usage: scripts/sanitize.sh [sm_121]
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

log=$(mktemp -t cuda-oxide-sanitize-log.XXXXXX)
trap 'trash-put "$log" 2>/dev/null || true' EXIT

gpu-lock run --tag issue-2831 -- \
    cargo oxide sanitize --tool memcheck --arch "$target_arch" | tee "$log"

python3 scripts/oxide_artifact.py check-launch \
    --manifest artifacts/vecadd.manifest.json \
    --log "$log"

if ! grep -q 'ERROR SUMMARY: 0 errors' "$log"; then
    echo "cuda-oxide-smoke: Compute Sanitizer did not report a clean summary" >&2
    exit 1
fi
echo "PASS sanitizer memcheck=clean"
