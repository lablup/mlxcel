#!/usr/bin/env bash
# Check that an unsupported target is rejected with an actionable error
# (issue #2831 acceptance criterion). Compile-only: no GPU is touched.
#
# `compute_121` is the libNVVM spelling, not a device target. The device
# codegen must reject it before emitting anything and say what it accepts.
# The value goes through CUDA_OXIDE_DEVICE_ARCH because `--arch` selects the
# cargo-oxide run/materialization path, while the backend validates the
# device-arch hint itself.
set -euo pipefail

spike_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$spike_dir"
# shellcheck source=scripts/env.sh
. "$spike_dir/scripts/env.sh"

log=$(mktemp -t cuda-oxide-unsupported-target.XXXXXX)
trap 'trash-put "$log" 2>/dev/null || true' EXIT

set +e
CUDA_OXIDE_DEVICE_ARCH=compute_121 cargo oxide build >"$log" 2>&1
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "FAIL unsupported target compute_121 was accepted" >&2
    tail -40 "$log" >&2
    exit 1
fi
if ! grep -q 'invalid CUDA_OXIDE_DEVICE_ARCH `compute_121`' "$log"; then
    echo "FAIL unsupported target failed without naming the rejected value" >&2
    tail -40 "$log" >&2
    exit 1
fi
if ! grep -q 'expected sm_<capability>' "$log"; then
    echo "FAIL unsupported target failed without an actionable message" >&2
    tail -40 "$log" >&2
    exit 1
fi

echo "PASS unsupported-target exit=$status message='invalid CUDA_OXIDE_DEVICE_ARCH \`compute_121\`: expected sm_<capability> with an optional \`a\` suffix'"
