#!/usr/bin/env bash
# Format and lint gate for the cuda-oxide package (issue #2831).
#
# This runs with the spike's own 1.99.1 toolchain. RUSTC_BOOTSTRAP=1 is needed
# because the pinned upstream cuda-macros expands unstable attributes on a
# stable toolchain; the upstream workspace sets the same variable for its own
# gates.
#
# Usage: scripts/check-lint.sh
set -euo pipefail

spike_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$spike_dir"
# shellcheck source=scripts/env.sh
. "$spike_dir/scripts/env.sh"

cargo fmt --check
RUSTC_BOOTSTRAP=1 cargo clippy --release --locked -- -D warnings
python3 scripts/check-guards.py

echo "PASS lint fmt=clean clippy=-D warnings guards=clean"
