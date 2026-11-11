#!/usr/bin/env bash
# Format and lint gate for the cuda-oxide fused-norm package (issue #2832).
#
# Runs with the package's own 1.99.1 toolchain. RUSTC_BOOTSTRAP=1 is needed
# because the pinned upstream cuda-macros expands unstable attributes on a
# stable toolchain (and this crate uses the `f16` primitive the same way).
#
# Usage: scripts/check-lint.sh
set -euo pipefail

pkg_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$pkg_dir"
# shellcheck source=scripts/env.sh
. "$pkg_dir/scripts/env.sh"

cargo fmt --check
RUSTC_BOOTSTRAP=1 cargo clippy --release --locked -- -D warnings
python3 scripts/check-guards.py

echo "PASS lint fmt=clean clippy=-D warnings guards=clean"
