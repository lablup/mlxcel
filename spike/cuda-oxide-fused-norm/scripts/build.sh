#!/usr/bin/env bash
# Reproducible build for the cuda-oxide fused-norm package (issue #2832).
#
# Compiles the five kernels with the pinned cuda-oxide revision, exports the
# PTX payload out of the `.oxart` container, assembles one cubin with the CUDA
# 13.0 toolkit, writes one manifest per entry under artifacts/, and validates
# every manifest against the emitted PTX and cubin.
#
# Usage: scripts/build.sh [sm_121]
set -euo pipefail

pkg_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$pkg_dir"
# shellcheck source=scripts/env.sh
. "$pkg_dir/scripts/env.sh"

target_arch=${1:-sm_121}
case "$target_arch" in
    sm_*) ;;
    *)
        echo "cuda-oxide-fused-norm: target must use the sm_ prefix, got '$target_arch'" >&2
        exit 2
        ;;
esac

mkdir -p build artifacts

# The harness must stay MLX-free. Capture the tree first: `grep -q` on a pipe
# can exit early and make the producer fail with SIGPIPE under pipefail.
dependency_tree=$(cargo tree --locked --prefix none)
if grep -qE '^mlxcel' <<<"$dependency_tree"; then
    echo "cuda-oxide-fused-norm: the package resolves an mlxcel crate; this package must stay standalone" >&2
    exit 1
fi

# `cargo oxide build` writes <package>.ptx next to the package.
cargo oxide build --arch "$target_arch"

python3 scripts/oxide_artifact.py export \
    --binary target/release/cuda-oxide-fused-norm \
    --ptx cuda_oxide_fused_norm.ptx \
    --schema-dir abi \
    --out-dir artifacts \
    --build-dir build \
    --target-arch "$target_arch"

python3 scripts/oxide_artifact.py validate \
    --manifest-dir artifacts \
    --schema-dir abi
