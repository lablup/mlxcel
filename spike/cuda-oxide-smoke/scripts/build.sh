#!/usr/bin/env bash
# Reproducible build for the cuda-oxide smoke package (issue #2831).
#
# Compiles the kernel with the pinned cuda-oxide revision, exports the PTX
# payload out of the `.oxart` container, assembles a cubin with the CUDA 13.0
# toolkit, writes artifacts/vecadd.manifest.json, and validates that manifest
# against the emitted PTX and cubin.
#
# Usage: scripts/build.sh [sm_121]
set -euo pipefail

spike_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$spike_dir"
# shellcheck source=scripts/env.sh
. "$spike_dir/scripts/env.sh"

target_arch=${1:-sm_121}
case "$target_arch" in
    sm_*) ;;
    *)
        echo "cuda-oxide-smoke: target must use the sm_ prefix, got '$target_arch'" >&2
        exit 2
        ;;
esac

mkdir -p build artifacts

# The harness must stay MLX-free. This catches a dependency added by mistake
# before the device artifact is produced. Capture the tree first: `grep -q` on
# a pipe can exit early and make the producer fail with SIGPIPE under pipefail.
dependency_tree=$(cargo tree --locked --prefix none)
if grep -qE '^mlxcel v' <<<"$dependency_tree"; then
    echo "cuda-oxide-smoke: the package resolves the mlxcel crate; this spike must stay standalone" >&2
    exit 1
fi

# `cargo oxide build` writes <package>.ptx next to the package.
cargo oxide build --arch "$target_arch"

python3 scripts/oxide_artifact.py export \
    --binary target/release/cuda-oxide-smoke \
    --ptx cuda_oxide_smoke.ptx \
    --schema abi/vecadd.schema.json \
    --out-dir artifacts \
    --build-dir build \
    --target-arch "$target_arch"

python3 scripts/oxide_artifact.py validate \
    --manifest artifacts/vecadd.manifest.json \
    --schema abi/vecadd.schema.json
