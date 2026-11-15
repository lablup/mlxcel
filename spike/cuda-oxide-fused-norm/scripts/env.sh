#!/usr/bin/env bash
# Toolchain discovery for the cuda-oxide fused-norm package (issue #2832).
#
# Source this file. The prerequisites are exactly the #2831 smoke package's
# (CUDA 13.0, LLVM 21 with NVPTX, Rust 1.99.1 with rust-src/rustc-dev,
# cargo-oxide from the pinned revision), so this delegates to its env.sh
# rather than restating the checks; see spike/cuda-oxide-smoke/README.md.

if [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
    echo "cuda-oxide-fused-norm: source this file instead of executing it: . scripts/env.sh" >&2
    exit 2
fi

_fused_norm_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_smoke_env="$_fused_norm_dir/../cuda-oxide-smoke/scripts/env.sh"
if [ ! -f "$_smoke_env" ]; then
    echo "cuda-oxide-fused-norm: sibling package env.sh not found at $_smoke_env" >&2
    unset _fused_norm_dir _smoke_env
    return 1
fi
# shellcheck source=../../cuda-oxide-smoke/scripts/env.sh
. "$_smoke_env" || return 1
export CUDA_OXIDE_SMOKE_SCRIPTS="$_fused_norm_dir/../cuda-oxide-smoke/scripts"
unset _fused_norm_dir _smoke_env
