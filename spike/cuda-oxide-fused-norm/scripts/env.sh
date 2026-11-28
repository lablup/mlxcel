#!/usr/bin/env bash
# Toolchain discovery for the cuda-oxide fused-norm package (issue #2832).
#
# Source this file. The prerequisites are exactly the #2831 smoke package's
# (CUDA 13.0, LLVM 21 with NVPTX, Rust 1.99.1 with rust-src/rustc-dev,
# cargo-oxide from the pinned revision), so this delegates to its env.sh
# rather than restating the checks; see spike/cuda-oxide-smoke/README.md.

# The path this file was sourced by, and whether it was executed instead.
# bash and zsh spell both differently, and neither spelling parses in the
# other shell (or in a POSIX sh), so each one sits inside `eval`.
if [ -n "${BASH_VERSION:-}" ]; then
    eval '_fused_norm_src=${BASH_SOURCE[0]}'
    [ "$_fused_norm_src" != "$0" ] || _fused_norm_src=
elif [ -n "${ZSH_VERSION:-}" ]; then
    eval '_fused_norm_src=${(%):-%x}'
    case "${ZSH_EVAL_CONTEXT:-}" in
        *:file*) ;;
        *) _fused_norm_src= ;;
    esac
else
    echo "cuda-oxide-fused-norm: source this file from bash or zsh" >&2
    return 1 2>/dev/null || exit 1
fi
if [ -z "$_fused_norm_src" ]; then
    echo "cuda-oxide-fused-norm: source this file instead of executing it: . scripts/env.sh" >&2
    exit 2
fi

_fused_norm_dir=$(cd "$(dirname "$_fused_norm_src")/.." && pwd)
unset _fused_norm_src
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
