#!/usr/bin/env bash
# Clean-build check: ordinary mlxcel builds need no cuda-oxide compiler
# (issue #2831 acceptance criterion).
#
# Proves three things without LLVM 21 or any cuda-oxide environment variable:
#   1. the cuda-oxide smoke package is not a member of the mlxcel workspace;
#   2. the mlxcel lockfile resolves no cuda-oxide crate;
#   3. a real application crate builds with the application toolchain (1.97.1).
set -euo pipefail

spike_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(cd "$spike_dir/../.." && pwd)
cd "$repo_root"

# rustup shims live in ~/.cargo/bin. Leave out any LLVM/cuda-oxide additions so
# this really is the ordinary application environment.
clean_path="$HOME/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

echo "==> workspace membership"
members=$(
    env -u CUDA_OXIDE_LLC -u LLVM_CONFIG_PATH -u LIBCLANG_PATH -u LLVM21_PREFIX \
        PATH="$clean_path" \
        cargo metadata --no-deps --format-version 1 --manifest-path "$repo_root/Cargo.toml" |
        python3 -c 'import json,sys; print("\n".join(sorted(p["name"] for p in json.load(sys.stdin)["packages"])))'
)
if grep -qx 'cuda-oxide-smoke' <<<"$members"; then
    echo "FAIL cuda-oxide-smoke is a member of the mlxcel workspace" >&2
    exit 1
fi
echo "workspace members: $(tr '\n' ' ' <<<"$members")"

echo "==> lockfile"
if grep -qE 'name = "(cuda-oxide|cuda-device|cuda-host|cuda-core)"' Cargo.lock; then
    echo "FAIL the mlxcel lockfile resolves a cuda-oxide crate" >&2
    exit 1
fi
echo "no cuda-oxide crate in Cargo.lock"

echo "==> application dependency graph"
if env -u CUDA_OXIDE_LLC -u LLVM_CONFIG_PATH -u LIBCLANG_PATH -u LLVM21_PREFIX \
    PATH="$clean_path" \
    cargo tree --locked -p mlxcel --prefix none |
    grep -qE '^(cuda-oxide|cuda-device|cuda-host|cuda-core) v'; then
    echo "FAIL the mlxcel dependency graph resolves a cuda-oxide crate" >&2
    exit 1
fi
echo "mlxcel dependency graph has no cuda-oxide crate"

echo "==> application build"
# mlxcel-mlx-pin is a real workspace member with no MLX C++ dependency, so it
# exercises the application toolchain and lockfile without an hour of C++.
env -u CUDA_OXIDE_LLC -u LLVM_CONFIG_PATH -u LIBCLANG_PATH -u LLVM21_PREFIX \
    PATH="$clean_path" \
    cargo check --locked -p mlxcel-mlx-pin
rustc_version=$(
    env -u CUDA_OXIDE_LLC -u LLVM_CONFIG_PATH -u LIBCLANG_PATH -u LLVM21_PREFIX \
        PATH="$clean_path" rustc --version
)
echo "PASS clean-build toolchain='$rustc_version' oxide_compiler=absent"
