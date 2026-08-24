#!/usr/bin/env bash
# Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
bridge="$repo_root/src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp"

# Every production shapeless=true construction must pass through the opt-in
# eager-oracle wrapper. Inventory the compile token itself so a future direct
# call cannot evade this check merely by putting its arguments on another line.
if command -v rg >/dev/null 2>&1; then
    direct_compile="$(rg -nF 'mlx::core::compile(' "$bridge")"
else
    # The self-hosted Apple runner intentionally carries only the build tools;
    # keep the on-demand audit usable there without installing ripgrep.
    direct_compile="$(grep -nF 'mlx::core::compile(' "$bridge")"
fi

wrapper_count=0
unexpected_compile=''
while IFS= read -r line; do
    case "$line" in
        *'mlx::core::compile(eager_fn, /*shapeless=*/true)'*)
            ((wrapper_count += 1))
            ;;
        *'mlx::core::compile('*false*)
            [[ $line != *true* ]] || unexpected_compile+="$line"$'\n'
            ;;
        *)
            unexpected_compile+="$line"$'\n'
            ;;
    esac
done <<< "$direct_compile"
if [[ $wrapper_count -ne 1 || -n $unexpected_compile ]]; then
    printf 'unexpected direct compile site(s):\n' >&2
    printf '%s' "$unexpected_compile" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin)
        features="metal,accelerate"
        ;;
    Linux)
        command -v nvidia-smi >/dev/null || {
            printf 'nvidia-smi is required for the CUDA audit\n' >&2
            exit 1
        }
        compute_capability="$(
            nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
                | sed -n '1{s/\.//g;p;}'
        )"
        [[ -n $compute_capability ]] || {
            printf 'could not determine the NVIDIA compute capability\n' >&2
            exit 1
        }
        export MLX_CUDA_ARCHITECTURES="${MLX_CUDA_ARCHITECTURES:-$compute_capability}"
        features="cuda"
        ;;
    *)
        printf 'unsupported audit platform: %s\n' "$(uname -s)" >&2
        exit 1
        ;;
esac

export MLXCEL_SHAPELESS_COMPILE_AUDIT=1

# Synthetic tensors only. This hardware qualification never downloads or opens
# a model checkpoint, regardless of backend.
exec cargo test \
    -p mlxcel-core \
    --profile test-fast \
    --features "$features" \
    --lib \
    ffi_tests::shapeless_compile_audit_harness \
    -- \
    --ignored \
    --exact \
    --nocapture \
    --test-threads=1
