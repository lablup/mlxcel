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
# eager-oracle wrapper. Comments do not match this literal call pattern.
direct_shapeless="$(
    rg -n 'mlx::core::compile\([^\n]*(true|shapeless=\*/true)' "$bridge"
)"
direct_count="$(printf '%s\n' "$direct_shapeless" | sed '/^$/d' | wc -l | tr -d ' ')"
if [[ $direct_count -ne 1 ]] \
    || [[ $direct_shapeless != *'mlx::core::compile(eager_fn, /*shapeless=*/true)'* ]]; then
    printf 'unexpected direct shapeless compile site(s):\n' >&2
    printf '%s\n' "$direct_shapeless" >&2
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
