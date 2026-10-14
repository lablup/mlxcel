#!/usr/bin/env bash
# Copyright 2026 Lablup Inc.
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

# Exercise the move-only ROCm allocation boundary with a host mock, without HIP.
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
overlay="$repo_root/src/lib/mlxcelverse/overlays/rocm"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/rocm-ownership.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT
mkdir -p "$test_dir/mlx"
# api.h only supplies symbol visibility; the actual allocator header is tested.
printf '#pragma once\n#define MLX_API\n' > "$test_dir/mlx/api.h"
awk '
  /^allocator::Data malloc_async\(size_t size, CommandEncoder& encoder\) \{/ { copying = 1 }
  copying { print }
  copying && /^}/ { found = 1; exit }
  END { if (!found) exit 1 }
' "$overlay/mlx/backend/rocm/allocator.cpp" > "$test_dir/rocm_malloc_async.inc"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter \
  -I"$overlay" -I"$test_dir" \
  "$repo_root/scripts/mlxcelverse/allocator_ownership_test.cpp" \
  -o "$test_dir/allocator-ownership-test"
"$test_dir/allocator-ownership-test"
printf '[rocm-ownership] owned, borrowed, empty, move, retained-batch and failure-cleanup contracts passed (host mock; no HIP execution)\n'
