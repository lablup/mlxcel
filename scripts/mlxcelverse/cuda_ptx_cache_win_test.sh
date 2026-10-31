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

# Test the CUDA PTX cache's Windows trust check (#2797) on a Windows host, without
# CUDA or a GPU, with real ACLs set by icacls. Needs a C++ compiler that targets
# MSVC on PATH (CXX, default clang++) and the Visual Studio environment. Run by the
# overlay-host-tests-windows job in .github/workflows/ci.yml.
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
overlay="$repo_root/src/lib/mlxcelverse/overlays/cuda"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/cuda-ptx-cache.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT

"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror \
  -I"$overlay" \
  "$repo_root/scripts/mlxcelverse/cuda_ptx_cache_win_test.cpp" \
  -o "$test_dir/cuda-ptx-cache-test.exe"
mkdir "$test_dir/base"
"$test_dir/cuda-ptx-cache-test.exe" "$test_dir/base"
printf '[cuda-ptx-cache] Windows ownership and ACL rule, walk, device directory and entry checks passed (host only; no CUDA)\n'
