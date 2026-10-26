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

# Test the Windows CUDA runtime and include directory choice (#2640) on the
# host, without CUDA: the pure functions of the CUDA overlay's runtime_dir.h.
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
overlay="$repo_root/src/lib/mlxcelverse/overlays/cuda"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/cuda-runtime-dir.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"$overlay" \
  "$repo_root/scripts/mlxcelverse/cuda_runtime_dir_test.cpp" \
  -o "$test_dir/cuda-runtime-dir-test"
"$test_dir/cuda-runtime-dir-test"
printf '[cuda-runtime-dir] runtime directory order, override errors, marker, preload check and include order passed (host only; no CUDA)\n'
