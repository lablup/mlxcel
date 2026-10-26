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

# Test the Windows ROCm runtime and include directory choice (#2606) on the
# host, without HIP: the pure functions from the overlay's runtime_dir.h and
# jit_module.cpp, the latter copied out by awk as allocator_ownership_test.sh
# does. Run by `make verify-rocm-overlay` and, on every PR that can change the
# choice, by the overlay-host-tests job in .github/workflows/ci.yml (#2642).
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
overlay="$repo_root/src/lib/mlxcelverse/overlays/rocm"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/rocm-runtime-dir.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT
awk '
  /^std::string choose_rocm_include_dir\(/ { copying = 1 }
  copying { print }
  copying && /^}/ { found = 1; exit }
  END { if (!found) exit 1 }
' "$overlay/mlx/backend/rocm/jit_module.cpp" > "$test_dir/rocm_include_dir.inc"
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"$overlay" -I"$test_dir" \
  "$repo_root/scripts/mlxcelverse/rocm_runtime_dir_test.cpp" \
  -o "$test_dir/rocm-runtime-dir-test"
"$test_dir/rocm-runtime-dir-test"
printf '[rocm-runtime-dir] runtime directory order, override errors, preload check and include order passed (host only; no HIP)\n'
