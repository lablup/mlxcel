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
# does. Run by `make verify-rocm-overlay` and, on every PR and push to main, by
# the overlay-host-tests job in .github/workflows/ci.yml (#2642).
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
overlay="$repo_root/src/lib/mlxcelverse/overlays/rocm"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/rocm-runtime-dir.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT
# A deleted call compiles and passes everything else, so check the source. Comment
# lines do not count (#2641).
delayload="$overlay/mlx/backend/rocm/delayload.cpp"
code_has() {
  grep -v '^[[:space:]]*//' "$1" | grep -qF "$2"
}
if ! code_has "$delayload" 'SetDllDirectoryW(L"")'; then
  printf 'delayload.cpp no longer calls SetDllDirectoryW(L""); the current directory is back in the DLL search\n' >&2
  exit 1
fi
delayload_hook=$(awk '
  /delayload_helper\(unsigned dliNotify, PDelayLoadInfo pdli\) \{/ { copying = 1 }
  copying { print }
  copying && /^}/ { exit }
' "$delayload")
if ! grep -qF 'remove_current_dir_from_dll_search();' <<<"$delayload_hook"; then
  printf 'the delay-load hook no longer calls remove_current_dir_from_dll_search()\n' >&2
  exit 1
fi
for script in build.rs src/lib/mlxcel-core/build.rs; do
  if ! code_has "$repo_root/$script" '/DEPENDENTLOADFLAG:0xA00'; then
    printf '%s no longer links with /DEPENDENTLOADFLAG:0xA00; static imports would search the current directory\n' "$script" >&2
    exit 1
  fi
done
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
