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

# Test the Windows CUDA runtime and include directory choice (#2640) and the PTX
# cache path rules (#2797) on the host, without CUDA: the pure functions of the
# CUDA overlay's runtime_dir.h and ptx_cache_dir.h.
# Run by `make verify-cuda-runtime-dir` and, on every PR and push to main, by
# the overlay-host-tests job in .github/workflows/ci.yml (#2642).
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
overlay="$repo_root/src/lib/mlxcelverse/overlays/cuda"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/cuda-runtime-dir.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT
# A deleted call compiles and passes everything else, so check the source. Comment
# lines do not count (#2641).
delayload="$overlay/mlx/backend/cuda/delayload.cpp"
code_has() {
  # Not a pipe into grep -q: it exits at the first match, and under pipefail the
  # SIGPIPE that gives the first grep then reads as "not found" on a large file.
  local code
  code=$(grep -v '^[[:space:]]*//' "$1")
  grep -qF "$2" <<<"$code"
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
if ! code_has "$delayload" 'dll_search_set_at_startup = [] {'; then
  printf 'delayload.cpp lost the static initializer that runs ahead of the static cudart\n' >&2
  exit 1
fi
if ! code_has "$delayload" '#pragma init_seg(lib)'; then
  printf 'delayload.cpp lost the init_seg pragma that orders its initializer\n' >&2
  exit 1
fi
for script in build.rs src/lib/mlxcel-core/build.rs; do
  if ! code_has "$repo_root/$script" '/DEPENDENTLOADFLAG:0xA00'; then
    printf '%s no longer links with /DEPENDENTLOADFLAG:0xA00; static imports would search the current directory\n' "$script" >&2
    exit 1
  fi
done
# The cache's location rules are tested below, and jit_module.cpp has to call them.
jit_module="$overlay/mlx/backend/cuda/jit_module.cpp"
ptx_cache_dir=$(awk '
  index($0, "const std::filesystem::path& ptx_cache_dir()") == 1 { copying = 1 }
  copying { print }
  copying && /^}/ { exit }
' "$jit_module")
for needle in 'ptx_cache_env_from_process()' 'ptx_cache_env_refusal(' 'resolve_ptx_cache_dir('; do
  if ! grep -qF "$needle" <<<"$ptx_cache_dir"; then
    printf 'ptx_cache_dir() in jit_module.cpp no longer calls %s\n' "$needle" >&2
    exit 1
  fi
done
if ! code_has "$jit_module" 'resolve_ptx_cache_subdir('; then
  printf 'jit_module.cpp no longer checks the device subdirectory on Windows\n' >&2
  exit 1
fi
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"$overlay" \
  "$repo_root/scripts/mlxcelverse/cuda_runtime_dir_test.cpp" \
  -o "$test_dir/cuda-runtime-dir-test"
"$test_dir/cuda-runtime-dir-test"
printf '[cuda-runtime-dir] runtime directory order, override errors, marker, preload check, include order and PTX cache path rules passed (host only; no CUDA)\n'
