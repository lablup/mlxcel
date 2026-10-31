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

# Test the ROCm JIT kernel disk cache's location and trust check (#2641) on the
# host, without HIP: hsaco_cache_dir.h from the overlay, and the cache read and
# write functions of jit_module.cpp, copied out by awk so the code tested is the
# code built. Also checks that hsaco_cache_dir() in jit_module.cpp really calls
# the header, and that the shared temp default did not come back. Run by
# `make verify-rocm-overlay` and, on every PR and push to main, by the
# overlay-host-tests job in .github/workflows/ci.yml.
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
overlay="$repo_root/src/lib/mlxcelverse/overlays/rocm"
jit_module="$overlay/mlx/backend/rocm/jit_module.cpp"
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/rocm-hsaco-cache.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT

copy_function() {
  awk -v start="$1" '
    index($0, start) == 1 { copying = 1 }
    copying { print }
    copying && /^}/ { found = 1; exit }
    END { if (!found) exit 1 }
  ' "$jit_module"
}
{
  copy_function 'std::filesystem::path get_hsaco_path('
  copy_function 'bool read_cached_hsaco('
  copy_function 'void write_cached_hsaco('
} > "$test_dir/hsaco_cache_read.inc"

hsaco_cache_dir=$(awk '
  index($0, "const std::filesystem::path& hsaco_cache_dir()") == 1 { copying = 1 }
  copying { print }
  copying && /^}/ { exit }
' "$jit_module")
for needle in 'resolve_hsaco_cache_dir(' 'hsaco_cache_env_from_process()' 'report_hsaco_cache_disabled('; do
  if ! grep -qF "$needle" <<<"$hsaco_cache_dir"; then
    printf 'hsaco_cache_dir() in jit_module.cpp no longer calls %s\n' "$needle" >&2
    exit 1
  fi
done
if grep -q 'temp_directory_path' "$jit_module"; then
  printf 'jit_module.cpp uses temp_directory_path again; the kernel cache must not default to a shared directory\n' >&2
  exit 1
fi

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"$overlay" -I"$test_dir" \
  "$repo_root/scripts/mlxcelverse/rocm_hsaco_cache_test.cpp" \
  -o "$test_dir/rocm-hsaco-cache-test"
mkdir "$test_dir/base"
chmod 700 "$test_dir/base" 2>/dev/null || true
"$test_dir/rocm-hsaco-cache-test" "$test_dir/base"
printf '[rocm-hsaco-cache] default location, ownership and mode refusals and entry checks passed (host only; no HIP)\n'
