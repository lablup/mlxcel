#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_kernel_dtype_keys.py (#1875).
#
# The checker's scope used to be invisible: it globbed `*.cpp` in two
# directories and skipped any file without `cuda_kernel(`, so moving a launch
# into a header, into a shared helper, or into a HIP-only file emptied the
# scope while the check still printed OK. A scope rule that is only ever run
# against a tree satisfying it is indistinguishable from no rule, so this test
# copies the launcher sources into a throwaway tree, mutates the copy the way a
# refactor would, and asserts the checker rejects it (or, for the positive
# controls, still accepts it).
#
# The copy is passed through --root. No build, GPU or network is needed.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
under_test="$script_dir/check_kernel_dtype_keys.py"

work="$(mktemp -d "${TMPDIR:-/tmp}/kernel-dtype-keys-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

# A fresh copy of every directory that holds a pinned launcher. The vendored
# MLX trees under src/lib/mlx-cpp come along, so the entry-point definitions
# (`CustomKernelFunction hip_kernel(`) are present and must not count.
make_tree() {
  local dir="$work/$1"
  # A failed copy must stop the test, not yield a tree that passes vacuously.
  mkdir -p "$dir/src/lib/mlxcel-core" || return 1
  cp -R "$repo_root/src/lib/mlx-cpp" "$dir/src/lib/mlx-cpp" || return 1
  cp -R "$repo_root/src/lib/mlxcel-core/cpp" "$dir/src/lib/mlxcel-core/cpp" || return 1
  echo "$dir"
}

run_case() {
  local name="$1" dir="$2" expected_status="$3"
  local stdout="$work/$name.out"
  set +e
  python3 "$under_test" --root "$dir" >"$stdout" 2>&1
  local status=$?
  set -e
  if [ "$status" -ne "$expected_status" ]; then
    echo "FAIL $name: expected exit $expected_status, got $status" >&2
    sed -n '1,40p' "$stdout" >&2
    failures=$((failures + 1))
    return 1
  fi
  echo "ok   $name -> exit $status"
  return 0
}

assert_contains() {
  local name="$1" needle="$2"
  if ! grep -qF -- "$needle" "$work/$name.out"; then
    echo "FAIL $name: output does not mention '$needle'" >&2
    sed -n '1,40p' "$work/$name.out" >&2
    failures=$((failures + 1))
  fi
}

assert_not_contains() {
  local name="$1" needle="$2"
  if grep -qF -- "$needle" "$work/$name.out"; then
    echo "FAIL $name: output unexpectedly mentions '$needle'" >&2
    sed -n '1,40p' "$work/$name.out" >&2
    failures=$((failures + 1))
  fi
}

# Write a launcher whose one `template_args` initialiser names no dtype.
# $1 file, $2 entry point (cuda_kernel or hip_kernel), $3 extra entry line.
write_launcher() {
  local file="$1" entry="$2" extra="${3:-}"
  mkdir -p "$(dirname "$file")"
  cat >"$file" <<CPP
#pragma once
inline void moved_launch(const mlx::core::array& x, int n) {
    auto kernel = mlx::core::fast::${entry}(
        "moved_launch", {"x"}, {"out"}, "out[0] = x[0];");
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
        {"N", n},${extra}
    };
}
CPP
}

sampling="src/lib/mlx-cpp/turbo/sampling.cpp"
bridge="src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp"

# Control: an untouched copy passes and reports both counts, so every failure
# below is caused by its mutation and not by the copy.
dir="$(make_tree control)"
if run_case control "$dir" 0; then
  assert_contains control "9 in scope (launching cuda_kernel or hip_kernel) of"
  assert_contains control "source files scanned"
fi

# The rule itself still bites: dropping the dtype key from a real launch fails.
dir="$(make_tree dtype-key-dropped)"
sed -i '/{"LogitsType", logits.dtype()},/d' "$dir/$sampling"
if run_case dtype-key-dropped "$dir" 1; then
  assert_contains dtype-key-dropped "$sampling:"
  assert_contains dtype-key-dropped "names no input dtype; keys are ['TgSize', 'NumSplits']"
fi

# A launch that moves into a header stays in scope and is checked. Before
# #1875 only `*.cpp` was globbed, and this passed.
dir="$(make_tree launch-in-header)"
write_launcher "$dir/src/lib/mlx-cpp/turbo/moved_launch.h" cuda_kernel
if run_case launch-in-header "$dir" 1; then
  assert_contains launch-in-header "src/lib/mlx-cpp/turbo/moved_launch.h:5: \`template_args\` names no input dtype"
fi

# Same for a header in a directory the old checker never searched.
dir="$(make_tree launch-in-new-directory)"
write_launcher "$dir/src/lib/mlxcel-core/cpp/detail/moved_launch.cuh" cuda_kernel
if run_case launch-in-new-directory "$dir" 1; then
  assert_contains launch-in-new-directory "src/lib/mlxcel-core/cpp/detail/moved_launch.cuh:5:"
fi

# A launch that moves into a shared helper takes its file out of scope while
# the `template_args` stay behind. The pin names the file that left, and the
# helper that arrived.
dir="$(make_tree launch-moved-to-helper)"
sed -i 's/mlx::core::fast::cuda_kernel(/mlxcel::make_jit_kernel(/' "$dir/$sampling"
cat >"$dir/src/lib/mlxcel-core/cpp/jit_helper.cpp" <<'CPP'
mlx::core::fast::CustomKernelFunction make_jit_kernel(const std::string& name) {
    return mlx::core::fast::cuda_kernel(name, {"x"}, {"out"}, "");
}
CPP
if run_case launch-moved-to-helper "$dir" 1; then
  assert_contains launch-moved-to-helper "$sampling: pinned in EXPECTED_IN_SCOPE but no longer calls"
  assert_contains launch-moved-to-helper "src/lib/mlxcel-core/cpp/jit_helper.cpp: launches a CUDA or HIP JIT kernel but is not pinned"
  assert_not_contains launch-moved-to-helper "names no input dtype"
fi

# A launch that is commented out is not a launch: its file leaves scope.
dir="$(make_tree launch-commented-out)"
sed -i 's|kernel = mlx::core::fast::cuda_kernel(|// kernel = mlx::core::fast::cuda_kernel(|' "$dir/$sampling"
if run_case launch-commented-out "$dir" 1; then
  assert_contains launch-commented-out "$sampling: pinned in EXPECTED_IN_SCOPE but no longer calls"
fi

# A deleted launcher is reported as deleted, distinguishable from a moved one.
dir="$(make_tree launcher-deleted)"
rm "$dir/$sampling"
if run_case launcher-deleted "$dir" 1; then
  assert_contains launcher-deleted "$sampling: pinned in EXPECTED_IN_SCOPE but no longer exists"
fi

# A HIP-only launcher is held to the same rule: ROCm keys its JIT cache as CUDA
# does. Before #1875 only `cuda_kernel(` scoped a file, and this passed.
dir="$(make_tree hip-only-new-file)"
write_launcher "$dir/src/lib/mlx-cpp/turbo/hip_only.cpp" hip_kernel
if run_case hip-only-new-file "$dir" 1; then
  assert_contains hip-only-new-file "src/lib/mlx-cpp/turbo/hip_only.cpp:5: \`template_args\` names no input dtype"
  assert_contains hip-only-new-file "src/lib/mlx-cpp/turbo/hip_only.cpp: launches a CUDA or HIP JIT kernel but is not pinned"
fi

# The real HIP-only launcher in the tree (the #1804 fault probe in the bridge,
# which has no `cuda_kernel(`) is in scope: an int-only initialiser added there
# fails, and a dtype-keyed one passes.
dir="$(make_tree hip-only-bridge-int-keys)"
cat >>"$dir/$bridge" <<'CPP'
void hip_only_probe(int n) {
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> probe_args = {
        {"N", n},
    };
}
CPP
if run_case hip-only-bridge-int-keys "$dir" 1; then
  assert_contains hip-only-bridge-int-keys "$bridge:"
  assert_contains hip-only-bridge-int-keys "\`probe_args\` names no input dtype"
  assert_not_contains hip-only-bridge-int-keys "EXPECTED_IN_SCOPE"
fi

dir="$(make_tree hip-only-bridge-dtype-keys)"
cat >>"$dir/$bridge" <<'CPP'
void hip_only_probe(const mlx::core::array& x, int n) {
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> probe_args = {
        {"T", x.dtype()},
        {"N", n},
    };
}
CPP
run_case hip-only-bridge-dtype-keys "$dir" 0

# A new launcher that keys correctly still has to be pinned, so the pin cannot
# go stale; the message is about the pin, not the rule.
dir="$(make_tree unpinned-launcher)"
write_launcher "$dir/src/lib/mlx-cpp/turbo/new_launcher.hpp" cuda_kernel '
        {"T", x.dtype()},'
if run_case unpinned-launcher "$dir" 1; then
  assert_contains unpinned-launcher "src/lib/mlx-cpp/turbo/new_launcher.hpp: launches a CUDA or HIP JIT kernel but is not pinned"
  assert_not_contains unpinned-launcher "names no input dtype"
fi

# Positive control for comment handling: a header that only mentions the
# token in comments is not a launcher, so its int-only initialiser is out of
# scope and the tree still passes. The raw string holds a quote and a `/*`, so
# a lexer that did not recognise raw strings would misread what follows it.
dir="$(make_tree token-in-comment)"
mkdir -p "$dir/src/lib/mlx-cpp/turbo"
cat >"$dir/src/lib/mlx-cpp/turbo/metal_only.h" <<'CPP'
// The CUDA port would call fast::cuda_kernel(...); this file is Metal-only.
/* Likewise fast::hip_kernel( is not called here. */
inline void metal_only(int n) {
    const char* src = R"(// a raw kernel source with a " quote and /* an unclosed comment
    )"; // fast::cuda_kernel( after the literal is still a comment
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> template_args = {
        {"N", n},
    };
}
CPP
run_case token-in-comment "$dir" 0

# An empty scope never passes, whatever the pin says.
mkdir -p "$work/empty-tree/src"
if run_case empty-tree "$work/empty-tree" 1; then
  assert_contains empty-tree "0 in scope"
  assert_contains empty-tree "would pass over nothing"
fi

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
echo "all kernel dtype-key checker cases passed"
