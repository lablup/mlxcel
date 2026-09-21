#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_cuda_arch_lists.py.
#
# A guard that only ever passes proves nothing. Each case below builds a
# throwaway root holding one workflow, one installation document and one
# CMakeLists, runs the checker against it with --root, and asserts both the
# exit status and the line the operator is supposed to read.
#
# The case that matters most is the architecture-specific Blackwell list. That
# is the obvious fix for issue #1934 and the one an earlier revision of that
# issue's own PR shipped: it works, and it charges every translation unit for a
# converter only one of them reaches. The `121f` case is the second most likely
# wrong turn, since `f` reads as a milder `a` and does not compile at all.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
under_test="$script_dir/check_cuda_arch_lists.py"

work="$(mktemp -d "${TMPDIR:-/tmp}/cuda-arch-lists-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

# Builds a root with one workflow carrying $2 as its architecture list, and an
# installation document quoting $3 (default: the same list, so a case that is
# not about documentation drift never trips the documentation rule).
setup_root() {
  local root="$1" list="$2" documented="${3-$2}" injection="${4-present}"
  mkdir -p "$root/.github/workflows" "$root/docs" "$root/src/lib/mlx-cpp"
  cat > "$root/.github/workflows/release.yml" <<YAML
jobs:
  build:
    env:
      MLX_CUDA_ARCHITECTURES: "$list"
YAML
  printf 'The release workflow builds `%s` on this target.\n' "$documented" > "$root/docs/installation.md"
  if [ "$injection" = "present" ]; then
    cat > "$root/src/lib/mlx-cpp/CMakeLists.txt" <<'CM'
set_source_files_properties(
  "${mlx_SOURCE_DIR}/mlx/backend/cuda/quantized/fp_quantize.cu"
  TARGET_DIRECTORY mlx PROPERTIES COMPILE_OPTIONS "${_fp_quantize_arch_flags}")
CM
  else
    printf '# the injection was removed\n' > "$root/src/lib/mlx-cpp/CMakeLists.txt"
  fi
}

# Runs the checker and asserts the exit status and, when given, that the output
# contains an expected fragment.
expect() {
  local name="$1" root="$2" want_status="$3" want_text="${4-}"
  local output status
  set +e
  output="$(python3 "$under_test" --root "$root" 2>&1)"
  status=$?
  set -e
  if [ "$status" -ne "$want_status" ]; then
    echo "FAIL: $name: expected exit $want_status, got $status" >&2
    echo "$output" | sed 's/^/    /' >&2
    failures=$((failures + 1))
    return
  fi
  if [ -n "$want_text" ] && ! printf '%s' "$output" | grep -qF -- "$want_text"; then
    echo "FAIL: $name: output did not mention '$want_text'" >&2
    echo "$output" | sed 's/^/    /' >&2
    failures=$((failures + 1))
    return
  fi
  echo "ok: $name"
}

# 1. What release.yml ships, and what it must keep shipping.
setup_root "$work/plain" "90a;100;121"
expect "plain Blackwell list passes" "$work/plain" 0 "CUDA architecture lists OK"

# 2. `f` compiles neither here nor in CI; rejecting it by name saves a CUDA job.
setup_root "$work/family" "121f;121"
expect "family-specific f is rejected" "$work/family" 1 "does not compile"

# 3. The whole-target form an earlier revision of #1938 shipped. It compiles the
#    converter and charges every other kernel for it.
setup_root "$work/combined" "90a;100a-real;100;121a-real;121"
expect "architecture-specific Blackwell fails" "$work/combined" 1 "architecture-specific at the target level"

# 4. The same mistake in its simplest spelling.
setup_root "$work/aonly" "90a;121a"
expect "bare 121a fails" "$work/aonly" 1 "architecture-specific at the target level"

# 5. Hopper keeps its suffix: it is load-bearing for MLX's own quantized-kernel
#    gate and cannot reach these converters anyway.
setup_root "$work/hopper" "90a"
expect "Hopper 90a is untouched" "$work/hopper" 0 "CUDA architecture lists OK"

# 6. Pre-Blackwell lists are untouched: Hopper fails the converter's
#    __CUDA_ARCH__ >= 1000 gate however it is spelled.
setup_root "$work/ampere" "70"
expect "pre-Blackwell list passes untouched" "$work/ampere" 0 "CUDA architecture lists OK"

# 7. A spelling nvcc would reject must not reach nvcc.
setup_root "$work/garbage" "sm_121a"
expect "unparseable entry fails" "$work/garbage" 1 "not a CMake architecture spelling"

# 8. Documentation drift, which is how the plain-only lists stayed invisible.
setup_root "$work/docs" "121a-real;121" "121"
expect "release list absent from installation.md fails" "$work/docs" 1 "does not name"

# 9. The per-source injection is what lets the plain lists above be correct.
#    Without it they are the original defect again, silently.
setup_root "$work/noinject" "90a;100;121" "90a;100;121" "absent"
expect "missing per-source injection fails" "$work/noinject" 1 "no longer carries the per-source architecture injection"

# 10. A workflow set that pins nothing hands the decision to auto-detection, so
#    what ships stops being reviewable at all.
setup_root "$work/empty" "placeholder"
trash-put "$work/empty/.github/workflows/release.yml" 2>/dev/null || rm -f "$work/empty/.github/workflows/release.yml"
printf 'jobs:\n  build:\n    runs-on: ubuntu-latest\n' > "$work/empty/.github/workflows/ci.yml"
expect "no architecture pin at all fails" "$work/empty" 1 "no MLX_CUDA_ARCHITECTURES assignment"

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
echo "all cases passed"
