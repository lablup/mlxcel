#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_cuda_arch_lists.py.
#
# A guard that only ever passes proves nothing. Each case below builds a
# throwaway root holding one workflow and one installation document, runs the
# checker against it with --root, and asserts both the exit status and the line
# the operator is supposed to read. The plain-only Blackwell case is the one
# that matters most, because `90a;100;121` is what this repository actually
# shipped (issue #1934), and the `121f` case is the one a well-meaning fix is
# most likely to reach for, since `f` looks like a milder `a`.

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
  local root="$1" list="$2" documented="${3-$2}"
  mkdir -p "$root/.github/workflows" "$root/docs"
  cat > "$root/.github/workflows/release.yml" <<YAML
jobs:
  build:
    env:
      MLX_CUDA_ARCHITECTURES: "$list"
YAML
  printf 'The release workflow builds `%s` on this target.\n' "$documented" > "$root/docs/installation.md"
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

# 1. The regression this guard exists for: exactly what release.yml shipped.
setup_root "$work/plain" "90a;100;121"
expect "plain-only Blackwell fails" "$work/plain" 1 "is plain-only"

# 2. `f` compiles neither here nor in CI; rejecting it by name saves a CUDA job.
setup_root "$work/family" "121f;121"
expect "family-specific f is rejected" "$work/family" 1 "does not compile"

# 3. The combined form: hardware SASS plus forward-JIT-capable plain PTX.
setup_root "$work/combined" "90a;100a-real;100;121a-real;121"
expect "combined form passes" "$work/combined" 0 "CUDA architecture lists OK"

# 4. The `a`-only form. Review may prefer it; the guard must not force the other.
setup_root "$work/aonly" "90a;121a"
expect "arch-specific-only form passes" "$work/aonly" 0 "CUDA architecture lists OK"

# 5. An `a` entry that emits no cubin leaves the SASS gate unsatisfied.
setup_root "$work/virtual" "121a-virtual;121"
expect "arch-specific PTX without a cubin fails" "$work/virtual" 1 "is plain-only"

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

# 9. A workflow set that pins nothing hands the decision to auto-detection, so
#    what ships stops being reviewable at all.
mkdir -p "$work/empty/.github/workflows" "$work/empty/docs"
printf 'jobs:\n  build:\n    runs-on: ubuntu-latest\n' > "$work/empty/.github/workflows/ci.yml"
: > "$work/empty/docs/installation.md"
expect "no architecture pin at all fails" "$work/empty" 1 "no MLX_CUDA_ARCHITECTURES assignment"

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
echo "all cases passed"
