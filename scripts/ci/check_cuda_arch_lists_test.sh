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

# Writes the two Rust sources that spell an architecture list from a detected
# capability, in the shape named by $2 (build.rs) and $3 (cuda_arch.rs):
# `bounded` is the shipped rule, `unbounded` is the pre-#1943 one that suffixes
# Blackwell too, and `wrong-boundary` keeps the constant but moves it off the
# first Blackwell capability. Only the parts the checker reads are reproduced.
write_suffix_rules() {
  local root="$1" build_rs="$2" cuda_arch_rs="$3"
  mkdir -p "$root/src/lib/mlxcel-core/src"
  case "$build_rs" in
    bounded)
      cat > "$root/src/lib/mlxcel-core/build.rs" <<'RS'
const FIRST_PLAIN_SM: u32 = 100;

fn sm_arch_with_suffix(sm: &str) -> String {
    match sm.parse::<u32>() {
        Ok(n) if (90..FIRST_PLAIN_SM).contains(&n) => format!("{sm}a"),
        _ => sm.to_string(),
    }
}
RS
      ;;
    wrong-boundary)
      cat > "$root/src/lib/mlxcel-core/build.rs" <<'RS'
const FIRST_PLAIN_SM: u32 = 120;

fn sm_arch_with_suffix(sm: &str) -> String {
    match sm.parse::<u32>() {
        Ok(n) if (90..FIRST_PLAIN_SM).contains(&n) => format!("{sm}a"),
        _ => sm.to_string(),
    }
}
RS
      ;;
    *)
      cat > "$root/src/lib/mlxcel-core/build.rs" <<'RS'
fn sm_arch_with_suffix(sm: &str) -> String {
    match sm.parse::<u32>() {
        Ok(n) if n >= 90 => format!("{sm}a"),
        _ => sm.to_string(),
    }
}
RS
      ;;
  esac
  if [ "$cuda_arch_rs" = "bounded" ]; then
    cat > "$root/src/lib/mlxcel-core/src/cuda_arch.rs" <<'RS'
const FIRST_PLAIN_SM: u32 = 100;

impl CudaArchMismatch {
    pub fn suggested_architecture(&self) -> String {
        let (major, minor) = self.device;
        let sm = major * 10 + minor;
        let suffix = if (90..FIRST_PLAIN_SM).contains(&sm) {
            "a"
        } else {
            ""
        };
        format!("{sm}{suffix}")
    }
}
RS
  else
    cat > "$root/src/lib/mlxcel-core/src/cuda_arch.rs" <<'RS'
impl CudaArchMismatch {
    pub fn suggested_architecture(&self) -> String {
        let (major, minor) = self.device;
        let sm = major * 10 + minor;
        let suffix = if sm >= 90 { "a" } else { "" };
        format!("{sm}{suffix}")
    }
}
RS
  fi
}

# Builds a root with one workflow carrying $2 as its architecture list, and an
# installation document quoting $3 (default: the same list, so a case that is
# not about documentation drift never trips the documentation rule). $5 and $6
# select the shape of the two auto-detect suffix rules, both correct by default
# so a case that is not about them never trips rule 6.
setup_root() {
  local root="$1" list="$2" documented="${3-$2}" injection="${4-present}"
  local build_rs="${5-bounded}" cuda_arch_rs="${6-bounded}"
  mkdir -p "$root/.github/workflows" "$root/docs" "$root/src/lib/mlx-cpp"
  write_suffix_rules "$root" "$build_rs" "$cuda_arch_rs"
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

# 5. Hopper keeps its suffix: 9.0 cannot reach these converters however it is
#    spelled, and `90a` is what both release lists ship.
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

# 11. The list nobody writes down. With MLX_CUDA_ARCHITECTURES unset, build.rs
#    spells one from nvidia-smi, and the pre-#1943 rule suffixed everything from
#    SM 90 up, so this host's own default build resolved to `121a`: the shape
#    case 4 rejects in a workflow, reached by a path no workflow names.
setup_root "$work/autodetect" "90a;100;121" "90a;100;121" "present" "unbounded"
expect "unbounded auto-detect suffix rule fails" "$work/autodetect" 1 "sm_arch_with_suffix"

# 12. The same rule on the runtime side. This one is user-facing: the startup
#    mismatch message is the only place the project hands an operator an
#    MLX_CUDA_ARCHITECTURES value to paste, and unbounded it pastes `121a`.
setup_root "$work/suggestion" "90a;100;121" "90a;100;121" "present" "bounded" "unbounded"
expect "unbounded startup suggestion fails" "$work/suggestion" 1 "MLX_CUDA_ARCHITECTURES=121a"

# 13. A boundary that exists but sits somewhere other than the first Blackwell
#    capability. `120` looks plausible and still suffixes sm_100.
setup_root "$work/boundary" "90a;100;121" "90a;100;121" "present" "wrong-boundary"
expect "misplaced plain-spelling boundary fails" "$work/boundary" 1 "sets FIRST_PLAIN_SM to 120"

# The other half of the #1943 decision, that Hopper keeps the `90a` the release
# lists ship, is not checkable here: this script reads text and cannot evaluate
# the rule. `the_suggested_rebuild_matches_the_shipped_spelling` in
# src/lib/mlxcel-core/src/cuda_arch_tests.rs asserts the produced spelling for
# H100 and GB10 instead, and runs under `cargo test`.

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
echo "all cases passed"
