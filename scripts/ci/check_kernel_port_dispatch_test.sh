#!/usr/bin/env bash
# Negative coverage for the ROCm wavefront rules of
# scripts/ci/check_kernel_port_dispatch.py (rules 5 and 6, issue #2147).
#
# Rule 5 lets a port table run its HIP body on a 64-lane device that no one has
# run it on, because the body has no lane-level operation. A rule like that is
# only worth something if it fails when the claim stops being true, so this
# copies the launcher sources into a throwaway tree, mutates the copy the way a
# careless change would, and asserts the checker rejects it (or, for the
# controls, still accepts it). Rule 6 keeps the wave-size test seam out of
# production code and is exercised the same way.
#
# The copy is passed through --root. No build, GPU or network is needed.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
under_test="$script_dir/check_kernel_port_dispatch.py"

work="$(mktemp -d "${TMPDIR:-/tmp}/kernel-port-dispatch-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

# The launcher sources the rules read: the turbo kernels, mlxcel-core's C++
# bridge, and the Rust bridge declaration in lib.rs.
make_tree() {
  local dir="$work/$1"
  mkdir -p "$dir/src/lib/mlx-cpp" "$dir/src/lib/mlxcel-core/src" || return 1
  cp -R "$repo_root/src/lib/mlx-cpp/turbo" "$dir/src/lib/mlx-cpp/turbo" || return 1
  cp -R "$repo_root/src/lib/mlxcel-core/cpp" "$dir/src/lib/mlxcel-core/cpp" || return 1
  cp "$repo_root/src/lib/mlxcel-core/src/lib.rs" "$dir/src/lib/mlxcel-core/src/lib.rs" || return 1
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

# Replace one literal string in a file, failing when it matches nothing, so a
# case cannot pass because its mutation silently stopped applying.
replace_in() {
  python3 - "$@" <<'PY'
import pathlib
import sys

path, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text()
if old not in text:
    raise SystemExit(f"fixture mutation: {old!r} not found in {path}")
path.write_text(text.replace(old, new, 1))
PY
}

norm="src/lib/mlx-cpp/turbo/fused_norm.cpp"
gumbel_hip="src/lib/mlx-cpp/turbo/sampling_gumbel_hip.h"
rejection="src/lib/mlx-cpp/turbo/sampling_rejection.cpp"
kernels="src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp"
bridge="src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp"
norm_rocm_entry='            return get_fused_norm_kernel_hip().get();
        },'
gumbel_body_start='GUMBEL_MAX_SAMPLE_HIP_SOURCE = R"('

# Control: an untouched copy passes and reports the five marked tables.
dir="$(make_tree control)"
if run_case control "$dir" 0; then
  assert_contains control "5 any-wave ROCm table(s) free of lane operations"
fi

# Marking a shuffle-based port any-wave fails twice: its body uses a lane
# intrinsic, and the table is not in the pin.
dir="$(make_tree mark-shuffle-port)"
replace_in "$dir/$norm" "$norm_rocm_entry" "$norm_rocm_entry
        .rocm_any_wave_size = true,"
if run_case mark-shuffle-port "$dir" 1; then
  assert_contains mark-shuffle-port "fused_norm_ports is marked rocm_any_wave_size, but FUSED_ADD_RMS_NORM_HIP_SOURCE"
  assert_contains mark-shuffle-port "not in EXPECTED_ANY_WAVE"
fi

# A shuffle added to a marked body fails.
dir="$(make_tree shuffle-in-marked-body)"
replace_in "$dir/$gumbel_hip" "$gumbel_body_start" "$gumbel_body_start
    float probe = __shfl_xor(0.0f, 1, 32);"
if run_case shuffle-in-marked-body "$dir" 1; then
  assert_contains shuffle-in-marked-body "gumbel_ports is marked rocm_any_wave_size, but GUMBEL_MAX_SAMPLE_HIP_SOURCE"
fi

# So does a ballot, a different intrinsic family.
dir="$(make_tree ballot-in-marked-body)"
replace_in "$dir/$gumbel_hip" "$gumbel_body_start" "$gumbel_body_start
    unsigned long long probe = __ballot(1);"
if run_case ballot-in-marked-body "$dir" 1; then
  assert_contains ballot-in-marked-body "uses \`__ballot\`"
fi

# The AMDGCN builtin a ballot lowers to is caught too.
dir="$(make_tree builtin-ballot-in-marked-body)"
replace_in "$dir/$gumbel_hip" "$gumbel_body_start" "$gumbel_body_start
    unsigned probe = __builtin_amdgcn_ballot_w32(true);"
if run_case builtin-ballot-in-marked-body "$dir" 1; then
  assert_contains builtin-ballot-in-marked-body "uses \`__builtin_amdgcn_ballot_w32\`"
fi

# Pointing a marked table's .rocm entry at another holder fails, although the
# pinned source is still compiled elsewhere in the same file.
dir="$(make_tree marked-entry-swapped)"
replace_in "$dir/$kernels" "            return get_xielu_kernel_hip().get();" "            return get_ssm_kernel_hip().get();"
if run_case marked-entry-swapped "$dir" 1; then
  assert_contains marked-entry-swapped "xielu_ports is pinned to XIELU_HIP_SOURCE, but its .rocm entry compiles ['SSM_HIP_SOURCE']"
fi

# A marked holder that passes more after the source (a header) is not trusted.
dir="$(make_tree marked-with-header)"
replace_in "$dir/$kernels" "                    XIELU_HIP_SOURCE);" "                    XIELU_HIP_SOURCE, XIELU_HEADER);"
if run_case marked-with-header "$dir" 1; then
  assert_contains marked-with-header "passes arguments after XIELU_HIP_SOURCE"
fi

# A lane intrinsic named only in a comment inside the body is not an operation.
dir="$(make_tree shuffle-in-comment)"
replace_in "$dir/$gumbel_hip" "$gumbel_body_start" "$gumbel_body_start
    // no __shfl_xor here, every reduction goes through shared memory"
run_case shuffle-in-comment "$dir" 0 || true

# Dropping a mark the pin expects fails, so a table cannot silently move to
# "refused on wave64" either.
dir="$(make_tree unmark-pinned-table)"
replace_in "$dir/$rejection" "        .rocm_any_wave_size = true," ""
if run_case unmark-pinned-table "$dir" 1; then
  assert_contains unmark-pinned-table "EXPECTED_ANY_WAVE pins rejection_ports"
fi

# Production code calling the test seam fails ...
dir="$(make_tree seam-in-production)"
mkdir -p "$dir/src/models"
cat >"$dir/src/models/wave.rs" <<'RS'
pub fn pretend_wave32() {
    mlxcel_core::set_rocm_port_warp_size_for_tests(32);
}
RS
if run_case seam-in-production "$dir" 1; then
  assert_contains seam-in-production "src/models/wave.rs:2: calls set_rocm_port_warp_size_for_tests outside a test"
fi

# ... in C++ too ...
dir="$(make_tree seam-in-cpp)"
replace_in "$dir/src/lib/mlx-cpp/turbo/kernel_port.cpp" "namespace mlxcel {" "namespace mlxcel {
static int forced = (set_rocm_port_warp_size_for_tests(32), 0);"
if run_case seam-in-cpp "$dir" 1; then
  assert_contains seam-in-cpp "kernel_port.cpp: calls set_rocm_port_warp_size_for_tests outside its definition"
fi

# ... including a second call inside the bridge file that defines it ...
dir="$(make_tree seam-in-bridge)"
replace_in "$dir/$bridge" "int32_t rocm_device_warp_size() {" "int32_t rocm_device_warp_size() {
    mlxcel::set_rocm_port_warp_size_for_tests(32);"
if run_case seam-in-bridge "$dir" 1; then
  assert_contains seam-in-bridge "mlx_cxx_bridge.cpp: calls set_rocm_port_warp_size_for_tests outside its definition (3 uses, 2 expected)"
fi

# ... and a test module may call it.
dir="$(make_tree seam-in-test-module)"
mkdir -p "$dir/src/models"
cat >"$dir/src/models/wave_tests.rs" <<'RS'
#[test]
fn wave64() {
    mlxcel_core::set_rocm_port_warp_size_for_tests(64);
}
RS
run_case seam-in-test-module "$dir" 0 || true

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
echo "all kernel port dispatch checker cases passed"
