#!/usr/bin/env bash
# Regression coverage for the A/B script default output directories.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${MLXCEL_AB_SCRIPTS_TEST_ROOT:-$(cd "$script_dir/../.." && pwd)}"
output_equality="$repo_root/scripts/ab_output_equality.sh"
paged_counter="$repo_root/scripts/paged_decode_counter_ab.sh"

work="$(mktemp -d "${TMPDIR:-/tmp}/ab-scripts-outdir-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

fail() {
  echo "FAIL [$1]: $2" >&2
  failures=$((failures + 1))
}

write_fake_mlxcel() {
  local path="$1" marker="$2"
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    "# $marker" \
    'if [[ -n "${AB_SCRIPTS_OUTDIR_TEST_MODEL_MARKER:-}" ]]; then' \
    '  : > "$AB_SCRIPTS_OUTDIR_TEST_MODEL_MARKER"' \
    'fi' \
    'case "${1:-}" in' \
    '  generate)' \
    '    printf "Generating...\\n"' \
    '    printf "same generated text\\n"' \
    '    printf "[Generated 1 tokens in 0.01s]\\n"' \
    '    ;;' \
    '  serve)' \
    '    while :; do sleep 1; done' \
    '    ;;' \
    '  *)' \
    '    exit 1' \
    '    ;;' \
    'esac' > "$path"
  chmod +x "$path"
}

write_fake_curl() {
  local path="$1"
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'output=""' \
    'url=""' \
    'while [[ $# -gt 0 ]]; do' \
    '  case "$1" in' \
    '    -o) output="$2"; shift 2 ;;' \
    '    -H|--data-binary) shift 2 ;;' \
    '    -*) shift ;;' \
    '    *) url="$1"; shift ;;' \
    '  esac' \
    'done' \
    'case "$url" in' \
    '  */health) body="{}" ;;' \
    '  */metrics) body=$(printf "mlxcel_paged_decode_launches_total{path=\\\"fused_v2\\\"} 1\\nmlxcel_paged_decode_launches_total{path=\\\"gather\\\"} 0\\nmlxcel_paged_decode_launches_total{path=\\\"cascade\\\"} 0") ;;' \
    '  */v1/completions) body="{\\\"choices\\\":[{\\\"text\\\":\\\"same\\\"}]}" ;;' \
    '  *) exit 1 ;;' \
    'esac' \
    'if [[ -n "$output" ]]; then' \
    '  printf "%s\\n" "$body" > "$output"' \
    'else' \
    '  printf "%s\\n" "$body"' \
    'fi' > "$path"
  chmod +x "$path"
}

write_fake_jq() {
  local path="$1"
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'if [[ "${1:-}" == "-n" ]]; then' \
    '  printf "{}\\n"' \
    '  exit 0' \
    'fi' \
    'if [[ "${1:-}" == "-r" ]]; then' \
    '  printf "same\\n"' \
    '  exit 0' \
    'fi' \
    'exit 1' > "$path"
  chmod +x "$path"
}

assert_default_dir() {
  local name="$1" prefix="$2" output="$3"
  local out_dir
  out_dir="$(sed -n 's/^outputs:[[:space:]]*//p' <<< "$output")"
  case "$out_dir" in
    "$work/tmp/$prefix".??????) ;;
    *)
      fail "$name" "expected an output directory below $work/tmp, got '${out_dir:-<empty>}'"
      return
      ;;
  esac
  [[ -d "$out_dir" ]] || fail "$name" "output directory was not created: $out_dir"
}

assert_create_failure() {
  local name="$1" script="$2"
  shift 2
  local output="$work/$name.failure.out"
  local status
  set +e
  "$script" "$@" --out-dir "$work/not-a-directory/child" > "$output" 2>&1
  status=$?
  set -e
  [[ "$status" -eq 2 ]] || fail "$name create failure" "expected exit 2, got $status"
  grep -Fq 'error: could not create output directory' "$output" || fail "$name create failure" "missing output-directory error"
}

assert_mktemp_failure() {
  local name="$1" script="$2"
  shift 2
  local marker="$work/$name.model-invoked"
  local output="$work/$name.mktemp-failure.out"
  local status
  rm -f "$marker"
  set +e
  AB_SCRIPTS_OUTDIR_TEST_MODEL_MARKER="$marker" TMPDIR="$work/not-a-directory/missing" "$script" "$@" > "$output" 2>&1
  status=$?
  set -e
  [[ "$status" -eq 2 ]] || fail "$name mktemp failure" "expected exit 2, got $status"
  grep -Fq 'error: could not create output directory' "$output" || fail "$name mktemp failure" "missing output-directory error"
  [[ ! -e "$marker" ]] || fail "$name mktemp failure" "model binary was invoked after output-directory creation failed"
}

mkdir -p "$work/tmp" "$work/bin"
write_fake_mlxcel "$work/baseline" baseline
write_fake_mlxcel "$work/arm" arm
write_fake_curl "$work/bin/curl"
write_fake_jq "$work/bin/jq"
printf 'not a directory\n' > "$work/not-a-directory"

set +e
output="$(TMPDIR="$work/tmp" "$output_equality" --baseline "$work/baseline" --arm "$work/arm" --model models/fake --prompt test -n 1 2>&1)"
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  assert_default_dir "output equality default" ab_output_equality "$output"
else
  fail "output equality default" "expected exit 0, got $status"
fi
assert_create_failure "output equality" "$output_equality" --baseline "$work/baseline" --arm "$work/arm" --model models/fake --prompt test -n 1
assert_mktemp_failure "output equality" "$output_equality" --baseline "$work/baseline" --arm "$work/arm" --model models/fake --prompt test -n 1

set +e
output="$(PATH="$work/bin:$PATH" TMPDIR="$work/tmp" "$paged_counter" --baseline "$work/baseline" --arm "$work/arm" --model models/fake --requests 1 -n 1 2>&1)"
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  assert_default_dir "paged counter default" paged_decode_counter_ab "$output"
else
  fail "paged counter default" "expected exit 0, got $status"
fi
assert_create_failure "paged counter" "$paged_counter" --baseline "$work/baseline" --arm "$work/arm" --model models/fake --requests 1 -n 1
assert_mktemp_failure "paged counter" "$paged_counter" --baseline "$work/baseline" --arm "$work/arm" --model models/fake --requests 1 -n 1

if [[ "$failures" -ne 0 ]]; then
  echo "A/B output-directory regression test: $failures case(s) failed" >&2
  exit 1
fi

echo "A/B output-directory regression test: all cases passed"
