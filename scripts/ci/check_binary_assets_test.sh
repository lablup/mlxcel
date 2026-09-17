#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_binary_assets.py.
#
# A guard that only ever passes proves nothing: the useful question is whether
# it fails on the case it exists for. Each case below builds a throwaway
# repository, copies the checker in, plants one violation, and asserts both the
# exit status and the line the operator is supposed to read. The undeclared-PNG
# case is the one that matters most, because it is the regression that
# webui/tests/screenshots/ actually was.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
under_test="$script_dir/check_binary_assets.py"

work="$(mktemp -d "${TMPDIR:-/tmp}/binary-assets-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

# A minimal real PNG: 1x1, and genuinely NUL-bearing so the sniffer sees it.
write_png() {
  printf '\211PNG\r\n\032\n\000\000\000\015IHDR\000\000\000\001\000\000\000\001\010\006\000\000\000\037\025\304\211\000\000\000\012IDATx\234c\000\001\000\000\005\000\001\r\n\055\262\000\000\000\000IEND\256B\140\202' > "$1"
}

# Builds a repo whose checker sees `tests/fixtures/*.png` as the only rule, with
# the ceilings passed in, so cases can drive each failure mode independently.
setup_repo() {
  local repo="$1" max_bytes="$2" total_budget="$3"
  mkdir -p "$repo/scripts/ci" "$repo/tests/fixtures"
  git init -q "$repo"
  git -C "$repo" config user.name "mlxcel test"
  git -C "$repo" config user.email "mlxcel-test@example.com"

  python3 - "$under_test" "$repo/scripts/ci/check_binary_assets.py" \
    "$max_bytes" "$total_budget" <<'PY'
import pathlib, re, sys
src, dst, max_bytes, total_budget = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
text = pathlib.Path(src).read_text()
text = re.sub(r"TOTAL_BUDGET_BYTES = .*", f"TOTAL_BUDGET_BYTES = {total_budget}", text, count=1)
# Replace the real declaration set with a single rule the cases can violate.
start = text.index("DECLARED = (")
end = text.index("\n)\n", start) + len("\n)\n")
rule = (
    "DECLARED = (\n"
    "    Rule(\n"
    '        pattern="tests/fixtures/*.png",\n'
    f"        max_bytes={max_bytes},\n"
    '        why="test fixture",\n'
    "    ),\n"
    ")\n"
)
pathlib.Path(dst).write_text(text[:start] + rule + text[end:])
PY
}

run_case() {
  local name="$1" repo="$2" expected_status="$3" expected_text="$4"
  local stdout="$work/$name.stdout"
  set +e
  (cd "$repo" && python3 scripts/ci/check_binary_assets.py) > "$stdout" 2>&1
  local status=$?
  set -e

  if [[ "$status" -ne "$expected_status" ]]; then
    echo "FAIL [$name]: expected exit $expected_status, got $status"
    sed 's/^/    /' "$stdout"
    failures=$((failures + 1))
    return
  fi
  if ! grep -qF "$expected_text" "$stdout"; then
    echo "FAIL [$name]: output did not contain '$expected_text'"
    sed 's/^/    /' "$stdout"
    failures=$((failures + 1))
    return
  fi
  echo "ok   [$name]"
}

# 1. The regression this guard exists for: a binary nobody declared.
repo="$work/undeclared"
setup_repo "$repo" $((32 * 1024)) $((256 * 1024))
mkdir -p "$repo/webui/tests/screenshots/darwin"
write_png "$repo/webui/tests/screenshots/darwin/1440-light-gallery-data.png"
git -C "$repo" add -A
run_case "undeclared-png" "$repo" 1 "undeclared: webui/tests/screenshots/darwin/1440-light-gallery-data.png"

# 2. A declared path is not a blank cheque: the per-file ceiling still applies.
repo="$work/oversize"
setup_repo "$repo" 64 $((256 * 1024))
write_png "$repo/tests/fixtures/probe.png"
git -C "$repo" add -A
run_case "oversize-declared" "$repo" 1 "oversize:   tests/fixtures/probe.png"

# 3. Growth inside the rules trips the aggregate ceiling even when each file fits.
repo="$work/budget"
setup_repo "$repo" $((32 * 1024)) 32
write_png "$repo/tests/fixtures/a.png"
write_png "$repo/tests/fixtures/b.png"
git -C "$repo" add -A
run_case "over-total-budget" "$repo" 1 "over the 0 KB ceiling"

# 4. An untracked binary is not the checker's business; .gitignore is.
repo="$work/untracked"
setup_repo "$repo" $((32 * 1024)) $((256 * 1024))
write_png "$repo/tests/fixtures/declared.png"
git -C "$repo" add -A
mkdir -p "$repo/traces"
write_png "$repo/traces/capture.png"
run_case "untracked-ignored" "$repo" 0 "binary-assets: OK"

# 5. Text files stay out of scope no matter how large.
repo="$work/text"
setup_repo "$repo" $((32 * 1024)) $((256 * 1024))
python3 -c "open('$repo/tests/fixtures/big.txt','w').write('x' * 400000)"
git -C "$repo" add -A
run_case "large-text-ignored" "$repo" 0 "binary-assets: OK"

# 6. The real tree must pass, or the rules above do not describe it.
run_case "repository-as-committed" "$(cd "$script_dir/../.." && pwd)" 0 "binary-assets: OK"

if [[ "$failures" -ne 0 ]]; then
  echo "binary-assets self-test: $failures case(s) failed"
  exit 1
fi
echo "binary-assets self-test: all cases passed"
