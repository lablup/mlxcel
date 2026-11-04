#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_benchmark_csv_schema.py.
#
# Each case copies the four CSV producers into a throwaway tree, plants one
# defect, and asserts the checker's exit status and the message the operator
# is supposed to read. The passing baseline runs first, so a case that fails
# for an unrelated reason cannot pass as a successful detection.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
work="$(mktemp -d "${TMPDIR:-/tmp}/benchmark-csv-schema-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT
failures=0

fixture="$work/repo"
mkdir -p "$fixture/scripts/ci"
cp "$script_dir/check_benchmark_csv_schema.py" "$fixture/scripts/ci/"
for name in bench_decode.sh bench_longprompt.sh bench_embeddings.py bench_imagegen.py; do
  cp "$repo_root/scripts/$name" "$fixture/scripts/$name"
done

# Replace one exact snippet, and refuse a no-op so a typo in a case cannot
# turn into a test that passes without planting anything.
mutate() {
  local file="$1" old="$2" new="$3"
  python3 - "$file" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
if old not in text:
    raise SystemExit(f"mutation target not found in {path}: {old!r}")
open(path, "w", encoding="utf-8").write(text.replace(old, new, 1))
PY
}

run_case() {
  local name="$1" expected_status="$2" expected_text="$3"
  local output status=0
  output="$(python3 -I "$fixture/scripts/ci/check_benchmark_csv_schema.py" \
      --root "$fixture" 2>&1)" || status=$?
  if [ "$status" -ne "$expected_status" ] || ! grep -qF "$expected_text" <<<"$output"; then
    echo "FAIL: $name (status $status): $output"
    failures=$((failures + 1))
  else
    echo "ok: $name"
  fi
}

reset_fixture() {
  cp "$repo_root/scripts/bench_decode.sh" "$fixture/scripts/bench_decode.sh"
  cp "$repo_root/scripts/bench_longprompt.sh" "$fixture/scripts/bench_longprompt.sh"
  cp "$repo_root/scripts/bench_embeddings.py" "$fixture/scripts/bench_embeddings.py"
  cp "$repo_root/scripts/bench_imagegen.py" "$fixture/scripts/bench_imagegen.py"
}

checker="$fixture/scripts/ci/check_benchmark_csv_schema.py"

# 1. The unmodified copy passes.
run_case baseline 0 "OK"

# 2. A header without the backend column.
reset_fixture
mutate "$fixture/scripts/bench_decode.sh" ',backend,decode_path"' ',decode_path"'
run_case header-missing-backend 1 "header is"

# 3. The backend column moved after decode_path.
reset_fixture
mutate "$fixture/scripts/bench_decode.sh" ',backend,decode_path"' ',decode_path,backend"'
run_case header-backend-last 1 "header is"

# 4. A row that keeps the old double comma, so the status lands in the backend
#    column and the row is one field short.
reset_fixture
mutate "$fixture/scripts/bench_decode.sh" \
  '${COMMIT_FIELDS},${BACKEND_LABEL},,SKIP:not_a_checkpoint' \
  '${COMMIT_FIELDS},,SKIP:not_a_checkpoint'
run_case row-drops-backend 1 "backend column is"

# 5. A row with one extra empty field.
reset_fixture
mutate "$fixture/scripts/bench_decode.sh" \
  '${COMMIT_FIELDS},${BACKEND_LABEL},,SKIP:missing_weights' \
  '${COMMIT_FIELDS},${BACKEND_LABEL},,,SKIP:missing_weights'
run_case row-extra-field 1 "row has"

# 6. The long-prompt skip row was not brought along.
reset_fixture
mutate "$fixture/scripts/bench_longprompt.sh" \
  '${len},,,${BACKEND_LABEL},,${classification}' \
  '${len},${classification}'
run_case longprompt-skip-row 1 "bench_longprompt.sh"

# 7. One embedding row lost the backend value.
reset_fixture
mutate "$fixture/scripts/bench_embeddings.py" \
  'meta["commit"], meta["mlx_commit"], meta["backend"], note])' \
  'meta["commit"], meta["mlx_commit"], note])'
run_case embeddings-row 1 "omits the backend value"

# 8. The Python writer reverted to CRLF.
reset_fixture
mutate "$fixture/scripts/bench_imagegen.py" \
  'csv.writer(csv_file, lineterminator="\n")' 'csv.writer(csv_file)'
run_case imagegen-crlf 1 "does not pin LF"

# 9. The env override no longer reaches the label.
reset_fixture
mutate "$fixture/scripts/bench_imagegen.py" \
  '    override = os.environ.get("MLXCEL_BENCH_BACKEND")
    if override:
        return override' \
  '    override = None
    if override:
        return override'
run_case override-ignored 1 "did not reach detect_backend"

# 10. The checker itself catches a header that no producer wrote: the failure
#     must name every function, not stop at the first one.
reset_fixture
mutate "$fixture/scripts/bench_decode.sh" ',backend,decode_path"' ',decode_path"'
mutate "$fixture/scripts/bench_longprompt.sh" ',backend,decode_path"' ',decode_path"'
run_case two-producers 1 "bench_longprompt.sh"

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed"
  exit 1
fi
echo "all schema cases passed"
