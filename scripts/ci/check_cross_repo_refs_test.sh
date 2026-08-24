#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_cross_repo_refs.py.
#
# The classifier is advisory in CI and depends on diff content plus an optional
# live GitHub boundary lookup. This companion test exercises both code paths in
# a temporary repository with a fake gh binary, so the behavior stays stable
# without network access or a real token.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
under_test="$script_dir/check_cross_repo_refs.py"

work="$(mktemp -d "${TMPDIR:-/tmp}/cross-repo-refs-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

setup_repo() {
  local repo="$1"
  mkdir -p "$repo"
  git init -q "$repo"
  git -C "$repo" config user.name "mlxcel test"
  git -C "$repo" config user.email "mlxcel-test@example.com"
  printf 'base\n' > "$repo/notes.md"
  git -C "$repo" add notes.md
  git -C "$repo" commit -q -m "base"
}

write_fake_gh() {
  local dir="$1" body="$2"
  mkdir -p "$dir"
  printf '%s\n' '#!/usr/bin/env bash' "$body" > "$dir/gh"
  chmod +x "$dir/gh"
}

run_case() {
  local name="$1" repo="$2" status_ref="$3" expected_status="$4" gh_body="$5"
  shift 5
  local fake_bin="$work/$name-bin"
  local stdout="$work/$name.stdout"
  write_fake_gh "$fake_bin" "$gh_body"
  set +e
  (
    cd "$repo"
    env PATH="$fake_bin:$PATH" "$@" python3 "$under_test" "$status_ref"
  ) >"$stdout" 2>&1
  local status=$?
  set -e
  if [ "$status" -ne "$expected_status" ]; then
    echo "FAIL $name: expected exit $expected_status, got $status" >&2
    sed -n '1,120p' "$stdout" >&2
    failures=$((failures + 1))
  else
    echo "ok   $name -> exit $status"
  fi
}

assert_contains() {
  local name="$1" needle="$2"
  local stdout="$work/$name.stdout"
  if ! grep -Fq "$needle" "$stdout"; then
    echo "FAIL $name: missing '$needle'" >&2
    sed -n '1,120p' "$stdout" >&2
    failures=$((failures + 1))
  fi
}

assert_not_contains() {
  local name="$1" needle="$2"
  local stdout="$work/$name.stdout"
  if grep -Fq "$needle" "$stdout"; then
    echo "FAIL $name: unexpectedly found '$needle'" >&2
    sed -n '1,120p' "$stdout" >&2
    failures=$((failures + 1))
  fi
}

repo_same="$work/repo-same"
setup_repo "$repo_same"
printf 'same repo refs #1023 and #1355 stay bare\n' >> "$repo_same/notes.md"
git -C "$repo_same" add notes.md
git -C "$repo_same" commit -q -m "same repo refs"
base_same="$(git -C "$repo_same" rev-parse HEAD~1)"
run_case same_repo "$repo_same" "$base_same" 0 'printf "1387\n"' GH_TOKEN=token
assert_contains same_repo "Verify each is a real lablup/mlxcel #N"
assert_contains same_repo "#1023"
assert_contains same_repo "#1355"
assert_not_contains same_repo "Likely UPSTREAM"

repo_upstream="$work/repo-upstream"
setup_repo "$repo_upstream"
printf 'mlx-lm parity note tracks #1240\n' >> "$repo_upstream/notes.md"
git -C "$repo_upstream" add notes.md
git -C "$repo_upstream" commit -q -m "upstream ref"
base_upstream="$(git -C "$repo_upstream" rev-parse HEAD~1)"
run_case upstream_line "$repo_upstream" "$base_upstream" 0 'printf "1387\n"' GH_TOKEN=token
assert_contains upstream_line "Likely UPSTREAM"
assert_contains upstream_line "#1240"

repo_qualified="$work/repo-qualified"
setup_repo "$repo_qualified"
printf 'qualified ref ml-explore/mlx-lm#1240 stays ignored\n' >> "$repo_qualified/notes.md"
git -C "$repo_qualified" add notes.md
git -C "$repo_qualified" commit -q -m "qualified ref"
base_qualified="$(git -C "$repo_qualified" rev-parse HEAD~1)"
run_case qualified_ref "$repo_qualified" "$base_qualified" 0 'printf "1387\n"' GH_TOKEN=token
assert_contains qualified_ref "OK"
assert_not_contains qualified_ref "#1240"

repo_boundary="$work/repo-boundary"
setup_repo "$repo_boundary"
printf 'future upstream-looking number #1400 needs review\n' >> "$repo_boundary/notes.md"
git -C "$repo_boundary" add notes.md
git -C "$repo_boundary" commit -q -m "boundary ref"
base_boundary="$(git -C "$repo_boundary" rev-parse HEAD~1)"
run_case live_boundary "$repo_boundary" "$base_boundary" 0 'printf "1387\n"' GH_TOKEN=token
assert_contains live_boundary "Likely UPSTREAM"
assert_contains live_boundary "#1400"

repo_offline="$work/repo-offline"
setup_repo "$repo_offline"
printf 'offline fallback leaves #1355 in manual review\n' >> "$repo_offline/notes.md"
git -C "$repo_offline" add notes.md
git -C "$repo_offline" commit -q -m "offline fallback"
base_offline="$(git -C "$repo_offline" rev-parse HEAD~1)"
run_case no_token "$repo_offline" "$base_offline" 0 'printf "1387\n"'
assert_contains no_token "fallback to manual review for non-upstream bare refs (no GH_TOKEN/GITHUB_TOKEN)."
assert_contains no_token "Verify each is a real lablup/mlxcel #N"
assert_not_contains no_token "Likely UPSTREAM"

repo_api_fail="$work/repo-api-fail"
setup_repo "$repo_api_fail"
printf 'API failure fallback leaves #1355 in manual review\n' >> "$repo_api_fail/notes.md"
git -C "$repo_api_fail" add notes.md
git -C "$repo_api_fail" commit -q -m "api failure fallback"
base_api_fail="$(git -C "$repo_api_fail" rev-parse HEAD~1)"
run_case api_failure "$repo_api_fail" "$base_api_fail" 0 'echo "simulated gh failure" >&2; exit 1' GH_TOKEN=token
assert_contains api_failure "fallback to manual review for non-upstream bare refs (simulated gh failure)."
assert_contains api_failure "Verify each is a real lablup/mlxcel #N"

repo_strict="$work/repo-strict"
setup_repo "$repo_strict"
printf 'strict mode still fails for mlx-lm #1240\n' >> "$repo_strict/notes.md"
git -C "$repo_strict" add notes.md
git -C "$repo_strict" commit -q -m "strict mode"
base_strict="$(git -C "$repo_strict" rev-parse HEAD~1)"
run_case strict_mode "$repo_strict" "$base_strict" 1 'printf "1387\n"' GH_TOKEN=token STRICT=1
assert_contains strict_mode "Likely UPSTREAM"

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi

echo "all cases passed"
