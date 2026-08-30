#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_llama_compat_manifest.py.
#
# The validator's most load-bearing rule is negative: a `supported` entry that
# also records an externally observable divergence from b10621 must be
# rejected. A rule that only ever runs against a manifest satisfying it is
# indistinguishable from no rule at all, so this companion test mutates a
# throwaway copy of compat/llama-server/b10621/ and asserts the validator
# fails, with the guidance text that tells the author which state to pick.
#
# The copy is passed through --manifest-dir; referenced test paths still
# resolve against the real repository root, so the only thing that changes is
# the document under test. No network and no b10621 archive are needed.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
under_test="$script_dir/check_llama_compat_manifest.py"
manifest="$repo_root/compat/llama-server/b10621"

work="$(mktemp -d "${TMPDIR:-/tmp}/llama-compat-manifest-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

# Rewrite one entry in one shard of a fresh copy of the manifest. The mutation
# is applied through json.load/json.dump with the validator's own canonical
# serialization, so a case never fails for the unrelated "not in canonical
# JSON serialization" reason.
make_case() {
  local name="$1" shard="$2" entry_id="$3" mutation="$4"
  local dir="$work/$name"
  cp -R "$manifest" "$dir"
  MLXCEL_CASE_SHARD="$dir/$shard.json" MLXCEL_CASE_ID="$entry_id" \
  MLXCEL_CASE_MUTATION="$mutation" python3 - <<'PY'
import json
import os

path = os.environ["MLXCEL_CASE_SHARD"]
entry_id = os.environ["MLXCEL_CASE_ID"]
mutation = os.environ["MLXCEL_CASE_MUTATION"]

with open(path, encoding="utf-8") as fh:
    doc = json.load(fh)
for entry in doc["entries"]:
    if entry["id"] != entry_id:
        continue
    exec(mutation, {"entry": entry})  # noqa: S102 - fixture mutation, not input
    break
else:
    raise SystemExit(f"fixture entry {entry_id} not found in {path}")
with open(path, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
PY
  echo "$dir"
}

run_case() {
  local name="$1" dir="$2" expected_status="$3"
  local stdout="$work/$name.out"
  set +e
  python3 "$under_test" --manifest-dir "$dir" >"$stdout" 2>&1
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

# Control: an untouched copy validates, so every failure below is caused by
# the mutation and not by copying the manifest somewhere else.
control="$work/control"
cp -R "$manifest" "$control"
run_case control "$control" 0

# The rule this file exists for: `supported` plus a divergence is rejected,
# and the message names the three honest alternatives.
dir="$(make_case supported-with-divergence sampling-and-grammar --min-p \
  'entry["divergence"] = ["mlxcel rounds the value; b10621 does not (#1436)."]')"
if run_case supported-with-divergence "$dir" 1; then
  assert_contains supported-with-divergence "state 'supported' with a non-empty divergence"
  assert_contains supported-with-divergence "'aliased'"
  assert_contains supported-with-divergence "'not_applicable'"
  assert_contains supported-with-divergence "'deferred'"
  assert_contains supported-with-divergence "name the owning issue"
fi

# The positive control for the same rule: a divergence on a non-`supported`
# entry is exactly how the manifest is meant to record a known gap.
dir="$(make_case divergence-on-deferred sampling-and-grammar --min-p \
  'entry["state"] = "deferred"; entry["issue"] = 1436; entry["divergence"] = ["mlxcel rounds the value; b10621 does not (#1436)."]')"
run_case divergence-on-deferred "$dir" 0

# A misspelled field name must fail rather than silently record nothing.
dir="$(make_case misspelled-divergence-key sampling-and-grammar --min-p \
  'entry["divergance"] = ["typo"]')"
if run_case misspelled-divergence-key "$dir" 1; then
  assert_contains misspelled-divergence-key "unknown key(s) ['divergance']"
fi

# `divergence` is an entry-level field; burying it in the mlxcel claim block
# is the other way an author could believe a divergence was recorded.
dir="$(make_case divergence-inside-mlxcel-claim sampling-and-grammar --min-p \
  'entry["mlxcel"]["divergence"] = ["misplaced"]')"
if run_case divergence-inside-mlxcel-claim "$dir" 1; then
  assert_contains divergence-inside-mlxcel-claim "mlxcel claim block has unknown key(s) ['divergence']"
fi

# Shape: the field is a list of non-empty strings, never a bare string.
dir="$(make_case divergence-not-a-list sampling-and-grammar --min-p \
  'entry["divergence"] = "one divergence"')"
if run_case divergence-not-a-list "$dir" 1; then
  assert_contains divergence-not-a-list "divergence must be a list of short strings"
fi

dir="$(make_case divergence-empty-string routes 'POST /rerank' \
  'entry["divergence"] = [""]')"
if run_case divergence-empty-string "$dir" 1; then
  assert_contains divergence-empty-string "divergence entries must be non-empty strings"
fi

# Dropping the field entirely must fail too, so the schema cannot regress by
# omission once a shard is hand-edited.
dir="$(make_case missing-divergence routes 'POST /rerank' \
  'entry.pop("divergence")')"
if run_case missing-divergence "$dir" 1; then
  assert_contains missing-divergence "missing key(s) ['divergence']"
fi

# A `file::name` test pointer must name a function that is really there. The
# file half was always checked; the function half was not, so an entry could
# point at a renamed test, or at one covering a different option, and nothing
# noticed. Several #1445 entries drifted exactly that way before the rule
# existed.
dir="$(make_case drifted-test-pointer routes 'POST /rerank' \
  'entry["test"] = "src/server/llama_compat_tests.rs::a_test_that_does_not_exist"')"
if run_case drifted-test-pointer "$dir" 1; then
  assert_contains drifted-test-pointer "names no \`fn a_test_that_does_not_exist\`"
fi

# The positive control: a pointer naming a real function still passes, so the
# rule cannot degrade into one that rejects every pointer.
dir="$(make_case valid-test-pointer routes 'POST /rerank' \
  'entry["test"] = "src/server/llama_compat_tests.rs::manifest_route_claims_match_the_mounted_router"')"
run_case valid-test-pointer "$dir" 0

# ── by_design (#1499) ───────────────────────────────────────────────────────
# The fifth state is also a way to stop trying, so every obligation that
# forces "permanent" to be argued per entry gets a case proving the validator
# actually rejects its absence.

# A by_design entry with no divergence does not differ from b10621, so it is
# `supported`, and the message must say so.
dir="$(make_case by-design-with-empty-divergence routes 'POST /infill' \
  'entry["divergence"] = []')"
if run_case by-design-with-empty-divergence "$dir" 1; then
  assert_contains by-design-with-empty-divergence "by_design entry with an empty divergence"
  assert_contains by-design-with-empty-divergence "'supported'"
fi

# A policy choice without a revisit condition is indistinguishable from an
# impossibility, which is what kind 'architectural' is for.
dir="$(make_case policy-without-revisit routes 'POST /infill' \
  'entry["rationale"]["revisit_if"] = None')"
if run_case policy-without-revisit "$dir" 1; then
  assert_contains policy-without-revisit "requires a non-empty revisit_if"
fi

# The inverse: an architectural impossibility carrying a revisit condition is
# really a policy choice, and the two must not be recordable interchangeably.
dir="$(make_case architectural-with-revisit embeddings-and-rerank --pooling \
  'entry["rationale"]["revisit_if"] = "someday"')"
if run_case architectural-with-revisit "$dir" 1; then
  assert_contains architectural-with-revisit "requires revisit_if null"
fi

# A typo'd rationale key must fail rather than silently record an argument
# nothing reads, the same hazard the mlxcel claim-key allowlist closes.
dir="$(make_case unknown-rationale-key routes 'POST /infill' \
  'entry["rationale"]["reviist_if"] = "typo"')"
if run_case unknown-rationale-key "$dir" 1; then
  assert_contains unknown-rationale-key "unknown key(s) ['reviist_if']"
fi

# `rationale` argues a by_design permanence claim and must be null on every
# other state, on `supported` and `deferred` alike.
dir="$(make_case rationale-on-supported sampling-and-grammar --min-p \
  'entry["rationale"] = {"kind": "policy", "reason": "r", "revisit_if": "c"}')"
if run_case rationale-on-supported "$dir" 1; then
  assert_contains rationale-on-supported "non-null rationale"
fi

# The mutation sets the state itself rather than borrowing whichever entry
# happens to be `deferred` today. It borrowed one twice and rotted twice: the
# case pointed at `POST /completion` until #1477 made it `by_design`, then at
# `--sleep-idle-seconds` until #1440 did the same, and each time the mutation
# became legal and the case passed while checking nothing. With the epic closed
# there is no `deferred` entry left to borrow, so it cannot depend on one.
dir="$(make_case rationale-on-deferred sampling-and-grammar --min-p \
  'entry["state"] = "deferred"; entry["rationale"] = {"kind": "policy", "reason": "r", "revisit_if": "c"}')"
if run_case rationale-on-deferred "$dir" 1; then
  assert_contains rationale-on-deferred "non-null rationale"
fi

# "Permanent" without a pinning test is an assertion, not an argument.
dir="$(make_case by-design-without-test routes 'POST /infill' \
  'entry["test"] = None')"
if run_case by-design-without-test "$dir" 1; then
  assert_contains by-design-without-test "must name the test that pins"
fi

# The shard-owner check is state-independent: a by_design entry keeping its
# `issue` for provenance still needs that issue in its shard's owner set.
dir="$(make_case by-design-foreign-issue routes 'POST /infill' \
  'entry["issue"] = 999999')"
if run_case by-design-foreign-issue "$dir" 1; then
  assert_contains by-design-foreign-issue "is not an owner of shard"
fi

# The positive control: a well-formed by_design entry constructed from a
# supported one passes, so none of the rules above degrades into one that
# rejects every by_design entry. (The untouched control at the top already
# covers the eleven real by_design entries.)
dir="$(make_case well-formed-by-design sampling-and-grammar --min-p \
  'entry["state"] = "by_design"; entry["divergence"] = ["mlxcel rounds the value; b10621 does not."]; entry["rationale"] = {"kind": "policy", "reason": "the rounding is deliberate", "revisit_if": "upstream changes its own rounding"}')"
run_case well-formed-by-design "$dir" 0

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
echo "all llama-compat manifest validator cases passed"
