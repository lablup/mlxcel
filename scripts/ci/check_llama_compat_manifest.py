#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate the checked-in llama-server b10621 compatibility manifest.

Offline structural gate for ``compat/llama-server/b10621/`` (issue #1443,
epic #1431). Runs without network or the b10621 archive; the binary-facing
half of the gate lives in ``tests/llama_compat_manifest.rs`` (option
spellings, env bindings, and defaults against the two server binaries) and
``src/server/llama_compat_tests.rs`` (mounted routes and native request
fields).

Enforced here:

- pin.json counts equal what the shards actually contain, and the shard
  names listed in pin.json's ``shards`` map equal the files on disk.
- Every shard in pin.json's ``shards`` map declares a non-empty, sorted,
  de-duplicated set of owning issue numbers (``shards[name].owners``), and
  every entry's ``issue`` is a member of its own shard's owner set. This is
  what makes shard ownership machine-checked rather than prose in
  docs/llama-server-compat.md: two concurrent chains editing one file is
  now a gate failure, not just a merge conflict.
- Every entry carries exactly the fact keys of its kind plus the seven
  policy keys (``state``, ``issue``, ``test``, ``notes``, ``divergence``,
  ``rationale``, ``mlxcel``), in that order. A misspelled or missing key
  fails loudly instead of being a silent no-op that nothing downstream ever
  reads.
- An entry's ``mlxcel`` claim block only carries keys from a known
  allowlist (``accepted_spellings``, ``accepted_on_one_binary_only``,
  ``env``, ``env_binding``, ``env_test``, ``defaults``, ``hidden``,
  ``route``, ``field``); an unrecognized key fails loudly for the same
  reason. ``divergence`` is deliberately NOT in that set: it is an
  entry-level field, so burying it inside the claim block fails too.
- ``divergence`` is the structured, machine-checked half of what used to
  live only in ``notes`` prose: a list of short strings, each naming one
  externally observable way mlxcel differs from b10621 for that entry. A
  non-empty ``divergence`` makes ``supported`` a hard error. Epic #1431
  defines ``supported`` as "the spelling, value domain, default,
  precedence, and externally observable behavior match", and its remaining
  sub-issues will each flip entries into it; scanning free prose for an
  admitted divergence is too brittle to be that gate.
- The b10621 inventory invariants hold against constants frozen HERE, not
  only against pin.json: 249 help entries, 323 distinct long-option
  spellings, 134 ``LLAMA_*`` environment variables, 53 routes, 74 native
  request fields. pin.json is written by the same extractor run that writes
  the shards, so comparing against it alone cannot catch an extractor
  regression that drops entries and lowers the recorded count to match.
  Duplicate aliases live inside their entry (no spelling appears in two
  entries), so aliases never inflate the feature count.
- Every entry has exactly one policy state out of ``supported`` /
  ``aliased`` / ``not_applicable`` / ``deferred`` / ``by_design``. The
  extractor's ``unclassified`` marker is rejected, which is what turns a
  nightly bump into a merge-blocking, reviewable diff.
- ``deferred`` entries carry an issue number (``--check-issues-open``
  additionally asserts each referenced issue is still open via ``gh``; the
  CI job passes it, the offline ``make verify-llama-compat`` does not).
- ``by_design`` entries are implemented and served, differ from b10621 on
  purpose, permanently, and must argue that: a non-empty ``divergence``
  (an entry with none is ``supported``), non-empty ``notes``, a resolving
  ``test`` pointer, and a well-formed ``rationale`` object whose ``kind``
  distinguishes an architectural impossibility (``revisit_if`` null) from
  a policy choice (``revisit_if`` states what would reopen the decision).
  ``by_design`` deliberately requires no open issue: being closed to
  further work is the state's meaning, so its entries never feed
  ``--check-issues-open``. On every other state ``rationale`` must be
  null.
- ``not_applicable`` entries carry a diagnostic/documentation test id and
  an explanation.
- ``aliased`` entries carry an mlxcel mapping and a test id, and the
  mapping names a DIFFERENT mlxcel identity than the b10621 one. When
  mlxcel answers the b10621 spelling / route / field itself the entry is
  ``supported``; keeping the two mechanically distinguishable is what
  lets ``src/server/llama_compat_tests.rs`` assert that an aliased
  b10621 route or field is NOT served under its own name.
- ``supported`` entries carry a self-consistent mlxcel claim (canonical
  spelling accepted, env binding recorded when b10621 defines one, route
  and field claims matching the entry identity).
- Every referenced test (``test``, and ``mlxcel.env_test`` for a runtime env
  binding) names an existing file INSIDE the repository: a repo-relative path
  with no ``..`` component, resolving under the repository root. An absolute
  or upward-walking value is rejected rather than satisfying the gate with a
  file that is not in the tree.
- Shard and pin.json serialization is canonical (sorted entries, two-space
  indent, trailing newline), so regeneration diffs stay minimal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO / "compat" / "llama-server" / "b10621"

VALID_STATES = {"supported", "aliased", "not_applicable", "deferred", "by_design"}
KIND_ORDER = {"option": 0, "route": 1, "native_request_field": 2}

# Manifest document schema, independent of the pinned llama.cpp release.
# Bumped alongside `scripts/compat/extract_b10621_manifest.py`,
# `tests/llama_compat_manifest.rs`, and `src/server/llama_compat_tests.rs`:
# 2 when pin.json's `shards` field changed from a bare name list to a mapping
# of shard name to its owning-issue set, 3 when every entry gained the
# structured `divergence` list (both issue #1443 follow-ups), 4 when every
# entry gained the `rationale` object and `by_design` joined the state
# vocabulary (#1499).
MANIFEST_SCHEMA_VERSION = 4

# Fact keys the extractor owns, per entry kind, in the order it emits them.
# Mirrors `FACT_KEYS` in `scripts/compat/extract_b10621_manifest.py`.
ENTRY_FACT_KEYS = {
    "option": [
        "kind",
        "id",
        "section",
        "spellings",
        "long_spellings",
        "value_hint",
        "env",
        "default",
        "description",
        "discovery",
    ],
    "route": ["kind", "id", "method", "path", "source", "condition", "discovery"],
    "native_request_field": [
        "kind",
        "id",
        "field",
        "aliases",
        "parent",
        "field_type",
        "description",
        "discovery",
    ],
}

# Policy keys every entry carries, in the order the extractor writes them.
# Mirrors `NEW_POLICY` in `scripts/compat/extract_b10621_manifest.py`; the two
# orders must stay byte-identical, because the extractor's canonical merge is
# what fixes key order on regeneration and a mismatch would make a manifest
# that validates once and then churns on every regeneration.
ENTRY_POLICY_KEYS = [
    "state",
    "issue",
    "test",
    "notes",
    "divergence",
    "rationale",
    "mlxcel",
]

# Keys of a `by_design` entry's `rationale` object, all three required. Like
# `MLXCEL_CLAIM_KEYS`, this is a closed set so a typo'd key fails instead of
# silently recording an argument nothing reads.
RATIONALE_KEYS = {"kind", "reason", "revisit_if"}
RATIONALE_KINDS = {"architectural", "policy"}

# Keys recognized inside an entry's `mlxcel` claim block. A key outside this
# set is very likely a typo (`accepted_spelling` for `accepted_spellings`,
# say), and without this allowlist a typo'd key is a silent no-op: nothing
# reads it, nothing checks it, and a downstream chain can believe it
# recorded a claim that this gate never verifies.
MLXCEL_CLAIM_KEYS = {
    "accepted_spellings",
    "accepted_on_one_binary_only",
    "env",
    "env_binding",
    "env_test",
    "defaults",
    "hidden",
    "route",
    "field",
}

# Frozen-reference invariants for the b10621 pin. These are properties of
# the pinned upstream release, so they can never drift while the pin holds;
# a mismatch means the manifest was corrupted or regenerated against the
# wrong binary.
EXPECTED_HELP_ENTRIES = 249
EXPECTED_LONG_SPELLINGS = 323
EXPECTED_LLAMA_ENVS = 134
# Routes and native request fields are frozen properties of the pin too.
# They need their OWN constants rather than a pin.json comparison alone:
# pin.json is written by the same extractor run that writes the shards, so
# a regression in `parse_routes` / `parse_schema` that drops entries also
# lowers the recorded count and the two agree on the wrong number.
EXPECTED_ROUTES = 53
EXPECTED_NATIVE_FIELDS = 74
EXPECTED_ARCHIVE_SHA256 = (
    "429c8270608600188035e5e92f7d78dffb7900904fe7dd7e6a84f48068cd13cf"
)

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def canonical_dump(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def repo_relative_file(value: object) -> bool:
    """True when ``value`` names an existing file INSIDE the repository.

    A bare ``(REPO / value).is_file()`` does not answer that question. Under
    pathlib join semantics an absolute ``value`` discards ``REPO`` entirely
    (``REPO / "/etc/passwd"`` is ``/etc/passwd``), and a ``..``-laden relative
    value walks out of the tree just as easily, so a manifest entry could
    satisfy the "referenced test exists" gate with a path that is not in the
    repository at all, and could probe the CI runner's filesystem while doing
    it. Require a repo-relative POSIX path with no upward component and
    confirm the resolved path is still under ``REPO``.
    """
    if not isinstance(value, str) or not value:
        return False
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts:
        return False
    # A Windows-style drive or UNC prefix is absolute on Windows but not
    # under PurePosixPath, so reject backslashes outright.
    if "\\" in value:
        return False
    try:
        resolved = (REPO / rel).resolve()
    except OSError:
        return False
    return resolved.is_relative_to(REPO) and resolved.is_file()


def check_entry_keys(where: str, e: dict) -> None:
    """Every entry carries exactly its kind's fact keys plus the policy keys.

    Without this, a misspelled key is a silent no-op: ``divergance`` on a
    ``supported`` entry records nothing, fails nothing, and leaves the author
    believing the divergence is on file. The same hazard applies to ``state``,
    ``notes`` and the rest, so the allowlist covers the whole entry rather than
    just the new field. Order is pinned too, because the extractor emits fact
    keys in ``FACT_KEYS`` order followed by policy keys in ``NEW_POLICY``
    order; requiring the same on disk is what keeps in-place regeneration a
    byte-level no-op.
    """
    expected = ENTRY_FACT_KEYS[e["kind"]] + ENTRY_POLICY_KEYS
    actual = list(e)
    if set(actual) != set(expected):
        unknown = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        parts = []
        if unknown:
            parts.append(f"unknown key(s) {unknown}")
        if missing:
            parts.append(f"missing key(s) {missing}")
        err(f"{where}: {', and '.join(parts)}; an entry carries exactly {expected}")
    elif actual != expected:
        err(
            f"{where}: keys are not in canonical order; expected {expected}, "
            f"found {actual}"
        )


def check_divergence(where: str, state: str, e: dict) -> None:
    """``divergence`` is well-formed, and a non-empty one forbids ``supported``.

    Epic #1431 defines ``supported`` as "the spelling, value domain, default,
    precedence, and externally observable behavior match", and its Definition
    of Done says no option with observable b10621 semantics is accepted and
    silently ignored. Twenty-two entries once claimed ``supported`` while
    diverging observably; nineteen of them said so in their own ``notes``,
    two of those opening with "BEHAVIOR DRIFT". Prose cannot be the gate that
    catches that, so the divergence moves into a field a machine can read.
    """
    divergence = e.get("divergence")
    if not isinstance(divergence, list):
        err(f"{where}: divergence must be a list of short strings, found {divergence!r}")
        return
    for item in divergence:
        if not isinstance(item, str) or not item.strip():
            err(f"{where}: divergence entries must be non-empty strings, found {item!r}")
    if divergence and state == "supported":
        err(
            f"{where}: state 'supported' with a non-empty divergence "
            f"({len(divergence)} recorded). An entry that differs from b10621 "
            "in externally observable behavior is 'aliased' (a tested "
            "translation with no loss of requested semantics), "
            "'not_applicable' (no MLX/CUDA equivalent, rejected with a "
            "diagnostic), 'deferred' (a linked implementation issue owns "
            "closing the gap), or 'by_design' (the difference is permanent, "
            "argued in `rationale` and pinned by a test). Pick one and name "
            "the owning issue."
        )


def check_rationale(where: str, state: str, e: dict) -> None:
    """``rationale`` argues a ``by_design`` permanence claim, and only that.

    On a ``by_design`` entry it is an object with exactly the keys ``kind``,
    ``reason`` and ``revisit_if``. ``kind`` is ``"architectural"`` (mlxcel
    structurally cannot produce b10621's behavior; ``revisit_if`` must be
    null) or ``"policy"`` (mlxcel could produce it and deliberately does not;
    ``revisit_if`` must state, non-empty, what would have to change for the
    decision to be revisited). Keeping the two kinds mechanically
    distinguishable is the point: an architectural impossibility and a policy
    choice must not be recordable interchangeably. On every other state
    ``rationale`` must be null, mirroring the rule that rejects a
    ``divergence`` misfiled inside the ``mlxcel`` block.
    """
    rationale = e.get("rationale")
    if state != "by_design":
        if rationale is not None:
            err(
                f"{where}: state {state!r} with a non-null rationale. "
                "`rationale` argues the permanence of a 'by_design' entry; "
                "on every other state it must be null. If this divergence "
                "really is permanent, the state is 'by_design'."
            )
        return
    if not isinstance(rationale, dict):
        err(
            f"{where}: by_design entry must carry a rationale object "
            f"{{kind, reason, revisit_if}}, found {rationale!r}"
        )
        return
    unknown = sorted(set(rationale) - RATIONALE_KEYS)
    missing = sorted(RATIONALE_KEYS - set(rationale))
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"unknown key(s) {unknown}")
        if missing:
            parts.append(f"missing key(s) {missing}")
        err(
            f"{where}: rationale has {', and '.join(parts)}; it carries "
            f"exactly {sorted(RATIONALE_KEYS)}"
        )
        return
    kind = rationale.get("kind")
    if kind not in RATIONALE_KINDS:
        err(
            f"{where}: rationale.kind {kind!r} is not one of "
            f"{sorted(RATIONALE_KINDS)}"
        )
        return
    reason = rationale.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        err(
            f"{where}: rationale.reason must be a non-empty string naming why "
            "the divergence is permanent, not merely restating it"
        )
    revisit = rationale.get("revisit_if")
    if kind == "policy":
        if not isinstance(revisit, str) or not revisit.strip():
            err(
                f"{where}: rationale.kind 'policy' requires a non-empty "
                "revisit_if stating what would have to change for the "
                "decision to be revisited; a choice with no revisit "
                "condition is indistinguishable from an impossibility, "
                "which is kind 'architectural'"
            )
    elif revisit is not None:
        err(
            f"{where}: rationale.kind 'architectural' requires revisit_if "
            f"null ({revisit!r} found); an impossibility has no revisit "
            "condition. If there is one, the kind is 'policy'."
        )


def check_entry(shard: str, e: dict) -> None:
    where = f"{shard}.json entry {e.get('id')!r}"
    check_entry_keys(where, e)
    state = e.get("state")
    if state not in VALID_STATES:
        err(f"{where}: state {state!r} is not one of {sorted(VALID_STATES)}")
        return
    mlxcel = e.get("mlxcel")
    test = e.get("test")

    check_divergence(where, state, e)
    check_rationale(where, state, e)

    if isinstance(mlxcel, dict):
        unknown = sorted(set(mlxcel) - MLXCEL_CLAIM_KEYS)
        if unknown:
            err(
                f"{where}: mlxcel claim block has unknown key(s) {unknown}; "
                f"allowed keys are {sorted(MLXCEL_CLAIM_KEYS)}. `divergence` "
                "is an entry-level field, not an mlxcel claim key."
            )

    if state == "deferred":
        issue = e.get("issue")
        # `isinstance(True, int)` is True in Python, and a JSON `true` would
        # otherwise reach `gh issue view` as the literal string "True".
        if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
            err(
                f"{where}: deferred entry must link a positive implementation "
                "issue number"
            )
    if state == "not_applicable":
        if not test:
            err(f"{where}: not_applicable entry must name a diagnostic/documentation test")
        if not e.get("notes"):
            err(f"{where}: not_applicable entry must explain why in notes")
    if state == "by_design":
        # The state means: implemented, served, differing from b10621 on
        # purpose, permanently. Each half of that claim has an obligation, so
        # "permanent" is argued per entry with a test behind it rather than
        # asserted to make a checkbox go green.
        divergence = e.get("divergence")
        if not (isinstance(divergence, list) and divergence):
            err(
                f"{where}: by_design entry with an empty divergence. An entry "
                "that does not observably differ from b10621 is 'supported'."
            )
        if not e.get("notes"):
            err(f"{where}: by_design entry must explain the behavior in notes")
        if not test:
            err(
                f"{where}: by_design entry must name the test that pins the "
                "permanent behavior"
            )
    if state == "aliased":
        if not isinstance(mlxcel, dict) or not mlxcel:
            err(f"{where}: aliased entry must record the mlxcel mapping")
        if not test:
            err(f"{where}: aliased entry must name its translation test")
        # An alias maps the b10621 surface onto a DIFFERENT mlxcel identity.
        # When mlxcel answers the b10621 identity itself the entry is
        # `supported`, and the route/field conformance test relies on that
        # split to know whether to assert the b10621 name is served or not.
        if isinstance(mlxcel, dict):
            kind = e.get("kind")
            if kind == "route":
                if not mlxcel.get("route"):
                    err(f"{where}: aliased route must record the mlxcel method/path in mlxcel.route")
                elif mlxcel["route"] == e.get("id"):
                    err(
                        f"{where}: aliased route must map onto a different "
                        "method/path; use state 'supported' when mlxcel serves "
                        "the b10621 route itself"
                    )
            elif kind == "native_request_field":
                if not mlxcel.get("field"):
                    err(f"{where}: aliased field must record the mlxcel field name in mlxcel.field")
                elif mlxcel["field"] == e.get("field"):
                    err(
                        f"{where}: aliased field must map onto a different field "
                        "name; use state 'supported' when mlxcel accepts the "
                        "b10621 field itself"
                    )
            elif kind == "option" and not mlxcel.get("accepted_spellings"):
                err(f"{where}: aliased option must record the mlxcel spellings it accepts instead")
    if state == "supported":
        if not isinstance(mlxcel, dict):
            err(f"{where}: supported entry must carry an mlxcel claim")
            return
        if not test:
            err(f"{where}: supported entry must name its conformance test")
        kind = e.get("kind")
        if kind == "option":
            longs = e.get("long_spellings") or []
            if not longs:
                err(f"{where}: option entry has no long_spellings")
                return
            canonical = longs[0]
            if canonical not in mlxcel.get("accepted_spellings", []):
                err(
                    f"{where}: supported option must accept its canonical "
                    f"spelling {canonical}"
                )
            if e.get("env") and mlxcel.get("env") != e["env"]:
                err(
                    f"{where}: b10621 binds {e['env']} but the supported claim "
                    f"records {mlxcel.get('env')!r}"
                )
        elif kind == "route":
            if mlxcel.get("route") != e.get("id"):
                err(f"{where}: supported route claim must match the entry id")
        elif kind == "native_request_field":
            if mlxcel.get("field") != e.get("field"):
                err(f"{where}: supported field claim must match the field name")

    if isinstance(mlxcel, dict):
        if mlxcel.get("env_binding") == "runtime":
            env_test = mlxcel.get("env_test")
            if not env_test:
                err(f"{where}: runtime env binding must name its covering test")
            elif not repo_relative_file(env_test):
                err(
                    f"{where}: env_test {env_test!r} is not an existing "
                    "repository-relative file"
                )
    if test:
        test_file = str(test).split("::")[0]
        if not repo_relative_file(test_file):
            err(f"{where}: test {test!r} is not an existing repository-relative file")
        else:
            check_test_function_exists(where, str(test), test_file)


def check_test_function_exists(where: str, test: str, test_file: str) -> None:
    """A ``file::name`` test pointer must name a function that is really there.

    The file half was already checked. The function half was not, so an entry
    could point at a test that was renamed, or at one that covers a different
    option entirely, and nothing noticed. That is not hypothetical: several
    #1445 entries initially pointed at a fourteen-case integration test that
    did not cover them, because a pointer nobody validates drifts the moment
    the tests are reorganized.

    Only the last ``::`` segment is treated as a function name, so a nested
    path (``file.rs::mod::name``) still works: the name is searched for
    anywhere in the file as ``fn <name>``.
    """
    if "::" not in test:
        return
    name = test.split("::")[-1]
    if not name:
        err(f"{where}: test {test!r} ends in an empty function name")
        return
    try:
        source = (REPO / test_file).read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - the file existed a line ago
        err(f"{where}: test file {test_file!r} became unreadable: {exc}")
        return
    if f"fn {name}(" not in source:
        err(
            f"{where}: test {test!r} names no `fn {name}` in {test_file}; the "
            "pointer has drifted from the test that covers this entry"
        )


def check_shard_ownership(shard: str, e: dict, owners: set[int]) -> None:
    """An entry's ``issue`` must belong to its shard's declared owner set.

    ``pin.json``'s ``shards`` map is the only thing that makes "which chain
    may touch which file" machine-checked rather than prose in
    docs/llama-server-compat.md. Without this, two concurrent chains editing
    the same shard is caught only by a merge conflict or a human reviewer,
    not by this gate.
    """
    issue = e.get("issue")
    if not isinstance(issue, int) or isinstance(issue, bool) or issue <= 0:
        return
    if issue not in owners:
        where = f"{shard}.json entry {e.get('id')!r}"
        err(
            f"{where}: issue #{issue} is not an owner of shard {shard!r} "
            f"(owners: {sorted(owners)}). Add #{issue} to pin.json "
            f"shards[{shard!r}].owners, or move the entry to a shard that "
            "issue's chain already owns."
        )


def issue_state(issue: int) -> tuple[str | None, str | None]:
    """Return ``(state, None)`` for ``issue``, or ``(None, reason)`` on failure.

    ``issue`` is already known to be an ``int``, and it is passed as a separate
    ``subprocess`` argv element with no shell, so no manifest-controlled string
    ever reaches a command line.
    """
    try:
        proc = subprocess.run(
            ["gh", "issue", "view", str(issue), "--json", "state", "-q", ".state"],
            capture_output=True,
            text=True,
            cwd=REPO,
            check=False,
        )
    except OSError as exc:  # `gh` not installed / not executable
        return None, f"cannot run gh: {exc}"
    if proc.returncode != 0:
        return None, proc.stderr.strip()
    return proc.stdout.strip(), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check-issues-open",
        action="store_true",
        help="additionally verify via `gh` that every issue referenced by a "
        "deferred entry is still open (requires network and GH_TOKEN; the "
        "CI job passes this, local `make verify-llama-compat` does not)",
    )
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=MANIFEST_DIR,
        help="manifest directory to validate (default: the checked-in "
        "compat/llama-server/b10621). Referenced test paths are still resolved "
        "against the repository root, so a copy elsewhere validates the same "
        "way; this exists for the companion negative test "
        "scripts/ci/check_llama_compat_manifest_test.sh",
    )
    args = ap.parse_args()
    manifest_dir: Path = args.manifest_dir

    pin_path = manifest_dir / "pin.json"
    if not pin_path.is_file():
        print(f"error: {pin_path} missing", file=sys.stderr)
        return 1
    pin_raw = pin_path.read_text(encoding="utf-8")
    pin = json.loads(pin_raw)
    if pin.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        err(
            f"pin.json: unsupported schema_version {pin.get('schema_version')!r} "
            f"(expected {MANIFEST_SCHEMA_VERSION})"
        )
    if canonical_dump(pin) != pin_raw:
        err("pin.json is not in canonical JSON serialization")
    if pin["reference"]["archive_sha256"] != EXPECTED_ARCHIVE_SHA256:
        err("pin.json archive_sha256 does not match the pinned official archive")

    shard_files = sorted(
        p for p in manifest_dir.glob("*.json") if p.name != "pin.json"
    )
    shards_field = pin.get("shards")
    if not isinstance(shards_field, dict):
        err("pin.json shards must be an object mapping shard name to its owner-issue set")
        shards_field = {}
    if sorted(shards_field) != [p.stem for p in shard_files]:
        err(
            "pin.json shards keys do not match the shard files on disk: "
            f"{sorted(shards_field)} vs {[p.stem for p in shard_files]}"
        )

    shard_owners: dict[str, set[int]] = {}
    for name, meta in shards_field.items():
        owners = meta.get("owners") if isinstance(meta, dict) else None
        if not isinstance(owners, list) or not owners:
            err(f"pin.json shards[{name!r}] must declare a non-empty owners list")
            owners = []
        cleaned: set[int] = set()
        for o in owners:
            if not isinstance(o, int) or isinstance(o, bool) or o <= 0:
                err(f"pin.json shards[{name!r}].owners contains a non-issue value {o!r}")
                continue
            cleaned.add(o)
        if sorted(cleaned) != owners:
            err(
                f"pin.json shards[{name!r}].owners must be a sorted, de-duplicated "
                "list of positive issue numbers"
            )
        shard_owners[name] = cleaned

    entries_by_id: dict[str, dict] = {}
    deferred_issues: set[int] = set()
    options: list[dict] = []
    routes = 0
    fields = 0
    for shard_path in shard_files:
        raw = shard_path.read_text(encoding="utf-8")
        doc = json.loads(raw)
        if doc.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            err(f"{shard_path.name}: unsupported schema_version")
        if doc.get("area") != shard_path.stem:
            err(f"{shard_path.name}: area field must equal the file stem")
        entries = doc.get("entries", [])
        if not entries:
            err(f"{shard_path.name}: shard has no entries")
        keys = [(KIND_ORDER.get(e.get("kind"), 9), e.get("id", "")) for e in entries]
        if keys != sorted(keys):
            err(f"{shard_path.name}: entries are not in canonical (kind, id) order")
        if canonical_dump(doc) != raw:
            err(f"{shard_path.name}: not in canonical JSON serialization")
        for e in entries:
            eid = e.get("id")
            if eid in entries_by_id:
                err(f"entry {eid!r} appears in more than one shard")
            entries_by_id[eid] = e
            kind = e.get("kind")
            if kind == "option":
                options.append(e)
            elif kind == "route":
                routes += 1
            elif kind == "native_request_field":
                fields += 1
            else:
                err(f"{shard_path.name} entry {eid!r}: unknown kind {kind!r}")
                continue
            check_entry(shard_path.stem, e)
            check_shard_ownership(shard_path.stem, e, shard_owners.get(shard_path.stem, set()))
            issue = e.get("issue")
            if (
                e.get("state") == "deferred"
                and isinstance(issue, int)
                and not isinstance(issue, bool)
                and issue > 0
            ):
                deferred_issues.add(issue)

    # Inventory invariants of the frozen reference.
    all_spellings: list[str] = []
    envs: set[str] = set()
    for o in options:
        all_spellings.extend(o.get("long_spellings", []))
        if o.get("env"):
            envs.add(o["env"])
    if len(options) != EXPECTED_HELP_ENTRIES:
        err(f"expected {EXPECTED_HELP_ENTRIES} help entries, found {len(options)}")
    if len(set(all_spellings)) != EXPECTED_LONG_SPELLINGS:
        err(
            f"expected {EXPECTED_LONG_SPELLINGS} distinct long-option "
            f"spellings, found {len(set(all_spellings))}"
        )
    if len(all_spellings) != len(set(all_spellings)):
        counts_by_spelling = Counter(all_spellings)
        dupes = sorted(s for s, n in counts_by_spelling.items() if n > 1)
        err(f"spellings appear in more than one entry (alias inflation): {dupes}")
    llama_envs = {v for v in envs if v.startswith("LLAMA_")}
    if len(llama_envs) != EXPECTED_LLAMA_ENVS:
        err(f"expected {EXPECTED_LLAMA_ENVS} LLAMA_* env vars, found {len(llama_envs)}")
    if routes != EXPECTED_ROUTES:
        err(f"expected {EXPECTED_ROUTES} b10621 routes, found {routes}")
    if fields != EXPECTED_NATIVE_FIELDS:
        err(
            f"expected {EXPECTED_NATIVE_FIELDS} native request fields, "
            f"found {fields}"
        )

    counts = pin.get("counts", {})
    for key, actual in [
        ("help_entries", len(options)),
        ("long_option_spellings", len(set(all_spellings))),
        ("environment_variables", len(envs)),
        ("llama_environment_variables", len(llama_envs)),
        ("routes", routes),
        ("native_request_fields", fields),
    ]:
        if counts.get(key) != actual:
            err(f"pin.json counts.{key}={counts.get(key)} but shards contain {actual}")

    if args.check_issues_open:
        # One `gh` round trip per issue, but issued concurrently: serially this
        # was ~0.6s x the deferred-issue count on every CI run for the whole
        # epic. Results are collected in sorted issue order so the error output
        # stays deterministic regardless of completion order.
        issues = sorted(deferred_issues)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for issue, result in zip(issues, pool.map(issue_state, issues)):
                state, failure = result
                if failure is not None:
                    err(f"deferred issue #{issue}: gh lookup failed: {failure}")
                elif state != "OPEN":
                    err(
                        f"deferred issue #{issue} is {state}; flip its manifest "
                        "entries or reopen the issue"
                    )

    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"\n{len(errors)} manifest violations", file=sys.stderr)
        return 1

    states: dict[str, int] = {}
    for e in entries_by_id.values():
        states[e["state"]] = states.get(e["state"], 0) + 1
    print(
        f"llama-compat manifest OK: {len(options)} help entries, "
        f"{len(set(all_spellings))} spellings, {len(envs)} env vars, "
        f"{routes} routes, {fields} native fields; states {states}; "
        f"{len(deferred_issues)} distinct deferred issues"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
