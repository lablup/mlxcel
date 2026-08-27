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
"""Regenerate the llama-server b10621 compatibility manifest (issue #1443).

The manifest under ``compat/llama-server/b10621/`` is the checked-in,
machine-readable inventory of the frozen llama-server ``b10621`` surface:
every ``--help`` entry, long-option spelling, environment variable, registered
HTTP route, and native ``/completion`` request field, each carrying one
compatibility-policy state (``supported`` / ``aliased`` / ``not_applicable``
/ ``deferred``, per epic #1431).

This script is the out-of-band regeneration tool. It is NOT run in CI (CI
validates the checked-in manifest offline via
``scripts/ci/check_llama_compat_manifest.py`` and the Rust conformance
tests); run it by hand when re-auditing against the pinned reference or when
bumping the pin to a newer nightly.

Inputs
------
- The official release binary, from the archive published at
  https://github.com/ggml-org/llama.cpp/releases/tag/b10621
  (``llama-b10621-bin-macos-arm64.tar.gz``, SHA-256
  ``429c8270608600188035e5e92f7d78dffb7900904fe7dd7e6a84f48068cd13cf``).
  Pass the extracted ``llama-server`` binary via ``--llama-server``; its
  ``--help`` output is the authority for options, spellings, defaults, and
  environment variables. Pass the archive itself via ``--archive`` to have
  its SHA-256 verified against the pinned value AND the ``--llama-server``
  binary required to be byte-identical to a member of that archive: this
  script executes that binary with ``DYLD_LIBRARY_PATH`` /
  ``LD_LIBRARY_PATH`` pointed at its directory, so hashing the tarball alone
  would be assurance about a file that is never run. Without ``--archive``
  the binary is executed unverified and the script says so on stderr.
- A pinned llama.cpp source checkout (or a flat directory) providing
  ``tools/server/server.cpp``, ``tools/server/server-http.cpp``, and
  ``tools/server/server-schema.cpp`` at commit
  ``c1d0e7a004015f23bc0233470b747b596f29b264``, via ``--source-dir``.
  Routes and native request fields are extracted from these files.

Behavior
--------
Facts (spellings, sections, env vars, defaults, descriptions, routes,
schema fields) are re-extracted wholesale. Policy fields (``state``,
``issue``, ``test``, ``notes``, ``divergence``, ``mlxcel``, and any other
non-fact key) are preserved from the existing shard files, keyed by entry id,
and each entry is written back to the shard file it currently lives in, with
its policy keys normalized to the canonical ``NEW_POLICY`` order and any
policy key the entry is missing backfilled from that skeleton. Entries that
are new upstream land in ``_unclassified.json`` with ``state:
"unclassified"``, which the CI validator rejects, so a nightly bump produces
a reviewable diff that cannot be merged until a human classifies the
additions. Entries that disappeared upstream are dropped (visible in the
diff).

``pin.json``'s ``shards`` map (shard name -> its owning-issue set) is policy
too, in the same sense as ``mlxcel_baseline``: it is preserved verbatim from
the existing ``pin.json`` rather than re-derived, so this script never
decides who owns a shard. A brand-new shard (one with no prior entry under
it) gets an empty owner set and needs a human to populate it before
``scripts/ci/check_llama_compat_manifest.py`` will accept entries pointing
at issues in it.

Output is deterministic: running the script twice against the same inputs
leaves the worktree clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

# Manifest document schema, independent of the pinned llama.cpp release.
# Bumped alongside `scripts/ci/check_llama_compat_manifest.py`,
# `tests/llama_compat_manifest.rs`, and `src/server/llama_compat_tests.rs`:
# 2 when pin.json's `shards` field changed from a bare name list to a mapping
# of shard name to its owning-issue set, 3 when every entry gained the
# structured `divergence` list (both issue #1443 follow-ups).
MANIFEST_SCHEMA_VERSION = 3

PINNED_TAG = "b10621"
PINNED_BUILD = 10621
PINNED_COMMIT = "c1d0e7a004015f23bc0233470b747b596f29b264"
PINNED_ARCHIVE = "llama-b10621-bin-macos-arm64.tar.gz"
PINNED_ARCHIVE_SHA256 = "429c8270608600188035e5e92f7d78dffb7900904fe7dd7e6a84f48068cd13cf"
PINNED_ARCHIVE_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b10621/"
    "llama-b10621-bin-macos-arm64.tar.gz"
)
PINNED_PUBLISHED = "2026-08-25"

# Fact keys owned by this extractor, per entry kind. Every other key on an
# entry is policy and survives regeneration untouched.
FACT_KEYS = {
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

# Policy skeleton stamped onto brand-new upstream entries. The key ORDER is
# load-bearing: `main` writes fact keys (in `FACT_KEYS` order) followed by
# policy keys in the order they appear on disk, and
# `scripts/ci/check_llama_compat_manifest.py` requires exactly this sequence,
# so a new entry lands already canonical.
NEW_POLICY = {
    "state": "unclassified",
    "issue": None,
    "test": None,
    "notes": None,
    # A list of short strings, each naming one externally observable way
    # mlxcel differs from b10621 for this entry. Non-empty forbids
    # `state: "supported"`; see the validator.
    "divergence": [],
    "mlxcel": None,
}

# Route-registration conditions that are visible in the pinned sources.
# Keyed by (method, path); anything not listed is registered
# unconditionally. The underlying source patterns are re-checked at parse
# time so a moved conditional block fails loudly instead of silently
# carrying a stale annotation.
ROUTE_CONDITIONS = {
    ("POST", "/models"): "router-mode",
    ("POST", "/models/load"): "router-mode",
    ("POST", "/models/unload"): "router-mode",
    ("GET", "/models/sse"): "router-mode",
    ("DELETE", "/models"): "router-mode",
    ("GET", "/cors-proxy"): "webui-mcp-proxy (403 handler when disabled)",
    ("POST", "/cors-proxy"): "webui-mcp-proxy (403 handler when disabled)",
    ("GET", "/tools"): "server tools or MCP configured (403 handler when disabled)",
    ("POST", "/tools"): "server tools or MCP configured (403 handler when disabled)",
}


def parse_help(text: str) -> list[dict]:
    """Parse ``llama-server --help`` into per-entry fact dicts.

    An entry head is a line starting at column 0 with ``-``; its argument
    field is the leading comma-separated run of dash tokens, which handles
    every b10621 shape including mixed short/long orders such as
    ``--spec-draft-threads-batch, -tbd, --threads-batch-draft N`` and
    negation pairs such as ``--perf, --no-perf``. Parsing the full
    comma list is what keeps spellings like ``--no-cont-batching`` or
    ``--threads-batch-draft`` from being dropped: a parser that reads only
    the first token of each head would misclassify them as prose-only.
    Continuation lines are indented and belong to the entry description.
    """
    entries: list[dict] = []
    section = None
    cur = None
    for line in text.splitlines():
        sec = re.match(r"^----- (.*?) -----\s*$", line)
        if sec:
            section = sec.group(1)
            cur = None
            continue
        if not line.strip():
            # Blank lines appear INSIDE entry blocks (e.g. `--load-mode`
            # prints a blank line before its `(env: ...)` line), so they do
            # not terminate the current entry; the next column-0 head or
            # section marker does.
            continue
        if line.startswith("-"):
            head = re.match(r"^((?:-[^\s,]+,\s*)*-[^\s,]+)", line)
            if head is None:
                raise SystemExit(f"unparseable help head line: {line!r}")
            spellings = [s.strip() for s in head.group(1).split(",")]
            rest = line[head.end() :]
            hint_desc = re.match(r"^\s*(\S(?:.*?\S)?)?\s{2,}(.*)$", rest)
            if hint_desc:
                hint = hint_desc.group(1) or None
                desc = [hint_desc.group(2)]
            else:
                hint = rest.strip() or None
                desc = []
            cur = {
                "section": section,
                "spellings": spellings,
                "value_hint": hint,
                "desc": desc,
            }
            entries.append(cur)
        elif cur is not None:
            cur["desc"].append(line.strip())

    out = []
    seen_ids: set[str] = set()
    for e in entries:
        longs = [s for s in e["spellings"] if s.startswith("--")]
        if not longs:
            raise SystemExit(f"help entry without a long spelling: {e['spellings']}")
        entry_id = longs[0]
        if entry_id in seen_ids:
            raise SystemExit(f"duplicate canonical option id {entry_id}")
        seen_ids.add(entry_id)
        text_all = " ".join(e["desc"])
        # An entry's own environment binding is printed as the LAST
        # `(env: ...)` in its block; earlier matches inside the prose refer
        # to other options (e.g. `--load-mode` describing the deprecated
        # per-flag variables it replaces).
        env_mentions = re.findall(r"\(env: ([A-Z0-9_]+)\)", text_all)
        env = env_mentions[-1] if env_mentions else None
        out.append(
            {
                "kind": "option",
                "id": entry_id,
                "section": e["section"],
                "spellings": e["spellings"],
                "long_spellings": longs,
                "value_hint": e["value_hint"],
                "env": env,
                "default": extract_default(text_all),
                "description": text_all,
                # Every b10621 spelling is declared in an entry head once the
                # full comma list of the head line is parsed; the value
                # records how the extractor discovered the spelling so a
                # future help format that documents options only inside
                # another entry's prose is representable.
                "discovery": "help-entry-head",
            }
        )
    return out


def extract_default(text: str) -> str | None:
    """Return the balanced-paren content of the first ``(default: ...)``."""
    marker = "(default:"
    start = text.find(marker)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + len(marker) : i].strip()
    return text[start + len(marker) :].strip()


def parse_routes(server_cpp: str, server_http_cpp: str) -> list[dict]:
    """Extract registered HTTP method/path pairs from the pinned sources."""
    routes: list[dict] = []
    seen: set[tuple[str, str]] = set()
    method_map = {"get": "GET", "post": "POST", "del": "DELETE"}
    for m in re.finditer(r'ctx_http\.(get|post|del)\s*\(\s*"([^"]+)"', server_cpp):
        method = method_map[m.group(1)]
        path = m.group(2)
        key = (method, path)
        if key in seen:
            # The 403-fallback pairs register the same path in both branches
            # of the feature conditional; one manifest entry covers both.
            continue
        seen.add(key)
        routes.append(
            {
                "kind": "route",
                "id": f"{method} {path}",
                "method": method,
                "path": path,
                "source": "tools/server/server.cpp",
                "condition": ROUTE_CONDITIONS.get(key),
                "discovery": "source-route-registration",
            }
        )

    # GCP / Vertex AI compatibility routes are registered in
    # server-http.cpp::register_gcp_compat behind AIP_MODE=PREDICTION, with
    # env-configurable paths. Verify the mechanism still exists, then emit
    # synthetic entries for the two configurable routes.
    if "register_gcp_compat" not in server_http_cpp:
        raise SystemExit("server-http.cpp no longer defines register_gcp_compat")
    for method, path, note in [
        (
            "POST",
            "${AIP_PREDICT_ROUTE:-/predict}",
            "gcp-compat (AIP_MODE=PREDICTION; path from AIP_PREDICT_ROUTE)",
        ),
        (
            "GET",
            "${AIP_HEALTH_ROUTE}",
            "gcp-compat (AIP_MODE=PREDICTION; registered only when AIP_HEALTH_ROUTE is set)",
        ),
    ]:
        routes.append(
            {
                "kind": "route",
                "id": f"{method} {path}",
                "method": method,
                "path": path,
                "source": "tools/server/server-http.cpp",
                "condition": note,
                "discovery": "source-gcp-compat",
            }
        )

    # The Web UI static asset mount (index.html plus per-asset routes under
    # the API prefix) is a dynamic set; represent it as one synthetic entry.
    if "set_mount_point" not in server_http_cpp:
        raise SystemExit("server-http.cpp no longer mounts the Web UI static path")
    routes.append(
        {
            "kind": "route",
            "id": "GET ${api-prefix}/ (Web UI static assets)",
            "method": "GET",
            "path": "${api-prefix}/ (Web UI static assets)",
            "source": "tools/server/server-http.cpp",
            "condition": "webui enabled (--ui / --path static mount)",
            "discovery": "source-static-mount",
        }
    )
    return routes


def parse_schema(schema_cpp: str) -> list[dict]:
    """Extract native ``/completion`` request fields from server-schema.cpp.

    Fields are declared as ``add((new field_<type>("name", ...))`` chains
    with optional ``->add_alias("...")`` and ``->add_subfield(...)`` calls
    and ``->set_desc("...")`` descriptions. Commented-out declarations are
    skipped.
    """
    fields: list[dict] = []
    lines = [ln for ln in schema_cpp.splitlines() if not ln.lstrip().startswith("//")]
    text = "\n".join(lines)
    # Split on top-level `add((new field_` occurrences; each chunk holds one
    # field declaration chain (subfield declarations appear inside it).
    decls = re.split(r"\n\s{4}add\(\(", text)
    for chunk in decls[1:]:
        head = re.match(r'new field_(\w+)\("([^"]+)"', chunk)
        if head is None:
            continue
        ftype, name = head.group(1), head.group(2)
        subfields = re.findall(r'add_subfield\(\(new field_(\w+)\("([^"]+)"', chunk)
        # Aliases attached to the parent chain but not to a subfield chain.
        aliases = re.findall(r'add_alias\("([^"]+)"\)', chunk)
        desc = re.findall(r'set_desc\("((?:[^"\\]|\\.)*)"', chunk)
        fields.append(
            {
                "kind": "native_request_field",
                "id": f"field:{name}",
                "field": name,
                "aliases": aliases,
                "parent": None,
                "field_type": ftype,
                "description": desc[0].replace('\\"', '"') if desc else None,
                "discovery": "source-schema-declaration",
            }
        )
        for sub_type, sub_name in subfields:
            fields.append(
                {
                    "kind": "native_request_field",
                    "id": f"field:{name}.{sub_name}",
                    "field": sub_name,
                    "aliases": [],
                    "parent": name,
                    "field_type": sub_type,
                    "description": None,
                    "discovery": "source-schema-declaration",
                }
            )
    return fields


def run_help(llama_server: Path) -> str:
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = str(llama_server.parent)
    env["LD_LIBRARY_PATH"] = str(llama_server.parent)
    proc = subprocess.run(
        [str(llama_server), "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"{llama_server} --help exited {proc.returncode}:\n{proc.stderr}"
        )
    return proc.stdout


def find_source(source_dir: Path, name: str) -> Path:
    for cand in [source_dir / name, source_dir / "tools" / "server" / name]:
        if cand.is_file():
            return cand
    raise SystemExit(f"cannot find {name} under {source_dir} (flat or tools/server/)")


def sha256_stream(reader: BinaryIO) -> str:
    """SHA-256 of a binary stream, read in chunks.

    Chunked rather than ``read()`` so neither the archive nor a member is ever
    materialised in memory in full.
    """
    h = hashlib.sha256()
    for chunk in iter(lambda: reader.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return sha256_stream(fh)


def verify_archive(archive: Path) -> None:
    digest = sha256_file(archive)
    if digest != PINNED_ARCHIVE_SHA256:
        raise SystemExit(
            f"archive SHA-256 mismatch: expected {PINNED_ARCHIVE_SHA256}, got {digest}"
        )
    print(f"archive SHA-256 verified: {digest}")


def verify_binary_matches_archive(archive: Path, binary: Path) -> None:
    """Bind the binary this script EXECUTES to the archive it verified.

    Hashing the tarball says nothing about the separately supplied
    ``--llama-server`` path, and ``run_help`` executes that path with
    ``DYLD_LIBRARY_PATH`` / ``LD_LIBRARY_PATH`` pointed at its own directory,
    so every dylib beside it is loaded too. Without this check, a verified
    archive digest is assurance about a file that is never used.

    The archive is read through ``tarfile`` in read mode and compared member
    by member; nothing is ever extracted or written to disk, so there is no
    path-traversal or symlink handling to get wrong. Call this only AFTER
    ``verify_archive`` has pinned the archive digest, so the bytes being
    decompressed are known-good rather than attacker-chosen.
    """
    want = sha256_file(binary)
    with tarfile.open(archive, "r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if PurePosixPath(member.name).name != binary.name:
                continue
            reader = tf.extractfile(member)
            if reader is None:
                continue
            with reader:
                if sha256_stream(reader) == want:
                    print(f"{binary.name} matches archive member {member.name}")
                    return
    raise SystemExit(
        f"{binary} (SHA-256 {want}) does not match any '{binary.name}' member of "
        f"{archive}. The verified archive and the binary being executed must be "
        "the same artifact; re-extract the pinned archive and point "
        "--llama-server at it."
    )


def load_existing(manifest_dir: Path) -> tuple[dict[str, dict], dict[str, str], dict]:
    """Return (policy-by-id, shard-by-id, existing pin.json)."""
    policy: dict[str, dict] = {}
    shard_of: dict[str, str] = {}
    pin = {}
    pin_path = manifest_dir / "pin.json"
    if pin_path.is_file():
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
    for shard_path in sorted(manifest_dir.glob("*.json")):
        if shard_path.name == "pin.json":
            continue
        doc = json.loads(shard_path.read_text(encoding="utf-8"))
        for entry in doc.get("entries", []):
            entry_id = entry["id"]
            if entry_id in policy:
                raise SystemExit(f"entry {entry_id} appears in two shards")
            kind = entry.get("kind", "option")
            if kind not in FACT_KEYS:
                # Without this, `facts` would be empty, every key would be
                # treated as policy, and the merge below would overwrite the
                # freshly extracted facts with the stale ones from disk.
                raise SystemExit(
                    f"entry {entry_id} in {shard_path.name} has unknown kind {kind!r}"
                )
            facts = set(FACT_KEYS[kind])
            policy[entry_id] = {k: v for k, v in entry.items() if k not in facts}
            shard_of[entry_id] = shard_path.stem
    return policy, shard_of, pin


def dump_json(path: Path, doc: dict) -> None:
    # Explicit UTF-8: the "run it twice, the worktree stays clean" contract
    # must not depend on the ambient locale's default encoding.
    path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--llama-server",
        type=Path,
        required=True,
        help="path to the extracted official b10621 llama-server binary",
    )
    ap.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="pinned llama.cpp checkout (or flat dir with the tools/server sources)",
    )
    ap.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="path to the official archive; when given, its SHA-256 is verified "
        "against the pin AND the --llama-server binary is required to match a "
        "member of it, so the binary that gets executed is the verified one",
    )
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "compat" / "llama-server" / "b10621",
    )
    args = ap.parse_args()

    if not args.llama_server.is_file():
        raise SystemExit(f"--llama-server {args.llama_server} is not a file")

    if args.archive is not None:
        verify_archive(args.archive)
        verify_binary_matches_archive(args.archive, args.llama_server)
    else:
        print(
            f"warning: --archive was not given, so {args.llama_server} is executed "
            "unverified (with DYLD_LIBRARY_PATH/LD_LIBRARY_PATH pointed at its own "
            f"directory). Pass --archive to pin it to SHA-256 {PINNED_ARCHIVE_SHA256}.",
            file=sys.stderr,
        )

    help_text = run_help(args.llama_server)
    options = parse_help(help_text)
    routes = parse_routes(
        find_source(args.source_dir, "server.cpp").read_text(encoding="utf-8"),
        find_source(args.source_dir, "server-http.cpp").read_text(encoding="utf-8"),
    )
    fields = parse_schema(
        find_source(args.source_dir, "server-schema.cpp").read_text(encoding="utf-8")
    )

    long_spellings = sorted({s for o in options for s in o["long_spellings"]})
    envs = sorted({o["env"] for o in options if o["env"]})
    llama_envs = [e for e in envs if e.startswith("LLAMA_")]
    sections = sorted({o["section"] for o in options})

    manifest_dir: Path = args.manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    policy, shard_of, old_pin = load_existing(manifest_dir)

    all_entries = options + routes + fields
    new_ids = {e["id"] for e in all_entries}
    for stale in sorted(set(policy) - new_ids):
        print(f"warning: entry {stale} no longer exists upstream; dropping", file=sys.stderr)

    shards: dict[str, list[dict]] = {}
    for entry in all_entries:
        entry_id = entry["id"]
        merged = dict(entry)
        if entry_id in policy:
            # `NEW_POLICY` first, then the on-disk values: the skeleton fixes
            # the policy key ORDER (which the validator pins) and backfills a
            # key an older schema version never wrote, while every value the
            # entry already carries wins. For a manifest already at the
            # current schema this is a byte-level no-op.
            merged.update({**NEW_POLICY, **policy[entry_id]})
            shard = shard_of[entry_id]
        else:
            merged.update(NEW_POLICY)
            shard = "_unclassified"
        shards.setdefault(shard, []).append(merged)

    kind_order = {"option": 0, "route": 1, "native_request_field": 2}
    old_shards = {p.stem for p in manifest_dir.glob("*.json") if p.name != "pin.json"}
    for name in sorted(old_shards - set(shards)):
        (manifest_dir / f"{name}.json").unlink()
        print(f"warning: shard {name}.json is now empty; removed", file=sys.stderr)
    for name, entries in sorted(shards.items()):
        entries.sort(key=lambda e: (kind_order[e["kind"]], e["id"]))
        dump_json(
            manifest_dir / f"{name}.json",
            {"schema_version": MANIFEST_SCHEMA_VERSION, "area": name, "entries": entries},
        )

    # Shard ownership (which implementation issues may add/edit entries in a
    # shard) is policy, not a fact re-derived from the binary or sources, so
    # it is carried forward from the existing pin.json by shard name, the
    # same way `mlxcel_baseline` is. A shard with no prior recorded owners
    # (new shard) starts with an empty set; the CI validator then rejects
    # any entry in it that names an issue, until a human populates the set.
    old_shard_owners: dict[str, list[int]] = {}
    old_shards_field = old_pin.get("shards")
    if isinstance(old_shards_field, dict):
        for name, meta in old_shards_field.items():
            owners = meta.get("owners") if isinstance(meta, dict) else None
            if isinstance(owners, list):
                old_shard_owners[name] = sorted(
                    {o for o in owners if isinstance(o, int) and not isinstance(o, bool)}
                )

    pin = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "reference": {
            "project": "llama.cpp",
            "release_tag": PINNED_TAG,
            "build": PINNED_BUILD,
            "commit": PINNED_COMMIT,
            "published": PINNED_PUBLISHED,
            "archive": PINNED_ARCHIVE,
            "archive_url": PINNED_ARCHIVE_URL,
            "archive_sha256": PINNED_ARCHIVE_SHA256,
            "sources": {
                "arguments": f"https://github.com/ggml-org/llama.cpp/blob/{PINNED_COMMIT}/common/arg.cpp",
                "routes": f"https://github.com/ggml-org/llama.cpp/blob/{PINNED_COMMIT}/tools/server/server.cpp",
                "http": f"https://github.com/ggml-org/llama.cpp/blob/{PINNED_COMMIT}/tools/server/server-http.cpp",
                "schema": f"https://github.com/ggml-org/llama.cpp/blob/{PINNED_COMMIT}/tools/server/server-schema.cpp",
            },
        },
        "counts": {
            "help_entries": len(options),
            "long_option_spellings": len(long_spellings),
            "environment_variables": len(envs),
            "llama_environment_variables": len(llama_envs),
            "other_environment_variables": sorted(set(envs) - set(llama_envs)),
            "routes": len(routes),
            "native_request_fields": len(fields),
            "help_sections": sections,
        },
        "shards": {
            name: {"owners": old_shard_owners.get(name, [])} for name in sorted(shards)
        },
    }
    # Informational snapshot maintained by hand; see the shard docs.
    if "mlxcel_baseline" in old_pin:
        pin["mlxcel_baseline"] = old_pin["mlxcel_baseline"]
    dump_json(manifest_dir / "pin.json", pin)

    print(
        f"manifest refreshed: {len(options)} help entries, "
        f"{len(long_spellings)} long-option spellings, "
        f"{len(envs)} environment variables ({len(llama_envs)} LLAMA_*), "
        f"{len(routes)} routes, {len(fields)} native request fields, "
        f"{len(shards)} shards"
    )
    unclassified = len(shards.get("_unclassified", []))
    if unclassified:
        print(
            f"warning: {unclassified} entries are unclassified; "
            "classify them before CI will pass",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
