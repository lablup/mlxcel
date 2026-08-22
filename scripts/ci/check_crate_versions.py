#!/usr/bin/env python3
"""Assert every version-tracking workspace crate carries the root's version.

Rationale
---------
From v0.3.3 onward the published root ``mlxcel`` and the member crates that ship
with it carry the *same* ``[package] version``. A release therefore has to bump
several manifests at once, and the list is not visible from any single file: the
root manifest names the members, but nothing in it says which of them track the
root version and which are versioned independently.

That gap has already cost a release. ``CLAUDE.md`` and the release skill both
enumerated three crates (``mlxcel``, ``mlxcel-core``, ``mlxcel-surgery``) at a
point when the workspace had five members, and ``mlxcel-mlx-pin`` (added with
issue #1047) had been silently tracking the root version since it was created.
Preparing v0.5.0-beta.1 caught it by hand. A hand-maintained list in prose is
the wrong mechanism for this: it goes stale exactly when a member is added,
which is the moment nobody is thinking about the release.

So the list lives here instead, and the rule is inverted. Every workspace member
must either match the root version or appear in ``INDEPENDENT`` below with a
reason. Adding a sixth member fails this check until someone decides which of
the two it is, which is the decision that kept getting skipped.

As of v0.6.0 ``INDEPENDENT`` is empty: every member tracks the root.

Usage
-----
    python3 scripts/ci/check_crate_versions.py          # check
    python3 scripts/ci/check_crate_versions.py --set X  # rewrite to version X

``--set`` is the release path: it writes the new version into the root and every
tracking member, so ``make prepare-release-version VERSION=x.y.z`` (or a manual
run) cannot bump a subset. It does not touch ``Cargo.lock``; run
``cargo update -p <each>`` afterwards, which the Makefile target does.

Exit status is 0 when every tracking member matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Members deliberately versioned independently of the root. A member listed here
# is skipped by the check; anything not listed must match the root version.
# Keep the reason: it is the whole justification for the exemption.
#
# Empty since v0.6.0. `mlxcel-xla` was the one entry, exempted because a
# default-off backend that nothing shipped could not be part of the release
# contract. That stopped being true: the crate now carries model support users
# select (Molmo2 indexed attention pooling, LLaVA and Qwen2-VL image context
# floors), its worker is on the server's prompt-cache path, and CI compiles its
# feature combinations. A version that stands still while the code moves tells a
# reader the backend is dormant, so it tracks the root and ships in the release
# notes like every other crate.
INDEPENDENT: dict[str, str] = {}

VERSION_RE = re.compile(r'^version = "(?P<version>[^"]+)"$', re.MULTILINE)


def _manifest_paths() -> list[Path]:
    """Return the root manifest followed by each workspace member's manifest."""
    root_manifest = REPO_ROOT / "Cargo.toml"
    with root_manifest.open("rb") as handle:
        root = tomllib.load(handle)

    paths = [root_manifest]
    for member in root["workspace"]["members"]:
        if member == ".":
            continue  # the workspace root *is* the `mlxcel` package
        paths.append(REPO_ROOT / member / "Cargo.toml")
    return paths


def _read(path: Path) -> tuple[str, str]:
    """Return ``(package name, package version)`` for a manifest."""
    with path.open("rb") as handle:
        package = tomllib.load(handle)["package"]
    return package["name"], package["version"]


def _rewrite(path: Path, new_version: str) -> None:
    """Replace the first top-level ``version = "..."`` line in a manifest.

    The manifests keep ``[package]`` first, so the first match is the package
    version. Anchoring on the line start keeps it off the indented `version`
    keys inside dependency tables.
    """
    text = path.read_text()
    updated, count = VERSION_RE.subn(f'version = "{new_version}"', text, count=1)
    if count != 1:
        raise SystemExit(f"{path.relative_to(REPO_ROOT)}: no top-level `version = \"...\"` line found")
    path.write_text(updated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--set",
        metavar="VERSION",
        help="rewrite the root and every tracking member to VERSION instead of checking",
    )
    parser.add_argument(
        "--print-update-args",
        action="store_true",
        help="print the `-p <crate>` arguments for `cargo update`, one per tracking crate",
    )
    args = parser.parse_args()

    manifests = _manifest_paths()
    root_path = manifests[0]
    _, root_version = _read(root_path)

    tracking: list[tuple[Path, str, str]] = []
    skipped: list[str] = []
    for path in manifests[1:]:
        name, version = _read(path)
        if name in INDEPENDENT:
            skipped.append(f"{name} (independent: {version})")
            continue
        tracking.append((path, name, version))

    if args.print_update_args:
        names = ["mlxcel"] + [name for _, name, _ in tracking]
        print(" ".join(f"-p {name}" for name in names))
        return 0

    if args.set:
        target = args.set
        for path in [root_path] + [p for p, _, _ in tracking]:
            _rewrite(path, target)
        names = ["mlxcel"] + [name for _, name, _ in tracking]
        print(f"set version {target} in {len(names)} crates: {', '.join(names)}")
        for note in skipped:
            print(f"  left alone: {note}")
        print("  next: cargo update " + " ".join(f"-p {n}" for n in names))
        return 0

    drifted = [(path, name, version) for path, name, version in tracking if version != root_version]
    if drifted:
        print(f"crate versions disagree with the root `mlxcel` version {root_version}:", file=sys.stderr)
        for path, name, version in drifted:
            print(f"  {name}: {version}  ({path.relative_to(REPO_ROOT)})", file=sys.stderr)
        print(
            "\nEvery workspace member must either carry the root version or be listed in\n"
            "INDEPENDENT in this script with a reason. Fix with:\n"
            f"  python3 scripts/ci/check_crate_versions.py --set {root_version}",
            file=sys.stderr,
        )
        return 1

    names = ", ".join(name for _, name, _ in tracking)
    print(f"crate versions OK: mlxcel {root_version} == {names}")
    for note in skipped:
        print(f"  skipped: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
