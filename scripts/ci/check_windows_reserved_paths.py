#!/usr/bin/env python3
"""Reject tracked paths that Windows cannot create.

Rationale
---------
Windows reserves a set of legacy device names. A path component whose stem (the
part before its first ``.``) matches one of them is unusable regardless of case
or extension: ``aux.rs``, ``Con.md`` and ``nul/b.txt`` all fail. Git for Windows
refuses to write such a path because ``core.protectNTFS`` defaults to true, so a
single offender makes the whole repository impossible to check out there.

That is not hypothetical here. ``src/lib/mlxcel-xla/src/aux.rs`` sat in the tree
for months: ``git clone`` aborted mid-checkout and ``git reset --hard`` failed
with ``invalid path``. macOS and Linux accept the name, and every CI job runs on
one of them, so nothing noticed.

The rule
--------
For every tracked path, split on ``/`` and take each component up to its first
``.``. Compare that stem, case-insensitively, against ``CON``, ``PRN``, ``AUX``,
``NUL``, ``COM0``-``COM9``, ``LPT0``-``LPT9``, and ``COM``/``LPT`` followed by
``¹``, ``²`` or ``³``. Names that merely start with a reserved word
(``aux_manifest.rs``, ``console.rs``, ``nullable.rs``) are fine.

Usage
-----
    scripts/ci/check_windows_reserved_paths.py

Exits non-zero and names every offending path.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"{prefix}{suffix}" for prefix in ("com", "lpt") for suffix in "0123456789¹²³"}
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def tracked_files(root: pathlib.Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    return [path for path in out.split("\0") if path]


def reserved_component(path: str) -> str | None:
    """Return the first reserved component of `path`, or None."""
    for component in path.split("/"):
        if component.split(".", 1)[0].casefold() in RESERVED:
            return component
    return None


def main() -> int:
    offenders = []
    for rel in sorted(tracked_files(repo_root())):
        component = reserved_component(rel)
        if component is not None:
            offenders.append((rel, component))

    if offenders:
        print("windows-reserved-paths: FAIL")
        for rel, component in offenders:
            print(f"  reserved: {rel} (component '{component}')")
        print()
        print(
            "Windows reserves CON, PRN, AUX, NUL, COM0-COM9, LPT0-LPT9 and COM/LPT\n"
            "followed by a superscript 1, 2 or 3 as device names, with any extension\n"
            "and in any case. Git for Windows refuses to check out a path containing\n"
            "one, so the repository cannot be cloned there. Rename the file or\n"
            "directory (for example aux.rs to auxiliary.rs) and update every\n"
            "reference to it."
        )
        return 1

    print("windows-reserved-paths: OK: no tracked path uses a Windows device name.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
