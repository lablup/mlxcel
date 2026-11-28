#!/usr/bin/env python3
"""Keep tracked technical reports inside their minor-version folders.

Rationale
---------
``TECHNICAL_REPORTS/`` held more than 1,400 reports in one flat directory before
they were split into ``<MAJOR>.<MINOR>/`` folders (see
``TECHNICAL_REPORTS/README.md``). The shared ``report`` and ``chain-report``
workflows still default to the directory root, so a report filed without
reading the README lands there silently. This check makes that visible.

The rule
--------
For every tracked path under ``TECHNICAL_REPORTS/``:

* at the root, only ``README.md`` and ``.keep-reports`` are allowed;
* every other file sits exactly one level down, in a folder named
  ``<MAJOR>.<MINOR>`` (``0.9``, ``0.10``);
* that folder is not newer than the ``MAJOR.MINOR`` of the root ``Cargo.toml``
  workspace version, since no release beyond it can contain the PR yet.

Usage
-----
    scripts/ci/check_technical_reports_layout.py
    scripts/ci/check_technical_reports_layout.py --self-test

Exits non-zero and names every offending path.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = "TECHNICAL_REPORTS"
ROOT_FILES = frozenset({"README.md", ".keep-reports"})
FOLDER_RE = re.compile(r"(\d+)\.(\d+)")
VERSION_RE = re.compile(r'^version\s*=\s*"(\d+)\.(\d+)\.', re.M)


def current_line(cargo_toml: str) -> tuple[int, int]:
    match = VERSION_RE.search(cargo_toml)
    if not match:
        raise SystemExit("check_technical_reports_layout: no version in Cargo.toml")
    return int(match.group(1)), int(match.group(2))


def violations(paths: list[str], current: tuple[int, int]) -> list[str]:
    problems = []
    for path in paths:
        parts = path.split("/")
        if parts[0] != REPORTS:
            continue
        rest = parts[1:]
        if len(rest) == 1:
            if rest[0] not in ROOT_FILES:
                problems.append(f"{path}: reports go in a <MAJOR>.<MINOR>/ folder, not the root")
            continue
        folder = FOLDER_RE.fullmatch(rest[0])
        if not folder:
            problems.append(f"{path}: folder {rest[0]!r} is not a <MAJOR>.<MINOR> version")
        elif len(rest) != 2:
            problems.append(f"{path}: reports sit directly in the version folder")
        elif (int(folder.group(1)), int(folder.group(2))) > current:
            problems.append(
                f"{path}: folder {rest[0]} is newer than the Cargo.toml line "
                f"{current[0]}.{current[1]}"
            )
    return problems


def self_test() -> int:
    current = (0, 9)
    good = [
        "TECHNICAL_REPORTS/README.md",
        "TECHNICAL_REPORTS/.keep-reports",
        "TECHNICAL_REPORTS/0.9/3491-yarn-mtp-20261127.en.md",
        "TECHNICAL_REPORTS/0.5/pr-3049-metal-prefill-sdpa-chunking.ko.md",
        "src/main.rs",
    ]
    bad = [
        "TECHNICAL_REPORTS/3491-yarn-mtp-20261127.en.md",
        "TECHNICAL_REPORTS/v0.9/3491-yarn-mtp-20261127.en.md",
        "TECHNICAL_REPORTS/0.9.1/3491-yarn-mtp-20261127.en.md",
        "TECHNICAL_REPORTS/0.9/extra/3491-yarn-mtp-20261127.en.md",
        "TECHNICAL_REPORTS/0.10/3491-yarn-mtp-20261127.en.md",
    ]
    failures = violations(good, current)
    caught = [p for p in bad if violations([p], current)]
    if failures or len(caught) != len(bad):
        print("self-test failed", failures, sorted(set(bad) - set(caught)), file=sys.stderr)
        return 1
    if current_line('[workspace.package]\nversion = "0.10.0-beta.1"\n') != (0, 10):
        print("self-test failed: version parse", file=sys.stderr)
        return 1
    print("check_technical_reports_layout self-test: ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        return self_test()
    tracked = subprocess.run(
        ["git", "ls-files", "--", REPORTS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    current = current_line((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    problems = violations(tracked, current)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} misplaced report path(s); see TECHNICAL_REPORTS/README.md", file=sys.stderr)
        return 1
    print(f"check_technical_reports_layout: {len(tracked)} tracked paths ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
