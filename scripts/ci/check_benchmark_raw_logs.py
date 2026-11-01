#!/usr/bin/env python3
"""Keep raw run logs out of docs/benchmark_results/data/.

Rationale
---------
``check_binary_assets.py`` deliberately ignores text, so nothing stopped server
logs, build logs and empty stderr captures from piling up beside the benchmark
reports: 690 such files were pruned at once. A report's tables, summaries and
machine-readable results carry its evidence; the logs were noise in checkouts
and searches.

The rule
--------
Under ``docs/benchmark_results/data/``, a tracked file fails this check when it
is empty, is named ``server*.log``, or is a build, format or lint log, unless it
is listed in ``benchmark_raw_logs_allowlist.txt``. The allowlist holds files a
report or its README cites by name or indexes in a manifest, so adding one is a
deliberate, reviewable act. Test logs have no reliable name, so they are out of
scope. An entry
that no longer matches a violation also fails, so the list can only shrink.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

DATA_PREFIX = "docs/benchmark_results/data/"
BUILD_LOG = re.compile(r"(^|[-_.])(fmt|clippy|build|lint|nextest)[-_.a-z0-9]*\.log$")
SERVER_LOG = re.compile(r"^server[^/]*\.log$")


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def violation(root: pathlib.Path, rel: str) -> str | None:
    name = rel.rsplit("/", 1)[-1]
    if SERVER_LOG.search(name):
        return "server log"
    if BUILD_LOG.search(name):
        return "build log"
    if (root / rel).stat().st_size == 0:
        return "empty file"
    return None


def main() -> int:
    root = repo_root()
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", DATA_PREFIX], cwd=root, text=True
    ).split("\0")
    found = {
        rel: kind
        for rel in tracked
        if rel and (root / rel).is_file()
        for kind in [violation(root, rel)]
        if kind
    }
    if "--list" in sys.argv[1:]:
        print("\n".join(sorted(found)))
        return 0

    allow_path = pathlib.Path(__file__).with_name("benchmark_raw_logs_allowlist.txt")
    allowed = {
        line.strip()
        for line in allow_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    failed = False
    for rel in sorted(set(found) - allowed):
        failed = True
        print(f"::error file={rel}::{found[rel]} under benchmark data: {rel}")
    for rel in sorted(allowed - set(found)):
        failed = True
        print(f"::error::stale allowlist entry, remove it: {rel}")
    if failed:
        print(
            "Raw logs belong outside the repository. Keep the summary or results "
            "file instead, or list a log a report cites by name in "
            "scripts/ci/benchmark_raw_logs_allowlist.txt.",
            file=sys.stderr,
        )
        return 1
    print(f"benchmark-raw-logs: OK, {len(allowed)} allowlisted files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
