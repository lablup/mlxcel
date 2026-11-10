#!/usr/bin/env python3
"""Keep raw records and run captures out of the entire benchmark result tree.

Final reports, measurement CSVs and compact correctness/provenance evidence
belong here; JSONL/NDJSON, server/build logs, stdout/stderr captures and empty
files do not. The historical allowlist and raw-data baseline are empty after
the 2026-11-10 consolidation. Stale entries fail as before. Reusable harnesses
and their inputs belong in scripts/ or tests/fixtures/, not result campaigns.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

DATA_PREFIX = "docs/benchmark_results/"
BUILD_LOG = re.compile(r"(^|[-_.])(fmt|clippy|build|lint|nextest)[-_.a-z0-9]*\.log$")
SERVER_LOG = re.compile(r"^server[^/]*\.log$")
RAW_RECORD = re.compile(r"\.(jsonl|ndjson)$")
RAW_CAPTURE = re.compile(r"\.(log|out|err|stdout|stderr)$")
RAW_RECORD_KIND = "raw record stream (commit the CSV result instead)"


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def read_list(path: pathlib.Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def violation(root: pathlib.Path, rel: str) -> str | None:
    name = rel.rsplit("/", 1)[-1]
    if RAW_RECORD.search(name):
        return RAW_RECORD_KIND
    if SERVER_LOG.search(name):
        return "server log"
    if BUILD_LOG.search(name):
        return "build log"
    if (root / rel).stat().st_size == 0:
        return "empty file"
    if RAW_CAPTURE.search(name):
        return "raw output capture"
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

    allowed = read_list(pathlib.Path(__file__).with_name("benchmark_raw_logs_allowlist.txt"))
    baseline = read_list(pathlib.Path(__file__).with_name("benchmark_raw_data_baseline.txt"))
    excused = {
        rel
        for rel, kind in found.items()
        if (rel in baseline if kind == RAW_RECORD_KIND else rel in allowed)
    }
    excusing_allowlist = {
        rel for rel, kind in found.items() if kind != RAW_RECORD_KIND and rel in allowed
    }
    excusing_baseline = {
        rel for rel, kind in found.items() if kind == RAW_RECORD_KIND and rel in baseline
    }

    failed = bool(allowed or baseline)
    if failed:
        print("::error::legacy benchmark exception lists must stay empty; archive captures outside the result tree")
    for rel in sorted(set(found) - excused):
        failed = True
        print(f"::error file={rel}::{found[rel]} under benchmark data: {rel}")
    for rel in sorted(allowed - excusing_allowlist):
        failed = True
        print(f"::error::stale allowlist entry, remove it: {rel}")
    for rel in sorted(baseline - excusing_baseline):
        failed = True
        print(f"::error::stale baseline entry, remove it: {rel}")
    if failed:
        print(
            "Raw logs and raw record streams belong outside the repository. Keep "
            "the Markdown record and its CSV result instead. Legacy exception "
            "lists must stay empty after consolidation.",
            file=sys.stderr,
        )
        return 1
    print(
        f"benchmark-raw-logs: OK, {len(allowed)} allowlisted files, "
        f"{len(baseline)} raw record files predating the CSV-only rule."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
