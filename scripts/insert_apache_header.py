from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_HEADER = """\
// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

"""
PROJECT_COPYRIGHT_LINE = "// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin"
TARGET_ROOTS = ("src", "examples", "tests", "benches")
# Repo-relative files that ship with the crate but sit outside every
# TARGET_ROOTS directory, so the walk above cannot reach them. `build.rs` is
# the crate's own build script (`build = "build.rs"` in Cargo.toml), so it is
# part of the published package and needs the header like any other source.
#
# `spike/` is deliberately absent: those are scratch crates outside the
# workspace `members` list that are never published. Add them here, not by
# guessing, if that ever changes.
TARGET_FILES = ("build.rs",)
ALLOWED_SUFFIXES = {".rs", ".cpp", ".h"}
SKIP_DIR_NAMES = {".git", "target", "__pycache__"}
HEADER_SCAN_LINE_COUNT = 40
REPO_ROOT = Path(__file__).resolve().parents[1]


def should_process_path(path: Path) -> bool:
    """Return True when the file is part of our code surface."""
    if path.suffix not in ALLOWED_SUFFIXES:
        return False
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return False
    if path.as_posix() in TARGET_FILES:
        return True
    return path.parts[0] in TARGET_ROOTS


def leading_header_window(content: str) -> str:
    """Return the leading lines used to detect existing provenance."""
    return "\n".join(content.splitlines()[:HEADER_SCAN_LINE_COUNT])


def has_existing_header(content: str) -> bool:
    """Detect either our project header or another existing provenance header."""
    leading = leading_header_window(content)
    return any(
        needle in leading
        for needle in (
            PROJECT_COPYRIGHT_LINE,
            "Licensed under the Apache License, Version 2.0",
            "SPDX-License-Identifier:",
            "Copyright",
        )
    )


def insert_header(content: str) -> str:
    """Prepend the project header at the very top of the file."""
    return PROJECT_HEADER + content.lstrip("\n")


def insert_header_in_file(path: Path) -> bool:
    """Insert the project header unless the file already has provenance metadata."""
    content = path.read_text(encoding="utf-8")
    if has_existing_header(content):
        print(f"⏭️  Skipped existing header: {path}")
        return False

    path.write_text(insert_header(content), encoding="utf-8")
    print(f"📝 Header inserted: {path}")
    return True


def iter_target_files(repo_root: Path):
    """Yield target Rust/C++ files under the repository roots we own."""
    for file_name in TARGET_FILES:
        path = repo_root / file_name
        if path.is_file() and should_process_path(path.relative_to(repo_root)):
            yield path
    for root_name in TARGET_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and should_process_path(path.relative_to(repo_root)):
                yield path


def files_missing_header(repo_root: Path) -> list[Path]:
    """Return the target files that `main()` would insert a header into.

    Deliberately expressed as "what would the writer touch" rather than as its
    own notion of a valid header: it reuses `has_existing_header`, the same
    predicate `insert_header_in_file` branches on, so `--check` and the default
    insert mode can never disagree about which files are covered. A check that
    re-derived the rule would eventually demand an edit the script itself does
    not make.
    """
    return [
        path
        for path in iter_target_files(repo_root)
        if not has_existing_header(path.read_text(encoding="utf-8"))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Insert the project Apache 2.0 header into owned Rust/C++ sources. "
            "With --check, report files that are missing it and exit non-zero "
            "instead of writing."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files missing the header and exit 1; write nothing",
    )
    args = parser.parse_args(argv)

    if args.check:
        missing = sorted(files_missing_header(REPO_ROOT))
        if missing:
            print("❌ Missing the project license header:")
            for path in missing:
                print(f"   {path.relative_to(REPO_ROOT).as_posix()}")
            print(
                f"\n{len(missing)} file(s) need a header. "
                "Run `python3 scripts/insert_apache_header.py` to insert it."
            )
            return 1
        print("✅ Every target file carries a license header.")
        return 0

    for path in iter_target_files(REPO_ROOT):
        insert_header_in_file(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
