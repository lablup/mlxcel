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
    for root_name in TARGET_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and should_process_path(path.relative_to(repo_root)):
                yield path


def main() -> None:
    for path in iter_target_files(REPO_ROOT):
        insert_header_in_file(path)


if __name__ == "__main__":
    main()
