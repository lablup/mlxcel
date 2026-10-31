#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
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
"""Fail on a production write of the process environment outside native_env.rs.

On Windows `std::env::set_var` does not reach the C runtime's `getenv`, which is
what MLX reads, so such a write is silently inert there (#2803). Every production
write goes through `mlxcel_core::native_env`, including the ones only Rust reads
today: whether native code reads a name is not visible at the write, and a
reader added later must not need the writer to change.

Test code is exempt: a `#[cfg(test)]` item, or a file declared by a `#[cfg(test)]
mod` along with the files below it. A file counts as test code by that
declaration alone, never by its name.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ALLOWED = {"src/lib/mlxcel-core/src/native_env.rs"}

WRITE = re.compile(
    r"\b(?:set_var|remove_var|_wputenv_s|_putenv_s|_wputenv|_putenv"
    r"|SetEnvironmentVariable[AW]?)\b"
)
CFG_TEST = re.compile(r"#\[cfg\(\s*(?:all\(\s*)?test\b[^\]]*\]")
OTHER_ATTRIBUTE = re.compile(r"\s*#\[[^\]]*\]")
PATH_ATTRIBUTE = re.compile(r'#\[path\s*=\s*"([^"]+)"\s*\]')
MOD_DECLARATION = re.compile(r"\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;")
BACKSLASH = chr(92)


def blank_comments_and_strings(source: str) -> tuple[str, str]:
    """Return (no comments, no comments or literals), keeping every offset."""
    without_comments = list(source)
    code = list(source)
    i = 0
    n = len(source)

    def blank(buffer: list[str], start: int, end: int) -> None:
        for at in range(start, min(end, n)):
            if buffer[at] != "\n":
                buffer[at] = " "

    while i < n:
        c = source[i]
        if source.startswith("//", i):
            end = source.find("\n", i)
            end = n if end < 0 else end
            blank(without_comments, i, end)
            blank(code, i, end)
            i = end
        elif source.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if source.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif source.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(without_comments, i, j)
            blank(code, i, j)
            i = j
        elif c == "r" and (raw := re.match(r'r(#*)"', source[i:i + 64])):
            closer = '"' + raw.group(1)
            end = source.find(closer, i + raw.end())
            end = n if end < 0 else end + len(closer)
            blank(code, i, end)
            i = end
        elif c == '"':
            j = i + 1
            while j < n and source[j] != '"':
                j += 2 if source[j] == BACKSLASH else 1
            blank(code, i, j + 1)
            i = j + 1
        elif c == "'":
            # A char literal, or the tick of a lifetime, which is left alone.
            if i + 1 < n and source[i + 1] == BACKSLASH:
                end = source.find("'", i + 3)
                end = n if end < 0 else end + 1
            elif i + 2 < n and source[i + 2] == "'":
                end = i + 3
            else:
                i += 1
                continue
            blank(code, i, end)
            i = end
        else:
            i += 1
    return "".join(without_comments), "".join(code)


def item_end(code: str, start: int) -> int:
    """Offset past the item at `start`: its `;`, or its matching `}`."""
    depth = 0
    for at in range(start, len(code)):
        c = code[at]
        if c == ";" and depth == 0:
            return at + 1
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return at + 1
    return len(code)


def mod_file(path: pathlib.Path, name: str, explicit: str | None) -> list[pathlib.Path]:
    """The files `mod name;` in `path` can name, by Rust's lookup rules."""
    if explicit:
        return [path.parent / explicit]
    is_root = path.name in {"lib.rs", "main.rs", "mod.rs"}
    base = path.parent if is_root else path.parent / path.stem
    return [base / f"{name}.rs", base / name / "mod.rs"]


def scan(root: pathlib.Path) -> list[str]:
    sources = sorted((root / "src").rglob("*.rs"))
    cleaned = {}
    test_files: set[pathlib.Path] = set()
    masked: dict[pathlib.Path, list[tuple[int, int]]] = {}
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        without_comments, code = blank_comments_and_strings(text)
        cleaned[path] = code
        spans = []
        for match in CFG_TEST.finditer(code):
            start = match.end()
            explicit = None
            while attribute := OTHER_ATTRIBUTE.match(code, start):
                literal = PATH_ATTRIBUTE.match(
                    without_comments[attribute.start():attribute.end()].strip()
                )
                if literal:
                    explicit = literal.group(1)
                start = attribute.end()
            spans.append((match.start(), item_end(code, start)))
            declaration = MOD_DECLARATION.match(code, start)
            if declaration:
                test_files.update(mod_file(path, declaration.group(1), explicit))
        masked[path] = spans

    # A file declared under test takes the directory of its own children along.
    test_dirs = {
        file.parent / file.stem for file in test_files if file.name != "mod.rs"
    } | {file.parent for file in test_files if file.name == "mod.rs"}

    findings = []
    for path in sources:
        relative = path.relative_to(root).as_posix()
        if relative in ALLOWED or path in test_files:
            continue
        if any(directory in path.parents for directory in test_dirs):
            continue
        code = cleaned[path]
        for match in WRITE.finditer(code):
            if any(start <= match.start() < end for start, end in masked[path]):
                continue
            line = code.count("\n", 0, match.start()) + 1
            findings.append(
                f"{relative}:{line}: direct environment write `{match.group()}`"
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
        help="tree to check (default: this repository)",
    )
    args = parser.parse_args()
    findings = scan(args.root.resolve())
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        print(
            "Use mlxcel_core::native_env::{set_native_var, remove_native_var}: on "
            "Windows std::env does not reach the C runtime's getenv (#2803).",
            file=sys.stderr,
        )
        return 1
    print("OK: production environment writes all go through native_env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
