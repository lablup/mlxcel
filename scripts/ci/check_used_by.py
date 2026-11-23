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
"""Hold the `Used by:` annotations on the shared core helpers to the call sites.

Rationale
---------
``docs/code-guidelines.md`` opens with the shared-function rule: a function
that several models depend on records which of them call it. Nothing checked
it, and the failure that matters is not a missing annotation but a wrong one.
Issue #1110 was filed on the belief that ``create_causal_mask`` carried no
annotation. It carried a roster naming Llama, Mixtral, Cohere, Phi, GLM4 and
StarCoder2, none of which had called it for several releases, while the files
that did call it went unnamed (PR #1121 rewrote it). A check for absence
passes that file. Run against the tree before #1121, this one reports 41 of
its 48 caller files as unaccounted for.

Scope
-----
Every top-level ``pub fn`` in the files of ``SCOPED_FILES``, which are the two
that ``docs/code-guidelines.md`` names first among the shared components to
track. The functions are read from the files, never listed here, so a new
helper is covered the day it is added. Methods, trait items and
``pub(crate)`` functions are out of scope.

What counts as a caller
-----------------------
A file is a caller of ``utils.rs::softcap`` when, outside test code, it

* imports the function by name (``use mlxcel_core::utils::softcap;``, also
  inside a brace group or under ``as``),
* names it by a path that resolves to it (``mlxcel_core::utils::softcap(..)``,
  ``utils::softcap(..)`` after ``use mlxcel_core::utils;``, ``crate::utils::``
  inside the defining crate), or
* names it under a glob import of its module, including as a function value.

Paths are resolved through the file's own ``use`` declarations, its child
``mod`` declarations, ``crate``, ``self`` and ``super``. Matching the bare
identifier would be wrong by a wide margin. On the tree this was written
against, 74 files contain the string ``softcap`` and ``grep '\\bsoftcap('``
lists four, but ``src/lib/mlxcel-xla/src/emitter/model.rs`` has its own
private ``fn softcap``, three other files define an unrelated
``softcap_logits``, and most of the rest only have a ``softcap: f32``
parameter. Exactly one file calls ``utils.rs::softcap``. The crate root also
re-exports the raw FFI functions (``pub use ffi::*``), so
``mlxcel_core::silu`` and ``mlxcel_core::gelu_approx`` are *not* the
``utils.rs`` wrappers of the same name, and a model can import a
``stack_arrays`` from its own helper module. None of those resolve to the
function, so none are counted.

Test code is not a caller: a ``#[cfg(test)]`` item, a file declared by a
``#[cfg(test)] mod`` along with the files below it (by that declaration, never
by its name), and anything outside a crate's ``src`` (integration tests,
benches, examples, build scripts). The defining file itself is not a caller
either: the rule is about other modules.

The rules
---------
1. **Absence.** A function with more than ``ABSENCE_THRESHOLD`` caller files
   needs a ``Used by:`` annotation. The three functions that were above the
   threshold and unannotated when the check landed are grandfathered in
   ``UNANNOTATED_DEBT``; an entry that no longer applies fails the check, so
   the list can only shrink.
2. **Form.** The annotation is a line of the function's doc block that starts
   with ``Used by:``, in a ``///`` doc comment so that it survives into
   rustdoc. It runs to the end of the doc block.
3. **Every caller is accounted for**, once a function has more than
   ``ENUMERATION_THRESHOLD`` caller files. Each caller file must be either

   * named by the annotation: a path (``recurrent_gemma.rs``, or
     ``src/vision/dense`` for everything below a directory), or a family name
     that equals the file's stem, or the stem up to one of its underscores,
     once case and ``_``, ``-``, ``.`` are dropped and a space before a digit
     is closed (``Qwen 3.5`` names ``qwen3_5.rs``, ``MiMoV2`` names
     ``mimo_v2_flash.rs``, ``Gemma`` does not name ``gemma3.rs``). The name of
     the directory a file sits in works the same way, as long as the
     directory is a family and not a container
     (``FAMILY_DIRECTORY_MAX_FILES``). A lower-case dictionary word is never a
     name: a name has an upper-case letter, a digit or an inner separator; or
   * listed by the annotation's regenerating command, which the check re-runs:
     ``grep -rln '\\bNAME(' DIR --include='*.rs'``, optionally piped through
     ``grep -v 'PATTERN'`` stages. This is the form the guidelines prescribe
     once a caller set is too long to enumerate.

4. **A recorded count is true.** When an annotation with a ``grep -rln``
   command states the size of the set as ``N files`` or ``N non-test files``,
   N must be the number of files that command lists at this commit.
5. **"Not used by" is true.** A paragraph of the annotation that starts with
   ``Not used by`` must not name a file that calls the function.

Rules 3 to 5 are the staleness half. Rule 3 is the one that fails on the
pre-#1121 text: its roster accounts for seven of the callers and it has no
regenerating command for the rest.

Staleness strategy, and what was rejected
-----------------------------------------
Issue #1141 listed three ways to detect a stale annotation and asked for the
choice to be recorded.

*Chosen: the machine-checkable trailer (option 1), widened so that a complete
roster also passes, with the negative claims (option 2) on top.* The trailer
is exact: the check runs the annotation's own ``grep`` and compares the
recorded count. But requiring it of every function above the threshold would
have meant rewriting rosters that were correct. When this check was written,
three functions in ``utils.rs`` carried rosters for 14, 17 and 33 caller files
(``prefill_sliding_window_mask``, ``materialized_state_tail``,
``prefill_causal_mask``) and every one of those 64 files was named. What makes
a roster go stale is that nothing objects when a caller is added, so the check
objects: it maps each caller *file* to the names that would account for it,
which is mechanical (normalise the file stem) and needs no table, and fails
when the annotation has none of them. A roster stays legal at any length for
as long as it is complete, and an annotation that stops enumerating must carry
the command instead. Either way a new caller cannot land unrecorded.

*Rejected: the family-name heuristic as specified (option 3)*, which maps
each name in a roster to candidate files and flags a listed family with no
call site. It needs the reverse map, from prose to paths, and the prose does
not say which names are claims. On the tree this was written against, the
``softplus`` annotation's ``Used by:`` paragraph names eight families in order
to say that they call the raw FFI function *instead*, and
``layers.rs::attention`` named Llama3, Qwen3, Gemma2 and Gemma3, which reach
it only through ``cache.attend`` and ``attention_from_ptr``. Both read as
stale rosters to that heuristic, and only the second one was. Telling them
apart needs a reader, so a name that no caller matches is not an error here.
The consequence is stated under Limits.

*Rejected as the only rule: negative claims (option 2).* Every hit is real,
but the pre-#1121 text made no negative claim, so it would have passed. It is
kept as rule 5 because the guidelines give the "not used by" half as much
weight as the list, and it costs one comparison per caller.

*Rejected: a mandatory trailer above the threshold (option 1 as specified).*
It catches #1110 by construction and is simpler, but it fails the three
correct rosters above for their length alone.

Why rule 3 has a threshold
--------------------------
Applied to every annotated function, rule 3 failed 28 annotations on the tree
it landed on. Seven were above ``ENUMERATION_THRESHOLD`` and were fixed in the
same change; each had real callers nobody had recorded
(``array_to_vec_f32`` named none of its 32). The other 21 are small, and they
are a mix that a script cannot sort: real omissions (``pipeline_hint`` had not
been told about Nanochat) next to accurate prose that names a caller by what
it does (``materialized_axis_tail`` says "``RotatingKVCache``'s first fill"
for ``cache/first_fill.rs``). Gating those would have meant rewording correct
annotations to suit the matcher. They are counted in the summary line and
listed by ``--report``, and the threshold is the knob to lower once they are
cleaned up.

Limits
------
* A roster that still names a family which has stopped calling the function
  passes, unless the name sits in a ``Not used by`` paragraph. Only the
  additions are caught mechanically. (``create_causal_mask`` still listed
  Gemma3, whose only remaining calls are in a test module; that was found by
  reading ``--report``, not by a failure.)
* Name matching is deliberately loose in one direction: ``DeepSeek`` accounts
  for every ``deepseek_*.rs`` caller, and a capitalised word that happens to
  equal a file stem accounts for that file. A looser match can hide a missing
  name; it cannot fail a correct annotation.
* A public or visibility-restricted ``use`` of a scoped function fails at
  the re-export, including a glob of the defining module. Downstream callers
  can otherwise use a path the checker cannot resolve. Macros are not
  expanded. Block-local imports are collected per file. Complex binding
  patterns and nested block comments are not parsed.
* The regenerating command is matched textually, as ``grep`` would, so the
  files it lists can include test files and comment mentions. That is what
  the annotation's own count is measured against; the caller set above is
  not.

Usage
-----
    scripts/ci/check_used_by.py [--root DIR] [--report]

``--root`` points the check at another tree; the companion test
``check_used_by_test.sh`` uses it on synthetic trees. ``--report`` prints
every scoped function with its caller files and how each one is accounted
for, which is what to read before changing a threshold or writing an
annotation. Exits non-zero and names every offending function.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import posixpath
import re
import sys

# The files `docs/code-guidelines.md` names as the shared components to track.
SCOPED_FILES = (
    "src/lib/mlxcel-core/src/utils.rs",
    "src/lib/mlxcel-core/src/layers.rs",
)

# More caller files than this, and a function needs an annotation. Measured
# against the tree the check landed on: outside UNANNOTATED_DEBT the largest
# unannotated caller count was 5 (`utils.rs::relu_squared`).
ABSENCE_THRESHOLD = 5

# Functions above ABSENCE_THRESHOLD that had no annotation when the check
# landed, with their caller counts then (71, 25 and 18). Annotate one and
# delete its entry in the same change; an entry that no longer applies fails.
UNANNOTATED_DEBT = frozenset(
    {
        "src/lib/mlxcel-core/src/utils.rs::slice_axis",
        "src/lib/mlxcel-core/src/utils.rs::silu",
        "src/lib/mlxcel-core/src/utils.rs::stack_arrays",
    }
)

# More caller files than this, and the annotation has to account for every one
# of them (rule 3). It is the guidelines' "past roughly a dozen callers", the
# point where a hand-written roster stops being repaired. At or below it an
# unnamed caller is reported by --report and counted in the summary line, but
# does not fail: see "Why rule 3 has a threshold" in the module docstring.
ENUMERATION_THRESHOLD = 12

# A directory's name stands for the files directly in it only while the
# directory is a family. `src/models` (several hundred files) is a container:
# a sentence that starts with "Models" must not account for all of them.
FAMILY_DIRECTORY_MAX_FILES = 64

# How many unaccounted callers a failure lists before it summarises.
LISTED_CALLERS = 12

# Comments, string literals and character literals. Every alternative starts
# with a literal character so the regex engine can skip to candidates instead
# of trying each position, which keeps lexing several hundred files cheap. The
# `r` and `b` prefixes check, after the fact, that they do not end an
# identifier.
LEXEME_RE = re.compile(
    r"/(?:/[^\n]*|\*.*?\*/)"
    r'|"(?:\\.|[^"\\])*"'
    r"|'(?:\\[^\n]+?|[^\\'\n])'"
    r'|r(?<!\wr)(?P<raw>#*)".*?"(?P=raw)'
    r"|b(?<!\wb)(?:\"(?:\\.|[^\"\\])*\"|'(?:\\[^\n]+?|[^\\'\n])'"
    r'|r(?P<braw>#*)".*?"(?P=braw))',
    re.S,
)
CFG_TEST_RE = re.compile(r"#\[cfg\(\s*(?:all\(\s*)?test\b[^\]]*\]")
# `#[cfg(test)]`, any further attributes or comments, then `mod name;`.
TEST_MOD_RE = re.compile(
    r"#\[cfg\(\s*(?:all\(\s*)?test\b[^\]]*\]"
    r"((?:\s+|//[^\n]*|#\[[^\]]*\])*)"
    r"(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;"
)
PATH_ATTRIBUTE_RE = re.compile(r'#\[path\s*=\s*"([^"]+)"\s*\]')
ATTRIBUTE_RE = re.compile(r"\s*#\[[^\]]*\]")
BRACE_RE = re.compile(r"[{};]")
MOD_RE = re.compile(r"\bmod\s+(\w+)\s*[;{]")
USE_RE = re.compile(r"\b(?P<public>pub(?:\s*\([^)]*\))?\s+)?use\s+(?P<tree>[^;]+);")
USE_TOKEN_RE = re.compile(r"::|[{},*]|\w+")
QUALIFIER_RE = re.compile(r"(?:\b\w+\s*::\s*)+\Z")
# Delimiters after a bare function expression, including calls, callbacks,
# assignments and casts. A field label, type qualifier or macro name differs.
VALUE_FOLLOW_RE = re.compile(r"\s*(?:[(),;\]}?.]|as\b)")
FN_HEADER_RE = re.compile(r"\bfn\s+\w+[^;{}]*\{")
FN_KEYWORD_RE = re.compile(r"\bfn\s+\Z")
TOP_LEVEL_PUB_FN_RE = re.compile(
    r"^pub\s+(?:(?:const|unsafe|async)\s+)*fn\s+(\w+)", re.M
)
TOP_LEVEL_FN_RE = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|unsafe|async)\s+)*fn\s+(\w+)", re.M
)
PACKAGE_SECTION_RE = re.compile(r"^\[package\]\s*$(.*?)(?=^\[|\Z)", re.M | re.S)
PACKAGE_NAME_RE = re.compile(r'^name\s*=\s*"([^"]+)"', re.M)

USED_BY_RE = re.compile(r"Used by:")
NOT_USED_BY_RE = re.compile(r"Not used by\b")
COMMAND_RE = re.compile(r"`(grep\s[^`]*)`")
# The one supported shape of a regenerating command.
COMMAND_SHAPE_RE = re.compile(
    r"grep -r(?P<flags>ln|l|n) '(?P<pattern>[^']+)' (?P<dir>[\w./-]+)"
    r" --include='\*\.rs'(?P<filters>(?: \| grep -v '[^']+')*)\Z"
)
FILTER_RE = re.compile(r"\| grep -v '([^']+)'")
RECORDED_COUNT_RE = re.compile(r"\b(\d+)\s+(?:non-test\s+)?files\b")
PATH_TOKEN_RE = re.compile(r"[\w.\-]+(?:/[\w.\-]+)+|[\w\-]+\.rs\b")
WORD_RE = re.compile(r"[A-Za-z0-9][\w.\-]*")
NAME_LIKE_RE = re.compile(r"[A-Z0-9]|[a-z][_\-][a-z]")


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def blank_lexemes(source: str) -> str:
    """Blank comments and literals, keeping every offset and newline."""

    def blank(match: re.Match[str]) -> str:
        text = match.group(0)
        if "\n" not in text:
            return " " * len(text)
        return re.sub(r"[^\n]", " ", text)

    return LEXEME_RE.sub(blank, source)


def item_end(code: str, start: int) -> int:
    """Offset past the item at `start`: its `;`, or its matching `}`."""
    depth = 0
    for match in BRACE_RE.finditer(code, start):
        token = match.group()
        if token == ";":
            if depth == 0:
                return match.end()
        elif token == "{":
            depth += 1
        else:
            depth -= 1
            if depth <= 0:
                return match.end()
    return len(code)


@dataclasses.dataclass(frozen=True)
class Use:
    path: tuple[str, ...]
    local: str
    at: int
    public: bool


def expand_use(tree: str) -> list[tuple[tuple[str, ...], str]]:
    """Flatten a `use` tree into (path, local name); the name is `*` for a glob."""
    tokens = USE_TOKEN_RE.findall(tree)
    flat: list[tuple[tuple[str, ...], str]] = []

    def parse(at: int, prefix: tuple[str, ...]) -> int:
        path = list(prefix)
        while at < len(tokens):
            token = tokens[at]
            if token == "::":
                at += 1
            elif token == "{":
                at += 1
                while at < len(tokens) and tokens[at] != "}":
                    at = at + 1 if tokens[at] == "," else parse(at, tuple(path))
                return at + 1
            elif token == "*":
                flat.append((tuple(path), "*"))
                return at + 1
            elif token in ",}":
                break
            elif token == "as":
                if at + 1 < len(tokens):
                    flat.append((tuple(path), tokens[at + 1]))
                return at + 2
            elif token == "self" and path:
                at += 1  # `{self}` names the prefix itself.
            else:
                path.append(token)
                at += 1
        if path:
            flat.append((tuple(path), path[-1]))
        return at

    parse(0, ())
    return flat


@dataclasses.dataclass
class RustFile:
    path: pathlib.Path
    rel: str
    crate: str | None  # None: not under a crate's `src`, so not production code.
    module: tuple[str, ...]
    raw: str
    is_test: bool = False
    code: str | None = None
    test_spans: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    # Flattened imports keep their own location for caller and re-export diagnostics.
    imports: list[Use] = dataclasses.field(default_factory=list)
    use_spans: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    bindings: dict[str, set[tuple[str, ...]]] = dataclasses.field(default_factory=dict)
    globs: list[tuple[str, ...]] = dataclasses.field(default_factory=list)
    child_mods: set[str] = dataclasses.field(default_factory=set)

    def lex(self) -> None:
        """Blank the file and read its test spans, `use` trees and child mods."""
        if self.code is not None:
            return
        code = self.code = blank_lexemes(self.raw)
        for match in CFG_TEST_RE.finditer(code):
            start = match.end()
            while attribute := ATTRIBUTE_RE.match(code, start):
                start = attribute.end()
            self.test_spans.append((match.start(), item_end(code, start)))
        for match in USE_RE.finditer(code):
            if self.in_test(match.start()):
                continue
            self.use_spans.append((match.start(), match.end()))
            for path, local in expand_use(match["tree"]):
                self.imports.append(Use(path, local, match.start(), bool(match["public"])))
                if local == "*":
                    self.globs.append(path)
                else:
                    self.bindings.setdefault(local, set()).add(path)
        for match in MOD_RE.finditer(code):
            if not self.in_test(match.start()):
                self.child_mods.add(match.group(1))

    def in_test(self, at: int) -> bool:
        return any(start <= at < end for start, end in self.test_spans)

    def in_use(self, at: int) -> bool:
        return any(start <= at < end for start, end in self.use_spans)

    def absolute(self, path: tuple[str, ...], depth: int = 0) -> set[tuple[str, ...]]:
        """The crate-rooted paths `path` can mean in this file."""
        if not path:
            return set()
        head = path[0]
        crate = self.crate or ""
        if head == "crate":
            return {(crate,) + path[1:]}
        if head in ("self", "super"):
            module = list(self.module)
            at = 0
            while at < len(path) and path[at] in ("self", "super"):
                if path[at] == "super" and module:
                    module.pop()
                at += 1
            return {(crate,) + tuple(module) + path[at:]}
        if depth < 4 and head in self.bindings:
            resolved: set[tuple[str, ...]] = set()
            for bound in self.bindings[head]:
                if bound != (head,):
                    resolved |= self.absolute(bound + path[1:], depth + 1)
            if resolved:
                return resolved
        if head in self.child_mods:
            return {(crate,) + self.module + path}
        return {path}  # An external crate, or something this file does not bind.


@dataclasses.dataclass
class Tree:
    root: pathlib.Path
    files: dict[str, RustFile]
    directory_sizes: dict[str, int]


def load_tree(root: pathlib.Path) -> Tree:
    """Every `*.rs` under `root/src`, with its crate, module path and test status."""
    crates: dict[pathlib.Path, str] = {}

    def note_crate(manifest: pathlib.Path) -> None:
        if manifest.is_file():
            section = PACKAGE_SECTION_RE.search(manifest.read_text(errors="replace"))
            name = PACKAGE_NAME_RE.search(section.group(1)) if section else None
            if name:
                crates[manifest.parent] = name.group(1).replace("-", "_")

    note_crate(root / "Cargo.toml")
    paths = []
    for dirpath, dirnames, filenames in os.walk(root / "src"):
        dirnames.sort()
        directory = pathlib.Path(dirpath)
        if "Cargo.toml" in filenames:
            note_crate(directory / "Cargo.toml")
        paths.extend(directory / name for name in sorted(filenames) if name.endswith(".rs"))

    files: dict[str, RustFile] = {}
    directory_sizes: dict[str, int] = {}
    for path in paths:
        crate, module = None, ()
        for parent in path.parents:
            if parent in crates:
                try:
                    parts = list(path.relative_to(parent / "src").with_suffix("").parts)
                except ValueError:
                    break  # tests/, benches/, examples/, build.rs
                if parts and parts[-1] in ("lib", "main", "mod"):
                    parts.pop()
                crate, module = crates[parent], tuple(parts)
                break
        rel = path.relative_to(root).as_posix()
        files[rel] = RustFile(
            path, rel, crate, module, path.read_text(encoding="utf-8", errors="replace")
        )
        parent_rel = rel.rsplit("/", 1)[0]
        directory_sizes[parent_rel] = directory_sizes.get(parent_rel, 0) + 1

    # Test files: declared by a `#[cfg(test)] mod`, never recognised by name.
    # Paths are compared as root-relative strings; pathlib is too slow here.
    test_files: set[str] = set()
    for rel, file in files.items():
        if "cfg(" not in file.raw:
            continue
        directory, _, filename = rel.rpartition("/")
        for match in TEST_MOD_RE.finditer(file.raw):
            explicit = PATH_ATTRIBUTE_RE.search(match.group(1))
            name = match.group(2)
            if explicit:
                test_files.add(posixpath.normpath(f"{directory}/{explicit.group(1)}"))
                continue
            is_root = filename in {"lib.rs", "main.rs", "mod.rs"}
            base = directory if is_root else f"{directory}/{filename[:-3]}"
            test_files.update({f"{base}/{name}.rs", f"{base}/{name}/mod.rs"})
    # A file declared under test takes the directory of its own children along.
    test_dirs = {
        t.rpartition("/")[0] if t.endswith("/mod.rs") else t[:-3] for t in test_files
    }
    for rel, file in files.items():
        parts = rel.split("/")
        file.is_test = (
            file.crate is None
            or rel in test_files
            or any("/".join(parts[:end]) in test_dirs for end in range(1, len(parts)))
        )
    return Tree(root, files, directory_sizes)


@dataclasses.dataclass
class Function:
    file: str
    name: str
    line: int
    # (kind, text) per comment line of the doc block: kind is "doc" or "plain".
    doc: list[tuple[str, str]]
    callers: dict[str, int] = dataclasses.field(default_factory=dict)  # rel -> line

    @property
    def key(self) -> str:
        return f"{self.file}::{self.name}"

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line}: `{self.name}`"

    def annotation_start(self) -> int | None:
        for at, (_, text) in enumerate(self.doc):
            if USED_BY_RE.match(text):
                return at
        return None


def doc_block(lines: list[str], fn_line: int) -> list[tuple[str, str]]:
    """The comment lines directly above a function, attributes skipped over."""
    block = []
    at = fn_line - 2
    while at >= 0:
        stripped = lines[at].strip()
        if stripped.startswith("///") and not stripped.startswith("////"):
            block.append(("doc", stripped[3:].strip()))
        elif stripped.startswith("//"):
            block.append(("plain", stripped.lstrip("/").strip()))
        elif stripped.endswith("]") and not stripped.startswith("#["):
            # The last line of an attribute that spans several: find its `#[`.
            top = at
            while top >= 0 and at - top < 16 and not lines[top].lstrip().startswith("#["):
                top -= 1
            if top < 0 or at - top >= 16:
                break
            at = top
        elif not stripped.startswith("#["):
            break
        at -= 1
    block.reverse()
    return block


def scoped_functions(file: RustFile) -> list[Function]:
    file.lex()
    assert file.code is not None
    lines = file.raw.split("\n")
    functions = []
    for match in TOP_LEVEL_PUB_FN_RE.finditer(file.code):
        if file.in_test(match.start()):
            continue
        line = file.code.count("\n", 0, match.start()) + 1
        functions.append(Function(file.rel, match.group(1), line, doc_block(lines, line)))
    return functions


def identifier_offsets(code: str, name: str):
    """Offsets where `name` stands as a whole identifier and not as `.name`."""
    at = code.find(name)
    while at >= 0:
        end = at + len(name)
        before = code[at - 1] if at else " "
        after = code[end] if end < len(code) else " "
        if not (before.isalnum() or before in "_.") and not (
            after.isalnum() or after == "_"
        ):
            yield at
        at = code.find(name, end)


def local_value_spans(code: str, name: str) -> list[tuple[int, int]]:
    """Simple parameters and let bindings that shadow a glob in their block."""
    spans = []
    parameter = re.compile(rf"\b{name}\s*:(?!:)")
    for header in FN_HEADER_RE.finditer(code):
        if parameter.search(header.group()):
            spans.append((header.start(), item_end(code, header.end() - 1)))
    for binding in re.finditer(rf"\blet\s+(?:mut\s+)?{name}\b", code):
        # The initializer still sees the outer binding. Shadowing starts after
        # the let statement and ends at the nearest enclosing closing brace.
        depth = 0
        start = None
        for token in BRACE_RE.finditer(code, binding.end()):
            if token.group() == "{":
                depth += 1
            elif token.group() == "}":
                if depth == 0:
                    if start is not None:
                        spans.append((start, token.start()))
                    break
                depth -= 1
            elif depth == 0 and start is None:
                start = token.end()
        spans.append(binding.span())
    return spans


def find_callers(tree: Tree, scoped: RustFile, functions: list[Function]) -> list[str]:
    """Fill in `callers` for every function defined in `scoped`."""
    by_name: dict[str, list[Function]] = {}
    for function in functions:
        by_name.setdefault(function.name, []).append(function)
    prefix = (scoped.crate or "",) + scoped.module
    leaf = scoped.module[-1] if scoped.module else ""

    def record(name: str, file: RustFile, at: int) -> None:
        assert file.code is not None
        line = file.code.count("\n", 0, at) + 1
        for function in by_name[name]:
            function.callers.setdefault(file.rel, line)

    failures = []
    for file in tree.files.values():
        if file.is_test:
            continue
        # A reference from another crate has to spell the module's name, in a
        # `use` or as a qualifier; inside the crate a `super::*` glob need not.
        # Plain substring tests first: they rule out most of the tree before
        # anything is lexed.
        if file.crate != scoped.crate and leaf not in file.raw:
            continue
        present = [name for name in by_name if name in file.raw]
        if not present and "*" not in file.raw:
            continue
        file.lex()
        code = file.code
        assert code is not None
        local_fns = set(TOP_LEVEL_FN_RE.findall(code)) if file.globs else set()
        glob_modules = {module for glob in file.globs for module in file.absolute(glob)}
        for use in file.imports:
            targets = file.absolute(use.path)
            # A re-binding can name a previous import: `use self::mask as pcm;`.
            short = use.path[1:] if use.path[:1] == ("self",) else use.path
            if short and short[0] in file.bindings and short != use.path:
                targets |= file.absolute(short)
            elif len(short) == 1 and short[0] not in file.bindings and short[0] not in local_fns:
                targets.update(module + short for module in glob_modules)
            for target in sorted(targets):
                if use.local == "*" and target == prefix and use.public:
                    exported = f"scoped module `{'::'.join(prefix)}` through a glob"
                elif target[:-1] == prefix and target[-1] in by_name:
                    if not use.public:
                        if file.rel != scoped.rel:
                            record(target[-1], file, use.at)
                        continue
                    exported = f"scoped function `{target[-1]}`"
                else:
                    continue
                line = code.count("\n", 0, use.at) + 1
                failures.append(
                    f"{file.rel}:{line}: re-exports {exported}. "
                    "Import the function at the caller so its path can be checked."
                )
        if file.rel == scoped.rel:
            continue
        for name in present:
            target = prefix + (name,)
            glob_visible = (
                prefix in glob_modules and name not in file.bindings and name not in local_fns
            )
            shadowed = local_value_spans(code, name) if glob_visible else []
            for at in identifier_offsets(code, name):
                if file.in_test(at) or file.in_use(at):
                    continue
                before = code[max(0, at - 240) : at]
                if FN_KEYWORD_RE.search(before):
                    continue  # A definition of the same name, not a reference.
                qualifier = QUALIFIER_RE.search(before)
                if qualifier:
                    segments = tuple(re.findall(r"\w+", qualifier.group(0)))
                    if target in file.absolute(segments + (name,)):
                        record(name, file, at)
                    continue
                if glob_visible and VALUE_FOLLOW_RE.match(code, at + len(name)):
                    if not any(start <= at < end for start, end in shadowed):
                        record(name, file, at)
    return failures


def normalise(name: str) -> str:
    return re.sub(r"[_\-.]", "", name).lower()


def family_keys(rel: str, directory_sizes: dict[str, int]) -> set[str]:
    """The normalised names that account for `rel` in a roster."""
    parent, _, filename = rel.rpartition("/")
    stem = filename[:-3]
    labels = [] if stem in ("mod", "lib", "main") else [stem]
    parent_name = parent.rpartition("/")[2]
    is_family = directory_sizes.get(parent, 0) <= FAMILY_DIRECTORY_MAX_FILES
    if parent_name != "src" and (is_family or not labels):
        labels.append(parent_name)
    keys = set()
    for label in labels:
        parts = label.split("_")
        for end in range(1, len(parts) + 1):
            keys.add(normalise("".join(parts[:end])))
    return keys


def exact_keys(rel: str) -> set[str]:
    parent, _, filename = rel.rpartition("/")
    stem = filename[:-3]
    return {normalise(parent.rpartition("/")[2] if stem in ("mod", "lib", "main") else stem)}


def names_in(text: str) -> set[str]:
    """Normalised family names in a stretch of annotation prose."""
    text = re.sub(r"(?<=[A-Za-z0-9])[ \t]+(?=\d)", "", text)  # "Qwen 3.5" -> "Qwen3.5"
    names = set()
    for word in WORD_RE.findall(text):
        word = word.rstrip(".-")
        if NAME_LIKE_RE.search(word):
            names.add(normalise(word))
    return names


def path_mentions(text: str, rel: str) -> bool:
    """True when `text` names `rel`, or a directory above it, as a path."""
    haystack = f"/{rel}/"
    for token in PATH_TOKEN_RE.findall(text):
        if f"/{token.strip('./')}/" in haystack:
            return True
    return False


def bre_to_python(pattern: str) -> str:
    """Translate the subset of GNU basic regular expressions the commands use."""
    out = []
    at = 0
    while at < len(pattern):
        char = pattern[at]
        if char == "\\" and at + 1 < len(pattern):
            escaped = pattern[at + 1]
            if escaped in "bBwWsS":
                out.append("\\" + escaped)
            elif escaped in "(){}|+?":
                out.append(escaped)  # An operator in a basic expression.
            else:
                out.append(re.escape(escaped))
            at += 2
        elif char in "(){}|+?":
            out.append("\\" + char)  # A literal in a basic expression.
            at += 1
        elif char == "[":
            end = pattern.find("]", at + 2)
            end = len(pattern) - 1 if end < 0 else end
            out.append(pattern[at : end + 1])
            at = end + 1
        else:
            out.append(char)
            at += 1
    return "".join(out)


@dataclasses.dataclass
class Command:
    text: str
    listed: set[str]
    lists_files: bool  # `-l`: the output is one path per file.


def run_command(tree: Tree, function: Function, text: str) -> Command | str:
    """Re-run a regenerating command over the tree, or say why it cannot be."""
    shape = COMMAND_SHAPE_RE.match(text)
    if not shape:
        return (
            "is not in the supported form "
            f"`grep -rln '\\b{function.name}(' DIR --include='*.rs'` "
            "(optionally followed by `| grep -v 'PATTERN'` stages)"
        )
    pattern, directory = shape.group("pattern"), shape.group("dir").strip("/")
    if function.name not in pattern:
        return f"does not search for `{function.name}`"
    if directory != "src" and not directory.startswith("src/") or ".." in directory:
        return f"searches `{directory}`, not `src` or a directory below it"
    if not (tree.root / directory).is_dir():
        return f"searches `{directory}`, which does not exist"
    try:
        search = re.compile(bre_to_python(pattern), re.M)
        filters = [
            re.compile(bre_to_python(f), re.M)
            for f in FILTER_RE.findall(shape.group("filters"))
        ]
    except re.error as error:
        return f"has a pattern this check cannot read ({error})"
    lists_files = "l" in shape.group("flags")
    listed = set()
    for rel, file in tree.files.items():
        if not rel.startswith(directory + "/") or function.name not in file.raw:
            continue
        if lists_files:
            outputs = [rel] if search.search(file.raw) else []
        else:
            outputs = [
                f"{rel}:{number}:{line}"
                for number, line in enumerate(file.raw.split("\n"), 1)
                if search.search(line)
            ]
        if any(not any(f.search(output) for f in filters) for output in outputs):
            listed.add(rel)
    return Command(text, listed, lists_files)


def summarise(rels: list[str]) -> str:
    shown = ", ".join(rels[:LISTED_CALLERS])
    more = len(rels) - LISTED_CALLERS
    return shown + (f" and {more} more" if more > 0 else "")


def check_function(
    tree: Tree, function: Function, report: list[str], advisory: list[int]
) -> list[str]:
    """The failures for one function; `advisory` collects ungated unnamed callers."""
    failures = []
    callers = sorted(function.callers)
    start = function.annotation_start()
    in_debt = function.key in UNANNOTATED_DEBT

    if start is None:
        loose = any("Used by" in text for _, text in function.doc)
        if in_debt and len(callers) <= ABSENCE_THRESHOLD:
            failures.append(
                f"{function.where} is listed in UNANNOTATED_DEBT but has "
                f"{len(callers)} caller file(s), which is not above the threshold of "
                f"{ABSENCE_THRESHOLD}. Remove the entry."
            )
        elif len(callers) > ABSENCE_THRESHOLD and not in_debt:
            failures.append(
                f"{function.where} has {len(callers)} caller files outside its own "
                f"module (threshold {ABSENCE_THRESHOLD}) and no `/// Used by:` "
                "annotation"
                + (
                    "; its doc block mentions `Used by` but no line starts with "
                    "`Used by:`"
                    if loose
                    else ""
                )
                + f". Callers: {summarise(callers)}."
            )
        report.append(
            f"  {function.name}:{function.line} callers={len(callers)} "
            f"annotation={'debt' if in_debt else 'none'}"
        )
        report.extend(f"      {rel}:{function.callers[rel]}" for rel in callers)
        return failures

    if in_debt:
        failures.append(
            f"{function.where} is listed in UNANNOTATED_DEBT but now carries a "
            "`Used by:` annotation. Remove the entry."
        )
    if function.doc[start][0] != "doc":
        failures.append(
            f"{function.where} keeps its `Used by:` annotation in a `//` comment. "
            "Use `///` on a public item so the annotation survives into rustdoc."
        )

    # The annotation runs from its `Used by:` line to the end of the doc block.
    paragraphs: list[list[str]] = [[]]
    for _, text in function.doc[start:]:
        if text:
            paragraphs[-1].append(text)
        elif paragraphs[-1]:
            paragraphs.append([])
    joined = [" ".join(paragraph) for paragraph in paragraphs if paragraph]
    positive = " ".join(p for p in joined if not NOT_USED_BY_RE.match(p))
    negative = " ".join(p for p in joined if NOT_USED_BY_RE.match(p))
    whole = " ".join(joined)

    commands = []
    for text in COMMAND_RE.findall(whole):
        result = run_command(tree, function, re.sub(r"\s+", " ", text).strip())
        if isinstance(result, str):
            failures.append(
                f"{function.where}: the regenerating command `{text}` {result}."
            )
        else:
            commands.append(result)

    file_lists = [command for command in commands if command.lists_files]
    recorded = RECORDED_COUNT_RE.search(whole)
    if recorded and file_lists:
        stated, measured = int(recorded.group(1)), len(file_lists[0].listed)
        if stated != measured:
            failures.append(
                f"{function.where}: the annotation states {stated} files, but its "
                f"regenerating command lists {measured} at this commit. Its "
                "extra files may be comment or test mentions. "
                f"Review `{file_lists[0].text}`, then add a `grep -v` stage for files to "
                "exclude or update the count and the groups it summarises."
            )

    positive_names = names_in(positive)
    negative_names = names_in(negative)
    how = {}
    for rel in callers:
        if any(rel in command.listed for command in commands):
            how[rel] = "command"
        elif path_mentions(positive, rel):
            how[rel] = "path"
        elif family_keys(rel, tree.directory_sizes) & positive_names:
            how[rel] = "name"
    unaccounted = [rel for rel in callers if rel not in how]
    gated = len(callers) > ENUMERATION_THRESHOLD
    if unaccounted and gated:
        failures.append(
            f"{function.where}: {len(unaccounted)} of {len(callers)} caller files "
            f"are not accounted for by the `Used by:` annotation: "
            f"{summarise(unaccounted)}."
        )
    elif unaccounted:
        advisory.append(len(unaccounted))
    contradicted = [
        rel
        for rel in callers
        if exact_keys(rel) & negative_names or path_mentions(negative, rel)
    ]
    if contradicted:
        failures.append(
            f"{function.where}: the annotation says `Not used by` about "
            f"{summarise(contradicted)}, which call(s) it."
        )

    form = "command" if commands else "roster"
    report.append(
        f"  {function.name}:{function.line} callers={len(callers)} annotation={form}"
        + (f" unaccounted={len(unaccounted)}" if unaccounted else "")
        + ("" if gated or not unaccounted else " (advisory)")
    )
    for rel in callers:
        report.append(
            f"      {rel}:{function.callers[rel]} {how.get(rel, 'NOT ACCOUNTED FOR')}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=repo_root(),
        help="tree to check (default: this repository)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print every scoped function, its callers and how each is accounted for",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"used-by: FAIL\n  --root {root} is not a directory")
        return 1

    tree = load_tree(root)
    failures: list[str] = []
    report: list[str] = []
    functions: list[Function] = []
    advisory: list[int] = []
    for rel in SCOPED_FILES:
        scoped = tree.files.get(rel)
        if scoped is None or scoped.crate is None:
            failures.append(
                f"{rel}: listed in SCOPED_FILES but "
                + ("does not exist" if scoped is None else "is not under a crate's `src`")
                + ". If it was renamed or split on purpose, update SCOPED_FILES in "
                "the same change."
            )
            continue
        found = scoped_functions(scoped)
        if not found:
            failures.append(
                f"{rel}: no top-level `pub fn`, so the check would pass over nothing."
            )
        failures.extend(find_callers(tree, scoped, found))
        report.append(f"{rel}: {len(found)} top-level pub fn")
        for function in found:
            failures.extend(check_function(tree, function, report, advisory))
        functions.extend(found)

    known = {function.key for function in functions}
    scanned = {function.file for function in functions}
    for key in sorted(UNANNOTATED_DEBT - known):
        if key.partition("::")[0] in scanned:  # A missing file is reported above.
            failures.append(
                f"{key}: listed in UNANNOTATED_DEBT but no such top-level `pub fn` "
                "exists. Remove the entry."
            )

    annotated = sum(1 for f in functions if f.annotation_start() is not None)
    callers = sum(len(f.callers) for f in functions)
    counts = (
        f"{len(functions)} top-level pub fn in {len(SCOPED_FILES)} shared files, "
        f"{annotated} annotated, {callers} caller-file references resolved, "
        f"{len(tree.files)} Rust files scanned"
    )
    if advisory:
        counts += (
            f"; not gated, at or below the enumeration threshold of "
            f"{ENUMERATION_THRESHOLD}: {len(advisory)} annotation(s) with "
            f"{sum(advisory)} unnamed caller file(s), see --report"
        )
    if args.report:
        print("\n".join(report))
        print()
    if failures:
        print("used-by: FAIL")
        print(f"  {counts}.")
        for failure in failures:
            print(f"  {failure}")
        print()
        print(
            "A `Used by:` annotation has to account for every file that calls the\n"
            "function. Name the caller in the roster (its file stem, as `FooBar` for\n"
            "`foo_bar.rs`, or its path), or, when the set is too long to enumerate,\n"
            "close the annotation with the command that regenerates it:\n"
            "  `grep -rln '\\bNAME(' src --include='*.rs' | grep -v 'tests\\.rs$'`\n"
            "Run this script with --report to see each caller and how it is\n"
            "accounted for. See docs/code-guidelines.md and issue #1141."
        )
        return 1

    print(f"used-by: OK, {counts}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
