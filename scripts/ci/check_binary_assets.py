#!/usr/bin/env python3
"""Require every tracked binary file to be declared, with a size budget.

Rationale
---------
Binary blobs enter a git repository silently. Nothing about ``git add`` warns
that a file will never delta-compress, that every later revision of it is stored
whole, and that removing it later does not shrink the history. The cost only
becomes visible as clone time, long after the commit that caused it.

That is not hypothetical here. ``webui/tests/screenshots/`` accumulated 24 PNG
baselines totalling ~1.9 MB across two platform directories without any review
step noticing, because each individual commit looked reasonable and no gate
looked at the aggregate. They were pixel baselines: renderer-, font- and
platform-specific, regenerated on a hosted runner and manually reviewed on every
font, Chromium or UI change. The browser suite's accessibility and geometry
assertions covered the same ground, so the images were dropped and the directory
is now ignored.

This check makes the next one impossible to add quietly.

The rule
--------
Every tracked file that git would treat as binary must match one ``Rule`` below.
A binary file matching nothing fails the job, and so does one that exceeds its
rule's ``max_bytes`` or pushes the declared set past ``TOTAL_BUDGET_BYTES``.

The list is inverted on purpose, the same way ``check_crate_versions.py``
inverts the crate list: a *new* binary path is the failure mode, so the default
for an undeclared path has to be "fail", not "ignore". An allowlist of paths to
*skip* would have stayed silent on exactly the directory that prompted this.

Adding a binary asset is therefore a deliberate act: add a ``Rule`` with a
reason, and the reason is what review reads. Generated artifacts, profiling
captures and test output belong in ``.gitignore`` instead.

Text bloat is out of scope. Large text files delta-compress and stay diffable,
so they do not carry the same one-way cost, and a ceiling on them would fire on
the generated WebUI bundle and the ``.mlir`` fixtures without telling anyone
anything they did not already know.

Usage
-----
    scripts/ci/check_binary_assets.py

Exits non-zero and names every undeclared or oversize file.
"""
import dataclasses
import fnmatch
import pathlib
import subprocess
import sys

# git's own binary heuristic: a NUL byte anywhere in the first 8000 bytes.
# See `buffer_is_binary()` in git's xdiff-interface.c.
BINARY_SNIFF_BYTES = 8000

# Ceiling on the declared set as a whole, so growth inside an existing rule is
# also a review event rather than a slow drift.
TOTAL_BUDGET_BYTES = 256 * 1024


@dataclasses.dataclass(frozen=True)
class Rule:
    """One declared binary asset path, its per-file ceiling, and why it exists."""

    pattern: str
    max_bytes: int
    why: str


DECLARED = (
    Rule(
        pattern="tests/fixtures/*.png",
        max_bytes=32 * 1024,
        why=(
            "Synthetic images for the vision preprocessing tests: a solid probe, "
            "two aspect-ratio shape fixtures and one page of rendered text for "
            "the GOT-OCR path. Checked in because the expected patch grids are "
            "asserted against exact pixel content."
        ),
    ),
    Rule(
        pattern="tests/fixtures/kimi_k3_vision/*.png",
        max_bytes=32 * 1024,
        why=(
            "NaViT dynamic-resolution probe for the Kimi K3 vision tower. Its "
            "dimensions pick the tile grid under test, so it cannot be generated "
            "at an arbitrary size at run time."
        ),
    ),
    Rule(
        pattern="tests/fixtures/audio/tone.*",
        max_bytes=32 * 1024,
        why=(
            "One second of a pure tone in WAV, MP3 and FLAC. The three encodings "
            "are the point: they cover the decoder branches behind /v1/audio, "
            "and the MP3 and FLAC framing cannot be synthesized in the test."
        ),
    ),
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def tracked_files(root: pathlib.Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [path for path in out.split("\0") if path]


def is_binary(path: pathlib.Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(BINARY_SNIFF_BYTES)
    except OSError:
        return False  # A submodule or broken symlink is not ours to judge.


def main() -> int:
    root = repo_root()
    undeclared: list[str] = []
    oversize: list[str] = []
    declared_bytes = 0
    declared_count = 0

    for rel in sorted(tracked_files(root)):
        path = root / rel
        if not path.is_file() or path.is_symlink() or not is_binary(path):
            continue

        size = path.stat().st_size
        rule = next(
            (r for r in DECLARED if fnmatch.fnmatch(rel, r.pattern)),
            None,
        )
        if rule is None:
            undeclared.append(f"{rel} ({size / 1024:.1f} KB)")
            continue

        declared_count += 1
        declared_bytes += size
        if size > rule.max_bytes:
            oversize.append(
                f"{rel} is {size / 1024:.1f} KB, over the "
                f"{rule.max_bytes / 1024:.0f} KB ceiling on '{rule.pattern}'"
            )

    over_budget = declared_bytes > TOTAL_BUDGET_BYTES

    if undeclared or oversize or over_budget:
        print("binary-assets: FAIL")
        for entry in undeclared:
            print(f"  undeclared: {entry}")
        for entry in oversize:
            print(f"  oversize:   {entry}")
        if over_budget:
            print(
                f"  budget:     declared binaries total "
                f"{declared_bytes / 1024:.1f} KB, over the "
                f"{TOTAL_BUDGET_BYTES / 1024:.0f} KB ceiling"
            )
        print()
        if undeclared:
            print(
                "A tracked binary file has to be declared in DECLARED in\n"
                "scripts/ci/check_binary_assets.py, with a per-file ceiling and a\n"
                "reason review can read. Binary blobs never delta-compress and\n"
                "deleting one later does not shrink the history, so adding one is\n"
                "a decision, not a detail.\n"
                "\n"
                "If the file is generated output, a profiling capture or test\n"
                "debris, add it to .gitignore and `git rm --cached` it instead."
            )
        if oversize or over_budget:
            print(
                "A ceiling was raised by the file, not by a reviewer. Confirm the\n"
                "asset still needs to be this large, then raise the number in the\n"
                "rule (or TOTAL_BUDGET_BYTES) in the same commit."
            )
        return 1

    print(
        f"binary-assets: OK — {declared_count} declared binary files, "
        f"{declared_bytes / 1024:.1f} KB of "
        f"{TOTAL_BUDGET_BYTES / 1024:.0f} KB budget."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
