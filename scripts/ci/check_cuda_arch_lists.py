#!/usr/bin/env python3
"""Require every Blackwell CUDA architecture list in CI and release to carry a
hardware-capable, architecture-specific entry.

Rationale
---------
MLX compiles its hardware block-float converters (``cvt.rn.satfinite.e2m1x2.f32``
and siblings in ``mlx/backend/cuda/quantized/nvfp4_quantize.cuh``) only when the
translation unit is compiled for an *architecture-specific* target, which nvcc
signals with ``__CUDA_ARCH_SPECIFIC__`` and CMake spells as the ``a`` suffix
(``121a``). Every list this repository shipped or tested before issue #1934 named
Blackwell in its plain form (``100``, ``120``, ``121``), so NVFP4 and MXFP4
quantization silently took the scalar CUTLASS fallback in every release artifact
and in every CI job, and stochastic-rounding quantization was statically
unavailable. Nothing failed; the capability was simply compiled out everywhere,
including in the tests that were supposed to cover it.

That regression is invisible in review: ``"90a;100;121"`` and
``"90a;100a-real;100;121a-real;121"`` differ by two tokens in a YAML ``env``
block, and no build output names the difference. This check makes the plain-only
spelling fail instead.

The rule
--------
For every ``MLX_CUDA_ARCHITECTURES`` value in ``.github/workflows/*.yml``:

1. Every entry must parse as CMake spells one: ``<sm>[a|f][-real|-virtual]``.
2. No entry may use the family-specific ``f`` suffix. ``sm_121f`` satisfies the
   dispatcher gate in ``nvfp4_quantize.cuh`` (line 315, ``__CUDA_ARCH_FAMILY_SPECIFIC__
   >= 1000``) but not the converter gate (line 26, ``__CUDA_ARCH_SPECIFIC__``), so the
   fast path calls converters that were never defined and nvcc fails with three
   errors in that header. It is rejected here rather than discovered in a
   40-minute CUDA job.
3. Every Blackwell compute capability (major >= 10) named by a plain entry must
   also be named by an architecture-specific entry that emits a cubin, in the
   same list: ``121`` alone fails, ``121a-real;121`` and ``121a`` both pass.

Rule 3 deliberately accepts both surviving forms. The combined
``121a-real;121`` carries hardware SASS *and* keeps forward-JIT-capable plain
PTX; the bare ``121a`` carries hardware SASS and gives the PTX up. Which one
ships is a review decision (issue #1934); compiling the converter out is not.

Pre-Blackwell entries are untouched. Hopper fails the converter's
``__CUDA_ARCH__ >= 1000`` gate no matter how it is spelled, so ``90a`` gains
nothing here and ``80``/``86``/``89``/``70`` have no ``a`` variant at all.

The second half of the check is documentation drift, which is how this went
unnoticed for as long as it did: ``docs/installation.md`` names the two release
lists in prose, and nothing made it move when the workflow did. Every list in
``release.yml`` must appear verbatim in that document.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The first Blackwell compute capability major version. The converter gate in
#: ``nvfp4_quantize.cuh`` is ``__CUDA_ARCH__ >= 1000``, i.e. compute capability
#: 10.0 and newer, which is exactly "Blackwell and later".
BLACKWELL_MAJOR = 10

#: ``env:`` assignment of the architecture list, in any workflow, at any indent.
ENV_ASSIGNMENT = re.compile(r"^\s*MLX_CUDA_ARCHITECTURES:\s*(?P<value>\S.*?)\s*$")

#: One CMake architecture-list entry: digits, an optional variant suffix, an
#: optional code-object restriction. Mirrors ``parse_cuda_arch_entry`` in
#: ``src/lib/mlxcel-core/src/cuda_arch.rs``, which parses the same strings at
#: runtime for the startup mismatch check.
ENTRY = re.compile(r"^(?P<digits>\d{2,})(?P<variant>[af]?)(?P<restriction>-real|-virtual)?$")


@dataclass(frozen=True)
class Entry:
    """One parsed architecture-list entry."""

    raw: str
    major: int
    minor: int
    variant: str  # "", "a" or "f"
    emits_cubin: bool
    emits_ptx: bool

    @property
    def capability(self) -> tuple[int, int]:
        return (self.major, self.minor)


@dataclass(frozen=True)
class ArchList:
    """One ``MLX_CUDA_ARCHITECTURES`` assignment, located."""

    path: Path
    line: int
    value: str


def _parse_entry(raw: str) -> Entry | None:
    match = ENTRY.match(raw.strip())
    if match is None:
        return None
    digits = match.group("digits")
    restriction = match.group("restriction")
    # The minor version is the last digit: 70 is 7.0 and 121 is 12.1.
    return Entry(
        raw=raw.strip(),
        major=int(digits[:-1]),
        minor=int(digits[-1]),
        variant=match.group("variant"),
        emits_cubin=restriction != "-virtual",
        emits_ptx=restriction != "-real",
    )


def _strip_yaml_scalar(value: str) -> str:
    """Return the string a YAML plain or quoted scalar denotes.

    Only the two shapes these workflows use are handled, because anything else
    in this position should be read by a human before it reaches nvcc.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    # A plain scalar can carry a trailing comment; a quoted one cannot.
    return value.split("#", 1)[0].strip()


def collect_arch_lists(workflow_dir: Path) -> list[ArchList]:
    """Every architecture-list assignment in every workflow, in file order."""
    found: list[ArchList] = []
    for path in sorted(workflow_dir.glob("*.yml")) + sorted(workflow_dir.glob("*.yaml")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            match = ENV_ASSIGNMENT.match(line)
            if match is None:
                continue
            found.append(ArchList(path=path, line=number, value=_strip_yaml_scalar(match.group("value"))))
    return found


def check_list(arch_list: ArchList) -> list[str]:
    """Every rule violation in one list, as operator-readable lines."""
    problems: list[str] = []
    entries: list[Entry] = []
    for raw in arch_list.value.split(";"):
        if not raw.strip():
            continue
        entry = _parse_entry(raw)
        if entry is None:
            problems.append(
                f"entry {raw.strip()!r} is not a CMake architecture spelling "
                "(<sm>[a|f][-real|-virtual], e.g. 121a-real)"
            )
            continue
        entries.append(entry)

    for entry in entries:
        if entry.variant == "f":
            problems.append(
                f"entry {entry.raw!r} uses the family-specific `f` suffix, which does not compile: "
                "it satisfies the dispatcher gate in mlx/backend/cuda/quantized/nvfp4_quantize.cuh "
                "(__CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000) but not the converter gate "
                "(__CUDA_ARCH_SPECIFIC__), so the fast path calls converters that were never "
                "defined and nvcc fails with three errors in that header. Use the `a` suffix."
            )

    hardware_capable = {
        entry.capability for entry in entries if entry.variant == "a" and entry.emits_cubin
    }
    for entry in entries:
        if entry.variant != "" or entry.major < BLACKWELL_MAJOR:
            continue
        if entry.capability not in hardware_capable:
            sm = f"{entry.major}{entry.minor}"
            problems.append(
                f"Blackwell entry {entry.raw!r} is plain-only: nothing in this list compiles "
                f"sm_{sm} with the `a` suffix, so MLX's hardware NVFP4/MXFP4 converters "
                "(cvt.rn.satfinite.e2m1x2.f32, gated on __CUDA_ARCH_SPECIFIC__) are compiled out "
                f"and quantization silently takes the scalar fallback. Add {sm}a-real beside it "
                f"(keeps forward-JIT PTX) or replace it with {sm}a (drops it). See issue #1934."
            )

    return problems


def check_docs(release_lists: list[ArchList], installation: Path) -> list[str]:
    """Release lists that `docs/installation.md` does not quote verbatim."""
    if not installation.exists():
        return [f"{installation} does not exist, so the release lists cannot be checked against it"]
    text = installation.read_text()
    return [
        f"{arch_list.path.name}:{arch_list.line} builds {arch_list.value!r}, which "
        f"{installation.name} does not name. The document tells operators which architectures the "
        "published archives carry; a list that moves without it is how issue #1934 stayed "
        "invisible. Quote the list verbatim there."
        for arch_list in release_lists
        if arch_list.value not in text
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root to check (default: this script's repository)",
    )
    args = parser.parse_args()

    workflow_dir = args.root / ".github" / "workflows"
    if not workflow_dir.is_dir():
        print(f"no workflow directory at {workflow_dir}", file=sys.stderr)
        return 1

    arch_lists = collect_arch_lists(workflow_dir)
    if not arch_lists:
        print(
            f"no MLX_CUDA_ARCHITECTURES assignment found under {workflow_dir}. Either the CUDA "
            "jobs lost their architecture pin (auto-detection then decides what ships) or this "
            "check no longer knows where to look.",
            file=sys.stderr,
        )
        return 1

    failures: list[str] = []
    for arch_list in arch_lists:
        for problem in check_list(arch_list):
            relative = arch_list.path.relative_to(args.root)
            failures.append(f"{relative}:{arch_list.line}: {problem}")

    release_lists = [a for a in arch_lists if a.path.name == "release.yml"]
    failures.extend(check_docs(release_lists, args.root / "docs" / "installation.md"))

    if failures:
        print("CUDA architecture lists are not shippable:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"CUDA architecture lists OK ({len(arch_lists)} checked):")
    for arch_list in arch_lists:
        print(f"  {arch_list.path.relative_to(args.root)}:{arch_list.line}  {arch_list.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
