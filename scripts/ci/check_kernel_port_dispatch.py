#!/usr/bin/env python3
"""Require every fused-kernel launcher to choose its port through one helper.

Rationale
---------
mlxcel dispatches each fused kernel between per-backend ports. The hand-written
form of that choice was ``use_cuda ? cuda_port : metal_port``, which reads "not
CUDA" as "Metal". That was true while Metal and CUDA were the only backends and
became a defect the moment a third one existed: on ROCm the false arm was taken,
``fast::metal_kernel`` threw, and because the bridge declarations were not
``Result`` the throw crossed a ``noexcept`` cxx extern into ``std::terminate``.

Issue #1803 replaced the underlying ``!metal::is_available()`` test with a named
backend kind, and left nine sites still shaped that way. #1885 and #2018 then
added a refusal to each of the nine by hand, one launcher at a time, months
apart, after each had already aborted in a gate run. The hand-written guards
drifted: two cited support predicates that did not exist, and one hardcoded an
entry-point name that was wrong for one of its two callers.

The lesson is not that the guards were written badly. It is that a convention
which has to be re-applied at every new launcher will be missed at some of them,
and the failure is silent until that backend runs that kernel. So the choice now
goes through ``mlxcel::select_kernel_port`` (``src/lib/mlx-cpp/turbo/
kernel_port.h``), which owns the refusal and reads a per-kernel ``KernelPorts``
table, and this check keeps it that way.

The rules
---------
For every file that launches a custom kernel (it calls ``fast::metal_kernel(``,
``fast::cuda_kernel(`` or ``fast::hip_kernel(``):

1. It must not select a port with a conditional on the backend kind. Any
   ``gpu_kernel_backend() == ...Cuda`` outside ``kernel_port.cpp`` and
   ``gpu_backend.cpp`` is that shape, whatever it is spelled as.
2. It must not hand-roll the refusal. A ``custom_kernels_available()`` test
   followed by a throw belongs in ``select_kernel_port``; a launcher that writes
   its own drifts in wording and in which predicate it names.
3. It must not reach a kernel holder directly. Rules 1 and 2 only describe how a
   multi-port launcher goes wrong; a Metal-only one goes wrong by calling
   ``get_x_kernel().get()`` with no port resolution at all, which neither of them
   sees. This rule was added after reverting such a launcher by hand and watching
   the check still pass, which is the only reason it is known to be needed.
   Occurrences inside a ``KernelPorts`` table are the intended use and exempt.

Files listed in ``UNCONVERTED`` are exempt from rule 1 and 2 while their ports
are still Metal-only and their reachability off Apple is unresolved. The list is
meant to shrink; adding to it needs a reason in review.

And on the Rust side, for every ``.rs`` file:

4. A gate must not be spelled ``metal_is_available() || cuda_is_available()``, nor
   its De Morgan twin. The condition is not wrong today, and that is the problem:
   it names the two backends that happen to have ports, so on a third backend it
   reads as a missing term rather than as what it means, and "add
   ``rocm_is_available()``" is the natural conclusion and the wrong one. It would
   widen the gate past the port table and run the caller into the launcher's
   refusal. Say which question is being asked instead: the kernel's own
   ``*_available()`` predicate for "does this backend have this port", or
   ``gpu_backend_available()`` for "is there a GPU at all".

Rust files listed in ``BACKEND_ENUMERATION_TODO`` (empty since #2061) are
exempt from rule 4, each with the predicate it is waiting on. Unlike
``UNCONVERTED`` these are not a convention that was skipped: they need a support
predicate to be exported first (#1814).
"""

from __future__ import annotations

import pathlib
import re
import sys

# No exemptions.
#
# There were two while `turbo4_delegated_sdpa.cpp` and `sparse_v_sdpa.cpp` were
# Metal-only with no refusal of their own. Both are now in the table with
# `.cuda = nullptr, .rocm = nullptr`, which says "Metal only" as a value rather
# than as a convention, so every launcher in the tree routes through one helper
# and this set is empty.
#
# Keep it that way. An exemption list is how the pattern this check exists to
# prevent would come back: the entry gets added for a good reason, the reason is
# resolved, and the entry stays. If a launcher genuinely cannot use the table,
# that is worth a review conversation rather than a line here.
UNCONVERTED: set[str] = set()

# Rust gates that still enumerate Metal and CUDA, with what each one needs.
#
# Not a general exemption list: every entry names a predicate that does not exist
# yet, so the entry disappears when that predicate lands rather than when someone
# remembers to look. Empty since #2061, which ran the last entry's tests
# (`grouped_gemm_numeric_tests.rs`, MLX's own `gather_mm`) on gfx1151, saw them
# pass, and moved them to `gpu_backend_available()`.
BACKEND_ENUMERATION_TODO: set[str] = set()

# The helper's own translation units, which are allowed to name the backend.
HELPERS = {
    "src/lib/mlx-cpp/turbo/kernel_port.cpp",
    "src/lib/mlx-cpp/turbo/gpu_backend.cpp",
}

LAUNCHES = re.compile(r"fast::(metal|cuda|hip)_kernel\s*\(")
BACKEND_COMPARE = re.compile(r"gpu_kernel_backend\(\)\s*==")
HAND_ROLLED_GUARD = re.compile(
    r"if\s*\(\s*!\s*(?:mlxcel::)?custom_kernels_available\(\)\s*\)")
# A holder reached directly. Inside a port table this is the intended spelling,
# so table bodies are cut out before this is applied.
DIRECT_HOLDER = re.compile(r"\bget_[A-Za-z0-9_]*kernel[A-Za-z0-9_]*\s*\([^)]*\)\s*\.get\s*\(\)")
PORT_TABLE = re.compile(r"KernelPorts&\s+\w+\(\)\s*\{.*?\n\}", re.S)

# `metal_is_available() || cuda_is_available()` and `!metal && !cuda`, in either
# order, with or without a `crate::` / `mlxcel_core::` path. Matching on the two
# calls joined by one operator keeps a chain that tests a third backend as well
# from being reported, since that chain is not the defect.
_METAL = r"(?:crate::|mlxcel_core::)?metal_is_available\(\)"
_CUDA = r"(?:crate::|mlxcel_core::)?cuda_is_available\(\)"
# Only operators, whitespace and negation between the two calls, so this catches
# both `a || b` and `!a && !b` in either order and does not reach across an
# intervening third condition.
_JOIN = r"\s*(?:\|\||&&)\s*!?\s*"
BACKEND_ENUMERATION = re.compile(
    rf"{_METAL}{_JOIN}{_CUDA}|{_CUDA}{_JOIN}{_METAL}")
# A doc comment quoting the defect to explain why it is not used. Rule 4 reads
# code, so comment lines are dropped before it is applied.
RUST_COMMENT = re.compile(r"^\s*(?://|/\*|\*).*$", re.M)


def without_port_tables(text: str) -> str:
    """Blank out port-table bodies, keeping byte offsets so line numbers hold."""
    out = list(text)
    for m in PORT_TABLE.finditer(text):
        for i in range(m.start(), m.end()):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)


def blank_matches(text: str, pattern: re.Pattern[str]) -> str:
    """Blank out every match, keeping byte offsets so line numbers hold."""
    out = list(text)
    for m in pattern.finditer(text):
        for i in range(m.start(), m.end()):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)


def check_rust_gates(root: pathlib.Path) -> tuple[list[str], int]:
    """Rule 4: no gate that enumerates Metal and CUDA as the backends with ports."""
    failures: list[str] = []
    checked = 0
    for path in sorted(root.glob("src/**/*.rs")) + sorted(root.glob("examples/*.rs")):
        rel = path.relative_to(root).as_posix()
        if rel in BACKEND_ENUMERATION_TODO:
            continue
        text = path.read_text(encoding="utf-8")
        checked += 1
        for m in BACKEND_ENUMERATION.finditer(blank_matches(text, RUST_COMMENT)):
            line = text.count("\n", 0, m.start()) + 1
            failures.append(
                f"{rel}:{line}: gates on metal_is_available() and "
                "cuda_is_available(); ask the kernel's own *_available() "
                "predicate for \"has this port\", or gpu_backend_available() for "
                "\"is there a GPU\", so a third backend needs no edit here")
    return failures, checked


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    failures: list[str] = []
    checked = 0

    for path in sorted(root.glob("src/lib/**/*.cpp")):
        rel = path.relative_to(root).as_posix()
        if rel in HELPERS:
            continue
        text = path.read_text(encoding="utf-8")
        if not LAUNCHES.search(text):
            continue
        if rel in UNCONVERTED:
            continue
        checked += 1
        outside_tables = without_port_tables(text)
        for pattern, rule, haystack in (
            (BACKEND_COMPARE,
             "selects a port by comparing the backend kind; call "
             "mlxcel::select_kernel_port with a KernelPorts table instead",
             text),
            (HAND_ROLLED_GUARD,
             "hand-rolls the no-port refusal; select_kernel_port owns it, so "
             "the message and the predicate cannot drift per launcher",
             text),
            (DIRECT_HOLDER,
             "reaches a kernel holder directly; resolve it through "
             "mlxcel::select_kernel_port so a backend without a port gets an "
             "error instead of the Metal arm",
             outside_tables),
        ):
            for m in pattern.finditer(haystack):
                line = text.count("\n", 0, m.start()) + 1
                failures.append(f"{rel}:{line}: {rule}")

    rust_failures, rust_checked = check_rust_gates(root)
    failures += rust_failures

    if failures:
        print("Kernel port dispatch check failed:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nSee src/lib/mlx-cpp/turbo/kernel_port.h for the pattern and why "
            "it exists (lablup/mlxcel#1801, #1803, #1885, #2018).",
            file=sys.stderr)
        return 1

    print(f"Kernel port dispatch check passed: {checked} launcher file(s) "
          f"route through select_kernel_port, {len(UNCONVERTED)} exempt; "
          f"{rust_checked} Rust file(s) free of backend-enumerating gates, "
          f"{len(BACKEND_ENUMERATION_TODO)} awaiting a predicate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
