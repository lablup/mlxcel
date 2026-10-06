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

And for the ROCm wavefront hold (issue #2147):

5. A table marked ``.rocm_any_wave_size = true`` is selected on a 64-lane
   (CDNA) device, where no one has run it, on the strength of its HIP body
   having no lane-level operation. That claim is checked here rather than
   trusted: the marked tables are pinned in ``EXPECTED_ANY_WAVE`` with the HIP
   source each compiles, the table's ``.rocm`` getter is followed to its
   holder and must compile exactly that source with no further arguments (no
   unscanned header), and the source (comments stripped, one raw literal) must
   contain no shuffle, ballot, lane builtin or 32-lane index arithmetic
   (``LANE_OPS``). Marking another table, or adding a shuffle to
   a marked one, fails until the pin is reviewed. The ``#error`` guards on
   ``__AMDGCN_WAVEFRONT_SIZE`` cannot do this job: AMD clang 23 defines neither
   spelling of the macro.
6. ``set_rocm_port_warp_size_for_tests``, the seam that replaces the wavefront
   width the port tables are checked against, is called only from tests: a
   Rust file under ``tests/``, a ``*_tests.rs`` module or ``test_support/``.
   Its bridge declaration and C++ definitions are the only other places it
   may appear, an exact number of times per file (``SEAM_DEFINITIONS``).
"""

from __future__ import annotations

import argparse
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

# Tables allowed to run their ROCm port at any wavefront width (rule 5), each
# mapped to the HIP source its `.rocm` entry compiles. Adding a table here says
# its body was read and has no lane-level operation; validating a
# shuffle-based port on a wave64 device and marking it is a separate,
# per-kernel decision (#2147).
EXPECTED_ANY_WAVE = {
    "fused_rope_ports": "FUSED_ROPE_APPEND_HIP_SOURCE",
    "paged_merge_ports": "PAGED_ATTENTION_MERGE_HIP_SOURCE",
    "gumbel_ports": "GUMBEL_MAX_SAMPLE_HIP_SOURCE",
    "rejection_ports": "REJECTION_SAMPLE_HIP_SOURCE",
    "xielu_ports": "XIELU_HIP_SOURCE",
}

# Cross-lane operations a body correct at any wavefront width cannot use: the
# HIP shuffle, vote and mask intrinsics, the AMDGCN lane builtins they lower
# to, the wavefront-size macros and constants, and hard-coded 32-lane index
# arithmetic on `threadIdx.x`.
LANE_OPS = re.compile(
    r"__shfl\w*|__ballot\w*|__activemask|\b__any(?:_sync)?\s*\(|"
    r"\b__all(?:_sync)?\s*\(|__lane_id|\bwarpSize\b|__reduce_\w+_sync|"
    r"__syncwarp|__match_\w+|\b__fns\w*\s*\(|__lanemask_\w+|__hip_move_dpp|"
    r"__hip_ds_\w+|__AMDGCN_WAVEFRONT_SIZE\w*|"
    r"__builtin_amdgcn_(?:ds_swizzle|mov_dpp\w*|update_dpp|readlane|"
    r"readfirstlane|writelane|ds_bpermute|ds_permute|ballot\w*|permlane\w*|"
    r"mbcnt_\w+|wavefrontsize|icmp|fcmp|wave_\w+)|"
    r"threadIdx\.x\s*(?:%\s*32u?|&\s*31u?|>>\s*5u?|/\s*32u?)\b")
ANY_WAVE_FLAG = re.compile(r"\.rocm_any_wave_size\s*=\s*true")
CPP_LINE_COMMENT = re.compile(r"//[^\n]*")
CPP_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)

# The test seam of rule 6, and how many times each file that declares or
# defines it may name it outside comments: the declaration in each header, the
# definition in each .cpp, and the bridge wrapper's one forwarding call. An
# exact count rather than a file exemption, so a production call added
# anywhere in those (large) files still fails.
SEAM = re.compile(r"\bset_rocm_port_warp_size_for_tests\b")
SEAM_DEFINITIONS = {
    "src/lib/mlx-cpp/turbo/gpu_backend.h": 1,
    "src/lib/mlx-cpp/turbo/gpu_backend.cpp": 1,
    "src/lib/mlxcel-core/cpp/mlx_cxx_bridge.h": 1,
    "src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp": 2,
}
# The cxx bridge declaration in mlxcel-core's lib.rs.
SEAM_DECLARATION = re.compile(r"\bfn\s+set_rocm_port_warp_size_for_tests\s*\(")

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


def hip_source_body(root: pathlib.Path, name: str) -> tuple[str, str] | None:
    """The raw-string body of the HIP source constant ``name`` and its file."""
    definition = re.compile(rf"\b{name}\s*=\s*R\"(\w*)\(")
    for path in sorted(root.glob("src/lib/**/*.h")) + sorted(root.glob("src/lib/**/*.cpp")):
        text = path.read_text(encoding="utf-8", errors="replace")
        m = definition.search(text)
        if not m:
            continue
        end = text.find(f"){m.group(1)}\"", m.end())
        if end < 0:
            continue
        rest = text[end + len(m.group(1)) + 2:].lstrip()
        if not rest.startswith(";"):
            # Adjacent literals concatenate; only the first would be scanned.
            return None
        return text[m.end():end], path.relative_to(root).as_posix()
    return None


def braced_block(text: str, open_idx: int) -> str:
    """The text from the brace at ``open_idx`` to its matching close."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx:i + 1]
    return text[open_idx:]


def function_body(text: str, name: str) -> str | None:
    """The body of the function or struct ``name`` defined in ``text``."""
    # A struct first: its constructor would otherwise match as a function.
    for pattern in (rf"\bstruct\s+{name}\b[^;{{]*\{{",
                    rf"\b{name}\s*\([^;{{}}]*\)\s*(?:const\s*)?\{{"):
        m = re.search(pattern, text)
        if m:
            return braced_block(text, m.end() - 1)
    return None


def rocm_entry_sources(text: str, table: str) -> tuple[list[str], str | None]:
    """HIP sources compiled by a table's ``.rocm`` getter, and any problem.

    Follows the getter named in the ``.rocm`` lambda to its definition, the
    ``static <Holder> holder`` it returns, the holder struct, and one level of
    helper functions those call, collecting every ``fast::hip_kernel`` call.
    """
    entry = re.search(r"\.rocm\s*=\s*\+\[\]\(\)[^{]*\{(.*?)\}\s*,", table, re.S)
    if not entry:
        return [], "no .rocm lambda to follow"
    getter = re.search(r"\b(get_\w+)\s*(?:<[^>]*>)?\s*\(\s*\)", entry.group(1))
    if not getter:
        return [], "the .rocm lambda calls no get_*() holder accessor"
    bodies: list[str] = []
    getter_body = function_body(text, getter.group(1))
    if getter_body is None:
        return [], f"{getter.group(1)} is not defined in this file"
    bodies.append(getter_body)
    for holder in re.findall(r"\bstatic\s+(\w+)\s*\*?\s*holder\b", getter_body):
        body = function_body(text, holder)
        if body:
            bodies.append(body)
    # One level of helpers (for example a `make_*_kernel(backend)` factory).
    for body in list(bodies):
        for call in set(re.findall(r"\b([a-z_]\w*)\s*\(", body)):
            if call in {"get", "if", "for", "while", "switch", "return",
                        "sizeof", "static_cast", "call_once"}:
                continue
            helper = function_body(text, call)
            if helper and helper not in bodies and "hip_kernel" in helper:
                bodies.append(helper)
    sources: list[str] = []
    for body in bodies:
        for call in LAUNCHES.finditer(body):
            if call.group(1) != "hip":
                continue
            named = re.search(r"\b([A-Z][A-Z0-9_]*_HIP_SOURCE)\b(\s*\)?\s*\)\s*;)?",
                              body[call.end():call.end() + 800])
            if not named:
                return [], "its fast::hip_kernel call names no *_HIP_SOURCE constant"
            if not named.group(2):
                return [], (f"its fast::hip_kernel call passes arguments after "
                            f"{named.group(1)} (a header?), which this check "
                            "does not scan for lane operations")
            if named.group(1) not in sources:
                sources.append(named.group(1))
    return sources, None


def check_any_wave_tables(root: pathlib.Path) -> tuple[list[str], int]:
    """Rule 5: a table marked any-wave compiles a HIP body with no lane op."""
    failures: list[str] = []
    marked: dict[str, str] = {}
    for path in sorted(root.glob("src/lib/**/*.cpp")):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in PORT_TABLE.finditer(text):
            if not ANY_WAVE_FLAG.search(m.group(0)):
                continue
            name = re.match(r"KernelPorts&\s+(\w+)\(", m.group(0)).group(1)
            marked[name] = rel
            line = text.count("\n", 0, m.start()) + 1
            source = EXPECTED_ANY_WAVE.get(name)
            if source is None:
                failures.append(
                    f"{rel}:{line}: {name} is marked rocm_any_wave_size but is "
                    "not in EXPECTED_ANY_WAVE; marking a port runs it on wave64 "
                    "devices, so pin it there, with its HIP source, in the same "
                    "change for review")
                # Still read the body when the file compiles a single HIP
                # source, so the message says what is wrong with it too.
                launches = [c for c in LAUNCHES.finditer(text) if c.group(1) == "hip"]
                named = re.search(r"\b([A-Z][A-Z0-9_]*_HIP_SOURCE)\b",
                                  text[launches[0].end():launches[0].end() + 800]) \
                    if len(launches) == 1 else None
                if not named:
                    continue
                source = named.group(1)
            # The pinned source must be the one the table's own `.rocm` getter
            # compiles, followed from the getter to its holder, so pointing the
            # entry at another holder cannot pass on the strength of the pinned
            # body still being compiled elsewhere in the file.
            compiled, problem = rocm_entry_sources(text, m.group(0))
            if problem:
                failures.append(f"{rel}:{line}: {name}: {problem}")
                continue
            if compiled != [source]:
                failures.append(
                    f"{rel}:{line}: {name} is pinned to {source}, but its .rocm "
                    f"entry compiles {compiled or 'no HIP source'}")
                continue
            found = hip_source_body(root, source)
            if found is None:
                failures.append(
                    f"{rel}:{line}: {name}'s HIP source {source} has no "
                    "single raw-string definition under src/lib (a source "
                    "split across concatenated literals is not scanned)")
                continue
            body, where = found
            code = CPP_LINE_COMMENT.sub("", CPP_BLOCK_COMMENT.sub("", body))
            op = LANE_OPS.search(code)
            if op:
                failures.append(
                    f"{rel}:{line}: {name} is marked rocm_any_wave_size, but "
                    f"{source} ({where}) uses `{op.group(0)}`, a lane-level "
                    "operation; a wave64 device would run it unvalidated. Drop "
                    "the mark, or validate the port on wave64 and make the body "
                    "width-independent")
    for name in sorted(set(EXPECTED_ANY_WAVE) - set(marked)):
        failures.append(
            f"scripts/ci/check_kernel_port_dispatch.py: EXPECTED_ANY_WAVE pins "
            f"{name}, but no port table by that name is marked "
            "rocm_any_wave_size; it would now be refused on wave64. Update the "
            "pin if that was intended")
    return failures, len(marked)


def check_seam_callers(root: pathlib.Path) -> list[str]:
    """Rule 6: only tests replace the wavefront width the port tables read."""
    failures: list[str] = []
    for path in sorted(root.glob("src/lib/**/*")):
        if path.suffix not in {".cpp", ".h", ".hpp", ".cc", ".hip", ".mm"}:
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        code = CPP_LINE_COMMENT.sub("", CPP_BLOCK_COMMENT.sub("", text))
        found = len(SEAM.findall(code))
        allowed = SEAM_DEFINITIONS.get(rel, 0)
        if found > allowed:
            failures.append(
                f"{rel}: calls set_rocm_port_warp_size_for_tests outside its "
                f"definition ({found} uses, {allowed} expected); it is a test "
                "seam, and production code must read the hardware wavefront "
                "width")
    rust = []
    for pattern in ("src/**/*.rs", "examples/**/*.rs", "benches/**/*.rs", "crates/**/*.rs"):
        rust += sorted(root.glob(pattern))
    for path in rust:
        rel = path.relative_to(root).as_posix()
        if rel.endswith("_tests.rs") or "/test_support/" in rel or "/tests/" in rel:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        code = blank_matches(text, RUST_COMMENT)
        for m in SEAM.finditer(code):
            line_start = code.rfind("\n", 0, m.start()) + 1
            line_end = code.find("\n", m.end())
            if SEAM_DECLARATION.search(code[line_start:line_end if line_end >= 0 else None]) \
                    and rel == "src/lib/mlxcel-core/src/lib.rs":
                continue
            line = text.count("\n", 0, m.start()) + 1
            failures.append(
                f"{rel}:{line}: calls set_rocm_port_warp_size_for_tests outside "
                "a test; it replaces the wavefront width every ROCm port table "
                "is checked against, so only tests under tests/, *_tests.rs or "
                "test_support/ may use it")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
        help="repository root to check (default: this checkout)")
    root = parser.parse_args().root.resolve()
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
    wave_failures, any_wave = check_any_wave_tables(root)
    failures += wave_failures
    failures += check_seam_callers(root)

    if failures:
        print("Kernel port dispatch check failed:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nSee src/lib/mlx-cpp/turbo/kernel_port.h for the pattern and why "
            "it exists (lablup/mlxcel#1801, #1803, #1885, #2018, #2147).",
            file=sys.stderr)
        return 1

    print(f"Kernel port dispatch check passed: {checked} launcher file(s) "
          f"route through select_kernel_port, {len(UNCONVERTED)} exempt; "
          f"{rust_checked} Rust file(s) free of backend-enumerating gates, "
          f"{len(BACKEND_ENUMERATION_TODO)} awaiting a predicate; "
          f"{any_wave} any-wave ROCm table(s) free of lane operations, "
          f"wave-size test seam used only by tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
