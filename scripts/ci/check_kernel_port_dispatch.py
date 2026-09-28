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

Files listed in ``UNCONVERTED`` are exempt from rule 1 and 2 while their ports
are still Metal-only and their reachability off Apple is unresolved. The list is
meant to shrink; adding to it needs a reason in review.
"""

from __future__ import annotations

import pathlib
import re
import sys

# Launcher files whose ports are Metal-only and whose reachability on other
# backends has not been established. Tracked by lablup/mlxcel#1814; two of them
# have no refusal at all today, and `sparse_v_available()` gates on an env
# threshold and a KV cache mode with no backend term, so adding a guard is a
# behavior change that needs its own analysis rather than a mechanical edit.
UNCONVERTED = {
    "src/lib/mlx-cpp/turbo/turbo4_delegated_sdpa.cpp",
    "src/lib/mlx-cpp/turbo/sparse_v_sdpa.cpp",
}

# The helper's own translation units, which are allowed to name the backend.
HELPERS = {
    "src/lib/mlx-cpp/turbo/kernel_port.cpp",
    "src/lib/mlx-cpp/turbo/gpu_backend.cpp",
}

LAUNCHES = re.compile(r"fast::(metal|cuda|hip)_kernel\s*\(")
BACKEND_COMPARE = re.compile(r"gpu_kernel_backend\(\)\s*==")
HAND_ROLLED_GUARD = re.compile(
    r"if\s*\(\s*!\s*(?:mlxcel::)?custom_kernels_available\(\)\s*\)")


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
        for pattern, rule in (
            (BACKEND_COMPARE,
             "selects a port by comparing the backend kind; call "
             "mlxcel::select_kernel_port with a KernelPorts table instead"),
            (HAND_ROLLED_GUARD,
             "hand-rolls the no-port refusal; select_kernel_port owns it, so "
             "the message and the predicate cannot drift per launcher"),
        ):
            for m in pattern.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                failures.append(f"{rel}:{line}: {rule}")

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
          f"route through select_kernel_port, {len(UNCONVERTED)} exempt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
