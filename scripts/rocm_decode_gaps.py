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

"""Host-gap accounting per role and port unit for the ROCm decode profile.

Issue #2148. A port unit's share of decode GPU time implies a speedup ceiling
of ``1 / (1 - share)``, but a port that replaces many small dispatches also
removes the host time spent launching them, which that share leaves out. Here
every idle stretch of the decode window is charged to the dispatch that ends
it, so each role (and each port unit, a set of roles) owns its kernels' time
plus the launch gaps in front of them, and its share of the wall time gives a
ceiling that counts both.

``scripts/rocm_decode_profile.py`` imports this module; its tests live in
``tests/test_rocm_decode_profile.py``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Protocol, Sequence


class Interval(Protocol):
    start: int
    end: int


def attribute_gaps(decode: Sequence[Interval], roles: Sequence[str | None],
                   lo: int, hi: int) -> dict[str, int]:
    """Idle nanoseconds of ``[lo, hi]`` per role of the dispatch that ends them.

    ``decode`` must be in start order and start inside the window (the decode
    cut guarantees both). Walking it, ``frontier`` is the latest end seen so
    far (at least ``lo``); a dispatch starting after the frontier ends a gap of
    ``start - frontier``, charged to its role (``"unattributed"`` for None). A
    dispatch that overlaps the frontier ends no gap. The idle stretch from the
    last end to ``hi`` is ``"tail"``. Ends are clipped to ``hi`` as
    ``busy_ns`` clips them, so the values sum to ``(hi - lo) - busy_ns(decode,
    lo, hi)``.
    """
    if len(decode) != len(roles):
        raise ValueError(f"{len(decode)} dispatches but {len(roles)} roles")
    gaps: dict[str, int] = defaultdict(int)
    frontier = lo
    for d, r in zip(decode, roles):
        gaps[r or "unattributed"] += max(0, min(d.start, hi) - frontier)
        frontier = max(frontier, min(d.end, hi))
    gaps["tail"] += max(0, hi - frontier)
    return dict(gaps)


def ceiling(fraction: float | None) -> float | None:
    """``1 / (1 - fraction)`` to 2 decimals; None when undefined (a share of 100%)."""
    if fraction is None or fraction >= 1.0:
        return None
    return round(1.0 / (1.0 - max(fraction, 0.0)), 2)


def gap_scale(plain_host_gap_ms_per_token: float | None,
              host_gap_ms_per_token: float | None) -> float | None:
    """How much of the traced host gap a run without the profiler keeps.

    The tracer adds host time per dispatch but barely changes kernel durations,
    so each role's traced gap is scaled by the plain run's estimated gap over
    the traced one. None without a plain run; never negative.
    """
    if plain_host_gap_ms_per_token is None or host_gap_ms_per_token is None:
        return None
    if host_gap_ms_per_token <= 0:
        return 0.0
    return max(plain_host_gap_ms_per_token, 0.0) / host_gap_ms_per_token


def wall_fields(prefix: str, roles: Iterable[str], role_ns: Mapping[str, int],
                role_gap_ns: Mapping[str, int], role_calls: Mapping[str, int], *,
                gpu_sum: int, wall: int, tokens: int, scale: float | None,
                plain_wall_ms_per_token: float | None) -> dict[str, float | None]:
    """Host gap, wall share and both ceilings of the roles a port would replace.

    ``prefix`` names the role set (``fallback`` or ``reached_default``).
    ``ceiling_gpu_share`` is the old GPU-time bound; ``ceiling_wall`` adds the
    roles' launch gaps and divides by the window's wall time;
    ``plain_ceiling_wall_est`` does the same against the run without the
    profiler, the bound to compare a port's measured speedup against. Every key
    takes the prefix except the fallback set's three ceilings, which are
    ``ceiling_gpu_share``, ``ceiling_wall`` and ``plain_ceiling_wall_est``.
    """
    roles = tuple(roles)
    kernel_ns = sum(role_ns.get(r, 0) for r in roles)
    gap_ns = sum(role_gap_ns.get(r, 0) for r in roles)
    calls = sum(role_calls.get(r, 0) for r in roles)
    wall_frac = (kernel_ns + gap_ns) / wall if wall > 0 else None
    plain_frac = None
    if scale is not None and plain_wall_ms_per_token and tokens:
        unit_ms_per_token = (kernel_ns + gap_ns * scale) / 1e6 / tokens
        plain_frac = unit_ms_per_token / plain_wall_ms_per_token
    ceil = "" if prefix == "fallback" else f"{prefix}_"
    return {
        f"{prefix}_host_gap_ms_per_token": round(gap_ns / 1e6 / tokens, 4) if tokens else None,
        f"{prefix}_host_gap_us_per_dispatch": round(gap_ns / 1e3 / calls, 2) if calls else 0.0,
        f"{prefix}_wall_share_pct": round(100.0 * wall_frac, 2) if wall_frac is not None else None,
        f"{ceil}ceiling_gpu_share": ceiling(kernel_ns / gpu_sum) if gpu_sum > 0 else None,
        f"{ceil}ceiling_wall": ceiling(wall_frac),
        f"{ceil}plain_ceiling_wall_est": ceiling(plain_frac),
    }


def ceiling_cell(unit: Mapping[str, object]) -> str:
    """``wall (gpu share)`` for the report; the plain estimate when there is one."""
    wall = unit.get("plain_ceiling_wall_est")
    if wall is None:
        wall = unit.get("ceiling_wall")
    gpu = unit.get("ceiling_gpu_share")
    if wall is None and gpu is None:
        return "-"
    return f"{_fmt(wall)} ({_fmt(gpu)})"


def _fmt(v: object) -> str:
    return "-" if v is None else f"{v:.2f}"
