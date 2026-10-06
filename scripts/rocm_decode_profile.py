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

"""Cut the measured decode out of a rocprofv3 kernel trace and attribute it.

Issue #2061. `scripts/rocm_decode_profile.sh` runs `mlxcel-bench-decode` under
`rocprofv3 --kernel-trace` with `MLXCEL_BENCH_PHASE_MARKS=1`; this script reads
the trace and the bench's log and writes, per run:

* ``<name>_decode_kernels.csv``: one row per kernel name dispatched during the
  measured decode, with calls, total GPU time, share of decode GPU time, its
  kernel class and the #1814 port unit it is attributed to;
* ``<name>_summary.json``: decode GPU time and host gap per token, tok/s with
  and without the profiler, share per port unit, and the checks below.

The measured decode is the dispatches that start between the bench's
``decode_start`` and ``measured_end`` marks (see
``src/bin/bench_decode/phase_marks.rs``). The prefill before it ends in a
blocking ``eval`` of the first token, so no kernel should straddle the cut;
the summary records how many do (expected 0) and the idle gap before the
first decode dispatch.

``report`` renders the summaries of a directory as the Markdown tables of the
published profile.

Run the unit tests with ``python3 -m unittest tests/test_rocm_decode_profile.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Kernel classes. First match wins. Patterns run on the demangled kernel name
# as rocprofv3 writes it; see the profile doc for how each was checked against
# the MLX ROCm overlay source.
# --------------------------------------------------------------------------
KERNEL_CLASSES: list[tuple[str, re.Pattern[str]]] = [
    ("gather_qmv", re.compile(r"gather.*q(mv|mm)|q(mv|mm).*gather", re.I)),
    ("gather_mm", re.compile(r"gather_mm|segmented|sorted_rhs", re.I)),
    ("qmv", re.compile(r"\bqmv|qmv_|_qmv|qvm", re.I)),
    ("qmm", re.compile(r"\bqmm|qmm_|_qmm", re.I)),
    ("gemm", re.compile(r"Cijk_|gemm|gemv|rocblas|hipblaslt|wmma_matmul", re.I)),
    ("rms_norm", re.compile(r"rms_?norm", re.I)),
    ("layer_norm", re.compile(r"layer_?norm", re.I)),
    ("rope", re.compile(r"rope", re.I)),
    ("sdpa", re.compile(r"sdpa|attention|flash", re.I)),
    ("softmax", re.compile(r"softmax", re.I)),
    ("logsumexp", re.compile(r"logsumexp", re.I)),
    ("arg_reduce", re.compile(r"arg_?reduce|argmax|argmin", re.I)),
    ("scan", re.compile(r"scan|cumsum", re.I)),
    ("sort", re.compile(r"sort|partition", re.I)),
    ("random", re.compile(r"rbits|random|philox|threefry", re.I)),
    ("conv", re.compile(r"conv", re.I)),
    ("reduce", re.compile(r"reduce|all_reduce|row_reduce|col_reduce", re.I)),
    ("gather_scatter", re.compile(r"gather|scatter|take|slice_update|masked", re.I)),
    ("copy", re.compile(r"copy|fill|arange", re.I)),
    ("binary", re.compile(r"binary", re.I)),
    ("ternary", re.compile(r"ternary|select|where", re.I)),
    ("unary", re.compile(r"unary", re.I)),
    ("dequantize", re.compile(r"dequantize", re.I)),
    # MLX's JIT-compiled elementwise kernels (`mlx::core::compile`) are named
    # after their encoded tape plus `_contiguous` / `_strided`, e.g.
    # `CV2ISigmoid...Multiply..._contiguous` for mlxcel's compiled SwiGLU and
    # `BV2ISigmoid...` for SiLU; `compiled.cpp` in the overlay builds the name.
    ("compiled", re.compile(r"rocm::[A-Z][A-Za-z0-9]+_[A-Za-z0-9_]*_(contiguous|strided)<")),
]

# Port units split from #1814, in the order #1814 listed them.
PORT_UNITS = {
    "2063": "fused_add_rms_norm + fused_rope_qk_append",
    "2064": "samplers (gumbel_max_sample, rejection_sample)",
    "2065": "fused MoE decode (moe_gateup, moe_down)",
    "2067": "SSM update (ssm_update_kernel)",
    "2068": "paged attention (v1, v2, merge)",
}


def classify(name: str) -> str:
    for cls, pat in KERNEL_CLASSES:
        if pat.search(name):
            return cls
    return "other"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
# slots: a full kernel trace is millions of rows, so keep each one small.
@dataclass(slots=True)
class Dispatch:
    name: str
    start: int
    end: int
    grid: int = 0

    @property
    def dur(self) -> int:
        return self.end - self.start


PHASE_RE = re.compile(r"^\[phase\] (\w+) monotonic_ns=(\S+) boottime_ns=(\S+)")
DECODE_RE = re.compile(r"^\s*Decode:\s+([\d.]+) ms \(([\d.]+) tok/s\)")
PREFILL_RE = re.compile(r"^\s*Prefill:\s+([\d.]+) ms \(([\d.]+) tok/s\)")
GEN_RE = re.compile(r"^\s*Generated tokens:\s+(\d+)")
PROMPT_RE = re.compile(r"^\s*Prompt tokens:\s+(\d+)")


@dataclass
class BenchLog:
    marks: dict[str, dict[str, int]] = field(default_factory=dict)
    decode_ms: float | None = None
    decode_tok_s: float | None = None
    prefill_ms: float | None = None
    prefill_tok_s: float | None = None
    generated: int | None = None
    prompt: int | None = None


def parse_log(text: str) -> BenchLog:
    log = BenchLog()
    for line in text.splitlines():
        if m := PHASE_RE.match(line):
            clocks = {}
            for clock, value in (("monotonic", m.group(2)), ("boottime", m.group(3))):
                if value != "na":
                    clocks[clock] = int(value)
            log.marks[m.group(1)] = clocks
        elif m := DECODE_RE.match(line):
            log.decode_ms, log.decode_tok_s = float(m.group(1)), float(m.group(2))
        elif m := PREFILL_RE.match(line):
            log.prefill_ms, log.prefill_tok_s = float(m.group(1)), float(m.group(2))
        elif m := GEN_RE.match(line):
            log.generated = int(m.group(1))
        elif m := PROMPT_RE.match(line):
            log.prompt = int(m.group(1))
    return log


def _col(header: list[str], *names: str) -> int:
    for n in names:
        if n in header:
            return header.index(n)
    raise ValueError(f"kernel trace has none of the columns {names}; header: {header}")


def read_trace(path: pathlib.Path) -> list[Dispatch]:
    with path.open(newline="") as fh:
        rows = csv.reader(fh)
        header = next(rows)
        i_name = _col(header, "Kernel_Name", "KernelName")
        i_start = _col(header, "Start_Timestamp", "BeginNs")
        i_end = _col(header, "End_Timestamp", "EndNs")
        grid_cols = [header.index(c) for c in ("Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z")
                     if c in header]
        out = []
        for r in rows:
            grid = 1
            for c in grid_cols:
                grid *= max(int(r[c] or 1), 1)
            out.append(Dispatch(r[i_name], int(r[i_start]), int(r[i_end]),
                                grid if grid_cols else 0))
    out.sort(key=lambda d: d.start)
    return out


# --------------------------------------------------------------------------
# Decode window
# --------------------------------------------------------------------------
def pick_clock(log: BenchLog, dispatches: list[Dispatch]) -> str:
    """The host clock whose marks actually bracket the traced dispatches.

    rocprofv3 stamps dispatches on one host clock; the bench prints two. The
    right one is the one under which the bench's own passes contain the
    dispatches, so count dispatches inside [warmup_start, measured_end] for
    each and keep the larger (they tie when the host has never suspended).
    """
    best, best_n = None, -1
    for clock in ("boottime", "monotonic"):
        try:
            lo = log.marks["warmup_start"][clock]
            hi = log.marks["measured_end"][clock]
        except KeyError:
            continue
        n = sum(1 for d in dispatches if lo <= d.start <= hi)
        if n > best_n:
            best, best_n = clock, n
    if best is None or best_n <= 0:
        raise ValueError("no dispatch lies between the warmup_start and measured_end marks "
                         "on either clock; was MLXCEL_BENCH_PHASE_MARKS=1 set?")
    return best


def busy_ns(ds: list[Dispatch], lo: int, hi: int) -> int:
    """Union of dispatch intervals clipped to [lo, hi]."""
    total, cur_s, cur_e = 0, None, None
    for d in ds:
        s, e = max(d.start, lo), min(d.end, hi)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


# --------------------------------------------------------------------------
# Roles: what mlxcel op each decode dispatch belongs to
# --------------------------------------------------------------------------
# Generic kernels (binary, copy, reduce) are emitted by many ops, so a name alone
# cannot say which op dispatched them. MLX evaluates a decode step's graph in a
# fixed order, so each op leaves a recognisable run of dispatches; the rules
# below read those runs. Each was checked against the model code and a printed
# decode step (see the profile doc, "Attribution").
ROLES = (
    "sampler_tail",            # after the lm_head GEMV: logit bias, sampling
    "add_rms_join_post_attn",  # residual add + RMSNorm after attention's o_proj
    "add_rms_join",            # the other residual add + RMSNorm pairs
    "rope_append",             # q/k RoPE and the K/V cache writes between them
    "moe_expert_gemv",         # gather_qmm expert GEMVs
    "moe_activation",          # SwitchGLU activation between the expert GEMVs
    "moe_weighted_sum",        # score-weighted sum after the down GEMV
    "moe_gather_indices",      # the arange each gather_qmm builds
    "ssm_step",                # the Mamba2 SSD step graph (ssm_step)
    "ssm_conv",                # depthwise conv1d and its state copies
    "ssm_silu",                # SiLU on the conv output and on the gate
    "ssm_gated_norm",          # gated RMSNorm closing the mixer
    "paged_attention",         # paged-attention graph fallback
)


def assign_roles(decode: list[Dispatch]) -> list[str | None]:
    n = len(decode)
    cls = [classify(d.name) for d in decode]
    role: list[str | None] = [None] * n

    def is_add(i: int) -> bool:
        return cls[i] == "binary" and "::Add," in decode[i].name

    # Sampler tail: the lm_head is the widest qmv of the step (vocab rows).
    qmv_grids = [d.grid for d, c in zip(decode, cls) if c == "qmv"]
    if qmv_grids:
        head_grid = max(qmv_grids)
        stop = {"qmv", "rms_norm", "dequantize", "gather_qmv", "sdpa", "gemm"}
        for i in range(n):
            if cls[i] == "qmv" and decode[i].grid == head_grid:
                j = i + 1
                while j < n and cls[j] not in stop and "gather_rows" not in decode[j].name:
                    role[j] = "sampler_tail"
                    j += 1

    # Mamba2 mixer: the dispatches between the qmv before a conv1d (in_proj)
    # and the qmv after it (out_proj).
    for i in range(n):
        if cls[i] != "conv" or role[i] is not None:
            continue
        lo = i
        while lo > 0 and cls[lo - 1] not in ("qmv", "gather_qmv"):
            lo -= 1
        hi = i
        while hi + 1 < n and cls[hi + 1] not in ("qmv", "gather_qmv"):
            hi += 1
        span = range(lo, hi + 1)
        norm_at = max((k for k in span if cls[k] == "rms_norm"), default=None)
        for k in span:
            if norm_at is not None and k >= norm_at:
                role[k] = "ssm_gated_norm"
            elif cls[k] == "conv" or "copy_gg" in decode[k].name:
                role[k] = "ssm_conv"
            elif cls[k] == "compiled" and "Sigmoid" in decode[k].name:
                role[k] = "ssm_silu"
            else:
                role[k] = "ssm_step"

    # MoE: clusters of gather_qmv dispatches.
    g = [i for i in range(n) if cls[i] == "gather_qmv"]
    clusters: list[list[int]] = []
    for i in g:
        if clusters and i - clusters[-1][-1] <= 12:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    for c in clusters:
        for i in c:
            role[i] = "moe_expert_gemv"
        for k in range(c[0] + 1, c[-1]):
            if role[k] is None and cls[k] in ("compiled", "binary", "copy", "unary") \
                    and "::Divide," not in decode[k].name:
                role[k] = "moe_activation"
        k = c[-1] + 1
        while k < n and role[k] is None and not is_add(k) \
                and cls[k] in ("binary", "copy", "reduce", "unary"):
            role[k] = "moe_weighted_sum"
            k += 1
    if g:
        for i in range(n):
            if role[i] is None and "arange_kernel<unsigned int>" in decode[i].name:
                role[i] = "moe_gather_indices"

    # RoPE and the cache writes it brackets.
    for i in range(n):
        if role[i] is None and cls[i] == "rope":
            role[i] = "rope_append"
            k = i + 1
            while k < n and role[k] is None and "copy_gg" in decode[k].name:
                role[k] = "rope_append"
                k += 1

    # Residual add + RMSNorm pairs; post-attention when the add follows o_proj
    # (a qmv) that follows the attention kernel.
    for i in range(n - 1):
        if role[i] is None and role[i + 1] is None and is_add(i) and cls[i + 1] == "rms_norm":
            post = i >= 2 and cls[i - 1] == "qmv" and cls[i - 2] == "sdpa"
            r = "add_rms_join_post_attn" if post else "add_rms_join"
            role[i] = role[i + 1] = r

    for i in range(n):
        if role[i] is None and "paged" in decode[i].name.lower():
            role[i] = "paged_attention"
    return role


# --------------------------------------------------------------------------
# Port units: which roles each would replace, and which the shipped code reaches
# --------------------------------------------------------------------------
UNIT_ROLES = {
    "2063": ("add_rms_join_post_attn", "rope_append"),
    "2064": ("sampler_tail",),
    "2065": ("moe_expert_gemv", "moe_activation", "moe_weighted_sum", "moe_gather_indices"),
    "2067": ("ssm_step",),
    "2068": ("paged_attention",),
}


def reach(unit: str, config: dict, sampled: bool) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """(roles reached with shipped defaults, roles reached with opt-ins, note).

    Read from mlxcel at the profiled commit; see the doc for the line numbers.
    """
    mt = config.get("model_type", "")
    rope_scaling = config.get("rope_scaling") or {}
    rope_table = (rope_scaling.get("rope_type") or rope_scaling.get("type")) in ("llama3", "yarn")
    if unit == "2063":
        if mt != "llama":
            return (), (), f"{mt} never calls fused_add_rms_norm or forward_fused_rope_append"
        reached = ("add_rms_join_post_attn",) + (() if rope_table else ("rope_append",))
        note = ("both paths are on by default in a rocm build (FUSED_ADD_RMSNORM_DEFAULT / "
                "FUSED_ROPE_APPEND_DEFAULT = cfg!(feature = \"rocm\"), #2145); only the "
                "post-attention join calls the fused norm")
        if rope_table:
            note += "; rope_scaling builds a frequency table, which the RoPE kernel cannot take"
        return reached, reached, note
    if unit == "2064":
        if sampled:
            return ("sampler_tail",), ("sampler_tail",), "sampled run: the draw is the fallback chain"
        return (), (), "greedy argmax dispatches neither sampler kernel"
    if unit == "2065":
        if mt in ("qwen3_moe", "qwen3_next", "mixtral", "dbrx", "cohere2_moe", "klear",
                  "afmoe", "lfm2", "bailing_moe"):
            return UNIT_ROLES["2065"], UNIT_ROLES["2065"], "forward_fused_kernel caller"
        if mt == "nemotron_h":
            return (), (), ("fused_moe_forward's default branch is gather_qmm; the kernel path "
                            "needs MLXCEL_FUSED_MOE_RELU2 and moe_fc1_relu2 (#2069)")
        return (), (), f"{mt} does not call forward_fused_kernel"
    if unit == "2067":
        if mt in ("granitemoehybrid", "nemotron_h", "falcon_h1", "plamo2"):
            return ("ssm_step",), ("ssm_step",), "ssm_kernel_available() gates the decode step"
        return (), (), "no Mamba2 layer"
    if unit == "2068":
        return (), (), "the bench decodes into a dense KVCache; the paged path is not taken"
    raise ValueError(unit)


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------
def summarize(trace: pathlib.Path, log_path: pathlib.Path, plain_log: pathlib.Path | None,
              model_dir: pathlib.Path | None, name: str, out_dir: pathlib.Path,
              temperature: float = 0.0) -> dict:
    log = parse_log(log_path.read_text(errors="replace"))
    dispatches = read_trace(trace)
    clock = pick_clock(log, dispatches)
    lo = log.marks["decode_start"][clock]
    hi = log.marks["measured_end"][clock]
    decode = [d for d in dispatches if lo <= d.start <= hi]
    if not decode:
        raise ValueError("no dispatch inside the decode window")
    before = [d for d in dispatches if d.end <= lo]
    straddling = sum(1 for d in dispatches if d.start < lo < d.end)
    tokens = log.generated or 0
    gpu_sum = sum(d.dur for d in decode)
    gpu_busy = busy_ns(decode, lo, hi)
    wall = hi - lo

    config = {}
    if model_dir is not None and (model_dir / "config.json").exists():
        config = json.loads((model_dir / "config.json").read_text())

    roles = assign_roles(decode)
    per_kernel: dict[str, list] = {}
    for d, r in zip(decode, roles):
        row = per_kernel.setdefault(d.name, [0, 0, defaultdict(int)])
        row[0] += 1
        row[1] += d.dur
        row[2][r or ""] += d.dur
    role_ns: dict[str, int] = defaultdict(int)
    role_calls: dict[str, int] = defaultdict(int)
    class_ns: dict[str, int] = defaultdict(int)
    for d, r in zip(decode, roles):
        role_ns[r or "unattributed"] += d.dur
        role_calls[r or "unattributed"] += 1
        class_ns[classify(d.name)] += d.dur

    def share(ns: int) -> float:
        return round(100.0 * ns / gpu_sum, 2)

    unit_of_role = {r: u for u, rs in UNIT_ROLES.items() for r in rs}
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{name}_decode_kernels.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kernel", "class", "roles", "port_units", "calls", "calls_per_token",
                    "total_ns", "avg_ns", "share_pct"])
        for kname, (calls, ns, by_role) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1]):
            rs = sorted(r for r in by_role if r)
            units = sorted({unit_of_role[r] for r in rs if r in unit_of_role})
            w.writerow([kname, classify(kname), ";".join(rs), ";".join(units), calls,
                        round(calls / tokens, 2) if tokens else "", ns, round(ns / calls),
                        f"{100.0 * ns / gpu_sum:.3f}"])

    units = {}
    for unit, rs in UNIT_ROLES.items():
        default_roles, optin_roles, note = reach(unit, config, temperature > 0)
        units[unit] = {
            "port": PORT_UNITS[unit],
            "fallback_share_pct": share(sum(role_ns[r] for r in rs)),
            "fallback_dispatches_per_token": round(sum(role_calls[r] for r in rs) / tokens, 1)
            if tokens else None,
            "reached_default_share_pct": share(sum(role_ns[r] for r in default_roles)),
            "reached_default_dispatches_per_token":
                round(sum(role_calls[r] for r in default_roles) / tokens, 1) if tokens else None,
            "reached_optin_share_pct": share(sum(role_ns[r] for r in optin_roles)),
            "note": note,
        }

    plain = parse_log(plain_log.read_text(errors="replace")) if plain_log else None
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:10]
    summary = {
        "name": name,
        "model_type": config.get("model_type"),
        "temperature": temperature,
        "clock": clock,
        "prompt_tokens": log.prompt,
        "generated_tokens": tokens,
        "profiled_decode_tok_s": log.decode_tok_s,
        "profiled_prefill_tok_s": log.prefill_tok_s,
        "plain_decode_tok_s": plain.decode_tok_s if plain else None,
        "plain_prefill_tok_s": plain.prefill_tok_s if plain else None,
        "profiler_decode_slowdown_pct": round(100.0 * (plain.decode_tok_s / log.decode_tok_s - 1), 2)
        if plain and plain.decode_tok_s and log.decode_tok_s else None,
        "decode_dispatches": len(decode),
        "dispatches_per_token": round(len(decode) / tokens, 1) if tokens else None,
        "decode_wall_ms": round(wall / 1e6, 3),
        "decode_gpu_sum_ms": round(gpu_sum / 1e6, 3),
        "decode_gpu_busy_ms": round(gpu_busy / 1e6, 3),
        "gpu_ms_per_token": round(gpu_busy / 1e6 / tokens, 4) if tokens else None,
        "host_gap_ms_per_token": round((wall - gpu_busy) / 1e6 / tokens, 4) if tokens else None,
        "host_gap_pct_of_wall": round(100.0 * (wall - gpu_busy) / wall, 2),
        # The tracer adds host time per dispatch but barely changes kernel
        # durations, so the unprofiled run's wall time per token minus the
        # traced GPU time is the better host-gap estimate for dispatch-heavy
        # models. Same denominator as tok/s (generated tokens).
        "plain_wall_ms_per_token": round(1000.0 / plain.decode_tok_s, 4)
        if plain and plain.decode_tok_s else None,
        "plain_host_gap_ms_per_token_est":
            round(1000.0 / plain.decode_tok_s - gpu_busy / 1e6 / tokens, 4)
        if plain and plain.decode_tok_s and tokens else None,
        "checks": {
            "kernels_straddling_decode_start": straddling,
            "idle_gap_before_first_decode_dispatch_us":
                round((decode[0].start - before[-1].end) / 1e3, 1) if before else None,
            "last_dispatch_before_decode": before[-1].name[:120] if before else None,
        },
        "top_kernels": [
            {"kernel": k, "class": classify(k), "calls_per_token": round(c / tokens, 2) if tokens else None,
             "share_pct": share(ns)}
            for k, (c, ns, _) in top
        ],
        "class_share_pct": {c: share(ns) for c, ns in sorted(class_ns.items(), key=lambda kv: -kv[1])},
        "role_share_pct": {r: share(ns) for r, ns in sorted(role_ns.items(), key=lambda kv: -kv[1])},
        "role_dispatches_per_token": {r: round(c / tokens, 1) for r, c in role_calls.items()}
        if tokens else {},
        "port_units": units,
    }
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def short_kernel(name: str) -> str:
    """`binary_vv<Add, float>` style: kernel, op functor and first dtype."""
    if name.startswith("Cijk_"):
        return "Cijk_* (Tensile GEMM)"
    m = re.search(r"::([A-Za-z0-9_]+)(<[^(]*>)?\(", name)
    base = m.group(1) if m else name.split("(")[0]
    if base[:1].isupper() and ("_contiguous" in base or "_strided" in base):
        return "compiled " + base[:40] + "..."
    parts = []
    if m and m.group(2):
        args = m.group(2)[1:-1]
        if op := re.search(r"rocm::(\w+)(<[^>]*>)?,", args):
            parts.append(op.group(1))
            args = args[op.end():]
        if dt := re.search(r"(hip_bfloat16|__half|float|int|unsigned int|bool)", args):
            parts.append({"hip_bfloat16": "bf16", "__half": "f16", "float": "f32",
                          "unsigned int": "u32"}.get(dt.group(1), dt.group(1)))
    return f"{base}<{', '.join(parts)}>" if parts else base


def report(out_dir: pathlib.Path) -> str:
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_summary.json"))]
    lines = ["| Run | plain tok/s | profiled tok/s | profiler cost | dispatches/token | "
             "GPU ms/token | host gap ms/token | host gap % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in rows:
        lines.append(
            f"| `{s['name']}` | {s['plain_decode_tok_s'] or '-'} | {s['profiled_decode_tok_s']} | "
            f"{'-' if s['profiler_decode_slowdown_pct'] is None else str(s['profiler_decode_slowdown_pct']) + '%'} | "
            f"{s['dispatches_per_token']} | {s['gpu_ms_per_token']} | {s['host_gap_ms_per_token']} | "
            f"{s['host_gap_pct_of_wall']} |")
    lines += ["", "| Run | " + " | ".join(f"#{u}" for u in UNIT_ROLES) + " |",
              "|---|" + "---:|" * len(UNIT_ROLES)]
    for s in rows:
        cells = []
        for u in UNIT_ROLES:
            p = s["port_units"][u]
            cells.append(f"{p['reached_default_share_pct']} ({p['fallback_share_pct']})")
        lines.append(f"| `{s['name']}` | " + " | ".join(cells) + " |")
    for s in rows:
        lines += ["", f"Top kernels, `{s['name']}` (share of decode GPU time, calls per token):", ""]
        for k in s["top_kernels"]:
            lines.append(f"- `{short_kernel(k['kernel'])}` {k['share_pct']}%, {k['calls_per_token']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize", help="cut and attribute one profiled run")
    s.add_argument("--trace", type=pathlib.Path, required=True)
    s.add_argument("--log", type=pathlib.Path, required=True)
    s.add_argument("--plain-log", type=pathlib.Path)
    s.add_argument("--model-dir", type=pathlib.Path)
    s.add_argument("--name", required=True)
    s.add_argument("--out-dir", type=pathlib.Path, required=True)
    s.add_argument("--temperature", type=float, default=0.0,
                   help="the run's sampling temperature (0 = greedy)")
    r = sub.add_parser("report", help="render the summaries in a directory as Markdown")
    r.add_argument("out_dir", type=pathlib.Path)
    args = ap.parse_args(argv)
    if args.cmd == "summarize":
        s = summarize(args.trace, args.log, args.plain_log, args.model_dir, args.name,
                      args.out_dir, args.temperature)
        print(f"{s['name']}: {s['decode_dispatches']} decode dispatches, "
              f"{s['gpu_ms_per_token']} GPU ms/token, {s['host_gap_ms_per_token']} host gap ms/token")
    else:
        sys.stdout.write(report(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
