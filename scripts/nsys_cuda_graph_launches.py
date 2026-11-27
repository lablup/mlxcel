#!/usr/bin/env python3
"""CUDA graphs per decode step, and what cuts them, from an nsys trace.

MLX's CUDA backend captures each eval into CUDA graphs and commits the graph
it is building when its op count passes `MLX_MAX_OPS_PER_BUFFER`, when the
summed `data_size()` of its inputs passes `MLX_MAX_MB_PER_BUFFER`, or when the
eval ends (`CommandEncoder::needs_commit`). On DGX Spark (sm_121) the defaults
are 20 ops and 25 "MB", in fact 25 << 20 elements, so a launch whose inputs
hold more than that cuts the graph on its own. This script counts the graphs a
decode step costs and says which kernels sit in the graphs that were cut short,
which is how lablup/mlxcel#3387 found the launches that pay for a whole
preallocated buffer they read a few rows of.

Take the trace with node-level graph tracing, so that every kernel a graph
runs is recorded with the correlation id of its `cudaGraphLaunch`:

    nsys launch --session-new=s --trace=cuda --cuda-graph-trace=node \\
        mlxcel-server -m <model> --parallel 4 -c 65536 --no-prompt-cache &
    # warm up, then
    nsys start --session=s --sample=none --cpuctxsw=none -o trace
    # send the requests to measure, then
    nsys stop --session=s
    nsys export --type sqlite -o trace.sqlite trace.nsys-rep

Stop the collection with `nsys stop` while the server is alive. A profile
ended by killing the server loses the events still in CUPTI's buffers.

A decode step is the interval between two launches of the marker kernel, by
default the greedy argmax (`arg_reduce`), which a temperature-0 decode runs
once per scheduler step. Collect a stretch of pure decode, or name a kernel
that only prefill runs with `--prefill-kernels` to drop the steps that hold it
(`qmm_sm80` for a single 4-bit request; a batched decode step runs it too).
With `--row-kernel NAME --per-row N` the steps are grouped by batch size, the
count of that kernel in the step divided by N (for Qwen3 0.6B, `rope_single`
runs twice per layer per row, so N is 56); steps whose count is not a whole
multiple, and steps next to a change of batch size, are dropped. A graph
belongs to the step in which its first kernel starts.

For each group it prints the graphs and kernels per step, then the graphs of
at most `--ops` nodes (kernels, memcpys and memsets) per step and the kernels
those short graphs hold. The op budget cuts a graph only once it holds more
than `--ops` nodes, so a short graph was cut by the byte budget or by the end
of an eval. That reading is a guide, not a proof: a captured child graph is
one node however many kernels it runs. Run the same workload again with
`MLX_MAX_MB_PER_BUFFER=1000`; the short graphs that disappear were byte-budget
cuts.

Usage:
  scripts/nsys_cuda_graph_launches.py trace.sqlite
  scripts/nsys_cuda_graph_launches.py trace.sqlite --row-kernel rope_single --per-row 56
"""
from __future__ import annotations

import argparse
import bisect
import sqlite3
import sys
from collections import Counter, defaultdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("sqlite", help="`nsys export --type sqlite` of a --cuda-graph-trace=node trace")
    ap.add_argument("--marker", default="arg_reduce", help="kernel name part that runs once per decode step")
    ap.add_argument("--row-kernel", help="kernel name part that runs --per-row times per row per step")
    ap.add_argument("--per-row", type=int, default=0)
    ap.add_argument("--prefill-kernels", default="",
                    help="comma-separated kernel name parts; a step that holds one is dropped as prefill work")
    ap.add_argument("--ops", type=int, default=20, help="the op budget (MLX_MAX_OPS_PER_BUFFER)")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    if bool(args.row_kernel) != bool(args.per_row):
        ap.error("--row-kernel and --per-row go together")

    db = sqlite3.connect(args.sqlite)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    kernels = [(start, names.get(name, "?"), corr) for start, name, corr in db.execute(
        "SELECT start, shortName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    launch_ids = [i for i, v in names.items() if v.startswith("cudaGraphLaunch")]
    if not kernels or not launch_ids:
        print("no kernels or no cudaGraphLaunch calls in the trace", file=sys.stderr)
        return 1
    launched = {row[0] for row in db.execute(
        "SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN (%s)"
        % ",".join(map(str, launch_ids)))}

    # Memcpy and memset nodes are graph nodes too: a graph the op budget cut
    # can hold fewer than `--ops` kernels when some of its nodes are copies.
    other_nodes: Counter = Counter()
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            for (corr,) in db.execute(f"SELECT correlationId FROM {table}"):
                other_nodes[corr] += 1

    marks = [start for start, name, _ in kernels if args.marker in name]
    if len(marks) < 3:
        print(f"fewer than three launches of a kernel named *{args.marker}*", file=sys.stderr)
        return 1
    prefill_parts = [p for p in args.prefill_kernels.split(",") if p]

    # Kernels and graphs of each step. A graph is the kernels that share the
    # correlation id of one cudaGraphLaunch; it goes to the step its first
    # kernel starts in.
    step_kernels: dict[int, list[str]] = defaultdict(list)
    graph_kernels: dict[int, list[str]] = defaultdict(list)
    graph_step: dict[int, int] = {}
    for start, name, corr in kernels:
        step = bisect.bisect_right(marks, start) - 1
        if 0 <= step < len(marks) - 1:
            step_kernels[step].append(name)
        if corr in launched:
            graph_kernels[corr].append(name)
            graph_step.setdefault(corr, step)
    step_graphs: dict[int, list[int]] = defaultdict(list)
    for corr, step in graph_step.items():
        if 0 <= step < len(marks) - 1:
            step_graphs[step].append(corr)

    def group_of(step: int):
        ks = step_kernels.get(step, [])
        if any(part in name for name in ks for part in prefill_parts):
            return None
        if not args.row_kernel:
            return "all"
        rows = sum(1 for name in ks if args.row_kernel in name)
        if rows == 0 or rows % args.per_row:
            return None
        return rows // args.per_row

    groups = [group_of(step) for step in range(len(marks) - 1)]
    kept: dict[object, list[int]] = defaultdict(list)
    for step, group in enumerate(groups):
        if group is None:
            continue
        before = groups[step - 1] if step > 0 else group
        after = groups[step + 1] if step + 1 < len(groups) else group
        if before == group and after == group:
            kept[group].append(step)

    print(f"steps={len(groups)} kept={sum(len(v) for v in kept.values())}")
    for group in sorted(kept, key=str):
        steps = kept[group]
        n = len(steps)
        counts = sorted(len(step_graphs.get(s, [])) for s in steps)
        short = Counter()
        holds = Counter()
        for s in steps:
            for corr in step_graphs.get(s, []):
                ks = graph_kernels[corr]
                if len(ks) + other_nodes[corr] <= args.ops:
                    short[len(ks)] += 1
                    # `noop` is the single root MLX gives a graph on sm_121.
                    for name in {k for k in ks if k != "noop"}:
                        holds[name] += 1
        label = "all steps" if group == "all" else f"batch {group}"
        print(f"{label}: steps={n} graphs/step mean={sum(counts) / n:.2f} min={counts[0]} "
              f"median={counts[n // 2]} max={counts[-1]} "
              f"kernels/step={sum(len(step_kernels[s]) for s in steps) / n:.1f} "
              f"short graphs/step={sum(short.values()) / n:.2f}")
        for name, count in holds.most_common(args.top):
            print(f"    {count / n:8.2f} short graphs/step hold  {name[:96]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
