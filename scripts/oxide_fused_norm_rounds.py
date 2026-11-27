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
"""Interleaved end-to-end rounds for the cuda-oxide fused-norm experiment (#2834).

`engine_bench_rounds.py` rotates arms that differ by flags. The three
implementations of the fused residual join differ by process environment, and
each run has to prove which one served it, so this driver owns that:

  graph       MLXCEL_FUSED_ADD_RMSNORM unset            (add + fast_rms_norm)
  fused       MLXCEL_FUSED_ADD_RMSNORM=1                (existing CUDA kernel)
  oxide       fused, plus MLXCEL_CUDA_OXIDE_FUSED_NORM=1 (cuda-oxide node)
  fused-null  the fused arm again: the method's noise floor

Every arm is a fresh process of the same binary
(`examples/oxide_fused_norm_checkpoint_bench`, built with
MLXCEL_BUILD_CUDA_OXIDE_FUSED_NORM=1) run under `gpu-lock`, and is started
with `--expect-route`, so a run whose route counters do not match its arm
fails instead of becoming a data point. The run order rotates by one position
each round. Cells are (path, batch): cli B=1, server B=1, server B=N.

  run        measure and append one CSV row per (arm, round, cell, prompt)
  summarize  medians, paired per-round deltas against the fused arm, the null
             floor, and the frozen gate rule of the #2834 protocol

The gate rule (docs/mlxcelverse/cuda-oxide-benchmark-evidence.md, Protocol):
a cell and metric is a flagged regression when the oxide arm's median paired
delta against fused is worse than 2 %, it is worse in at least all but one
round, and it exceeds the cell's null floor (the median absolute paired delta
of fused-null against fused). A cell whose null floor is itself above 2 %
cannot resolve the threshold and is reported as unresolved.
"""
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MARK = "[oxide-ckpt-bench] "
ARMS = {
    "graph": ({}, "graph"),
    "fused": ({"MLXCEL_FUSED_ADD_RMSNORM": "1"}, "fused"),
    "oxide": ({"MLXCEL_FUSED_ADD_RMSNORM": "1", "MLXCEL_CUDA_OXIDE_FUSED_NORM": "1"}, "oxide"),
    "fused-null": ({"MLXCEL_FUSED_ADD_RMSNORM": "1"}, "fused"),
}
SWITCHES = ("MLXCEL_FUSED_ADD_RMSNORM", "MLXCEL_CUDA_OXIDE_FUSED_NORM",
            "MLXCEL_CUDA_OXIDE_FUSED_NORM_DIR")
# metric -> True when a larger value is better
METRICS = {"ttft_ms": False, "decode_tok_s": True}
THRESHOLD = 0.02
FIELDS = ["arm", "round", "order", "started_unix", "ci_job_running", "path", "batch",
          "prompt_target_len", "prompt_tokens", "max_tokens", "generated_tokens", "ttft_ms",
          "prefill_tok_s", "decode_ms", "decode_tok_s", "peak_memory_bytes",
          "active_memory_bytes", "route", "adapter_built", "installed", "oxide_routes",
          "native_routes", "launches", "last_fallback", "env_fused", "env_oxide",
          "env_cuda_graphs", "tokens_sha256_16", "first_row_tokens"]


def ci_job_running():
    """Whether a GitHub Actions job is executing on this host right now."""
    # -x: match the process name, not any command line that mentions it.
    out = subprocess.run(["pgrep", "-x", "Runner.Worker"], capture_output=True, text=True)
    return bool(out.stdout.strip())


def wait_for_runner(max_wait_s):
    waited = 0
    while ci_job_running() and waited < max_wait_s:
        time.sleep(10)
        waited += 10
    return waited


def run_one(a, arm, cell, round_no, order):
    env_add, expect = ARMS[arm]
    env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    env.update(env_add)
    path, batch = cell
    cmd = ["gpu-lock", "run", "--tag", a.tag, "--", a.bin, "--model", a.model, "--path", path,
           "--batch", str(batch), "--prompt-tokens", ",".join(str(p) for p in a.prompt_tokens),
           "--max-tokens", str(a.max_tokens), "--warmup-tokens", str(a.warmup_tokens),
           "--label", arm, "--expect-route", expect]
    waited = wait_for_runner(a.runner_wait)
    busy = ci_job_running()
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=a.run_timeout)
    rows = []
    for line in proc.stdout.splitlines():
        if line.startswith(MARK):
            rec = json.loads(line[len(MARK):])
            rec.update(arm=arm, round=round_no, order=order, started_unix=f"{started:.0f}",
                       ci_job_running=str(busy or ci_job_running()).lower())
            rows.append(rec)
    if proc.returncode != 0 or len(rows) != len(a.prompt_tokens):
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise SystemExit(f"arm {arm} round {round_no} cell {path}/B{batch} failed "
                         f"(exit {proc.returncode}, {len(rows)} records); the run is invalid, "
                         "not a data point")
    if waited:
        print(f"  (waited {waited} s for a CI job on this host)")
    return rows


def cmd_run(a):
    cells = [("cli", 1), ("server", 1), ("server", a.batch)]
    names = list(ARMS)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore",
                                lineterminator="\n")
        if new:
            writer.writeheader()
        for r in range(a.first_round, a.first_round + a.rounds):
            shift = r % len(names)
            schedule = names[shift:] + names[:shift]
            for cell in cells:
                for order, arm in enumerate(schedule):
                    for rec in run_one(a, arm, cell, r, order):
                        writer.writerow(rec)
                        fh.flush()
                        print(f"round {r} {arm:<10} {rec['path']:<6} B{rec['batch']} prompt "
                              f"{rec['prompt_target_len']:>5} route {rec['route']:<5} "
                              f"TTFT {rec['ttft_ms']:.2f} ms  decode {rec['decode_tok_s']:.2f} tok/s  "
                              f"peak {rec['peak_memory_bytes'] / 2**30:.2f} GiB")


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def paired(rows, arm, ref, cell, metric):
    """Per-round relative deltas of `arm` against `ref`, positive = better."""
    higher_better = METRICS[metric]
    by = {}
    for row in rows:
        key = (row["path"], row["batch"], row["prompt_target_len"])
        if key == cell and row["arm"] in (arm, ref):
            by.setdefault(row["round"], {})[row["arm"]] = float(row[metric])
    deltas = []
    for r in sorted(by, key=int):
        if arm in by[r] and ref in by[r] and by[r][ref] > 0:
            d = (by[r][arm] - by[r][ref]) / by[r][ref]
            deltas.append(d if higher_better else -d)
    return deltas


def cmd_summarize(a):
    rows = load(a.csv)
    cells = sorted({(r["path"], r["batch"], r["prompt_target_len"]) for r in rows},
                   key=lambda c: (c[0], int(c[1]), int(c[2])))
    arms = [n for n in ARMS if any(r["arm"] == n for r in rows)]
    out_rows = []
    print("| cell | metric | " + " | ".join(f"{n} median (min..max)" for n in arms)
          + " | oxide vs fused, paired median | rounds worse | null floor | verdict |")
    print("|" + " --- |" * (len(arms) + 6))
    for cell in cells:
        for metric, higher_better in METRICS.items():
            med = {}
            cells_txt = []
            for arm in arms:
                vals = [float(r[metric]) for r in rows
                        if r["arm"] == arm and (r["path"], r["batch"], r["prompt_target_len"]) == cell]
                med[arm] = statistics.median(vals) if vals else float("nan")
                cells_txt.append(f"{med[arm]:.2f} ({min(vals):.2f}..{max(vals):.2f})" if vals else "n/a")
            d_oxide = paired(rows, "oxide", "fused", cell, metric)
            d_null = paired(rows, "fused-null", "fused", cell, metric)
            d_graph = paired(rows, "graph", "fused", cell, metric)
            m_oxide = statistics.median(d_oxide) if d_oxide else float("nan")
            floor = statistics.median(abs(d) for d in d_null) if d_null else float("nan")
            worse = sum(1 for d in d_oxide if d < 0)
            better = sum(1 for d in d_oxide if d > 0)
            n = len(d_oxide)
            # A cell without a complete set of rounds for every arm, or with a
            # non-finite value, decides nothing: a NaN floor or median compares
            # false against every threshold and would otherwise read as a pass.
            rounds_in_cell = {r["round"] for r in rows
                              if (r["path"], r["batch"], r["prompt_target_len"]) == cell}
            complete = (n > 0 and n == len(d_null) == len(rounds_in_cell)
                        and all(map(math.isfinite, d_oxide + d_null)))
            if not complete:
                verdict = "unresolved (incomplete or non-finite pairs)"
            elif floor > THRESHOLD:
                verdict = "unresolved (null floor above 2 %)"
            elif m_oxide < -THRESHOLD and worse >= n - 1 and abs(m_oxide) > floor:
                verdict = "FLAGGED regression"
            elif m_oxide > floor and better >= n - 1:
                verdict = "improvement beyond the null floor"
            else:
                verdict = "no regression"
            name = f"{cell[0]} B={cell[1]} prompt {cell[2]}"
            print(f"| {name} | {metric} | " + " | ".join(cells_txt)
                  + f" | {m_oxide * 100:+.2f} % | {worse}/{n} | {floor * 100:.2f} % | {verdict} |")
            out_rows.append({
                "path": cell[0], "batch": cell[1], "prompt_target_len": cell[2], "metric": metric,
                **{f"median_{arm}": f"{med[arm]:.4f}" for arm in arms},
                "rounds": n,
                "oxide_vs_fused_paired_median_pct": f"{m_oxide * 100:.3f}",
                "oxide_vs_fused_paired_min_pct": f"{min(d_oxide) * 100:.3f}" if d_oxide else "",
                "oxide_vs_fused_paired_max_pct": f"{max(d_oxide) * 100:.3f}" if d_oxide else "",
                "oxide_rounds_worse": worse,
                "null_floor_median_abs_pct": f"{floor * 100:.3f}",
                "null_paired_median_pct": f"{statistics.median(d_null) * 100:.3f}" if d_null else "",
                "graph_vs_fused_paired_median_pct":
                    f"{statistics.median(d_graph) * 100:.3f}" if d_graph else "",
                "verdict": verdict,
            })
    print("\nPaired deltas are signed so that positive is better (lower TTFT, higher tok/s).")
    # Greedy output and peak memory per cell, across arms and rounds.
    print("\n| cell | arm | distinct greedy outputs across rounds | peak memory GiB (min..max) |")
    print("| --- | --- | --- | --- |")
    for cell in cells:
        for arm in arms:
            sel = [r for r in rows
                   if r["arm"] == arm and (r["path"], r["batch"], r["prompt_target_len"]) == cell]
            digests = sorted({r["tokens_sha256_16"] for r in sel})
            peaks = [int(r["peak_memory_bytes"]) / 2**30 for r in sel]
            peak_txt = f"{min(peaks):.3f}..{max(peaks):.3f}" if peaks else "unresolved (no rows)"
            digest_txt = f"{len(digests)}: {', '.join(digests)}" if digests else "unresolved (no rows)"
            print(f"| {cell[0]} B={cell[1]} prompt {cell[2]} | {arm} | {digest_txt} | {peak_txt} |")
    if a.out and out_rows:
        with open(a.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(out_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--bin", default=os.path.join(
        REPO, "target", "release", "examples", "oxide_fused_norm_checkpoint_bench"))
    run.add_argument("--model", required=True)
    run.add_argument("--out", required=True, help="CSV, one row per measured pass (appended)")
    run.add_argument("--rounds", type=int, default=7)
    run.add_argument("--first-round", type=int, default=0)
    run.add_argument("--batch", type=int, default=4)
    run.add_argument("--prompt-tokens", type=int, nargs="+", default=[128, 2048])
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--warmup-tokens", type=int, default=16)
    run.add_argument("--tag", default="issue-2834-e2e")
    run.add_argument("--run-timeout", type=float, default=900.0)
    run.add_argument("--runner-wait", type=int, default=1800,
                     help="seconds to wait for a CI job on this host before each run")
    run.set_defaults(func=cmd_run)
    summ = sub.add_parser("summarize")
    summ.add_argument("csv")
    summ.add_argument("--out", help="write the per-cell summary as CSV")
    summ.set_defaults(func=cmd_summarize)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
