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
"""Interleaved A/B rounds for mlxcel-bench-engine, with a null arm.

Each round runs every arm once, then runs the first arm a second time as the
null arm (every arm with --null-every-arm). The run order rotates by one
position each round, so no arm always runs first or right after a given arm
and a drift across the round (thermals, allocator growth) does not land on one
arm. The first arm and its repeat use the same binary and the same flags, so
their paired delta is the method's noise floor; an A/B delta inside the null
spread is unresolved. Every arm is a fresh process, so
process-wide settings (MLXCEL_PREFILL_CHUNK on the CLI path) can differ per
arm, and no arm inherits another's allocator or kernel caches.

This is the protocol ADR 0007 uses for its regression threshold and for the
performance rows of its decision table (prefill chunk 2048 vs 512, dense vs
paged single-sequence storage). Example, the prefill-chunk A/B on the CLI path
at long context:

  scripts/engine_bench_rounds.py --bin target/release/mlxcel-bench-engine \\
      --model models/mlx/qwen3-1.7b-4bit --prompt-tokens 8192 --rounds 5 \\
      --arm chunk2048="--path cli --prefill-chunk 2048" \\
      --arm chunk512="--path cli --prefill-chunk 512" \\
      --out chunk-cli-qwen3.jsonl

--hostgate waits for a quiet host before every arm with the #1820 gate from
docs/benchmark_results/data/sdpa-plan-bucket-gb10-2026-09-12/harness/.

Every benchmark process runs under --run-timeout seconds (default 3600, 0
disables it) so a hung model load or decode cannot stall a round forever. An
arm name ending in -null is reserved for the generated null arms and is
rejected.
"""
import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HOSTGATE_DIR = os.path.join(REPO, "docs", "benchmark_results", "data",
                            "sdpa-plan-bucket-gb10-2026-09-12", "harness")
MARK = "[engine-bench] {"
METRICS = ("ttft_ms", "decode_tok_s")
NULL_SUFFIX = "-null"
DEFAULT_RUN_TIMEOUT_S = 3600.0


def parse_arm(spec):
    name, sep, flags = spec.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"--arm expects NAME=\"FLAGS\", got {spec!r}")
    return name, shlex.split(flags)


def check_arm_names(names):
    """Reject duplicate arm names and names that collide with a generated null
    arm. A null arm is named `<source>-null`, so a user arm that ends in -null
    would share its records and its summary row with it."""
    if len(set(names)) != len(names):
        raise SystemExit("arm names must be unique")
    reserved = [n for n in names if n.endswith(NULL_SUFFIX)]
    if reserved:
        raise SystemExit(
            f"arm name(s) {', '.join(reserved)} end in {NULL_SUFFIX!r}, which is reserved for "
            "the generated null arms; rename them")


def run_arm(a, name, flags, round_no, gate):
    gate_info = {}
    if gate is not None:
        gate_info["gate_wait_s"] = gate.wait_quiet(log=sys.stderr)
        gate_info["ci_job_running"] = gate.ci_job_running()
    cmd = [a.bin, "--model", a.model, "--max-tokens", str(a.max_tokens),
           "--prompt-tokens", ",".join(str(p) for p in a.prompt_tokens),
           "--label", name] + flags + a.extra
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=a.run_timeout or None)
    except subprocess.TimeoutExpired as exc:
        for stream in (exc.stdout, exc.stderr):
            if stream:
                sys.stderr.write(stream if isinstance(stream, str) else stream.decode(errors="replace"))
        raise SystemExit(f"arm {name} round {round_no} timed out after {a.run_timeout:g} s "
                         "(raise --run-timeout if the run is expected to take longer)")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"arm {name} round {round_no} failed with exit {proc.returncode}")
    records = []
    for line in proc.stdout.splitlines():
        if line.startswith(MARK):
            rec = json.loads(line[len("[engine-bench] "):])
            rec.update(arm=name, round=round_no, started=started, **gate_info)
            records.append(rec)
    if not records:
        raise SystemExit(f"arm {name} round {round_no} printed no [engine-bench] records")
    return records


def key(rec):
    return (rec["path"], rec["prompt_target_len"])


def summarize(records, arms, nulls):
    """Print medians per arm, A/B deltas against the first arm, and each null
    arm's delta against the arm it repeats. `nulls` maps null name to source."""
    base = arms[0]
    by = {}
    for rec in records:
        by.setdefault((rec["arm"], key(rec)), {})[rec["round"]] = rec
    keys = sorted({key(r) for r in records})
    print("\narm medians (min..max) per path and prompt length")
    for k in keys:
        for arm in arms + list(nulls):
            rows = list(by.get((arm, k), {}).values())
            if not rows:
                continue
            cells = []
            for m in METRICS:
                vals = [r[m] for r in rows]
                cells.append(f"{m} {statistics.median(vals):.2f} ({min(vals):.2f}..{max(vals):.2f})")
            print(f"  {k[0]:<6} prompt {k[1]:>5}  {arm:<18} n={len(rows)}  " + "  ".join(cells))
    pairs = [(arm, base) for arm in arms[1:]] + list(nulls.items())
    print("\npaired per-round deltas, arm vs reference (positive = higher than the reference)")
    for k in keys:
        for arm, ref in pairs:
            ref_rounds, other = by.get((ref, k), {}), by.get((arm, k), {})
            for m in METRICS:
                deltas = [(other[r][m] - ref_rounds[r][m]) / ref_rounds[r][m] * 100.0
                          for r in sorted(ref_rounds) if r in other and ref_rounds[r][m]]
                if not deltas:
                    continue
                print(f"  {k[0]:<6} prompt {k[1]:>5}  {arm:<18} vs {ref:<14} {m:<12} "
                      f"median {statistics.median(deltas):+.2f}%  range {min(deltas):+.2f}%..{max(deltas):+.2f}%")
    print("\nthe *-null rows are the noise floor: an A/B delta whose range overlaps the null "
          "range is unresolved")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.join(REPO, "target", "release", "mlxcel-bench-engine"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", action="append", type=parse_arm, required=True,
                    help='NAME="FLAGS" passed to the benchmark; repeat for each arm')
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[256, 8192])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True, help="JSONL file, one record per measurement")
    ap.add_argument("--hostgate", action="store_true", help="wait for a quiet host before each arm")
    ap.add_argument("--null-every-arm", action="store_true",
                    help="repeat every arm as its own null arm, not only the first")
    ap.add_argument("--run-timeout", type=float, default=DEFAULT_RUN_TIMEOUT_S, metavar="SECONDS",
                    help="kill and fail an arm whose benchmark process runs longer than this "
                         f"(default {DEFAULT_RUN_TIMEOUT_S:g}, 0 disables)")
    ap.add_argument("--extra", default="", help='flags passed to every arm, e.g. --extra="--ignore-eos"')
    a = ap.parse_args()
    a.extra = shlex.split(a.extra)
    if a.run_timeout < 0:
        raise SystemExit("--run-timeout must be 0 or positive")
    names = [n for n, _ in a.arm]
    check_arm_names(names)
    gate = None
    if a.hostgate:
        sys.path.insert(0, HOSTGATE_DIR)
        import hostgate as gate  # noqa: E402
    flags_of = dict(a.arm)
    null_sources = names if a.null_every_arm else names[:1]
    nulls = {f"{n}{NULL_SUFFIX}": n for n in null_sources}
    records = []
    with open(a.out, "a") as out:
        header = {"kind": "header", "model": a.model, "bin": a.bin, "arms": dict(a.arm),
                  "prompt_tokens": a.prompt_tokens, "max_tokens": a.max_tokens,
                  "rounds": a.rounds, "host": os.uname().nodename, "time": time.time(),
                  "sdpa_deterministic": os.environ.get("MLXCEL_SDPA_DETERMINISTIC")}
        out.write(json.dumps(header) + "\n")
        for r in range(a.rounds):
            schedule = list(a.arm) + [(null, flags_of[src]) for null, src in nulls.items()]
            shift = r % len(schedule)
            schedule = schedule[shift:] + schedule[:shift]
            for name, flags in schedule:
                for rec in run_arm(a, name, flags, r, gate):
                    records.append(rec)
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"round {r} {name:<18} {rec['path']:<6} prompt {rec['prompt_target_len']:>5} "
                          f"TTFT {rec['ttft_ms']:.2f} ms  decode {rec['decode_tok_s']:.2f} tok/s")
    summarize(records, names, nulls)


if __name__ == "__main__":
    main()
