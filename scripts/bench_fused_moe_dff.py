#!/usr/bin/env python3
"""Fused decode-MoE vs gather_qmm decode harness with interleaved arms (issue #2540).

Primary:   mlxcel-bench-decode, prompt "Hello, how are you today?", 20 warmup
           tokens, 100 measured, one process per run.
Secondary: mlxcel generate --temp 0, one process per run, prompt/length cells.

Arms: fused (MLXCEL_FUSED_MOE_MAX_DFF=20000), gather (MLXCEL_FUSED_MOE=0),
null (fused again). One discarded warm-up run per arm, then N rounds with the
arm order rotating every round. Every run goes through `gpu-lock run`. Before
every run the harness waits until no cargo/rustc/nvcc/cc1plus process is
active. A measured run that prints the JIT notice is discarded and rerun.

Every run's full output is kept under <out>/raw/, and one CSV row per run
(discarded ones included, flagged) is appended to <out>/runs.csv. Point --out
at a directory outside the checkout or under the ignored bench-results/; only
the final runs.csv belongs in docs/benchmark_results/data/. Summarize with
scripts/bench_fused_moe_dff_report.py.

Example (primary harness, one model):
  python3 scripts/bench_fused_moe_dff.py --harness primary \
      --bin-dir target/release --model <dir> --name phi-3.5-moe-4bit \
      --cell hello-n100 --rounds 5 --out bench-results/fused-moe-dff
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

JIT_NOTICE = "Compiling CUDA kernels"
ARMS = {
    "fused": {"MLXCEL_FUSED_MOE_MAX_DFF": "20000"},
    "gather": {"MLXCEL_FUSED_MOE": "0"},
    "null": {"MLXCEL_FUSED_MOE_MAX_DFF": "20000"},
}
ARM_ORDER = ["fused", "gather", "null"]
BUSY = re.compile(r"\b(cargo|rustc|nvcc|cc1plus|ptxas)\b")

RE_BENCH_DECODE = re.compile(r"^\s*Decode:\s+([\d.]+) ms \(([\d.]+) tok/s\)", re.M)
RE_BENCH_PREFILL = re.compile(r"^\s*Prefill:\s+([\d.]+) ms \(([\d.]+) tok/s\)", re.M)
RE_BENCH_GEN = re.compile(r"^\s*Generated tokens:\s+(\d+)", re.M)
RE_BENCH_PROMPT = re.compile(r"^\s*Prompt tokens:\s+(\d+)", re.M)
RE_GEN = re.compile(r"\[Generated (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s\]")
RE_GEN_PROMPT = re.compile(r"[Pp]rompt[^\n]*?(\d+) tokens")


def busy_processes():
    out = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True).stdout
    return sorted({line.strip() for line in out.splitlines() if BUSY.search(line)})


def wait_quiet(log):
    waited = 0
    while True:
        busy = busy_processes()
        if not busy:
            return waited
        if waited % 60 == 0:
            log(f"host busy ({','.join(busy)}); waiting")
        time.sleep(5)
        waited += 5


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


def run_once(cmd, env_extra, raw_path, tag):
    env = dict(os.environ)
    for key in ("MLXCEL_FUSED_MOE", "MLXCEL_FUSED_MOE_MAX_DFF"):
        env.pop(key, None)
    env.update(env_extra)
    start = time.time()
    proc = subprocess.run(
        ["gpu-lock", "run", "--tag", tag, "--"] + cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.time() - start
    text = proc.stdout + "\n--- stderr ---\n" + proc.stderr
    with open(raw_path, "w") as f:
        f.write("# cmd: " + " ".join(cmd) + "\n# env: " + repr(env_extra) + "\n" + text)
    return proc.returncode, text, wall


def parse(harness, text):
    row = {"rate": "", "decode_ms": "", "gen_tokens": "", "prompt_tokens": "", "prefill_ms": ""}
    if harness == "primary":
        m = RE_BENCH_DECODE.search(text)
        if m:
            row["decode_ms"], row["rate"] = m.group(1), m.group(2)
        m = RE_BENCH_GEN.search(text)
        if m:
            row["gen_tokens"] = m.group(1)
        m = RE_BENCH_PROMPT.search(text)
        if m:
            row["prompt_tokens"] = m.group(1)
        m = RE_BENCH_PREFILL.search(text)
        if m:
            row["prefill_ms"] = m.group(1)
    else:
        m = RE_GEN.search(text)
        if m:
            row["gen_tokens"] = m.group(1)
            row["decode_ms"] = f"{float(m.group(2)) * 1000:.0f}"
            row["rate"] = m.group(3)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=["primary", "secondary"], required=True)
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--model", required=True, help="model directory")
    ap.add_argument("--name", required=True, help="model label for the CSV")
    ap.add_argument("--cell", default="default", help="label of the prompt/length cell")
    ap.add_argument("--prompt", default="Hello, how are you today?")
    ap.add_argument("--prompt-file")
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--warmup-tokens", type=int, default=20)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--cpu-list", help="pin every run to these CPUs with taskset -c")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    logf = open(os.path.join(args.out, "harness.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} [{args.name}/{args.cell}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()

    if args.harness == "primary":
        cmd = [
            os.path.join(args.bin_dir, "mlxcel-bench-decode"),
            "-m", args.model, "-p", prompt,
            "-n", str(args.max_tokens), "--warmup-tokens", str(args.warmup_tokens),
        ]
        if args.ignore_eos:
            cmd.append("--ignore-eos")
    else:
        cmd = [
            os.path.join(args.bin_dir, "mlxcel"), "generate",
            "-m", args.model, "-p", prompt,
            "-n", str(args.max_tokens), "--temp", "0",
        ]

    if args.cpu_list:
        cmd = ["taskset", "-c", args.cpu_list] + cmd

    csv_path = os.path.join(args.out, "runs.csv")
    new = not os.path.exists(csv_path)
    csvf = open(csv_path, "a", newline="")
    fields = [
        "harness", "model", "cell", "round", "position", "arm", "attempt", "status",
        "rate", "decode_ms", "gen_tokens", "prompt_tokens", "prefill_ms",
        "jit_notice", "rc", "wall_s", "loadavg", "quiet_wait_s", "raw",
    ]
    writer = csv.DictWriter(csvf, fieldnames=fields)
    if new:
        writer.writeheader()

    def one(round_no, position, arm, measured):
        attempt = 0
        while True:
            attempt += 1
            waited = wait_quiet(log)
            la = loadavg()
            raw = os.path.join(
                args.out, "raw",
                f"{args.harness}-{args.name}-{args.cell}-r{round_no}-{arm}-a{attempt}.log",
            )
            rc, text, wall = run_once(cmd, ARMS[arm], raw, f"issue-2540-{args.harness}")
            row = parse(args.harness, text)
            jit = JIT_NOTICE in text
            if not measured:
                status = "warmup" if rc == 0 and row["rate"] else "warmup_failed"
            elif rc != 0 or not row["rate"]:
                status = "failed"
            elif jit:
                status = "discarded_jit"
            else:
                status = "ok"
            writer.writerow({
                "harness": args.harness, "model": args.name, "cell": args.cell,
                "round": round_no, "position": position, "arm": arm, "attempt": attempt,
                "status": status, **row, "jit_notice": int(jit), "rc": rc,
                "wall_s": f"{wall:.1f}", "loadavg": la, "quiet_wait_s": waited,
                "raw": os.path.basename(raw),
            })
            csvf.flush()
            log(f"round={round_no} pos={position} arm={arm} attempt={attempt} status={status} "
                f"rate={row['rate']} gen={row['gen_tokens']} jit={int(jit)} rc={rc} wall={wall:.1f}s")
            if status in ("ok", "warmup"):
                return status
            if attempt >= 3:
                log(f"giving up on round={round_no} arm={arm} after {attempt} attempts")
                return status

    log(f"start harness={args.harness} rounds={args.rounds} cmd={' '.join(cmd[:1] + cmd[1:3])} ...")
    for position, arm in enumerate(ARM_ORDER):
        if one(0, position, arm, measured=False) != "warmup":
            log("warm-up run failed; recording this model/cell as NA and stopping")
            return 3
    for r in range(1, args.rounds + 1):
        shift = (r - 1) % len(ARM_ORDER)
        order = ARM_ORDER[shift:] + ARM_ORDER[:shift]
        for position, arm in enumerate(order):
            one(r, position, arm, measured=True)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
