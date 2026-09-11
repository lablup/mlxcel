#!/usr/bin/env python3
"""Same-process decode/prefill A/B sweep for issue #1798 (CUDA graph budgets, GB10).

Runs `mlxcel-bench-decode` once per (config, round), round-robin over the
configs so drift spreads across the table. Same binary on every arm; each
config is a named set of environment variables (presets). Records the
`[Profile Results]` fields (prompt tokens, prefill ms and tok/s, decode ms
and tok/s, MLX peak memory), the `[Load]` peak, and the 1-minute load average
before each run (host idleness evidence).

Usage:
  bench_bd.py --bin B --model M --prompt-file P --out results.jsonl \
      --rounds 3 --warmup 1 --configs default,both [--preset NAME=K=V,K=V ...] \
      [--max-tokens 200] [--prompt-tokens N] [--wrap "nsys profile ..."]
The config named `default` sets nothing.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import hostgate

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RE = {
    "prompt_tokens": re.compile(r"Prompt tokens:\s+(\d+)"),
    "prefill_ms": re.compile(r"Prefill:\s+([0-9.]+) ms \(([0-9.]+) tok/s\)"),
    "decode_ms": re.compile(r"Decode:\s+([0-9.]+) ms \(([0-9.]+) tok/s\)"),
    "peak_gb": re.compile(r"MLX peak memory:\s+([0-9.]+) GB"),
    "load_peak_gb": re.compile(r"\[Load\] wall: ([0-9.]+) s\s+MLX peak: ([0-9.]+) GB"),
}


def run_once(a, cfg, presets, prompt, wrap=""):
    env = dict(os.environ)
    env.setdefault("MLX_ENABLE_TF32", "0")
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    if cfg != "default":
        for kv in presets[cfg]:
            k, v = kv.split("=", 1)
            env[k] = v
    cmd = shlex.split(wrap) + [a.bin, "-m", a.model, "-n", str(a.max_tokens),
                               "--ignore-eos", "--warmup-tokens", str(a.warmup_tokens)]
    if a.prompt_tokens:
        cmd += ["--prompt-tokens", str(a.prompt_tokens), "-p", "x"]
    else:
        cmd += ["-p", prompt]
    if a.no_chat_template:
        cmd += ["--no-chat-template"]
    gate_wait_s = hostgate.wait_quiet(log=sys.stderr)
    load1 = os.getloadavg()[0]
    ci_job = hostgate.ci_job_running()
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    wall = time.perf_counter() - t0
    out = ANSI.sub("", p.stdout)
    err = ANSI.sub("", p.stderr)
    rec = {"cfg": cfg, "load1_before": load1, "ci_job_running": ci_job, "gate_wait_s": gate_wait_s, "wall_s": wall, "rc": p.returncode,
           "env": {k: env[k] for k in env if k.startswith("MLX_")}}
    m = RE["prompt_tokens"].search(out)
    if m:
        rec["prompt_tokens"] = int(m.group(1))
    m = RE["prefill_ms"].search(out)
    if m:
        rec["prefill_ms"] = float(m.group(1)); rec["prefill_tok_s"] = float(m.group(2))
    m = RE["decode_ms"].search(out)
    if m:
        rec["decode_ms"] = float(m.group(1)); rec["decode_tok_s"] = float(m.group(2))
    m = RE["peak_gb"].search(out)
    if m:
        rec["peak_gb"] = float(m.group(1))
    m = RE["load_peak_gb"].search(out)
    if m:
        rec["load_s"] = float(m.group(1)); rec["load_peak_gb"] = float(m.group(2))
    if p.returncode != 0 or "decode_tok_s" not in rec:
        rec["stderr_tail"] = err[-2000:]
        rec["stdout_tail"] = out[-1500:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--warmup-tokens", type=int, default=20)
    ap.add_argument("--prompt-tokens", type=int, default=0)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--preset", action="append", default=[])
    ap.add_argument("--wrap", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    presets = {}
    for p in a.preset:
        name, _, kvs = p.partition("=")
        presets[name] = kvs.split(",")
    prompt = open(a.prompt_file).read()
    cfgs = a.configs.split(",")
    with open(a.out, "a") as f:
        for i in range(a.warmup):
            r = run_once(a, cfgs[0], presets, prompt, a.wrap)
            r["warmup"] = True; r["tag"] = a.tag; r["model"] = a.model
            print(f"[warmup {i}] {cfgs[0]} dec={r.get('decode_tok_s')} rc={r['rc']}", file=sys.stderr, flush=True)
            if r["rc"] != 0:
                print(r.get("stderr_tail", "")[-800:], file=sys.stderr, flush=True)
            f.write(json.dumps(r) + "\n"); f.flush()
        for rd in range(a.rounds):
            n = len(cfgs)
            for k in range(n):
                cfg = cfgs[(rd + k) % n]
                r = run_once(a, cfg, presets, prompt, a.wrap)
                r["warmup"] = False; r["round"] = rd; r["tag"] = a.tag; r["model"] = a.model
                print(f"[round {rd}] {cfg} dec={r.get('decode_tok_s')} pre={r.get('prefill_tok_s')} "
                      f"ptok={r.get('prompt_tokens')} peak={r.get('peak_gb')} load1={r['load1_before']:.2f} rc={r['rc']}",
                      file=sys.stderr, flush=True)
                if r["rc"] != 0:
                    print(r.get("stderr_tail", "")[-800:], file=sys.stderr, flush=True)
                f.write(json.dumps(r) + "\n"); f.flush()


if __name__ == "__main__":
    main()
