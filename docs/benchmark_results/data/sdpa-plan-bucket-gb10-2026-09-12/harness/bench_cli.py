#!/usr/bin/env python3
"""CLI-driven A/B sweep for issue #1799 (Laguna DFlash verify cost, GB10).

Runs `mlxcel generate` once per (config, round), round-robin over the configs
so drift spreads across the table instead of pooling on one arm. Same binary
on every arm; the classic arm is the same command without --draft-model.
Records decode tok/s as the CLI reports it, the `DFlash:` diagnostics line,
the generated token ids (greedy identity), and the 1-minute load average
before each run (host idleness evidence).

Usage:
  bench_cli.py --bin B --target T --draft D --prompt-file P --out results.jsonl \
      --rounds 3 --configs off,b2,b4,b8,b16 [--env K=V ...] [--max-tokens 200]
  config grammar:  off | b<N> | <prefix>-b<N> | <prefix>-off
      prefix maps through --preset NAME=K=V,K=V (e.g. ops100=MLX_MAX_OPS_PER_BUFFER=100)
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import threading

import hostgate


def _peak_rss_kib(pid, stop):
    """Poll VmHWM for the run's own peak resident size. Cheap (one small read
    per 200 ms) and it is the number the memory floor should be set from."""
    peak = 0
    while not stop.is_set():
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        peak = max(peak, int(line.split()[1]))
                        break
        except (OSError, ValueError):
            break
        stop.wait(0.2)
    return peak

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GEN_RE = re.compile(r"\[Generated (\d+) tokens in ([0-9.]+)s = ([0-9.]+) tok/s\]")
DFLASH_RE = re.compile(r"^DFlash: (.*)$", re.M)
KV_RE = re.compile(r"(\w+)=([-0-9.]+)")
IDS_RE = re.compile(r"\[token ids \((\d+)\): ([0-9 ]*)\]")


def run_once(a, cfg, presets, prompt, wrap=""):
    env = dict(os.environ)
    env["MLXCEL_PRINT_TOKEN_IDS"] = "1"
    env["MLXCEL_MTP_ALLOW_INEXACT"] = "1"
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    prefix, _, arm = cfg.rpartition("-") if "-" in cfg else ("", "", cfg)
    if prefix:
        for kv in presets[prefix]:
            k, v = kv.split("=", 1)
            env[k] = v
    cmd = shlex.split(wrap) + [a.bin, "generate", "-m", a.target, "-p", prompt,
                               "-n", str(a.max_tokens), "--temp", "0"]
    block = None
    if arm != "off":
        block = int(arm[1:])
        cmd += ["--draft-model", a.draft, "--draft-kind", "dflash",
                "--draft-block-size", str(block)]
    gate_wait_s = hostgate.wait_quiet(log=sys.stderr)
    # Driver trip wire, checked immediately before EVERY run rather than once
    # per rung: the 2026-09-17 halt lost a rung because the check was manual
    # and periodic. Raises DriverHalt, which the caller turns into a clean stop
    # with everything so far already written (#1820).
    driver_wait_s, nvrm_window, nvrm_total = hostgate.driver_gate(log=sys.stderr)
    gate_wait_s += driver_wait_s
    mem_wait_s, mem_avail = hostgate.mem_gate(log=sys.stderr)
    gate_wait_s += mem_wait_s
    load1 = os.getloadavg()[0]
    ci_job = hostgate.ci_job_running()
    # Re-checked immediately before the load and again after: the gate above can
    # clear and a foreign session can start in the gap (#1820).
    foreign_before = hostgate.foreign_model_procs()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    stop = threading.Event()
    peak = {}
    sampler = threading.Thread(target=lambda: peak.__setitem__("kib", _peak_rss_kib(proc.pid, stop)))
    sampler.start()
    try:
        out_s, err_s = proc.communicate(timeout=3600)
    finally:
        stop.set()
        sampler.join(timeout=5)
    p = subprocess.CompletedProcess(cmd, proc.returncode, out_s, err_s)
    wall = time.perf_counter() - t0
    out = ANSI.sub("", p.stdout)
    err = ANSI.sub("", p.stderr)
    rec = {"cfg": cfg, "block": block, "load1_before": load1, "ci_job_running": ci_job, "gate_wait_s": gate_wait_s, "wall_s": wall,
           "foreign_models_before": foreign_before, "foreign_models_after": hostgate.foreign_model_procs(),
           "nvrm_window_before": nvrm_window, "nvrm_total_before": nvrm_total,
           "mem_available_gib_before": round(mem_avail, 1), "peak_rss_kib": peak.get("kib", 0),
           "nvrm_total_after": hostgate.nvrm_counts()[1],
           "prompt_file": a.prompt_file, "max_tokens": a.max_tokens, "rc": p.returncode}
    m = GEN_RE.search(out)
    if m:
        rec["gen_tokens"] = int(m.group(1))
        rec["decode_s"] = float(m.group(2))
        rec["tok_s"] = float(m.group(3))
    m = DFLASH_RE.search(out)
    if m:
        rec["diag"] = {k: float(v) for k, v in KV_RE.findall(m.group(1))}
    m = IDS_RE.search(err)
    if m:
        rec["ids"] = m.group(2).strip()
    if p.returncode != 0 or "tok_s" not in rec:
        rec["stderr_tail"] = err[-2000:]
        rec["stdout_tail"] = out[-1000:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=200)
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
            try:
                r = run_once(a, cfgs[0], presets, prompt, a.wrap)
            except hostgate.DriverHalt as e:
                print(f"[driver] HALT before warmup: {e}", file=sys.stderr, flush=True)
                return 3
            r["warmup"] = True
            r["tag"] = a.tag
            print(f"[warmup {i}] {cfgs[0]} tok/s={r.get('tok_s')} rc={r['rc']}", file=sys.stderr, flush=True)
            f.write(json.dumps(r) + "\n"); f.flush()
        for rd in range(a.rounds):
            n = len(cfgs)
            for k in range(n):
                cfg = cfgs[(rd + k) % n]
                try:
                    r = run_once(a, cfg, presets, prompt, a.wrap)
                except hostgate.DriverHalt as e:
                    print(f"[driver] HALT before {cfg}: {e}", file=sys.stderr, flush=True)
                    return 3
                r["warmup"] = False
                r["round"] = rd
                r["tag"] = a.tag
                d = r.get("diag", {})
                extra = ""
                if d.get("rounds"):
                    n_r = d["rounds"]
                    extra = (f" rounds={n_r:.0f} acc={d['accepted']/n_r:.2f}"
                             f" draft/r={d['draft_ms']/n_r:.1f} vgraph/r={d['verify_graph_ms']/n_r:.1f}"
                             f" sync/r={d['verify_sync_ms']/n_r:.1f}")
                print(f"[round {rd}] {cfg} tok/s={r.get('tok_s')} load1={r['load1_before']:.2f}"
                      f" rc={r['rc']}{extra}", file=sys.stderr, flush=True)
                if r["rc"] != 0:
                    print(r.get("stderr_tail", "")[-800:], file=sys.stderr, flush=True)
                f.write(json.dumps(r) + "\n"); f.flush()


if __name__ == "__main__":
    # Propagate the driver-halt code so the caller stops the rung instead of
    # walking on to the next configuration (#1820).
    sys.exit(main() or 0)
