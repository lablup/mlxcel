#!/usr/bin/env python3
"""Batched-serving A/B sweep for issue #1798: one mlxcel-server per (config, round),
`scripts/bench_serving_concurrency.py` at the given concurrency ladder, round-robin
over configs. Records each level's row (ttft mean/p95, per-request decode tok/s,
aggregate tok/s) and load1 before the server start. Same binary on every arm.
Usage: bench_srv.py --server BIN --client scripts/bench_serving_concurrency.py --model M
       --out X.jsonl --configs default,both --preset both=K=V,K=V --rounds 3
       [--concurrency 1,4] [--max-tokens 200] [--prompt-tokens 128] [--port 18798]
"""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request
import hostgate

ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", re.M)


def wait_health(port, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_once(a, cfg, presets):
    env = dict(os.environ)
    env.setdefault("MLX_ENABLE_TF32", "0")
    if cfg != "default":
        for kv in presets[cfg]:
            k, v = kv.split("=", 1)
            env[k] = v
    gate_wait_s = hostgate.wait_quiet(log=sys.stderr)
    load1 = os.getloadavg()[0]
    ci_job = hostgate.ci_job_running()
    cmd = [a.server, "-m", a.model, "--port", str(a.port), "--max-batch-size", str(a.max_batch),
           "--ignore-eos"]
    log = open(f"{a.out}.{cfg}.server.log", "a")
    t0 = time.time()
    srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rec = {"cfg": cfg, "load1_before": load1, "ci_job_running": ci_job, "gate_wait_s": gate_wait_s, "model": a.model,
           "env": {k: env[k] for k in env if k.startswith("MLX_")}}
    try:
        if not wait_health(a.port):
            rec["error"] = "server never became healthy"
            return rec
        rec["startup_s"] = time.time() - t0
        cl = ["python3", a.client, "--port", str(a.port), "--concurrency", a.concurrency,
              "--prompt-tokens", str(a.prompt_tokens), "--max-tokens", str(a.max_tokens)]
        p = subprocess.run(cl, capture_output=True, text=True, timeout=1800)
        rec["rc"] = p.returncode
        levels = []
        for m in ROW.finditer(p.stdout):
            levels.append({"conc": int(m.group(1)), "ok": int(m.group(2)), "fail": int(m.group(3)),
                           "ttft_ms_mean": float(m.group(4)), "ttft_ms_p95": float(m.group(5)),
                           "decode_tok_s_mean": float(m.group(6)), "aggregate_tok_s": float(m.group(7))})
        rec["levels"] = levels
        if not levels or p.returncode != 0:
            rec["stdout_tail"] = p.stdout[-1500:]; rec["stderr_tail"] = p.stderr[-1500:]
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=60)
        except Exception:
            os.killpg(srv.pid, signal.SIGKILL)
        log.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True); ap.add_argument("--client", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True); ap.add_argument("--preset", action="append", default=[])
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--concurrency", default="1,4")
    ap.add_argument("--max-tokens", type=int, default=200); ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-batch", type=int, default=8); ap.add_argument("--port", type=int, default=18798)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    presets = {}
    for p in a.preset:
        name, _, kvs = p.partition("="); presets[name] = kvs.split(",")
    cfgs = a.configs.split(",")
    with open(a.out, "a") as f:
        for rd in range(a.rounds):
            n = len(cfgs)
            for k in range(n):
                cfg = cfgs[(rd + k) % n]
                r = run_once(a, cfg, presets); r["round"] = rd; r["tag"] = a.tag
                lv = " ".join(f"c{l['conc']}:agg={l['aggregate_tok_s']:.1f}/dec={l['decode_tok_s_mean']:.1f}/ttft={l['ttft_ms_mean']:.0f}" for l in r.get("levels", []))
                print(f"[round {rd}] {cfg} load1={r['load1_before']:.2f} start={r.get('startup_s', 0):.0f}s {lv} {r.get('error', '')}", file=sys.stderr, flush=True)
                if not r.get("levels"):
                    print(r.get("stderr_tail", "")[-600:], file=sys.stderr, flush=True)
                f.write(json.dumps(r) + "\n"); f.flush()


if __name__ == "__main__":
    main()
