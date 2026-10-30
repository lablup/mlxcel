#!/usr/bin/env python3
"""Interleaved server-ladder A/B for the cuBLASLt version question.

Arms: ``sys`` (the cuBLASLt the binary links) and ``alt`` (the libraries in
$MLXCEL_AB_CUBLAS_DIR, through LD_PRELOAD), each repeated as its own null arm.
Every arm is a fresh ``mlxcel-server`` followed by
``scripts/bench_serving_concurrency.py``; the run order rotates by one arm per
round. Each record names the libcublasLt the server process actually mapped,
read from /proc/<pid>/maps, so a preload that silently failed is visible.

    MLXCEL_AB_CUBLAS_DIR=/path/to/nvidia/cu13/lib \\
    ladder_rounds.py --model models/llama-3.2-1b-instruct-bf16 --rounds 3 \\
        --hostgate --out ladder.jsonl
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
HOSTGATE_DIR = os.path.join(REPO, "docs", "benchmark_results", "data",
                            "sdpa-plan-bucket-gb10-2026-09-12", "harness")
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
COLS = ("conc", "ok", "fail", "ttft_mean_ms", "ttft_p95_ms", "decode_tok_s", "aggregate_tok_s")


def parse_table(stdout):
    rows = []
    for line in stdout.splitlines():
        m = ROW_RE.match(line)
        if m:
            vals = [int(v) if i < 3 else float(v) for i, v in enumerate(m.groups())]
            rows.append(dict(zip(COLS, vals)))
    return rows


def run_arm(a, arm, rnd, preload):
    env = dict(os.environ)
    if preload:
        env["LD_PRELOAD"] = preload
    log_path = f"{a.out}.{arm}.r{rnd}.server.log"
    with open(log_path, "w") as log:
        srv = subprocess.Popen([os.path.join(a.bin_dir, "mlxcel-server"), "-m", a.model,
                                "--host", "127.0.0.1", "--port", str(a.port),
                                "--parallel", str(a.parallel), "--max-batch-prefill", "4"],
                               env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.time() + a.load_timeout
            while True:
                if srv.poll() is not None:
                    raise SystemExit(f"server exited during load ({arm} r{rnd}), see {log_path}")
                if time.time() > deadline:
                    raise SystemExit(f"server not healthy after {a.load_timeout}s ({arm} r{rnd})")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                    break
                except OSError:
                    time.sleep(1)
            with open(f"/proc/{srv.pid}/maps") as f:
                libs = sorted(set(re.findall(r"(/\S*libcublasLt\.so\S*)", f.read())))
            out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "bench_serving_concurrency.py"),
                                  "--port", str(a.port), "--concurrency", a.levels,
                                  "--prompt-tokens", str(a.prompt_tokens), "--max-tokens", str(a.max_tokens)],
                                 capture_output=True, text=True, timeout=3600, check=False)
            return out.stdout, libs
        finally:
            srv.terminate()
            try:
                srv.wait(60)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait()


def summarize(records):
    by = {}
    for r in records:
        for row in r["rows"]:
            by.setdefault((row["conc"], r["arm"]), []).append(row)
    print("\nconc  arm       n  ttft_mean_ms(med)  decode_tok_s(med)  aggregate_tok_s(med)")
    for (conc, arm), rows in sorted(by.items()):
        med = {k: statistics.median(x[k] for x in rows) for k in ("ttft_mean_ms", "decode_tok_s", "aggregate_tok_s")}
        print(f"{conc:>4}  {arm:<8} {len(rows):>2}  {med['ttft_mean_ms']:>17.1f}  "
              f"{med['decode_tok_s']:>17.1f}  {med['aggregate_tok_s']:>20.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="JSONL file, one record per arm run")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--levels", default="1,4,7", help="concurrency ladder (CUDA batched decode covers <= 7 rows)")
    p.add_argument("--parallel", type=int, default=7)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--port", type=int, default=18741)
    p.add_argument("--load-timeout", type=int, default=900)
    p.add_argument("--bin-dir", default=os.environ.get("MLXCEL_AB_BIN_DIR", os.path.join(REPO, "target", "release")))
    p.add_argument("--hostgate", action="store_true", help="wait for a quiet host before each arm")
    a = p.parse_args()
    cublas_dir = os.environ.get("MLXCEL_AB_CUBLAS_DIR")
    if not cublas_dir:
        raise SystemExit("set MLXCEL_AB_CUBLAS_DIR to the alternate cuBLAS lib dir")
    alt = f"{cublas_dir}/libcublasLt.so.13:{cublas_dir}/libcublas.so.13"
    arms = {"sys": None, "alt": alt, "sys-null": None, "alt-null": alt}
    gate = None
    if a.hostgate:
        sys.path.insert(0, HOSTGATE_DIR)
        import hostgate as gate  # noqa: E402
    order = list(arms)
    records = []
    with open(a.out, "a") as f:
        for rnd in range(a.rounds):
            k = rnd % len(order)
            for arm in order[k:] + order[:k]:
                if gate:
                    gate.wait_quiet(log=sys.stderr)
                stdout, libs = run_arm(a, arm, rnd, arms[arm])
                rec = {"model": a.model, "arm": arm, "round": rnd, "cublaslt": libs,
                       "rows": parse_table(stdout), "raw": stdout, "host": os.uname().nodename, "t": time.time()}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                records.append(rec)
                print(f"[{time.strftime('%H:%M:%S')}] r{rnd} {arm:<8} rows={len(rec['rows'])} libs={libs}", flush=True)
    summarize(records)


if __name__ == "__main__":
    main()
