#!/usr/bin/env python3
"""Greedy /v1/completions byte identity, DFlash vs classic, for #2191.

One server per arm (classic, then DFlash at each width, `auto` meaning no
--draft-block-size so the server resolves it), three prompts per arm at
temperature 0, compared by sha256 against the classic arm. No env overrides.

  served_identity.py --server BIN --target DIR --drafter DIR --out DIR
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request

PROMPTS = [
    "def quicksort(arr):\n    \"\"\"Sort a list of integers.\"\"\"\n",
    "The three most important ideas in thermodynamics are",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.\n\n",
]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="classic,2,4,8,16,auto")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--port", type=int, default=18977)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = {}
    for arm in a.arms.split(","):
        cmd = [a.server, "-m", a.target, "--port", str(a.port), "--max-batch-size", "1"]
        if arm != "classic":
            cmd += ["--model-draft", a.drafter, "--draft-kind", "dflash"]
            if arm != "auto":
                cmd += ["--draft-block-size", arm]
        log = open(os.path.join(a.out, f"server.{arm}.log"), "w")
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(600):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                    break
                except Exception:
                    time.sleep(1)
            outs = []
            for p in PROMPTS:
                r = post(a.port, {"model": "m", "prompt": p, "max_tokens": a.max_tokens,
                                  "temperature": 0})
                t = r["choices"][0]["text"]
                outs.append({"sha": hashlib.sha256(t.encode()).hexdigest()[:12],
                             "chars": len(t),
                             "completion_tokens": r["usage"]["completion_tokens"],
                             "text": t})
            results[arm] = outs
        finally:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=120)
            log.close()
        body = open(os.path.join(a.out, f"server.{arm}.log"), errors="replace").read()
        results[arm + "_log"] = {
            "bursts": body.count("Speculative burst completed"),
            "declined": "falling back to classic decode" in body,
            "probe_lines": [ln for ln in body.splitlines() if "exactness" in ln][:4],
            "block_size_lines": [ln[ln.find("speculative="):][:160] for ln in body.splitlines()
                                 if "speculative=dflash" in ln][:1],
        }
        same = [o["sha"] == c["sha"] for o, c in zip(outs, results["classic"])]
        print(f"[{arm}] shas {[o['sha'] for o in outs]} equal_to_classic={same} "
              f"bursts={results[arm + '_log']['bursts']} "
              f"declined={results[arm + '_log']['declined']}", flush=True)
    with open(os.path.join(a.out, "served_identity.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("SERVED DONE", flush=True)


if __name__ == "__main__":
    main()
