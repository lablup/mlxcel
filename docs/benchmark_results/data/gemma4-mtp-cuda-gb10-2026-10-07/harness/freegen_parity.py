#!/usr/bin/env python3
"""Greedy free-generation byte identity, Gemma 4 MTP vs classic decode.

Starts a classic server, runs each prompt once as a chat completion at
temperature 0 with a 256-token budget, stops it, then does the same with the
MTP drafter attached, and compares text and completion-token counts. The MTP
server's log must show the exactness probe passed and at least one burst, so a
silent fallback to classic cannot pass as parity.

  freegen_parity.py --server BIN --target DIR --drafter DIR --out parity.json
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    "Write a detailed explanation of how a hash map handles collisions, with a short Python example.",
    "Tell a story about a lighthouse keeper who finds a message in a bottle. Make it at least three paragraphs.",
    "List the steps to set up a reproducible Python project with a virtual environment, linting, tests, and CI, explaining each step.",
]


def wait_health(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def chat(port, prompt, max_tokens):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        obj = json.loads(r.read())
    return obj["choices"][0]["message"]["content"], obj["usage"]["completion_tokens"]


def arm(a, label, extra):
    log_path = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"parity.{label}.log")
    env = dict(os.environ)
    env.setdefault("RUST_LOG", "info")
    cmd = [a.server, "-m", a.target, "--port", str(a.port), "--parallel", "1"] + a.server_arg + extra
    with open(log_path, "w") as log:
        srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
        try:
            if not wait_health(a.port):
                raise RuntimeError(f"{label}: server never became healthy")
            outs = [chat(a.port, p, a.max_tokens) for p in PROMPTS]
        finally:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=120)
    return outs, open(log_path, errors="replace").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--port", type=int, default=18961)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-arg", action="append", default=[],
                    help="extra mlxcel-server argument for both arms (repeatable)")
    ap.add_argument("--null", action="store_true",
                    help="run the second arm without the drafter (classic vs classic)")
    a = ap.parse_args()
    classic, _ = arm(a, "classic", [])
    if a.null:
        mtp, log = arm(a, "classic2", [])
    else:
        mtp, log = arm(a, "mtp", ["--model-draft", a.drafter, "--draft-kind", "mtp",
                                  "--draft-block-size", str(a.width)])
    rows = []
    for p, (ct, cn), (mt, mn) in zip(PROMPTS, classic, mtp):
        rows.append({"prompt": p, "classic_tokens": cn, "mtp_tokens": mn,
                     "identical": ct == mt and cn == mn, "classic": ct, "mtp": mt})
    result = {"target": a.target, "drafter": a.drafter, "width": a.width,
              "server_args": a.server_arg,
              "probe_passed": "MTP exactness probe passed" in log,
              "declined": "MTP declined" in log,
              "bursts": log.count("Speculative burst completed"), "rows": rows}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    for r in rows:
        print(f"identical={r['identical']} tokens classic={r['classic_tokens']} mtp={r['mtp_tokens']}")
    print(f"probe_passed={result['probe_passed']} declined={result['declined']} bursts={result['bursts']}")
    engaged = a.null or (result["probe_passed"] and result["bursts"])
    return 0 if all(r["identical"] for r in rows) and engaged else 1


if __name__ == "__main__":
    sys.exit(main())
