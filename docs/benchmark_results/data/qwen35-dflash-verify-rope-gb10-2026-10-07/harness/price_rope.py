#!/usr/bin/env python3
"""Throughput of the #2191 fix: DFlash at the width main resolves vs classic.

Reuses `dflash-sdpav-options-gb10-2026-09-21/harness/price_options.py` for one
arm (its host gate, server lifecycle, discarded warm-up and streaming
request), and only replaces the arm table: per round, classic-a, the DFlash
burst with no `--draft-block-size` (so the server resolves the width itself),
and classic-b, a classic-vs-classic null arm. Rounds are interleaved and the
arm order rotates per round so drift does not land on one arm.

  price_rope.py --server BIN --target DIR --drafter DIR --rounds 3 --out run.jsonl
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRICE = os.path.normpath(
    os.path.join(HERE, "..", "..", "dflash-sdpav-options-gb10-2026-09-21", "harness",
                 "price_options.py")
)
spec = importlib.util.spec_from_file_location("price_options", PRICE)
po = importlib.util.module_from_spec(spec)
spec.loader.exec_module(po)


def server_cmd(a, width):
    # Each server holds the shared GPU lock only for its own lifetime, so the
    # host gate's wait for an idle host does not hold the GPU from others.
    lock = ["gpu-lock", "run", "--tag", a.lock_tag, "--"] if a.lock_tag else []
    cmd = lock + [a.server, "-m", a.target, "--port", str(a.port), "--ignore-eos",
           "--max-batch-size", "1"]
    if width is not None:
        cmd += ["--model-draft", a.drafter, "--draft-kind", "dflash"]
        if width != "auto":
            cmd += ["--draft-block-size", str(width)]
    return cmd


po.server_cmd = server_cmd

ARMS = [("classic-a", None), ("dflash-auto", "auto"), ("classic-b", None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--start-round", type=int, default=0,
                    help="resume an interrupted session at this round")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt-file", default=os.path.join(po.S1797, "prompt_retry.txt"))
    ap.add_argument("--port", type=int, default=18935)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--lock-tag", default=None,
                    help="wrap each server in `gpu-lock run --tag TAG`")
    a = ap.parse_args()
    a.logdir = a.logdir or os.path.dirname(os.path.abspath(a.out))
    os.makedirs(a.logdir, exist_ok=True)
    prompt = open(a.prompt_file).read()
    with open(a.out, "a") as f:
        for rnd in range(a.start_round, a.rounds):
            order = ARMS[rnd % len(ARMS):] + ARMS[:rnd % len(ARMS)]
            for tag, width in order:
                rec = po.run_arm(a, f"{tag}-r{rnd}", width, {}, prompt)
                rec["round"] = rnd
                rec["arm_base"] = tag
                f.write(json.dumps(rec) + "\n")
                f.flush()
                rs = [r["e2e_tok_s"] for r in rec.get("runs", [])]
                shas = sorted({r["sha"] for r in rec.get("runs", [])})
                print(f"[r{rnd} {tag}] "
                      + (f"e2e {min(rs):.2f}..{max(rs):.2f} tok/s" if rs
                         else f"ERROR {rec.get('error')}")
                      + f" | sha {shas} | declined={rec.get('declined_to_classic')}",
                      flush=True)
    print("ROPE PRICE DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
