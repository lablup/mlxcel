#!/usr/bin/env python3
"""Decode tok/s A/B for #3254: main vs branch vs main again (null arm).

One server per (arm, model). Each cell sends B concurrent streaming
completions of a ~P-token prompt with the prompt cache off, and reports the
aggregate decode rate: generated tokens after the first one, divided by the
wall time from the first streamed token to the last, summed over the rows.
Arms are interleaved round by round.
"""
import argparse
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

WORDS = ["alpha", "river", "stone", "cloud", "ember", "field", "glass", "harbor"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def prompt_of(tokens, salt):
    # One short common word is one token in both vocabularies; the salt keeps
    # prompts distinct so nothing is shared between requests.
    words = [WORDS[(i * 7 + salt) % len(WORDS)] for i in range(tokens)]
    return " ".join(words)


def stream(port, prompt, max_tokens, out, idx):
    body = json.dumps({
        "model": "default", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    stamps, usage, text = [], None, []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices") and obj["choices"][0].get("text"):
                stamps.append(time.perf_counter())
                text.append(obj["choices"][0]["text"])
    out[idx] = (stamps, usage, "".join(text))


def cell(port, prompt_tokens, batch, max_tokens, salt):
    out = [None] * batch
    threads = [threading.Thread(target=stream, args=(
        port, prompt_of(prompt_tokens, salt * 31 + i), max_tokens, out, i)) for i in range(batch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    prompt_seen, completion, digests = [], [], []
    for stamps, usage, text in out:
        if usage is None or len(stamps) < 8:
            raise RuntimeError("a row returned no usage or too few chunks")
        if abs(usage["prompt_tokens"] - prompt_tokens) > 0.05 * prompt_tokens + 16:
            raise RuntimeError(f"prompt is {usage['prompt_tokens']} tokens, wanted {prompt_tokens}")
        if usage["completion_tokens"] != max_tokens:
            raise RuntimeError(f"row stopped at {usage['completion_tokens']} of {max_tokens} tokens")
        prompt_seen.append(usage["prompt_tokens"])
        completion.append(usage["completion_tokens"])
        digests.append(hashlib.sha256(text.encode()).hexdigest()[:12])
    # Decode rate over the window in which every row is streaming: from the
    # latest first token to the earliest last token. Rows finish their
    # prefill at different times, so anything outside it is not a B-row step.
    lo = max(stamps[0] for stamps, _, _ in out)
    hi = min(stamps[-1] for stamps, _, _ in out)
    if hi - lo < 1.0:
        raise RuntimeError(f"rows overlap for only {hi - lo:.2f} s")
    rate = 0.0
    for stamps, usage, _ in out:
        per_chunk = usage["completion_tokens"] / len(stamps)
        inside = sum(1 for t in stamps if lo < t <= hi)
        rate += inside * per_chunk / (hi - lo)
    return rate, prompt_seen, completion, digests, hi - lo


def start(binary, model, port, log):
    cmd = [binary, "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "--parallel", "4", "--no-prompt-cache", "-c", "65536"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    raise RuntimeError("server did not become healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arms = [("main-a", args.main), ("branch", args.branch), ("main-b", args.main)]
    cells = [(p, b) for p in (256, 8192) for b in (1, 4)]
    if os.environ.get("BENCH_CELLS"):
        cells = [tuple(int(x) for x in c.split("x")) for c in os.environ["BENCH_CELLS"].split(",")]
    results = []
    log = open(args.out + ".server.log", "w")
    for model in args.models:
        for rnd in range(args.rounds):
            order = arms[rnd % 3:] + arms[:rnd % 3]
            for arm, binary in order:
                port = free_port()
                proc = start(binary, model, port, log)
                try:
                    stream(port, prompt_of(64, 999), 16, [None], 0)  # warm-up, discarded
                    for prompt_tokens, batch, rep in [(p, b, r) for (p, b) in cells
                                                      for r in range(args.reps)]:
                        # Rows prefill one after another, so a B > 1 cell decodes long
                        # enough for every row to be streaming at once.
                        max_tokens = args.max_tokens if batch == 1 else (
                            1024 if prompt_tokens <= 256 else 3072)
                        rate, seen, completion, digests, window = cell(
                            port, prompt_tokens, batch, max_tokens, rnd * 100 + batch)
                        row = {"model": os.path.basename(model), "round": rnd, "rep": rep,
                               "arm": arm,
                               "prompt_target": prompt_tokens, "prompt_tokens": seen,
                               "batch": batch, "completion_tokens": completion,
                               "text_sha": digests, "window_s": round(window, 2),
                               "decode_tok_s": round(rate, 2)}
                        results.append(row)
                        print(json.dumps(row), flush=True)
                finally:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
    def text_eq(model, prompt_tokens, batch, x, y):
        def digests(arm):
            return {(r["round"], r["rep"]): r["text_sha"] for r in results if r["model"] == model
                    and r["prompt_target"] == prompt_tokens and r["batch"] == batch
                    and r["arm"] == arm}
        dx, dy = digests(x), digests(y)
        return all(dx[k] == dy[k] for k in dx)

    summary = []
    for model in sorted({r["model"] for r in results}):
        for prompt_tokens, batch in cells:
            def vals(arm):
                return [r["decode_tok_s"] for r in results if r["model"] == model
                        and r["prompt_target"] == prompt_tokens and r["batch"] == batch
                        and r["arm"] == arm]
            a, b, br = vals("main-a"), vals("main-b"), vals("branch")
            main_all = a + b
            null = abs(statistics.median(a) - statistics.median(b))
            summary.append({
                "model": model, "prompt": prompt_tokens, "batch": batch,
                "main_a_median": statistics.median(a), "main_b_median": statistics.median(b),
                "branch_median": statistics.median(br),
                "main_min": min(main_all), "main_max": max(main_all),
                "branch_min": min(br), "branch_max": max(br),
                "null_spread_medians": round(null, 2),
                "branch_vs_main_pct": round(
                    100 * (statistics.median(br) / statistics.median(main_all) - 1), 2),
                "within_main_range": min(main_all) <= statistics.median(br) <= max(main_all),
                "b1_text_main_a_eq_main_b": text_eq(model, prompt_tokens, batch, "main-a", "main-b"),
                "b1_text_main_eq_branch": text_eq(model, prompt_tokens, batch, "main-a", "branch"),
            })
    with open(args.out, "w") as f:
        json.dump({"rows": results, "summary": summary}, f, indent=1)
    for s in summary:
        print(json.dumps(s))


if __name__ == "__main__":
    sys.exit(main())
