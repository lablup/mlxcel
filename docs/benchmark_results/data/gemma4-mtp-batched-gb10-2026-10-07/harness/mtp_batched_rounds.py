#!/usr/bin/env python3
"""Batched Gemma 4 MTP vs classic batched decode on CUDA (issue #2190).

The #2160 method (`gemma4-mtp-cuda-gb10-2026-10-07/harness/mtp_rounds.py`)
with B concurrent requests per timed batch instead of one. Each round runs
three servers: classic (open bracket), batched MTP, classic (close bracket);
the classic pair is the null arm. Every arm discards one warm-up batch, then
times `--n` batches of B concurrent streaming completions, each a distinct
prompt (so rows diverge) at temperature 0. A batch's aggregate rate is the
rows' total completion tokens over the wall time until the last row finishes.

The classic arms run `--server` with `--max-batch-size B --parallel B`. The
MTP arm runs `--mtp-server`, a build with the `requires_singleton` decline
removed (the shipped binary declines B>1 for these geometries, which is the
decision this measures), with `MLXCEL_ENABLE_MTP_BATCH=1` and
`MLXCEL_ENABLE_MTP_BATCH_RAGGED=1`.

The scheduler forms a batched MTP window only from requests already queued
when it dequeues the head, and B client threads do not arrive inside one
tick: without help the first row starts a B=1 burst alone. So every batch
(warm-up and timed, in every arm) first sends a short primer request, then
the B rows 0.25 s later. In the MTP arm the primer runs as a run-to-completion
B=1 burst (`MLXCEL_MTP_TICK_SLICE=0`), the rows queue behind it, and they
dequeue as one window. Two queued rows otherwise take the classic batched
prefill, so the MTP arm also needs `--mtp-server-arg=--max-batch-prefill
--mtp-server-arg=1`. The rate is measured from the primer's completion to
the last row's completion (`agg_tok_s`), and also from the rows' submission
(`agg_tok_s_submit`). The record counts per-row burst completions and B=1
burst finalizations in the server log, so a run whose rows did not form one
window is visible.

  mtp_batched_rounds.py --server BIN --mtp-server BIN --target DIR \
      --drafter DIR --batch 2 --out run.jsonl
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time

HARNESS = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HARNESS, "..", ".."))
S1797 = os.path.join(DATA, "draft-block-width-gb10-2026-09-20", "harness")
sys.path.insert(0, os.path.join(DATA, "sdpa-plan-bucket-gb10-2026-09-12", "harness"))
import hostgate  # noqa: E402

_sweep = {"__file__": os.path.join(S1797, "sweep_server_widths.py")}
with open(_sweep["__file__"]) as fh:
    exec(compile(fh.read().split('if __name__ == "__main__"')[0], _sweep["__file__"], "exec"), _sweep)
wait_health = _sweep["wait_health"]
first_model_id = _sweep["first_model_id"]
_json = _sweep["json"]
_urllib = _sweep["urllib"]

# Every request carries the same explicit seed. Without one each request draws
# a random seed, and the window collector (`sampling_config_eq`) only groups
# rows whose sampling configs match, so no batched window would ever form.
# Greedy decoding ignores the seed, so outputs are unaffected.
SEED = 2190


def complete(port, prompt, max_tokens, model, timeout=1800):
    """`sweep_server_widths.complete` plus a fixed seed. Returns (wall seconds,
    text, chunk count, prompt tokens, completion tokens)."""
    body = _json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "seed": SEED, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = _urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                  headers={"Content-Type": "application/json"})
    text, chunks, ptok, ctok = [], 0, None, None
    t0 = time.time()
    with _urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = _json.loads(payload)
            except _json.JSONDecodeError:
                continue
            usage = obj.get("usage") or {}
            ptok = usage.get("prompt_tokens") or ptok
            ctok = usage.get("completion_tokens") or ctok
            for choice in obj.get("choices", []):
                if choice.get("text"):
                    text.append(choice["text"])
                    chunks += 1
    return time.time() - t0, "".join(text), chunks, ptok, ctok

BURST = "Speculative burst completed"
B1_BURST = "speculative B=1 burst finalized"
DECLINE = "declined"

# Distinct openings so the B rows are different requests whose accept counts
# diverge; the shared body is the #1797 prompt.
OPENINGS = (
    "You are reviewing a retry helper for a payments service.",
    "A junior engineer asked you to explain this retry helper.",
    "Summarize the risks in the following retry helper before it ships.",
    "Rewrite the following retry helper so it is easier to test.",
)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def server_cmd(a, mtp):
    binary = a.mtp_server if mtp else a.server
    # No `--ignore-eos`: it suppresses EOS through the request's token bias,
    # and the window collector (`sampling_config_eq`) refuses rows with a
    # non-empty bias, so it alone keeps every row out of a batched window.
    cmd = [binary, "-m", a.target, "--port", str(a.port),
           "--max-batch-size", str(a.batch), "--parallel", str(a.batch),
           "--no-cache-prompt"] + a.server_arg
    if mtp:
        cmd += ["--model-draft", a.drafter, "--draft-kind", "mtp",
                "--draft-block-size", str(a.width)] + a.mtp_server_arg
    return cmd


PRIMER = "Say hello."


def timed_batch(a, prompts, model_id):
    """Primer, then the B rows. Returns (wall from primer end, wall from row
    submission, per-row results)."""
    results = [None] * len(prompts)
    ends = [0.0] * len(prompts)
    primer_end = [0.0]

    def primer():
        complete(a.port, PRIMER, 8, model_id)
        primer_end[0] = time.time()

    def one(i):
        results[i] = complete(a.port, prompts[i], a.max_tokens, model_id)
        ends[i] = time.time()

    p = threading.Thread(target=primer)
    p.start()
    time.sleep(0.25)
    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    p.join()
    last = max(ends)
    return last - max(primer_end[0], t0), last - t0, results


def wait_no_builds(max_wait, samples=12, interval=5):
    """`hostgate.wait_quiet` without its CI-job clause: a minute of no
    compiler or foreign model process, bounded by `max_wait` seconds.

    The CI runner on this host takes the same GPU lock this run holds, so a
    queued CI job sits in `Runner.Worker` waiting on us and `wait_quiet` would
    wait on it in turn. Builds are what skew GB10 decode; they still gate.
    Returns (seconds waited, timed out)."""
    start = time.time()
    quiet = 0
    while quiet < samples:
        if hostgate.busy_procs() or hostgate.foreign_model_procs():
            quiet = 0
            if time.time() - start > max_wait:
                return round(time.time() - start, 1), True
        else:
            quiet += 1
        time.sleep(interval)
    return round(time.time() - start, 1), False


def run_arm(a, tag, mtp, prompts):
    gate_wait = None
    if not a.no_gate:
        gate_wait, gate_timed_out = wait_no_builds(a.gate_max_wait)
    else:
        gate_timed_out = None
    driver_wait, _window, nvrm_before = hostgate.driver_gate(log=sys.stderr)
    mem_wait, _avail = hostgate.mem_gate(log=sys.stderr)
    log_path = os.path.join(a.logdir, f"server.b{a.batch}.{tag}.log")
    env = dict(os.environ)
    env.setdefault("RUST_LOG", "info")
    if mtp:
        env["MLXCEL_ENABLE_MTP_BATCH"] = "1"
        env["MLXCEL_ENABLE_MTP_BATCH_RAGGED"] = "1"
        env["MLXCEL_MTP_TICK_SLICE"] = "0"
    rec = {"arm": tag, "batch": a.batch, "mtp": mtp, "cmd": server_cmd(a, mtp), "log": log_path,
           "gate_wait_s": gate_wait, "gate_timed_out": gate_timed_out,
           "driver_wait_s": driver_wait, "mem_wait_s": mem_wait,
           "ci_job_running": hostgate.ci_job_running(),
           "busy_procs": hostgate.busy_procs(), "load1_before": os.getloadavg()[0]}
    with open(log_path, "w") as log:
        srv = subprocess.Popen(server_cmd(a, mtp), env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if not wait_health(a.port):
                rec["error"] = "server never became healthy"
                return rec
            model_id = first_model_id(a.port)
            timed_batch(a, prompts, model_id)  # discarded warm-up batch
            runs = []
            for _ in range(a.n):
                wall, wall_submit, results = timed_batch(a, prompts, model_id)
                tokens = sum(r[4] or 0 for r in results)
                runs.append({
                    "wall_s": round(wall, 3),
                    "wall_submit_s": round(wall_submit, 3),
                    "agg_tok_s": round(tokens / wall, 3),
                    "agg_tok_s_submit": round(tokens / wall_submit, 3),
                    "completion_tokens": [r[4] for r in results],
                    "sha": [sha(r[1]) for r in results],
                })
            rec["runs"] = runs
            rec["load1_after"] = os.getloadavg()[0]
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
                srv.wait(timeout=120)
            except Exception:  # noqa: BLE001
                os.killpg(srv.pid, signal.SIGKILL)
    body = open(log_path, errors="replace").read()
    rec["burst_rows"] = body.count(BURST)
    rec["b1_bursts"] = body.count(B1_BURST)
    # Rows served inside a batched window (primers and stray rows run as B=1).
    rec["batched_rows"] = rec["burst_rows"] - rec["b1_bursts"]
    rec["thrash_abort"] = "Cache thrashing" in body
    rec["declines"] = sum(1 for line in body.splitlines() if "MTP" in line and DECLINE in line)
    _, nvrm_after = hostgate.nvrm_counts()
    rec["nvrm_delta"] = nvrm_after - nvrm_before
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--mtp-server", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt-file", default=os.path.join(S1797, "prompt_retry.txt"))
    ap.add_argument("--port", type=int, default=18970)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--gate-max-wait", type=float, default=1800.0,
                    help="seconds to wait for a quiet host before running anyway (recorded)")
    ap.add_argument("--server-arg", action="append", default=[])
    # Two or more queued requests take the classic batched prefill
    # (`execute_batched_prefill`) whenever `--max-batch-prefill` > 1, which
    # never reaches the speculative window; the MTP arm passes
    # `--max-batch-prefill 1` here so the rows dequeue through it.
    ap.add_argument("--mtp-server-arg", action="append", default=[])
    a = ap.parse_args()
    a.logdir = os.path.dirname(os.path.abspath(a.out))
    body = open(a.prompt_file).read()
    prompts = [f"{OPENINGS[i % len(OPENINGS)]}\n\n{body}" for i in range(a.batch)]
    with open(a.out, "a") as out:
        for r in range(a.rounds):
            for tag, mtp in ((f"r{r}-classic-open", False), (f"r{r}-mtp", True),
                             (f"r{r}-classic-close", False)):
                rec = run_arm(a, tag, mtp, prompts)
                rec["round"] = r
                out.write(json.dumps(rec) + "\n")
                out.flush()
                rs = [x["agg_tok_s"] for x in rec.get("runs", [])]
                shas = sorted({tuple(x["sha"]) for x in rec.get("runs", [])})
                print(f"[b{a.batch} {tag}] {rs or rec.get('error')} sha={shas} "
                      f"batched_rows={rec['batched_rows']}/{a.batch * (a.n + 1) if mtp else 0} b1_bursts={rec['b1_bursts']} "
                      f"thrash={rec['thrash_abort']} "
                      f"gate_timed_out={rec['gate_timed_out']} nvrm+{rec['nvrm_delta']}",
                      flush=True)
    print("ROUNDS DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
