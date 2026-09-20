#!/usr/bin/env python3
"""Served draft-block-width sweep for issue #1797: one `mlxcel-server` per arm,
n streaming `/v1/completions` requests per arm after a discarded warm-up.

Why a separate driver from `scripts/bench_block_width.sh`: that script drives the
offline `mlxcel generate` path and interleaves widths per sample, which it can do
because a sample is one process invocation. A served width cannot be interleaved
that cheaply. `draft_block_size` is worker-owned in the live settings API and is
refused with "owned by the model worker; restart required", so changing the width
means restarting the server, and a per-request generator (`dflash_target.rs`
builds one per request and drops it at the end) gives a warm-up nothing to
amortize into.

So the drift control is a bracket instead of an interleave: the classic arm runs
FIRST and LAST, with the widths in between. Two classic ranges that overlap say
the session did not drift under the arms they bracket. Two that separate say the
middle rows are suspect, and the record has to say so rather than quietly
reporting the ratio.

Host safety comes from the hardened `hostgate` of #1820 (imported, not copied):
a sustained-quiet CPU predicate that matches on `/proc/<pid>/comm` and drops
stopped processes, a foreign-model check, a memory floor, and the cumulative
NV_ERR_NO_MEMORY trip wire that this host needs. Per-arm NVRM deltas are recorded
so a sweep can be projected from its first arm rather than discovered at the end.

  sweep_server_widths.py --server BIN --target DIR --drafter DIR \\
      --widths 2,3,4,5,6,7,8,16 --n 3 --out run.jsonl
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HARNESS = os.path.dirname(os.path.abspath(__file__))
# The #1820 gate, used from where it lives rather than duplicated. A second copy
# would drift, and the three predicates it encodes were each paid for once.
sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(HARNESS, "..", "..", "sdpa-plan-bucket-gb10-2026-09-12", "harness")
    ),
)
import hostgate  # noqa: E402

SPEC_LINE = re.compile(r"speculative=(\w+) \(([^)]*)\)")
DIAG_LINE = re.compile(r"block_size[=:]\s*(\d+)")


def wait_health(port, timeout=900):
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


def complete(port, prompt, max_tokens, timeout=1800):
    """One streaming completion. Returns (wall seconds, text, chunk count).

    End to end for a fixed token budget, because a DFlash burst delivers its
    chunks at the end of a round: an inter-chunk rate would measure the burst
    shape rather than the throughput.
    """
    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    text = []
    chunks = 0
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices", []):
                piece = choice.get("text")
                if piece:
                    text.append(piece)
                    chunks += 1
    return time.time() - t0, "".join(text), chunks


def server_cmd(a, width):
    """Build one arm's command line.

    `classic` drops the drafter entirely. `default` keeps the drafter and
    passes no width, which is the arm the whole issue is about: what an
    operator gets without knowing the flag exists. `envN` also passes no flag
    and sets `MLXCEL_DRAFT_BLOCK_SIZE` instead, which is the override surface
    that has to keep winning over the new default. Anything else is an
    explicit `--draft-block-size`.
    """
    cmd = [a.server, "-m", a.target, "--port", str(a.port), "--ignore-eos",
           "--max-batch-size", "1"]
    if width != "classic":
        cmd += ["--draft-model", a.drafter, "--draft-kind", "dflash"]
        if width != "default" and not str(width).startswith("env"):
            cmd += ["--draft-block-size", str(width)]
    return cmd


def arm_env(width):
    """The environment override an `envN` arm sets, if any."""
    w = str(width)
    return {"MLXCEL_DRAFT_BLOCK_SIZE": w[3:]} if w.startswith("env") else {}


def run_arm(a, width, prompt, tag):
    """Gate the host, start one server, measure n requests, tear it down."""
    gate_wait = hostgate.wait_quiet(log=sys.stderr)
    driver_wait, nvrm_window, nvrm_before = hostgate.driver_gate(log=sys.stderr)
    mem_wait, mem_avail_gib = hostgate.mem_gate(log=sys.stderr)

    log_path = os.path.join(a.logdir, f"server.{tag}.log")
    log = open(log_path, "w")
    env = dict(os.environ)
    # MLX defaults TF32 on; nothing here is a float comparison, but the record
    # has to be able to say which way it was set.
    env.setdefault("MLX_ENABLE_TF32", "0")
    env.setdefault("RUST_LOG", "info")
    # Pinned, not auto-detected: build.rs yields `121a` here while the shipped
    # release and every prior GB10 record use plain `121`, and a block-width
    # crossover is a kernel-dispatch result.
    env.setdefault("MLX_CUDA_ARCHITECTURES", "121")
    env.update(arm_env(width))
    t0 = time.time()
    srv = subprocess.Popen(
        server_cmd(a, width), env=env, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    rec = {
        "arm": tag,
        "width": width,
        "cmd": server_cmd(a, width),
        "arm_env": arm_env(width),
        "load1_before": os.getloadavg()[0],
        "ci_job_running": hostgate.ci_job_running(),
        "gate_wait_s": gate_wait,
        "driver_wait_s": driver_wait,
        "mem_wait_s": mem_wait,
        "mem_available_gib_before": round(mem_avail_gib, 1),
        "nvrm_window_before": nvrm_window,
        "nvrm_total_before": nvrm_before,
        "log": log_path,
    }
    try:
        if not wait_health(a.port):
            rec["error"] = "server never became healthy"
            return rec
        rec["startup_s"] = round(time.time() - t0, 1)
        # Discarded: the first request pays MLX kernel and graph compilation
        # for every shape this arm will use.
        warm_s, warm_text, _ = complete(a.port, prompt, a.max_tokens)
        rec["warmup_s"] = round(warm_s, 3)
        rec["warmup_chars"] = len(warm_text)
        runs = []
        for _ in range(a.n):
            wall, text, chunks = complete(a.port, prompt, a.max_tokens)
            runs.append(
                {
                    "wall_s": round(wall, 3),
                    "e2e_tok_s": round(a.max_tokens / wall, 3),
                    "chunks": chunks,
                    "chars": len(text),
                    "text": text,
                }
            )
        rec["runs"] = runs
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=120)
        except Exception:
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except Exception:
                pass
        log.close()
        # The GPU is released only once the process is really gone; a peer
        # session starting here would otherwise share the device.
        time.sleep(5)

    blob = open(log_path, errors="replace").read()
    spec = SPEC_LINE.search(blob)
    rec["speculative_line"] = spec.group(0) if spec else None
    rec["diagnostics_block_sizes"] = sorted(
        {int(m) for m in DIAG_LINE.findall(blob)}
    )
    window_after, total_after = hostgate.nvrm_counts()
    rec["nvrm_total_after"] = total_after
    rec["nvrm_delta"] = total_after - nvrm_before
    rec["nvrm_window_after"] = window_after
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--widths", default="2,3,4,5,6,7,8,16")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt-file", default=os.path.join(HARNESS, "prompt_retry.txt"))
    ap.add_argument("--port", type=int, default=18797)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", default=None)
    ap.add_argument(
        "--nvrm-budget",
        type=int,
        default=400,
        help="stop before an arm whose projected cumulative NVRM would exceed this",
    )
    ap.add_argument(
        "--no-trailing-classic",
        action="store_true",
        help="skip the closing classic bracket (only for a resumed partial sweep)",
    )
    a = ap.parse_args()
    a.logdir = a.logdir or os.path.dirname(os.path.abspath(a.out))
    os.makedirs(a.logdir, exist_ok=True)
    prompt = open(a.prompt_file).read()

    widths = [w.strip() for w in a.widths.split(",") if w.strip()]
    # classic first, the widths, classic again: the bracket IS the drift check.
    arms = [("classic", "classic-open")]
    arms += [(w, "default" if w == "default" else f"w{w}") for w in widths]
    if not a.no_trailing_classic:
        arms.append(("classic", "classic-close"))

    per_arm_nvrm = []
    with open(a.out, "a") as f:
        for i, (width, tag) in enumerate(arms):
            _, total = hostgate.nvrm_counts()
            if per_arm_nvrm:
                worst = max(per_arm_nvrm)
                projected = total + worst * (len(arms) - i)
                if projected > a.nvrm_budget:
                    print(
                        f"[halt] projected cumulative NVRM {projected} "
                        f"(now {total}, worst arm {worst}, {len(arms) - i} arms left) "
                        f"exceeds the {a.nvrm_budget} budget; stopping here",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
            rec = run_arm(a, width, prompt, tag)
            per_arm_nvrm.append(rec.get("nvrm_delta", 0))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            rates = [r["e2e_tok_s"] for r in rec.get("runs", [])]
            print(
                f"[{tag}] "
                + (
                    f"e2e {min(rates):.2f} to {max(rates):.2f} tok/s"
                    if rates
                    else f"ERROR {rec.get('error')}"
                )
                + f" | nvrm +{rec.get('nvrm_delta', '?')} (total {rec.get('nvrm_total_after', '?')})"
                + f" | {rec.get('speculative_line')}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
