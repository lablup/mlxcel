#!/usr/bin/env python3
"""Price the options for issue #1935's byte-identity problem, in one session.

The question this answers is not "is the burst faster" but "what does each way
of getting byte-identical greedy text cost, measured against the same classic
brackets". Every arm is one `mlxcel-server` with its own environment, so the
arms differ in exactly the variable named in their tag and in nothing else.

Reuses the #1797 harness rather than reimplementing it: the same prompt, the
same streaming request, the same discarded warm-up, and the same #1820 host
gate (sustained-quiet CPU predicate matching on `/proc/<pid>/comm` with stopped
processes dropped, foreign-model check, memory floor, cumulative
NV_ERR_NO_MEMORY trip wire). Only the arm table and the per-arm environment are
new.

  price_options.py --server BIN --target DIR --drafter DIR --out run.jsonl
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import re
import time

HARNESS = os.path.dirname(os.path.abspath(__file__))
S1797 = os.path.normpath(
    os.path.join(HARNESS, "..", "..", "draft-block-width-gb10-2026-09-20", "harness")
)
sys.path.insert(0, S1797)
sys.path.insert(
    0,
    os.path.normpath(
        os.path.join(HARNESS, "..", "..", "sdpa-plan-bucket-gb10-2026-09-12", "harness")
    ),
)
import hostgate  # noqa: E402

_sweep = {}
with open(os.path.join(S1797, "sweep_server_widths.py")) as fh:
    src = fh.read().split('if __name__ == "__main__"')[0]
_sweep["__file__"] = os.path.join(S1797, "sweep_server_widths.py")
exec(compile(src, _sweep["__file__"], "exec"), _sweep)
wait_health = _sweep["wait_health"]
first_model_id = _sweep["first_model_id"]
complete = _sweep["complete"]
DIAG_LINE = _sweep["DIAG_LINE"]
DIAG_FIELDS = _sweep["DIAG_FIELDS"]
DECLINE = _sweep["DECLINE"]

# tag, draft width (None = no drafter), extra environment.
#
# The brackets are the SHIPPED configuration: no drafter, MLXCEL_SDPA_VECTOR_LARGE_D
# left at its default of on. Everything between them is priced against those.
ARMS = [
    ("classic-open", None, {}),
    ("classic-novec", None, {"MLXCEL_SDPA_VECTOR_LARGE_D": "0"}),
    ("w3-novec", 3, {"MLXCEL_SDPA_VECTOR_LARGE_D": "0"}),
    ("w4-novec", 4, {"MLXCEL_SDPA_VECTOR_LARGE_D": "0"}),
    ("w3-fused", 3, {}),
    ("w4-fused", 4, {}),
    ("classic-close", None, {}),
]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:10]


def server_cmd(a, width):
    cmd = [a.server, "-m", a.target, "--port", str(a.port), "--ignore-eos",
           "--max-batch-size", "1"]
    if width is not None:
        cmd += ["--model-draft", a.drafter, "--draft-kind", "dflash",
                "--draft-block-size", str(width)]
    return cmd


def run_arm(a, tag, width, arm_env, prompt):
    gate_wait = hostgate.wait_quiet(log=sys.stderr)
    driver_wait, nvrm_window, nvrm_before = hostgate.driver_gate(log=sys.stderr)
    mem_wait, mem_avail = hostgate.mem_gate(log=sys.stderr)

    log_path = os.path.join(a.logdir, f"server.{tag}.log")
    log = open(log_path, "w")
    env = dict(os.environ)
    env.setdefault("MLX_ENABLE_TF32", "1")
    env.setdefault("RUST_LOG", "info")
    env.setdefault("MLX_CUDA_ARCHITECTURES", "121")
    env.update(arm_env)
    srv = subprocess.Popen(server_cmd(a, width), env=env, stdout=log,
                           stderr=subprocess.STDOUT, start_new_session=True)
    rec = {"arm": tag, "width": width, "arm_env": arm_env,
           "cmd": server_cmd(a, width), "log": log_path,
           "gate_wait_s": gate_wait, "driver_wait_s": driver_wait,
           "mem_wait_s": mem_wait, "nvrm_total_before": nvrm_before,
           "ci_job_running": hostgate.ci_job_running(),
           "load1_before": os.getloadavg()[0]}
    try:
        if not wait_health(a.port):
            rec["error"] = "server never became healthy"
            return rec
        model_id = first_model_id(a.port)
        # Discarded: the first request pays MLX kernel and graph compilation.
        complete(a.port, prompt, a.max_tokens, model_id)
        runs = []
        for _ in range(a.n):
            wall, text, chunks, ptok, ctok = complete(
                a.port, prompt, a.max_tokens, model_id
            )
            runs.append({"wall_s": round(wall, 3),
                         "e2e_tok_s": round(a.max_tokens / wall, 3),
                         "completion_tokens": ctok, "chars": len(text),
                         "sha": sha(text), "text": text})
        rec["runs"] = runs
        rec["prompt_tokens"] = ptok
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
    body = open(log_path, errors="replace").read()
    rec["declined_to_classic"] = DECLINE in body
    diags = []
    for m in DIAG_LINE.finditer(body):
        fields = {}
        for f in DIAG_FIELDS:
            mm = re.search(rf"\b{f}=([0-9.]+)", m.group(1))
            if mm:
                fields[f] = float(mm.group(1))
        if fields:
            diags.append(fields)
    rec["diagnostics"] = diags
    _, nvrm_after = hostgate.nvrm_counts()
    rec["nvrm_delta"] = nvrm_after - nvrm_before
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt-file", default=os.path.join(S1797, "prompt_retry.txt"))
    ap.add_argument("--port", type=int, default=18935)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", default=None)
    a = ap.parse_args()
    a.logdir = a.logdir or os.path.dirname(os.path.abspath(a.out))
    os.makedirs(a.logdir, exist_ok=True)
    prompt = open(a.prompt_file).read()

    with open(a.out, "a") as f:
        for tag, width, arm_env in ARMS:
            rec = run_arm(a, tag, width, arm_env, prompt)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            rs = [r["e2e_tok_s"] for r in rec.get("runs", [])]
            shas = sorted({r["sha"] for r in rec.get("runs", [])})
            print(
                f"[{tag}] "
                + (f"e2e {min(rs):.2f} to {max(rs):.2f} tok/s" if rs
                   else f"ERROR {rec.get('error')}")
                + f" | sha {shas} | declined={rec.get('declined_to_classic')}"
                + f" | nvrm +{rec.get('nvrm_delta')}",
                flush=True,
            )
    print("OPTIONS DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
