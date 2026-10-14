#!/usr/bin/env python3
"""Gemma 4 MTP vs classic decode on CUDA, interleaved rounds with a null arm.

Each round runs three servers in order: classic (open bracket), MTP at the
given draft width, classic (close bracket). The two classic arms run the same
binary with the same flags, so their paired delta is the method's noise floor
(the null arm); an MTP speedup inside that spread is unresolved. Every arm
discards one warm-up request, then times `--n` streaming completions of a
fixed token budget at temperature 0.

Reuses the #1797 helpers (streaming `complete`, `wait_health`) and the #1820
host gate exactly as `price_options.py` (#1935) does.

  mtp_rounds.py --server BIN --target DIR --drafter DIR --out run.jsonl
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys

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
complete = _sweep["complete"]

MTP_DIAG = re.compile(r"MTP round-loop diagnostics ([^\n]*)")
MTP_FIELDS = ("rounds", "proposed_tokens", "accepted_draft_tokens", "acceptance_rate",
              "generated_tokens")
DECLINE = "MTP declined"
PROBE_PASS = "MTP exactness probe passed"
BURST = "Speculative burst completed"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def server_cmd(a, width):
    cmd = [a.server, "-m", a.target, "--port", str(a.port), "--ignore-eos",
           "--max-batch-size", "1", "--parallel", "1"] + a.server_arg
    if width is not None:
        cmd += ["--model-draft", a.drafter, "--draft-kind", "mtp",
                "--draft-block-size", str(width)]
    return cmd


def run_arm(a, tag, width, prompt):
    # --no-gate skips only the sustained-quiet CPU wait (the CI runner shares
    # this host); the driver and memory gates still run, and the record keeps
    # ci_job_running and load1 so a noisy arm stays visible.
    gate_wait = None if a.no_gate else hostgate.wait_quiet(log=sys.stderr)
    driver_wait, _window, nvrm_before = hostgate.driver_gate(log=sys.stderr)
    mem_wait, _avail = hostgate.mem_gate(log=sys.stderr)
    log_path = os.path.join(a.logdir, f"server.{tag}.log")
    env = dict(os.environ)
    env.setdefault("RUST_LOG", "info")
    rec = {"arm": tag, "width": width, "cmd": server_cmd(a, width), "log": log_path,
           "gate_wait_s": gate_wait, "driver_wait_s": driver_wait, "mem_wait_s": mem_wait,
           "ci_job_running": hostgate.ci_job_running(), "load1_before": os.getloadavg()[0]}
    with open(log_path, "w") as log:
        srv = subprocess.Popen(server_cmd(a, width), env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if not wait_health(a.port):
                rec["error"] = "server never became healthy"
                return rec
            model_id = first_model_id(a.port)
            complete(a.port, prompt, a.max_tokens, model_id)  # discarded warm-up
            runs = []
            for _ in range(a.n):
                wall, text, _chunks, ptok, ctok = complete(a.port, prompt, a.max_tokens, model_id)
                runs.append({"wall_s": round(wall, 3), "e2e_tok_s": round(a.max_tokens / wall, 3),
                             "completion_tokens": ctok, "sha": sha(text)})
            rec["runs"] = runs
            rec["prompt_tokens"] = ptok
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
                srv.wait(timeout=120)
            except Exception:  # noqa: BLE001
                os.killpg(srv.pid, signal.SIGKILL)
    body = open(log_path, errors="replace").read()
    rec["declined"] = DECLINE in body
    rec["probe_passed"] = PROBE_PASS in body
    rec["bursts"] = body.count(BURST)
    diags = []
    for line in MTP_DIAG.findall(body):
        fields = {}
        for f in MTP_FIELDS:
            m = re.search(rf"\b{f}=([0-9.]+)", line)
            if m:
                fields[f] = float(m.group(1))
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
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt-file", default=os.path.join(S1797, "prompt_retry.txt"))
    ap.add_argument("--port", type=int, default=18960)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gate", action="store_true")
    # The timed requests repeat the warm-up's prompt. With the prompt cache on,
    # classic serves them from a whole-prompt hit while the MTP burst prefills
    # cold, so the arms would differ in prefill work and in output bytes.
    ap.add_argument("--server-arg", action="append", default=[],
                    help="extra mlxcel-server argument for every arm (repeatable)")
    a = ap.parse_args()
    a.logdir = os.path.dirname(os.path.abspath(a.out))
    prompt = open(a.prompt_file).read()
    with open(a.out, "a") as out:
        for r in range(a.rounds):
            for tag, width in ((f"r{r}-classic-open", None), (f"r{r}-mtp-w{a.width}", a.width),
                               (f"r{r}-classic-close", None)):
                rec = run_arm(a, tag, width, prompt)
                rec["round"] = r
                out.write(json.dumps(rec) + "\n")
                out.flush()
                rs = [x["e2e_tok_s"] for x in rec.get("runs", [])]
                print(f"[{tag}] {rs or rec.get('error')} sha={sorted({x['sha'] for x in rec.get('runs', [])})}"
                      f" declined={rec['declined']} bursts={rec['bursts']} nvrm+{rec['nvrm_delta']}",
                      flush=True)
    print("ROUNDS DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
