#!/usr/bin/env python3
"""Issue #1935: capture one served arm's token transcript AND its per-round
DFlash round-loop transcript, so the round algebra can be checked offline.

One server per arm, `--ignore-eos --max-batch-size 1`, non-streaming
`/v1/completions` at temperature 0 with `logprobs`, repeated `--n` times
against the SAME server process so a cross-request state difference (burst 1
against burst 2) shows up as two different token lists rather than as one
number.

Identity only: nothing here is a timing measurement, so the host gate is the
foreign-model check alone.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

FOREIGN = ("mlxcel", "mlxcel-server", "python3", "python")


def foreign_model_procs():
    """Processes that could be holding the GPU, matched on /proc/<pid>/comm
    exactly and with stopped ones dropped. Our own renamed server is excluded
    by construction (its comm is mlxcel1935-serv)."""
    hits = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            comm = open(f"/proc/{pid}/comm").read().strip()
            if comm not in ("mlxcel", "mlxcel-server"):
                continue
            state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
            if state == "T":
                continue
            cwd = os.readlink(f"/proc/{pid}/cwd")
            hits.append((pid, comm, cwd))
        except OSError:
            continue
    return hits


def wait_health(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def first_model_id(port, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                data = json.load(r)
                return data["data"][0]["id"]
        except Exception:
            time.sleep(1)
    raise RuntimeError("no model id")


def complete(port, prompt, max_tokens, model, timeout=1800, want_logprobs=True):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if want_logprobs:
        payload["logprobs"] = 5
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter")
    ap.add_argument("--width", default="classic")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--port", type=int, default=18935)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--extra-env", default="", help="k=v,k=v applied to the server process")
    a = ap.parse_args()

    tag = a.tag or f"w{a.width}"
    os.makedirs(a.outdir, exist_ok=True)
    prompt = open(a.prompt_file).read()

    foreign = foreign_model_procs()
    if foreign:
        print(f"[1935] foreign model processes present, refusing: {foreign}", file=sys.stderr)
        return 2

    run_dir = os.path.join(a.outdir, "bin")
    os.makedirs(run_dir, exist_ok=True)
    run_bin = os.path.join(run_dir, "mlxcel1935-server")
    if not os.path.exists(run_bin) or os.path.getmtime(run_bin) < os.path.getmtime(a.server):
        subprocess.run(["/bin/cp", "-f", a.server, run_bin], check=True)
        os.chmod(run_bin, 0o755)

    cmd = [run_bin, "-m", a.target, "--port", str(a.port), "--ignore-eos",
           "--max-batch-size", "1"]
    if a.width != "classic":
        cmd += ["--draft-model", a.drafter, "--draft-kind", "dflash",
                "--draft-block-size", str(a.width)]

    env = dict(os.environ)
    env.setdefault("MLX_ENABLE_TF32", "1")
    env.setdefault("MLX_CUDA_ARCHITECTURES", "121")
    env["RUST_LOG"] = (
        "info,mlxcel_core::drafter::dflash::round_loop=debug,"
        "mlxcel::server::batch::dflash_target=debug"
    )
    for pair in filter(None, a.extra_env.split(",")):
        k, _, v = pair.partition("=")
        env[k] = v

    log_path = os.path.join(a.outdir, f"server.{tag}.log")
    log = open(log_path, "w")
    srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
    rec = {"tag": tag, "width": a.width, "cmd": cmd, "log": log_path,
           "extra_env": a.extra_env, "responses": []}
    try:
        if not wait_health(a.port):
            rec["error"] = "server never became healthy"
        else:
            model_id = first_model_id(a.port)
            rec["model_id"] = model_id
            for i in range(a.n):
                try:
                    resp = complete(a.port, prompt, a.max_tokens, model_id)
                except urllib.error.HTTPError as exc:
                    # A server that refuses `logprobs` must not cost the arm:
                    # the round transcript carries the ids either way, and the
                    # record says omitting logprobs does not move the output.
                    print(f"[1935] logprobs refused ({exc.code}), retrying without",
                          file=sys.stderr)
                    resp = complete(a.port, prompt, a.max_tokens, model_id,
                                    want_logprobs=False)
                choice = resp["choices"][0]
                lp = choice.get("logprobs") or {}
                rec["responses"].append(
                    {
                        "index": i,
                        "text": choice["text"],
                        "tokens": lp.get("tokens"),
                        "token_logprobs": lp.get("token_logprobs"),
                        "usage": resp.get("usage"),
                    }
                )
                print(f"[1935] {tag} request {i}: "
                      f"{len(lp.get('tokens') or [])} tokens, "
                      f"{len(choice['text'])} chars", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
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

    out = os.path.join(a.outdir, f"arm.{tag}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[1935] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
