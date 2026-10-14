#!/usr/bin/env python3
"""How fast MTP serving spends MLX's CUDA graph-cache lifetime budget.

MLX keys its CUDA graph cache by graph topology and throws a fatal "Cache
thrashing" error once lifetime misses pass twice the capacity (#818). Shapes
that change every round therefore spend a budget that never refills. This
starts one MTP server with a deliberately small `MLX_CUDA_GRAPH_CACHE_SIZE`
and serves batches (the same primer + B rows as `mtp_batched_rounds.py`) until
the server aborts or `--batches` is reached, so B=1 MTP and the per-row
batched MTP of issue #2190 can be compared by how many tokens each serves on
the same budget.

  graph_cache_budget.py --server BIN --target DIR --drafter DIR --batch 2 \
      --cache-size 100 --out budget-b2.json
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_batched_rounds as rounds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--cache-size", type=int, default=100)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--port", type=int, default=18975)
    ap.add_argument("--prompt-file", default=os.path.join(rounds.S1797, "prompt_retry.txt"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    body = open(a.prompt_file).read()
    prompts = [f"{rounds.OPENINGS[i % len(rounds.OPENINGS)]}\n\n{body}" for i in range(a.batch)]
    log_path = os.path.splitext(a.out)[0] + ".server.log"
    env = dict(os.environ, RUST_LOG="info", MLX_CUDA_GRAPH_CACHE_SIZE=str(a.cache_size),
               MLXCEL_ENABLE_MTP_BATCH="1", MLXCEL_ENABLE_MTP_BATCH_RAGGED="1",
               MLXCEL_MTP_TICK_SLICE="0")
    cmd = [a.server, "-m", a.target, "--port", str(a.port), "--max-batch-size", str(a.batch),
           "--parallel", str(a.batch), "--no-cache-prompt", "--max-batch-prefill", "1",
           "--model-draft", a.drafter, "--draft-kind", "mtp", "--draft-block-size", "4"]
    rec = {"cmd": cmd, "batch": a.batch, "cache_size": a.cache_size, "batches": []}
    with open(log_path, "w") as log:
        srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
        try:
            if not rounds.wait_health(a.port):
                rec["error"] = "server never became healthy"
            else:
                model_id = rounds.first_model_id(a.port)
                for i in range(a.batches):
                    try:
                        wall, _, results = rounds.timed_batch(
                            argparse.Namespace(port=a.port, max_tokens=a.max_tokens),
                            prompts, model_id)
                    except Exception as exc:  # noqa: BLE001 - the abort is the result
                        rec["stopped"] = f"batch {i}: {type(exc).__name__}: {exc}"
                        break
                    rec["batches"].append({"wall_s": round(wall, 3),
                                           "tokens": sum(r[4] or 0 for r in results)})
                    if srv.poll() is not None:
                        rec["stopped"] = f"server exited after batch {i}"
                        break
        finally:
            time.sleep(2)
            rec["server_alive"] = srv.poll() is None
            try:
                os.killpg(srv.pid, signal.SIGTERM)
                srv.wait(timeout=120)
            except Exception:  # noqa: BLE001
                os.killpg(srv.pid, signal.SIGKILL)
    text = open(log_path, errors="replace").read()
    rec["thrash_abort"] = "Cache thrashing" in text
    rec["row_tokens_served"] = sum(b["tokens"] for b in rec["batches"])
    with open(a.out, "w") as out:
        json.dump(rec, out, indent=1)
    print(json.dumps({k: rec[k] for k in ("batch", "cache_size", "row_tokens_served",
                                          "thrash_abort", "server_alive")}
                     | {"batches_served": len(rec["batches"]), "stopped": rec.get("stopped")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
