#!/usr/bin/env python3
# Copyright 2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cold against warm time to first token of mlxcel-server over a PTX cache dir.

Used for issue #3503: does a CUDA kernel cache under a container volume root
(a 0777 root with MLXCEL_CACHE_TRUSTED_ROOT, or a group-writable root of a
private group) actually get hit on the second start.

One run starts the server with the given environment, waits for /health,
sends one streaming chat request, takes the time of the first content token,
and stops the server. A cold run empties the kernel cache first; a warm run
leaves it.

The timing is not the evidence that the cache was hit. Each run also records
two signals that can only move one way: whether the server printed the JIT
notice ("Compiling CUDA kernels"), and whether any file under the cache
directory was created, replaced or touched. A cold run must show both; a warm
run must show neither; an arm declared with --expect refused must print the
refusal line on every run. When a run disagrees with its arm the script exits
non-zero and prints no table, so a table always compares a miss with a hit.

Run it under the GPU lock on a GB10 host:

    gpu-lock run --tag ptx-cache-trust -- python3 scripts/bench/ptx_cache_trust_ttft.py \\
        --server target/release/mlxcel-server --model models/qwen3-4b-4bit \\
        --cache-dir /vol/cuda-ptx --env MLXCEL_CACHE_DIR=/vol \\
        --env MLXCEL_CACHE_TRUSTED_ROOT=/vol --runs 5 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

JIT_NOTICE = "Compiling CUDA kernels"
REFUSAL = "mlxcel: CUDA kernel disk cache disabled: "
IGNORED = "mlxcel: MLXCEL_CACHE_TRUSTED_ROOT ignored: "
SCRUBBED = (
    "MLX_PTX_CACHE_DIR",
    "MLXCEL_CACHE_DIR",
    "MLXCEL_CACHE_TRUSTED_ROOT",
    "MLXCEL_QUIET_JIT",
)


class HarnessError(RuntimeError):
    """A run could not be completed or contradicted its arm."""


def snapshot(cache_dir: pathlib.Path) -> dict[str, tuple[int, int]]:
    """Inode and mtime of every file under the cache directory."""
    files: dict[str, tuple[int, int]] = {}
    if not cache_dir.is_dir():
        return files
    for root, _dirs, names in os.walk(cache_dir):
        for name in names:
            path = pathlib.Path(root) / name
            stat = path.lstat()
            files[str(path.relative_to(cache_dir))] = (stat.st_ino, stat.st_mtime_ns)
    return files


def wait_for_health(port: int, server: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise HarnessError(
                f"the server exited with {server.returncode} before /health"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.05)
    raise HarnessError(f"/health did not answer within {timeout:.0f} s")


def first_token_seconds(port: int, model: str, timeout: float) -> float:
    """Seconds from sending a streaming chat request to its first content token."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Name three colors."}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                delta = json.loads(line[5:])["choices"][0].get("delta", {})
            except (ValueError, LookupError) as error:
                raise HarnessError(f"unreadable stream line {line!r}") from error
            if delta.get("content") or delta.get("reasoning_content"):
                return time.monotonic() - start
    raise HarnessError("the stream ended without a content token")


def one_run(
    args: argparse.Namespace, cold: bool, log_dir: pathlib.Path, index: int
) -> dict:
    cache_dir = pathlib.Path(args.cache_dir)
    if cold and cache_dir.exists():
        shutil.rmtree(cache_dir)
    before = snapshot(cache_dir)
    env = {key: value for key, value in os.environ.items() if key not in SCRUBBED}
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    log_path = log_dir / f"{'cold' if cold else 'warm'}-{index}.stderr"
    with open(log_path, "wb") as log:
        spawned = time.monotonic()
        server = subprocess.Popen(
            [args.server, "-m", args.model, "--port", str(args.port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=log,
        )
        try:
            wait_for_health(args.port, server, args.timeout)
            ready = time.monotonic() - spawned
            request = first_token_seconds(args.port, args.model, args.timeout)
            total = time.monotonic() - spawned
        finally:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    stderr = log_path.read_text(errors="replace")
    after = snapshot(cache_dir)
    return {
        "mode": "cold" if cold else "warm",
        "start_to_ready_s": ready,
        "request_to_first_token_s": request,
        "start_to_first_token_s": total,
        "jit_notice": JIT_NOTICE in stderr,
        "refusals": [line for line in stderr.splitlines() if REFUSAL in line],
        "ignored": [line for line in stderr.splitlines() if IGNORED in line],
        "files_before": len(before),
        "files_after": len(after),
        "files_changed": before != after,
        "load_1m": os.getloadavg()[0],
    }


def check_attribution(run: dict, expect: str) -> None:
    """Fail unless the hit-only signals agree with what the run claims to be."""
    what = f"{run['mode']} run ({expect})"
    if expect == "refused":
        if len(run["refusals"]) != 1 or not run["jit_notice"] or run["files_after"]:
            raise HarnessError(
                f"{what}: expected one refusal, a compile and no entries: {run}"
            )
        return
    if run["refusals"] or run["ignored"]:
        raise HarnessError(f"{what}: the cache was refused: {run}")
    if run["mode"] == "cold":
        if not run["jit_notice"] or not run["files_changed"] or run["files_before"]:
            raise HarnessError(f"{what}: a cold run must compile and publish: {run}")
    elif run["jit_notice"] or run["files_changed"] or not run["files_before"]:
        raise HarnessError(f"{what}: a warm run must hit and write nothing: {run}")


def summarize(runs: list[dict], key: str) -> str:
    values = [run[key] for run in runs]
    return (
        f"{statistics.median(values):.2f} "
        f"(min {min(values):.2f}, max {max(values):.2f}, n={len(values)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", required=True, help="mlxcel-server binary")
    parser.add_argument("--model", required=True, help="model directory")
    parser.add_argument(
        "--cache-dir", required=True, help="directory holding the PTX entries"
    )
    parser.add_argument(
        "--env", action="append", default=[], help="KEY=VALUE for the server"
    )
    parser.add_argument("--expect", choices=("accepted", "refused"), default="accepted")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--port", type=int, default=18503)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--label", default="arm")
    parser.add_argument("--json", help="write every run here")
    args = parser.parse_args()

    log_dir = pathlib.Path(tempfile.mkdtemp(prefix="ptx-cache-trust-"))
    modes = [True] if args.expect == "refused" else [True, False]
    runs: list[dict] = []
    try:
        for cold in modes:
            # A warm arm follows a populating run that is not timed.
            if not cold:
                check_attribution(one_run(args, True, log_dir, -1), args.expect)
            for index in range(args.runs):
                run = one_run(args, cold, log_dir, index)
                check_attribution(run, args.expect)
                runs.append(run)
    except HarnessError as error:
        print(f"{args.label}: attribution failed, no table: {error}", file=sys.stderr)
        print(f"server logs: {log_dir}", file=sys.stderr)
        return 1
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"label": args.label, "runs": runs}, indent=1)
        )
    print(f"## {args.label} ({args.expect})")
    for mode in ("cold", "warm"):
        subset = [run for run in runs if run["mode"] == mode]
        if not subset:
            continue
        loads = [run["load_1m"] for run in subset]
        print(
            f"{mode}: start to first token {summarize(subset, 'start_to_first_token_s')} s; "
            f"start to ready {summarize(subset, 'start_to_ready_s')} s; "
            f"request to first token {summarize(subset, 'request_to_first_token_s')} s; "
            f"jit notice {sum(run['jit_notice'] for run in subset)}/{len(subset)}; "
            f"runs that wrote cache files {sum(run['files_changed'] for run in subset)}/{len(subset)}; "
            f"load {min(loads):.2f}-{max(loads):.2f}"
        )
    shutil.rmtree(log_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
