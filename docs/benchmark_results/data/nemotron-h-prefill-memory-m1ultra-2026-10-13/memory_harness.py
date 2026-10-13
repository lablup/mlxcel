#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
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

import argparse
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3}
PEAK_RE = re.compile(r"Physical footprint \(peak\):\s+([0-9.]+)([KMG])")
MARKER = "[NemotronH] chunked SSD prefill forward"
LOAD_MARKER = "[NemotronH] chunked SSD prefill dispatch enabled"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request_json(url, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def wait_ready(proc, base, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup with {proc.returncode}")
        try:
            status, body = request_json(base + "/v1/models", timeout=3)
            if status == 200:
                return body["data"][0]["id"]
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError("server readiness timeout")


def vmmap_peak(pid):
    result = subprocess.run(["vmmap", "-summary", str(pid)], text=True, capture_output=True, check=True, timeout=60)
    match = PEAK_RE.search(result.stdout)
    if not match:
        raise RuntimeError("vmmap omitted physical footprint peak")
    return int(float(match.group(1)) * UNITS[match.group(2)])


def run_one(args, rep, target):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = pathlib.Path(args.output).with_suffix(f".rep{rep}.tokens{target}.log")
    command = [
        args.binary, "--model", args.model, "--ctx-size", "8192", "--jinja",
        "--host", "127.0.0.1", "--port", str(port), "--alias", "repro",
        "--parallel", "1", "--offline",
    ]
    load_before = os.getloadavg()
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True, env={**os.environ, "RUST_LOG": "info"})
        try:
            model_id = wait_ready(proc, base)
            ready_s = time.time() - started
            prompt = f"Run {rep}, target {target}. " + "benchmark " * max(1, target - 20)
            body = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "Read the filler and reply with the single word OK."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request_started = time.time()
            status, response = request_json(base + "/v1/chat/completions", body)
            request_s = time.time() - request_started
            peak_bytes = vmmap_peak(proc.pid)
            health_status, health = request_json(base + "/health", timeout=10)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
    logs = log_path.read_text(errors="replace")
    forward_marker = MARKER in logs
    load_marker = LOAD_MARKER in logs
    if args.marker == "absent" and (forward_marker or load_marker):
        raise RuntimeError("baseline unexpectedly emitted fixed-arm dispatch marker")
    if args.marker == "present" and not (forward_marker and load_marker):
        raise RuntimeError("fixed arm did not execute and log chunked SSD forward dispatch")
    if args.marker == "loadonly" and not load_marker:
        raise RuntimeError("preliminary fixed arm did not emit load-time chunk marker")
    usage = response.get("usage") or {}
    choice = (response.get("choices") or [{}])[0]
    return {
        "rep": rep,
        "target_tokens": target,
        "status": status,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "choice": choice,
        "peak_bytes": peak_bytes,
        "peak_gib": peak_bytes / 1024**3,
        "ready_s": ready_s,
        "request_s": request_s,
        "load_before": load_before,
        "load_after": os.getloadavg(),
        "health_status": health_status,
        "health": health,
        "forward_marker": forward_marker,
        "load_marker": load_marker,
        "log": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--marker", choices=["present", "absent", "loadonly"], required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--points", default="1500,3000,4500,6000")
    args = parser.parse_args()
    result = {
        "arm": args.arm,
        "binary": args.binary,
        "model": args.model,
        "marker_expectation": args.marker,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [],
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2))
    for rep in range(1, args.reps + 1):
        for target in [int(value) for value in args.points.split(",")]:
            run = run_one(args, rep, target)
            result["runs"].append(run)
            pathlib.Path(args.output).write_text(json.dumps(result, indent=2))
            print(json.dumps({key: run[key] for key in ["rep", "target_tokens", "prompt_tokens", "peak_gib", "request_s", "load_before", "forward_marker"]}), flush=True)
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
