#!/usr/bin/env python3
"""Real checkpoint admission lifecycle, not a speed benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def request(base: str, route: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(base + route, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read().decode()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw
        return {"status": response.status, "content_type": response.headers.get("Content-Type"), "body": body, "elapsed_s": time.monotonic() - started}


def footprint(pid: int) -> dict:
    text = subprocess.run(["vmmap", "-summary", str(pid)], capture_output=True, text=True, check=True, timeout=60).stdout
    result = {}
    for field, pattern in [("peak", r"Physical footprint \(peak\):\s+([0-9.]+)([KMG])"), ("current", r"Physical footprint:\s+([0-9.]+)([KMG])")]:
        match = re.search(pattern, text)
        require(match is not None, f"missing vmmap {field}")
        result[field + "_bytes"] = int(float(match.group(1)) * {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)])
    return result


def admissions(path: Path) -> list[dict]:
    result = []
    for line in ANSI.sub("", path.read_text(errors="replace")).splitlines():
        if "request memory admission" in line:
            values = {k: int(v) for k, v in re.findall(r"\b([a-z_]+)=(\d+)", line)}
            values["admitted"] = "admitted=true" in line
            result.append(values)
    return result


def run(args: argparse.Namespace, repetition: int) -> dict:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log_path = Path(f"{args.output}.rep{repetition}.log")
    env = {**os.environ, "RUST_LOG": "info", "NO_COLOR": "1", "MLXCEL_REQUEST_MEMORY_BUDGET": args.budget}
    cmd = [args.binary, "--model", args.model, "--ctx-size", "8192", "--jinja", "--parallel", "1", "--host", "127.0.0.1", "--port", str(port), "--alias", "repro", "--offline"]
    if args.no_warmup:
        cmd.append("--no-warmup")
    if args.no_batch:
        cmd.append("--no-batch")
    result = {"rep": repetition, "binary_sha256": hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(), "command": cmd, "budget": args.budget, "load_before": os.getloadavg(), "requests": [], "log": str(log_path)}
    started = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 180
            while True:
                require(proc.poll() is None, f"server exited: {log_path.read_text()[-3000:]}")
                try:
                    ready = request(base, "/v1/models")
                    if ready["status"] == 200:
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                require(time.monotonic() < deadline, "server readiness deadline")
                time.sleep(0.25)
            result["ready_s"] = time.monotonic() - started
            result["startup_footprint"] = footprint(proc.pid)
            for route in ["/v1/completions", "/v1/chat/completions"]:
                for streaming in [False, True]:
                    for target, count in [("reject", 6000), ("accept_after_reject", 16)]:
                        filler = "benchmark " * count
                        body = {"model": "repro", "max_tokens": 8, "temperature": 0.0, "stream": streaming}
                        if route.endswith("/chat/completions"):
                            body["messages"] = [{"role": "user", "content": filler + " Reply OK."}]
                            body["chat_template_kwargs"] = {"enable_thinking": False}
                        else:
                            body["prompt"] = filler
                        before = len(admissions(log_path))
                        response = request(base, route, body)
                        response.update({"case": target, "route": route, "stream": streaming, "footprint": footprint(proc.pid)})
                        entries = admissions(log_path)
                        require(len(entries) > before, "request never reached real memory guard")
                        response["admission"] = entries[-1]
                        require(response["admission"]["prompt_tokens"] < 8192, "must reject within context")
                        if target == "reject":
                            require(response["status"] == 503, f"expected memory503: {response}")
                            require("text/event-stream" not in response["content_type"], "SSE headers committed on rejection")
                            error = response["body"]["error"]
                            require(error["type"] == error["code"] == "request_memory_budget_exceeded", "wrong typed error")
                            require(not response["admission"]["admitted"], "not a guard rejection")
                        else:
                            require(response["status"] == 200, f"following short failed: {response}")
                            require(response["admission"]["admitted"], "short not admitted")
                            if streaming:
                                require("data:" in response["body"] and "[DONE]" in response["body"], "stream incomplete")
                        health = request(base, "/health")
                        require(health["status"] == 200, "health failed after request")
                        response["health_status"] = health["status"]
                        result["requests"].append(response)
                        print(json.dumps({k: response[k] for k in ["case", "route", "stream", "status", "admission", "footprint"]}), flush=True)
            medium = request(base, "/v1/completions", {"model": "repro", "prompt": "benchmark " * 1500, "max_tokens": 8, "temperature": 0.0})
            require(medium["status"] == 200, f"medium request failed: {medium}")
            medium.update({"case": "accepted_1500", "admission": admissions(log_path)[-1], "footprint": footprint(proc.pid)})
            result["requests"].append(medium)
            if not args.no_batch:
                context = request(base, "/v1/completions", {"model": "repro", "prompt": "benchmark " * 8400, "max_tokens": 8, "stream": True})
                require(context["status"] == 400, f"context precedence: {context}")
                result["context_rejection"] = context
            result["final_health"] = request(base, "/health")
            require(result["final_health"]["status"] == 200, "final health not200")
            result["load_after"] = os.getloadavg()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", default="24G")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-batch", action="store_true")
    args = parser.parse_args()
    results = []
    for repetition in range(1, args.reps + 1):
        results.append(run(args, repetition))
        Path(args.output).write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
