#!/usr/bin/env python3
"""Validate the public CLI and streaming OpenAI LLaVA XLA surfaces."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

FIXTURE_SHA256 = "5e7d54e8a7d21802378c87d2d70cf551e29739fe27599ddf129ebccdad1e6261"


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def get(url: str, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def wait_ready(base_url: str, process: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            fail(f"server exited before readiness with status {process.returncode}")
        try:
            get(f"{base_url}/health", timeout=2.0)
            return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    fail(f"server did not become healthy within {timeout:.0f}s")


def metric_value(metrics: str, name: str) -> float:
    match = re.search(rf"(?m)^{re.escape(name)} ([0-9.eE+-]+)$", metrics)
    if match is None:
        fail(f"missing metric {name}")
    return float(match.group(1))


def run_cli(args: argparse.Namespace, case: dict[str, Any], env: dict[str, str]) -> None:
    command = [
        str(args.mlxcel_bin),
        "generate",
        "--model",
        str(args.model),
        "--prompt",
        case["user_prompt"],
        "--image",
        str(args.image),
        "--max-tokens",
        str(args.max_new),
        "--temp",
        "0",
    ]
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            f"CLI exited {completed.returncode}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    prefix = f"Generating...\n{case['user_prompt']}"
    if prefix not in completed.stdout or "\n\n[Generated " not in completed.stdout:
        fail(f"unrecognized CLI output:\n{completed.stdout}")
    generated = completed.stdout.split(prefix, 1)[1].rsplit("\n\n[Generated ", 1)[0]
    if generated != case["greedy_text"]:
        fail(
            f"CLI text mismatch: expected {case['greedy_text']!r}, got {generated!r}"
        )
    print(f"[cli] PASS content={generated!r}")


def stream_server(
    args: argparse.Namespace,
    case: dict[str, Any],
    base_url: str,
) -> None:
    request_body = {
        "model": args.alias,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri(args.image)},
                    },
                    {"type": "text", "text": case["user_prompt"]},
                ],
            }
        ],
        "max_tokens": args.max_new,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(request_body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    events: list[dict[str, Any]] = []
    done = False
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                done = True
                break
            events.append(json.loads(data))
    if not done:
        fail("stream ended without [DONE]")
    if not events or events[0]["choices"][0]["delta"].get("role") != "assistant":
        fail("first streaming event did not declare the assistant role")

    content: list[str] = []
    finish_index = None
    usage_index = None
    finish_reason = None
    usage = None
    for index, event in enumerate(events):
        choices = event.get("choices", [])
        if choices:
            choice = choices[0]
            piece = choice.get("delta", {}).get("content")
            if piece:
                if finish_index is not None:
                    fail("content event appeared after finish event")
                content.append(piece)
            if choice.get("finish_reason") is not None:
                if finish_index is not None:
                    fail("multiple finish events")
                finish_index = index
                finish_reason = choice["finish_reason"]
        if event.get("usage") is not None:
            if event.get("choices") != []:
                fail("usage event must have an empty choices array")
            usage_index = index
            usage = event["usage"]

    if finish_index is None or usage_index is None or usage_index <= finish_index:
        fail("stream must order content, finish, usage, then [DONE]")
    generated = "".join(content)
    if generated != case["greedy_text"]:
        fail(
            f"server text mismatch: expected {case['greedy_text']!r}, got {generated!r}"
        )
    if finish_reason != "length":
        fail(f"expected finish_reason='length', got {finish_reason!r}")
    logical_prompt_tokens = len(case["unexpanded_input_ids"])
    expected_usage = {
        "prompt_tokens": logical_prompt_tokens,
        "completion_tokens": args.max_new,
        "total_tokens": logical_prompt_tokens + args.max_new,
    }
    for key, expected in expected_usage.items():
        if usage.get(key) != expected:
            fail(f"usage.{key}: expected {expected}, got {usage.get(key)}")
    print(
        f"[server] PASS content={generated!r} finish_reason={finish_reason} "
        f"usage={expected_usage}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlxcel-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="local-task")
    parser.add_argument("--port", type=int, default=18062)
    parser.add_argument("--alias", default="llava-reference")
    parser.add_argument("--max-new", type=int, default=4)
    parser.add_argument("--context-capacity", type=int, default=1536)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    for path, label in (
        (args.mlxcel_bin, "mlxcel binary"),
        (args.model, "model"),
        (args.reference / "manifest.json", "reference manifest"),
        (args.image, "image"),
    ):
        if not path.exists():
            fail(f"{label} not found: {path}")
    actual_fixture_sha = hashlib.sha256(args.image.read_bytes()).hexdigest()
    if actual_fixture_sha != FIXTURE_SHA256:
        fail(
            "image fixture SHA-256 mismatch: "
            f"expected {FIXTURE_SHA256}, got {actual_fixture_sha}"
        )
    manifest = json.loads((args.reference / "manifest.json").read_text())
    case = next(case for case in manifest["cases"] if case["name"] == "image_text")
    if "greedy_text" not in case:
        fail("reference manifest predates greedy_text; recapture it with the oracle")

    env = os.environ.copy()
    env.update(
        MLXCEL_BACKEND="xla",
        MLXCEL_XLA_DEVICE=args.device,
        MLXCEL_XLA_CONTEXT_CAPACITY=str(args.context_capacity),
        MLXCEL_XLA_REFERENCE_EXPECT_UNEXPANDED_IDS=json.dumps(
            case["unexpanded_input_ids"], separators=(",", ":")
        ),
    )
    run_cli(args, case, env)

    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(args.mlxcel_bin),
        "serve",
        "--model",
        str(args.model),
        "--alias",
        args.alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--n-predict",
        str(args.max_new),
        "--max-batch-size",
        "4",
        "--max-batch-prefill",
        "1",
        "--metrics",
        "--no-warmup",
    ]
    with tempfile.NamedTemporaryFile(
        prefix="mlxcel-llava-server.", suffix=".log", delete=False
    ) as log:
        log_path = Path(log.name)
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
    try:
        wait_ready(base_url, process, args.timeout)
        stream_server(args, case, base_url)
        print(
            "[server-render] PASS unexpanded_input_ids="
            f"{case['unexpanded_input_ids']}"
        )
        metrics = get(f"{base_url}/metrics", timeout=10.0).decode()
        effective_prompt_tokens = case["arrays"]["expanded_token_ids"]["shape"][1]
        checks = {
            "mlxcel_batch_sequences_started": 1,
            "mlxcel_batch_sequences_completed": 1,
            "mlxcel_batch_prefill_tokens_total": effective_prompt_tokens,
            "mlxcel_batch_decode_tokens_total": args.max_new,
        }
        for name, minimum in checks.items():
            actual = metric_value(metrics, name)
            if actual < minimum:
                fail(f"{name}: expected >= {minimum}, got {actual}")
        print(f"[metrics] PASS {checks}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if process.returncode not in (0, -15):
            print(f"server log retained at {log_path}", file=sys.stderr)
        else:
            log_path.unlink(missing_ok=True)
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
