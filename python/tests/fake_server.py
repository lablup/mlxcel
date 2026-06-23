"""A tiny stdlib-only fake of the mlxcel HTTP server for lifecycle tests.

Binds either a Unix domain socket or a TCP port and serves the subset of routes
the client touches: ``/health`` (HTTP 503 for a short warmup window, then 200),
``/v1/models``, ``/v1/completions`` (plain and SSE), ``/v1/chat/completions``
(plain and SSE), ``/tokenize``, and ``/detokenize``. Responses are canned.

Run as a script for the lifecycle tests::

    python fake_server.py --uds /tmp/x.sock [--ready-after 0.3] [--model alias]
    python fake_server.py --host 127.0.0.1 --port 8080
    python fake_server.py --uds /tmp/x.sock --crash    # exit before binding

Uses only the standard library so it can stand in for the real binary in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn, UnixStreamServer
from typing import Any, Dict

MODEL_ID = "fake-model"
START_TIME = time.monotonic()
READY_AFTER = 0.0


def _now() -> int:
    return int(time.time())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # noqa: D401 - silence default logging
        return

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed: Dict[str, Any] = json.loads(raw.decode("utf-8"))
            return parsed
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _is_ready(self) -> bool:
        return (time.monotonic() - START_TIME) >= READY_AFTER

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/health":
            if not self._is_ready():
                self._send_json(503, {"status": "loading model"})
                return
            self._send_json(
                200,
                {"status": "ok", "model": MODEL_ID, "context_size": 0, "tool_call_parser": None},
            )
            return
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": _now(),
                            "owned_by": "user",
                        }
                    ],
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = self._read_body()
        stream = bool(body.get("stream"))

        if self.path == "/v1/completions":
            if stream:
                self._stream_completions()
            else:
                self._send_json(200, self._completion_payload())
            return
        if self.path == "/v1/chat/completions":
            if stream:
                self._stream_chat()
            else:
                self._send_json(200, self._chat_payload())
            return
        if self.path == "/tokenize":
            content = str(body.get("content", ""))
            tokens = [ord(c) % 256 for c in content]
            self._send_json(200, {"tokens": tokens})
            return
        if self.path == "/detokenize":
            tokens = body.get("tokens", [])
            content = "".join(chr(int(t)) for t in tokens)
            self._send_json(200, {"content": content})
            return
        self._send_json(404, {"error": "not found"})

    # -- canned payloads ---------------------------------------------------

    def _completion_payload(self) -> Dict[str, Any]:
        return {
            "id": "cmpl-fake",
            "object": "text_completion",
            "created": _now(),
            "model": MODEL_ID,
            "choices": [
                {"index": 0, "text": "hello world", "finish_reason": "stop", "logprobs": None}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    def _chat_payload(self) -> Dict[str, Any]:
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": _now(),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    def _write_sse(self, chunks: list[Dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _stream_completions(self) -> None:
        base = {
            "id": "cmpl-fake",
            "object": "text_completion",
            "created": _now(),
            "model": MODEL_ID,
        }
        chunks = [
            {**base, "choices": [{"index": 0, "text": "hello ", "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "text": "world", "finish_reason": "stop"}]},
        ]
        self._write_sse(chunks)

    def _stream_chat(self) -> None:
        base = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": _now(),
            "model": MODEL_ID,
        }
        chunks = [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {**base, "choices": [{"index": 0, "delta": {"content": "hi "}}]},
            {
                **base,
                "choices": [{"index": 0, "delta": {"content": "there"}, "finish_reason": "stop"}],
            },
        ]
        self._write_sse(chunks)


class ThreadingUnixServer(ThreadingMixIn, UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class ThreadingTCPServer(ThreadingMixIn, TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    global READY_AFTER, MODEL_ID

    parser = argparse.ArgumentParser()
    parser.add_argument("--uds")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ready-after", type=float, default=0.0)
    parser.add_argument("--model", default="fake-model")
    parser.add_argument("--crash", action="store_true", help="exit before binding")
    # Accept and ignore the flags the client always passes so this can stand in
    # for `mlxcel serve` invoked by ManagedServer.
    parser.add_argument("serve", nargs="?")
    parser.add_argument("-m", "--model-path")
    parser.add_argument("--api-key")
    parser.add_argument("--ctx-size")
    parser.add_argument("--n-predict")
    parser.add_argument("-a", "--alias")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args, _unknown = parser.parse_known_args()

    READY_AFTER = args.ready_after
    MODEL_ID = args.alias or args.model

    if args.crash:
        # Emit some stderr so the lifecycle test can assert it is captured.
        sys.stderr.write("fatal: simulated startup failure\n")
        sys.stderr.flush()
        sys.exit(3)

    # Honor UDS mode signalled either by --uds or by `--port 0 --host <path>`
    # (the way ManagedServer invokes the real binary).
    uds = args.uds
    if uds is None and args.port == 0 and args.host:
        uds = args.host

    # Simulate the real server's startup ordering: warm up (sleep) before bind.
    if READY_AFTER > 0:
        time.sleep(0)  # binding happens immediately; /health gates readiness.

    if uds:
        if os.path.exists(uds):
            os.unlink(uds)
        server: Any = ThreadingUnixServer(uds, Handler)
        sys.stderr.write(f"Starting mlxcel server on unix:{uds}\n")
    else:
        host = args.host or "127.0.0.1"
        port = args.port or 8080
        server = ThreadingTCPServer((host, port), Handler)
        sys.stderr.write(f"Starting mlxcel server on {host}:{port}\n")
    sys.stderr.flush()

    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if uds and os.path.exists(uds):
            try:
                os.unlink(uds)
            except OSError:
                pass


if __name__ == "__main__":
    # Reduce the chance of a noisy traceback on broken-pipe during shutdown.
    try:
        main()
    except (BrokenPipeError, ConnectionResetError):
        pass
    except OSError as exc:
        sys.stderr.write(f"fake_server bind error: {exc}\n")
        sys.exit(4)
