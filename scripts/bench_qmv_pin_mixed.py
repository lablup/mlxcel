#!/usr/bin/env python3
"""Mixed-workload client for the qmv_wide narrow-pin tax (issue #1261).

One MTP stream plus N classic streams against a running ``mlxcel-server``,
reading the classic streams' decode throughput only. This is the second arm
of the issue #1261 measurement: the B-sweep (``bench_qmv_wide_pin.sh sweep``)
prices the narrow pin on batched decode in isolation, and this harness prices
it in the mixed shape the pin actually arises in, where an MTP stream bought
byte-identity for itself and everything admitted beside it pays the kernel.

How the two request classes are kept distinct, since with a drafter loaded
every eligible request wants speculative service: the server must run with
``MLXCEL_MTP_SLICE_GRANT_ROUNDS=0``, which disables slice-slot rotation
(issue #746). The first stream then holds the tick-cooperative MTP slice
slot for its whole generation, and every eligible request arriving while the
slot is busy falls back to classic decode, exactly the pre-#746 behaviour.
That gives one MTP stream and N classic rows deterministically, with the
same env in both kernel arms so the comparison isolates the kernel.

Phases:

1. Wait for the server to answer ``/v1/models``.
2. Start ONE long streaming request (the MTP stream) and wait for its first
   token. On the first window of a server process this includes the
   exactness probe. Then wait ``--settle-s`` so the slice loop is steady.
3. Fire ``--classic-streams`` identical streaming requests concurrently
   (the classic streams) and record each one's TTFT and decode rate.
4. When the last classic stream finishes, close the MTP stream's connection
   and report. A window is only valid if the MTP stream was still decoding
   when the last classic stream finished; the overlap fraction is printed
   and checked so a window that quietly lost its MTP stream cannot pass as
   a mixed measurement.

The classic streams' decode tok/s is the reported quantity. The MTP
stream's own throughput is deliberately not compared across kernel arms:
its generated text differs between them by construction.

Only the Python standard library is used, matching the sibling harnesses
(``bench_serving_concurrency.py``, ``bench_mixed_step_admission.py``).

The pairing must be one whose narrow retry actually passes on the host, or
arm A has no pin to price; on M3 Ultra that is the Qwen pairing (the Gemma
31B + bf16 pairing probes non-identical under both kernels there and
declines MTP).

Example (arm A, the default-env narrow pin):
    MLXCEL_MTP_SLICE_GRANT_ROUNDS=0 MLXCEL_ENABLE_MTP_B1=1 MLXCEL_MTP_ADAPTIVE=0 \
        target/release/mlxcel-server -m models/qwen3.8-27b-4bit \
        --model-draft models/qwen3.8-27b-mtp-4bit \
        --draft-block-size 3 --parallel 8 --metrics --port 8114 &
    python3 scripts/bench_qmv_pin_mixed.py --port 8114 --classic-streams 4
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
from dataclasses import dataclass


@dataclass
class StreamResult:
    """Timing of one streaming request."""

    ok: bool
    ttft_s: float | None = None
    decode_tok_s: float | None = None
    completion_tokens: int = 0
    start_s: float = 0.0
    first_token_s: float = 0.0
    end_s: float = 0.0
    error: str | None = None


_BASE_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the benchmark harness "
    "measures prefill and decode throughput under concurrent streaming load. "
)
_TOKENS_PER_WORD = 1.3

_MTP_PROMPT = (
    "Write a long, detailed essay about the history of numerical computing, "
    "covering mechanical calculators, the stored-program concept, floating "
    "point arithmetic, vector supercomputers, and modern accelerators. Use "
    "flowing prose with no lists."
)


def build_prompt(prompt_tokens: int) -> str:
    words_per_copy = len(_BASE_SENTENCE.split())
    target_words = max(words_per_copy, int(prompt_tokens / _TOKENS_PER_WORD))
    copies = max(1, target_words // words_per_copy + 1)
    return (_BASE_SENTENCE * copies).strip()


def wait_ready(host: str, port: int, timeout_s: float) -> str:
    """Poll ``/v1/models`` until it answers, returning the model id."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            if resp.status == 200:
                data = json.loads(body)
                models = data.get("data") or []
                if models and models[0].get("id"):
                    return str(models[0]["id"])
                return "default"
        except (OSError, ValueError):
            pass
        time.sleep(1.0)
    raise SystemExit(f"server on {host}:{port} not ready after {timeout_s:.0f}s")


class StreamRunner(threading.Thread):
    """One streaming chat completion on its own thread.

    ``stop()`` closes the connection from the client side, which is how the
    MTP stream is ended once the classic window has been measured.
    """

    def __init__(
        self,
        host: str,
        port: int,
        model: str,
        prompt: str,
        max_tokens: int,
        timeout: float,
    ) -> None:
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._model = model
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._conn: http.client.HTTPConnection | None = None
        self._stop_event = threading.Event()
        self.result = StreamResult(ok=False, error="not started")
        self.first_token = threading.Event()
        self.last_token_s = 0.0

    def stop(self) -> None:
        self._stop_event.set()
        conn = self._conn
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def run(self) -> None:  # noqa: C901 - one linear protocol loop
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": self._prompt}],
                "max_tokens": self._max_tokens,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        start = time.perf_counter()
        ttft: float | None = None
        delta_tokens = 0
        usage_tokens: int | None = None
        try:
            self._conn = http.client.HTTPConnection(
                self._host, self._port, timeout=self._timeout
            )
            self._conn.request(
                "POST", "/v1/chat/completions", body=payload, headers=headers
            )
            resp = self._conn.getresponse()
            if resp.status != 200:
                body = resp.read().decode("utf-8", "replace")[:200]
                self.result = StreamResult(ok=False, error=f"HTTP {resp.status}: {body}")
                return
            buf = b""
            while not self._stop_event.is_set():
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk
                if not buf.endswith(b"\n"):
                    continue
                line = buf.strip()
                buf = b""
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:") :].strip()
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                usage = event.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    usage_tokens = int(usage["completion_tokens"])
                for choice in event.get("choices", []) or []:
                    delta = choice.get("delta") or {}
                    # Reasoning models stream their thinking channel as
                    # `reasoning_content`; both channels are decoded tokens
                    # (same fix as bench_serving_concurrency.py).
                    content = delta.get("content") or delta.get("reasoning_content")
                    if content:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - start
                            self.first_token.set()
                        self.last_token_s = now
                        delta_tokens += 1
        except (OSError, http.client.HTTPException, AttributeError, ValueError) as exc:
            # AttributeError / ValueError cover the race where stop()
            # closes the connection while the read loop is inside
            # http.client (its file object becomes None mid-read); a
            # stop-induced close is expected for the MTP stream and its
            # partial stats below are still the measurement.
            if not self._stop_event.is_set():
                self.result = StreamResult(ok=False, error=str(exc))
                return
        finally:
            if self._conn is not None:
                try:
                    self._conn.close()
                except OSError:
                    pass

        end = time.perf_counter()
        completion_tokens = usage_tokens if usage_tokens is not None else delta_tokens
        decode_tok_s: float | None = None
        if ttft is not None and completion_tokens > 1:
            span = (self.last_token_s or end) - (start + ttft)
            if span > 0:
                decode_tok_s = (completion_tokens - 1) / span
        self.result = StreamResult(
            ok=True,
            ttft_s=ttft,
            decode_tok_s=decode_tok_s,
            completion_tokens=completion_tokens,
            start_s=start,
            first_token_s=start + (ttft or 0.0),
            end_s=end,
        )


def run_window(args: argparse.Namespace, model: str, window: int) -> dict:
    """One mixed window: MTP stream up, then N classic streams measured."""
    mtp = StreamRunner(
        args.host, args.port, model, _MTP_PROMPT, args.mtp_max_tokens, args.timeout
    )
    mtp_started = time.perf_counter()
    mtp.start()
    if not mtp.first_token.wait(timeout=args.mtp_first_token_timeout):
        mtp.stop()
        mtp.join(timeout=10)
        raise SystemExit(
            f"window {window}: MTP stream produced no token within "
            f"{args.mtp_first_token_timeout:.0f}s ({mtp.result.error})"
        )
    mtp_ttft = time.perf_counter() - mtp_started
    time.sleep(args.settle_s)

    prompt = build_prompt(args.classic_prompt_tokens)
    classics = [
        StreamRunner(
            args.host, args.port, model, prompt, args.classic_max_tokens, args.timeout
        )
        for _ in range(args.classic_streams)
    ]
    classic_start = time.perf_counter()
    for c in classics:
        c.start()
    for c in classics:
        c.join()
    classic_end = time.perf_counter()

    # The MTP stream must have outlived the classic window for the window to
    # count as mixed. last_token_s is the wall-clock of its latest token.
    mtp_alive_until = mtp.last_token_s
    mtp.stop()
    mtp.join(timeout=15)

    window_span = classic_end - classic_start
    overlap = 0.0
    if window_span > 0:
        overlap = max(0.0, min(mtp_alive_until, classic_end) - classic_start)
        overlap /= window_span
    valid = overlap >= args.min_overlap

    rows = []
    for i, c in enumerate(classics):
        r = c.result
        rows.append(
            {
                "stream": i,
                "ok": r.ok,
                "ttft_s": round(r.ttft_s, 3) if r.ttft_s is not None else None,
                "decode_tok_s": round(r.decode_tok_s, 2)
                if r.decode_tok_s is not None
                else None,
                "completion_tokens": r.completion_tokens,
                "error": r.error if not r.ok else None,
            }
        )
    ok_rates = [
        r["decode_tok_s"]
        for r in rows
        if r["ok"] and r["decode_tok_s"] is not None
    ]
    summary = {
        "window": window,
        "valid": valid,
        "overlap": round(overlap, 3),
        "mtp_ttft_s": round(mtp_ttft, 3),
        "mtp_tokens_in_window": mtp.result.completion_tokens,
        "classic_streams": args.classic_streams,
        "classic_mean_decode_tok_s": round(sum(ok_rates) / len(ok_rates), 2)
        if ok_rates
        else None,
        "classic_aggregate_tok_s": round(
            sum(
                (r["completion_tokens"] - 1)
                for r in rows
                if r["ok"] and r["completion_tokens"] > 1
            )
            / window_span,
            2,
        )
        if window_span > 0
        else None,
        "window_span_s": round(window_span, 2),
        "streams": rows,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One MTP stream plus N classic streams (issue #1261 arm 2)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--windows",
        type=int,
        default=1,
        help="Mixed windows to run back to back (default: 1)",
    )
    parser.add_argument(
        "--classic-streams",
        type=int,
        default=4,
        help="Concurrent classic streams per window (default: 4)",
    )
    parser.add_argument("--classic-prompt-tokens", type=int, default=512)
    parser.add_argument("--classic-max-tokens", type=int, default=256)
    parser.add_argument(
        "--mtp-max-tokens",
        type=int,
        default=3000,
        help="Token budget for the MTP stream; it is closed once the classic "
        "window ends, so this only needs to outlast the window (default: 3000)",
    )
    parser.add_argument(
        "--mtp-first-token-timeout",
        type=float,
        default=420.0,
        help="The first window's MTP TTFT includes the one-time exactness "
        "probe (and, in the default-env arm, its retry), which on a 31B "
        "target takes tens of seconds (default: 420)",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=2.0,
        help="Delay between the MTP stream's first token and the classic "
        "window, so the slice loop is steady (default: 2)",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.95,
        help="Minimum fraction of the classic window the MTP stream must have "
        "been decoding for; below it the window is reported invalid "
        "(default: 0.95)",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    args = parser.parse_args()

    model = args.model or wait_ready(args.host, args.port, args.ready_timeout)
    print(f"Server:  http://{args.host}:{args.port}")
    print(f"Model:   {model}")
    print(
        f"Windows: {args.windows}, classic streams per window: "
        f"{args.classic_streams} x {args.classic_max_tokens} tokens"
    )

    any_valid = False
    for w in range(args.windows):
        summary = run_window(args, model, w)
        any_valid = any_valid or summary["valid"]
        print(f"RESULT {json.dumps(summary)}", flush=True)
        mean = summary["classic_mean_decode_tok_s"]
        print(
            f"window {w}: valid={summary['valid']} overlap={summary['overlap']} "
            f"classic mean decode {mean} tok/s over "
            f"{summary['window_span_s']}s, MTP emitted "
            f"{summary['mtp_tokens_in_window']} tokens",
            flush=True,
        )
    return 0 if any_valid else 1


if __name__ == "__main__":
    sys.exit(main())
