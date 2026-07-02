#!/usr/bin/env python3
"""Concurrency load generator for the mlxcel OpenAI-compatible server.

Fires N concurrent streaming chat-completion requests against a running
``mlxcel-server`` and reports, per concurrency level, the mean/p50/p95
time-to-first-token (TTFT), the mean per-request decode throughput, and the
aggregate server throughput (all completion tokens divided by the wall-clock
span of the level). This is the serving-side companion to the offline
``bench_decode.sh`` / ``bench_longprompt.sh`` prefill benchmarks (epic #623
#624).

Only the Python standard library is used: each request runs on a worker thread
via ``asyncio``'s default executor and speaks HTTP/1.1 through ``http.client``,
so no third-party HTTP client is required.

Examples:
    # Sweep the default concurrency ladder (1, 2, 4, 8):
    python3 scripts/bench_serving_concurrency.py

    # Single concurrency level against a non-default port:
    python3 scripts/bench_serving_concurrency.py --concurrency 4 --port 8081

    # Longer prompts and more decode tokens:
    python3 scripts/bench_serving_concurrency.py --prompt-tokens 2048 --max-tokens 256
"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import statistics
import time
from dataclasses import dataclass, field

# Base sentence repeated to synthesize a prompt of an approximate token length.
# The server tokenizes it; the client only needs a roughly-sized input, so a
# words-per-token heuristic is sufficient here.
_BASE_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the benchmark harness "
    "measures prefill and decode throughput under concurrent streaming load. "
)
# Rough tokens-per-word factor for English BPE tokenizers; only used to size the
# synthetic prompt, never for reporting.
_TOKENS_PER_WORD = 1.3


@dataclass
class RequestResult:
    """Outcome of a single streaming request."""

    ok: bool
    ttft_s: float | None = None
    decode_tok_s: float | None = None
    completion_tokens: int = 0
    total_s: float = 0.0
    start_s: float = 0.0
    end_s: float = 0.0
    error: str | None = None


@dataclass
class LevelSummary:
    """Aggregated results for one concurrency level."""

    concurrency: int
    ok_count: int
    fail_count: int
    ttft_ms: list[float] = field(default_factory=list)
    decode_tok_s: list[float] = field(default_factory=list)
    total_completion_tokens: int = 0
    wall_s: float = 0.0

    @property
    def aggregate_tok_s(self) -> float:
        """Total completion tokens divided by the level's wall-clock span."""
        return self.total_completion_tokens / self.wall_s if self.wall_s > 0 else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile (0-100) of ``values`` (nearest-rank)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[rank]


def build_prompt(prompt_tokens: int) -> str:
    """Build a synthetic prompt of roughly ``prompt_tokens`` tokens."""
    words_per_copy = len(_BASE_SENTENCE.split())
    target_words = max(words_per_copy, int(prompt_tokens / _TOKENS_PER_WORD))
    copies = max(1, target_words // words_per_copy + 1)
    return (_BASE_SENTENCE * copies).strip()


def resolve_model(host: str, port: int, override: str | None) -> str:
    """Resolve the served model id, preferring an explicit override.

    Falls back to querying ``/v1/models`` and finally to ``"default"`` so the
    script works even when the server does not enforce the model name.
    """
    if override:
        return override
    try:
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        if resp.status == 200:
            data = json.loads(body)
            models = data.get("data") or []
            if models and isinstance(models[0], dict) and models[0].get("id"):
                return str(models[0]["id"])
    except (OSError, ValueError):
        pass
    return "default"


def stream_request(
    host: str,
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> RequestResult:
    """Issue one streaming chat completion and time it (blocking).

    Runs on a worker thread. Counts completion tokens from the final ``usage``
    chunk when the server emits one, otherwise from the number of non-empty
    content deltas as a proxy.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
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
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", "/v1/chat/completions", body=payload, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            body = resp.read().decode("utf-8", "replace")[:200]
            conn.close()
            return RequestResult(ok=False, error=f"HTTP {resp.status}: {body}")

        buf = b""
        while True:
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
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    delta_tokens += 1
        conn.close()
    except (OSError, http.client.HTTPException) as exc:
        return RequestResult(ok=False, error=str(exc))

    end = time.perf_counter()
    completion_tokens = usage_tokens if usage_tokens is not None else delta_tokens
    total_s = end - start
    decode_tok_s: float | None = None
    if ttft is not None and completion_tokens > 1:
        decode_span = total_s - ttft
        if decode_span > 0:
            decode_tok_s = (completion_tokens - 1) / decode_span
    return RequestResult(
        ok=True,
        ttft_s=ttft,
        decode_tok_s=decode_tok_s,
        completion_tokens=completion_tokens,
        total_s=total_s,
        start_s=start,
        end_s=end,
    )


async def run_level(
    concurrency: int,
    host: str,
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> LevelSummary:
    """Run ``concurrency`` requests in parallel and aggregate the results."""
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            None, stream_request, host, port, model, prompt, max_tokens, timeout
        )
        for _ in range(concurrency)
    ]
    results: list[RequestResult] = await asyncio.gather(*tasks)

    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    summary = LevelSummary(
        concurrency=concurrency,
        ok_count=len(ok),
        fail_count=len(fail),
    )
    for r in ok:
        if r.ttft_s is not None:
            summary.ttft_ms.append(r.ttft_s * 1000.0)
        if r.decode_tok_s is not None:
            summary.decode_tok_s.append(r.decode_tok_s)
        summary.total_completion_tokens += r.completion_tokens
    if ok:
        summary.wall_s = max(r.end_s for r in ok) - min(r.start_s for r in ok)
    for r in fail:
        print(f"    request error: {r.error}")
    return summary


def print_table(summaries: list[LevelSummary]) -> None:
    """Print the per-level summary table to stdout."""
    header = (
        f"{'conc':>4}  {'ok':>3}  {'fail':>4}  "
        f"{'ttft_ms(mean)':>13}  {'ttft_ms(p95)':>12}  "
        f"{'decode_tok_s(mean)':>18}  {'aggregate_tok_s':>15}"
    )
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        ttft_mean = statistics.mean(s.ttft_ms) if s.ttft_ms else 0.0
        ttft_p95 = _percentile(s.ttft_ms, 95)
        decode_mean = statistics.mean(s.decode_tok_s) if s.decode_tok_s else 0.0
        print(
            f"{s.concurrency:>4}  {s.ok_count:>3}  {s.fail_count:>4}  "
            f"{ttft_mean:>13.1f}  {ttft_p95:>12.1f}  "
            f"{decode_mean:>18.1f}  {s.aggregate_tok_s:>15.1f}"
        )
    print()


def parse_concurrency(spec: str) -> list[int]:
    """Parse a comma-separated concurrency spec into a list of positive ints."""
    levels: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"concurrency must be positive, got {value}")
        levels.append(value)
    if not levels:
        raise ValueError("no concurrency levels parsed")
    return levels


async def _amain(args: argparse.Namespace) -> int:
    model = resolve_model(args.host, args.port, args.model)
    prompt = build_prompt(args.prompt_tokens)
    levels = parse_concurrency(args.concurrency)

    print(f"Server:       http://{args.host}:{args.port}")
    print(f"Model:        {model}")
    print(f"Prompt:       ~{args.prompt_tokens} tokens ({len(prompt)} chars)")
    print(f"Max tokens:   {args.max_tokens}")
    print(f"Concurrency:  {levels}")

    summaries: list[LevelSummary] = []
    for level in levels:
        print(f"\n>>> concurrency={level} ...")
        summary = await run_level(
            level,
            args.host,
            args.port,
            model,
            prompt,
            args.max_tokens,
            args.timeout,
        )
        summaries.append(summary)

    print_table(summaries)
    # Non-zero exit if every request at some level failed (server likely down).
    any_ok = any(s.ok_count > 0 for s in summaries)
    return 0 if any_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concurrency load generator for the mlxcel OpenAI-compatible server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument(
        "--model",
        default=None,
        help="Model id (default: resolved from /v1/models, else 'default')",
    )
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8",
        help="Comma-separated concurrency levels (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=512,
        help="Approximate synthetic prompt length in tokens (default: 512)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per request (default: 128)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request socket timeout in seconds (default: 600)",
    )
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
