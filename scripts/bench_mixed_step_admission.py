#!/usr/bin/env python3
"""Admission-during-decode latency harness for issue #908 (mixed prefill/decode steps).

``bench_serving_concurrency.py`` fires N identical requests simultaneously and
reports per-request aggregates. That cannot express the scenario issue #908 is
about, which needs two things it does not have: **staggered admission** (a long
prompt arriving while other streams are already decoding) and **per-token
arrival timestamps** (so inter-token latency can be split into a quiet window
and an admission window). This harness adds both.

Scenario, matching the issue:

1. Start ``--streams`` streaming requests with short prompts and a generous
   ``--stream-max-tokens``, so they are all in steady-state decode.
2. Wait ``--settle-s`` for the decode rate to stabilise.
3. Admit one request carrying an ``--admit-prompt-tokens`` prompt (8192 by
   default) and time its first token.
4. Report per-stream ITL p50/p95 **before** the admission and **during** it,
   plus the admitted request's TTFT.

Read the output with the scheduler in mind. Under the shipped tick policy a
parked chunked prefill yields every tick to any active decode batch, so the
admitted prompt does not advance until the decode streams finish. The
``admitted_first_token_after_last_stream`` line reports exactly that, and it is
the measurement ADR 0005 turns on.

Preconditions the harness checks and reports, because getting any of them wrong
silently measures nothing:

* The server must run with ``--metrics``; without it no counter can attribute a
  run and the harness says so instead of printing an unattributed table.
* ``--parallel`` on the server must exceed ``--streams``. With ``--parallel 4``
  and 4 streams the batch is full, the admitted request never leaves the queue,
  and there is no prefill during decode to measure at all.
* The admitted prompt must exceed the server's ``--prefill-chunk-size``, else it
  prefills in one unchunked forward and the chunked path is never entered.

Examples:
    # Baseline (shipped tick policy). Server started with --parallel 8 --metrics.
    python3 scripts/bench_mixed_step_admission.py --expect baseline

    # Prototype arm. Server started with MLXCEL_MIXED_STEP=1 --parallel 8 --metrics.
    python3 scripts/bench_mixed_step_admission.py --expect mixed
"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the prompt sizing and model resolution from the concurrency harness so
# a "512-token prompt" means the same thing in both benchmarks.
from bench_serving_concurrency import (  # noqa: E402
    _percentile,
    build_prompt,
    resolve_model,
)

# Scheduler counters that say which tick policy served the run. `mixed_steps` is
# the dispatch proof for the issue #908 prototype: it can only move when
# `MLXCEL_MIXED_STEP` is set and a tick advanced prefill and decode together.
#
# `prefill_grants` is the same kind of proof for the issue #1011 fairness
# policy: it can only move when a parked chunked prefill took a tick away from a
# live decode batch, so it separates a server running the shipped grant from one
# running `--prefill-grant-interval 0`. Comparing two grant intervals without it
# is comparing a code path against itself and calling the noise a result.
_TICK_METRICS = (
    "mlxcel_batch_mixed_steps_total",
    "mlxcel_batch_prefill_grants_total",
    "mlxcel_batch_prefill_chunks_total",
    "mlxcel_batch_decode_steps_total",
)


@dataclass
class StreamTrace:
    """Per-token arrival times for one streaming request."""

    label: str
    ok: bool = True
    error: str | None = None
    start_s: float = 0.0
    end_s: float = 0.0
    # Wall-clock arrival of every non-empty content delta.
    token_times: list[float] = field(default_factory=list)

    @property
    def ttft_s(self) -> float | None:
        return self.token_times[0] - self.start_s if self.token_times else None

    def itls_ms(self, lo: float | None = None, hi: float | None = None) -> list[float]:
        """Inter-token gaps in ms lying wholly within ``[lo, hi]``.

        Bounds are absolute ``perf_counter`` values; ``None`` means unbounded.
        Splitting one stream's tokens by time is what separates the quiet window
        from the admission window.

        A gap is counted only when **both** of its endpoints fall inside the
        bounds. The gap that straddles the admission instant belongs to neither
        window: it is part quiet and part contended, and charging it wholly to
        the admission window would inflate that window by one boundary sample
        per stream, which at four streams is a visible bias on a p95.
        """
        out: list[float] = []
        for prev, cur in zip(self.token_times, self.token_times[1:]):
            if lo is not None and prev < lo:
                continue
            if hi is not None and cur > hi:
                continue
            out.append((cur - prev) * 1000.0)
        return out


def stream_tokens(
    host: str,
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    label: str,
    start_barrier: float = 0.0,
) -> StreamTrace:
    """Issue one streaming completion, timestamping every content delta.

    ``start_barrier`` is an absolute ``perf_counter`` deadline to wait for
    before sending, which is how the admitted request is delayed until the
    decode streams have settled.
    """
    trace = StreamTrace(label=label)
    if start_barrier:
        while time.perf_counter() < start_barrier:
            time.sleep(0.005)

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
    trace.start_s = time.perf_counter()
    conn: http.client.HTTPConnection | None = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", "/v1/chat/completions", body=payload, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            body = resp.read().decode("utf-8", "replace")[:200]
            trace.ok = False
            trace.error = f"HTTP {resp.status}: {body}"
            trace.end_s = time.perf_counter()
            return trace

        # `readline()` frames in C. The obvious `read(1)` loop spends the GIL
        # once per byte across `streams + 1` threads, and that scheduling noise
        # lands directly in the inter-token gaps this harness exists to measure.
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if data == b"[DONE]":
                break
            try:
                event = json.loads(data)
            except ValueError:
                continue
            for choice in event.get("choices", []) or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    trace.token_times.append(time.perf_counter())
    except (OSError, http.client.HTTPException) as exc:
        trace.ok = False
        trace.error = str(exc)
    finally:
        if conn is not None:
            conn.close()

    trace.end_s = time.perf_counter()
    return trace


def scrape_tick_metrics(host: str, port: int) -> dict[str, float]:
    """Read the scheduler tick counters from ``/metrics``.

    Returns an empty dict when the endpoint is unavailable, which the caller
    treats as "attribution impossible" rather than "counters were zero".
    """
    try:
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")
        conn.close()
        if resp.status != 200:
            return {}
    except (OSError, ValueError, http.client.HTTPException):
        return {}
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        name = name.strip()
        if name in _TICK_METRICS:
            try:
                out[name] = float(value)
            except ValueError:
                continue
    return out


def _fmt(values: list[float]) -> str:
    if not values:
        return "n/a (no samples)"
    return (
        f"p50 {_percentile(values, 50):.1f} ms  "
        f"p95 {_percentile(values, 95):.1f} ms  "
        f"mean {statistics.mean(values):.1f} ms  "
        f"n={len(values)}"
    )


async def _amain(args: argparse.Namespace) -> int:
    model = resolve_model(args.host, args.port, args.model)
    stream_prompt = build_prompt(args.stream_prompt_tokens)
    # Salt the admitted prompt per run. The prompt cache is on by default, and a
    # byte-identical 8K prompt would be adopted wholesale on the second run of
    # an A/B, skipping the chunked path the harness is here to observe. The salt
    # goes at the front so no cached prefix matches. Repeated runs are the point
    # of this harness, so this is not a detail that can be left to the operator.
    salt = f"Session {os.getpid()}-{int(time.time())}. "
    admit_prompt = salt + build_prompt(args.admit_prompt_tokens)

    print("# mixed prefill/decode admission bench (issue #908)")
    print(f"model={model}  streams={args.streams}  settle={args.settle_s}s")
    print(
        f"stream_prompt~{args.stream_prompt_tokens} tok  "
        f"admit_prompt~{args.admit_prompt_tokens} tok  "
        f"stream_max_tokens={args.stream_max_tokens}"
    )
    print(f"expect={args.expect}")
    print()

    before = scrape_tick_metrics(args.host, args.port)
    if not before:
        print(
            "NOT COMPARABLE: /metrics is unavailable, so no counter can say which "
            "tick policy served this run. Restart the server with --metrics. "
            "Refusing to print a latency table that cannot be attributed."
        )
        return 2
    missing = [n for n in _TICK_METRICS if n not in before]
    if missing:
        print(
            "NOT COMPARABLE: /metrics is missing "
            + ", ".join(missing)
            + ". This server predates the counters this harness attributes runs "
            "with, so a zero delta would mean 'not built' rather than 'did not "
            "engage'. Rebuild the server from this branch."
        )
        return 2

    loop = asyncio.get_running_loop()
    admit_at = time.perf_counter() + args.settle_s

    # Size the pool explicitly. asyncio's default executor caps at
    # `min(32, cpu_count + 4)`, so a large `--streams` on a small host would
    # silently queue requests instead of running them concurrently, and the
    # admitted request could be the one left waiting for a thread. That would
    # measure client-side queueing and report it as server latency.
    executor = ThreadPoolExecutor(max_workers=args.streams + 1)

    tasks = [
        loop.run_in_executor(
            executor,
            stream_tokens,
            args.host,
            args.port,
            model,
            f"{stream_prompt} Please answer question {i} at length.",
            args.stream_max_tokens,
            args.timeout,
            f"stream{i}",
            0.0,
        )
        for i in range(args.streams)
    ]
    tasks.append(
        loop.run_in_executor(
            executor,
            stream_tokens,
            args.host,
            args.port,
            model,
            admit_prompt,
            args.admit_max_tokens,
            args.timeout,
            "admitted",
            admit_at,
        )
    )

    try:
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        executor.shutdown(wait=True)

    traces: list[StreamTrace] = []
    for label, item in zip(
        [f"stream{i}" for i in range(args.streams)] + ["admitted"], gathered
    ):
        if isinstance(item, BaseException):
            traces.append(StreamTrace(label=label, ok=False, error=repr(item)))
        else:
            traces.append(item)
    after = scrape_tick_metrics(args.host, args.port)
    if [n for n in _TICK_METRICS if n not in after]:
        print(
            "NOT COMPARABLE: /metrics stopped serving the tick counters partway "
            "through the run, so the deltas below cannot be trusted."
        )
        return 2

    streams = [t for t in traces if t.label != "admitted"]
    admitted = next(t for t in traces if t.label == "admitted")

    failed = [t for t in traces if not t.ok]
    for t in failed:
        print(f"    request error ({t.label}): {t.error}")

    deltas = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in _TICK_METRICS}
    print("## scheduler tick attribution")
    for name, delta in deltas.items():
        print(f"    {name.replace('mlxcel_batch_', '')}: {delta:+.0f}")
    print()

    mixed = deltas["mlxcel_batch_mixed_steps_total"]
    grants = deltas["mlxcel_batch_prefill_grants_total"]
    chunks = deltas["mlxcel_batch_prefill_chunks_total"]

    admit_sent = admitted.start_s
    admit_first = admitted.token_times[0] if admitted.token_times else None

    # ---- Dispatch guards. Each one, when it fires, means the run measured
    # something other than what the invocation claimed, so no table is printed.
    if failed:
        print(
            "NOT COMPARABLE: "
            + str(len(failed))
            + " request(s) failed, so the batch composition during this run is not "
            "the scenario. Errors are listed above."
        )
        return 2

    # The scenario is "a long prompt arrives while others decode". That requires
    # every stream to be mid-generation at the admission instant. Checking the
    # prefill-chunk counter alone does not establish it: with --parallel equal to
    # --streams the admitted request simply queues until a stream finishes and
    # then prefills against a draining batch, which moves the counter for a run
    # in which the scenario never occurred.
    not_decoding: list[str] = []
    for t in streams:
        before_admit = any(ts < admit_sent for ts in t.token_times)
        after_admit = any(ts > admit_sent for ts in t.token_times)
        if not (before_admit and after_admit):
            not_decoding.append(t.label)
    if not_decoding:
        print(
            "NOT COMPARABLE: "
            + ", ".join(not_decoding)
            + " did not straddle the admission instant, so the long prompt did not "
            "arrive next to a live decode batch. Raise --stream-max-tokens so the "
            "streams outlast the admission, lower --settle-s, and make sure the "
            "server's --parallel exceeds --streams so the request is admitted "
            "immediately instead of queueing until a slot frees."
        )
        return 2

    if chunks <= 0:
        print(
            "NOT COMPARABLE: the prefill-chunk counter did not move, so the admitted "
            "prompt never entered the chunked-prefill path at all. Either it was "
            "never admitted (the server's --parallel must exceed --streams; with a "
            "full batch the request sits in the queue), or it fit in one unchunked "
            "forward (--admit-prompt-tokens must exceed the server's "
            "--prefill-chunk-size). There was no prefill-during-decode window."
        )
        return 2
    if args.expect == "mixed" and mixed <= 0:
        print(
            "NOT COMPARABLE: --expect mixed, but no mixed step ran. The server was "
            "not started with MLXCEL_MIXED_STEP=1, so this run measured the default "
            "tick policy against itself. This is the failure mode issue #899 shipped; "
            "refusing to print a timing comparison."
        )
        return 2
    if args.expect == "baseline" and mixed > 0:
        print(
            "NOT COMPARABLE: --expect baseline, but the mixed-step counter moved. "
            "MLXCEL_MIXED_STEP is set in the server's environment, so this is not a "
            "baseline measurement."
        )
        return 2

    # Issue #1011 fairness arm, same guard shape. `--expect baseline` says
    # nothing about the grant, because the grant IS the default policy; this is
    # the separate axis.
    if args.expect_grant == "on" and grants <= 0:
        print(
            "NOT COMPARABLE: --expect-grant on, but no fairness grant fired. The "
            "server is running --prefill-grant-interval 0, or the interval is so "
            "large that no grant came due inside the window, so this run measured "
            "the starved policy rather than the fair one."
        )
        return 2
    if args.expect_grant == "off" and grants > 0:
        print(
            "NOT COMPARABLE: --expect-grant off, but the fairness grant fired. The "
            "server is not running --prefill-grant-interval 0, so this is not a "
            "grant-disabled measurement."
        )
        return 2

    # ---- Latency windows.
    #
    # The window must end where the prefill stops competing with decode, and
    # that is NOT the same instant in the two arms. In the mixed arm the
    # admitted request's first token marks its terminal chunk, so it closes the
    # window correctly. In the baseline arm the parked prefill does not resume
    # until the batch drains, which is the finding itself, so that first token
    # lands after every stream has already ended; using it would sweep the whole
    # post-admission run into a window labelled "prefill live", during almost
    # all of which no prefill ran. Clamping to the earliest stream end keeps
    # both arms measuring gaps that genuinely overlap the prefill.
    earliest_stream_end = min((t.end_s for t in streams), default=admit_sent)
    if admit_first is None:
        window_end = earliest_stream_end
        window_note = "clamped to first stream end; admitted request produced no token"
    elif admit_first > earliest_stream_end:
        window_end = earliest_stream_end
        window_note = "clamped to first stream end; prefill outlived the decode batch"
    else:
        window_end = admit_first
        window_note = "closed by the admitted request's first token"

    quiet: list[float] = []
    during: list[float] = []
    for t in streams:
        quiet.extend(t.itls_ms(hi=admit_sent))
        during.extend(t.itls_ms(lo=admit_sent, hi=window_end))

    print("## per-stream inter-token latency")
    print(f"    quiet window (before admission): {_fmt(quiet)}")
    print(f"    admission window ({window_note}): {_fmt(during)}")
    if quiet and during:
        quiet_p95 = _percentile(quiet, 95)
        if quiet_p95 > 0:
            print(
                "    p95 inflation during admission: "
                f"{_percentile(during, 95) / quiet_p95:.2f}x"
            )
        else:
            print("    p95 inflation during admission: n/a (quiet p95 is zero)")
    print()

    print("## admitted request")
    if admit_first is None:
        print("    never produced a token within the timeout")
    else:
        last_stream_end = max((t.end_s for t in streams), default=admit_sent)
        starved = admit_first > last_stream_end
        print(f"    TTFT: {(admit_first - admit_sent) * 1000.0:.0f} ms")
        print(f"    admitted_first_token_after_last_stream: {starved}")
        if starved:
            print(
                "    NOTE: the admitted prompt produced its first token only after "
                "every decode stream had finished. That is the chunked-prefill "
                "starvation ADR 0005 documents, not a slow prefill."
            )
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Admission-during-decode latency harness (issue #908).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument(
        "--model",
        default=None,
        help="Model id (default: resolved from /v1/models, else 'default')",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=4,
        help=(
            "Decode streams running before the admission (default: 4). The "
            "server's --parallel must be strictly greater, or the admitted "
            "request never leaves the queue."
        ),
    )
    parser.add_argument(
        "--stream-prompt-tokens",
        type=int,
        default=128,
        help="Approximate prompt length for the decode streams (default: 128)",
    )
    parser.add_argument(
        "--stream-max-tokens",
        type=int,
        default=512,
        help=(
            "Generation budget per decode stream (default: 512). Must be large "
            "enough that the streams are still decoding when the admission lands."
        ),
    )
    parser.add_argument(
        "--admit-prompt-tokens",
        type=int,
        default=8192,
        help=(
            "Approximate prompt length of the admitted request (default: 8192). "
            "Must exceed the server's --prefill-chunk-size to enter the chunked path."
        ),
    )
    parser.add_argument(
        "--admit-max-tokens",
        type=int,
        default=32,
        help="Generation budget for the admitted request (default: 32)",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=5.0,
        help=(
            "Seconds to let the decode streams reach steady state before the "
            "admission is sent (default: 5). The quiet-window ITL is measured "
            "over this span."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Per-request socket timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--expect",
        choices=("baseline", "mixed", "any"),
        required=True,
        help=(
            "Which arm this invocation claims to measure. 'baseline' fails if a "
            "mixed step ran, 'mixed' fails if none did. Required, because the "
            "point of this harness is an A/B and 'any' silently disables both "
            "dispatch guards; pass 'any' only for exploratory runs whose numbers "
            "will not be recorded."
        ),
    )
    parser.add_argument(
        "--expect-grant",
        choices=("on", "off", "any"),
        default="any",
        help=(
            "Whether this invocation claims the issue #1011 prefill fairness "
            "grant is active. 'on' fails if no grant fired, 'off' fails if one "
            "did. Orthogonal to --expect, which is about the #908 mixed-step "
            "prototype. Default 'any' for exploratory runs."
        ),
    )
    args = parser.parse_args()
    if args.streams < 1:
        parser.error("--streams must be at least 1")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
