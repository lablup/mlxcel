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

"""Batch-scheduler counter arithmetic for the serving benchmark (issue #2156).

``bench_serving_concurrency.py --metrics`` snapshots ``/metrics`` around each
concurrency level. This module parses those snapshots and turns the batch
counters into the per-level figures the harness prints: decode steps, mean
batch occupancy, interleaved prefill chunks and mixed steps, and the mean
decode step time. Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass

# Batch scheduler counters (issue #2156). The client-side decode rate divides
# by the span after a request's first token, which also contains every tick
# the scheduler spent on other requests' prefill chunks; these counters say
# how many decode steps actually ran, how full they were, and how many prefill
# chunks were interleaved, so the real batched step cost is visible.
_DECODE_STEPS = "mlxcel_batch_decode_steps_total"
_DECODE_TOKENS = "mlxcel_batch_decode_tokens_total"
_MIXED_STEPS = "mlxcel_batch_mixed_steps_total"
_PREFILL_CHUNKS = "mlxcel_batch_prefill_chunks_total"
_PREDICTED_SECONDS = "llamacpp:tokens_predicted_seconds_total"
_PREDICTED_TOKENS = "llamacpp:tokens_predicted_total"
BATCH_METRICS = (
    _DECODE_STEPS,
    _DECODE_TOKENS,
    _MIXED_STEPS,
    _PREFILL_CHUNKS,
    _PREDICTED_SECONDS,
    _PREDICTED_TOKENS,
)


@dataclass
class BatchDelta:
    """What the batch counters did across one concurrency level.

    ``decode_tokens`` is the sum of batch rows over decode steps (the server
    adds the batch size on every step), so ``occupancy`` is the mean number of
    sequences one decode step advanced. ``predicted_s`` is the sum over
    completed requests of each request's generation time, first token to
    last, as llama-server's ``tokens_predicted_seconds_total`` defines it.
    """

    decode_steps: float
    decode_tokens: float
    mixed_steps: float
    prefill_chunks: float
    predicted_s: float
    predicted_tokens: float

    @property
    def occupancy(self) -> float | None:
        """Mean batch occupancy: decode tokens over decode steps."""
        return self.decode_tokens / self.decode_steps if self.decode_steps > 0 else None

    @property
    def request_seconds_per_step_ms(self) -> float | None:
        """``tokens_predicted_seconds`` delta over decode steps, in ms.

        This is the issue's raw ratio. With one request it is the time per
        decode step; with ``k`` requests decoding side by side every step is
        counted once per request, so it is ``occupancy`` times the step time.
        """
        if self.decode_steps <= 0:
            return None
        return self.predicted_s * 1000.0 / self.decode_steps

    @property
    def step_ms(self) -> float | None:
        """Mean decode step time in ms, as one request sees it.

        The raw ratio divided by the occupancy, which is
        ``tokens_predicted_seconds / decode_tokens``. It still contains any
        prefill tick that ran while a request was decoding, which
        ``prefill_chunks`` and ``mixed_steps`` count.
        """
        raw = self.request_seconds_per_step_ms
        occ = self.occupancy
        if raw is None or not occ:
            return None
        return raw / occ


def batch_delta(before: dict[str, float], after: dict[str, float]) -> BatchDelta | None:
    """Difference the batch counters across one level.

    Returns ``None`` when neither snapshot carries the decode-step counter (no
    ``/metrics`` endpoint, or a build without the batch scheduler).
    """
    if _DECODE_STEPS not in before and _DECODE_STEPS not in after:
        return None

    def d(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)

    return BatchDelta(
        decode_steps=d(_DECODE_STEPS),
        decode_tokens=d(_DECODE_TOKENS),
        mixed_steps=d(_MIXED_STEPS),
        prefill_chunks=d(_PREFILL_CHUNKS),
        predicted_s=d(_PREDICTED_SECONDS),
        predicted_tokens=d(_PREDICTED_TOKENS),
    )


def format_batch_delta(delta: BatchDelta | None) -> list[str]:
    """Render a level's batch counters as the lines the harness prints."""
    if delta is None:
        return ["    batch steps: /metrics unavailable, step cost unknown"]

    def fmt(value: float | None, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    counters = (
        "    batch counters: "
        f"decode_steps={delta.decode_steps:+.0f}  "
        f"decode_tokens={delta.decode_tokens:+.0f}  "
        f"mixed_steps={delta.mixed_steps:+.0f}  "
        f"prefill_chunks={delta.prefill_chunks:+.0f}  "
        f"tokens_predicted_seconds={delta.predicted_s:+.3f}"
    )
    steps = (
        "    batch steps: "
        f"occupancy={fmt(delta.occupancy, '.2f')}  "
        f"decode_step_ms={fmt(delta.step_ms, '.2f')}  "
        f"(predicted_seconds/steps={fmt(delta.request_seconds_per_step_ms, '.2f')} ms)"
    )
    return [counters, steps]


def parse_metrics(body: str, names: tuple[str, ...]) -> dict[str, float]:
    """Pick the samples named in ``names`` out of a Prometheus text body.

    Comment lines and samples with an unparsable value are skipped, so a body
    from a build that lacks some of the counters yields a partial dict.
    """
    wanted = set(names)
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        name = name.strip()
        if name in wanted:
            try:
                out[name] = float(value)
            except ValueError:
                continue
    return out
