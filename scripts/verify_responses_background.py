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

"""Check background Responses against a running server with the OpenAI SDK.

Issue #2902. Drives ``background: true`` on ``/v1/responses`` through the
official ``openai`` Python package, the way a client application would, and
prints one ``PASS``/``FAIL``/``SKIP`` line per check. The first failure ends
the run with exit code 1.

Checks:

1. create(background=True) answers queued/in_progress at once; polling
   reaches completed with output text.
2. A store=False job is retrievable once finished.
3. cancel on a long job returns cancelled; a repeat and a retrieve return the
   same object.
4. create(background=True, stream=True): sequence numbers increase and the
   stream ends with a terminal event.
5. A stream dropped mid-way resumes with retrieve(stream=True,
   starting_after=cursor); the deltas add up to the final output text.
6. previous_response_id chains onto a finished job.
7. Going past the job limit raises RateLimitError (needs --expect-max-jobs,
   the server's --responses-background-max-jobs).

``--jobs N`` then runs N sequential jobs while sampling the server's RSS
(``--pid``); after a warm-up of N/2 jobs the growth over the second half must
stay under 64 MiB. That is a leak bound, not a performance measurement.

The repository has no OpenAI SDK dependency; run it through uv::

    uv run --with openai python scripts/verify_responses_background.py \\
        --base-url http://127.0.0.1:18080/v1 --model qwen3-0.6b-4bit \\
        --jobs 200 --pid "$(pgrep -n mlxcel-server)"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable
from itertools import pairwise

import openai
from openai import OpenAI

TERMINAL = {"completed", "incomplete", "failed", "cancelled"}
TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}
RSS_BOUND_KIB = 64 * 1024
CREATE_PROMPT_SECONDS = 5.0

# Every request disables thinking (reasoning.effort "none" clears the
# template's enable_thinking), so a qwen3 answer lands in output_text.
SHORT_PROMPT = "Reply with one short sentence about rain."
LONG_PROMPT = "Write a very long story about a lighthouse keeper, at least 3000 words."


class CheckFailed(Exception):
    """A check observed behavior other than what it expects."""


class Skipped(Exception):
    """A check that cannot run against this server configuration."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def poll(client: OpenAI, response_id: str, timeout: float) -> openai.types.responses.Response:
    """Retrieve until the response is terminal."""
    deadline = time.monotonic() + timeout
    while True:
        response = client.responses.retrieve(response_id)
        if response.status in TERMINAL:
            return response
        if time.monotonic() > deadline:
            raise CheckFailed(f"{response_id} still {response.status} after {timeout:.0f}s")
        time.sleep(0.05)


def wait_status(client: OpenAI, response_id: str, wanted: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = client.responses.retrieve(response_id).status
        if status == wanted:
            return
        expect(status not in TERMINAL, f"{response_id} ended {status} before {wanted}")
        expect(time.monotonic() < deadline, f"{response_id} never reached {wanted}")
        time.sleep(0.02)


class Checks:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0, timeout=args.timeout)
        self.first_id: str | None = None

    def create(self, **kwargs: object) -> openai.types.responses.Response:
        params: dict[str, object] = {
            "model": self.args.model,
            "input": SHORT_PROMPT,
            "max_output_tokens": 256,
            "background": True,
            "reasoning": {"effort": "none"},
        }
        params.update(kwargs)
        return self.client.responses.create(**params)  # type: ignore[call-overload]

    def check_1_create_and_poll(self) -> str:
        started = time.monotonic()
        created = self.create()
        elapsed = time.monotonic() - started
        expect(
            created.status in ("queued", "in_progress"),
            f"create answered {created.status}",
        )
        expect(elapsed < CREATE_PROMPT_SECONDS, f"create took {elapsed:.2f}s")
        expect(created.background is True, "create object lacks background: true")
        done = poll(self.client, created.id, self.args.timeout)
        expect(done.status == "completed", f"ended {done.status}: {done.error}")
        expect(bool(done.output_text.strip()), "completed with empty output_text")
        self.first_id = done.id
        return f"{created.status} in {elapsed * 1000:.0f} ms, {len(done.output_text)} chars"

    def check_2_store_false(self) -> str:
        created = self.create(store=False)
        done = poll(self.client, created.id, self.args.timeout)
        expect(done.status == "completed", f"ended {done.status}: {done.error}")
        expect(bool(done.output_text.strip()), "completed with empty output_text")
        again = self.client.responses.retrieve(created.id)
        expect(again.status == "completed", f"second retrieve says {again.status}")
        return f"retrievable after completion ({len(done.output_text)} chars)"

    def check_3_cancel(self) -> str:
        created = self.create(input=LONG_PROMPT, max_output_tokens=8192)
        try:
            wait_status(self.client, created.id, "in_progress", self.args.timeout)
            time.sleep(1.0)  # let it generate, so the cancel carries partial output
            raw = self.client.responses.with_raw_response
            first = raw.cancel(created.id)
            cancelled = first.parse()
            expect(cancelled.status == "cancelled", f"cancel answered {cancelled.status}")
            expect(cancelled.id == created.id, "cancel answered another id")
            second = raw.cancel(created.id)
            expect(second.http_response.content == first.http_response.content, "repeat cancel differs")
            fetched = raw.retrieve(created.id)
            expect(fetched.http_response.content == first.http_response.content, "retrieve differs from cancel")
            return f"cancelled with {len(cancelled.output_text)} chars generated; repeat and retrieve identical"
        finally:
            try:
                self.client.responses.cancel(created.id)
            except openai.APIError:
                pass

    def check_4_stream(self) -> str:
        deadline = time.monotonic() + self.args.timeout
        sequence: list[int] = []
        names: list[str] = []
        with self.create(stream=True) as stream:
            for event in stream:
                if not names:
                    expect(
                        event.type == "response.created" and event.response.background is True,
                        "the created event lacks background: true",
                    )
                sequence.append(event.sequence_number)
                names.append(event.type)
                expect(time.monotonic() < deadline, "the stream ran past the timeout")
        expect(bool(names), "the stream sent no event")
        expect(names[0] == "response.created", f"first event is {names[0]}")
        expect(
            all(b > a for a, b in pairwise(sequence)),
            f"sequence numbers not increasing: {sequence}",
        )
        expect(names[-1] in TERMINAL_EVENTS, f"last event is {names[-1]}")
        expect(
            sum(name in TERMINAL_EVENTS for name in names) == 1,
            "more than one terminal event",
        )
        return f"{len(names)} events, sequence {sequence[0]}..{sequence[-1]}, ends {names[-1]}"

    def check_5_resume(self) -> str:
        deadline = time.monotonic() + self.args.timeout
        response_id: str | None = None
        cursor = -1
        before: list[str] = []
        with self.create(stream=True, input=LONG_PROMPT, max_output_tokens=200) as stream:
            for event in stream:
                if event.type == "response.created":
                    response_id = event.response.id
                cursor = event.sequence_number
                if event.type == "response.output_text.delta":
                    before.append(event.delta)
                    if len(before) >= 5:
                        break
                expect(event.type not in TERMINAL_EVENTS, "the job ended before the drop")
                expect(time.monotonic() < deadline, "the stream ran past the timeout")
        expect(response_id is not None, "no response.created event")
        assert response_id is not None
        resumed = self.client.responses.retrieve(response_id, stream=True, starting_after=cursor)
        after: list[str] = []
        last = cursor
        final = None
        for event in resumed:
            expect(time.monotonic() < deadline, "the resumed stream ran past the timeout")
            expect(
                event.sequence_number == last + 1,
                f"resume gap or duplicate: {last} then {event.sequence_number}",
            )
            last = event.sequence_number
            if event.type == "response.output_text.delta":
                after.append(event.delta)
            if event.type in TERMINAL_EVENTS:
                final = event.response
        expect(final is not None, "the resumed stream had no terminal event")
        assert final is not None
        text = "".join(before) + "".join(after)
        expect(
            text == final.output_text,
            f"deltas ({len(text)} chars) differ from output_text ({len(final.output_text)} chars)",
        )
        expect(bool(text), "no output text")
        return f"dropped after event {cursor}, resumed to {last}; {len(text)} chars match output_text"

    def check_6_chain(self) -> str:
        if self.first_id is None:
            raise Skipped("check 1 did not run")
        created = self.create(
            input="Now say it again in other words.",
            previous_response_id=self.first_id,
        )
        done = poll(self.client, created.id, self.args.timeout)
        expect(done.status == "completed", f"ended {done.status}: {done.error}")
        expect(bool(done.output_text.strip()), "completed with empty output_text")
        expect(done.previous_response_id == self.first_id, "previous_response_id not echoed")
        return f"chained onto {self.first_id}"

    def check_7_job_limit(self) -> str:
        limit = self.args.expect_max_jobs
        if limit is None:
            raise Skipped("pass --expect-max-jobs N for a server started with --responses-background-max-jobs N")
        started: list[str] = []
        try:
            for _ in range(limit):
                started.append(self.create(input=LONG_PROMPT, max_output_tokens=8192).id)
            try:
                extra = self.create()
            except openai.RateLimitError as error:
                return f"job {limit + 1} refused with 429 ({error.code})"
            started.append(extra.id)
            raise CheckFailed(f"job {limit + 1} was admitted as {extra.status}")
        finally:
            for response_id in started:
                try:
                    self.client.responses.cancel(response_id)
                except openai.APIError:
                    pass

    def rss_leak_bound(self) -> str:
        jobs = self.args.jobs
        samples: list[int] = []
        for index in range(jobs):
            created = self.create(max_output_tokens=32)
            done = poll(self.client, created.id, self.args.timeout)
            expect(done.status in ("completed", "incomplete"), f"job {index} ended {done.status}")
            samples.append(rss_kib(self.args.pid))
        warm = jobs // 2
        measured = samples[warm:]
        expect(bool(measured), "too few jobs to measure")
        baseline = samples[warm - 1] if warm > 0 else measured[0]
        growth = max(measured) - baseline
        detail = (
            f"{jobs} jobs; measured half RSS start {baseline / 1024:.1f} MiB, "
            f"end {measured[-1] / 1024:.1f} MiB, peak growth {growth / 1024:.1f} MiB "
            f"(bound {RSS_BOUND_KIB // 1024} MiB)"
        )
        expect(growth < RSS_BOUND_KIB, detail)
        return detail


def rss_kib(pid: int) -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False)
    value = out.stdout.strip()
    if not value:
        raise CheckFailed(f"process {pid} is gone")
    try:
        return int(value)
    except ValueError:
        raise CheckFailed(f"unexpected ps output for pid {pid}: {value!r}") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True, help="server base URL including /v1")
    parser.add_argument("--model", required=True, help="model name sent on create")
    parser.add_argument("--api-key", default="sk-local", help="API key (default: sk-local)")
    parser.add_argument("--jobs", type=int, default=0, help="sequential jobs for the RSS leak bound (0 skips it)")
    parser.add_argument("--pid", type=int, help="server pid whose RSS --jobs samples")
    parser.add_argument("--checks", default="1,2,3,4,5,6,7", help="comma-separated check numbers to run")
    parser.add_argument(
        "--expect-max-jobs",
        type=int,
        help="the server's --responses-background-max-jobs; enables check 7",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for one job")
    args = parser.parse_args()
    if args.jobs > 0 and args.pid is None:
        parser.error("--jobs needs --pid")

    checks = Checks(args)
    table: dict[str, tuple[str, Callable[[], str]]] = {
        "1": ("create-and-poll", checks.check_1_create_and_poll),
        "2": ("store-false-retrievable", checks.check_2_store_false),
        "3": ("cancel-idempotent", checks.check_3_cancel),
        "4": ("stream-sequence", checks.check_4_stream),
        "5": ("stream-resume", checks.check_5_resume),
        "6": ("chain-previous-response", checks.check_6_chain),
        "7": ("job-limit-429", checks.check_7_job_limit),
    }
    selected = [c.strip() for c in args.checks.split(",") if c.strip()]
    unknown = [c for c in selected if c not in table]
    if unknown:
        parser.error(f"unknown checks: {', '.join(unknown)}")
    plan = [(number, *table[number]) for number in selected]
    if args.jobs > 0:
        plan.append(("rss", "rss-leak-bound", checks.rss_leak_bound))

    for number, name, check in plan:
        try:
            detail = check()
        except Skipped as reason:
            print(f"SKIP {number} {name}: {reason}", flush=True)
            continue
        except (CheckFailed, openai.APIError) as error:
            print(f"FAIL {number} {name}: {type(error).__name__}: {error}", flush=True)
            return 1
        print(f"PASS {number} {name}: {detail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
