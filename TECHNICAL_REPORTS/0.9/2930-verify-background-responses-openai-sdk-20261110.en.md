# PR #2930: Verify background Responses with the OpenAI SDK

**Date**: 2026-11-10
**Status**: Completed (PR open as draft; closes #2902, part of epic #2903)
**Languages**: Python, Rust, Markdown
**Risk Level**: Low

## Executive Summary

PR #2930 is the closing unit of the background Responses epic. It adds no product behavior. It adds an end-to-end conformance script that drives a running server through the official OpenAI Python SDK, a 2,000-job scripted leak test, and a reconciliation pass over `docs/responses-api.md`. On `qwen3-0.6b-4bit` (GB10, release CUDA build) all seven checks pass on the single-model server, check 7 (the 429 job limit) passes on a server started with `--responses-background-max-jobs 2`, checks 1 and 3 pass against the multi-model router with no `model` on retrieve or cancel, and 200 sequential jobs grew RSS by 0.2 MiB over the measured half (bound 64 MiB). A server with background mode disabled fails at check 1, which shows the script detects a missing feature. No product bug was found.

## 1. Problem Statement

Sub-issues #2896 to #2901 built background mode (job store, create and poll, cancel, streaming and resume, router and disaggregated roles) and tested it with route tests on a scripted provider. Those tests use the server's own request shapes, so they could not show that the official SDK's calls (`responses.create(background=True)`, `retrieve(id, stream=True, starting_after=n)`, `cancel(id)`) produce requests and parse answers the way the server expects, and nothing had run the feature on a real model. There was also no test that many jobs of mixed outcomes leave nothing behind, and the docs needed a final pass against the merged behavior.

## 2. Change Summary

| Area | Change |
|---|---|
| `scripts/verify_responses_background.py` (new) | Seven checks plus an RSS leak bound; stdlib plus `openai` only, run with `uv run --with openai`; `--checks`, `--expect-max-jobs`, `--jobs`/`--pid`, `--timeout` |
| `src/server/routes/responses_background_leak_tests.rs` (new) | `background_jobs_do_not_leak`: 2,000 scripted jobs over seven kinds |
| `src/server/routes/responses_background_route_tests.rs` | registers the `leak` child module |
| `docs/responses-api.md` | separate `?stream=true&starting_after=N` endpoint row, rewritten cancel row, `store` retention for background, media failures as `failed`/`invalid_prompt`, pointer to the script |

### The seven checks

1. Create with `background=True` returns `queued`/`in_progress` within 5 s with `background: true`, and polling reaches `completed` with non-empty `output_text`.
2. A `store=False` job is retrievable after it finishes.
3. Cancel during `in_progress` on an 8192-token job: the raw HTTP bodies of the cancel, a repeated cancel and a retrieve are byte-identical and `cancelled`.
4. A streamed background create starts with `response.created` carrying `background: true`, `sequence_number` strictly increases, and exactly one terminal event comes last.
5. The stream is dropped after the 5th delta and resumed with `starting_after`; the resumed stream must start at `cursor + 1` and stay contiguous, and the deltas before and after the drop must join to the final `output_text`.
6. A `previous_response_id` follow-up on the finished job completes.
7. With `max_retries=0`, the job after the limit raises `openai.RateLimitError`; the check cancels its own jobs in a `finally`.

## 3. Technical Decisions

### 3.1 Reconcile the docs instead of following the issue's line references

The issue was written against `8d3e16ad` and asked to replace the "no background job mode" text at `:89` and `:475`. By the time this unit ran, #2898 to #2901 had already written the Background mode, Streaming and resuming, and Serving roles sections and added the three flags to the store table. The PR therefore audits the existing text against the issue's checklist and the code, and fills only the gaps. The most notable correction is the cancel row: it described only the foreground stream case and said a background job "answers instead", which hid that a foreground stream cancel still returns a minimal `{id, object, status}` envelope. The new row states both.

### 3.2 Byte-level comparison for cancel idempotency

Check 3 compares the raw HTTP bodies of the cancel, the repeat and the retrieve rather than the parsed SDK objects. Parsed comparison would ignore field order and unknown fields; byte equality also catches a repeat cancel that rebuilds the object (for example with a new timestamp) instead of returning the recorded one, which is what #2899 promises.

### 3.3 `reasoning: {effort: "none"}` on every request

qwen3-0.6b honours `/no_think` on a single turn but ignored it on the chained `previous_response_id` turn, spent its output budget on reasoning, and returned an empty `output_text`, failing check 6. The server maps `reasoning.effort: "none"` to `enable_thinking: false` in the template, which is reliable on every turn. This is model behavior, not a server bug.

### 3.4 A fast, deterministic leak test

The leak test uses the scripted provider so 2,000 jobs take about 2 s. It uses a 256 KiB store budget, a 500 ms TTL and a 2 s `store: false` TTL. It reads each terminal object through `wait_background_terminal` rather than the store, because the small budget may already have evicted the stored copy. The `store: false` TTL started at 250 ms. The security review showed that a scheduling stall longer than that between a job finishing and the test reading it could expire the entry and panic the test, so it was raised to 2 s. The final empty-store assertion polls (each `get` sweeps) instead of sleeping a fixed time. The test asserts the byte budget after every job. At the end it asserts that the in-flight cancel registry is empty, `active_background_jobs()` is 0, exactly one generation ran per job, and the store is empty with 0 bytes after the TTL.

### 3.5 Bounded client calls

The SDK's default read timeout is 600 s, so one stalled `retrieve` or stream read could hang the script whatever `--timeout` said. The client is built with `timeout=args.timeout` and `max_retries=0`. The streaming checks also carry an overall deadline.

## 4. Validation

- `background_jobs_do_not_leak`: passed, 2,000 jobs in about 1.9 s. Under full CPU load the whole test took 3.6 s at most.
- `server::routes::responses_background`: 36 passed (`--test-threads=1`, under `gpu-lock`).
- clippy (`--lib --tests`, cuda, `-D warnings`), `cargo fmt --check`, `check_env_var_docs.py`, `insert_apache_header.py --check` and `ruff check` on the script are clean.
- Real model (`qwen3-0.6b-4bit`, release CUDA binary):
  - single model with `--jobs 200`: checks 1 to 6 passed; RSS over the measured half went from 1843.7 MiB to 1843.9 MiB.
  - `--responses-background-max-jobs 2`: all seven checks passed; the third job was refused with 429. The run was repeated after the review fixes and passed again.
  - `--responses-background-max-jobs 0`: FAIL at check 1 with the background-disabled 400.
  - router (`--models-dir`, `--models-max 2`, gemma-3-1b loaded first): checks 1 and 3 passed with no `model` on retrieve or cancel.

## 5. Learning Points

- The router's model discovery requires `config.json` to be a regular file, not only a real directory (the #1438 confinement). A scratch models dir for router tests must copy the small files and may symlink only the weights.
- An SDK-driven script is a cheap regression guard for wire-format drift that route tests cannot see, because route tests build the requests themselves.

## 6. Follow-up Actions

- None blocking. The script is not run in CI; it needs a real server and model. Running it before a release on one GB10 and one Apple Silicon host would keep the SDK contract checked.
- The script is not formatted with `ruff format`. The repository enforces ruff format only under `python/`, and 39 other `scripts/*.py` files are in the same state.
