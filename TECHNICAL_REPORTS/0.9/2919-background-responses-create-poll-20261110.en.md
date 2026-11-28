# PR #2919: Create and poll background Responses

**Date**: 2026-11-10
**Status**: Completed (PR open as a draft; closes #2898, part of epic #2903)

## Executive Summary

PR #2919 turns on OpenAI Responses API background mode for the single-model server. `POST /v1/responses` with `background: true` now validates the request synchronously, admits a job into the `ResponsesStore` background registry (#2897), and answers HTTP 200 with a `queued` object in about a millisecond. A detached Tokio task then runs the slow pre-generation work and hands the generation to `drive_response_run` (#2896) on the blocking pool with a new `BackgroundSink`. `GET /v1/responses/{id}` polls the job through `queued`, `in_progress` and a terminal status. Every exit path, including a panic in the job task or the driver, records exactly one terminal object, so no job stays active. Foreground responses serialize byte for byte as before: the new `ResponseObject.background` field is emitted only when it is `true`. Implementation review and security check found no CRITICAL or HIGH issues; the finalizer fixed two LOW items. 12 files, about +1350/-27, of which about 560 lines are tests.

## 1. Problem Statement

Before this PR the translator rejected `background: true` with a 400 (`BackgroundUnsupported`). The two dependencies had prepared the pieces but not connected them:

- #2896 split the Responses generation from the HTTP connection: `prepare_response_run` builds a `PreparedRun`, and `drive_response_run(run, cancelled, sink)` drives it into any `ResponseEventSink`.
- #2897 gave `ResponsesStore` an active-job map with admission limits, status transitions, an atomic terminal hand-off and ephemeral entries for `store: false`.

What was missing was the route behavior: which checks stay synchronous, which work moves into the job, how status moves from `queued` to `in_progress`, how failures that the driver only reports as stream events become a stored `failed` object, and how the existing GET, DELETE, `previous_response_id` and `conversation` paths treat a job that has not finished.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/server/routes/responses_background.rs` (new) | `BackgroundJob`, `validate_background_request`, `create_background_response` (503/429/400 mapping), `spawn_background_job` and `spawn_background_job_with` (driver injectable for tests), the job task `run_background_job`, and `run_prelude` |
| `src/server/routes/responses_background_sink.rs` (new) | `JobRecorder` (first-terminal recording, failure-code mapping, `Drop` guard) and `BackgroundSink` (the `ResponseEventSink` for jobs) |
| `src/server/routes/responses.rs` | `create_response` branches on `background`, skipping video expansion and auto compaction in the handler; `retrieve_response` falls back to `background_snapshot`; `delete_response` refuses an active job; `append_conversation_turn` split out of `persist_response` |
| `src/server/responses_translator.rs` | `BackgroundUnsupported` removed; `PreviousNotFinished { id, status }` when `previous_response_id` names an active job; `build_response_object` sets `background` |
| `src/server/responses_store.rs` | `set_background_input_items` to replace an active job's input items after compaction |
| `src/server/types/responses_response.rs` | `background: Option<bool>` with `skip_serializing_if = "Option::is_none"` |
| `src/server/routes/responses_background_route_tests.rs` (new) | 12 tests on the scripted provider covering the issue's acceptance criteria, plus cancel through the in-flight registry and a driver panic after the terminal |
| `docs/responses-api.md` | New "Background mode" section, field and endpoint tables, error cases, in-memory shutdown behavior |

## 3. Technical Decisions

### 3.1 What stays synchronous

Everything that only inspects the request remains a 400 before a job exists: `context_management` parsing, translation (including the new not-finished check), the media capability gate, effective input, sampling parameter ranges, tool shapes, reasoning budget, and the structured-output constraint. Background-specific refusals (`stream: true`, store disabled, a zero job limit, a zero entry or byte budget) run first in `validate_background_request`. The structured constraint is built at create on purpose: an invalid JSON schema is a client error the caller should see at once, not a `failed` job to poll for.

Three steps move into the job because they can take seconds: `expand_request_video_parts` (frame decoding), `apply_auto_compaction` (a full generation pass), and `prepare_response_run` (media reads and template rendering). Their errors become `failed` objects.

### 3.2 The queue check is a probe

`create_background_response` calls `can_accept_request()` and then `reserve_single_stream_queue_slot()`, answering the foreground's 503 when either fails, and releases the reservation at once. The job reserves its own slot inside `prepare_response_run`, after the prelude, which is the same point the foreground reserves it. Holding the slot across the prelude would deadlock compaction at queue depth 1, because the compaction pass needs a slot of its own. The cost: a queue that fills between create and job start ends the job as `failed` with `server_overloaded` instead of waiting. The security review raised this as MEDIUM because `--responses-background-max-jobs` (64) is independent of slots plus queue depth, so a burst larger than the queue fails some jobs. It is documented; a job that waits for a slot is a possible follow-up.

### 3.3 One cancellation token, registered before the response

`admit_background` returns an `InFlightToken`. `spawn_background_job_with` inserts that same `Arc` into the store's in-flight registry under the response id before create returns, and the job passes it to `arm_admission` and `drive_response_run`. So the existing `POST /v1/responses/{id}/cancel` (`cancel_in_flight`) reaches the generation without changes, and a cancel sent right after create cannot miss the job. Nothing references the client connection, so dropping the create response or a poll cancels nothing; a test checks this.

A cancel that lands while the job is still in its prelude is checked after the prelude: the job records `cancelled` with empty output and never dispatches. A cancel during a running compaction pass is not interrupted (the compaction API takes no token); it takes effect when compaction returns.

### 3.4 `queued` to `in_progress`

The provider answers the `AdmissionWaiter` at dispatch, which for media requests is before any slot starts, so the waiter cannot signal "started". The status flips in `BackgroundSink::on_started`, which the driver calls on the first prefill-progress or token callback. The waiter is still held until the provider answers, because dropping it unanswered cancels the generation, and its result is the fallback error if the driver returns without a terminal.

### 3.5 Exactly one terminal, from any exit

All terminal writes go through `ResponsesStore::finish_background`, where the first call wins. The sources are:

- The driver's own final response through `record_terminal`: `completed`, `incomplete` (as `final_status` decides), or `cancelled` during generation.
- A generation error, which the driver reports only as a `ResponseStreamEvent::Error` (it does not call `record_terminal` on that path). `BackgroundSink::emit` intercepts that event and records `failed` with the event's code and message. A worker that exited arrives this way as `ChatWorkerGoneError`.
- Prelude and `prepare_response_run` failures, through `JobRecorder::fail_with`, which maps the foreground's error response to `error.code`: a 4xx becomes `invalid_prompt`, a 503 that is not `server_error` becomes `server_overloaded`, anything else `server_error`.
- A `JoinError` from the blocking driver (a panic), recorded as `server_error` only while the job is still active, so a panic after the terminal was recorded leaves the result alone (finalizer fix).
- `Drop` on the last `JobRecorder` handle while the job is active: a panic in the async job task, or runtime shutdown.

`JobRecorder::fail` first calls `finish_in_flight`; if a cancel was already acknowledged to the client, it records `cancelled` instead of `failed`, so the client is never told "cancelled" and then sees "failed".

### 3.6 Polling shows no partial output

Snapshots keep `output: []` and zero usage until terminal, the issue's proposed default. Partial text belongs to streaming (#2900); exposing it here would mean copying driver state into the store on every token.

### 3.7 Stored input after compaction

`admit_background` stores the request's canonical input items, but automatic compaction runs later and rewrites them; the terminal response then carries the compaction capsule in its output. If the job were stored with the original items, a chained `previous_response_id` request would replay the pre-compaction history and the capsule. `set_background_input_items` replaces the items and adjusts the active byte count. The byte budget is not re-checked because compaction only shrinks the input.

### 3.8 Conversation and stored record

`finish_background` already stores the terminal object, so calling `persist_response` would store it twice. The PR splits `append_conversation_turn` out of `persist_response` (whose foreground behavior is unchanged) and the sink calls only that, at terminal, under the same `effective_store` rule. It skips the append for a cancel with empty output and when the store substituted a `failed` object for an oversized one.

### 3.9 Interactions with existing endpoints

- `retrieve_response` tries `store.get` first, then `background_snapshot`, which covers active jobs and `store: false` results until their ephemeral TTL.
- `delete_response` returns 400 for an active job and tells the client to cancel first; removing the job would leave its generation running unobserved.
- `previous_response_id` naming an active job returns 400 `previous response '<id>' has not finished (status: <status>)`. The translator checks `background_status` before `store.get`, so a job that finishes between the two calls is found as a stored response. This applies to foreground requests too.
- A `store: false` result cannot be chained (`get` hides ephemeral entries), matching the foreground.

## 4. Verification

- `cargo test --profile test-fast --features cuda -p mlxcel --lib server::routes::responses -- --test-threads=1` under `gpu-lock`: 79 passed after the finalizer commit. `responses_store` (30), `responses_translator` (24) and `server::types::responses` (18) passed on the implementation commit. The background tests passed repeatedly (8 runs, then 3 more in review).
- clippy (`-p mlxcel --lib --tests --features cuda -D warnings`), `cargo fmt --check`, the env-var docs check and the license header check pass.
- Real checkpoint on GB10 (`qwen3-0.6b-4bit`, `--responses-background-max-jobs 1`): create returned 200 `queued` in 1.2 ms; polls went `in_progress` then `completed` with output and usage; a `store: false` job completed by polling and could not be chained; a second job returned 429 `rate_limit_exceeded`; `previous_response_id` and `DELETE` on the active job returned the 400s; `background` with `stream` returned 400; a foreground create had no `background` key. On this small model the first poll 50 ms after create already saw `in_progress`; the scripted tests pin the `queued` poll.

## 5. Known Limitations and Follow-ups

- Cancel semantics belong to #2899. `request_background_cancel` flips the token without marking the cancel as acknowledged, so if #2899 switches the route to it, a cancel that stops generation before the first token would be recorded `failed`; `JobRecorder::fail` should then also treat a set token as `cancelled`. Cancel on a finished `store: false` job returns 404, and the active-job cancel still returns the old minimal envelope.
- A burst of jobs larger than slots plus queue depth fails the excess with `server_overloaded` (3.2).
- Background ids are not scoped to the API key, the same as foreground stored responses; the UUIDv4 id is the only protection. A failed job keeps the driver's error text in `error.message` for the store TTL.
- Each poll takes the `jobs` mutex and sweeps expired entries under the `inner` write lock (#2897 code, now on a hot GET path).
- Carried over from #2897: `DELETE` also removes a `store: false` terminal entry, and admitting a duplicate id replaces the active job (ids are server UUIDs).
- Jobs live in memory: the single-model server installs no signal handler, so a process exit loses them. This is documented.

## 6. Interfaces for the Sibling Issues

- `ResponseObject.background: Option<bool>` is set by `build_response_object` from `request.background.filter(|b| *b)`, so every object built for a background request, including the driver's final response and stream event payloads, carries `true`.
- #2900 (stream and resume) removes the `stream: true` refusal in `validate_background_request` and fans events out from `BackgroundSink::emit`, which today only intercepts `Error`.
- #2899 (cancel) works with the `InFlightToken` registered in `spawn_background_job_with`, the `finish_registration` acknowledgement, and `JobRecorder::fail`'s cancel rule (section 5).
- #2901 (router roles): nothing here touches `router_server.rs`. A job holds `AppState` and the model provider through its task and blocking thread.

## 7. Learning Points

- When a driver reports some outcomes only as events, a sink that persists results must read those events; trusting `record_terminal` alone leaves failed jobs active forever.
- A `Drop` guard on a shared recorder, combined with a first-write-wins store method, is a compact way to guarantee one terminal per job across panics without wrapping every await point.
- Moving work from a request handler into a detached job changes which resources may be held across it: a queue slot held across a compaction pass that itself needs a slot is a deadlock at depth 1.
