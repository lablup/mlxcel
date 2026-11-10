# PR #2924: Stream and resume background Responses

**Date**: 2026-11-10
**Status**: Completed (PR open as a draft; closes #2900, part of epic #2903)

## Executive Summary

PR #2924 lets a client combine `background: true` with `stream: true` on `POST /v1/responses` and reconnect later with `GET /v1/responses/{id}?stream=true&starting_after=N`, as the OpenAI Responses contract describes. Each streamed background job owns a `BackgroundEventLog`: a bounded, replayable list of serialized SSE events numbered from 0 without gaps. The create call and every resume are independent readers over that log. A dropped connection ends only its reader; the job keeps running to its terminal state. On `qwen3-0.6b-4bit` (GB10, CUDA) a stream killed after sequence 39 resumed at 40 and ran to `response.completed` at 496 with no gap or duplicate, and the OpenAI Python SDK (3.48.0) did the same through `responses.create(background=True, stream=True)` and `responses.retrieve(id, stream=True, starting_after=29)`. Foreground streaming is unchanged. The implementation review found no CRITICAL or HIGH issues; the security check found one HIGH (a stalled reader kept an evicted log alive) and fixed it together with three smaller findings. 28 files, about +2418/-52, of which about 1210 lines are tests.

## 1. Problem Statement

After #2898, background jobs ran through a `BackgroundSink` that discarded every stream event, and `validate_background_request` refused `stream: true` with a 400. A client that wanted progress had to poll. The OpenAI contract instead streams the job's events with a `sequence_number` and lets a client that lost its connection resume after the last number it saw.

The server already had a replay buffer, `src/server/stream_session.rs` (#1444), but it is keyed by `(owner, conversation_id)` and its cursor is a byte offset. Translating `sequence_number` into byte offsets per reader was rejected in the issue. What was needed was a new log addressed by sequence number, with a clear lifetime (who owns it while the job runs, where it goes when the job ends, when its memory is released) and a reader model in which any number of clients can replay and tail concurrently without ever slowing generation.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/server/responses_background_events.rs` (new) | `BackgroundEventLog` (`VecDeque<(u64, &'static str, Arc<str>)>` with a byte budget), cursor resolution (`CursorError::Evicted` / `Ahead`), bounded reads, terminal preparation, and `tail_events`, the SSE stream a reader consumes |
| `src/server/routes/responses_background_stream.rs` (new) | The streamed create response, `resume_stream`, and `RetrieveResponseQuery` (`stream`, `starting_after`) with the 400/404 mapping |
| `src/server/responses_store.rs` | `background_event_log(id)`, `attach_background_event_log`, `background_event_log_dropped`; the log on the job record and on the terminal `Entry`; terminal event appended in `finish_background`; log bytes in `size_bytes` and `approximate_total_bytes`; `background_event_log_bytes` config |
| `src/server/routes/responses_run.rs` | `PreparedRun::background_opening: Option<u64>`: when set, the driver skips `response.created`, numbers from that value and sends `response.in_progress` at the first progress callback |
| `src/server/routes/responses_background.rs` | The `stream: true` 400 removed; log created and `response.created` / `response.queued` appended at admission; zero log budget refuses a streamed create |
| `src/server/routes/responses_background_sink.rs` | `BackgroundSink::emit` fans events into the log; the driver's `Error` event still records `failed` |
| `src/server/routes/responses.rs` | `retrieve_response` takes the query and dispatches to `resume_stream` when `stream=true` |
| `src/server/types/responses_stream.rs` | `ResponseStreamEvent::Queued` (`response.queued`), `sequence_number_mut` |
| Flag wiring | `--responses-background-event-log-bytes` / `MLXCEL_RESPONSES_BACKGROUND_EVENT_LOG_BYTES` (default 4 MiB) in both binaries, `serve`, `cli_input`, `startup` |
| Tests (new) | Log unit tests, store event-log tests, two route test modules (stream behavior and error cases), one background compaction stream test |
| Docs | `docs/responses-api.md` "Streaming and resuming" section and flag row; `docs/environment-variables.md` |

## 3. Technical Decisions

### 3.1 The log numbers the events, not the writers

Three writers append to one log: the admission path (`response.created`, `response.queued`), the generation driver (everything in between), and the store (the terminal event). Each could assign numbers, but then a gap or repeat appears whenever two of them disagree, for example when a prelude failure ends the job before the driver runs. The log therefore assigns `sequence_number` itself on `append` and restamps an event whose number differs. The driver is still told where to start (`background_opening`) so that its own emitter matches and restamping is the exception, not the rule.

### 3.2 Opening events at admission

The issue placed the ordering change in a `PreparedRun` field. The PR keeps that field but appends `response.created` (status `queued`) and `response.queued` when the job is admitted, before the prelude. A streamed create would otherwise send nothing, not even the response id, until video decoding or automatic compaction finished, which can take seconds. The driver, seeing `background_opening: Some(2)`, skips its own `response.created`, numbers from 2 and emits `response.in_progress` from the first progress callback, which is the same point `on_started` flips the stored status. With `None` the driver's output is byte-identical to before; foreground streams always pass `None`.

### 3.3 One terminal event, from the store

Background jobs end through many paths: a normal driver result, a driver `Error` event converted to `failed`, `JobRecorder::fail`, `cancel_before_start`, `JobRecorder::drop`, a driver `JoinError`, and the store's own replacement of an oversized result by a `failed` object. Appending the terminal event in each of those would be easy to get wrong and could disagree with what a poll returns. The PR appends it inside `ResponsesStore::finish_background`, where the first call already wins, from the exact object that is stored. Every exit path, including any added later by the cancel work in #2899, therefore ends the stream with exactly one event (`response.completed`, `response.incomplete` or `response.failed`) that matches the polled object. A cancelled job ends with `response.completed` carrying `status: "cancelled"`, as the foreground does after #2870.

To keep the store's lock discipline, the terminal event is serialized before `finish_background` takes the jobs lock (`prepare_terminal` reads only the next sequence number). Under the lock the prepared event is pushed as is when its number still matches and the stored object was not replaced; otherwise it falls back to serializing under the log lock. The original version serialized under the jobs mutex; the security pass moved it.

### 3.4 Budget, eviction and lifetime

The log drops its oldest events when a new one does not fit `--responses-background-event-log-bytes`, but always keeps the newest, so the terminal event survives even a tiny budget. The cost is that a log can exceed its budget by one event. While the job runs, the job record owns the log; `finish_background` moves it onto the terminal `Entry`, adds its bytes to `size_bytes`, and store eviction frees it. A log that does not fit `--responses-store-max-bytes` beside its response is dropped at finish rather than evicting the response itself, and the entry remembers that it was dropped so a later resume gets an accurate message. Live jobs' logs count in `approximate_total_bytes` but not in the admission budget, so active streamed jobs can hold up to `max_jobs` times the log budget (256 MiB at defaults) beyond the store budget. This is bounded by configuration and documented.

### 3.5 Readers

A reader is a cursor and a `Weak` handle. Each read takes the log lock, copies at most 256 events or 64 KiB (always at least one event), releases the lock and yields the frames. When it has caught up it creates a `Notified` future before re-reading, so an append between the read and the await still wakes it; a one-second periodic wake is a backstop. Appends call `notify_waiters` after releasing the lock and never wait on readers, so a slow client cannot slow generation. A reader that falls behind the retained window gets one `response.error` event with `code: "stream_events_evicted"` and the same message as the resume 400, then ends. A reader stops needing the log once it has sent the terminal event.

The security check found that the first version held a strong `Arc` for the whole stream, so a client that opened a stream and stopped reading pinned up to 4 MiB after the store evicted the entry, limited only by connection count. The `Weak` handle fixes that: a reader upgrades it only while waiting on a live tail, when the job owns the log anyway, and a reader whose log disappears before the terminal event gets the `response.error` event and closes.

### 3.6 Resume cursor rules

`starting_after` is parsed as `u64`; anything else is a 400. With a cursor, events with a larger sequence are replayed and then tailed. Without one, replay starts at 0 only if nothing was dropped. A cursor below the oldest retained minus one is a 400 that names the oldest retained `sequence_number` and tells the client to GET the response. A cursor past the newest assigned number is a 400 naming the newest (added in the security pass; the first version returned an empty stream with no terminal event on a finished job). The newest number itself is a valid cursor: it tails on a live job and closes at once on a finished one. A response with no log is a 400 saying it was not created with `background: true` and `stream: true`; an unknown id is 404; without `stream=true` the handler behaves exactly as before. `X-Conversation-Id` is ignored for background jobs because the cursor is the resume mechanism.

## 4. Verification

- `cargo test --profile test-fast --features cuda -p mlxcel --lib responses -- --test-threads=1`: 231 passed, under `gpu-lock`. `cli_input` and the CLI flag tests of both binaries also pass.
- `cargo clippy -p mlxcel --lib --tests --profile test-fast --features cuda -- -D warnings`, `cargo fmt --all -- --check`, `scripts/ci/check_env_var_docs.py`, `scripts/insert_apache_header.py --check`: clean.
- Each of the issue's 13 acceptance criteria has a test, among them two concurrent readers (one replaying, one tailing) receiving identical sequences, a slow reader overrun, a tiny-budget log keeping its terminal event, keepalive frames during a scripted pause, and log bytes released on eviction.
- Real model at the final head (`0a0e49ba`), `mlxcel-server -m models/qwen3-0.6b-4bit`, 500 output tokens: curl killed at sequence 39, resume from 40 to `response.completed` at 496, contiguous; SDK dropped at 29, resume 30..498 ending `response.completed`, replay with no cursor identical; `starting_after=abc` 400, unknown id 404, foreground id with `stream=true` 400.

## 5. Remaining Risks and Follow-ups

- Live-job logs are outside the admission budget (section 3.4). A follow-up could reserve the log budget at admission.
- The number of readers per response is not capped. After the `Weak` fix each costs one connection and at most one read batch.
- A duplicate query key such as `?stream=a&stream=b` now gets axum's plain-text 400 from the `Query` extractor instead of being ignored.
- A cancel that stops generation before the first token through `request_background_cancel` is still recorded `failed` (a known gap left for #2899), so its stream ends with `response.failed`. The stream follows whatever the store records, so fixing the recorder fixes the stream.
- A colliding response id in `admit_background` drops the old job's log without a terminal event; ids are random, so this is practically unreachable.
- Not run: Metal, the router pool (#2901), the full lib suite, CI.

## 6. Learning Points

- When several writers share one ordered stream, let the stream assign the order. The writers then cannot produce gaps no matter which of them runs or fails.
- Put the terminal event where the terminal state is decided. Emitting it from the single first-call-wins store transition covered seven exit paths without touching any of them.
- A replay reader should hold the buffer weakly. A strong handle per connection quietly turns the store's eviction into a no-op for any client that stops reading.
- The `stream_session` pattern carries over directly: a `std::sync::Mutex` never held across an await, a `Notified` created before the re-check, and a bounded periodic wake as insurance.
