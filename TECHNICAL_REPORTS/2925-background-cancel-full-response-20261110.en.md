# PR #2925: Return the full Response from background cancel

**Date**: 2026-11-10
**Status**: Completed (PR open; closes #2899, part of epic #2903)

## Executive Summary

PR #2925 makes `POST /v1/responses/{id}/cancel` on a `background: true` response return the job's final Response object, byte for byte what a following `GET /v1/responses/{id}` returns. Before, the route answered with the foreground stream's three-field envelope, returned 404 for a finished `store: false` job, and could 404 an existing job for a moment before its terminal write. Cancel is now idempotent and its reply is always the recorded terminal object, which is what `client.responses.cancel()` expects. Foreground cancel is unchanged. The security review found one HIGH issue (a cancelled queued job kept running its setup work outside the job limit), fixed in the second commit. 9 files, +968/-36, of which about 600 lines are tests.

## 1. Problem Statement

PR #2919 enabled background mode but left cancel on the old foreground route, which was written for a stream that is open in the same process.

- The reply was the foreground envelope (three fields), not a Response object, so an OpenAI client could not parse it.
- A cancel on a finished `store: false` job returned 404, although a GET inside the retention window returns the job.
- Between the job's in-flight registration retiring and its terminal write, the job was in neither place the foreground route looked, so a cancel could 404 an existing job.
- A cancel that stopped generation before the first token surfaced as a driver error, and `JobRecorder::fail` recorded `failed`, not `cancelled` (the reviewer MEDIUM carried over from #2919, listed in its section 5).

The contract the issue asks for: cancel is idempotent, returns the final Response, and never disagrees with GET.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/server/routes/responses.rs` | `cancel_response` calls `cancel_background_response` first and falls through to the unchanged foreground code when it returns `None` (13 lines) |
| `src/server/routes/responses_background.rs` | `BACKGROUND_CANCEL_WAIT` (5 s), `cancel_background_response`, and `unless_cancelled`, which races the job's pre-generation work against its terminal write; module docs |
| `src/server/responses_store_cancel.rs` (new) | `cancel_background`, `background_cancel_requested`, `force_background_cancel`, `record_cancelled_locked` |
| `src/server/responses_store.rs` | `cancel_requested` flag on the job; `record_terminal_locked` split out of `finish_background` so the cancel paths share the terminal write; `request_background_cancel` sets the flag and stores the token with `Release`; lock-order doc |
| `src/server/routes/responses_background_sink.rs` | `JobRecorder::terminal_recorded`; `fail` records `cancelled` when a cancel was requested and logs the original error at debug level; `Drop` retires the cancel registration; the conversation append is skipped when the route recorded the terminal object first |
| `src/server/routes/responses_background_cancel_tests.rs` (new) | 10 tests, a child module of `responses_background_route_tests.rs` that reuses its harness (that file is over 500 lines) |
| `docs/responses-api.md` | Cancel semantics for background responses |

Commits: `60a3f67f` (the cancel route and store methods) and `d0a76f88` (drop queued work on cancel, debug log, `Release` ordering).

## 3. Technical Decisions

### 3.1 Routing: the background branch first

`cancel_response` asks `cancel_background_response` before anything else. It returns `None` for an unknown id and for a stored response whose `background` is not `true`, and the existing foreground code then runs untouched, including its `responses_cancel_route_tests.rs`, which passes unchanged. The background branch has to go first because an active job is in neither the stored entries nor, for a moment before its terminal write, the in-flight registry. The old order of "stored, then in-flight" is exactly what produced the transient 404.

### 3.2 Three cases by job status

`ResponsesStore::cancel_background` runs under the `jobs` mutex and flips the token, sets `cancel_requested`, and acknowledges the cancel in the in-flight registry. It then branches on the snapshot status:

- `queued`: `record_cancelled_locked` removes the job and records a `cancelled` copy of the snapshot (`output: []`, `completed_at: null`) in the same critical section as the status check. A job therefore cannot start generating between the check and the write. The reply is `BackgroundCancel::Terminal`.
- `in_progress`: the job stays active (`BackgroundCancel::Requested`). The route waits up to `BACKGROUND_CANCEL_WAIT` for the driver's own terminal object through `wait_background_terminal`. The driver checks its token once per token and records `cancelled` with the partial output, which is the more useful object to return.
- Timeout: `force_background_cancel` records `cancelled` from the snapshot (no output) and the driver's late write is a no-op. If the job finished in the meantime, it returns the object that was recorded first. The wait is a bound, not an expected path: it only runs out when the job is stuck in one long step.
- Finished or unknown: a retained object comes back as `Terminal` whatever its status; no job and no retained object is `NotFound`, which the route maps to `None` (foreground rules) before the wait, or to 404 if the object was evicted during the wait.

### 3.3 First writer wins, and the reply is the recorded object

Every terminal write, from the driver, `cancel_background` or `force_background_cancel`, goes through `record_terminal_locked`, which inserts the stored entry before the `jobs` guard is released, so no reader sees a job that is neither active nor stored. The route never builds its own reply. It returns the object that was recorded, so cancel and GET cannot disagree in any interleaving, including the window between `finish_registration` and `record_terminal`, where the job is still in `jobs` and the cancel waits instead of returning 404. When `completed`, `incomplete` or `failed` won the race, the reply is the existing 400 `has already finished and cannot be cancelled`, and GET is unchanged. A repeat cancel on a `cancelled` response returns the stored object.

### 3.4 A `cancel_requested` flag, not the token

`JobRecorder::fail` has to turn "the generation stopped before its first token and reported an error" into `cancelled`. The obvious signal is the job's token, but `AdmissionWaiter::drop` sets the same token when admission is refused. The first attempt read the token and recorded a driver panic as `cancelled`. The job now has a separate `cancel_requested` flag, set only by `request_background_cancel` and `cancel_background`, and `fail` reads it through `background_cancel_requested`. A revert check (flag disabled) makes `a_cancel_that_ends_generation_with_an_error_is_recorded_cancelled` fail with `failed`.

The accepted cost: once a cancel is requested, any later failure, including an unrelated driver panic, is recorded as `cancelled`. The reviewer rated this MEDIUM. The reply and GET still agree, and "cancel wins" matches the issue. To keep it diagnosable, the original code and message are logged at debug level (second commit).

### 3.5 Drop pre-generation work on a queued cancel (security HIGH)

Cancelling a `queued` job frees its slot immediately, but the job task kept running its prelude (video frame extraction, automatic compaction) and `prepare_response_run` until it next checked the token. A client looping create and cancel could keep any number of ffmpeg decodes and compaction passes running with no slot bounding them, bypassing `--responses-background-max-jobs`.

`unless_cancelled` wraps the prelude and the preparation in a `tokio::select!` against `JobRecorder::terminal_recorded`. When the cancel lands, the work future is dropped, as a foreground client disconnect drops it, and the cancel registration is retired. A compaction summary pass or clip decode that is already running finishes that one step. `terminal_recorded` loops on `wait_background_terminal` with a 60 s recheck bound; the wake comes from the terminal write. Test: `a_queued_cancel_drops_the_pre_generation_work`.

### 3.6 Concurrency and lock order

The documented order is `jobs`, then `in_flight` (read), then the ack mutex in `cancel_background`; `jobs` then `inner` in every terminal write. Nothing takes `jobs` while holding `in_flight`, the ack mutex or `inner`. No lock is held across the 5 s wait: the wait signal is cloned out of `jobs` first. Concurrent cancels on one id each hold one `Notified` future and one connection, and the first terminal write wins. Ack and registry entries are retired on every exit path, including `JobRecorder::drop`, which matters when the route recorded the terminal object and the job then panics.

### 3.7 Conversation append

When the route recorded `cancelled` first, the driver's later terminal write is a no-op. The sink now checks the object that was actually recorded (`recorded`, not its own `response`) when deciding to skip the append, so a cancel with empty output never appends a conversation turn.

## 4. Verification

- `cargo test --profile test-fast --features cuda -p mlxcel --lib server::routes::responses -- --test-threads=1`: 89 passed, including `responses_cancel_route_tests.rs` unchanged. `responses_store`: 32 passed. The background module was repeated (the race test runs 200 rounds each); the implementation review ran it 10 times with no failure.
- clippy `-D warnings`, `cargo fmt --all -- --check`, `check_env_var_docs.py` and `insert_apache_header.py --check` pass.
- Real checkpoint (`qwen3-0.6b-4bit`, GB10 CUDA, curl): cancel of an `in_progress` job returned 200 `cancelled` with partial output (281 tokens, then 316 after the review fix) and equalled GET (`cmp`, 3450 bytes in the first run); a repeat cancel was identical; a cancel right after create equalled GET 2 s later; `store: false` behaved the same; cancel of a completed job returned 400 with GET unchanged; an unknown id returned 404; no errors in the server log.

## 5. Unverified and Open Items

- LOW: `record_cancelled_locked` serializes the snapshot under the `jobs` mutex to size it (`responses_store_cancel.rs`). The snapshot echoes `instructions`, `tools` and `metadata`, so a large request costs a few ms of contention. Caching the snapshot size at admission would remove it. The code comment calls it cheap because the snapshot has no output, which holds only for the output part.
- LOW: for a foreground id, `cancel_background` clones the stored response under `jobs` plus the `inner` write lock, and the route then reads it again through `store.get`. A double read; harmless.
- The timeout path (`force_background_cancel`) is tested with an injected 50 ms wait, not the real 5 s constant, and no real checkpoint run exercised a job that ignores its token.
- No performance measurement was taken; cancel adds one mutex acquisition per call.
- The MEDIUM in 3.4 is accepted, not removed: an unrelated failure after a cancel is visible only in debug logs.
- Cancel of a job evicted while the route waits returns 404, a narrow window that is not tested.
- Carried over from #2919: background ids are not scoped to the API key.

## 6. Learning Points

- When the reply to an idempotent operation must match a later read, derive it from the stored record and never build it separately. Here the route cannot disagree with GET because both return the same object.
- A status check and the write that depends on it belong in one critical section. Recording `cancelled` for a queued job inside `cancel_background` removes the window in which the job could start generating.
- A flag that means "the client asked" must be separate from a token that several code paths set. `AdmissionWaiter::drop` sets the token on refused admission, so reading it as a cancel misclassifies a panic.
- Freeing a bounded resource (the job slot) must stop the work that resource bounded. Racing pre-generation work against the terminal write with `select!` ties the two together, as a client disconnect does in the foreground path.
- A wait that serves as a fallback needs an injectable duration so the timeout branch can be tested in 50 ms instead of 5 s.
