# Technical Report: PR #2914 - refactor(server): separate Responses generation from the HTTP connection

**Date**: 2026-11-10
**Status**: In review (open draft PR; closes #2896, part of epic #2903)
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

PR #2914 moves the streamed `/v1/responses` generation out of the HTTP handler `stream_create_response` into a new module, `src/server/routes/responses_run.rs`, so the same generation can run without an SSE connection. This is the first step of background mode (epic #2903). `prepare_response_run` produces a `PreparedRun` that owns everything the blocking task needs. `drive_response_run` is the former `spawn_blocking` closure body, moved intact, and it reports every event and lifecycle step through a new `ResponseEventSink` trait. `SseResponseSink` reproduces the old foreground behavior exactly. No route, flag, or wire format changed. The existing route tests pass unmodified (`server::routes`: 552 passed). A run on the real `qwen3-0.6b-4bit` checkpoint showed that streaming, non-streaming, an API cancel, and a client disconnect all behave as before. Nothing calls the non-SSE path yet: #2898 is the first consumer.

---

## 1. Problem Statement

### 1.1 Background

Background mode (`background: true`) must create a response, return at once, and keep generating with no client attached. Before this PR the only streaming code path was welded to its SSE channel in four places:

- **Event delivery:** `ResponseStreamSender::send_event` was called directly from the generation closure.
- **Cancellation:** a closed channel flipped the request's cancellation token, so a reader leaving was a cancel.
- **Persistence:** `persist_response` and the early-cancel `store.insert` were called inline.
- **Final output:** the record of completed output items (`done_items`, #2396), which builds the final object's `output`, lived inside the sender.

### 1.2 Existing Issues

- **No reusable driver.** About 600 lines of envelope sequencing (reasoning, message, tool-call items, compaction, cancel handling) sat inside one closure in the HTTP handler. Background mode would have had to copy it.
- **Output tied to the transport.** Because `streamed_output()` lived on the SSE sender, the final response's `output` depended on which transport ran the generation.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Duplicated driver drifts between foreground and background | High | High without this refactor |
| Refactor changes foreground SSE bytes or cancel semantics | High | Low (moved intact, tests unchanged, verified on a real checkpoint) |

---

## 2. Change Summary

| File | Change |
|------|--------|
| `src/server/routes/responses_run.rs` (new) | `ResponseEventSink`, `PreparedRun` (+ `arm_admission`), `prepare_response_run`, `drive_response_run`, the private `RunEvents` wrapper, `SseResponseSink`, and the moved `streamed_output_text` |
| `src/server/routes/responses_run_tests.rs` (new) | Five driver tests against a recording sink on the scripted provider |
| `src/server/routes/responses.rs` | `stream_create_response` reduced to prepare, channel and sink construction, spawn, admission wait, and `sse_response`. `persist_response` becomes `pub(crate)` and `count_reasoning_tokens` becomes `pub(super)` |
| `src/server/streaming_responses.rs` | `done_items` and `streamed_output()` removed from `ResponseStreamSender` |
| `src/server/routes/responses_auto_compact.rs` | `send_compaction_events` takes `&dyn ResponseEventSink` |

The call flow after the change:

```
create_response (validation, compaction)
  └─ stream_create_response
       ├─ prepare_response_run      → PreparedRun (render, options, cache ctx, queue slot)
       ├─ responses_sse_channel_resumable → sender, stream, cancelled
       ├─ SseResponseSink::new      (registers `cancelled` for /cancel)
       ├─ run.arm_admission(cancelled)
       ├─ spawn_blocking(drive_response_run(run, cancelled, sink))
       └─ admission.wait → sse_response(stream)
```

---

## 3. Technical Decisions

### 3.1 The sink owns side effects; the driver owns the response

**Context:** The issue fixed the trait's five methods because sibling #2898 was being written against them in parallel.

**Decision:** The driver decides the status, builds every event and the final object, and never touches the store or the channel. The sink decides what delivery, persistence, and registration mean. The driver keeps the completed-item record in a private `RunEvents` wrapper that implements the same trait and forwards to the caller's sink. Compaction events go through that wrapper too, so the compaction item stays in `streamed_output()` and `prepend_compaction_item` still de-duplicates it.

**Rationale:** The final object's `output` must be identical whichever sink ran. Keeping the record in the driver makes that structural, not a convention each sink must follow.

### 3.2 Early cancel goes through `record_terminal`

**Context:** A cancel acknowledged before the first token used to call `store.insert` directly, without the conversation append that `persist_response` performs. The trait has a single `record_terminal(&ResponseObject)` and no flag for this case.

**Alternatives considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep the direct store insert in the driver | No inference needed | The driver would touch the store, which a background sink must own |
| Add a trait method or parameter | Explicit | Deviates from the signatures sibling units were written against |
| **Chosen:** `record_terminal` with a `cancelled`, empty-`output` response; the sink skips the conversation for exactly that shape | Keeps the signatures; one rule for every sink | Relies on an invariant |

**Invariant relied on:** `build_response_object` always emits a message item when no function call is emitted (`responses_translator.rs`, "Message item" block). Every other terminal response therefore has at least one output item. The rule is written on the trait doc, so #2898 applies the same predicate.

### 3.3 The error path does not call `record_terminal`

This keeps the old behavior: a failed generation only emits `response.error`. The trait doc now says that a sink which drops events, as the planned background sink does, must read the `Error` event in `emit` to learn the code and message. `finish_registration` still runs once on every path.

### 3.4 `SseResponseSink` holds two senders

`generation_finished` cannot replace a sender behind `&self` without a lock. The sink builds `sender.after_generation()` up front and switches with an `AtomicBool`. Both are clones of one `tx`, so the SSE stream still ends exactly when the driver drops the sink.

---

## 4. Verification

- **Static checks:** `cargo fmt --check`, clippy (`-p mlxcel --lib --tests --features cuda -D warnings`), the env-var docs check, and the license-header check are all clean.
- **Unit tests:** `server::routes` 552 passed (1 ignored), `server::streaming_responses` 4 passed, `server::reasoning_echo` 30 passed. Run on GB10 with `--test-threads=1`.
- **New tests:**
  - A recording sink gets the same `(event, sequence_number)` list as the SSE wire for the same token script.
  - `record_terminal` runs once, after `generation_finished` and `finish_registration`, and before the last event.
  - `on_started` fires once.
  - The error path emits only `response.error`.
  - An early cancel records one empty `cancelled` response before `response.completed`.
- **Real checkpoint (`qwen3-0.6b-4bit`):**
  - A streamed request had contiguous sequence numbers 0..66, ended `completed`, and was retrievable.
  - A non-streamed request ended `completed`.
  - An API cancel mid-stream ended with `response.completed` carrying `status: cancelled`, and a repeat cancel was idempotent.
  - A client disconnect logged `cancelled (client disconnected)`.
- **Review:** an independent static review found no CRITICAL or HIGH issues. Its two MEDIUM findings were addressed: the terminal contract is now documented on the trait, and the empty-output predicate is narrowed to `cancelled`. The error and early-cancel tests were added in response to the same review.

### Not verified

- No non-SSE test checks the `streamed_output()` path with tool-call streaming (#2396). That path is still covered only through the SSE route tests.
- `on_started` on a prefill-progress callback was not tested, because the scripted provider never calls prefill progress. Only the token trigger is tested.

---

## 5. Learning Points

- **Move first, then diverge.** The driver was checked against the original closure with a normalized textual diff. The only differences are the sink calls and renamed locals. That makes "byte-compatible" something a reviewer can verify, rather than something the author only claims.
- **Put invariants where every implementer reads them.** The early-cancel rule had lived in an inline comment. Writing it on the trait doc is what lets a second sink implementation (#2898) get it right.
