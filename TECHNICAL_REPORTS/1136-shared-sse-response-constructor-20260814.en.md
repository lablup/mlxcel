# Technical Report: PR #1136 - refactor(server): attach the SSE keepalive through one constructor

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low (no observable behaviour change; every route emits the same frames on the same schedule)

---

## Executive Summary

Five route handlers each ended with the same three-line tail attaching the SSE keepalive by hand. A sixth route that forgot it would have compiled and passed the whole suite. A `TODO` asked for an axum integration harness to cover the gap.

The harness is the wrong tool, and the reasoning is the substance of this change. The property that can regress is "does every route attach the keepalive", and an integration test answers that only for routes someone remembered to write a test for. A test per route is five tests guarding five copy-paste sites: the same duplication relocated into the test module, still blind to the sixth route. Removing the duplication removes the property's ability to be false.

`streaming::sse_response` is now the only way to turn one of these streams into a `Response`, and it takes the keepalive by value. There is one `Sse::new` and one `.keep_alive` left in the entire tree, with seven call sites routed through them.

---

## 1. Problem Statement

### 1.1 Background

`src/server/streaming_tests.rs` carried a `TODO` recording a known coverage gap: no test drives `Sse::new(stream).keep_alive(...)` end to end and reads raw SSE frames, because the unit-level `payload_channel` tests cannot reach axum's `KeepAlive` layer. It was skipped to avoid the test infrastructure.

Behind it, five handlers each ended with the byte-identical tail:

```rust
Sse::new(stream)
    .keep_alive(keepalive.into_inner())
    .into_response()
```

in `routes/chat.rs`, `routes/completions.rs`, `routes/native_completion.rs`, `routes/responses.rs` and `routes/anthropic.rs`. No route varied the stream type, the keepalive, or the ordering.

### 1.2 Existing Issues

- **The invariant was maintained by five independent acts of remembering.** Nothing in the type system required a route to attach the keepalive. A handler that returned `Sse::new(stream).into_response()` compiled fine and served a stream that a reverse proxy would drop mid-prefill.
- **The proposed fix would have institutionalised the duplication.** One `tower::ServiceExt::oneshot` test per route is five tests that must be added whenever a route is added, guarding five sites that must be written whenever a route is added. The failure mode is identical: someone adds the sixth route and forgets. The test module gains a maintenance obligation and buys no new guarantee.
- **Half the `TODO` was testing upstream.** "Does axum emit a comment frame while the stream is idle" is a property of axum's `KeepAlive`, not of mlxcel. Upstream covers it. Reproducing it here would have added infrastructure for a result that says nothing about this codebase.
- **The `TODO`'s stated cost had already fallen, which turned out not to matter.** `tower = { version = "0.4", features = ["util"] }` is already a direct dependency and `util` is exactly what provides `ServiceExt::oneshot`, so the harness needed no new dependency. Worth checking before rejecting the plan, and irrelevant once the plan is rejected on other grounds.
- **`router_front.rs` had two more hand-assembled sites.** Outside `src/server/routes/`, so outside the issue's wording, but the same construction, and leaving them would have made the "one attach site" property true of a directory rather than of the tree.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| A route is added without a keepalive and streams are dropped behind a proxy during long prefills | High | Medium |
| Five per-route tests are added and then drift out of sync with the route list | Medium | High |
| The shared constructor over-constrains a future route that legitimately needs a different stream type | Low | Low |

---

## 2. Technical Review

### 2.1 The Two Halves of the `TODO` Want Different Answers

Splitting the property is what makes the decision obvious. "Does axum keep the connection alive" belongs to axum. "Does every route ask it to" belongs here, and is not a testable property so much as a structural one: a test can only sample the routes it knows about, while a type can constrain all of them, including the ones not written yet. The correct instrument for a universally quantified statement over future code is the compiler.

### 2.2 What Makes Forgetting Inexpressible

`sse_response(stream, keepalive)` takes the keepalive by value, and the newtypes' inner `KeepAlive` is private, so there is no public path from one of these streams to a `Response` that does not pass through it. A route that wanted to skip the keepalive would have to construct `Sse` itself, which means importing it, which is now visibly not what any route does. The guarantee is not that it is impossible to write, it is that it cannot be done by omission.

### 2.3 The Trait Is Deliberately Minimal

`IntoKeepAlive` has one method and exists for one reason: to let three distinct types reach one function. It does not unify the newtypes, which is the point. #1105 required that a route cannot attach another surface's keepalive, and that still holds, because the value a handler passes to `sse_response` is the one its own channel constructor returned. The trait widens what can be passed to the constructor; it does not widen what any given route has available to pass.

### 2.4 The Compiler Confirmed the Result

Deleting the five tails made `sse::Sse` unused in all five route files, and clippy failed the build under `-D warnings` naming each one. That is worth recording as evidence rather than as a chore: it is the compiler stating that no route constructs an SSE response on its own any longer. The imports were removed.

The end state is checkable in one line each. `grep -rn "Sse::new(" src/` and `grep -rn "\.keep_alive(" src/` return exactly one line apiece, both in `streaming.rs`, and seven call sites go through them.

---

## 3. Technical Decisions

### 3.1 A Constructor, Not Five Tests

Stated above. The short form: a test samples, a type quantifies. The regression this guards against is about code that does not exist yet, which a test cannot reach.

### 3.2 A Shared Trait, Not One Merged Newtype

Merging the three newtypes would have removed the duplication too, and would have cost the guarantee #1105 preserves. The duplication was never in the types; it was in the attach expression. Only that is consolidated.

### 3.3 `router_front.rs` Included Despite Being Out of Scope

The issue names `src/server/routes/`. Its two sites are in `src/server/`, do not come from `sse_channel`, and build the newtype directly. Leaving them would have satisfied the criterion while leaving two hand-assembled `Sse::new` calls that a future edit could copy. Including them is what makes "one attach site" a property of the tree.

### 3.4 The `TODO` Is Replaced, Not Just Deleted

A bare deletion loses the reasoning, and the next reader notices the same coverage gap and re-proposes the same harness. The replacement note records that the hazard was removed structurally, and that the other half of the `TODO` tests upstream.

---

## 4. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 10 |
| Hand-assembled attach sites removed | 7 |
| Attach sites remaining | 1 |
| New public API | 0 (both additions are `pub(crate)`) |
| Behaviour changes | 0 |

### Changes by Area

**`src/server/streaming.rs`**
- New `IntoKeepAlive` trait, with the impl for `SseKeepAlive`.
- New `sse_response<S>(stream, keepalive) -> Response`, the single attach site.

**`src/server/streaming_responses.rs`, `src/server/streaming_anthropic.rs`**
- One `IntoKeepAlive` impl each. Channel-constructor docs now point at `sse_response` instead of describing the hand assembly.

**`src/server/routes/{chat,completions,native_completion,responses,anthropic}.rs`**
- Each tail becomes `sse_response(stream, keepalive)`; the now-unused `sse::Sse` import is dropped.

**`src/server/router_front.rs`**
- Both sites route through `sse_response`; the `Sse` import is dropped.

**`src/server/streaming_tests.rs`**
- The `TODO` is replaced by a note recording why the hazard no longer exists.

---

## 5. Validation and Follow-up

### Passed

- `cargo test --profile test-fast --features cuda --lib server::streaming` on GB10: 25 passed, 0 failed.
- `cargo test --profile test-fast --features cuda --lib server::routes` on GB10: 116 passed, 0 failed.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`.
- `cargo fmt --all -- --check`.

### Not Covered

- No end-to-end SSE frame capture. That is the test this change argues against writing, and nothing about the emitted frames changed: the same `KeepAlive` value reaches the same `Sse` in the same order.
- The acceptance criterion names `--features metal,accelerate`. This is a Linux CUDA box and cannot build that feature set; no feature gate reaches this code.
- The criterion's `server::` selector was run as its two populated halves rather than unfiltered, because an unfiltered selector matches hundreds of tests and the full lib suite aborts under parallel execution on this CUDA host for reasons unrelated to this change.

### Follow-up

- With #1133 and this change landed, the keepalive has one interval, one guard, and one attach site. A future surface gets all three by calling `sse_response`, and a surface that does not call it is visibly outside the arrangement rather than silently missing from it.
