# Technical Report: PR #1133 - refactor(server): consolidate the SSE keepalive interval and its guard

**Date**: 2026-08-14
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low (no observable behaviour change; every keepalive value was 15 and still is)

---

## Executive Summary

The SSE keepalive interval was declared three times, once per streaming surface. The compile-time assertion that gives the number its meaning, that it stays under the 60-second reverse-proxy idle default, named only one of the three. `/v1/responses` and the Anthropic-compatible surface could be raised past a proxy timeout and the build would not notice.

This is not a case of three constants that happened to agree. It is one invariant with one enforcement point and two surfaces outside it. The fix keeps one definition, moves the assertion next to it so coverage follows from where it sits rather than from who remembered to import it, and routes the two `KeepAlive::default()` sites in `router_front.rs` through the newtype they were supposed to use.

---

## 1. Problem Statement

### 1.1 Background

`src/server/streaming.rs` documents the keepalive as a property of SSE in general: a long prefill leaves the stream open and silent for tens of seconds, and nginx, HAProxy and AWS ALB all default to a 60-second idle timeout, so the connection dies before the first token. The interval has to be under 60 for every surface that streams, not for one route family.

Three surfaces stream: the chat and completion routes through `SseKeepAlive`, `/v1/responses` through `ResponseSseKeepAlive`, and the Anthropic-compatible messages route through `AnthropicSseKeepAlive`. Each newtype had an identical `default_for_long_prefill()` body reading a constant declared in its own file.

### 1.2 Existing Issues

- **The guard covered a third of what it described.** `const _: () = assert!(SSE_KEEPALIVE_INTERVAL_SECS < 60, ...)` lived in `streaming_tests.rs` and named the `streaming.rs` constant. The other two were file-private and unnamed by anything. Setting either to 61 compiled cleanly.
- **The narrow coverage was drift, not scoping.** The module docs state the invariant for SSE generally, and the assertion's own message talks about "most reverse proxies", not about one route family. Nothing in the code said the Responses and Anthropic surfaces were deliberately exempt.
- **`router_front.rs` bypassed the design.** Two sites built their responses with `KeepAlive::default()`. Under axum 0.7.9 that is 15 seconds, so it was not a live bug, but `streaming.rs` states plainly why the inner `KeepAlive` is private: "to prevent callers from constructing a mismatched keepalive independently". These two sites were exactly that. Lowering the shared constant for proxy compatibility would have left them at 15 with no diagnostic.
- **A test module was load-bearing for a production invariant.** The assertion is a compile-time check, so putting it in `streaming_tests.rs` worked, but it made the guard's existence depend on a `#[cfg(test)]`-adjacent file that a reader auditing the constant would not think to open.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| A future proxy-compatibility change lowers one constant and misses the other two, leaving surfaces mismatched | Medium | Medium |
| The Responses or Anthropic interval is raised past 60 and streams are dropped mid-prefill behind a proxy | High | Low |
| `router_front.rs` silently keeps 15s after the shared value is lowered | Medium | Low |

---

## 2. Technical Review

### 2.1 What the Invariant Actually Constrains

The assertion is about the wire, not about a type. A proxy sitting in front of the server does not know which route family produced a stream; it applies one idle timeout to all of them. So the correct scope for the guard is "every SSE surface", and the correct number of definitions is one. Placing the assertion beside the definition makes that structural: any surface that reads the constant is covered, and a surface that does not read it is visibly not using the shared interval.

### 2.2 Why the Newtypes Stay Separate

The obvious next step, collapsing the three newtypes into one, is wrong and the issue says so. They are distinct types precisely so a route cannot attach another surface's keepalive: the value a handler receives comes from the channel constructor it called. Merging them would trade a compile-time guarantee for nothing, since the duplication was never in the types. It was in the constant and the assertion, and only those are consolidated here.

### 2.3 The `router_front.rs` Sites Are the Same Defect Seen From the Other End

`KeepAlive::default()` is not a third copy of the constant, it is the absence of one: it takes whatever axum's default happens to be. That it currently equals 15 is a coincidence of the pinned version, not a property anyone chose. Routing both through `SseKeepAlive::default_for_long_prefill()` ties them to the same definition as everything else, and leaves no way to construct an SSE keepalive in `src/server/` that is not derived from `SSE_KEEPALIVE_INTERVAL_SECS`.

### 2.4 Verifying the Guard Is Not Vacuous

An assertion that passes tells you nothing about whether it can fail. The acceptance criterion asks for a demonstration, and the demonstration is decisive:

```
error[E0080]: evaluation panicked: SSE keepalive interval must be less than the 60s default used by most reverse proxies
  --> src/server/streaming.rs:77:15
```

Setting the constant to 61 fails the build at the assertion, by name, with the message intact. The value was reverted immediately. The point of running it is that before this change the same edit to either of the other two constants compiled cleanly, so the guard's reach is what changed, not its wording.

The reach widened on a second axis that was not obvious going in. `streaming_tests.rs` is included as `#[cfg(test)] #[path = "streaming_tests.rs"] mod tests;`, so the assertion was const-evaluated only in test builds. As a module-level `const _` in `streaming.rs` it is evaluated in every build, which is why the check above reproduces under a plain `cargo check --lib`. The guard now covers three surfaces instead of one and all profiles instead of test-only.

---

## 3. Technical Decisions

### 3.1 One Definition, Not Three Assertions

The alternative was to leave the three constants and add two more assertions. It is fewer lines to change and strictly worse: it keeps three numbers that must agree by convention, and the next surface added starts uncovered again by default. Consolidating makes coverage the default and drift the thing that requires effort.

### 3.2 The Assertion Moves to the Definition, Not to a Shared Test

Keeping it in a test module and importing the other two constants would also work, and would also have to be updated whenever a surface is added. Next to the definition, it needs no maintenance: every consumer reads the constant, so every consumer is covered.

### 3.3 `router_front.rs` Goes Through the Newtype, Not the Constant

Constructing `KeepAlive::new().interval(Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS))` inline at both sites would satisfy the letter of the criterion and add two more copies of the construction expression. Calling `SseKeepAlive::default_for_long_prefill()` reuses the one place that expression lives.

### 3.4 The Module Docs Stop Restating the Number

The module docs said "The keepalive interval is 15 seconds". That is a fourth place for the value to drift, in prose where no assertion can reach it. They now name the constant and record that it is shared, which is the fact a reader needs.

---

## 4. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 5 |
| Constant definitions removed | 2 |
| `KeepAlive::default()` call sites removed | 2 |
| Behaviour changes | 0 |

### Changes by Area

**`src/server/streaming.rs`**
- `SSE_KEEPALIVE_INTERVAL_SECS` documented as the single definition for every SSE surface.
- The `const _: () = assert!(... < 60, ...)` moved here from `streaming_tests.rs`, directly under the definition.
- Module docs point at the constant instead of restating `15`, and record that the invariant now covers all surfaces.

**`src/server/streaming_responses.rs`, `src/server/streaming_anthropic.rs`**
- Local `const KEEPALIVE_INTERVAL_SECS: u64 = 15;` deleted; both import the shared constant. The newtypes themselves are untouched.

**`src/server/streaming_tests.rs`**
- Assertion and its now-unused import removed, replaced by a comment recording where the invariant lives and why it moved.

**`src/server/router_front.rs`**
- Both `KeepAlive::default()` sites build through `SseKeepAlive::default_for_long_prefill()`; the `KeepAlive` import is dropped.

---

## 5. Validation and Follow-up

### Passed

- `cargo test --profile test-fast --features cuda --lib server::streaming` on GB10: 25 passed, 0 failed.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`.
- `cargo fmt --all -- --check`.
- The `61` build-failure check quoted above, reverted afterwards.

### Not Covered

- The acceptance criterion names `--features metal,accelerate`. This is a Linux CUDA box and cannot build that feature set. The affected code is backend-agnostic SSE plumbing that no feature gate reaches, so the CUDA run is equivalent evidence rather than a substitute.
- No live proxy test. Nothing about the emitted frames changed, so there is nothing new to observe on the wire; the values are identical to those already in production.

### Follow-up

- #1107 consolidates the other half of this duplication, the three `Sse::new(..).keep_alive(..)` attach expressions, and lands directly on top of this.
