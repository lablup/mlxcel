# Technical Report: PR #2020, Audio routes answer 500 for server-side failures

**Date**: 2026-09-28
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

Six failure paths in `src/server/routes/audio.rs` (worker panics in `audio_speech`, `compat_transcribe`, and `transcribe`; response-builder failures in the transcription-stream and binary-audio-response paths; and the `AudioModelError::Inference` mapping) answered HTTP 400 for server-side failures instead of 500, because each built `ErrorResponse::new(message, "server_error")` without overriding `.status`, so the constructor's `BAD_REQUEST` default leaked through. The sibling embeddings and rerank routes handle the identical failure class and already return 500. This PR converts all six sites to `ErrorResponse::internal_server_error(message)`, the constructor added in the companion PR #2015 (issue #1690), and pins the previously-unchecked `Inference` arm's status in the existing test.

## 1. Problem Statement

### 1.1 Background

`ErrorResponse::new` defaults `status` to `BAD_REQUEST` (400). A caller has to override `.status` explicitly to signal a server-side fault. The rerank and embeddings routes already did this for their equivalent failure classes (worker panic, malformed provider result); the audio routes never had the override line at all.

### 1.2 Existing Issues

- **Issue 1**: A server-side failure (a panicked `spawn_blocking` task, a `Response::builder()` failure building the SSE or binary body, or `AudioModelError::Inference`) was reported to the client as HTTP 400 with `"type": "server_error"`. The status contradicts the error type: a proxy or client keying retry/alerting behavior off the status code would treat these as the caller's fault.
- **Issue 2**: The existing test `model_error_maps_kinds_and_inference` asserted `error_type == "server_error"` for the `Inference` arm but never asserted its status, so nothing in the test suite pinned the correct behavior once fixed, unlike the `KindNotLoaded` (501), `QueueFull` (503), and `Timeout` (504) arms, which were all already pinned.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A monitoring or alerting system filtering on 5xx to detect server health misses audio worker panics and inference failures entirely, since they surface as 4xx | Medium | Was certain before this fix; now resolved |

## 2. Technical Review

### 2.1 Code Quality

- **Test Coverage**: One assertion added (`assert_eq!(inference.status, StatusCode::INTERNAL_SERVER_ERROR)`), completing the existing `model_error_maps_kinds_and_inference` test's coverage of all four `AudioModelError` variants. The full `server::routes::audio` module (16 tests) passes.
- **Code Complexity**: Unchanged; each site remains a single expression, now calling the shared constructor instead of `ErrorResponse::new`.

### 2.2 Compatibility & Dependencies

- **Breaking Changes**: Yes, in the narrow sense that six response bodies now carry HTTP 500 instead of 400 for the same error body (message and `"type": "server_error"` unchanged). Any client or proxy that specifically branched on 400 for these paths, rather than on the error type, will observe the new status. This is the intended fix: these are server-side failures, and 500 is the correct status.
- **New Dependencies**: None.

## 3. Technical Decisions

### 3.1 Reusing the #1690 constructor instead of inlining a fix

**Context**: The issue offered two options: set `INTERNAL_SERVER_ERROR` directly at each site, or use the shared constructor from the companion issue #1690 if it landed first.

**Rationale**: #1690 (PR #2015) merged first in this chain, so `ErrorResponse::internal_server_error` was already available. Using it here keeps the audio routes consistent with rerank, embeddings, and gcp_compat, and avoids reintroducing the two-line pattern that caused this defect in the first place.

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/server/routes/audio.rs`** (the `AudioModelError::Inference` arm)
```rust
// Before
AudioModelError::Inference(message) => ErrorResponse::new(
    format!("audio model inference failed: {message}"),
    "server_error",
),

// After
AudioModelError::Inference(message) => {
    ErrorResponse::internal_server_error(format!("audio model inference failed: {message}"))
}
```

**File: `src/server/routes/audio.rs`** (test)
```rust
let inference = audio_model_error_response(AudioModelError::Inference("boom".into()));
assert_eq!(inference.error.error_type, "server_error");
assert!(inference.error.message.contains("boom"));
assert_eq!(inference.status, StatusCode::INTERNAL_SERVER_ERROR); // new
```

**Reason for change**: Matches the sibling embeddings/rerank convention and closes the gap between what the test asserted and what the route actually returned.

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 1 |
| Lines added | +9 |
| Lines deleted | -9 |
| Tests added | 1 assertion (in an existing test) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Bug fix | 6 call sites | Corrected HTTP status from 400 to 500 for genuine server-side failures |
| Code Quality | 1 assertion | Pinned the `Inference` arm's status in the existing test |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `93e3417` | fix | Audio routes answer 500 for server-side failures |

### Related PRs/Issues

- PR #2015 / Issue #1690: added `ErrorResponse::internal_server_error`, the constructor this PR consumes.

## 8. Follow-up Actions

### Required

- [ ] None.

### Monitoring Required

- None beyond normal deployment observation; any dashboard or alert that previously counted these six audio failure modes under 4xx will now see them correctly counted under 5xx.
