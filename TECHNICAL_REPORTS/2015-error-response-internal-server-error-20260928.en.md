# Technical Report: PR #2015, Add ErrorResponse::internal_server_error

**Date**: 2026-09-28
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

`ErrorResponse` had named constructors for every 5xx-family status it returns except 500: `service_unavailable` (503), `gateway_timeout` (504), `not_supported` and `not_implemented` (501). The 500 case was instead hand-rolled at eight call sites across `rerank.rs`, `embeddings.rs`, and `gcp_compat.rs`, each repeating `ErrorResponse::new(message, "server_error")` followed by a separate `response.status = StatusCode::INTERNAL_SERVER_ERROR` assignment. This PR adds `ErrorResponse::internal_server_error(message)` alongside its siblings and converts all eight sites to use it. No behavior changed: every site keeps its original status code and message text.

## 1. Problem Statement

### 1.1 Background

`ErrorResponse::new` defaults `status` to `StatusCode::BAD_REQUEST` (400). Every non-400 error path has to override `status` explicitly after construction, and the project had already grown named constructors that bundle the override with the error type and message, one per status. 500 was the only common status still built the two-line way.

### 1.2 Existing Issues

- **Issue 1**: A call site could set the `server_error` message and forget the trailing `response.status = StatusCode::INTERNAL_SERVER_ERROR` line, silently downgrading a server-side fault to the `new` default of 400. This is exactly the defect issue #1695 documents in the audio routes, which never had the second line at all.
- **Issue 2**: Duplication across eight call sites meant any future change to how a 500 is built (for example, adding a `code` field) required editing eight places instead of one.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Leaving the pattern hand-rolled invites another silent 400-for-500 defect at a new call site | Medium | Medium (already happened once, in audio.rs) |

## 2. Technical Review

### 2.1 Code Quality

- **Test Coverage**: Unchanged. The existing `rerank` and `embeddings` route test suites already assert `StatusCode::INTERNAL_SERVER_ERROR` on the converted paths (`non_finite_scores_return_500`, `non_finite_embeddings_return_500`, `provider_errors_map_to_the_shared_status_codes`, `error_mapping_matches_audio_routes`); all continued to pass unchanged, confirming the refactor preserved behavior.
- **Code Complexity**: Reduced. Each converted site collapsed from a `let mut response = ErrorResponse::new(...)` plus a status assignment (and, at three sites, an explicit final `response`/`return response.into_response()`) to a single expression.
- **Technical Debt**: Decreased. The `axum::http::StatusCode` import became dead in `rerank.rs`, `embeddings.rs`, and `gcp_compat.rs` once the last inline `StatusCode::INTERNAL_SERVER_ERROR` reference in each file was removed, and was deleted from all three.

### 2.2 Compatibility & Dependencies

- **Breaking Changes**: None. Wire format, status codes, and message text are unchanged.
- **New Dependencies**: None.

## 3. Technical Decisions

### 3.1 Constructor placement and doc comment

**Context**: The four existing named constructors are grouped together in `response.rs` (`service_unavailable`, `gateway_timeout`, `not_supported`, `not_implemented`), each with a short doc comment describing which routes use it.

**Rationale**: `internal_server_error` was placed directly after `gateway_timeout`, keeping the 5xx constructors adjacent, and given a doc comment in the same one-paragraph style describing the class of failure it represents (a panicked worker task, a malformed provider result, a response-builder error) rather than enumerating call sites, since call sites will grow with #1695 and an enumerated list would go stale.

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/server/types/response.rs`**
```rust
// After
pub fn internal_server_error(message: impl Into<String>) -> Self {
    Self {
        error: ErrorDetail {
            message: message.into(),
            error_type: "server_error".into(),
            code: None,
        },
        status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
    }
}
```

**File: `src/server/routes/rerank.rs`** (one of four converted sites)
```rust
// Before
let mut response = ErrorResponse::new(
    format!("rerank inference failed: {message}"),
    "server_error",
);
response.status = StatusCode::INTERNAL_SERVER_ERROR;
response

// After
ErrorResponse::internal_server_error(format!("rerank inference failed: {message}"))
```

**Reason for change**: Removes the possibility of setting the message without the status, and matches the sibling-constructor convention already used for 503/504/501.

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 4 |
| Lines added | +35 |
| Lines deleted | -45 |
| Tests added | 0 (existing tests already cover the converted paths) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Code Quality | 8 call sites + 1 new constructor | Collapsed duplicated 500-construction pattern into one named constructor |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `2798675` | refactor | Add ErrorResponse::internal_server_error |

## 8. Follow-up Actions

### Required

- [ ] None. Issue #1695 (audio routes answering 400 for server-side failures) is the direct follow-up and will consume this constructor.

### Future Improvements

- The six `server_error` sites in `audio.rs` that currently omit the status override entirely (issue #1695) are the natural next users of `internal_server_error`.
