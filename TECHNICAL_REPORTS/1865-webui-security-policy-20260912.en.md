# PR #1865: WebUI Security Policy and Router Harness

## Overview

PR #1865 implements the server-side security foundation for the bundled WebUI in epic #1834 and issue #1837. The change adds a shared WebUI security policy and middleware that can wrap both the router-mode harness implemented here and the single-app/startup builder owned by #1838. The policy keeps public health and static shell access available while treating model management, runtime operations, settings/properties, and events as admin-equivalent private surfaces guarded by the existing API-key registry.

## Problem statement

A private route can remain authenticated yet bypass administrative resource limits if classification enumerates only today's UI actions. Review found this gap for legacy properties/settings aliases and pending catalog/download/removal routes. A DTO round trip also checked only Rust serialization consistency, not the canonical error contract.

## Implementation

The new `src/server/webui/security.rs` middleware validates Host, Origin, Fetch Metadata, query parameters, bearer credentials, control request size, control concurrency, and SSE connection capacity before handing a request to the existing router stack. The secured router accessor installs this layer outside Trace, CORS, and legacy auth so hostile browser metadata and query-seeded secrets are rejected before lower layers can log or reflect them. The policy supports explicit WebUI and API path prefixes for reverse-proxy deployments, and it rejects percent-encoded credential parameter names as well as literal names.

`src/server/webui/security/policy.rs` centralizes CSP, referrer, nosniff, and permissions-policy headers, allowed Host/Origin validation, path prefix validation, and control/SSE semaphore limits. `src/server/webui/security/startup.rs` adds the startup resolver contract: disabled WebUI returns no policy, loopback interactive startup may generate a restart-local credential, noninteractive mode requires a configured key, non-loopback WebUI requires an explicit key plus TLS, and Unix sockets are rejected for the browser WebUI path. The generated credential is not cloneable, redacts in Debug, and exposes the terminal secret by consuming the value.

## Review corrections and decisions

`security/request.rs` now applies control limits by default to every non-GET/HEAD/OPTIONS request in the normalized `/ui-api/v1` namespace, including future routes. The existing route inventory identified legacy administrative mutations under `/models`, `/slots`, `/props`, `/settings`, `/v1/settings`, `/lora-adapters`, and `/v1/cache/reset`; these share rate, concurrent-response, and body limits. Inference, tokenization, response cancellation, and stream controls retain data-plane limits. Prefix boundaries prevent `/ui-api/v10` from being confused with v1.

Whole actual HTTP error JSON is compared against eight independently schema-validated canonical fixtures. Only `request_id` is normalized, after complete token-format validation; codes, messages, retryability, extra fields, and omitted-versus-null fields must match exactly. Negative controls reject missing fields, inserted nulls/extras, invalid error codes, altered messages/retryability, and malformed dynamic identifiers. No OpenAPI or generated TypeScript changes were required.

## Validation

`cargo test --profile test-fast --features metal,accelerate security` passed 26 selected tests (0 failed, 0 ignored), including 27 HTTP rate/capacity/body cases across nine administrative families. `make verify-webui-contract` validated 40 fixtures (32 before this finalization). Local validation covered formatting, targeted security tests, adversarial secured router tests, scoped clippy, no-default-feature compilation, WebUI contract fixtures, llama compatibility, workspace version consistency, and kernel dtype key static checks. The tests exercise missing and invalid keys, public health behavior, private route denial, hostile/null origins, cross-site Fetch Metadata, credential query rejection on public and private paths, percent-encoded credential names, DNS-rebinding Host rejection, preflight allowlisting, legacy `GET /models?reload`, encoded private paths, SSE permit retention until response-body drop, declared and streamed body limits, startup failure modes, strict allowed Origins, and custom path prefix classification.

## Change summary

The classifier moved into a focused module to keep production security files below 500 lines. Regression coverage includes direct HTTP enforcement for nine administrative route families, prefix-aware classification, and strict whole-error comparisons. The new classifier regression failed against the preceding implementation at `POST /props` (one failed test, exit 101), establishing a failing-before control rather than a pass-only test.

## Frozen body-limit alignment

The final cross-contract audit found the middleware default was 256 KiB while the existing OpenAPI `LimitSummary.json_body_bytes` contract was 2,097,152 bytes. The first follow-up aligned that production constant to 2 MiB without changing schemas, fixtures, or inference. A focused generic-wrapper harness constructs valid JSON at exactly 2,097,152 and 2,097,153 bytes, both with declared Content-Length and a multi-chunk body without Content-Length. The literal boundary is independently cross-checked against the canonical OpenAPI constant, rather than copied from the runtime limit. Its handler consumes `Request<Body>` directly so an extractor limit cannot mask middleware behavior.

After correction, both declared and chunked probes accept exactly 2,097,152 bytes (204) and reject 2,097,153 bytes (413); the existing large data-plane regression also passes unchanged. Both boundary regressions failed before the constant correction: valid 2,097,152-byte declared and chunked requests received 413 instead of 204 (2 failed tests, exit 101).

The preceding `92d77dbd` snapshot passed the root-run full workspace gate (11,184 passed, 0 failed, 361 ignored) and workspace Clippy. Those broad results precede this isolated constant correction; no new GPU/full-workspace run is claimed for the follow-up.

## Assembled-router body-limit correction

The generic wrapper proved the outer middleware boundary but did not exercise the real route's `Bytes` extractor. A subsequent review found this PR had also introduced a separate inner 256 KiB `DefaultBodyLimit` on UI routes and legacy model-management routes, so the assembled UI still rejected legal bodies and UI-off behavior had regressed. Removing that duplicate constant and its layers restores the original Axum 2 MiB extractor default; enabled WebUI keeps its canonical outer 2 MiB security policy. No feature-off dependency on the WebUI security module is introduced.

New assembled-router probes send valid model-action/load JSON padded with trailing whitespace to the independently schema-checked literal boundary. Both declared and multi-chunk requests must reach the actual missing-model handler at exactly 2,097,152 bytes (404 with the precise domain error, not a successful model load) and receive 413 at 2,097,153 bytes. These cover both the secured `/ui-api/v1/model-actions` route and UI-off `/models/load`. After removal, all eight assembled-router cases pass: two modes × two body transports × exact/over-limit sizes. Each action probe uses a distinct idempotency key so it reaches the real handler rather than replaying an earlier failed operation. Before removing the inner layers, both assembled-router tests failed on the first declared exact-limit request (413 instead of the handler’s 404; 2 failed tests, exit 101). The earlier `fded5454` 24-test/scoped-Clippy gate preceded this semantic plumbing correction, so it is not used as final assembled-route evidence.

## Deferred scope

Production CLI/startup mounting is intentionally deferred to #1838. Frontend credential handling, logout cleanup, and browser UX are owned by sibling WebUI client/design issues. No CUDA/GB10, real checkpoint, Safari, or VoiceOver validation is claimed for this server-policy PR; the user explicitly waived unavailable required GB10 CI after confirming that runner was down, while requiring local CI to pass and the change does not modify inference paths.
