# PR #1865: WebUI Security Policy and Router Harness

## Overview

PR #1865 implements the server-side security foundation for the bundled WebUI in epic #1834 and issue #1837. The change adds a shared WebUI security policy and middleware that can wrap both the router-mode harness implemented here and the single-app/startup builder owned by #1838. The policy keeps public health and static shell access available while treating model management, runtime operations, settings/properties, and events as admin-equivalent private surfaces guarded by the existing API-key registry.

## Implementation

The new `src/server/webui/security.rs` middleware validates Host, Origin, Fetch Metadata, query parameters, bearer credentials, control request size, control concurrency, and SSE connection capacity before handing a request to the existing router stack. The secured router accessor installs this layer outside Trace, CORS, and legacy auth so hostile browser metadata and query-seeded secrets are rejected before lower layers can log or reflect them. The policy supports explicit WebUI and API path prefixes for reverse-proxy deployments, and it rejects percent-encoded credential parameter names as well as literal names.

`src/server/webui/security/policy.rs` centralizes CSP, referrer, nosniff, and permissions-policy headers, allowed Host/Origin validation, path prefix validation, and control/SSE semaphore limits. `src/server/webui/security/startup.rs` adds the startup resolver contract: disabled WebUI returns no policy, loopback interactive startup may generate a restart-local credential, noninteractive mode requires a configured key, non-loopback WebUI requires an explicit key plus TLS, and Unix sockets are rejected for the browser WebUI path. The generated credential is not cloneable, redacts in Debug, and exposes the terminal secret by consuming the value.

## Validation

Local validation covered formatting, targeted security tests, adversarial secured router tests, scoped clippy, no-default-feature compilation, WebUI contract fixtures, llama compatibility, workspace version consistency, and kernel dtype key static checks. The tests exercise missing and invalid keys, public health behavior, private route denial, hostile/null origins, cross-site Fetch Metadata, credential query rejection on public and private paths, percent-encoded credential names, DNS-rebinding Host rejection, preflight allowlisting, legacy `GET /models?reload`, encoded private paths, SSE permit retention until response-body drop, declared and streamed body limits, startup failure modes, strict allowed Origins, and custom path prefix classification.

## Deferred scope

Production CLI/startup mounting is intentionally deferred to #1838. Frontend credential handling, logout cleanup, and browser UX are owned by sibling WebUI client/design issues. No CUDA/GB10, real checkpoint, Safari, or VoiceOver validation is claimed for this server-policy PR; the GB10 runner was known down during publication and the change does not modify inference paths.
