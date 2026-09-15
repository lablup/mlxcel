# Technical Report: PR #1896 — Patch rustls handshake boundaries

**Date:** 2026-09-15
**Issue:** #1895
**Scope:** Dependency security patch; no Rust source or inference arithmetic changes.

## Problem and decision

[RUSTSEC-2026-0285](https://rustsec.org/advisories/RUSTSEC-2026-0285) and [GHSA-2mjx-qc3c-rqvc](https://github.com/rustls/rustls/security/advisories/GHSA-2mjx-qc3c-rqvc) identify TLS 1.3 handshake messages accepted across encryption-level boundaries in rustls 0.23.13–0.23.44. The transcript remains authenticated; this is not an authentication-bypass claim. The newly published advisory caused the WebUI epic's cargo-deny gate to fail on rustls 0.23.43.

Run `cargo update -p rustls --precise 0.23.45` to select the first patched compatible release. Parsed lockfile comparison confirms that only the single rustls version and checksum change. Manifests, TLS features, explicit ring provider, and advisory policy remain unchanged. The existing grouped dependency PR #1880 does not contain this rustls patch and is not expanded into this fix.

## Validation

- Fresh `cargo deny --locked check`: advisories, bans, licenses, and sources pass using advisory database `e2e640471715167f73e22eaf761f2e547adafeec`, which contains this advisory. Existing warning-level duplicates and unused allowances remain unsuppressed.
- `cargo tree --locked -i rustls`: one rustls 0.23.45 shared by server TLS and HTTP clients. Parsed lockfile comparison and `git diff --check` pass.
- `cargo test --locked --profile test-fast --lib server::tls::tls_tests -- --test-threads=1`: 10 configuration/certificate tests pass.
- `cargo test --locked --profile test-fast --lib server::listen::listen_tests::a_tls_listener_completes_a_handshake_and_answers -- --exact --test-threads=1`: 1 actual loopback TLS handshake and HTTP response test passes. Both runs execute only CPU tests, with no ignored tests or failures. The build emits the pre-existing unused `BITLINEAR_HIP_SOURCE` C++ warning.
- Hosted source CI [34912229649](https://github.com/lablup/mlxcel/actions/runs/34912229649): cargo-deny, WebUI bundle/contract, and OpenXLA feature compile pass. Final branch status is checked separately before central merge; conditionally skipped jobs are not runtime evidence.
- Independent code, security, and performance review found no findings. Review was read-only; execution evidence above is separate.

## Limits

The existing TLS tests establish application compatibility, not a new reproduction of the malformed-record advisory. The security correction is supplied by the verified upstream release. No advisory exception, broad dependency update, GPU test, real-model inference, or performance benchmark is claimed. GPU timeout investigation and final integrated Safari verification remain separate epic work.
