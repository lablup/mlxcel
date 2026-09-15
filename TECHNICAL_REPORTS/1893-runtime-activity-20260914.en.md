# Technical Report: PR #1893 — Runtime observation and Activity

**Date**: 2026-09-14

**Status**: Partial — implementation and bounded automated reviews complete; root real-model/performance and final integrated manual gates remain outstanding.

**Languages**: Rust, TypeScript/React, JavaScript, OpenAPI/JSON Schema

**Risk level**: Medium

## Executive summary

Issue #1847 adds the Activity screen to epic #1834 using the published npm `@lablup/ui-common@0.1.0-alpha.19` adapters. Existing provider, lifecycle, slot and metrics owners remain authoritative. The UI distinguishes unavailable values from measured zero and does not initialize MLX, trigger autoload or add GPU sampling.

## Architecture and behavior

The canonical runtime projection moved from `webui/api.rs` into `webui/runtime.rs`, with required typed `slots` and strict whole-response fixtures. Ready provider observation clones a loaded state only when the visible catalog identity/revision still matches; CPU counters and a bounded redacted slot registry are the only new reads. `--metrics` and `--slots` remain effective. Requests, provider-lifetime completed counters and cache metrics carry units, scope, timestamps and source/window explanations. General effective parallelism, process/allocator/device memory, exact KV bytes, TTFT and interval decode rate remain N/A when no cheap authoritative source is available.

The merged #1800 source tracks current context occupancy including accepted decode tokens. Activity uses that counter without adding decoded tokens twice, and does not divide by unknown/zero context capacity. Shared-pool geometry remains separately nullable. The later #1815 worker-published width-one/context authority is consumed, not reconstructed from configured limits.

Activity shows active and terminal operations, explicit cancellation acknowledgement, typed failures, selected runtime, partial slot warnings, sanitized export and a lazy discrete history plot. One shared synchronizer owns two-second visible observation, ten-second snapshot timeout, hidden-tab request/stream/timer teardown and reconnect. Provider model selection now calls observation-only `selectionChanged()`, preserving in-flight inference; logout still aborts the complete session. A complete unfiltered catalog clears a removed selected model, with generation and page consistency checks.

The selected-model history is memory-only, five minutes/150 two-second buckets. Terminal operation retention is capped at 200 for one hour, including repeated same-sequence snapshots; active operations are never silently discarded. Diagnostics use a strict allowlist rather than attempting to redact arbitrary free text, excluding credentials, private paths, prompts, output, model identifiers and operation errors.

## Independent review corrections

- Correctness review found the new observation-only cancellation seam was not wired by the real provider. Ownership was explicitly transferred from the later Chat unit; provider wiring and a real shared-client stream-preservation regression now cover selection and logout.
- Security/performance review found terminal TTL bypass during snapshot ingestion, cross-model contamination of the selected history ring, and a hidden warning for available-but-truncated slots. All were corrected with targeted regressions and rereviewed with no remaining findings.
- Chat separately owns the pre-existing abort-reader rejection race fix; Activity does not duplicate it or infer successful inference completion from a resolved transport promise.

## Executed validation

Historical pre-integration implementation source: `51fe6579958718b72f1a7a89bbeb29d2c4cb886d`. Subsequent report/harness-only changes do not alter the runtime or frontend bundle.

| Gate | Result |
|---|---|
| Frontend unit suite | 94 passed, 13 files |
| Strict TypeScript and ESLint | Passed |
| CPU-only runtime projection selectors | 6 passed |
| Revision/LRU/lease observation regression | 1 passed |
| Scoped library/tests Clippy | Passed |
| Canonical fixtures, DTO drift and schema strictness | 48 passed |
| Compatibility/version/kernel structural gates, fmt/diff | Passed |
| Deterministic embedded bundle rebuild and verify | Passed; digest `6a887a0b8fb8c4ab7f61c783ed52d44de16160fc985bd76e9ede3c5dea898a57` |
| Local production browser, real model and GPU/performance | Not executed by this unit |

The initial runtime test build had an incorrect fixture-relative include path. A subsequent whole-ready JSON test exposed integer `0` versus the producer's f64 `0.0` representation in the authored fixture; only fixture representation was corrected, without weakening whole-response assertions or modifying numerical inference code. Original logs remain available under `/tmp/1847-runtime-*`. Initial timer cleanup tests also exposed a retained timeout; immediate teardown now clears it.

Hosted source `51fe6579` passed all executable checks. The WebUI job `103992971327` in run `34849305758` actually executed 94 unit tests, 22 browser cases (including both new Activity cases), and deterministic bundle verification successfully. Skipped compile/runtime steps are not claimed as passes. The current GB10 runner state is not a blanket waiver for actual failures.

## Remaining acceptance and integration

`webui/scripts/activity-performance.mjs` prepares five alternating-order UI-off/treatment pairs for one visible, two visible and actually hidden headed-browser clients. It requires native backend `timings.predicted_per_second`, records median paired degradation and noise, and flags >2% degradation or >5% baseline CV for investigation. `activity-slots.mjs` captures an actual long chat stream with before/runtime/after native slot samples, strips all text, bounds time and leaves the result explicitly requiring comparison. These scripts are prepared and syntax/lint-checked, not measured acceptance results.

Central integration order is #1846 → #1847 → #1844 → #1845. The Settings integration described below is complete. Root-owned full/real-model/production-CSP/performance acceptance remains outstanding. Safari/VoiceOver/native 200% checks are explicitly deferred to the final integrated manual session, not passed. No existing user checkpoints were modified or removed.

## September 15 Settings integration

Measured integrated source: `93c825b8a6dc432d8f2427fd1cc8edee1ba2dde0`, based on merged Settings `2a91bf3069fb84fdc729d688baad8747136af252` and the Rustls security update. Runtime metrics and actual effective configuration now share one revision-fenced loaded AppState; CLI override provenance survives. Settings session-fenced operations and Activity observation-only selection both retain their provider regressions. The generated bundle includes both screens, the corrected Field accessible names, and the canonical eight-character Vite asset hashes.

Executed integration gates: 137 frontend tests in 19 files, 14 offline SSE acceptance tests, TypeScript, ESLint, 49 canonical fixtures/schema checks, six CPU runtime tests and one revision/LRU observation test, formatting/diff checks, and deterministic bundle verification all passed. Integrated bundle digest: `d9102d2559e51de2f16dc2c9a409c6ad5860c65157fb10f8ba190181a8a83dc8`. Previous hosted browser evidence above applies only to its recorded historical source; no new local browser, full workspace, real-model or performance pass is claimed here.

The real slots harness now validates bounded UTF-8 SSE frames, rejects explicit errors, malformed/truncated responses and post-finish choices, and requires a successful finish reason followed by `[DONE]`. It retains no generated text. An independent read-only correctness/security review verified the integration seams and identified the post-finish acceptance gap; it was fixed with regressions and rereviewed with no remaining findings. The maintainer deferred GPU timeout root-cause diagnosis, not correctness assertions or failed acceptance results. Native Safari/VoiceOver/200% remains deferred to the final integrated session.
