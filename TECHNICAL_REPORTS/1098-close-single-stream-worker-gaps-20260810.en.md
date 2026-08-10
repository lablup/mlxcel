# Technical Report: PR #1098 - Close single-stream worker gaps

**Date**: 2026-08-10
**Status**: Completed for an open PR
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

PR #1098 closes three related server-boundary gaps for single-stream model families: late queue admission on dedicated workers, silent image-cardinality drift between declared and resolved media, and Florence-2 prompt-token undercounting. The implementation started by fixing the dedicated worker and shared request-preparation paths, then a review hardening pass found a remaining `/v1/completions` and native `/completion` queue-overload hole and closed that too in head commit `cd3af51e`.

The net effect is that overloaded single-stream families now fail at the HTTP boundary with consistent `503` behavior before SSE opens, bad image declarations fail before worker dispatch without poisoning the live worker, and Florence-2 usage accounting reports the real fused encoder length rather than a partial proxy.

## 1. Problem Statement

Dedicated single-stream families in `mlxcel-server` do not run through the main batched scheduler path, so they need their own queue-depth reservation discipline. Before this PR, that discipline was incomplete in two ways:

- DiffusionGemma, LLaDA-2 MoE, and Florence-2 could still admit a request after the queue-depth snapshot and only discover saturation later, after some routes had already opened SSE.
- Shared chat media acquisition tolerated partial image resolution, which is correct for low-level acquisition helpers, but unsafe at the HTTP request-preparation boundary because a declared image count could silently collapse into a text-only request or a partial multimodal request.

Florence-2 had a second family-specific correctness issue: `usage.prompt_tokens` did not reflect the actual fused encoder sequence. The model consumes projected image-feature tokens plus the prompt tokens that survive image-placeholder filtering, so counting only the text side under-reported actual work.

## 2. Initial Implementation

The first implementation commit, `34b10437`, made three core changes.

### 2.1 Atomic queue reservations for dedicated single-stream workers

`src/server/model_provider.rs` and `src/server/state.rs` add RAII-backed `SingleStreamQueueReservation` handling on the shared queue-depth gauge used by admission control. DiffusionGemma, LLaDA-2 MoE, and Florence-2 now reserve before enqueue and release on dequeue or failed send, which makes route-level admission and observability reflect the real pending depth instead of a best-effort approximation.

The chat, Anthropic, and Responses routes were then wired to reserve before starting streaming work, so the failure mode moves from "SSE opened, then overload surfaced later" to a clean HTTP `503 Service Unavailable` response at the route boundary.

### 2.2 Shared image-cardinality validation at request preparation

`src/server/chat_request.rs` now validates the declared-versus-resolved image count after tolerant media acquisition completes, and `src/server/media.rs` exposes that validation as shared logic. This preserves tolerant low-level helpers for internal callers while making the HTTP request boundary fail closed.

The provider repeats the same guard for internal callers, and the regression tests prove a rejected mismatch does not poison the same live worker's next request.

### 2.3 Florence-2 fused prompt accounting

`src/models/florence2/model.rs` introduces `fused_prompt_len` and returns the actual fused encoder length from the greedy generation path. That length is the projected image-token count plus the prompt tokens that remain after filtering out the placeholder image token. The server-side Florence-2 worker uses that real fused length for `usage.prompt_tokens`.

## 3. Review Hardening

Review found a remaining route-boundary inconsistency: `/v1/completions` and the native `/completion` route still opened or dispatched through a path that could surface single-stream queue saturation after the admission snapshot, and their non-streaming error mapping still collapsed `QueueFullError` into a generic server error.

Head commit `cd3af51e` closes that gap by:

- reserving the single-stream queue slot before opening SSE on `/v1/completions` and `/completion`,
- routing both streaming paths through the reserved generation entry point,
- mapping `QueueFullError` to HTTP `503` / "All slots are busy" on both streaming and non-streaming paths, and
- extending route tests so chat, Responses, Anthropic-compatible messages, OpenAI Completions, and native `/completion` all prove the same overload behavior.

This hardening matters because it closes the last externally visible inconsistency. Without it, the PR would have fixed most single-stream families while leaving two text-completions surfaces with older overload semantics.

## 4. Confirmed / Refuted Matrix

| Family | `--max-queue-depth` gap | `usage.prompt_tokens` gap | Declared/resolved image gap |
|------|------|------|------|
| Florence-2 | Confirmed and fixed. Dedicated seq2seq serving now reserves before dispatch and returns route-level `503` on saturation, including the review-hardening completions coverage. | Confirmed and fixed. Usage now reports the actual fused encoder length from image feature shape plus filtered prompt tokens. | Confirmed and fixed. Shared request preparation rejects image-cardinality mismatches before worker dispatch. |
| DiffusionGemma | Confirmed and fixed. Dedicated diffusion serving now uses the same single-stream reservation path. | Refuted. Existing accounting already used the expanded engine prompt slice that generation consumes. Regression coverage was added instead of changing behavior. | Confirmed and fixed. Shared request preparation rejects image-cardinality mismatches before dispatch. |
| LLaDA-2 MoE | Confirmed and fixed. Dedicated LLaDA serving now uses the same single-stream reservation path. | Refuted. LLaDA rejects media before generation and already counts the same text prompt slice it sends to the generator. Regression coverage was added instead of changing behavior. | Confirmed and fixed. Shared request preparation rejects image-cardinality mismatches before dispatch. |

## 5. Change Summary

| Item | Value |
|------|-------|
| Implementation files changed | 27 |
| Implementation lines added | +1001 |
| Implementation lines deleted | -82 |
| Implementation commits | 2 |
| Reviewed implementation head | `cd3af51e18709ef9f3308486838f398881012957` |

The two bilingual report artifacts and their documentation commit are excluded from the implementation counts above.

- Added shared, atomic queue-slot reservation and release for dedicated single-stream workers.
- Moved route-boundary overload handling to pre-SSE reservation on chat, Responses, Anthropic/messages, OpenAI Completions, and native `/completion`.
- Added shared image-cardinality validation at the request-preparation boundary and provider-level recovery coverage.
- Corrected Florence-2 usage accounting to the actual fused encoder length.
- Updated user-facing docs for supported models, audio preprocessing, and block diffusion behavior.

## 6. Validation

### Local validation run on 2026-08-10

- `cargo fmt --check`: passed.
- `git diff --check origin/main...HEAD`: passed.
- `cargo clippy -p mlxcel --lib --tests -- -D warnings`: passed.
- `cargo test -p mlxcel --lib server::`: passed with `1673 passed; 0 failed; 8 ignored`.
- `cargo test -p mlxcel --lib server::model_provider::tests`: passed with `18 passed`.
- `cargo test -p mlxcel --lib server::chat_request::tests`: passed with `76 passed`.
- `cargo test -p mlxcel --lib server::max_tokens_route_tests`: passed with `7 passed`.
- `cargo test -p mlxcel --lib server::media::tests`: passed with `36 passed`.
- `cargo test -p mlxcel --lib server::state::tests`: passed with `17 passed`.
- `cargo test -p mlxcel --lib server::diffusion_worker::tests`: passed with `13 passed`.
- `cargo test -p mlxcel --lib models::florence2::model::florence2_fusion_tests`: passed with `38 passed`.

### Hosted checks observed for PR #1098

- `Detect changes`: pass
- `crate versions`: pass
- `kernel dtype keys`: pass
- `cross-repo refs`: pass
- `cargo-deny`: pass
- `cargo-fmt`: pass
- `license/cla`: pass
- `MLX pin extraction`: skipped

### Honest unavailable gates

- Full CUDA qualification was not run from this report pass. This repository's meaningful CUDA gate is hardware- and backend-dependent, and no full-CUDA claim is made here.
- Real-checkpoint validation for Florence-2, DiffusionGemma, and LLaDA-2 was not available locally. The required family checkpoints were not present on this machine, so no live checkpoint pass is claimed.

## 7. Key Technical Takeaways

The load-bearing design choice is to make queue admission a reservation, not a later observation. That keeps route behavior, worker enqueueing, and the queue-depth metric on the same state transition instead of letting them drift apart under race.

The second reusable lesson is boundary placement for tolerant helpers. Media acquisition can stay permissive internally, but the request boundary must validate declared-versus-resolved cardinality explicitly, otherwise a best-effort helper quietly changes request semantics.

## 8. Related Work

- PR #1098: https://github.com/lablup/mlxcel/pull/1098
- Issue #1086: https://github.com/lablup/mlxcel/issues/1086
