# Technical Report: PR #1507 - Gate boundary snapshots by model capability

**Date**: 2026-08-30
**Status**: Completed
**Languages**: Rust
**Risk Level**: Medium

## Executive Summary

PR #1507 publishes the loaded model's snapshot-reuse capability from the worker thread to HTTP request preparation and uses it to skip the history-boundary render for dense-KV models. Snapshot-capable models keep the existing boundary path, while the startup window deliberately fails open until the worker publishes `loaded=true` so the first eligible request cannot lose a boundary snapshot.

## 1. Problem Statement

Issue #1153 identified wasted work in the prompt-cache chat path. Text-only chat requests rendered the conversation twice and queued a second tokenization even for dense-KV models whose scheduler path cannot consume model-owned recurrent snapshots.

The previous only end-to-end opt-out was `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1`. That protected operators who knew the switch, but it did not let the server choose automatically from the loaded model trait that already owns the real capability answer.

Leaving this path ungated makes long dense-KV conversations pay a full extra Minijinja render, `String` clone, and tokenizer encode per request with no cache benefit.

## 2. Technical Decisions

### 2.1 Publish the trait result through the provider

`ModelProvider` now carries `snapshot_reuse_capable` beside the existing `loaded` flag. Batch and legacy workers store `model.supports_snapshot_reuse()` immediately after successful model construction and before `loaded=true`; the XLA worker stores `false` because that path does not use model-owned snapshot reuse.

This avoids a duplicate static family table. The implementation follows the same trait method that the scheduler later consults, so future model additions only need to update the model implementation.

### 2.2 Gate request preparation, not only scheduler adoption

Every HTTP path that calls `prepare_chat_request_with_cache` now passes a separate `snapshot_reuse_capable` boolean. The history render only happens when prompt cache is enabled, the model can reuse snapshots, the request is text-only, and the manual kill switch is not set.

The disaggregated router remains text-only and passes `false`, so it never pays this single-node snapshot boundary work.

### 2.3 Fail open before readiness

`AppState::should_render_history_boundary_snapshot()` returns true until the worker marks the provider loaded. After readiness, it follows the worker-published trait result. This preserves issue #1153's ordering requirement: a request admitted during startup cannot skip the first boundary snapshot for a model that later proves snapshot-capable.

## 3. Change Summary

| Category | Count | Summary |
|---|---:|---|
| Capability plumbing | 1 | Added `snapshot_reuse_capable` to `ModelProvider`, worker constructors, and test fixtures. |
| Request preparation | 1 | Extended `prepare_chat_request_with_cache` and every caller with the capability gate. |
| Tests | 3 | Added history-render attempt instrumentation, dense-KV no-render coverage, and kept snapshot-capable prefix coverage. |
| Documentation comments | 2 | Updated the `PreparedChatRequest::history_prompt` and state accessor comments to describe the new gate and startup ordering. |

## 4. Validation

- `cargo test --lib history_render_is`: 5 passed.
- `cargo test --lib prompt_cache_on_`: 4 passed.
- `cargo test --lib single_stream`: 5 passed.

Real llama-3.2-1b-instruct and qwen3.5-0.8b-4bit benchmark parity was not run in this Linux worktree. The PR verifies the code path with targeted unit tests; checkpoint counter parity remains a qualified hardware/model validation item.

## 5. Review Notes

- **Correctness**: The capability write happens before `loaded=true` with release ordering, and route reads happen through provider accessors. This preserves the existing readiness ordering while avoiding false negative capability reads after readiness.
- **Security**: No new request data is exposed. The added flag is a process-local model capability, not prompt or user metadata.
- **Performance**: Dense-KV text-only requests skip the extra history render and therefore the downstream history-tokenization path. Snapshot-capable models retain the existing behavior.
- **Compatibility**: No public API changes. Route behavior changes only by omitting an internal optimization artifact for models that cannot consume it.

## 6. Follow-up Actions

- Run the real checkpoint matrix from issue #1153 on a qualified machine: llama dense-KV no-render counters, qwen3.5 snapshot hit parity, and the documented history-boundary benchmark numbers.
- Keep future snapshot-capable model additions tied to `LanguageModel::supports_snapshot_reuse()` rather than adding HTTP-side family tables.
