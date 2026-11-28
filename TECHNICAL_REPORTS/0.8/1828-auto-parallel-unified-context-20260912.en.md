# Technical Report: PR #1828 - fix: share auto parallel context budget

**Date**: 2026-09-12
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle (implementation review, security and performance review)
**Status**: Completed (the exact QAT checkpoint named by the issue was unavailable; a same-family local checkpoint was used)
**Languages**: Rust, JSON, Markdown
**Risk Level**: Medium-high (scheduler admission and decode limits change; inference arithmetic is unchanged)

---

## Executive Summary

PR #1828 completes llama-server b10621's automatic parallel-context behavior. Leaving `--parallel` on its `-1` default now resolves four visible slots and enables unified context budgeting, so each slot may use the whole configured context while all live sequences share one logical token budget. Explicit `--parallel` or `--max-batch-size` retains split windows unless `--kv-unified` is also explicit, and `--no-kv-unified` always selects split mode.

mlxcel continues to allocate KV state per sequence. Unified mode therefore implements the externally visible b10621 contract as scheduler accounting rather than changing the physical cache layout. The scheduler checks the combined prompt and generated-token footprint at admission, immediately before prefill, for the first sampled token, and before every decode growth step. When no growth fits, it evicts reusable prompt-cache state first and then length-finishes every live sequence with context-exhausted metadata.

---

## Problem and Resolved Behavior

Before this change, auto `--parallel` produced four slots but divided `--ctx-size` into four fixed shares. A deployment that requested `--ctx-size 20480` without an explicit parallel value therefore exposed four 5,120-token slots, while b10621 exposes four slots whose individual window is 20,480 tokens and limits only their combined live use. The mismatch was especially visible for a family that is discovered as non-batching after load: the worker reduced decode width to one but left the only usable request with a quarter of the configured context.

The resolved policy is:

- Auto `--parallel -1`: four slots, unified mode on, full context reported for every slot.
- Explicit `--parallel N` or `--max-batch-size N`: split windows by default.
- Explicit `--kv-unified` / `-kvu` / `LLAMA_ARG_KV_UNIFIED=true`: full per-slot window with one shared live-token budget.
- Explicit `--no-kv-unified` / `-no-kvu`: split windows.
- A post-load non-batching clamp: one effective decode row regains the whole configured context, and runtime metadata is republished so it does not retain the stale pre-load share.

The b10621 compatibility manifest now classifies `--parallel`, `--ctx-size`, and `--kv-unified` as supported and records the narrow non-batching post-load divergence.

---

## Implementation and Technical Decisions

The CLI preserves whether the original parallel value was automatic instead of inferring it from the resolved value of four. Startup combines that source bit with explicit unified/split flags and an explicit max-batch override, then stores both the total context and the effective per-slot context in `ServerConfig`.

`BatchScheduler` receives an optional shared budget. Live use is the saturating sum of prompt plus generated tokens for every unfinished decode-active sequence and the parked chunked-prefill sequence. Checked addition rejects arithmetic overflow. Admission and prefill recheck the budget after queueing because existing sequences can grow while a request waits. Decode reserves one logical token per active row before executing a tick, and speculative bursts are disabled in unified mode so one check cannot be bypassed by multi-token growth.

Prompt-cache entries are not counted as live sequence ownership, but are cleared before a shared-budget rejection or terminal exhaustion to mirror b10621's reclaim-first policy. If a decode tick still cannot fit, every live sequence is marked `FinishReason::Length` with `context_exhausted`, producing the documented deterministic fail-all result.

The worker can only discover some non-batching families after model load. It now publishes corrected runtime context and KV limits through release/acquire atomics in `BatchMetrics`. `/props`, `/slots`, and `/v1/models` read the effective runtime value, and a second geometry log records the post-load resolution. This keeps hot metadata reads lock-free and prevents control-plane reporting from disagreeing with the actual worker limit.

---

## Review, Security, and Performance

Implementation review corrected the compatibility shard pointers, assigned issue #1815 to the runtime-and-context shard, expanded shared-budget accounting tests, and added a static wiring test for every admission, prefill, decode, and runtime-publication boundary. A real-model smoke then exposed stale explicit-split metadata after the non-batching clamp; commit `b039be14` fixed that discrepancy and added regression coverage.

Security and performance review found no remaining CRITICAL or HIGH issue. Token totals use checked or saturating arithmetic, budget decisions are made before model work grows live state, and no new unbounded user-controlled allocation, filesystem surface, credential path, or lock ordering was introduced. Shared accounting is linear in the bounded active batch. Metadata publication uses atomics, while prompt-cache eviction occurs only after normal capacity checks fail.

No tensor operation, kernel selection, quantization rule, or numerical arithmetic changed, so a teacher-forced logit trace is not applicable to this control-plane and scheduler-budget change.

---

## Validation and Limitations

- `cargo test --workspace --profile test-fast --features metal,accelerate` passed on the final implementation head with zero failures, including all workspace test binaries and doctests.
- Focused suites passed for unified resolution and windows, shared live-token accounting and overflow rejection, non-batching clamp behavior, metadata geometry, post-load publication, and scheduler wiring.
- `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings`, `cargo fmt --check`, and `git diff --check` passed.
- `scripts/ci/check_llama_compat_manifest.py` passed with 249 help entries, 323 spellings, 136 environment variables, 53 routes, 74 native fields, and no deferred issue.
- GitHub CI passed crate-version, kernel dtype-key, license, compatibility, cross-repository reference, dependency-policy, formatting, clippy, and OpenXLA compile checks on the reviewed implementation head.
- The exact issue checkpoint `models/gemma-4-12B-it-qat-4bit` was absent. The same-family local checkpoint `models/gemma-4-12b-it-4bit` was used explicitly as a substitute:
  - Auto mode reported four slots, `kv_unified: true`, and `n_ctx: 20480` on `/props`, `/slots`, and `/v1/models`.
  - A 16,002-token completion request succeeded; a 21,002-token request was rejected with `exceed_context_size_error` and an available context of 20,480.
  - `--parallel 4 --kv-unified` reported the same unified geometry.
  - `--parallel 4` without unified mode triggered the model's non-batching clamp, restored the whole 20,480-token request window, and reported the corrected post-load value on every metadata surface.

The substitute validates the same family and the exact post-load non-batching path, but it is not evidence about the unavailable QAT checkpoint's weights. Explicit 5,120-token split geometry remains covered by deterministic startup and route tests for batching-capable configurations.
