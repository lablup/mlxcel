# Technical Report: PR #2201 - One post-sample finish step for every CLI and server decode site

**Date**: 2026-10-07

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: Rust

**Risk Level**: Medium. Every decode site now finishes through one function. Greedy output is byte-identical before and after on two real checkpoints. Four small, disclosed ordering or cadence changes remain, and they are unreachable or have no effect under default settings.

## Executive Summary

Phase 1 of epic #2166 (#2168). After a token is sampled, each decode site decides whether the sequence is done: EOS, history push, stop strings, generation bounds, structured stop, `max_tokens`, the context bound, loop detection and the cache-clear cadence. That decision was written out nine times, in four `CxxGenerator` loops and five server sites. `mlxcel_core::decode_finish::finish_step` is now the only implementation. The server reaches it through one hooks type in `src/server/batch/finish.rs`, which also holds the single `FinishCause` to `FinishReason` mapping. A side effect is that the speculative burst stream (MTP, DFlash) gains loop detection and the context bound.

## 1. Problem Statement

The finish logic had drifted between copies.
- Stop-string support needed five emission sites and two finalization sites.
- Generation bounds needed "all three paths".
- The speculative burst stream had no loop detection at all.
- A fix made at one site did not reach the others. That class of bug is what the epic exists to remove.

## 2. Change Summary

- **`decode_finish`:** `finish_step(input, hooks) -> Option<FinishCause>` checks, in one fixed order:
  1. EOS (not pushed)
  2. push and history
  3. stop string
  4. generation bound
  5. structured stop
  6. `max_tokens`
  7. context bound
  8. loop detection

  The cache-clear cadence runs only when the step did not finish. The CLI uses `NoStopHooks`.
- **`server::batch::finish`:** the server's `FinishHooks` implementation and the single cause-to-reason map. Used by the batched per-row, fused, single-step, prefill-completion and burst paths.
- **Burst stream:** finishes per token through the step. It gains loop detection and the context bound (`ContextBound` is threaded into `BurstContext`), while keeping matched-stop precedence and the detokenizer flush.
- **Burst donation gate** (from the security review): a burst that finishes before the last token of a batch the generator already produced no longer donates its state to the prompt cache, whatever the reason. Those trailing tokens are already in the target's model-owned state but not in `generated_tokens`, so a snapshot-reuse family would have stored a state ahead of its key. This also closes the same gap for stop strings and generation bounds, which existed before.
- **Removed:** `finish_on_generation_bound` and `context_bound_stop_due`; `ContextBound::stop_due` keeps the rule.

## 3. Technical Decisions

**One fixed order, shared by both sides.** The issue requires a single order, so a few edge cases now resolve the same way everywhere instead of per site. The changes are disclosed in the PR body:
- At prefill, a first token that trips both the structured stop and a generation bound reports `length`.
- On the CLI, a token that both spends `max_tokens` and completes a repetition loop is emitted and reported as `Length`. This is unreachable by default, because the CLI never enables loop detection.
- The cache-clear cadence is keyed on generated length (it moves by one token) and now also runs on the burst and prefill paths. The clear is off by default on CUDA.

**`generated_len` passed to the hooks.** The server hook borrows disjoint `SequenceInfo` fields while `finish_step` holds `generated_tokens` mutably, so it cannot read the length itself. Passing it in avoids a clone or an allocation per token.

## 4. Validation

- Unit tests:
  - `decode_finish` 9, `generate::finish_tests` 6, `generate` 61, `loop_detection` 22.
  - Server modules one at a time: `finish_step_tests` 13 (one test per site class over the same scenarios, plus the donation gate), `speculative_burst_tests` 58, `speculative_slice` 25, `stop_sequence_tests` 8, `scheduler::tests` 95, `sequence` 30, `scheduler_prompt_cache_tests` 31.
  - The burst loop test fails on `main`.
- Real checkpoints on GB10, `mlxcel-engine-parity` with `MLXCEL_SDPA_DETERMINISTIC=1 --no-prompt-cache-case --expect-identical`:
  - Models: qwen3-1.7b-4bit and llama-3.2-1b-instruct-4bit.
  - Runs: a 256-token run ending on `max_tokens`, and a one-word prompt ending on EOS.
  - CLI, server-dense and server-paged were identical in all 24 pairs.
  - Every token stream was byte-identical to the base commit's binary.
- Throughput is not measured here. The epic measures it once at the end against ADR 0007's 1.0 percent threshold.

## 5. Residual Risks

- Request-controlled loop-detection `min_count` has no upper cap (pre-existing, LOW).
- The thinking-budget override in bursts is unchanged (pre-existing).
- `prompt_lookup.rs` still calls `detect_repetition_loop` directly. It is outside the nine sites and is left for Phase 6.
- One intermittent `cudaStreamEndCapture` abort was seen when two scheduler test modules ran in one process. A later combined run passed, and it was not compared against `main`.

## 6. Learning Points

- **Collapsing copies forces the latent differences into the open.** Each disclosed behavior change is a place where the copies already disagreed.
- **A new finish path can create a new state-commit hazard.** Finishing mid-batch was safe for text output but not for snapshot donation. The security review found that because the change made mid-batch finishes more frequent.

## 7. Related

- Epic #2166, issue #2168.
- ADR 0007.
- Siblings #2169 and #2170.
