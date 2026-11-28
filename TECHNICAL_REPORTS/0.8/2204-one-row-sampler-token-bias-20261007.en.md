# Technical Report: PR #2204 - One per-row sampling step and token-bias composition

**Date**: 2026-10-07

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: Rust

**Risk Level**: Medium. Every per-row sampling site now runs through one type. CLI output is byte-identical before and after on two real checkpoints across greedy, seeded, penalty, DRY and language-bias configurations, and the CLI and server paths stay identical.

## Executive Summary

This is Phase 2 of epic #2166 (#2169).
- **`RowSampler`** (`mlxcel_core::sampling_row_step`) now owns the per-row sampling step on both sides:
  - the sampler-state lifecycle;
  - the history-ordering invariant behind #2090;
  - the structured-output mask;
  - the thinking-budget override.
- **Call sites:** the four `CxxGenerator` loops and the three server sites (batched per-row, `decode_single_step`, prefill first token).
- **Token bias:** `compose_token_bias` applies one precedence for the CLI and the server.
- **Fused path:** the fused batched path's eligibility is derived from the same stage list, so a new sampler stage cannot be missed by one side.

## 1. Problem Statement

The CLI and the server each had their own sampler-state handling.
- The CLI created `SamplerState` only for history-reading samplers and never called `accept_token`, so mirostat and adaptive-p state could not be carried. This was latent, because the CLI cannot enable those samplers.
- The token-history ordering that #2090 fixed in four CLI loops was implemented separately on the server.
- Token-bias composition (request bias, language bias, output suppression) had different precedence on each side.
- The fused-path gate `config_supports_fused_batch_except_bias` had to be extended by hand whenever a stage was added (#1485).

## 2. Change Summary

- **`RowSampler`:**
  - Creates the state once when `needs_token_history() || needs_sampler_feedback_state()`.
  - `needs_host_token_before_sample()` is true only for samplers that read history, so feedback-only samplers keep their state without forcing a host read.
  - The mask runs before the chain, and `resolve` returns the sampled and the final token for the thinking override.
  - It feeds `accept_token` on every accepted token.
- **`SamplerStage`:** fused eligibility comes from exhaustive matches over this enum, and still checks the lang-bias counters (#2188).
- **`compose_token_bias`** (`sampling_token_bias.rs`): request bias, then language bias only when the request map is empty, then output suppression last as a `-inf` overwrite that `+inf` and NaN request bias cannot override.
- **Windowed-penalty aggregates** (from review): `SamplerState::for_config` keeps full-history aggregates only for `penalty_last_n < 0`. Windowed configs, which include the server default 64, no longer absorb the whole prompt into aggregates they never read.

## 3. Technical Decisions

**Pipelining stays on the CLI.** The step tells the loop whether it needs the host token before sampling. The CLI keeps the lazy, pipelined decode for history-free samplers and reads the token first only when a sampler needs history, as #2117 established.

**The server's bias order is adopted.** The issue left the merge order open apart from "suppression always wins". The server's order was already the one requests depend on, so the CLI adopts it for the sources it has. Its output for those sources is unchanged.

## 4. Validation

- **Unit tests:**
  - mlxcel-core: `sampling` 200, `generate` 61, `decode_finish` 9, `sampler_state` 6, `sampling_token_bias` 5.
  - Root crate: `server::batch` 508 (7 ignored), `sampling` 54.
  - New tests: the windowed-aggregate case, and suppression beating `+inf`, NaN and accumulated-NaN bias.
- **Real checkpoints on GB10:**
  - CLI output was byte-identical to the base commit on qwen3-1.7b-4bit and llama-3.2-1b-instruct-4bit for greedy, seeded (temperature 0.8, repetition penalty 1.15, DRY 0.8) and language bias (6/6), and for windowed penalties (4/4).
  - `mlxcel-engine-parity --no-prompt-cache-case --expect-identical` gave 6 of 6 pairs identical on both models, matching the pre-change baseline.
- **Merge:** #2201 was merged into the branch. The only conflict was in imports, and one test fixture field was renamed.

## 5. Residual Risks

- The CLI-vs-server stream unit test compares against a written copy of the server step. The engine-parity harness covers the real scheduler.
- Pre-existing LOW issues for a separate issue:
  - `native_completion.rs` parses bias with `n as f32`, so `1e39` becomes `+inf`.
  - The handoff and CLI pipeline paths skip output suppression. This is text-only in practice.

## 6. Learning Points

- **Creating state eagerly has a cost.** Moving state creation to construction exposed full-history aggregates that the default windowed penalty never reads; the review caught the extra per-token work.
- **Derive gates from the stage list.** An exhaustive match over a stage enum turns a forgotten fused-path gate into a compile error.

## 7. Related

- Epic #2166, issue #2169.
- ADR 0007.
- #2090 / #2117 (history ordering).
- #2188 (lang-bias counters).
- #2201 (finish step).
