# Technical Report: PR #2117 - Sample each penalized step against the token just read

**Date**: 2026-10-06

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: Rust (decode loops, shared test fixtures, tests)

**Risk Level**: Low. Output changes only under a history-reading sampler, and there it moves to the documented semantics; the no-penalty path keeps its order, output and throughput.

## Executive Summary

`CxxGenerator` is the decode loop behind `mlxcel generate`, `run` and interactive chat. Its four pipelined loops sampled step t+1 before pushing token t into `token_history`, so repetition, frequency and presence penalties and DRY always saw a history one token stale (#2090). A shared helper now reads and records token t before the step t+1 sample is built, while the step t+1 forward is already running on the device. No-penalty output and throughput are unchanged; penalty-path throughput is unchanged as well.

## 1. Problem Statement

A pipelined loop builds step t+1 from the still-lazy sample of step t, so the GPU goes straight from one step to the next. The forward needs only that lazy token, but a history-reading sampler needs token t on the host, in `token_history`. All four loops (`generate_streaming`, `generate_streaming_with_embeddings`, `generate_with_stats_and_embeddings`, `generate_with_stats`) built the step t+1 sample at the top of the iteration and pushed token t at the bottom. The first decode token was sampled against the prompt alone. With `--repeat-last-n 3` the window was shifted by one; a token could repeat immediately after itself unpenalized. mlx-lm (`generate_step` concatenates the consumed token before its logits processors run) and llama.cpp (`common_sampler_accept` after each sample) both include it.

The bug was found while hardening PR #2074: prompt lookup, which samples verify positions sequentially, disagreed with `CxxGenerator` under a penalty, and a host-side simulation matched prompt lookup to the current history and `CxxGenerator` to a history lagging by one token.

## 2. Change Summary

- `sample_next_step(next_logits, y, sampling, needs_history, token_history, sampler_state)` in `generate.rs`: on the history path it calls `async_eval(next_logits)`, reads `y`, pushes it, then builds the sample, and returns the token so the loop reuses it. On the history-free path it returns the old `sample_token_optimized` result untouched.
- Each loop holds `current_token: Option<i32>`, reads `y` only when the helper did not, and pushes into `token_history` only when the helper did not.
- `test_support::induction`: `InductionModel`, `lcg_tokens`, `sequential_reference`, moved from `prompt_lookup_tests.rs` so both test files share them.
- `generate_history_tests.rs`: all four entry points against the sequential reference under a three-token repetition window, full-history counts (the incremental `SamplerState`), frequency/presence, DRY, and no penalty.
- Prompt lookup's penalty test checks `CxxGenerator` against the reference again.
- Docs at the helper on what diagnostics see on the penalty path.

## 3. Technical Decisions

**Submit the forward, then read.** The issue's own proposal. The step t+1 forward depends only on the lazy token, so submitting it before the host read keeps the device busy through the read; only the sampler, which needs the host history, waits. The alternative of turning the penalty path synchronous would have lost that overlap, and a device-side lazy history (mlx-lm's approach) would have rewritten `SamplerState` and DRY, which consume a host `&[i32]`.

**One helper for four loops.** The ordering rule lives in one function; the loops only thread `current_token` through their existing read and push sites.

**Discriminating tests, checked by reverting.** Restoring the old order fails the windowed repetition, frequency/presence, full-history count and DRY tests, and prompt lookup's penalty test. A first version of the full-history test used a repetition penalty and passed under the bug: once every token of a small vocabulary is in the history, the penalty set no longer changes. It was replaced with count-based penalties, which every occurrence moves.

## 4. Validation

- `generate::history_tests` and `speculative::prompt_lookup`: 44 passed.
- GB10, release binaries built from one tree (only `generate.rs` swapped), greedy, median of 3, interleaved: no-penalty throughput 0.990x to 1.007x of before, penalty throughput 0.997x to 1.010x; no-penalty output identical in four of five cases. The fifth (Qwen3-1.7B, 800-token story) is not reproducible run to run on this host even with the unchanged binary (three runs diverge at characters 1300 to 1351), so its before/after difference at character 1432 says nothing about this change.
- Server batch scheduler: unaffected. Its lookahead admits only rows passing `config_supports_fused_batch_except_bias`, which rejects `needs_token_history`; per-row paths read and push each token in the tick that sampled it, and prefill pushes the first token before decode.

## 5. Residual Risks and What Was Not Verified

- **Plain greedy decoding is not reproducible run to run on GB10** for at least one long Qwen3-1.7B reply. Pre-existing and untracked; root cause not investigated here.
- **Diagnostics on the penalty path** (decode-graph export, astype count, Metal capture, pipeline profiles) now see the sampler without the forward and attribute the host wait to sample time; documented at the helper.
- **`CxxGenerator` carries no mirostat or adaptive-p state.** Unreachable today: `generate` and `chat` fix both off, and the server never uses `CxxGenerator`.
- **`SpeculativeGenerator`** samples each draft against a history without that round's earlier drafts (out of scope here).
- Metal was not measured; the change is backend-independent host logic.

## 6. Learning Points

- **In a pipelined loop, check what each consumer needs on the host.** The forward could run from the lazy token; the sampler could not. Splitting the submission keeps the overlap and fixes the semantics.
- **A regression test must fail on the bug.** One of five tests here passed under the old ordering until its sampler was changed; only a deliberate revert showed it.
- **Before/after parity needs a self-parity baseline.** A greedy reply that is not reproducible against itself cannot attribute a difference to the change.

## 7. Related

- Issue #2090, PR #2074 (where it was found), PR #2092 (prompt-lookup policy, same test fixtures).
