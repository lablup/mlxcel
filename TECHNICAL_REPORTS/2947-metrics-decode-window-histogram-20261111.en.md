# PR #2947: Count only decode-window tokens in the /metrics decode histogram

**Date**: 2026-11-11
**Status**: Completed (PR open as a draft; closes #2938; the full `make verify-test` and clippy run was still pending when this report was written)

## Executive Summary

#2931 set one rule for every decode tok/s figure in the project: tokens produced inside the timed decode window divided by the window's duration, computed by `mlxcel_core::generate::decode_tok_per_sec`. It left the server's `/metrics` histogram `mlxcel_request_decode_tok_s` on the old rule, which divided all N completion tokens by `generation_only_ms`, a window that opens after the prefill-sampled first token. Issue #2938 tracked that gap. This PR adds `decode_window_tokens: Option<usize>` to `GenerationResult` next to `generation_only_ms`, gives the pair a written contract, and has the recorder observe `decode_tok_per_sec(decode_window_tokens, generation_only_ms)`. The batch scheduler path reports N-1. Florence-2 and Nemotron-Parse cannot separate their encoder pass from the decode loop, so they report `None` and are skipped rather than rated over wall time. The llama-server counters keep upstream's inputs. The last commit applies the same rule to the rate line `mlxcel run` prints. 17 files, +306/-70 (excluding this report).

## 1. Problem Statement

### 1.1 Two rules for one quantity

After #2931, `docs/benchmarks.md` "How decode tok/s is counted" defines decode tok/s as window tokens over window time: N-1 when the first token comes out of prefill, N when every token is produced inside the window. Bench tools, `generate`, the speculative engine and the native `/completion` response's `timings.predicted_per_second` all follow it. `BatchObservability::record_request_completion` in `src/server/batch/observability.rs` did not: it computed `completion_tokens * 1000 / generation_only_ms` locally. For the batch scheduler, `generation_only_ms` is `elapsed_ms - prompt_eval_ms`, which starts at the first-token instant, so the first token was counted against a window that did not contain it. A dashboard built on the histogram therefore read N/(N-1) higher than the response timings for the same request: 128/127 (0.79%) at N = 128, and double at N = 2.

### 1.2 `generation_only_ms` meant different things per producer

The issue also found that the field had no stated meaning. The shared builder in `src/server/model_worker.rs` reports a decode window. `src/server/florence2_worker.rs` and `src/server/nemotron_parse_worker.rs` report the whole wall time, encoder pass included, with `prompt_eval_ms: 0`. A consumer had no way to tell the two apart, so any rate computed from the field silently mixed a decode window with a wall time.

### 1.3 Degenerate single-token cases

A one-token completion has its only token come out of prefill, so its window holds no tokens. The old code skipped the observation only when `generation_only_ms` rounded to 0. With a nonzero tail (for example, the EOS step after the prefill token) it recorded 1 token over that tail, a rate that measured nothing.

## 2. Change Summary

### 2.1 The contract on `GenerationResult` (`src/server/model_provider.rs`)

`decode_window_tokens: Option<usize>` sits next to `generation_only_ms`, and both fields now carry doc comments that define the pair. With `Some(n)`, `generation_only_ms` is the interval from the first token to the finish and `n` is the tokens produced inside it: `completion_tokens - 1` for a prefill-seeded producer, `completion_tokens` for one that produces every token inside the window. With `None`, the producer has no isolated decode window and `generation_only_ms` is whole wall time. The doc also records that the llama-server `predicted_ms` field and counters read `generation_only_ms` either way.

### 2.2 Producers

- **Shared builder** (`build_generation_result_with_cache`, `src/server/model_worker.rs`): reports `Some(completion_tokens.saturating_sub(1))`. Its comment, written in `ab113b39` and `f50016cc`, says that every producer using the builder stamps the first-token instant through `StreamingDecodeState`: the batch scheduler (the only producer that reaches the histogram), the XLA worker and the diffusion worker. Diffusion commits a whole block at once, so part of its first block is computed before the window opens. For diffusion the N-1 figure is therefore a bound, not an exact count. This report takes those producer claims from the comment and the PR body; it did not trace each producer's stamping.
- **Count-only result** (`src/server/model_provider_prompt_count.rs`): `Some(0)`, an empty window, consistent with EOS at prefill.
- **Florence-2 and Nemotron-Parse**: the `Done` result construction moved out of the request handlers into pure functions, `florence2_generation_result` and `nemotron_parse_generation_result`. Both report `generation_only_ms: elapsed_ms` and `decode_window_tokens: None`, with comments explaining why. Other fields are unchanged. The extraction exists so the result can be tested without a model.
- **Test support** (`src/server/model_provider_test_support.rs`): the mock providers fill the new field (`Some(0)` and `Some(tokens_emitted - 1)`).

### 2.3 The recorder (`src/server/batch/observability.rs`)

`record_request_completion` gains a `decode_window_tokens: Option<usize>` parameter, passed through from the one call site in `src/server/batch/scheduler/decode_tick.rs`. A decode rate is observed only when the value is `Some(n)` with `n > 0` and `generation_only_ms > 0`, and it is computed by `decode_tok_per_sec` instead of a local division. TTFT and the llama-server counters are recorded as before, including for `None` results. The doc comment states that the llama counters keep b10621's accumulation of every completion token over `generation_only_ms`.

### 2.4 `mlxcel run` (`src/commands/run.rs`, commit `a04b39db`)

The rate in the `[Generated N tokens in S s = R tok/s]` line now uses `decode_tok_per_sec(window_tokens, generation_only_ms)` when the result carries `Some`, and keeps the old `completion_tokens / seconds` for `None`. A grep of non-test sources found no other consumer that still divides `completion_tokens` by `generation_only_ms` to get a decode rate. The native timings already use N-1 since #2931, and the engine probe only forwards the milliseconds.

### 2.5 Docs

- `docs/benchmarks.md` gains a paragraph under "How decode tok/s is counted" describing the histogram's inputs, the N-1 count, the skip cases and the unchanged llama counters.
- The `/metrics` HELP text for `mlxcel_request_decode_tok_s` says it counts only tokens produced inside the decode window.
- `CHANGELOG.md` has an Unreleased entry for the histogram change.
- `docs/llama-server-compat.md` is unchanged, because neither the `llamacpp:tokens_predicted_*` counters nor `timings.predicted_*` change.

## 3. Validation

Tests added or changed:

| Test | What it pins |
|------|--------------|
| `observability::decode_rate_counts_only_window_tokens_for_a_prefill_seeded_request` | 128 tokens, 127 in a 1270 ms window, records exactly 100 tok/s, equal to `decode_tok_per_sec(127, 1270.0)` |
| `observability::decode_rate_skips_a_request_without_a_decode_window` | a `None` result records no decode rate, while TTFT and the llama counters (30 tokens, 900000 us) still see it |
| `observability::record_request_completion_skips_zero_token_requests` | adds the one-token case with a 15 ms tail: no 0 tok/s observation |
| `observability::record_request_completion_populates_histograms`, `histogram_counts_plus_inf_observations` | existing expectations restated with window counts |
| `model_worker_tests::build_generation_result_computes_finish_reason_and_generation_split` | N-1 window tokens, including `Some(0)` for an empty generation |
| `florence2_worker_tests`, `nemotron_parse_worker_tests` (`result_reports_wall_time_without_a_decode_window`) | wall time, `None`, and an empty decode histogram after feeding the result to the recorder |
| `metrics_llama_tests` | llama counter values unchanged with the new argument |

- **Gate result: pending.** The full `make verify-test` and `cargo clippy --workspace --all-targets -- -D warnings` run was in progress in the worktree when this report was written. This report records no pass or fail for it.
- **No real-checkpoint measurement is recorded.** The PR body gives a procedure (one native `/completion` with `n_predict: 128` under `mlxcel-server --metrics`, then compare the change in `mlxcel_request_decode_tok_s_sum` with the response's `timings.predicted_per_second`), but no result from running it appears in the PR.
- The `mlxcel run` change in `a04b39db` touches only `src/commands/run.rs` and adds no test.

## 4. Technical Decisions

### 4.1 Carry the window count in the result instead of deriving it in the recorder

**Context:** the recorder could have computed `completion_tokens - 1` itself, since its only caller is the batch scheduler.

| Option | Pros | Cons |
|--------|------|------|
| `decode_window_tokens: Option<usize>` on `GenerationResult` (chosen) | The producer, which knows where its window opens, states the count; the `None` arm can express "no window"; adding a field to a struct built with literals forces every constructor to choose a value at compile time | Every `GenerationResult` literal had to change, including test support |
| Subtract 1 inside `record_request_completion` | One-line change | Bakes the scheduler's prefill-seeded shape into a shared recorder, and is wrong for any producer whose window holds all N |
| A boolean "prefill seeded" flag | Smaller than a count | Cannot describe a producer whose window count is neither N nor N-1, and still needs a third state for "no window" |

The second and third rows are this report's analysis; the PR body does not discuss them.

### 4.2 Skip, rather than approximate, the encoder-decoder workers

Florence-2 and Nemotron-Parse run encode and decode inside one model-owned call, so the worker cannot see where decoding started. Recording their wall-time rate would put encoder time into a decode histogram. Splitting the call would mean changing model code to expose an inner timestamp, which is outside a metrics fix. `None` keeps their rate out and documents why. Today these workers send `Done` directly and never reach `record_request_completion`, so the skip is a guard for a future routing change rather than a behavior change. The PR explicitly declines to wire them into the recorder, because that would also start feeding their `prompt_eval_ms: 0` into the TTFT histogram and the llama counters.

### 4.3 Leave the llama-server counters on upstream's inputs

`llamacpp:tokens_predicted_total` and `llamacpp:tokens_predicted_seconds_total` mirror b10621's `n_decoded` and `t_token_generation` accumulation, and deployment scripts compare them with a real `llama-server`. Changing them would trade upstream compatibility for internal consistency, and the compatibility manifest would then need an entry. The PR keeps them unchanged, so `/metrics` now exposes two rates built on different numerators: the histogram (window tokens) and the counter ratio (all tokens). That difference is documented in `docs/benchmarks.md`.

### 4.4 One helper for the division

Computing the rate through `decode_tok_per_sec` instead of a local expression means a future change to the rule (for example, its zero handling) applies to the histogram too. The recorder's own `window_tokens > 0 && generation_only_ms > 0` guard is still needed, because the helper returns 0.0 in those cases and a 0 tok/s observation would skew the histogram.

## 5. Limits and Open Items

- **The `mlxcel run` change is not in the PR body or CHANGELOG.** Both describe only the histogram. A user who compares a `run` rate before and after this PR sees it drop by 1/N without an entry explaining it.
- **The `run` line no longer divides out.** It still prints `completion_tokens` and the window seconds next to a rate computed from N-1 tokens, so N / S no longer equals the printed rate. A single-token turn now prints 0.00 tok/s, where it used to print 1 / S whenever the window was nonzero.
- **Size of the change in the histogram.** Each observation becomes (N-1)/N of its old value: 0.78% lower at N = 128 (the old value was 128/127, 0.79%, higher), 50% lower at N = 2. The PR body and CHANGELOG write "drops by (N-1)/N" next to those figures. (N-1)/N is the ratio of new to old. The drop itself is 1/N.
- **Diffusion's count is a bound.** It does not reach the histogram, but `mlxcel run` on a diffusion model would print a rate based on it.
- **The producer claims in the builder comment are untested.** No test checks that the XLA or diffusion worker stamps the first-token instant the way the comment says.
- **Gate pending and no real-checkpoint scrape**, see section 3.

## 6. Learning Points

- A duration field with no stated interval tends to collect producers that each measure something different. Writing the contract into the field doc, and pairing the duration with the count it is meant to divide, is what made the Florence-2 and Nemotron-Parse mismatch visible.
- `Option` with a documented `None` is a better way to say "not measurable" than a sentinel such as `0` or the wall time, because a consumer that ignores it fails to compile instead of producing a plausible number.
- Moving a struct literal out of a channel-sending handler into a pure function is a cheap way to make worker output testable without loading a model.
- When the project fixes a counting rule, search every consumer of the inputs, not only the one the issue names. Here the same N-over-window division also lived in `mlxcel run` and was found only in a follow-up commit.
