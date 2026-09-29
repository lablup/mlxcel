# Technical Report: Issue #1373 - Hyperparameter-Free p-less Sampling

**Date**: 2026-09-29

**Status**: Implemented and validated locally on Apple Silicon; pending merge.

**Languages**: Rust

**Risk Level**: Low (off by default, byte-identical when disabled)

## Executive Summary

mlxcel gains a `p_less` sampler: a boolean request field and a `--p-less` flag on `mlxcel generate` and `mlxcel chat`. The filter keeps every token whose probability under the temperature-scaled softmax is at least the row's collision probability `L = sum_v p(v)^2`, and masks the rest to `-inf`. It has no tunable parameter. It is a third filter on the row-filter hook that #1375 introduced, so it needed no new dispatch path.

## 1. Problem Statement

Truncation samplers (top-k, top-p, min-p, typical-p, top-n-sigma) all need a cutoff the user must tune per model and per temperature. p-less derives its cutoff from the distribution itself. Since `L <= max_v p(v)` the argmax always survives, a sharply peaked row collapses to the argmax, and a flat row keeps most of the vocabulary. mlxcel had no such sampler.

## 2. Change Summary

- `SamplingConfig::p_less: bool` (default `false`) and `SamplingConfig::effective_p_less()`, which folds to `false` on the greedy path and under mirostat.
- `sampling.rs`: `p_less_filter(logits, temperature)`, the `p_less` arm of `apply_row_filters` between `top_n_sigma` and `typical_p`, the same filter in `apply_extended_chain`, and `FusedSampleParams::p_less` (compared in `matches`, so uniform batches stay single-dispatch).
- Request plumbing on the #1375 pattern: `SamplingParams::p_less`, `RequestOptionOverrides::p_less`, `ResolvedSamplingParams::p_less` through both `build_sampling_config` branches, `NativeCompletionRequest::p_less`, the disaggregated `SerializableSamplingState::p_less` (`#[serde(default)]`), and the batched speculative window equality.
- CLI: `--p-less` on `SamplingOptions`, used by `generate` and `chat`.
- Docs: `docs/server-features.md`, `CHANGELOG.md`.

## 3. Technical Decisions

**The filter takes the temperature explicitly.** The fused C++ chain applies the temperature after the row filters, so the filter would otherwise see the untempered distribution. p-less is defined on the tempered one, so `p_less_filter` divides by `T` itself (skipped at `T == 1.0`) and the fused chain's own scaling is unchanged. Raising the temperature therefore never shrinks the kept set.

**Masked entries stay masked without special casing.** A `-inf` logit has probability exactly 0, `L > 0` for any row with a finite entry, and `0 >= L` is false, so the existing mask survives and appended `-inf` entries change neither `L` nor the kept set.

**Effective value, not raw field.** `FusedSampleParams::from_config` stores `effective_p_less()`, so greedy rows that differ only in an inert `p_less` do not split a fused batch or a speculative window. This mirrors `effective_top_n_sigma`.

**Extended chain.** `apply_extended_chain` does not go through `apply_row_filters`, so the filter is applied there too, in the same position. It is not applied twice on any path.

**No server-wide default.** A bool has no disabled sentinel to sanitize, and the issue specifies that an absent field resolves to `false`, so there is no `--p-less` server flag and no runtime setting.

## 4. Validation

- Unit tests in `sampling.rs` (host-reference agreement over 40 random rows, argmax survival, peaked row, temperature monotonicity, row independence, `-inf` padding, greedy skip, fused params equality, `batched_fused_sample` with 128 rows) and in the request, execution, wire and speculative-window test modules.
- Real checkpoint validation on Qwen3-4B-4bit is recorded in the PR description.

## 5. Not Done

No server-wide default, and no echo of `p_less` in `/props` or the native `generation_settings`, since neither is required by the issue.
