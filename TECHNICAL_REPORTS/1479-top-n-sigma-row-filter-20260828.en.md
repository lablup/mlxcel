# Technical Report: PR #1479 - Top-n-sigma logit filtering with a row-filter hook

**Date**: 2026-08-28

**Status**: Completed

**Languages**: Rust

**Risk Level**: Medium

## Executive Summary

Adds the top-n-sigma sampler (`SamplingConfig::top_n_sigma`, default `0.0` disabled) and the batch-safe row-filter hook `apply_row_filters` that later sampler stages (`typical_p` #1377, `p_less` #1373) plug into. The filter keeps only tokens whose raw logit lies within `n` standard deviations of the row maximum, which makes sampling stable at high temperature while remaining invariant to the temperature value itself. The disabled baseline is bit-exact: no graph nodes are added when the filter is off or the config is greedy.

## 1. Problem Statement

mlxcel had no top-n-sigma sampler and no shared, RNG-free, history-free hook where row-wise logit filters could run without disqualifying a request from the single-dispatch fused batch path. llama-server b10621 exposes `top_n_sigma` on its request schema and `--top-nsigma` on its CLI, so the epic #1431 compatibility work needs the semantics in the engine before the native spellings can be claimed in the manifest (#1436).

## 2. Technical Decisions

### 2.1 Rust pre-filter, not a C++ chain stage

The fused sampler chain (temperature, top_k, top_p, min_p, XTC) lives in C++ behind `ffi::fused_sample`. Top-n-sigma is instead a Rust-side graph transformation applied to the logits before that single dispatch. It draws no randomness and needs no per-step state, so MLX's lazy graph fuses it into the same evaluation; no bridge signature change, no MLX pin change, and the batched `[B, V]` path stays one dispatch.

### 2.2 Filter statistics in float32 over finite entries only

An f16 sum over a 150k-entry vocabulary overflows to `inf`, which would silently disable the filter, so all reductions run in f32 regardless of the logit dtype. Entries already masked to `-inf` by token bias or penalties are excluded from mean, std, and max, and stay masked in the output. A review pass hardened two edges: the row maximum is taken over the finite-masked row so one NaN entry cannot collapse the whole row to `-inf` (MLX's Max reducer propagates NaN), and the final mask fill carries the original logits dtype because MLX's `where` promotion was otherwise returning f32 for f16/bf16 inputs.

### 2.3 Effective-value normalization for batch gates

`FusedSampleParams::from_config` and the speculative-window `sampling_config_eq` compare `SamplingConfig::effective_top_n_sigma()`, which folds greedy, non-positive, and non-finite values to `0.0`. Without this, two greedy requests differing only in an inert `top_n_sigma` would produce byte-identical tokens yet fail bitwise `matches`, dropping the whole co-resident batch to the per-row loop and splitting speculative windows for no observable difference.

### 2.4 Native `/completion` deferred to #1436

`src/server/llama_compat_tests.rs` requires a `native_request_field` manifest entry to flip in the same change that declares the field on `NativeCompletionRequest`, and the `field:top_n_sigma` entry is owned by #1436 under the epic's shard-ownership rule. The field therefore lands together with its manifest flip in #1436. `apply_row_filters` already treats the upstream `-1.0` disabled sentinel as inert in preparation.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 27 (+105 in the review follow-up) |
| Lines | +642 / -13 across both commits |
| Tests added | 17 (core filter, gates, request layer, wire protocol, burst window) |

Touchpoints: `mlxcel-core` (`generate.rs`, `sampling.rs`), scheduler pipelined lookahead, speculative burst gate, request plumbing on three OpenAI-shaped endpoints, disaggregated wire struct (`#[serde(default)]` back-compat), CLI flag `--top-n-sigma` (alias `--top-nsigma`), `docs/python-client.md`.

Validation: 77 core sampling tests plus root-crate filter suites green; clippy `-D warnings` and `cargo fmt --check` green; real-checkpoint check on Qwen3-4B-4bit shows fluent output at temperature 2.0 with the filter versus degeneration without it, and token-identical greedy output with and without the flag.

## 4. Follow-up Actions

- #1377 adds `typical_p` on the hook; #1373 adds `p_less`; the ordering slots are pinned in `apply_row_filters`.
- #1436 adds the native `/completion` field, the `--top-nsigma` server flag surface, and flips the two manifest entries.
- Drafter-internal greedy `fused_sample` call sites (DFlash round loop, MTP heads) bypass all samplers by design; acceptance-rate impact only, tracked as a known pre-existing gap.
