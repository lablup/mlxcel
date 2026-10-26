# PR #2646: Adopt native window SDPA for Metal prefill

**Date**: 2026-10-27
**Status**: Completed
**Languages**: Rust, C++
**Risk Level**: Medium

## Executive Summary

A causal multi-query prefill whose keys outgrow a sliding window now runs through upstream MLX's native `window_size` argument on Metal when MLX reports a fused kernel for the shape, instead of materializing a full-width windowed mask (issue #2621). The route is on by default (`MLXCEL_NATIVE_WINDOW_SDPA=auto`). The decision rests on a criterion fixed before timing that one shape cleared at exactly 3.0%. bf16 Gemma 4 greedy text changes at near-tie positions, and `MLXCEL_NATIVE_WINDOW_SDPA=0` restores the previous route.

## 1. Problem Statement

The MLX pin bump in #2614 brought in ml-explore/mlx#4734, which adds window-aware Metal vector and matrix kernels, but kept every mlxcel call site at `window_size = -1`. For an over-window prefill, `causal_attention` built a `[q_len, k_len]` mask that encodes both causality and the window band, and passed it as an array mask. That has two costs. First, the mask itself: 5376 MiB of call-peak memory at the Gemma 4 31B geometry with 8192 rows. Second, on head dim 256 (every cached sliding-window checkpoint), MLX never fuses an array-mask call on pre-M5 GPUs, so the whole prefill ran the graph fallback.

The prefill semantics are subtle and had regressed before. A multi-query over-window prefill must keep every key, because truncating to the last window strands early queries (#401/#408). A rotating-cache continuation returns retained prior keys plus every new key, so query rows align to the suffix of visible keys, not to the absolute offset (#678).

## 2. Technical Decisions

**One route decision, explicit metadata.** `mlxcel_core::window_sdpa` owns the decision. Its input is a `WindowSdpaShape` (shapes and dtypes of q/k/v) plus `WindowSdpaExtras` (softcap, sinks, extra mask). It never infers meaning from a mask's shape. The native entry is reachable only from `causal_attention`'s over-window branch with softcap 0, which has no mask parameter and no sinks. Everything else (padding, segment and per-head masks, `q_len <= 8`, `q_len > k_len`, empty batches, unsupported head dims or dtypes, every non-Metal backend) keeps the legacy route. This matters because positive windows force a graph fallback on CUDA and ROCm.

**Ask MLX instead of copying its rules.** `auto` takes the native entry only when the pinned MLX's own `ScaledDotProductAttention::use_fallback` says a fused kernel runs. `sdpa_uses_fused_kernel` in `mlx_cxx_window_sdpa.cpp` gets that answer from lazy zero arrays of the call's shape. Hard-coding the dispatch table would drift with the next pin bump. The measurement showed why the gate is needed: graph-fallback shapes (head dim 256 continuation chunks) are neutral to 4% slower on the native arm. `1` forces native wherever the route is statically eligible, and `0` is the rollback.

**Plan at the model, confirm at the layer.** Gemma 3 (model, stage model, VLM backbone) and Gemma 4 text-only prefill decide once per forward whether the native route will run. When it will, they skip building the sliding mask. A layer that declines rebuilds the identical mask, so a disagreement costs time, never semantics. Review found such a disagreement. Under `fp16+turbo4` the cache returns f16 K/V. A bf16 query promotes the call to f32, where head dim 256 is never fused, so every sliding layer declined and rebuilt the mask. The plan now uses `RotatingKVCache::fetched_kv_dtype`, and goes native only when every sliding cache agrees (`sliding_caches_kv_dtype`), because `--turbo-boundary-v` mixes cache modes per layer.

**Fix the unmasked rotating branch instead of preserving it.** `RotatingKVCache::attend` ran an unmasked over-window multi-token step as plain unmasked attention, with no causality. Only Gemma 3 reaches that entry, and every Gemma 3 path built a mask in that case, so the branch was dead. It now calls `causal_attention(..., max_size)`, and Gemma 3 relies on it when the plan is native. Continuous-batching rows always arrive with `q_len == 1`, and speculative verify blocks are declined by `q_len <= 8`, so neither reaches the new branch.

**Make the probe cheap and non-fatal.** The probe is memoized per thread on the full shape, dtypes and arm. The memo is bounded at 64 entries and cleared when full, because `k_len` changes every chunk in a long-running server. cxx declares the bridge function infallible, so a C++ exception would terminate the process. The probe therefore returns false for non-positive dimensions and catches `std::exception`. Returning false means the legacy route runs.

## 3. Change Summary

- `src/lib/mlxcel-core/src/window_sdpa.rs` (new): mode parsing (read once, one warning for unknown values), eligibility, plan, fused-verdict memo, dispatch counters, `MLXCEL_LOG_WINDOW_SDPA` evidence, and the native and legacy arms.
- `src/lib/mlxcel-core/cpp/mlx_cxx_window_sdpa.cpp` (new) plus `mlx_cxx_bridge.h`, `lib.rs` cxx block and `build.rs`: `fast_scaled_dot_product_attention_causal_window` (upstream `sdpa(..., W, "causal")`) and `sdpa_uses_fused_kernel`.
- `lib.rs::causal_attention`: the over-window branch with softcap 0 goes through `window_sdpa::causal_window_attention`. The softcap composite is unchanged.
- `cache/attend.rs`, `cache.rs`: the rotating unmasked branch described above, and `fetched_kv_dtype`.
- `src/models/gemma3.rs`, `gemma3_backbone.rs`, `gemma4.rs`: native-plan mask skipping.
- Tests: `window_sdpa_tests.rs` (19).
- Bench and docs: `examples/window_sdpa_microbench.rs`; `docs/environment-variables.md`; report and raw CSV and trace artifacts under `docs/benchmark_results/native-window-sdpa-m1ultra-2026-10-26*`.

## Validation

M1 Ultra, macOS 27.0.1, MLX `7ca40033`.

**Correctness.** Tolerances were fixed before running. The native arm matches both the legacy arm and an f64 host reference over 8 shape cases at f32, f16 and bf16, MHA and GQA. Those cases include W=1, continuation chunks, query blocks longer than the window, and the Gemma D=256 MQA geometry. Out-of-window and future keys are bitwise inert. The rotating cache is checked through wrap, trim and 20 maskless decode steps.

**Logit trace.** Teacher-forced traces at the width the route runs at (`--decided 2`) show 0 disagreements on decided positions for Gemma 3 1B (f16) and Gemma 4 E4B (bf16), and the reverted arm is byte-identical to itself.

**Full gate.** `cargo test --workspace --profile test-fast --no-fail-fast` on the final head passed 13896 tests, 0 failed. Workspace clippy and fmt are clean.

**Performance.** At the kernel level, the shapes `auto` selects run in 0.50x to 0.11x of the legacy time at W=512, and call-peak memory drops 3.5x to 8x. At the full-model level:
- Gemma 4 E4B prefill is 3.0% faster at 2105 tokens (n=6, non-overlapping ranges) and within noise at longer prompts.
- The 31B points the same way at n=2.
- Gemma 3 1B is inside its noise band.
- Decode and process RSS are unchanged.

The gain at full-model level is small even though the kernel speedup is large. Two reasons: the route engages only on the first 2048-row prefill chunk, and attention is a minority of prefill time.

## Unverified and Follow-up

- **M5/NAX.** Not run. The pinned dispatch fuses head dim 256 at every width there, so `auto` engages on more shapes and needs its own measurement. On M5 the native entry bypasses `layers::attention`, so `MLXCEL_METAL4_ATTENTION` and `MLXCEL_LOG_NA_ATTENTION` do not cover it. `MLXCEL_NATIVE_WINDOW_SDPA=0` does.
- **Head-dim 128 sliding-window families (Ministral 3).** This is where the microbench predicts the large model-level gains. No checkpoint was available, and Ministral 3 still builds its own prefill mask, so it reaches the route only when that mask is absent.
- **TurboQuant cache modes.** Plan and layer now agree, but neither arm was timed.
- **Default-on rests on one shape.** The decision depends on a single shape clearing the 3.0% bar. Re-measure if the prefill chunk size or the MLX pin changes; raising `DEFAULT_PREFILL_CHUNK` would widen the engaged region at the cost of the chunking's memory purpose (#672).
- **Flaky paged-counter test.** `attend_tests::batched_paged_one_row_is_one_pooled_launch` failed once in a full parallel run. It reads a process-global paged launch counter, and the code it tests is untouched by this PR. It passed on every rerun, but the race is real and worth isolating.
