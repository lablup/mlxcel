# Technical Report: PR #3056 - update(models): use MLX causal mode for purely causal hybrid prefill

**Date**: 2026-11-15
**Reviewer**: -
**Status**: Verified on M3 Ultra and M5 Max
**Languages**: Rust, Bash, Markdown
**Risk Level**: Medium

---

## Executive Summary

The full-attention layers of Qwen 3.5, Qwen3-Next, GraniteMoeHybrid and the Qwen-VL models built an f32 causal array mask for every prefill. Before M5, MLX does not fuse SDPA with that mask at head_dim 256, so a single-pass prefill took #3049's query-chunked fallback, and the 3.65 GiB mask stayed alive at 28K tokens (#3051). The models now leave causality to MLX's causal mode when it masks the same keys and MLX's own verdict says it fuses at the call's shape. Elsewhere they keep the shared array mask. On an M3 Ultra a 28K single pass drops from 30.13 to 19.82 GB (4B) and from 64.20 to 53.96 GB (27B), and runs 21% and 15% faster. Chunked CLI and server prefill are unchanged, and greedy output is byte-identical. On an M5 Max, against a main build, the single pass drops from 14.53 to 8.76 GB (4B) and from 31.83 to 26.03 GB (27B) and runs 21% and 23% faster; the later 2048-token pieces take causal mode too, so the 28K chunked peak falls 6.0% and 1.7%. Output is byte-identical there as well.

---

## 1. Problem Statement

### 1.1 Background

MLX's SDPA has three arms: an additive array mask, causal mode and unmasked. Each arm is either fused (memory-linear) or falls back to a materialized `[B, H, L, S]` score matrix. #3033 found that before M5, at head_dim 256, the array-mask arm always falls back, while causal mode fuses when `q_len == k_len`. #3049 bounded that fallback by query-chunking it, but it is still slower and still needs the mask.

### 1.2 Existing Issues

- The hybrid and Qwen-VL families called `create_causal_mask(seq_len, offset)` for every multi-token forward and passed it to every full-attention layer.
- At 28K tokens that array is 3.65 GiB in f32 and lives for the whole single pass. Each full-attention layer then takes the chunked fallback (8 gated calls on the 4B, 16 on the 27B).

---

## 2. Change Summary

| Area | Change |
|---|---|
| `mlxcel-core` `utils.rs` | `prefill_causal_mask(seq_len, offset, batch, PrefillAttention, activation_dtype)` and `prefill_uses_causal_mode`, with a pure `causal_mode_applies` for tests. The kill switch is `MLXCEL_PREFILL_CAUSAL_MODE`. |
| Models | Qwen 3.5 (four prefill entry points and the pipeline stage), Qwen3-Next (sequence forward and stage), GraniteMoeHybrid, Qwen2-VL, Qwen3-VL and Qwen3-VL-MoE pass their full-attention geometry and the hidden dtype. The VL attention `forward` routes a mask-less multi-token call through `causal_attention`. |
| Bench harness | `ab_prefill.sh` adds per-arm `BASE_ENV`/`NEW_ENV`, a `gated_calls` column, and `SERVER_MODELS=""` support. |
| Docs | Env var row, CHANGELOG, and a benchmark doc with four CSVs (results and two rejected variants). |

---

## 3. Technical Decisions

### 3.1 When causal mode is equivalent

MLX's causal mode lets query `i` see keys `0..=k_len - q_len + i`. `create_causal_mask(L, offset)` is `[L, offset + L]`, so every forward that ran with it had `k_len == offset + L`, and the two allow the same keys at any offset. They differ only where a model's mask carries more than causality, and those cases keep the array:

- left padding or batched rows (`batch > 1`);
- caller-supplied masks (padding, windows, multimodal);
- a single-row decode, which needs no mask at all.

Position ids (MRoPE) change only the rotation, not the attended keys.

### 3.2 Speculative verify keeps the array

The generic `Engine::verify` runs a `[1, k+1]` block with no caller mask, so it qualified as well. A review found that verify blocks would then switch kernels, which could flip near-tie argmaxes against earlier verify results. The decision now also requires that no `DecodeLookaheadAppendScope` is active.

### 3.3 Ask MLX before choosing, and keep the shared mask otherwise

The first version used causal mode wherever it was equivalent. On pieces where MLX falls back for causal mode too (pre-M5, head_dim 256, `q_len != k_len`: every piece after the first), `causal_attention` then ran its own chunked causal fallback. That fallback slices K/V and builds a mask per query chunk in every layer. It raised the 28K, 2048-chunk peak by 2.6% (4B) and 4.0% (27B), the server's default path.

A second version, in which the fallback built `create_causal_mask` itself for the array-mask chunker, was worse: +11.4% on the 27B, with one `[2048, 30K]` f32 mask per layer instead of one per forward.

The shipped version asks `sdpa_mode_is_fused` (MLX's `use_fallback`, memoized per shape) with the layer geometry and the hidden dtype. It uses causal mode only where that verdict is fused, and otherwise keeps the one shared mask. Chunked peaks then match base to within 1 MB.

On non-Metal builds the probe reports no fused kernel, so CUDA and CPU keep the array mask and their behaviour does not change.

---

## 4. Verification

| 28K tokens, M3 Ultra | qwen3.5-4b-4bit | qwen3.8-27b-4bit |
|---|---|---|
| CLI single pass | 30.13 to 19.82 GB, 16.89 to 13.38 s | 64.20 to 53.96 GB, 86.09 to 73.33 s |
| CLI 2048 chunks | 8.97 GB, 13.87 to 13.85 s | 25.80 GB, 74.77 to 74.74 s |
| server `--parallel 1` | 9.10 GB, TTFT 13.90 to 13.89 s | 25.92 GB, TTFT 74.88 to 74.81 s |

- Both arms use one binary, `MLXCEL_PREFILL_CAUSAL_MODE=0` against the default, interleaved in one lock hold.
- Repetitions agree within 57 ms.
- Output is identical in every cell and for an image prompt.
- At 4.6K with 2048-token chunks, gated calls fall from 24 to 16 (4B) and from 48 to 32 (27B), with peaks unchanged.

Unit tests:

- They check causal mode numerically against the array mask at offset 0, at offset > 0, and in bf16 at head_dim 256.
- They also cover the padded, batched, decode, verify and unfused-shape branches.
- 96 model, 610 scheduler and 328 prompt-cache tests pass on Metal.

---

## 5. Remaining Work

- On M5 the short last piece of a chunked prefill (452 and 494 rows at 4.6K) still falls back: MLX reports causal mode unfused at that shape, so it keeps the shared array mask, as main does.
- Before M5, a chunked prefill's later pieces still take MLX's fallback; no mask choice avoids it there.
- Other hybrid families still build the array (`nemotron_h`, `falcon_h1`, `jamba`, `plamo2`, `lfm2`, `bailing_moe_linear`), as does the tensor-parallel Qwen 3.5 runtime. They are candidates for a grouped follow-up once each layer is confirmed to route a mask-less multi-token call through `causal_attention`.
