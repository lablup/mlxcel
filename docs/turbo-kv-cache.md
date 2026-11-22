# TurboQuant KV cache

TurboQuant modes reduce KV-cache memory by quantizing K and/or V cache tensors.
The implementation is experimental in the sense that quality and speed vary by
model family, cache mode, hardware, and server path. Use the default FP16 cache
unless you have measured the target model and workload.

The TurboQuant algorithms (PolarQuant rotation, Lloyd-Max codebooks, layer-aware V protection, sparse-V dequant) are a Rust port of [turboquant_plus](https://github.com/TheTom/turboquant_plus), Copyright 2026 Tom Turney, licensed under the Apache License 2.0. See the top-level [NOTICE](../NOTICE) file for the attribution carried forward under Apache-2.0 Section 4(d).

Implementation entry points:

- CLI flags: `src/cli/turbo_args.rs`
- cache modes: `src/lib/mlxcel-core/src/cache.rs`
- TurboQuant helpers: `src/lib/mlxcel-core/src/cache/turbo/`
- quality/behavior tests: `tests/turbo_kv_e2e.rs` and
  `src/lib/mlxcel-core/src/cache/*turbo*_tests.rs`

## Available modes

| User-facing mode | Effective cache mode | Notes |
|------------------|----------------------|-------|
| `fp16` | `KVCacheMode::Fp16` | Default and baseline. |
| `int8` | `KVCacheMode::Int8` | Per-token INT8 absmax quantization. On DeepSeek V2 under absorbed MLA decode, on the `DeepSeekV3Attention` families (DeepSeek V3, Kimi K2, Kimi-VL, Youtu) on the MLA layers of Kimi Linear and Kimi K3, and on both MLA sub-layers of LongCat, it stores the MLA latent as affine group-64 8-bit instead (see [MLA latent caches](#mla-latent-caches)). |
| `fp16+turbo4` / `turbo4-asym` | `KVCacheMode::Turbo4Asym` | FP16 K, 4-bit TurboQuant V. Safest TurboQuant starting point. |
| `turbo4` / `turbo4-sym` | `KVCacheMode::Turbo4` | 4-bit TurboQuant K and V. K-side quantization is quality-sensitive; use only on validated/allowlisted families. |
| `turbo4-delegated` | `KVCacheMode::Turbo4Delegated` | Hot/cold split on the V side with packed cold storage. Validate speed on target hardware. |
| `fp16+turbo3` / `turbo3-asym` / `turbo3` | `KVCacheMode::Turbo3Asym` | FP16 K, 3-bit TurboQuant V. More aggressive than Turbo4 V. |

Symmetric Turbo3 is not exposed.

## Measured speed and the memory trade-off

Quantized KV cache is a memory-footprint optimization, not a decode-speed one.
On the post-#369 M1 Ultra sweep (`benchmarks/turbo_kv/`, 2026-06-19) every
quantized mode decodes slower than fp16. The dequant work (Walsh-Hadamard
inverse rotation plus a codebook lookup for the Turbo modes, a scale-and-add for
int8) runs per token and, at the context lengths tested, costs more than the
memory bandwidth it saves. On dense models the gap is wider because FFN
dominates decode and the V read that compression shrinks is a small fraction of
the work.

Decode is reported as a fraction of the same model's fp16 throughput at that
context; prefill at 8K. Two representative checkpoints below; the full
four-model sweep (adding qwen2.5-7b and the 142B dots.llm1 MoE) is in
`benchmarks/turbo_kv/`.

| Mode | dense decode 4K / 16K | dense prefill 8K | MoE decode 4K / 16K | MoE prefill 8K | KV compression |
|------|-----------------------|------------------|---------------------|----------------|----------------|
| `fp16` | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 1.00 | 1x (baseline) |
| `int8` | 0.61 / 0.41 | 1.29 | 0.48 / 0.50 | 1.14 | ~2x |
| `turbo4-delegated` | 0.63 / 0.45 | 1.27 | 0.43 / 0.34 | 1.13 | ~4x V |
| `turbo4-asym` | 0.38 / 0.37 | 0.92 | 0.24 / 0.26 | 0.80 | ~3.8x V |
| `turbo4` (sym) | 0.22 / 0.20 | 0.73 | 0.17 / 0.17 | 0.62 | ~4x K+V |
| `turbo3-asym` | 0.04 / 0.02 | 0.91 | 0.05 / 0.03 | 0.77 | ~5x V |

Dense = `qwen3-8b-4bit`, MoE = `qwen3-30b-a3b-4bit`, M1 Ultra.

What the numbers say:

- **`int8` and `turbo4-delegated` are the fast picks.** int8 (2x) is robust on
  both dense and MoE; turbo4-delegated reaches ~4x V compression at a similar
  decode speed by keeping recent V in fp16 and only the cold tail in 4-bit. Both
  also speed up prefill (less KV written), 1.1-1.3x.
- **`turbo4-asym` is the memory-and-exactness pick, not a speed pick.** K stays
  fp16 and the #369 dequant-SDPA path is parity-exact with the fp16 reference,
  but decode is ~0.2-0.4x. Reach for it when you need ~4x V compression with an
  untouched K and accept the decode cost. If you want speed at the same fp16-K +
  4-bit-V trade, use `turbo4-delegated` (~0.7x) instead. #370 tried fusing the V
  dequant into the attention kernel to close the gap and measured it a 3-7x
  regression, not a win; see the 2026-06-21 addendum in ADR 0002.
- **Symmetric `turbo4` maximizes compression and is the slowest;** use only on
  an allowlisted family (see below).
- **`turbo3-asym` is near-unusable** (0.02-0.07x, about 0.4 tok/s at 32K). It
  exists for memory-extremis only and is not a recommended mode.

The slowdown is consistent with the upstream TurboQuant+ analysis: its headline
`+22.8%` decode-at-32K figure is measured against turbo3's own full-dequant
path, not against fp16, and even with that win turbo3 stays at 0.93x of int8.
The compression is a 4.6x memory trade; the decode cost is inherent to
dequantizing a rotated, codebook-quantized cache every step. The upstream
sparse-V skip that produces the +22.8% does not carry over to mlxcel, because
mlxcel's Turbo decode is a split dequant plus native SDPA rather than a fused
flash-attention; [ADR 0002](adr/0002-turbo-kv-split-dequant-vs-fused.md) records
the measured A/B. Its 2026-06-21 addendum closes #370: routing asym through
mlxcel's fused kernel regressed decode 3-7x, because the hand-written fused
kernel is slower than native SDPA over a materialized V, so the split dequant
plus native SDPA stays the fast arrangement.

### On CUDA (GB10)

The same trade holds on CUDA, and the numbers were measured on GB10 (sm_121) with `llama-3.1-8b-4bit` (2026-07-10, `benchmarks/cuda_gb10_issue635_kvquant_2026-07-10.csv`; full matrix in `benchmark_results/cuda-kv-quant-modes-gb10-2026-07-10.md`). All six modes run without a Metal-only abort: every custom Turbo/Sparse-V kernel is gated to macOS and falls back to a plain-MLX graph path on CUDA.

Int8 decode vs fp16 is 0.74x at 2K, 0.53x at 8K, and 0.31x at 32K, so it is slower at every length and the gap widens with context, because the path dequantizes the whole INT8 window to fp16 each step and then runs the standard fp16 SDPA (no fused INT8-KV kernel cuts the read bandwidth). What Int8 buys is memory: at 32K the MLX peak drops from 10.37 GB to 8.18 GB, matching a halved fp16 KV cache. Recommendation for CUDA: keep fp16 (default) for speed, and reach for `int8` only when an fp16 KV cache will not fit a long context or you need to pack more concurrent sequences into unified memory, accepting the decode-rate cost. The Turbo modes add V-quantization quality loss with no CUDA speed benefit today, so they stay experimental on CUDA. A fused INT8-KV / paged-attention kernel (#634) is the prerequisite for turning the smaller footprint into a decode win.

## CLI and server flags

The same TurboQuant flag group is flattened into `mlxcel generate`,
`mlxcel serve`, and `mlxcel-server`:

```bash
# Legacy shorthand.
mlxcel generate -m models/<checkpoint> \
    --kv-cache-mode fp16+turbo4 \
    -p "Hello" -n 100

# llama-server-style split flags.
mlxcel-server -m models/<checkpoint> \
    --cache-type-k fp16 \
    --cache-type-v turbo4 \
    --port 8080
```

Supported split-flag combinations:

| `--cache-type-k` | `--cache-type-v` | Result |
|------------------|------------------|--------|
| `fp16` or `f16` | `fp16` or `f16` | `fp16` |
| `int8` | `int8` | `int8` |
| `fp16` | `turbo4` or `turbo4-asym` | `fp16+turbo4` |
| `turbo4` | `turbo4` | `turbo4` |
| `fp16` | `turbo4-delegated` | `turbo4-delegated` |
| `fp16` | `turbo3` or `turbo3-asym` | `fp16+turbo3` |

If split flags and `--kv-cache-mode` are both supplied, split flags take
precedence and a warning is logged.

The exact llama.cpp `f16` spelling aliases FP16 storage. GGML quantizer names such as `q8_0` and `q4_0` remain errors because mlxcel has no mathematically equivalent cache format for them.

Environment variables:

| Variable | Purpose |
|----------|---------|
| `LLAMA_ARG_CACHE_TYPE_K` | Env fallback for `--cache-type-k`. |
| `LLAMA_ARG_CACHE_TYPE_V` | Env fallback for `--cache-type-v`. |
| `MLXCEL_KV_BOUNDARY_V_LAYERS` | Boundary-V layer count when the CLI flag is not set. |
| `MLXCEL_SPARSE_V_THRESHOLD` | Sparse-V threshold; `0` disables sparse-V behavior. |
| `MLXCEL_SPARSE_V_KERNEL` | Set to a falsy value to disable custom sparse/dequant Metal kernels where they are used. |
| `MLXCEL_TURBO4_ASYM_DEQUANT_SDPA` | Falsy value disables the default dequant-first SDPA path for `Turbo4Asym`, falling back to the sparse-V approximation. |
| `MLXCEL_TURBO4_DELEGATED_DEQUANT_SDPA` | Falsy value disables the default dequant-first SDPA path for delegated mode. |
| `MLXCEL_TURBO4_DELEGATED_FUSED` | Truthy value opts into an older fused delegated-kernel route for comparison. |
| `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH` | Truthy value keeps an FP16 V working set for delegated-mode speed experiments. |
| `MLXCEL_TURBO4_DELEGATED_FP16_SIDECARS` | Sidecar policy for the FP16 fast path. |

Most users should not need the experimental environment variables. Prefer
published CLI flags unless you are benchmarking implementation variants. For
the broader runtime and diagnostic environment-variable reference, see
[Environment variables](environment-variables.md).

## Boundary-V layer protection

`--turbo-boundary-v N` keeps the first `N` and last `N` transformer layers' V
cache at FP16 when a Turbo mode is active. This is intended to reduce quality
loss from aggressive V quantization.

```bash
mlxcel generate -m models/<checkpoint> \
    --kv-cache-mode fp16+turbo4 \
    --turbo-boundary-v 2 \
    -p "Hello" -n 100
```

`0` disables the policy. Values larger than half the layer count are clamped by
the runtime. The flag is inert for `fp16` and `int8` cache modes.

Under continuous batching the attention route follows the storage behind each
layer's cache ([ADR 0008](adr/0008-kv-attention-dispatch-in-the-cache.md)).
Turbo rows of a batched decode step take the same dequant-first variants as
single-sequence decode, and the layers Boundary-V (or `--kv-skip-last-layer`)
keeps at FP16 under a non-FP16 paged layout hold dense caches that run a
per-row dense update plus SDPA. Both are the same exact attention in a different
op order, so they are not bit-identical to the pre-#2171 batched routes.

## Symmetric Turbo4 allowlist

K-side quantization can strongly affect softmax quality. The code therefore has
an allowlist helper in `src/lib/mlxcel-core/src/cache/turbo/allowlist.rs`, which
is the source of truth for this list. The hard-coded allowlisted model-type
prefixes are:

- `qwen3_5`
- `qwen3_5_moe`
- `qwen3_next`

Do not assume `turbo4` is safe for other dense 4-bit checkpoints. The safer
starting point is `fp16+turbo4`, which keeps K in FP16.

The fallback is performed, not merely advertised. Every generation entry point
resolves the requested mode through
`cache::turbo::resolve_kv_cache_mode_for_model` before any cache is built, so
`--kv-cache-mode turbo4` on a family outside this list serves as
`fp16+turbo4` and says so on stderr and in the server log. The mode reported in
the CLI banner, the server startup log and the `/props` payload
(`kv_cache_mode`, `kv_bits`, when the endpoint is enabled with `--props`) is the
effective mode, not the requested one. If you are adding a new entry, include a
quality gate in the same change.

## Model-owned cache families

Some families do not use the homogeneous `&mut [KVCache]` slice handed to `forward`; they keep heterogeneous attention, sliding-window, recurrent, or convolutional state inside the model wrapper. The CLI generator and server scheduler resolve the effective per-layer KV-mode table once (including Boundary-V and `--kv-bits` / batch-KV policy) and inject that table into model-owned attention-cache constructors too. The CLI banner and server startup log report the effective mode and `applied to N of M layers`, and `GET /v1/cache/stats` includes `kv_cache_mode_effective`.

Attention KV caches owned by Gemma 3, AFMoE, Gemma 4 text/VLM/Unified, Qwen 3.5 text/MoE/VLM, Qwen3-Next, Bailing MoE Linear, LFM2/LFM2-VL, and the regular-attention arms of Llama 4 use the injected mode for their layer index and fall back to FP16 only when no table is available. Recurrent, convolutional, and gated-delta state is not a KV tensor and remains unchanged. For families that donate full model-state snapshots, non-FP16 attention KV modes are deliberately fail-closed for snapshot serialization today: the model logs a warning that names the mode (for example `cache mode Turbo4Asym is not supported by model-state snapshots`) and skips donating that snapshot rather than storing a partial or mislabeled state. An empty quantized layer, such as a KV-shared layer that never populates its own cache, is skipped without an error. The mode is still applied to live attention layers and reported as the effective mode.

Llama 4 `ChunkedKVCache` layers remain FP16. They do not yet have quantized sidecar storage, and the wrapper warns once when a non-FP16 table would otherwise apply to those chunked layers. Muse Glimmer still rejects non-FP16 modes at startup rather than silently running a partial mode.

## WHT head-dimension constraint

TurboQuant uses a Walsh-Hadamard transform. The implementation expects a
power-of-two head dimension for the production path. Models with unsupported
head dimensions must either reject TurboQuant for that cache path or use a
family-specific fallback; do not silently pad without a quality test.

## WHT scaling, and which scale a quantization target wants

`mlxcel_core::wht` leaves the scale to MLX, which applies `1/sqrt(N)`. That makes
the transform orthonormal and an involution, so `wht(wht(x))` returns `x` and a
codebook fitted on the rotated values sees the same magnitudes the unrotated
ones had. Every current caller wants exactly this, and a backend port that
hardcodes a different scale is wrong: measure the forward norm ratio, which is 1
for `1/sqrt(N)` and `sqrt(N)` for an unscaled matrix, rather than the round trip,
which cannot tell the two apart. `examples/wht_numeric_probe` reports both.

`mlxcel_core::wht_scaled` exists for the case where the orthonormal factor cannot
be carried for free, which is rotation feeding a block-scaled low-precision
format. What the format's scale can represent decides where the factor belongs:

| Target | Scale storage | Where `1/sqrt(N)` goes |
|---|---|---|
| `mxfp8`, `mxfp4` | `E8M0` block scale, powers of two | Folds exactly only when `log2(N)` is even: head_dim 64 (`1/8`) and 256 (`1/16`) yes, 128 (`2^-3.5`) no |
| `nvfp4` | `FP8` block scale plus an `FP32` per-tensor global scale | Absorbed by the global scale, at the cost of range |
| `fp8` `E4M3` | Saturates at 448 | Scale and saturation check are one decision: an unscaled transform grows magnitudes by up to `sqrt(N)`, 11.3x at head_dim 128 |

The failure this table prevents is the one lablup/mlxcel#1769 and the pin bump in
lablup/mlxcel#1772 already cost once: `E8M0` rounding saturated block maxima by up
to 29 percent, and nothing downstream said so.

## MLA latent caches

An absorbed MLA cache does not hold per-head K/V rows. It holds the shared latent `ckv` (`kv_lora_rank` wide, 512 in the shipping checkpoints) and the RoPE key stream `k_pe` (`qk_rope_head_dim` wide, 64). Which KV modes such a cache accepts depends on how the family reads it.

### DeepSeek V2: fp16 or an affine int8 latent

`deepseek_v2` (and the `deepseek_v2` decoder inside DeepSeek-VL2) reads its cache through `mla::MlaLatentCache`. DeepSeek-VL2 keeps that decoder under `language_config`, so the mode resolver, the memory estimate and the advisor read it as `deepseek_v2` and every rule in this section applies to it unchanged. Under absorbed decode, which is the default on Metal, CUDA and ROCm, that cache holds:

| Mode | Latent `ckv` | `k_pe` | Bytes per token per layer (V2-Lite) |
|------|--------------|--------|-------------------------------------|
| `fp16` (default) | fp16 | fp16 | 1152 |
| `int8` | MLX affine group-64 8-bit, fp16 scale and bias per group | fp16 | 672 (512 + 32 + 128) |

`--kv-cache-mode int8`, `--cache-type-k int8 --cache-type-v int8` and `--kv-bits 8` (uniform) all select the int8 latent. This is a different codec from per-token int8: one absmax scale for a 512-wide row that every query head reads is too coarse, and it would force a full-window dequantize on every decode step. The affine triple is what `quantized_matmul` consumes, so a decode step computes both the latent score and the latent-value sum on the packed rows without dequantizing the window. `k_pe` stays fp16 for RoPE precision. A multi-token step and a cold prefill dequantize the window once per step and run the same code as the fp16 latent, after the fresh rows were stored quantized, so prefill and decode read the same values. The group size is fixed at 64; `--kv-group-size` with another value fails server startup for such a model. `--kv-skip-last-layer` (the `--kv-bits` default) keeps the last layer as an fp16 latent.

On Metal an unmasked decode step over a window of up to 2048 keys runs the fused kernel's quantized variant (`absorbed_fused` in `MLXCEL_TRACE_MLA_STATS=1`, #3156); longer windows and the other backends run the composed path (`absorbed_composed`). The figures that follow were measured on the composed path, before #3156; the fused variant's own rows are in [mla-fused-quantized-decode-m3ultra-2026-11-19](benchmark_results/mla-fused-quantized-decode-m3ultra-2026-11-19.md). Measured on an M3 Ultra with DeepSeek-V2-Lite ([results](benchmark_results/mla-quantized-latent-m3ultra-2026-11-19.md)): the teacher-forced trace against the fp16 latent shows no disagreement on decided positions behind 1024 and 4096 tokens, and decode is 13 to 15% slower than the fp16 latent behind 512 and 2048 tokens and 36% slower behind 16384, while staying 1.15x to 2.8x faster than the decompressed per-head int8 cache it replaces.

A 4-bit affine latent (416 bytes per token per layer) was implemented and measured in the same change and is not offered by default. It also showed no decided-position disagreement, but it flipped the top-1 token at 8.8% and 5.5% of positions with reference top-two gaps up to 0.86 logits, against the gate of 0.11, so `int4` is not a KV cache mode and `--kv-bits 4` with the uniform scheme stays a startup error. It is available only through the explicit, experimental `--mla-latent-quant int4` ([below](#experimental---mla-latent-quant), #3179).

The Turbo modes do not fit the latent, for the reasons listed under the latent-pair families below, and a Turbo request on an absorbing `deepseek_v2` would decline the latent and store the decompressed per-head cache, which is larger than the fp16 latent (about 7.2 KB per token per layer for turbo4-asym on V2-Lite against 1152). The resolver therefore substitutes `fp16` with a warning that names `int8` as the quantized latent mode, and `--kv-quant-scheme turboquant` is disabled the same way. A Turbo-4 codec on the WHT-rotated latent itself (386 bytes per token per layer) was measured and not adopted: like 4-bit affine it flipped top-1 choices at reference top-two gaps up to 0.72 logits, outside the 0.11 near-tie bar, and it decoded 19 to 33% slower than `int8` ([results](benchmark_results/mla-turbo-latent-m3ultra-2026-11-19.md), #3157).

With `MLXCEL_MLA_ABSORBED=0` there is no latent: `deepseek_v2` caches decompressed per-head K/V, `int8` is per-token absmax there (5120 bytes per token per layer on V2-Lite) and the Turbo modes apply to ordinary rows. The memory estimate (`mlxcel inspect`, the `--estimate-memory` preflights, `--recommend-quant` and the server's prompt-cache store budget) sizes the layout the run resolves to in every case (#3153, #3155).

### DeepSeekV3Attention families: fp16 or an affine int8 latent

`deepseek_v3`, `kimi_k2`, `joyai_llm_flash`, `kimi_vl`, `kimi_k25`, `youtu`, `youtu_llm` and `youtu_vl` (`mla::QUANTIZED_LATENT_FAMILIES`) all run `DeepSeekV3Attention`, which writes the `(kv_latent, k_pe)` pair through `mla::MlaLatentCache` on every step, prefill and single-sequence decode alike, and the shared batched decode helper (`src/models/mla_batched.rs`) does the same per row. They hold the same two layouts as DeepSeek V2, at the same rates (1152 and 672 bytes per token per layer at `kv_lora_rank` 512 and `qk_rope_head_dim` 64), selected by the same flags: `--kv-cache-mode int8`, `--cache-type-k int8 --cache-type-v int8` or `--kv-bits 8`. These families cache the latent in every absorption state, so `MLXCEL_MLA_ABSORBED` does not change the layout. A one-query decode step over the int8 latent runs the fused Metal kernel's quantized variant on a window of up to 2048 keys (`absorbed_fused`, #3156) and the composed `quantized_matmul` decode beyond that and on other backends (`absorbed_composed`), multi-token steps dequantize the window once, the group size is fixed at 64, and `--kv-skip-last-layer` keeps the last layer as an fp16 latent. A checkpoint whose `kv_lora_rank` is not a multiple of 64 cannot be grouped, so its cache keeps per-token int8 rows and logs one warning.

Measured on an M3 Ultra with Moonlight-16B-A3B (`deepseek_v3`) and Kimi-VL-A3B-Thinking (`kimi_vl`) ([results](benchmark_results/mla-quantized-latent-v3-m3ultra-2026-11-19.md)): the teacher-forced trace against the fp16 latent shows no disagreement on decided positions behind 1024 and 4096 tokens on either checkpoint (behind 1024, on the fused int8 kernel, two Moonlight top-1 flips are outside the 0.11-logit near-tie bar, at 0.117 and 0.328, and none on Kimi-VL; on the composed path before #3156 it was one per checkpoint, 0.117 and 0.180). Decode is about 10% slower than the fp16 latent behind 512 tokens (fused int8 kernel), 13 to 14% slower behind 2048 and 36% slower behind 16384 on Kimi-VL (composed path), the same cost as on DeepSeek V2.

A Turbo request on these families resolves to `fp16` with a warning that names `int8` as the latent mode they take, for the reasons listed below.

### Kimi Linear and Kimi K3: fp16 or an affine int8 latent on the MLA layers

`kimi_linear` and `kimi_k3` (the Kimi K3 text backbone and its VLM wrapper) are hybrid KDA + MLA stacks whose caches are model-owned. Both take the resolved per-layer mode table (`set_kv_cache_layer_modes`) and build each MLA layer's cache in its entry's mode; a KDA layer ignores its entry and keeps its recurrent state. The MLA write goes through `mla::MlaLatentCache`, so an `int8` table stores the same affine group-64 latent with fp16 `k_pe` as the `DeepSeekV3Attention` families, at the same rates (672 against 1152 bytes per token per layer at rank 512, rope 64), in single-sequence decode, prefill and the native batched decode. A one-query absorbed step over the int8 latent runs `mla::quantized_absorbed_attention` (the fused kernel's quantized variant when the step is unmasked and the window is at most 2048 keys on Metal, the composed path otherwise); a caller mask is added to the rope-score term as the fp16 path adds it, and multi-token steps dequantize the window once and cast it back to the activation dtype, so a bf16 checkpoint stays bf16 (an FP16 window against a bf16 query promotes to float32, which the KDA state would then carry into every decode step). Only the MLA layers hold KV, so `mlxcel inspect`, the memory preflights and `--recommend-quant` charge the latent rate on the layers that `linear_attn_config.kda_layers` does not list (7 of 27 on Kimi-Linear-48B). These families opt out of the prompt cache (`server::prompt_cache::hybrid_ssm`), so no parked latent is restored.

Measured on an M3 Ultra with Kimi-Linear-48B-A3B-Instruct 4-bit ([results](benchmark_results/mla-quantized-latent-kimi-m3ultra-2026-11-19.md)): zero decided-position disagreements behind 1024 (0 / 201) and 4096 (0 / 52); five top-1 flips behind each lie outside the 0.11-logit near-tie bar (largest 0.500 and 0.312), within the int8 decode's own fused-against-composed noise (16 flips, largest 0.438). Kimi's fp16 decode is a hand-written composed path with no kernel choice, so its noise floor is zero. Perplexity moves +0.29% and -0.40%. Decode is +1.5% against fp16 behind 512 tokens (fused int8), -4.7% behind 2048 and -23.3% behind 16384 (composed int8), with prefill and peak memory unchanged. Kimi K3 is not measured: the only MLX export is an expert-pruned 2-bit conversion, which would mix weight quantization and pruning error into a cache-mode comparison; the tiny-model tests (`models::kimi_k3::latent_tests`) cover it.

### GLM-4.7-Flash and LongCat: ported to the latent view; LongCat ships int8, GLM stays fp16

GLM-4.7-Flash (`glm4_moe_lite`) and LongCat (`longcat_flash`, `longcat_flash_ngram`, both loaded by `LongcatFlashNgramModel`) write the `(kv_latent, k_pe)` pair through `mla::MlaLatentCache` since #3161: GLM in its single-sequence forward, its batched decode and its MTP verify and rollback, LongCat in both MLA sub-layers of each decoder layer, whose model-owned caches take the per-layer mode table the engine hands the model. Both LongCat types are in `mla::QUANTIZED_LATENT_FAMILIES`, so `int8` stores the affine latent in both sub-layers of every layer. `glm4_moe_lite` is not, so its int8 latent stays off and `int8` resolves to `fp16` with the substitution warning, exactly as before; listing it switches its int8 latent on (GLM reads the list into a per-layer flag at load, LongCat into the flag its mode table checks).

GLM-4.7-Flash failed the quality gate. Measured on an M3 Ultra with `glm-4.7-flash-4bit` ([results](benchmark_results/mla-quantized-latent-glm-longcat-m3ultra-2026-11-19.md)), the teacher-forced trace against the fp16 latent shows 1 disagreement on decided positions behind 1024 tokens (of 193, reference top-two gap 2.75) and 0 behind 4096 (of 95), against a gate of zero at both lengths. There are 18 and 15 top-1 flips above the 0.11-logit near-tie bar, most at 0.5, which is one bf16 unit at GLM's logit magnitudes. The failing count is at the int8 kernel-noise floor, not below it. Switching only the int8 kernel (fused against composed, over the same stored rows) also gives 1 decided disagreement (gap 6.5), and the composed int8 arm has 0 against fp16. GLM's fp16 latent decodes on the composed path on every trace arm, so there is no fp16 kernel noise floor. Perplexity on the int8 arm moves -3.1% and -0.4%. An earlier measurement (2 decided at each length) ran the int8 context pass in f32 under the bf16 model and is superseded. Decode on the int8 latent, measured before that fix, which does not touch single-token steps, was -0.6% behind 512 tokens (fused kernel), -11.3% behind 2048 and -39.3% behind 16384.

LongCat-Flash-Lite (`longcat_flash_ngram`, `longcat-flash-lite-4bit`) passed the same gate on the same host and corpus: 0 decided-position disagreements behind 1024 tokens (of 239) and behind 4096 (of 121), with 9 and 5 top-1 flips above the 0.11 bar, the largest at a reference gap of 0.375. Switching only the int8 kernel gives 11 top-1 flips and 0 decided, so the int8 latent's flips are at the kernel-choice scale. Perplexity on the int8 arm moves -0.7% and -1.2%. Decode on the int8 latent measured +8.0% behind 512 tokens (fused kernel, against composed fp16), -15.5% behind 2048 and -49.2% behind 16384. The 560B `longcat_flash` is not on the host; it loads the same `LongcatFlashNgramModel` and is listed on that shared path, covered by the tiny-model tests of both types. `mlxcel inspect` and the advisor now count LongCat's two latent caches per decoder layer, at either mode; before #3161 they sized it at half.

### Other latent-pair families: fp16 only

`glm4_moe_lite`, `deepseek_v32` (also spelled `deepseek_v3.2`) and `glm_moe_dsa` store the `(kv_latent, k_pe)` pair in one `KVCache` (through the latent view, at fp16): the "K" slot holds the latent and the "V" slot holds `k_pe`. `deepseek_v32`, `deepseek_v3.2` and `glm_moe_dsa` concatenate the `index_head_dim`-wide DSA indexer key onto the same "K" slot, so theirs is 640 wide at the default `index_head_dim` of 128. Neither slot is a per-head K/V row at any of those widths, so no per-token quantized mode is calibrated for what they contain. The same two reasons rule out the Turbo modes on every latent-pair family, the `DeepSeekV3Attention` ones included:

- Symmetric `turbo4` builds its sign vectors and codebook from the "V" width and applies them to the "K" slot as well, reading 448 floats past the end of a 64-entry sign vector (576 on the indexer-carrying families). On `glm4-flash` every generated token came out `!`.
- The asymmetric modes (`fp16+turbo4`, `fp16+turbo3`, `turbo4-delegated`) quantize only the "V" slot, which here is `k_pe`. They therefore compress a **key** and leave the latent exact, which is the reverse of the "FP16 K, 4-bit V" contract in the mode table above. Output stayed finite, which is why this went unnoticed.

Any non-`fp16` mode requested for one of these families resolves to `fp16` with one warning naming the family and the requested mode, and `--kv-bits` / `--kv-quant-scheme` are disabled the same way.

**Decision (#3155, #3160, #3161, #3162):** the affine int8 latent ships for `deepseek_v2`, the `DeepSeekV3Attention` families, the Kimi Linear and Kimi K3 MLA layers, and both LongCat types.

**Decision (#3161): GLM-4.7-Flash stays on fp16.** It failed its gate with 1 decided-position disagreement behind 1024 tokens, so `glm4_moe_lite` keeps the fp16 substitution. Its write path is in place behind the allowlist.

**Decision (#3163): the DSA families stay on fp16.** `deepseek_v32`, `deepseek_v3.2` and `glm_moe_dsa` can now hold their 640-wide `[kv_latent | indexer key]` slot as the affine int8 triple (808 against 1408 bytes per token per layer): the indexer dequantizes only its own two groups, sparse decode gathers the packed rows, and the attention reads the latent packed. It was measured on DeepSeek-V3.2-4bit on an M3 Ultra ([results](benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md)) and is not enabled. The teacher-forced trace shows no decided-position disagreement behind 1024 and 4096 tokens, with top-1 flips the same size as the fp16 latent's own fused-versus-composed kernel noise on that checkpoint. The top-k overlap gate is missed: the indexer's selection over the int8 cache overlaps the fp16 selection by less than 0.99 on 49 of 61 layers behind 4096 tokens (0.966 on the worst) and on 55 of 61 behind 16384 (0.744 on the worst, on an fp16 reference that is itself degenerate at that length). For scale, the two fp16 kernels overlap by less than 0.99 on 29 layers behind 4096 (0.985 on the worst). A quantized request on these families resolves to `fp16` with a warning. GLM-5 was not measured and inherits the decision, since it wraps the DeepSeek V3.2 model.

### Experimental: `--mla-latent-quant`

`--mla-latent-quant <auto|fp16|int8|int4>` (environment `MLXCEL_MLA_LATENT_QUANT`; the CLI flag wins, and any other value is a parse error that lists the four) picks the precision every MLA latent cache of the loaded model is stored at. It exists so an operator who accepts a measured risk can use a precision the default gate does not ship. It is flattened into the same flag group as `--kv-cache-mode`, so `mlxcel generate`, `mlxcel run`, `mlxcel serve`, `mlxcel inspect`, `mlxcel-server` and `mlxcel-bench-decode` all take it. `mlxcel tune`, `mlxcel predict`, `mlxcel-bench-engine`, `mlxcel-engine-parity` and `speculative_bench` take no flag but read the environment variable at startup, where an unknown or non-UTF-8 value is an error as well.

| Value | Latent | Bytes per token per layer (rank 512, rope 64) |
|-------|--------|-----------------------------------------------|
| `auto` (default) | The decisions above: `int8` stores the affine 8-bit latent on `deepseek_v2`, the `DeepSeekV3Attention` families, Kimi Linear, Kimi K3 and LongCat; GLM-4.7-Flash and the DSA families keep fp16. Unset, output is byte-identical to a build without the option. | 1152 or 672 |
| `fp16` | fp16 on every MLA family, even under `--kv-cache-mode int8` or `--kv-bits 8`. | 1152 |
| `int8` | Affine group-64 8-bit on every MLA family, including `glm4_moe_lite`, `deepseek_v32`, `deepseek_v3.2` and `glm_moe_dsa`. | 672 (808 with the DSA indexer key) |
| `int4` | Affine group-64 4-bit on every MLA family. | 416 (552 with the DSA indexer key) |

An explicit value does not need `--kv-cache-mode int8`: it decides the storage of the model's latent caches on its own, and wins when the two disagree (one warning names both). Every KV cache of an MLA family is a latent cache (the KDA layers of Kimi Linear and Kimi K3 hold recurrent state, no KV), so nothing else in such a model is affected, and `--kv-bits` steps aside there, last layer included. On a model with no MLA latent cache, and on `deepseek_v2` with `MLXCEL_MLA_ABSORBED=0`, the option is ignored with one warning and the model's own layers keep the mode they were given: `--mla-latent-quant fp16 --kv-cache-mode int8` on a Llama model still builds int8 caches. A `kv_lora_rank` (or DSA key slot) that is not a multiple of 64 keeps fp16 with the existing whole-groups reason.

`int4` uses the same affine codec and group size as `int8`, at 4 bits. It **never uses the fused Metal kernel**: that kernel reads 8-bit codes only (#3156), so every `int4` decode step runs the composed `quantized_matmul` path (`absorbed_composed` in `MLXCEL_TRACE_MLA_STATS=1`), on every backend and at every window length. A multi-token step and a cold prefill dequantize the window once, as they do for `int8`.

On the DSA families the indexer key cached beside the latent stays 8-bit under `int4`, because the #3163 overlap gate showed the indexer's top-k selection is the precision-sensitive part. The 640-wide slot is then two runs of whole groups in the same buffers, `[512-wide latent at 4 bits | 128-wide indexer key at 8 bits]`: 256 + 128 packed bytes, ten groups of fp16 scale and bias, and the fp16 rope keys, 552 bytes per token per layer.

An explicit `int8` on a family outside the default list, and any `int4`, prints one startup warning stating that the precision is experimental, with the measured result for that family and width only: under `int8`, GLM-4.7-Flash's 1 decided-position disagreement and the DSA families' top-k overlap below 0.99; under `int4`, the 4-bit near-tie flips (up to 0.86 logits on DeepSeek-V2-Lite, 0.84 on Moonlight, 1.13 on LongCat-Flash-Lite), GLM-4.7-Flash's 3 decided disagreements, and for the DSA families the statement that their 4-bit slot is unmeasured on the real checkpoint. The texts live in one table (`mla::EXPERIMENTAL_LATENT_NOTES`), one width per entry.

Everything that sizes the cache follows the stored precision: `mlxcel inspect`, the `--estimate-memory` preflights, the server's prompt-cache budget and the `generate` banner. Under `int4` the memory estimate's `KV dtype` line reads `int4 (MLA latent, --mla-latent-quant int4)` rather than the `int8` storage mode, while `inspect --json` keeps the storage mode in `kv_cache_mode`. `--recommend-quant` and the advisor do not: they describe the default gate and never suggest an explicit value. A prompt-cache entry records its latent width in its own buffers, so an entry saved at one precision is not adopted by a server running another (`kv_mode_mismatch`) and the prompt is prefilled again.

Measured on an M3 Ultra ([results](benchmark_results/mla-latent-quant-option-m3ultra-2026-11-20.md)), teacher-forced against the fp16 latent behind 1024 and 4096 tokens. `int4` shows 0 decided-position disagreements on Moonlight-16B-A3B, LongCat-Flash-Lite, Kimi-Linear-48B-A3B and DeepSeek-V2-Lite, and misses the 0.11-logit near-tie bar on all four: 6 to 38 top-1 flips per trace lie above it, the largest at 0.84 (Moonlight), 1.13 (LongCat), 0.63 (Kimi Linear) and 0.86 (DeepSeek-V2-Lite) logits, with perplexity moving between -3.9% and +2.3%. On GLM-4.7-Flash `int4` fails the decided gate as well, with 3 decided disagreements behind each length and perplexity moving by -19.5% and -16.1%, so it is a poor fit there. A forced `int8` reproduces the epic's numbers exactly, GLM-4.7-Flash's 1 decided disagreement behind 1024 tokens included. Decode on LongCat-Flash-Lite at 512, 2048 and 16384 prompt tokens: fp16 124.7, 125.1 and 99.3 tok/s; int8 135.5, 106.9 and 50.2; int4 124.3, 110.0 and 54.5. `int4` is 3% to 9% faster than `int8` where both run composed and 8% slower behind 512 tokens, where `int8` has the fused kernel. DeepSeek V3.2 was not traced under the option: its int8 slot was measured in #3163, and its `int4` split slot has the tiny-model tests, a coherent generate run and a 4096-token sparse decode run behind it, with no quality measurement on the real checkpoint.

## Paged cache and server batching

The paged decode layout accepts TurboQuant modes through
`PagedKvLayout::uniform_with_mode`. Server dispatch routes Turbo modes to the
paged layout when `--decode-storage-backend paged` is selected.

### Unified paged KV cache

The continuous-batching server keeps both the cross-request prompt-prefix cache
and per-sequence decode state in one refcounted, copy-on-write block pool
(epic #116). The paged backend is the default for batch-capable pool-backed
families (`--decode-storage-backend auto` resolves to paged when batching);
`dense` forces the legacy per-sequence caches.

Two requests that share a prompt prefix store that prefix's KV blocks once.
Adoption is non-consuming clone-and-pin: a borrower clones pinned references to
the matched prefix blocks, so the stored entry survives for concurrent siblings
and for deeper future matches, and any number of in-flight requests can share
one stored prefix simultaneously. Partial matches adopt too: with Automatic
Prefix Caching (APC, on by default) the match is verified per 16-token hash
block, floored to the 32-token pool block boundary, and the borrower
re-prefills only its divergent suffix on fresh blocks (full shared blocks are
never mutated; a shared partial tail forks copy-on-write). Cache entries are
accounted at their REAL pool bytes, so `--prompt-cache-capacity-bytes`
(default 2 GiB) genuinely bounds retention and the LRU eviction actually
triggers.

Paged adopt and donate are supported for the pool-backed Fp16 families
(the dense-natural backends such as qwen3 and llama3); model-owned-state families
and recurrent or hybrid SSM models keep dense or model-owned caches and stay out
of the pool.

Model-owned families need one clarification, because the paged backend does
apply to them and it is easy to read that as pool participation. Gemma 3 and
Llama 4 report `supports_batching()`, so under
`--decode-storage-backend auto` (the default) the scheduler allocates them on
the paged backend. AFMoE declares model-owned state too, but it reports
`supports_batching() == false`, so it stays on the dense backend where the
allocated backend already reads `ModelOwned` and donation was already skipped;
it is unaffected either way. That allocation is **accounting only**: their real K/V lives
in the model's `ModelOwnedSequenceState`, `make_caches()` hands the pool nothing,
and the paged block table exists so `sync_paged_state_with_lengths` can mirror
their lengths. Such a sequence is never donated to the prompt cache and never
adopted from it. `CachePool::detach_paged` refuses a paged sequence with no
per-layer handles, `DetachedPagedCacheSet::clone_eligible` requires at least one,
and the scheduler declines the donation up front against the model's own
`sequence_state_layout().backend` rather than the backend it was allocated on
(the allocated one reads `PagedKvCache` for exactly these families, which is what
issue #1346 turned into a wrong turn-2 answer). The decline is visible, not
silent: `GET /v1/cache/stats` reports `reject_model_owned_state` and `/metrics`
carries `mlxcel_prompt_cache_reject_total{reason="model_owned_state"}`, so a
family whose store legitimately stays empty is distinguishable from a cache that
is broken. Dense prompt-cache adoption also compares every detached layer's KV
mode against the currently resolved per-layer table; mismatches decline with
`reject_kv_mode_mismatch` and
`mlxcel_prompt_cache_reject_total{reason="kv_mode_mismatch"}` instead of
installing a stale-mode cache. Real cross-request reuse for these families
comes through the model-state snapshot path (#1335), described next; a family
that opts into `supports_snapshot_reuse()` is handled by that branch first and
keeps donating.

### Exact-prefix snapshots for recurrent state

Hybrid-SSM and linear-attention families remain excluded from block sharing:
their recurrent hidden state cannot be reconstructed from a radix/APC token
prefix, and it cannot be truncated to an arbitrary earlier token. The same is
true of any family that owns its caches per `SequenceId` rather than handing
them to the scheduler's pool, whether or not the state is recurrent. For those
families, `mlxcel-server` has an orthogonal exact-prefix snapshot bucket. Models
that implement `supports_snapshot_reuse()` can copy their full model-owned
state at turn end and restore it into a fresh sequence when the next request's
tokens begin with that exact stored prefix under the same session key.

The snapshot-capable families are:

- Recurrent and hybrid SSM: Mamba, Mamba2, Jamba, Nemotron-H, Falcon-H1,
  Granite 4 H, PLaMo 2, LFM2, Qwen3-Next, Bailing MoE Linear, Inkling.
- Attention-cache families: Gemma 3, Gemma 4, AFMoE, Llama 4, Muse Glimmer,
  Qwen 3.5 / 3.6 text and MoE.
- VLM wrappers, which forward the five snapshot hooks to their text model:
  Gemma 4 VL and Unified, Qwen 3.5 VL, LFM2-VL, Granite 4 Vision, Muse Glimmer
  VLM, Inkling VL, and the shared `vision::VisionLanguageModel` used by the
  Gemma 3 and Llama 4 VLM checkpoints. The wrapper carries no per-sequence
  state of its own, and the prompt-cache key folds in the request's multimodal
  digest, so a restore cannot serve KV computed from a different image.

Regenerate the list with `grep -rn 'fn supports_snapshot_reuse' src/models
src/vision` rather than trusting the roster above.

The attention-cache families share one serializer, `src/models/kv_snapshot.rs`,
which covers the three ordinary cache types and fixes what a truncating restore
may do to each. Truncation matters because a stored snapshot of N tokens can
serve a request whose prompt is a proper prefix of those N, but only if the
shortened cache is indistinguishable from a cold prefill of the shorter prompt:

- `KVCache` (full attention) is always truncatable. Every token keeps its own
  slot, so dropping the tail is a rewind of `offset`.
- `RotatingKVCache` (sliding window) is truncatable only while the ring is
  unwrapped, meaning `buffer_size == 0 && idx == offset && offset <= max_size`.
  Once the ring wraps, logical token `t` no longer sits at slot `t` and a
  rewind would expose the wrong window.
- `ChunkedKVCache` (Llama 4 iGQA) is truncatable only while its front is
  untrimmed (`start_position == 0`). After a front trim the buffer holds
  `[start_position, offset)`, while a cold prefill of `target_len` tokens would
  hold `[target_len - chunk_size, target_len)`, so the truncated restore would
  attend over strictly fewer tokens than the cold run.

Exact-prefix restore is unaffected by all three rules: a full-length restore
reproduces the cache as captured, wrapped ring and trimmed front included. Only
the shorter-prefix variant declines.

A snapshot is keyed on a token vector, so the cached state has to hold exactly
those tokens and no others. The M5 neural-accelerator prefill alignment pads a
chunk to a 32-token tile and then trims the padding back out of the
`CachePool`'s caches, and a model-owned sequence has none there, so the pad
positions would stay in the model's own caches and run `offset` past the real
token count. The scheduler therefore also calls
`LanguageModel::trim_state` (with the sequence's id) for a model-owned family, which rewinds
the model's own per-sequence state; the request is aborted if that rewind
fails. Gemma 3 implements it (a sliding-window layer also cuts its physical
ring buffer back, so no pad key survives the next decode step) and keeps the
aligned path. Gemma 4, AFMoE and Llama 4 do not implement it yet and answer
`supports_padded_prefill()` with `false`. Pool-backed families are unaffected.

Snapshot reuse is deliberately conservative with quantized attention caches.
Every family listed above serializes FP16 model-owned attention KV state only;
if a live layer is configured for Int8 or Turbo mode, the shared serializer
refuses donation and restoration with an explicit error rather than copying only
`keys`/`values` and restoring them as FP16. Restores also compare the serialized
snapshot mode with the live per-layer table, so a snapshot captured under a
different KV mode falls back to cold prefill instead of corrupting the cache.
`ChunkedKVCache` has no quantized variant, so its Llama 4 layers are FP16 by
construction and the quantized-mode request is warned about once at cache
construction instead.

The snapshot bucket has its own byte cap, entry cap, TTL, LRU counters, and
hit/miss metrics. `GET /v1/cache/stats` reports `snapshot_*` fields, while
`/metrics` exposes `mlxcel_prompt_cache_snapshot_hits_total`,
`mlxcel_prompt_cache_snapshot_misses_total`,
`mlxcel_prompt_cache_snapshot_tokens_reused_total`, and labeled snapshot
evictions. This path is deliberately whole-prefix only: it does not participate
in APC block matching, does not share SSM state across sessions, and does not
modify the block-sharing carve-out for hybrid SSM families.

The block pool is bounded with `--kv-cache-budget <bytes|auto|none>` (env
`MLXCEL_KV_CACHE_BUDGET`); the default is `auto`, which derives the cap from
the memory estimate so the batched-decode default (#628) cannot run
concurrent full-context sequences into an OOM abort. `none` (or `0`) restores
the previous unbounded pool. Under a budget the scheduler evicts cold cached
prefixes, then preempts, before rejecting a request. `GET /v1/cache/stats`
reports paged block usage (block size, allocated, live, free, bytes
reserved/in use, and the budget).

### Measured payoff

Numbers below are from an M1 Ultra with `models/qwen3-0.6b-4bit` (28 layers, 8 KV
heads, head dim 128, runtime KV dtype bf16, so about 112 KiB of KV per token).

**Memory saved per shared prefix.** A shared prefix of `P` tokens across `N`
concurrent requests occupies one pool copy instead of `N`. KV bytes per token are
`2 (K and V) * n_kv_heads * head_dim * num_layers * dtype_bytes`. A shared
1024-token system prompt is about 112 MiB; with 8 concurrent requests the pool
stores it once instead of eight times, saving roughly `7 * 112 MiB ≈ 784 MiB`.

**Prefill tokens avoided.** Only the first of `N` requests sharing a `P`-token
prefix prefills it; the other `N - 1` adopt the cached blocks and skip
`(N - 1) * P` prefill-token forward passes. `tests/paged_prefix_share_parity.rs`
confirms an adopting request decodes byte-identically to a cold run while
skipping the shared prefill.

**End-to-end server footprint.** Whole-process physical footprint
(`/usr/bin/footprint`) of `mlxcel serve` under fixed HTTP workloads
(`scripts/bench_memory_footprint.py`, `models/llama-3.2-1b-4bit`, M1 Ultra,
defaults, peak during the scenario phase):

| scenario | v0.1.4 | current default | shared tokens |
|---|---|---|---|
| idle (weights only) | 957 MiB | 983 MiB | - |
| 8 concurrent requests, shared ~3.7k-token system prompt | 1653 MiB | 1627 MiB | 29184 (all 8 adopt) |
| same prefix, 8 sequential requests | 1650 MiB | 1413 MiB | 25536 |
| one conversation, 8 turns | 2104 MiB | 1301 MiB | 10976 |
| 32 distinct prompts (churn) | 1558 MiB | 2276 MiB | 1984 |

v0.1.4 never engaged the prompt cache from the HTTP path (zero reuse), so its
footprint is the no-cache floor. The current default matches or beats it on
every sharing scenario while also skipping the shared prefill work (the
8-way burst completes its request phase 3.4x faster). The churn number is
higher because donated entries are now retained for future reuse; that
retention is real memory governed by `--prompt-cache-capacity-bytes` and is
released by LRU eviction at the cap.

**Decode throughput.** The parity evidence for paged decode is intentionally
scoped: `tests/paged_scheduler_parity.rs` covers scheduler-shaped qwen3 and
llama3 Fp16 dense-natural-cache allocations at B=1/B=2, plus a CUDA qwen3 arm
that requires greedy-token equality and bounded logit RMS with TF32 disabled.
That evidence does not claim byte identity for VLM front ends, model-owned
caches, Turbo/quantized-KV modes, or every CUDA reduction geometry. The live
batched path is the pooled paged decode of #899, dispatched from the cache
itself (`KVCache::attend`, ADR 0008) rather than by the model forward; a
gather-then-SDPA path is kept as a correctness reference. At batch 4 the native
kernel runs at 276 tok/s for a 512-token prompt and 84 tok/s for a 4096-token
prompt, versus 146 and 7.7 tok/s for the gather reference (1.9x and 10.9x). The
gather reference degrades sharply with context because it re-materializes the
visible window every step, which is why the live path uses the native kernel.
The separate fused split-K kernel (Metal, and since #634
also CUDA via `mx.fast.cuda_kernel`)
(`MLXCEL_PAGED_ATTENTION_NATIVE`, feeding `paged_decode_attention_pooled`) is a
different code path from the block-table kernel above. Since #331 it is no
longer a plain on/off switch: an adaptive selector dispatches it on Metal only
inside the batch>=4 / ctx<=4096 / single-slab island ADR 0001 measured it
winning; on CUDA (#634) it dispatches on any single-slab layer regardless of
batch or context, since the Metal ceilings do not apply there. Gather runs
everywhere else, and the env var still force-pins either arm for A/B testing.
The chunked slab storage narrows that island further, since
the kernel declines (falling back to gather) once a layer has grown past one
slab. #710 retired the pooled entry point and its selector to a library-only
API, off the `mlxcel serve` decode path, which stays on the block-table kernel
described above. The variable itself keeps two consumers today, per
`resolve_dispatch_decision` and `resolve_paged_v2_dispatch` in
`src/lib/mlxcel-core/src/layers.rs`: that library-only pooled entry point, and
the server's pool-backed batched paged decode. On the server side, issue #899
made the fused v2 kernel the production decode path and named this variable's
force-off values its kill switch; a force-on value pins v2 for every servable
shape, bypassing the measured token floors. See
[Paged decode v2 variables](environment-variables.md#paged-decode-v2-variables)
for the floors and defaults, and
[Continuous batching](CONTINUOUS_BATCHING.md#seeing-which-path-ran) for the
dispatch policy and per-outcome log lines. History: #710's retirement of the
library entry point is where the "not a server knob" reading came from; #899
gave the variable its second, server-side consumer. See ADR 0001's #710 decision
record, [ADR 0001](adr/0001-paged-attention-gather-vs-fused-kernel.md).

Pool growth appends fixed-size slabs instead of reallocating one big tensor
per layer, so extending the pool never copies existing KV and never strands a
ladder of old buffer sizes in the allocator cache. Eight concurrent requests
with distinct ~4k-token prompts (qwen3-0.6b, the nothing-shareable stress)
peak at 3739 MiB versus 5013 MiB before the change, 1.13x of the dense
backend on the same workload.

## Recommended starting points

| Workload | Recommendation |
|----------|----------------|
| General serving | Start with `fp16`. Quantized KV is for memory pressure, not speed. |
| Lower KV memory, fastest quantized option | Use `int8` (~2x) or `turbo4-delegated` (~4x V); both also speed up prefill. |
| Maximum V compression with exact K | `fp16+turbo4` (turbo4-asym), accepting ~0.2-0.4x decode; enable boundary-V for quality. |
| Considering symmetric `turbo4` | Use only on an allowlisted/validated family; it is the slowest mode. |
| Memory-extremis only | `fp16+turbo3` decodes near-unusably; confirm the footprint is worth the speed cost. |

## Advisor recommendations (`--recommend-quant`)

`mlxcel generate --recommend-quant` prints an advisory KV-cache-mode section
alongside the quantization advice. It suggests one of `fp16`, `int8`,
`turbo4-delegated`, `fp16+turbo4`, or `turbo4` per model family and context
range (it withholds `fp16+turbo3`, whose decode is near-unusable), so you have a
benchmark starting point instead of guessing. The suggestions follow the
measured trade-off above: int8 and turbo4-delegated as the fast picks,
fp16+turbo4 as the exact-K memory pick, symmetric turbo4 for allowlisted
families only.

The suggestions are advisory and opt-in only. The default inference path is
unchanged: with no flags the runtime still uses `fp16`. To use a suggested mode
you must pass `--kv-cache-mode` (or `--cache-type-k` / `--cache-type-v`)
yourself. The advisor reads only `config.json`; it never loads weights and
never changes the quantized-weight dtype. KV-cache modes quantize only the K/V
cache tensors and dequantize back to FP16 for attention, so a recommendation
cannot reintroduce the bf16-to-f16 quantized-weight promotion that the project
forbids (the `#289` regression).

How it keys the suggestion:

- **Model family** comes from the KV-architecture classifier
  (`src/execution/kv_arch.rs`): standard attention, sliding-window, MLA
  (DeepSeek), hybrid attention plus recurrent, and pure SSM. The raw
  `model_type` is also checked against the symmetric-Turbo4 allowlist
  (`src/lib/mlxcel-core/src/cache/turbo/allowlist.rs`).
- **Context range** is bucketed as short (`<=4K` tokens, interactive /
  single-request), medium (`4K-32K`), and long (`>32K`, long-context serving).
  Long context and memory-constrained serving are prioritized over raw
  short-decode tok/s, because that is where KV-cache pressure dominates.

The conservative rules the advisor applies:

| Family / context | Suggested | Also benchmark | Why |
|------------------|-----------|----------------|-----|
| Any family, short context | `fp16` | - | KV footprint is small; keep the baseline. |
| Standard / sliding-window / hybrid, medium | `int8` | `turbo4-delegated` | int8 is the fastest quantized mode (~2x); turbo4-delegated for ~4x V at a similar speed. |
| Standard / sliding-window / hybrid, long, allowlisted | `turbo4` | `turbo4-delegated` | Family passed the PPL gate; turbo4 is max compression, delegated is the faster fallback. |
| Standard / sliding-window / hybrid, long, not allowlisted | `turbo4-delegated` | `fp16+turbo4` | Fastest ~4x V compression; fp16+turbo4 is the exact-K alternative. Symmetric `turbo4` withheld off the allowlist. |
| MLA (DeepSeek V2 without absorbed decode), medium or long | `int8` | - | The cache holds decompressed per-head rows whose dimension is not a power of two, so the Turbo Walsh-Hadamard V path does not apply; per-token INT8 has no head-dim constraint. |
| MLA (DeepSeek V2 under absorbed decode), medium or long | `int8` | - | Stores the MLA latent as affine group-64 8-bit, about 1.7x smaller than the fp16 latent, and keeps absorbed decode (#3155). |
| MLA `DeepSeekV3Attention` families (DeepSeek V3, Kimi K2, Kimi-VL, Youtu), Kimi Linear and Kimi K3, medium or long | `int8` | - | Stores the MLA latent as affine group-64 8-bit, about 1.7x smaller than the fp16 latent (#3160, #3162). |
| MLA LongCat (`longcat_flash`, `longcat_flash_ngram`), medium or long | `int8` | - | Stores both sub-layer latents as affine group-64 8-bit (#3161). |
| Other MLA latent-pair families (GLM-4.7-Flash, DeepSeek V3.2) | `fp16` | - | Every quantized mode is substituted back to fp16 at startup. |
| Pure SSM (Mamba/Mamba2) | `fp16` | - | No context-proportional KV cache, so Turbo modes save almost nothing. |
| Non-power-of-two head dim (e.g. Phi-2 at head_dim 80) | `int8` (medium/long), `fp16` (short) | - | The Turbo Walsh-Hadamard transform requires a power-of-two head dimension; the advisor downgrades any Turbo suggestion to `int8` or `fp16` for these families, matching the MLA treatment. |

The head-dimension check reads `head_dim` or `head_size` directly from `config.json` when present. When neither explicit field exists, it divides the hidden size by the attention head count, checking the alternate field names used by each naming convention: `hidden_size` / `d_model` / `dim` / `model_dim` for hidden size, and `num_attention_heads` / `num_heads` / `n_heads` / `n_head` for head count. This mirrors the field-name coverage in the KV-architecture classifier so the two paths agree on the derived head dimension. When none of the required fields are present, the advisor conservatively assumes Turbo is applicable and leaves the suggestion unchanged.

Symmetric `turbo4` is only ever suggested for families on the allowlist; off
the allowlist the advisor leads with `turbo4-delegated` at long context and
`int8` at medium, matching the measured trade-off above. `fp16+turbo3` is never
suggested. Treat every suggestion as a hypothesis to validate per family with
the checklist below before adopting it in production.

## Validation checklist

Before recommending a TurboQuant mode for a model family:

1. Run a short smoke generation to confirm the mode loads and decodes.
2. Compare output quality against FP16 on a representative prompt set.
3. Run perplexity/NIAH or another task-appropriate quality gate.
4. Measure prefill and decode throughput separately.
5. Test the exact server path if the mode will be used with continuous batching
   or paged decode.
6. Record hardware, MLX commit/version, model checkpoint, prompt/decode shape,
   and all cache flags.

Ignored tests in `tests/turbo_kv_e2e.rs` are the right place for hardware/model
quality gates; they are not part of the default `cargo test` run because they
require local model checkouts.

## Known limitations

- TurboQuant is not uniformly validated across all text and VLM families.
- VLM long-context and multi-image prompts need separate validation.
- Older Apple Silicon generations and non-Hopper/non-Blackwell CUDA paths may
  have different bottlenecks from the developer benchmark machines.
- Experimental environment-variable paths are useful for A/B testing but should
  not appear in user-facing recommendations without fresh benchmark data.
- No TurboQuant mode is available on an MLA latent cache. `deepseek_v2` under absorbed decode, the `DeepSeekV3Attention` families, the Kimi Linear and Kimi K3 MLA layers and LongCat hold `fp16` or the affine `int8` latent; the other latent-pair families serve at FP16 whatever is requested.
