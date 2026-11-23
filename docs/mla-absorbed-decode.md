# MLA matrix-absorbed decode

Multi-head latent attention (MLA) is the DeepSeek-family attention that compresses K and V into a single latent vector per token. This page covers mlxcel's matrix-absorbed decode path, which serves attention directly against that latent instead of up-projecting it first.

Status: opt-in, off by default for DeepSeek V2. Measured against the default decompressed layout on deepseek-v2-lite-chat-4bit-mlx (issue #3102); see [Default layout for DeepSeek V2](#default-layout-for-deepseek-v2) for why it stays opt-in.

## The identity

MLA stores one latent `c` of width `kv_lora_rank` (512 in every shipping checkpoint) plus a shared rope stream `k_pe` of width `qk_rope_head_dim` (64). `kv_b_proj` up-projects `c` into per-head keys and values. Naming its two row blocks `W_UK` and `W_UV`:

```
score:  q_nope . (W_UK c)  ==  (W_UK^T q_nope) . c
output: attn @ (W_UV c)    ==  (attn @ c) @ W_UV
```

Folding `W_UK` into the query projection and `W_UV` after the attention removes the up-projection from the decode graph without changing the arithmetic. Attention then runs against `c` with `num_heads` query heads sharing one latent KV head, and the score becomes the sum of two dot products, one over `kv_lora_rank` and one over `qk_rope_head_dim`.

## What it buys

The KV cache stops scaling with head count.

| Geometry | Decompressed bytes/token/layer (f16) | Latent bytes/token/layer (f16) | Ratio |
|---|---|---|---|
| DeepSeek-V2-Lite (16 heads) | 10240 | 1152 | 8.9x |
| DeepSeek-V3 (128 heads) | 81920 | 1152 | 71.1x |

These come from `mlxcel_core::mla::decompressed_bytes_per_token` and `latent_bytes_per_token`, which are the same functions the benchmark harness prints, so the table cannot drift from the code.

The cost is fixed weight memory, and how much depends on the checkpoint. A quantized `kv_b_proj` folds without dequantizing: its packed rows are regrouped per head along the rank axis, so `absorb_queries` and `unabsorb_output` run as batched `quantized_matmul` on the checkpoint's own planes (30.4 MiB on DeepSeek-V2-Lite, against 28 MiB for the 4-bit original) (#3121). An unquantized layer, an NVFP4 global scale, runtime LoRA terms or a mode mismatch keep the dense f16 fold (about 113 MiB on V2-Lite); the load line says which, as `folded 27/27 layers (27 quantized, 30.4 MiB)`.

## Default layout for DeepSeek V2

DeepSeek V2 caches the MLA latent by default wherever the fused absorbed-decode kernel runs (#3131) and on CUDA and ROCm (#3133), and decompressed per-head K and V elsewhere or under `MLXCEL_MLA_ABSORBED=0`. The default followed two rounds of measurement. Issue #3102 measured both layouts on deepseek-v2-lite-chat-4bit-mlx on an M5 Max (2026-11-15) with the rule that the latent layout becomes the default only if it holds decode throughput within a few percent and the teacher-forced trace shows no material disagreement:

| measurement | decompressed (then default) | absorbed | |
|---|---|---|---|
| decode, pp512/tg128 (median tok/s) | 200.6 | 180.7 | -9.9% |
| decode, 2048-token prompt, tg128 | 168.1 | 161.9 | -3.7% |
| decode, 16384-token prompt, tg128 (mean) | 70.9 | 111.2 | +57% |
| peak, 28,705-token prompt, 2048-token pieces | 20.20 GB | 13.52 GB | -6.68 GB |
| peak, same prompt, single pass | 48.55 GB | 41.02 GB | -7.53 GB |
| logit trace, decided positions that disagree | reference | 0 of 3,653 | prefill width 2048 and decode width 1 |

The trace condition holds: every disagreement falls where the reference's top two were within 0.19 logits, so the two layouts differ only in the rounding class. The throughput condition does not: absorbed decode carries a fixed per-step cost of about 0.55 ms at 512 tokens of context, which its smaller cache read pays back only past roughly 3K tokens. A default that made short chat-length requests 10% slower to save memory they did not need was declined at that point, and the decompressed layout stayed the default.

What would change the decision is removing that fixed cost. Issue #3121 removed most of it, and the decision is open again:

- The folds now run quantized.
- Unmasked single-query absorbed attention runs on a fused Metal kernel for windows up to 8192 keys (`MLXCEL_MLA_FUSED_DECODE`, on by default). An int8 latent window takes the kernel's quantized variant up to 2048 keys (#3156, see the quantized latent section).

Re-measured on the same checkpoint (2026-11-16):

| measurement | decompressed (then default) | absorbed | |
|---|---|---|---|
| decode, pp512/tg128 (median tok/s) | 205.5 | 201.4 | -2.0% |
| decode, 2048-token prompt, tg128 | 170.1 | 180.7 | +6.2% |
| decode, 16384-token prompt, tg128 | 73.5 | 114.4 | +56% |
| logit trace, decided positions that disagree | reference | 0 of 393 | decode width 1 behind 1024, 4096 and 8192 |

Both conditions now hold, and issue #3131 made absorbed decode the default on Metal, where they were measured. Re-measured on the switched binary (2026-11-16, shared host, so absolute rates run low; arms interleaved):

| measurement | `MLXCEL_MLA_ABSORBED=0` | default (absorbed) | |
|---|---|---|---|
| decode, pp512/tg128 (median of 6) | 171.3 | 168.5 | -1.7% |
| decode, 2048-token prompt, tg128 (median of 4) | 150.9 | 153.5 | +1.8% |
| decode, 16384-token prompt, tg128 | 71.4 | 105.8 | +48% |
| logit trace, decided positions that disagree | reference | 0 of 315 | decode width 1 behind 1024 and 4096 |

`MLXCEL_MLA_ABSORBED=0` keeps the decompressed layout for anyone who measures their own workload differently. The same run validated the fused kernel on an always-absorbed family end to end: on kimi-vl-a3b-thinking-4bit (`deepseek_v3` decoder) the `absorbed_fused` counter covers every decode step behind 1024 and 4096 tokens, the trace against `MLXCEL_MLA_FUSED_DECODE=0` disagrees at 0 of 242 decided positions, and decode is 7% faster at both 512 and 2048 tokens. Rows: [the #3121 benchmark results](benchmark_results/mla-absorbed-decode-cost-m5max-2026-11-16.md) and [the #3131 results](benchmark_results/mla-absorbed-default-m5max-2026-11-16.md).

CUDA has no fused kernel, so absorbed decode there takes the composed path. Issue #3133 measured it on GB10 (sm_121, CUDA 13.0) against the same two conditions, ABBA order with five runs per arm at 512 tokens:

| measurement | `MLXCEL_MLA_ABSORBED=0` | `MLXCEL_MLA_ABSORBED=1` (composed) | |
|---|---|---|---|
| decode, pp512/tg128 (median of 5) | 96.05 | 98.52 | +2.6% |
| decode, pp512/tg128, 160 warmup tokens (median of 5) | 98.63 | 98.92 | +0.3% |
| decode, pp512/tg512 (median of 3) | 93.47 | 94.43 | +1.0% |
| decode, 2048-token prompt, tg128 (2 runs) | 83.8 / 79.1 | 98.7 / 96.9 | about +20% |
| decode, 16384-token prompt, tg128 (2 runs) | 30.1 / 30.4 | 72.0 / 73.7 | about 2.4x |
| logit trace, decided positions that disagree | reference | 0 of 208 and 0 of 106 | decode width 1 behind 1024 and 4096 |

Run-to-run spread within an arm is 4 to 8% at 512 tokens and the medians never put the composed path behind, so both conditions hold and the default absorbs on CUDA too. On Metal with `MLXCEL_MLA_FUSED_DECODE=0` the composed path is the one #3102 measured at about -10%, so that configuration keeps the decompressed default; the CPU backend is unmeasured and keeps it as well. Rows: [the #3133 results](benchmark_results/mla-and-prefill-memory-cuda-gb10-2026-11-18.md).

ROCm has no fused kernel either, so absorbed decode there is the same composed path. Issue #3133 measured it on a gfx1151 host (Radeon 8060S, 30 GB unified memory) in ABBA order with 60 s idle before each run:

| measurement | `MLXCEL_MLA_ABSORBED=0` | `MLXCEL_MLA_ABSORBED=1` (composed) | |
|---|---|---|---|
| decode, pp512/tg128 (median of 5) | 42.72 (41.62 to 43.57) | 41.28 (36.94 to 41.78) | -3.4% |
| decode, 2048-token prompt, tg128 (median of 2) | 35.28 (35.87, 34.68) | 41.41 (40.88, 41.93) | +17% |
| decode, 16384-token prompt, tg128 (median of 2) | 22.04 (22.05, 22.02) | 36.49 (36.19, 36.78) | +65% |
| phase peak, 16384-token prompt | 16.06 GB | 12.39 GB | -23% |
| logit trace, decided positions that disagree | reference | 0 of 208 and 0 of 105 | decode width 1 behind 1024 and 4096 |

The 512-token cell was run twice: an initial ABBA pair read +5.6% with a 14% spread in the decompressed arm, and a five-run-per-arm repeat reads -3.4%, close to the -1.7% Metal traded for its fused-kernel gains. From 2048 both arms are tight and the composed path is well ahead, and neither trace shows a decided disagreement, so both conditions hold on ROCm as well. Rows: [the #3133 ROCm results](benchmark_results/mla-and-prefill-memory-rocm-gfx1151-2026-11-18.md).

No other family shares the choice: `deepseek_v3` and `deepseek_v32` absorb unconditionally, and `minicpm3` and `dots1` cache decompressed K/V with no absorbed path. Every sample and trace row is in [the benchmark results](benchmark_results/mla-v2-default-layout-m5max-2026-11-15.md).

## Flags

| Variable | Effect |
|---|---|
| `MLXCEL_MLA_ABSORBED` | Cache `(ckv, kpe)` and run absorbed decode (Stage 1). On by default where the fused kernel runs (#3131) and on CUDA and ROCm (#3133); `1`, `true`, `on` or `yes` force it on every backend, and `0`, `false`, `off` or `no` restore the decompressed per-head K/V cache. |
| `MLXCEL_MLA_SPLIT_KV=1` | Additionally cut the latent range into chunks and merge the partial states (Stage 2). Off by default; ignored under `MLXCEL_MLA_ABSORBED=0`. |

`MLXCEL_MLA_ABSORBED` is a tri-state (`mlxcel_core::mla::AbsorbedMode`): an on spelling (`1`, `true`, `on`, `yes`) forces absorbed decode, an off spelling forces the decompressed layout, and unset, empty or unrecognized is the default. The default (`mlxcel_core::mla::default_absorbs`) absorbs where the fused absorbed-decode kernel runs, which today means Metal with `MLXCEL_MLA_FUSED_DECODE` not turned off, and on CUDA and ROCm, where #3133 measured the composed path within a few percent of the decompressed one at 512 tokens and ahead of it from 2K. Elsewhere absorbed decode takes the composed path without that evidence: Metal with the kernel off carried the roughly 10% short-context cost #3102 measured, and the CPU backend is unmeasured, so both keep the decompressed layout by default. A typo keeps the default layout instead of changing it. `MLXCEL_MLA_SPLIT_KV` stays opt-in and accepts only the on spellings.

Under `MLXCEL_MLA_ABSORBED=0`, no fold is built, no weight is dequantized, and the family runs the pre-existing decompressed path unchanged. Absorbed decode still declines per cache, keeping the decompressed path, for a Turbo KV cache mode and for paged-backed caches. With absorption on, `--kv-cache-mode int8` (or `--kv-bits 8`) stores the latent itself quantized (#3155, see below), and the `deepseek_v2` mode resolver substitutes fp16 for a Turbo request and warns, because the decompressed quantized layout is larger than the fp16 latent (about 7.2 KB per token per layer for turbo4-asym on V2-Lite, against 1152); `MLXCEL_MLA_ABSORBED=0` keeps the quantized decompressed layout. The memory estimate (`mlxcel inspect`, `--estimate-memory`, `--recommend-quant`, the prompt-cache store budget) sizes the layout the run resolves to (#3153, #3155).

## Where it applies

| Family | Before this work | Now |
|---|---|---|
| `deepseek_v2` | Decompressed cache | Absorbed by default on Metal (#3131) and on CUDA and ROCm (#3133), decompressed by default elsewhere; `MLXCEL_MLA_ABSORBED` forces either |
| `deepseek_v3` | Already absorbed (`embed_q` / `unembed_out` decomposition in `sanitize_weights`) | Unchanged |
| `deepseek_v32` | Already absorbed, same decomposition | Unchanged |
| `minicpm3` | Decompressed cache | Unchanged, still decompressed |
| `dots1` | Decompressed cache | Unchanged, still decompressed |

V3 and V3.2 have carried their own copy of the absorption transform since before this work. The shared `mlxcel_core::mla` module is where they can converge, and where MiniCPM3 and dots1 can be folded next.

## Cache layout and what stays working

The latent cache is a `KVCache` in FP16 mode whose two slots hold asymmetric tensors:

| Slot | Holds | Shape |
|---|---|---|
| `keys` | `ckv` | `[B, 1, L, kv_lora_rank]` |
| `values` | `kpe` | `[B, 1, L, qk_rope_head_dim]` |

The FP16 update path already reads each side's head dimension from its own shape, so this needs no new cache type. That matters: a genuinely new type would force the family into the `ModelOwnedSequenceState` escape hatch the SSM hybrids use, which costs batching, the paged decode backend, and the generic prompt-cache donation path. Instead `offset`, `live_start`, `trim`, `trim_front`, `nbytes`, and `can_trim_prompt_cache` all keep their existing meanings on latent rows.

The slot assignment matches what `src/models/deepseek_v3.rs` has always done, so V2 folded onto the shared type and V3's existing code describe the same buffer.

### Quantized latent (`int8`)

Under `--kv-cache-mode int8` the same cache stores the latent as the MLX affine triple `quantize_weights` produces, reusing the four per-token buffers a per-head INT8 cache has, so `trim`, `trim_front`, `gather_positions`, detach/adopt and the prompt-cache store move the rows and their scales together (#3155):

| Slot | Holds | Shape | dtype |
|---|---|---|---|
| `keys` | packed `ckv` | `[B, 1, L, kv_lora_rank * 8 / 32]` | uint32 |
| `values` | `kpe` | `[B, 1, L, qk_rope_head_dim]` | fp16 |
| `key_scales` | affine group scales | `[B, 1, L, kv_lora_rank / 64]` | fp16 |
| `val_scales` | affine group biases | `[B, 1, L, kv_lora_rank / 64]` | fp16 |

That is 672 bytes per token per layer on V2-Lite against 1152 for the fp16 latent. `MlaLatentCache::update_and_fetch` returns a `LatentWindow`, either `Dense { ckv, kpe }` or `Quantized { ckv_q, scales, biases, kpe, bits, group_size }`, and a reader takes the codec parameters from the window. A single-query step runs both contractions on the packed rows:

```text
scores   = scale * quantized_matmul(q_absorbed, ckv_q, transpose = true) + scale * q_pe @ kpe^T
o_latent = quantized_matmul(softmax(scores), ckv_q, transpose = false)
```

in f32, recorded as `absorbed_composed`. On Metal an unmasked single-query step over a window of up to 2048 keys runs on a second variant of the fused kernel instead (#3156, `mla::decode::fused_absorbed_decode_quantized`, recorded as `absorbed_fused`). It is the same kernel body with a different latent read: the packed codes are read as bytes and each group's scale and bias are factored out of the score and the value sum, so the window is not dequantized. `MLXCEL_MLA_FUSED_DECODE=0` turns it off together with the fp16 kernel. The limit is lower than the fp16 kernel's 8192 (`FUSED_DECODE_QUANTIZED_MAX_KEYS`) because the composed quantized path is the faster one on longer windows: on an M3 Ultra int8 decode on the kernel is 5.9% faster than the composed path behind 512 tokens, even near 2048 and 4.6% slower behind 3840, with no decided-position disagreement in the trace against either the composed int8 path or the fp16 latent ([mla-fused-quantized-decode-m3ultra-2026-11-19](benchmark_results/mla-fused-quantized-decode-m3ultra-2026-11-19.md)). A 4-bit window, a group size other than 64, a rank that is not a power of two from 64 to 1024, a masked step and every other backend keep the composed path. The dense fused kernel and split-KV read an fp16 latent and are not tried on a quantized window. A multi-token step and a cold prefill dequantize the live window once per step and run the fp16 code unchanged, after the fresh chunk was stored quantized. Only `quantize_weights`, `quantized_matmul` and `dequantize` are used, so the path runs on every backend the quantized weight path runs on. A `kv_lora_rank` that is not a multiple of 64 declines the quantized latent and keeps the decompressed path. Quality and speed: [mla-quantized-latent-m3ultra-2026-11-19](benchmark_results/mla-quantized-latent-m3ultra-2026-11-19.md). A 4-bit latent failed the same gate and is not offered, and so did a Turbo-4 codec on the WHT-rotated latent ([mla-turbo-latent-m3ultra-2026-11-19](benchmark_results/mla-turbo-latent-m3ultra-2026-11-19.md), #3157).

Declined, with a message, falling back to the decompressed path:

- Any Turbo KV cache mode (`fp16+turbo4`, `turbo4`, `fp16+turbo3`, `turbo4-delegated`). Their per-token quantization is calibrated for a per-head K/V row, not for a 512-wide latent whose reconstruction error is amplified by every query head that reads it.
- Paged backing. The pool allocates `[num_blocks, page_size, Hkv, head_dim]` with one head dimension for both sides, which the asymmetric `(512, 64)` split cannot fill.

The decline is decided per cache, and the answer is fixed for a cache's lifetime, so a run that declines on the first step declines on every step and a cache never mixes the two layouts.

## Multi-token steps

Absorption is not limited to single-token decode. Per head, with `r = kv_lora_rank`, `d = qk_nope_head_dim + v_head_dim`, `L` queries and `T` cached latent positions after this step's update, the absorbed path costs `L*r*d + 2*L*T*r` and materializing costs `T*r*d + L*T*d`. Absorbed wins while `L < r*d / (r*d/T + 2*r - d)`; `mlxcel_core::mla::use_absorbed_mla` evaluates that bound once per call and every MLA family branches on its result instead of on the query count.

A single query is always absorbed. As `T` grows the bound approaches `r*d / (2*r - d)` (170 queries for DeepSeek-V3, 398 for GLM-4.7-Flash), so a speculative verify block or a chunked-prefill continuation behind a long cache runs absorbed. On a cold cache (`T == L`) the bound falls below any real `L` for every shipped config, so a fresh prefill still materializes and stays numerically identical to the baseline.

Both paths write the same latent cache, so a step may switch between them without repacking anything. A multi-token absorbed step needs the causal mask that the materializing path gets from `causal_attention`: `deepseek_v3` and `deepseek_v32` run every multi-token step, absorbed or materializing, through `mla::rope_folded_attention`, which concatenates the rope half into the SDPA head dimension and applies causal mode or the caller's mask (#3071); the shared `absorbed_attention` takes the same folded call for a multi-token step, and `deepseek_v2`'s `forward_absorbed` constructs the causal mask when the caller passed none, with a caller-supplied padded mask taking precedence. The materializing side still keeps the original `kv_b_proj` loaded and re-up-projects the rows a previous chunk already handled, because the cache no longer holds them decompressed.

## Stage 2: split-KV

Absorbed decode is one query row per head against the whole cached window, so parallelism is bounded by `batch * num_heads` and adding context adds none. Stage 2 cuts the latent range into chunks whose partial softmax states are combined afterwards, the same observation issue #898 made about paged decode.

The merge is issue #898's `paged_attention_merge_states`, reused with no change to the kernel, the FFI signature, or the C++ launcher. Its contract is deliberately paging-agnostic: `v_in [N, H, D]` partials already divided by their own denominator, `lse_in [N, H]` in **log2** units, and `o_indptr [M + 1]` grouping. MLA fits it as `H = num_heads`, `D = kv_lora_rank`, `M = batch`, `N = batch * chunks`.

The log2 requirement is the one clause that fails silently. A natural-log LSE still merges and returns a plausible wrong weighted average, so the conversion is a named constant (`mla::split_kv::LOG2_E`) and `merge_rejects_natural_log_lse_units` is a negative control that asserts the natural-log variant does *not* match the reference.

The partial producer is currently composed from MLX ops. It is correct and it is the reference a fused partial kernel would be validated against, but it runs `C` small matmuls where Stage 1 runs one large one, so it is expected to be slower than Stage 1. Do not quote a Stage 2 throughput number before a fused partial kernel exists.

## Benchmark

```bash
export DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer
cargo run --release --features metal,accelerate --example mla_absorbed_decode_bench -- \
    --contexts 4096,16384,32768 --batches 1,4 --steps 64 --warmup 16
```

It measures one MLA attention block per decode step over three arms and prints the KV-memory table. Synthetic tensors, not a real checkpoint, so the numbers are the throughput of the layer that changed, not end-to-end model throughput.

Every row ends in a `paths=` field from `mlxcel_core::mla::stats`, taken and reset around the measured region, and the harness prints a `WARNING` on any row whose counters disagree with the arm's label. **Do not report a row whose `paths=` field does not name its own arm.** Issue #899 shipped a fused decode path that never activated and whose before/after benchmark compared the fallback against itself; that null looked clean and was nearly accepted.

Tests that assert which path a forward took read the counters with `mlxcel_core::mla::stats::scoped(|| ...)`, which returns only the calls made on the calling thread, not a `snapshot()` before and after: the global counters also count every MLA test the harness runs in parallel (#3298). Path selection happens while the graph is built, on the thread that calls `forward`, so a test's own forwards always land in its scope.

For a real checkpoint, `MLXCEL_MLA_ABSORBED=1 mlxcel generate -m models/<deepseek> ...` (the explicit on spelling; the default reports the same line through `tracing` at info level, `RUST_LOG=info`) prints one stderr line at load stating how many layers folded and the bytes/token before and after. A run whose line reads `0/27 layers` is running the fallback, and a decline is printed to stderr under the default as well. The line goes to stderr, not stdout, so it never mixes with the generated text, and it does not depend on `RUST_LOG`.

## Related

- `src/lib/mlxcel-core/src/mla/` for the implementation.
- `src/lib/mlxcel-core/src/paged_v2/` for the merge kernel and its contract.
- [`docs/turbo-kv-cache.md`](turbo-kv-cache.md) for the KV cache modes this path declines.
