# Technical Report: PR #3168 - Store the absorbed MLA latent as affine int8 on DeepSeek V2

**Date**: 2026-11-19
**Status**: Completed
**Risk Level**: Medium (new cache layout behind an opt-in flag; the fp16 default path is unchanged)

## Executive Summary

On DeepSeek V2 with absorbed decode, `--kv-cache-mode int8` now stores the MLA latent as affine group-64 8-bit values with fp16 rope keys, 672 bytes per token per layer on V2-Lite, against 1152 for the fp16 latent and 5120 for the decompressed int8 cache it replaced. On the M3 Ultra it decodes 1.15x to 2.78x faster than the decompressed int8 cache and 13% to 36% slower than the fp16 latent, and its teacher-forced traces show 0 decided disagreements against fp16. int4 failed the same gate and is not shipped.

## 1. Problem Statement

Before #3155, a quantized cache request on an absorbing V2 model either fell back to the decompressed per-head layout (int8) or was substituted with the fp16 latent (Turbo and int4), because `MlaLatentCache` held only dense `(ckv, kpe)` tensors. Asking for int8 to save memory therefore produced a cache 4.4x larger than the fp16 latent default. Epic #3158 needs a quantized latent storage type and a window contract that its later phases (#3156 fused Metal kernel, #3157 Turbo WHT evaluation, #3160-#3163 latent-pair families) can consume.

## 2. What Changed

- `mlxcel-core/src/mla/quant_latent.rs` adds the quantized latent window, the composed one-query decode and the sizing rate `quantized_latent_bytes_per_token`, which saturates to `usize::MAX` for a rank whose byte count would overflow.
- `MlaLatentCache::update_and_fetch` returns `LatentWindow::{Dense, Quantized { ckv_q, scales, biases, kpe, bits, group_size }}`. The quantized variant reuses the four buffers of the per-head INT8 cache, so `trim`, `trim_front`, gather, detach/adopt and prompt-cache parking handle it without new bookkeeping.
- Decode: one query runs the latent score and the value sum through `quantized_matmul` on the packed rows, and the rope score puts the query heads on the matmul row axis. A multi-token step and a cold prefill dequantize the window once per step. The fused kernel and split-KV never see a quantized window.
- Resolution: Turbo modes on an absorbing V2 keep resolving to the fp16 latent with a warning that now names `int8`. `MLXCEL_MLA_ABSORBED=0` keeps the decompressed int8 cache. DeepSeek-VL2 resolves its KV rules through its `deepseek_v2` language config. `--kv-group-size` other than 64 fails server startup on such a model.
- Sizing: `kv_arch`, `inspect`, the memory preflights, the prompt-cache budget and the KV advisor use the 672-byte rate.
- Docs: the "MLA latent caches are FP16 only" section of `docs/turbo-kv-cache.md` is rewritten with the int8 decision, the int4 numbers and the latent-pair follow-ups. `docs/mla-absorbed-decode.md`, `docs/environment-variables.md` and `CHANGELOG.md` are updated, and `docs/benchmark_results/mla-quantized-latent-m3ultra-2026-11-19.md` records the measurements with per-run CSV files.

## 3. Technical Decisions

**Compose on `quantized_matmul` rather than dequantize per step.** A single decode query reads the window once. Dequantizing it first would write and re-read an fp16 copy of the whole window each step, which costs the bandwidth the format is meant to save. `quantized_matmul` and `dequantize` are backend-generic MLX ops, so the path compiles for CUDA and ROCm without Metal-only symbols. Multi-token steps already pay for a matmul with many query rows, so they dequantize once and reuse the dense attention code.

**Keep the window contract general.** `LatentWindow::Quantized` carries `bits` and `group_size` instead of assuming V2's 8 and 64, because #3156 and #3160-#3163 consume the same type. Decode reads the bit width from the window, and a unit test covers that.

**Ship int8, not int4.** The gate is 0 decided disagreements and every top-1 flip at a near tie (top-two gap of 0.11 logits or less). int8 meets it: largest gap 0.078 behind 1024 and 0.109 behind 4096. int4 had 45 of 512 and 14 of 256 top-1 flips with a largest gap of 0.859, so `KVCacheMode::Int4` and the `(Uniform, 4)` mapping were not added and `--kv-bits 4` keeps its startup error.

**Ship only the numerically identical decode layout.** The security and performance review measured three single-query layouts. Putting the heads on the row axis of the rope score is byte-identical in the trace and gains 1% to 1.5%, so it shipped. Folding the heads into the latent score gains 3% at 512 and 2048 and 13% at 16384, but it changes the summation order, and one top-1 flip landed at a 0.219-logit gap, outside the 0.11 bar. It is recorded and left for a decision on the bar. Folding the value sum as well was slower at every length (38.5 against 71 tok/s at 16384).

**Saturate the sizing rate.** `kv_arch` passes `kv_lora_rank` straight from `config.json`, and `rank * bits` had no overflow check: a crafted rank that is a multiple of 64 panicked in a debug build and wrapped to a small rate in release, undercounting `inspect`, the memory estimate and the prompt-cache budget. The rate now saturates, and runtime shape checks (`MlaGeometry::check`, `check_quantizable`) still reject such a rank before any allocation.

## 4. Measurements

M3 Ultra, Metal, deepseek-v2-lite-chat-4bit-mlx, `mlxcel-bench-decode`, 128 tokens after 16 warmup, median tok/s, arms interleaved under one GPU lock with indexers paused:

| prompt | fp16 latent | int8 latent | int8 decompressed | int8 latent vs decompressed |
|---|---|---|---|---|
| 512 | 163.59 | 138.85 | 120.97 | 1.15x |
| 2048 | 146.69 | 127.58 (129.5 to 130.1 after the rope-score change) | 88.50 | 1.44x |
| 16384 | 109.89 | 70.67 (71.3 after) | 25.41 | 2.78x |

Teacher-forced trace against the fp16 latent (`compare_logit_traces.py --decided 2`): int8 has 0 of 209 and 0 of 105 decided disagreements behind 1024 and 4096, top-1 flips 12 of 512 and 3 of 256, perplexity +0.29% and -0.63%. The trace after the rope-score change is byte-identical behind 1024 and matches behind 4096.

16384-token prefill transient peak: 11.23 GB for the int8 latent against 11.45 GB for fp16 and 16.95 GB for decompressed int8. A 16384-token cache is 0.30 GB against 2.29 GB decompressed.

Gate: `make verify-clippy` clean at 8a0137ab; `mlxcel-core` `mla::` and `cache::` tests and the `mlxcel` sizing, advisor, CLI, prompt-cache and `models::deepseek_v2` tests pass serially on Metal; `deepseek_v2_greedy_parity` on the real checkpoint confirms the fp16 default is unchanged.

## 5. Learning Points

**The broadcast head is where the cost goes.** V2's latent has one head, and `quantized_matmul` broadcasts it over the 16 query heads, so each head reads the whole packed window. That is why int8 decode runs behind the fp16 latent, whose fused kernel reads the window once for all heads. The measured fold shows the cost is real (+13% at 16384), and #3156's fused kernel is the place to remove it without changing the summation order the gate measured.

**A near-tie bar has to be stated before the measurement.** The latent-score fold improved perplexity (-0.20%) and had 0 decided disagreements, which reads as better on two of three metrics. The bar settled the decision because it was written in the issue before the run.

**Workspace clippy catches API drift that crate checks miss.** The `LatentWindow` return type broke `examples/mla_absorbed_decode_bench.rs`, which only the workspace `--all-targets` clippy compiles. Narrow crate checks passed.

## 6. Follow-up Actions

- #3156: fused Metal kernel for the quantized latent, where the one-read-for-all-heads layout belongs.
- #3157: Turbo WHT evaluation on the latent.
- #3160-#3163: quantized latent for the latent-pair families, which keep fp16 in this PR.
- Decide whether the 0.11 near-tie bar should admit the latent-score fold (+13% at 16384), and whether f32 scale sidecars (704 instead of 672 bytes per token per layer) are worth removing the per-step f32 cast.
- CUDA GB10 and ROCm gfx1151 rows are not measured; this host cannot reach them.
