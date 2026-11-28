# Technical Report: PR #3173 - Read the int8 MLA latent inside the fused Metal decode kernel

**Date**: 2026-11-19
**Status**: Completed
**Risk Level**: Low (Metal only, int8 latent only, windows up to 2048 keys; every other case keeps the #3155 composed path)

## Executive Summary

The fused absorbed-decode Metal kernel now has a second variant that reads the affine int8 MLA latent from #3155 directly, without dequantizing the window. On the M3 Ultra, DeepSeek-V2-Lite int8 decode on it is 5.9% faster than the composed int8 path behind 512 tokens and 1.5% faster behind 1792. Past about 2048 keys the composed path is faster, so the kernel is limited to 2048 keys by `FUSED_DECODE_QUANTIZED_MAX_KEYS`. Teacher-forced traces show 0 decided disagreements against both the composed int8 path and the fp16 latent.

## 1. Problem Statement

#3155 stored the V2 latent as int8 and decoded it through `quantized_matmul`, which broadcasts the single latent head over all 16 query heads, so each head reads the whole packed window. The fp16 latent avoids that through the fused kernel (#3121), which reads the window once for all heads, but `fused_absorbed_decode` declined quantized windows. Behind 512 tokens the int8 latent decoded 15% slower than fp16.

## 2. What Changed

- `src/lib/mlxcelverse/kernels/mla_absorbed_decode.{cpp,h}`: a second instantiation of the existing kernel template. Only the latent load differs: it reads the packed codes as bytes and factors each group's scale and bias out of the score and the value sum. The rope half, softmax, value sum and merge are shared. The dense kernel source is unchanged character for character.
- Bridge: `mla_absorbed_decode_fused_quantized` and its `_available` check in `mlx_cxx_bridge.{cpp,h}` and the `#[cxx::bridge]` block in `mlxcel-core/src/lib.rs`. A kernel refusal comes back as an error and never aborts the step.
- `mla/decode.rs`: `fused_absorbed_decode_quantized` accepts a single unmasked query, int8, group size 64, a power-of-two rank from 64 to 1024 and 1 to 2048 keys, and records `AbsorbedFused`. The #3155 quantized decode calls it first and falls back to the composed path on `None`. `MLXCEL_MLA_FUSED_DECODE=0` disables both fused variants.
- Tests: `mla/fused_quant_decode_tests.rs` covers fused-vs-composed parity at 1, 255, 2048 and 8192 keys in f32, fp16 and bf16, plus refusals (4-bit, group size 32, rank 192, over the limit, two queries).
- Docs: `docs/mla-absorbed-decode.md`, `docs/environment-variables.md`, CHANGELOG, and `docs/benchmark_results/mla-fused-quantized-decode-m3ultra-2026-11-19.md` with four CSVs.

## 3. Technical Decisions

**Factor scale and bias out of the sums instead of dequantizing each element.** For an affine group, `sum(q_i * (s * c_i + b)) = s * sum(q_i * c_i) + b * sum(q_i)`. The kernel accumulates the integer-code dot product and the group sum of the query once per group, then applies scale and bias. That keeps the per-element work at one byte load and one multiply-add, and the window is never materialized in fp16.

**Set the window limit from the in-model measurement, not the isolated kernel.** In isolation the kernel stays ahead of the composed path up to about 6000 keys. Inside the model, decode measured +6.5% behind 512, +0.4% behind 2048 and -4.6% behind 3840, so the limit is 2048. The isolated benchmark does not include the per-step quantize-and-store work or the rest of the layer, and the cause of the gap between the two crossovers was not isolated further.

**Keep the bit-width dispatch but refuse 4.** #3155 shipped int8 only, so the kernel takes a `bits` parameter and refuses anything but 8. A future 4-bit latent needs only a new unpack path and its own measurement.

## 4. Measurements

M3 Ultra, Metal, deepseek-v2-lite-chat-4bit-mlx, 128 tokens, median tok/s, three arms interleaved in one GPU lock hold:

| prompt | int8 fused (new) | fp16 fused | int8 composed (#3155) | new vs composed |
|---|---|---|---|---|
| 512 | 148.43 | 164.64 | 140.16 | +5.9% |
| 1792 | 132.82 | 149.43 | 130.90 | +1.5% |
| 2048 | 128.82 (composed, window over 2048) | 146.72 | 129.14 | -0.2% |
| 16384 | 71.08 (composed) | 109.65 | 71.12 | -0.1% |

Trace with `--decided 2` and the fused int8 arm as candidate: against composed int8, 0 of 205 decided behind 1024 (6 of 512 top-1 flips, largest gap 0.078) and 0 of 113 behind 1920 (3 of 256, 0.109). Against the fp16 latent, 0 of 209 behind 1024 (10 of 512, 0.078) and 0 of 115 behind 1920 (5 of 256, 0.078). Behind 4096 both arms take the composed path and the traces are byte-identical. The fp16 traces at 1024 and 4096 are byte-identical to a build of the base commit.

Parity, worst relative error fused vs composed: 5e-7 in f32, 5.6e-4 in fp16, 6.25e-3 in bf16, against bounds of 1e-5, 2e-3 and 1.6e-2.

## 5. Learning Points

**An isolated kernel crossover overstates the in-model one.** The isolated sweep put the crossover near 6000 keys; the model put it near 2048. The cause was not isolated, so the constant comes from the decode benchmark, which measures what users run.

**The remaining gap to fp16 is not in attention.** At 512 tokens int8 fused is 148 tok/s against 165 for fp16, with the attention kernels comparable. The difference is the quantize-and-store step that #3155 adds on every write.

## 6. Follow-up Actions

- `MLXCEL_MLA_FUSED_DECODE=0` with int8 alone selects the decompressed path on Metal, not `absorbed_composed`, because of the #3131 default rule; reaching composed also needs `MLXCEL_MLA_ABSORBED=1`. The rule is unchanged here.
- The fp16 kernel's isolated crossover on M3 Ultra looks like about 4096 keys against the 8192 measured on M5 Max. It was not measured in the model, and `FUSED_DECODE_MAX_KEYS` is unchanged.
- #3160-#3163 bring the int8 latent to the latent-pair families, which can then use this kernel through the same seam.
- CUDA and ROCm have no fused kernel for either latent and were not measured.
