# Technical Report: PR #3129 - Quantized folds and a fused kernel for absorbed MLA decode

**Date**: 2026-11-16
**Status**: Completed (always-absorbed families covered by unit tests only; the DeepSeek V2 default switch is left for a separate change)
**Risk Level**: Medium (new Metal kernel on the decode hot path of every absorbed MLA model)

## Executive Summary

Absorbed MLA decode had a fixed per-step cost that kept DeepSeek V2's latent layout about 10% behind the decompressed layout at short context (#3102). Profiling at 512 keys attributed it to three parts:

- dense f16 `W_UK`/`W_UV` folds: 0.36 ms;
- a rope-score mask: 0.17 ms;
- an unfused, latency-bound 512-wide attention block: 0.93 ms.

Two changes address it (#3121):

- A quantized `kv_b_proj` now folds as batched `quantized_matmul` on its own packed planes: 30.4 MiB instead of about 113 MiB on V2-Lite.
- Unmasked single-query absorbed attention runs on a new fused Metal kernel for windows up to 8192 keys.

On deepseek-v2-lite, absorbed decode at pp512/tg128 goes from 182.6 to 201.4 tok/s. That puts it 2.0% behind the decompressed default (205.5), against 11% before. It is 6.2% ahead at 2048 tokens and 56% ahead at 16384. The teacher-forced trace shows 0 decided-position disagreements.

## 3. Technical Decisions

- **Two key halves plus a merge.** At decode the query is a single row, so one threadgroup per head would leave most of the GPU idle. Splitting the keys into halves doubles the parallelism, and the merge combines the two online-softmax states exactly.
- **8192-key limit.** That is where the fused and composed blocks crossed over in the prototype measurement. Longer windows, masked steps, non-Metal backends and latent widths that are not multiples of 32 keep the composed path.
- **Fallbacks.** A dense fold remains for an unquantized layer, an NVFP4 global scale, runtime LoRA terms, or a quantization mode mismatch. `MLXCEL_MLA_FUSED_DECODE=0` disables the kernel. A new `absorbed_fused` stats path makes every trace and benchmark show which path ran.
- **No default switch here.** Both of the conditions `docs/mla-absorbed-decode.md` sets for switching DeepSeek V2's default to absorbed now hold. Switching is a behaviour change for every V2 user, so it belongs in its own PR.

## 7. Change Summary

| prompt | main absorbed | branch absorbed | branch fold only | decompressed |
|---|---|---|---|---|
| 512 (median of 4) | 182.6 | 201.4 | 187.8 | 205.5 |
| 2048 | 148-161 | 180.7 | | 170.1 |
| 16384 | 113.5 | 114.4 | | 73.5 |

The 16384 row runs the composed path on both arms, because its window is over the kernel limit.

| trace, width 1 | candidate | decided disagree |
|---|---|---|
| behind 1024 | fused | 0 / 209 |
| behind 4096 | fused | 0 / 106 (all 256 positions agree) |
| behind 8192 | composed (over the limit) | 0 / 78 |

Tests:
- New `mla::fused_decode_tests` check the kernel against a host f64 reference, and fail on Apple Silicon if the kernel is not live.
- Quantized-fold cases were added to `absorb_tests`.
- The `mlxcel-core` lib suite (2,328) and the full `mlxcel` lib suite (11,986) pass, and workspace clippy is clean.

## 8. Follow-up Actions

- Switch DeepSeek V2's default to the absorbed layout, citing these measurements.
- Measure an always-absorbed family end to end (Kimi-VL, whose unquantized `kv_b_proj` exercises only the kernel).
