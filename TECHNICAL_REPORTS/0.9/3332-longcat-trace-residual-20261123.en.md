# PR #3332: Trace the LongCat-Flash residual to mlx-lm's kv_b_proj requantization

**Date**: 2026-11-23
**Status**: Draft PR (closes #3318). Documentation and a script docstring only; real-checkpoint arms recorded below.
**Risk Level**: Low (no runtime code changes)

## Executive Summary

After #3306, the teacher-forced trace of `longcat-flash-lite-4bit` against mlx-lm still had 39 decided disagreements out of 20486 over 36864 positions. The residual comes from the two runtimes loading different weights. mlxcel's code does not drift. The checkpoint ships a fused 4-bit `kv_b_proj`. mlxcel splits it into dense `embed_q` / `unembed_out`, and mlx-lm quantizes both halves again. With the reference's split kept dense, the residual falls to 4. The dtype noise floor (the bf16 reference against the same reference cast to float16) is 5.

## 1. Problem Statement

#3295's first acceptance criterion required no decided disagreements beyond the reference's own noise. #3306 left 39, rising from 0 to 1 per 4096-position chunk below 12288 to 4 to 10 per chunk from 12288 on. GigaChat3, which uses the same `MlaRope`, traced at 0, so the residual looked LongCat-specific.

## 2. Change Summary

| Area | Change |
| --- | --- |
| `scripts/mlx_lm_logit_trace.py` | The docstring no longer claims both arms run the checkpoint dtype. mlxcel converts bf16 to f16 only for unquantized checkpoints on Apple Silicon. The docstring also says that a fused quantized `kv_b_proj` loads different weights in each runtime. |
| `docs/supported-models.md` | The LongCat entry states the cause instead of "not yet identified". |
| `docs/benchmark_results/longcat-flash-lite-trace-residual-m3ultra-2026-11-23.md` | Every arm, per chunk. |

## 3. Technical Decisions

### 3.1 Arms in the issue's order

1. `MLXCEL_KEEP_BF16=1` produced byte-identical rows. `bf16_to_f16_at_load` returns false for a quantized checkpoint, so this checkpoint was never converted. The issue's premise did not hold.
2. `MLXCEL_TRACE_KV_CACHE_MODE=fp16` also produced identical rows. The dense cache already stores bf16 activations, as mlx-lm's `KVCache` does.
3. `MLXCEL_TRACE_MLA_STATS=1` reported every step on the materializing path, so there was no absorbed path to force. A local experiment put mlx-lm's formulation back: the rope term as a bf16 additive mask over a fused 128-wide SDPA, instead of the folded 192-wide SDPA that MLX runs unfused. It gave 33.

### 3.2 Finding the source without per-layer bisection

A float32 mlx-lm reference agreed with the float16 one at 0 decided disagreements. It still showed mlxcel 28 apart, and mlxcel's per-chunk perplexity was 5% to 7% lower on chunks 1 to 3. A systematic gap of that size pointed to different weights. Comparing the two sanitizers showed the `kv_b_proj` requantization. A reference shim that keeps the split dense brought the count to 4 (bf16) and 2 (float32).

### 3.3 No change to mlxcel's split

mlxcel's dense halves are the dequantized checkpoint weights exactly. mlx-lm's second quantization (of `embed_q` along the transposed axis) loses precision. Matching it would make mlxcel less faithful to the checkpoint.

## 4. Verification

| Reference | Candidate | Decided | Per chunk |
| --- | --- | --- | --- |
| bf16 stock | mlxcel main | 39 / 20486 | 0 1 0 4 10 4 6 8 6 |
| bf16 stock | float16 reference | 5 / 20486 | 0 0 0 1 1 0 0 1 2 |
| bf16 dense split | mlxcel main | 4 / 20489 | 0 0 0 2 1 0 0 0 1 |
| float32 dense split | mlxcel main | 2 / 20208 | 0 1 0 1 0 0 0 0 0 |
| bf16 dense split | bf16 stock | 32 / 20489 | 0 3 0 5 4 3 4 7 6 |

Contract checks pass: fmt, the `scripts/ci` scripts and the header check. The `coverage_tests` and `longcat` lib selectors also pass.

## 5. Unverified and Open Items

- The dense-split and cached-prefill reference is a local shim. The committed script still runs the stock split in one forward.
- The sparse variant loads the same kind of fused `kv_b_proj` from the HF layout. Whether its mlx-vlm reference requantizes was not checked here. Its trace was already at 0.

## 6. Learning Points

- When one arm is consistently more confident than an f32 oracle across content, compare what each runtime loads before bisecting arithmetic.
- A dtype-cast reference pair gives a real noise floor. The prefill-step pair in #3306 gave 0 and could not set one.

## Related Commits

- `cb8a03b5` docs: trace LongCat-Flash residual to mlx-lm's kv_b_proj requantization
