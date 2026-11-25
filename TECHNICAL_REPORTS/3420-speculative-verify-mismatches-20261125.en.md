# PR #3420: Tracing the last two speculative verify mismatches

**Date**: 2026-11-25
**Status**: Draft PR open (closes #3418). Validated on an M3 Ultra (512 GB) with `gemma-4-26b-a4b-it-4bit`, the gigachat3 bf16 and 8-bit checkpoints, Kimi-VL, moonlight, Xing4.0 and GLM-4.7-Flash; narrow lib suites, clippy and the contract checks pass

## Executive Summary

The exactness survey of PR #3415 left two places where a K-row speculative verify block is not the K single-token decode steps it stands for: the Gemma 4 26B A4B MoE checkpoint, which the MTP probe declined on both quantized kernels, and the engine's token-only verify round (prompt lookup, a full-model drafter), which differed on every model it takes.

Both are traced. Gemma 4 26B A4B is fixed and passes the probe on the narrow kernel by default. The token-only round has an exact block on the DeepSeek V3 family, Xing4.0 and the GLM-4.7-Flash engine path, but it is opt-in (`MLXCEL_MLA_ROW_INVARIANT=1`): byte identity costs prompt lookup 24% to 43% of its throughput, and the round's contract never promised it. Arcee, K2 Horizon and OLMoE are located and left as documented.

## 1. Gemma 4 26B A4B

New capture points inside the MoE feed-forward put the narrow-kernel divergence at layer 0, `MoE experts`, row 0. The shared MLP and both router outputs agree. `Experts::forward` picks its kernel by the token count: the fused GeGLU kernel for one token, a compiled `gather_qmm` graph below 64 routed pairs, a sorted gather from there.

`Experts::forward_verify_block` runs the experts one position at a time inside an MTP verify block. A K-token block reads K sets of routed experts either way, so this costs kernel launches and no extra weight reads.

That alone was not enough. With the experts fixed, the divergence moved to the attention output of the batched block (layers 1 to 11 depending on the draw). The 31B and 12B geometries already correct this with row-wise verify attention, so the 26B A4B geometry (16 heads, 8 KV heads, 2 global KV heads) joins that set on Metal. CUDA keeps its old policy for this geometry because it was not measured there.

| Kernel | Before | Experts only | Experts and row-wise attention |
|---|---|---|---|
| wide | differs, layer 0 attention | differs | differs (`qmv_wide`, as on every quantized Gemma 4) |
| narrow | differs, layer 0 `MoE experts` | differs, attention output | equal at widths 2, 4, 8, at startup, 600 and 2100 tokens |

Classic decode does not enter the verify path: 132.65 tok/s with the change against 132.90 without (8 ABAB pairs).

No 26B A4B drafter is on disk, so MTP throughput on this target is not measured.

## 2. Token-only verify rounds

On the DeepSeek V3 family a single query takes the fused absorbed MLA kernel and a block takes rope-folded SDPA, absorbed or materialized by the `use_absorbed_mla` cost model. On bf16 checkpoints the dense projections (gemv against gemm) and the float experts (`gather_mm`) differ too. On quantized checkpoints MLX's `qmv_wide` does.

With `MLXCEL_MLA_ROW_INVARIANT=1`, `DeepSeekV3Model` and `Xing40Model` enter a `row_invariant` scope. `DeepSeekV3Attention::forward` now has one `single_query_attention` helper, used by a decode step and, inside a verify block, by every row over the prefix of the stored latent that row's decode step would see. Dense projections take the row-invariant matvec kernel; float experts run per position. Quantized projections stay one product (`row_invariant::linear_whole_block`), because splitting them reads each weight once per row and they are already row-invariant with `qmv_wide` off. GLM-4.7-Flash routes the block to the row-wise verify forward its MTP gate already probes.

| Checkpoint | With the variable, wide | With the variable, narrow |
|---|---|---|
| gigachat3 bf16 | equal | not needed |
| gigachat3 8-bit, Kimi-VL 4-bit, moonlight 4-bit | differs | equal |
| xing4.0 8-bit and optiq 4-bit, GLM-4.7-Flash engine path | differs | equal |

Xing4.0 needed its hyper connection too: with only the shared layers in the scope, one optiq cell (width 8, 2100 tokens) still differed at layer 38. The mixer is a dense `matmul` and `hc_expand` a batched matmul over the positions; a review of the diff pointed at them, and both are now row-invariant inside the scope.

All cells are widths 2, 4, 8 at 8, 600 and 2100 prompt tokens. Without the variable every cell differs.

End to end, `--prompt-lookup` text against plain decode on the narrow kernel: six of six checkpoint and prompt pairs identical with the variable, four of six without (gigachat3 8-bit and moonlight left plain decode on a 2.5K-token prompt).

## 3. Why it is opt-in

Prompt lookup on a reply that copies its prompt, 4 ABAB cycles:

| Checkpoint | Plain decode | Lookup, off | Lookup, on |
|---|---|---|---|
| gigachat3 bf16 | 114.79 | 261.70 | 148.49 |
| gigachat3 8-bit | 144.07 | 343.77 | 260.84 |
| gigachat3 8-bit, `MLXCEL_QMV_WIDE=0` | 143.91 | 320.31 | 246.96 |

Plain decode does not regress (bf16 116.58 against 114.02 tok/s with the kernel, 8-bit 149.47 against 148.22), but the verify round does: K fused-kernel launches per layer in place of one SDPA, 8-row matvec products and per-position experts on bf16. Turning the exact block on by default would take a third to a half of prompt lookup's gain for a property its contract does not state, and on a quantized checkpoint the default kernel would still not be byte-identical. The issue asks for exactness "if the cost is acceptable"; this cost is recorded and the default is unchanged.

A variant that kept MLX `matmul` for decode and multiplied the block's dense products row by row ran prompt lookup below plain decode (113.6 tok/s) and was dropped.

## 4. Tests

- `gemma4::moe_verify_tests`: quantized expert planes, widths 2, 4, 8, 12; the verify-block experts equal single-token decode byte for byte, and the block-wide product does not (the pre-change behavior).
- `deepseek_v3::verify_block_tests`: a tiny model, widths 2, 4, 8, 12, with and without the matvec kernel, at a rope width that takes the fused MLA kernel and one that takes the composed form; the block equals the chain, and without the scope it does not. A second test counts the MLA paths to confirm every block row takes the single-query path.
- `a_token_only_verify_block_takes_the_row_wise_verify_forward` (GLM): the plain forward inside the append scope is the row-wise verify forward when the mode is on and the classic forward otherwise.
- `linear_singleton_geometry_and_long_probe_are_limited_to_measured_backends` now pins the 26B A4B geometry to Metal.

## 5. Not done

- MTP throughput on Gemma 4 26B A4B (no drafter on disk).
- DeepSeek V3.2, Kimi Linear, Kimi K3 and LongCat carry the same MLA split in their own attention code and are unchanged.
- A multi-query form of the fused MLA kernel is the likely way to cut the 8-bit cost of the exact block. It is not attempted here, and its effect is not measured.
