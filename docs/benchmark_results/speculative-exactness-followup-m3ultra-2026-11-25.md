# Speculative exactness follow-up: Gemma 4 26B A4B and token-only verify rounds (M3 Ultra, 2026-11-25)

Issue #3418. The [survey](speculative-exactness-survey-m3ultra-2026-11-25.md) left two block-versus-chain mismatches untraced: the Gemma 4 26B A4B MoE checkpoint, which declined the MTP exactness probe on both quantized kernels, and the engine's token-only verify round (prompt lookup, a full-model drafter), which differed from single-token decode on every model it takes. This page records where each one starts, what was changed, and what it costs.

Host: Apple M3 Ultra, 512 GB, Metal. Every run is serialized under the shared GPU lock. Verdicts are byte comparisons from `models::speculative_exactness_survey_tests` (widths 2, 4 and 8; startup or 8 prompt tokens, 600 and 2100). `wide` is MLX's default quantized kernel selection, `narrow` is `qmv_wide` off, the MTP gate's own retry and the operator switch `MLXCEL_QMV_WIDE=0`.

## 1. Gemma 4 26B A4B

### Where it parts

`diag::capture` points inside the MoE feed-forward (`DecoderLayer::ffn_branch`: shared MLP, router indices, router weights, experts, combine) put the narrow-kernel divergence at **layer 0, `MoE experts`, row 0**. The shared MLP and both router outputs agree. `Experts::forward` picks its kernel by the token count: the fused GeGLU kernel for one token, the compiled `gather_qmm` graph below 64 routed pairs, the sorted gather from there. The three do not agree bit for bit.

With the experts fixed, the narrow divergence moved to the **attention output** of the batched block (layer 5 full attention at startup and at 600 tokens, layer 2 sliding attention at 2100 tokens, layer 11 or 1 at width 8). That is the cause the 31B and 12B geometries already correct with row-wise verify attention.

On the wide kernel the first divergence is the layer 0 attention output before and after, as on every quantized Gemma 4 checkpoint: `qmv_wide` itself.

### Change

- Inside an MTP verify block the routed experts run one position at a time (`Experts::forward_verify_block`), so each position takes the kernel its decode step takes. A block of K tokens reads K sets of routed experts either way, so the split costs kernel launches, not weight reads. `MLXCEL_GEMMA4_MOE_VERIFY_PER_POSITION=0` restores the block-wide product.
- The 26B A4B geometry (16 heads, 8 KV heads, 2 global KV heads, head dims 256 and 512) joins the row-wise verify set (`TextConfig::mtp_requires_linear_singleton`) on the active Metal path. It was not measured on CUDA, so CUDA keeps the old policy for this geometry.

### Verdicts

| Kernel | Before | Experts per position only | Experts per position and row-wise attention |
|---|---|---|---|
| wide | differs, layer 0 attention output | differs, layer 0 attention output | differs, layer 0 attention output |
| narrow | differs, layer 0 `MoE experts` (41% to 77% of logit bytes) | differs, attention output of layer 1 to 11 | **equal** at widths 2, 4, 8, at startup, 600 and 2100 tokens |

The startup plan of a row-wise geometry includes a long draw past the 1024-token window, so the `startup` row covers it. The gate now behaves as it does on the 31B and 12B checkpoints: the wide probe diverges, the retry with `qmv_wide` off is equal, and MTP engages pinned narrow.

Not measured: MTP throughput and an end-to-end text comparison on this checkpoint. No drafter for the 26B A4B target is on disk (the 12B and 31B assistants do not pair with it), so whether the drafter gains more than row-wise verify costs is open. The 31B and 12B pairs, which run the same verify path, are in the survey.

### Decode

Classic decode does not enter the verify path, so the change cannot move it; the table confirms that. `mlxcel generate --temp 0 --profile -n 128`, one locked block, ABAB, 8 pairs, load average 1.4 to 1.5. Arm A is the default, arm B sets `MLXCEL_GEMMA4_MOE_VERIFY_PER_POSITION=0`. Both arms produce the same 128 tokens.

| Arm | Decode tok/s (8 runs) | Mean |
|---|---|---|
| A, per-position experts in verify blocks (default) | 132.53, 132.93, 132.70, 133.24, 132.51, 132.27, 132.59, 132.41 | 132.65 |
| B, block-wide experts | 133.43, 133.14, 132.95, 132.79, 132.73, 131.85, 132.81, 133.51 | 132.90 |

The means differ by 0.2%, inside the run-to-run spread of either arm. No decode regression.

## 2. Token-only verify rounds

### Where it parts

On the DeepSeek V3 family the survey's diagnostic put the first divergence at the layer 0 output, row 0. Three causes sit inside that layer, and one more outside it:

| Cause | Where |
|---|---|
| A single query takes the fused absorbed MLA kernel (or its composed fallback); a block takes rope-folded SDPA, absorbed or materialized by the `use_absorbed_mla` cost model. `embed_q` and `unembed_out` run as per-head batched products whose kernel depends on the row count | `DeepSeekV3Attention::forward` |
| gemv for one row, gemm for several, on unquantized checkpoints: q, kv_a, o, the dense MLP, the shared experts, the MoE gate, the LM head | every dense projection |
| Float routed experts multiply through `gather_mm`, which switches kernel with the token count | `MoEBlock::forward` on bf16 checkpoints |
| Quantized projections with two or more rows take `qmv_wide` | MLX, every quantized checkpoint |

### Change

`MLXCEL_MLA_ROW_INVARIANT=1` (off by default, see the cost below) makes a single-sequence forward of `DeepSeekV3Model` (deepseek_v3, kimi_k2, joyai_llm_flash, kimi_vl, kimi_k25, youtu) and `Xing40Model` enter a `mlxcel_core::row_invariant` scope:

- A verify block (speculative append scope, one sequence, 2 to 16 rows, no caller mask) runs every row of its MLA attention as the single query its decode step runs, over the prefix of the stored latent that step would see. Both a dense and a quantized latent window are sliced.
- Dense projections take the row-invariant matvec kernel for up to 16 rows, in decode steps and verify blocks alike. Float experts run one position at a time inside a verify block.
- Quantized projections are left as one product (`row_invariant::linear_whole_block`). Splitting them would read each weight once per row; they are row-invariant with `qmv_wide` off.
- Xing4.0's hyper connection takes the row-invariant product for its mixer and multiplies `hc_expand` per position inside a verify block.
- GLM-4.7-Flash routes a verify block of the token-only round to `Glm4MoeLiteModel::forward_verify`, the row-wise forward its MTP gate already probes.

### Verdicts with the variable on

Every cell is all three widths at 8, 600 and 2100 prompt tokens (9 comparisons).

| Checkpoint | Family, dtype | Before, wide | Before, narrow | After, wide | After, narrow |
|---|---|---|---|---|---|
| gigachat3-10b-a1.8b-bf16 | DeepSeek V3, bf16 | differs | same (no quantized product) | **equal** | not needed |
| gigachat3-10b-a1.8b-mlx-8bit | DeepSeek V3, affine 8-bit | differs | differs | differs | **equal** |
| Kimi-VL-A3B-Thinking-4bit | DeepSeek V3 text, 4-bit | differs | differs | differs | **equal** |
| moonlight-16b-a3b-instruct-4bit | DeepSeek V3, 4-bit | differs | differs | differs | **equal** |
| xing4.0-29b-a4b-mlx-8bit | xing4_0, affine 8-bit | differs | differs | differs | **equal** |
| xing4.0-29b-a4b-optiq-4bit | xing4_0, mixed 4/8-bit | differs | differs | differs | **equal** |
| glm-4.7-flash-4bit (engine path) | GLM-4.7-Flash, 4-bit | differs | differs | differs | **equal** |

The remaining wide-kernel difference on quantized checkpoints is `qmv_wide`: turning that one switch off is what makes the row equal. So a quantized checkpoint needs `MLXCEL_MLA_ROW_INVARIANT=1` together with `MLXCEL_QMV_WIDE=0`, and a bf16 checkpoint needs only the first.

Xing4.0 needed one more product. With only the DeepSeek V3 layers in the scope, the optiq checkpoint was equal in 8 of 9 cells and differed at width 8 after 2100 tokens, first at layer 38, row 6. The cause is the hyper connection: its mixer is a dense `matmul` over `[1, L, hc * D]` (gemv for one row, gemm for several) and `hc_expand` is a batched matmul over the positions. The mixer now goes through `row_invariant::dense` and `hc_expand` multiplies each position alone inside a verify block; all 9 cells are equal.

### End to end

`mlxcel generate --temp 0 --prompt-lookup` against plain decode, generated text compared. The copy prompt asks for a 16-line Python function to be reprinted with one variable renamed; the long prompt is the survey's (bytes 4000 to 14000 of `docs/supported-models.md` plus a summary request). Quantized checkpoints were compared with `MLXCEL_QMV_WIDE=0` in both arms, since a prefill is a multi-row product too.

| Checkpoint | Prompt | Variable off | Variable on |
|---|---|---|---|
| gigachat3 bf16 | copy, long | identical, identical | identical, identical |
| gigachat3 8-bit, narrow | copy | identical | identical |
| gigachat3 8-bit, narrow | long | **differs** (108 tokens against 63) | identical |
| Kimi-VL 4-bit, narrow | copy, long | identical, identical | identical, identical |
| moonlight 4-bit, narrow | copy | identical | identical |
| moonlight 4-bit, narrow | long | **differs** | identical |

With the variable on and the wide kernel left on, gigachat3 8-bit still left plain decode on the long prompt (84 tokens against 63); Kimi-VL and moonlight matched on both prompts.

### Cost

Decode, `mlxcel generate --temp 0 --profile -n 256`, one locked block per checkpoint, 8 cycles, load average 1.4 rising to 8.3 on the bf16 block and 8.3 falling to 6.7 on the 8-bit block (another build started on the host). Every arm produced the same 256 tokens.

| Checkpoint | Arm | Decode tok/s (8 runs) | Mean |
|---|---|---|---|
| gigachat3 bf16 | variable on (matvec kernel) | 117.46, 117.22, 118.10, 115.60, 114.85, 118.98, 115.37, 115.08 | 116.58 |
| gigachat3 bf16 | variable off (MLX `matmul`) | 114.81, 116.47, 116.57, 112.88, 113.30, 111.34, 113.51, 113.26 | 114.02 |
| gigachat3 8-bit | variable on | 149.69, 150.25, 151.89, 150.19, 151.23, 147.78, 147.36, 147.38 | 149.47 |
| gigachat3 8-bit | variable off | 148.45, 149.64, 149.83, 151.20, 151.09, 145.01, 145.51, 145.06 | 148.22 |

Decode does not regress: the kernel arm is 2.2% faster on bf16 and ahead in all 8 adjacent pairs, and the 8-bit arms are within 0.8%.

Prompt lookup on the copy prompt (130 generated tokens, 27 rounds, 4.78 tokens per target forward, the same in every arm), one locked block per checkpoint, 4 cycles, load average 1.6 to 5.6:

| Checkpoint | Arm | Decode tok/s (4 runs) | Mean | Against plain decode |
|---|---|---|---|---|
| gigachat3 bf16 | plain decode | 114.96, 114.80, 114.76, 114.66 | 114.79 | 1.00x |
| gigachat3 bf16 | prompt lookup, variable off | 261.01, 262.31, 261.26, 262.24 | 261.70 | 2.28x |
| gigachat3 bf16 | prompt lookup, variable on | 147.62, 149.73, 147.79, 148.83 | 148.49 | 1.29x |
| gigachat3 8-bit | plain decode | 143.55, 145.37, 143.16, 144.20 | 144.07 | 1.00x |
| gigachat3 8-bit | prompt lookup, variable off | 343.30, 344.21, 345.29, 342.27 | 343.77 | 2.39x |
| gigachat3 8-bit | prompt lookup, variable on | 261.78, 259.64, 262.92, 259.03 | 260.84 | 1.81x |
| gigachat3 8-bit | prompt lookup, variable off, `MLXCEL_QMV_WIDE=0` | 320.15, 320.81, 321.19, 319.08 | 320.31 | 2.22x |
| gigachat3 8-bit | prompt lookup, variable on, `MLXCEL_QMV_WIDE=0` (byte-identical) | 246.64, 247.47, 247.03, 246.71 | 246.96 | 1.71x |

On the long prompt, where almost no proposal lands (single runs): gigachat3 bf16 runs plain decode at 101.6 tok/s, prompt lookup at 89.6 with the variable off and 79.4 with it on.

The exact block costs prompt lookup 43% of its throughput on bf16 and 24% on 8-bit (28% once `qmv_wide` is also off). On bf16 the per-row MLA, the 8-row matvec products and the per-position float experts all contribute; on 8-bit the per-row MLA is the only product of any size that changed. A third arm that kept MLX `matmul` for decode and multiplied a verify block's dense products one row at a time ran prompt lookup at 113.6 tok/s on bf16, below plain decode, and was dropped.

### Decision per family

| Family | Decision |
|---|---|
| Gemma 4 26B A4B (MTP gate) | Fixed and on by default. The gate needs exactness to engage at all, and decode is unchanged. |
| DeepSeek V3 family (`DeepSeekV3Model`) | Exact verify block available, off by default. Byte identity costs prompt lookup 24% to 43% of its throughput and the round's contract does not promise it. `MLXCEL_MLA_ROW_INVARIANT=1` turns it on; quantized checkpoints also need `MLXCEL_QMV_WIDE=0`. |
| Xing4.0 | Same switch, same layers, plus the hyper connection's two products. Equal on the 8-bit and the mixed-width optiq checkpoint. |
| GLM-4.7-Flash engine path | Same switch: the token-only block takes the MTP gate's row-wise verify forward. Equal on narrow. |
| DeepSeek V3.2, Kimi Linear, Kimi K3, LongCat | Not changed. They carry the same `use_absorbed_mla` split in their own attention code. `deepseek-v3.2-4bit` is over this run's memory limit, and the others were not surveyed. |
| Arcee (`afm-4.5b`, bf16) | Kept as documented. First divergence: layer 0 output (rows 0 to 3 depending on the draw; layer 1 in one cell). The products are the Llama-class dense projections; making them row-invariant means moving that shared decode path onto the matvec kernel. |
| K2 Horizon (`k2-horizon-0.9b`, bf16, `Llama3Model`) | Kept as documented. First divergence: layer 0 output, row 0, in all 9 cells. Same reason. |
| OLMoE (`olmoe-1b-7b-0125-4bit`) | Kept as documented. First divergence: layer 0 output, row 0, on both kernels. |

No gate was added: prompt lookup and the full-model drafter run on every model they ran on before, with or without the variable.

## Commands

```bash
# Gemma 4 26B A4B, family probe with the MoE stages traced
/usr/bin/lockf -k /Users/devops/Development/.mlxcel-gpu.lock \
  env RUST_MIN_STACK=16777216 EXACTNESS_SURVEY_MODEL=models/gemma-4-26b-a4b-it-4bit \
  EXACTNESS_SURVEY_PATH=family EXACTNESS_SURVEY_DIAG=1 \
  cargo test --profile test-fast --features metal,accelerate --lib \
  models::speculative_exactness_survey_tests -- --ignored --nocapture

# Token-only round, exact block on
/usr/bin/lockf -k /Users/devops/Development/.mlxcel-gpu.lock \
  env RUST_MIN_STACK=16777216 MLXCEL_MLA_ROW_INVARIANT=1 \
  EXACTNESS_SURVEY_MODEL=models/gigachat3-10b-a1.8b-mlx-8bit \
  EXACTNESS_SURVEY_PATH=engine EXACTNESS_SURVEY_DIAG=1 \
  cargo test --profile test-fast --features metal,accelerate --lib \
  models::speculative_exactness_survey_tests -- --ignored --nocapture

# Byte-identical prompt lookup on a quantized DeepSeek V3 checkpoint
MLXCEL_MLA_ROW_INVARIANT=1 MLXCEL_QMV_WIDE=0 \
  mlxcel generate -m models/gigachat3-10b-a1.8b-mlx-8bit --temp 0 --prompt-lookup -p "..."
```

The decode and prompt-lookup tables, and the survey verdicts, were first taken on a build in which the row-invariant scope was on by default and `MLXCEL_DEEPSEEK_V3_ROW_INVARIANT=0` turned it off (`verify` selected the dropped third arm); the variable was then renamed and its default inverted without touching either code path. On the final build the checks were repeated in part: the survey gave the same verdicts for Gemma 4 26B A4B (all 18 rows), gigachat3 bf16 with the variable on (9 cells equal) and off (width 4, differs), gigachat3 8-bit with it on (widths 2 and 8) and GLM-4.7-Flash with it on (width 4); prompt lookup on gigachat3 bf16 ran at 264.9 tok/s by default and 149.8 with the variable on (2 cycles, another build running); and on the long prompt gigachat3 8-bit again left plain decode by default and matched it with the variable on, both with `MLXCEL_QMV_WIDE=0`. The Xing4.0 rows are from the final build, after the hyper-connection change (9 of 9 cells on each checkpoint with the variable on; the 8-bit checkpoint differs on both kernels with it off).
