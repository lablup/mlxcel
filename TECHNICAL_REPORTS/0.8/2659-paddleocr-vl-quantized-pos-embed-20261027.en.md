# PR #2659: Dequantize the PaddleOCR-VL vision position table

**Date**: 2026-10-27
**Status**: Complete on Metal (M1 Ultra) for PaddleOCR-VL 4-bit and bf16
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

PR #2659 (issue #2657) makes `mlx-community/PaddleOCR-VL-4bit` usable. That export quantizes the vision tower's learned position table, and `PositionEmbedding` loaded it as a dense table, so the interpolated position embedding was 144 packed words wide instead of 1152 features. Adding it to the patch embeddings aborted with `[broadcast_shapes] Shapes (2664,1152) and (2664,144)` in both the CLI and the server. The table is now a `UnifiedEmbedding`, the same fix #2459 applied to Qwen3-VL. The 4-bit checkpoint now produces the same OCR text as bf16, and bf16 output is byte-identical to the base.

## 1. Problem Statement

In the 4-bit export, `visual.embeddings.position_embedding.weight` is U32 `[729, 144]` with `.scales` and `.biases` of `[729, 18]` (group size 64, 4 bits; 1152 / 8 = 144). `PositionEmbedding::from_weights` in `src/vision/encoders/paddleocr_vl.rs` copied `.weight` and took the embedding width from `shape[1]`. `interpolate` then gathered packed rows, and the residual add onto `[N, 1152]` patch embeddings failed.

The developer read the vision tower and projector headers to rule out other cases of the same defect. `patch_embedding`, `fc2`, the layer norms and the projector `pre_norm` are BF16 in the export. `qkv`, `out_proj`, `fc1` and projector `linear_1`/`linear_2` are quantized but already load through `UnifiedLinear`. The position table was the only quantized tensor read as dense.

## 2. Change Summary

| Area | Change |
|---|---|
| `PositionEmbedding` | Holds a `UnifiedEmbedding` and takes the vision quantization `group_size`/`bits`. `interpolate` gathers rows through `UnifiedEmbedding::forward`, which dequantizes a quantized table and is a plain `take` on a dense one. The `embed` field and its no-op reshape are removed. |
| Load-time validation | The table must be 2-D with a square, positive row count. Added after the security review (see section 3). |
| Tests | `paddleocr_vl_pos_embed_tests.rs`: a quantized table interpolates to the dequantized dense result (failed on base with `[60, 16]` vs `[60, 128]`), a bf16 table matches the old row gather bit for bit, and 15- and 17-row tables are rejected at load. |

## 3. Technical Decisions

- **Drop the stored width instead of reading `config.hidden_size`.** The issue suggested taking the width from the config. Once lookups go through `UnifiedEmbedding`, nothing reads the width, and the old reshape to `[h_in * w_in, embed]` did nothing to a 2-D table. Removing the field avoids keeping a value that is wrong for one of the two layouts.
- **Validate the grid shape at load.** The removed reshape was the only thing that would have rejected a table whose row count is not a perfect square. `interpolate` computes indices from `sqrt(num_positions)` and gathers them unchecked, so a malformed checkpoint could read past the last row. The security checker flagged this as MEDIUM, and the load-time check restores a clear error.
- **Dequantize on each lookup.** `UnifiedEmbedding` dequantizes the gathered rows on every forward. The table is 729 x 1152 and the gather runs four times per image, so caching a dense copy at load was not worth the extra memory. The cost was not measured.

## 4. Validation

M1 Ultra, greedy, fixture `tests/fixtures/got_ocr_page.png`. The base binary was built at cbe7919f.

| Check | Base | Branch |
|---|---|---|
| 4-bit CLI `generate --image` | Abort (exit 134), the broadcast error | "GOT OCR two point zero renders this page" |
| 4-bit `mlxcel-server` `/v1/chat/completions` with a base64 image | Not run | Same text as CLI |
| bf16 image prompt | Baseline | Byte-identical |
| bf16 text prompt | Baseline | Byte-identical |
| 4-bit vs bf16 OCR text | n/a | Same string |

Orchestrator gate on the branch at main cbe7919f: fmt, cross-repo refs, workspace clippy with `-D warnings`, and the `vision::`, `models::paddleocr` and `vlm_family_batched_decode_tests` test modules.

## 5. Not Verified

- The 4-bit checkpoint was not run through the base server. The CLI crash and the shared encoder path make the server result predictable, but it was not observed in this run.
- The bf16 text prompt produced one token ("France"). The text path does not touch the changed code.
- No Linux/CUDA run.
- Three LOW items from the security review remain, and are listed in the PR body:
  - A table whose width differs from `hidden_size` still fails at the residual add instead of at load. This predates the PR.
  - The lookup cost is unmeasured.
  - A dense table with a stray `.scales` tensor takes the quantized path. Qwen3-VL behaves the same way.

## 6. Learning Points

- MLX quantized exports can quantize embedding tables inside the vision tower, not only linears. A loader that copies `.weight` directly needs to check for a sibling `.scales`. `UnifiedEmbedding` is the shared answer, and this is the second VLM (after Qwen3-VL in #2459) that needed it. A quick scan of the other vision encoders for direct `.weight` copies of embedding tables would find any remaining cases before a user's checkpoint does.
- Removing code that looks like a no-op can remove an implicit guard. Here the reshape doubled as the only shape check.
