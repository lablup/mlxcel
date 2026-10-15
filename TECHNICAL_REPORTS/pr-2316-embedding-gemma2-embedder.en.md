# PR #2316: Support EmbeddingGemma 2 (embedding_gemma2) multimodal embedder

**Date**: 2026-10-15

**Status**: Implemented and independently reviewed; unit and integration test suite passing

**Risk**: Medium (attention masking, per-layer geometry overrides, and projection-only PLE in bidirectional Gemma 4 stack)

---

## Executive Summary

PR #2316 implements full support for Google's EmbeddingGemma 2 (`google/embeddinggemma-2` and `mlx-community/embeddinggemma-2-*`), a 0.6B multimodal bidirectional embedding model based on the Gemma 4 backbone. The implementation provides hybrid local/global bidirectional attention masks, scale-1.0 SDPA, parameterless `v_norm` (`RMSNormNoScale`), projection-only Per-Layer Embeddings (PLE) with gated per-layer residual blocks, in-model `512 -> 768` embedding projection, masked mean pooling, Matryoshka dimension truncation, and optional Gemma 4 vision and audio tower loading. Review-time hardening fixed a placeholder string corruption in image prompt formatting, corrected fallback attention geometry for non-sliding layers, aligned the reference retrieval parity test with official Venus/Mars benchmark scores (`[0.6775, 0.8348]`), and added multimodal token expansion test coverage.

---

## 1. Problem Statement

### 1.1 Background

Google DeepMind released EmbeddingGemma 2 (`embedding_gemma2`, `embedding_gemma2_text`) on 2026-10-06. Unlike the first-generation EmbeddingGemma (`gemma3_text`), which was text-only with sequential external dense projection modules, EmbeddingGemma 2 is natively multimodal across text, image, and audio, and embeds an in-model linear projection (`512 -> 768`) directly after the final layer norm. Serving this model in mlxcel requires encoder-only bidirectional execution, custom per-layer GQA/MQA head dimension overrides, and projection-only PLE without token lookup tables.

### 1.2 Key Requirements & Gaps Addressed

- **Per-layer Attention Overrides**: 20 local layers (`sliding_attention`) use `num_attention_heads: 4`, `num_key_value_heads: 2`, `head_dim: 256`, `rope_theta: 10000.0`, while 4 global layers (`full_attention` at indices `05`, `11`, `17`, `23`) override to `head_dim: 512`, `num_key_value_heads: 1`, `rope_theta: 1000000.0`.
- **Bidirectional Attention Masks**: Symmetric sliding-window mask (`|q - k| <= sliding_window`) intersected with bidirectional padding mask for local layers, and bidirectional padding mask for global layers.
- **Projection-only PLE**: Scaled post-scatter input projection (`512 -> 24 * 512`, factor `hidden_size^-0.5`) through `per_layer_projection_norm`, feeding each layer's `ple_block` (`per_layer_input_gate` -> GeLU -> `* ple_slice` -> `per_layer_projection` -> `post_per_layer_input_norm`) and `layer_scalar`.
- **Dtype Policy**: Explicit exclusion from `bf16_to_f16_at_load` because `float16` overflows the activation dynamic range across the deep unscaled residual connections.
- **Multimodal Modular Loading**: Optional `vision_tower` (`Gemma4VisionModel`) and `audio_tower` (`AudioEncoder`), scattering embeddings into `<|image|>` and `<|audio|>` placeholder positions.

---

## 2. Technical Review & Findings

### 2.1 Security & Robustness

- Weight key sanitization strips untrusted or superfluous weights (`lm_head.*`, rotary tables, position IDs) and normalizes Hugging Face and MLX key paths to canonical names.
- Zero-length image inputs or missing tower configurations fail safely without panicking.

### 2.2 Findings Discovered and Resolved

1. **Corrupted Placeholder in `format_text` (High)**:
   - *Issue*: `format_text` returned a corrupted string `"<|image><|image|><image|>"` on empty input. For image embeddings via `embed_image`, this emitted malformed subword tokens, bypassed the `else` branch in `expand_image_tokens`, and corrupted the input sequence.
   - *Fix*: Replaced `format_text` with clean identity formatting (`text.to_string()`) and ensured the `else` branch of `expand_image_tokens` preserves leading tokens (such as BOS) before appending `boi_token_id + image_tokens + eoi_token_id`.
2. **Fallback Geometry in `per_layer_config` (High)**:
   - *Issue*: In `EmbeddingGemma2Config::from_json`, non-sliding layers defaulted to `fallback_global_kv_heads` and `fallback_global_head_dim` only if `per_layer_map.is_none()`. If `per_layer_config` was present but an entry omitted those specific fields, it fell back to local layer geometry (`head_dim: 256`, `kv_heads: 2`).
   - *Fix*: Made non-sliding fallback unconditionally use `fallback_global_kv_heads` (1) and `fallback_global_head_dim` (512), matching the `rope_theta` schedule.
3. **Reference Retrieval Benchmark Parity Test (Medium)**:
   - *Issue*: `real_checkpoint_embeddinggemma2_parity_and_ranking` tested arbitrary queries without Venus/Mars and omitted the reference validation score check.
   - *Fix*: Updated the test to use reference Venus/Mars query documents and asserted cosine similarity parity against `retrieval_scores_venus_mars: [0.6775, 0.8348]`.
4. **Multimodal Expansion Unit Coverage (Medium)**:
   - *Issue*: `expand_image_tokens` had no direct unit test.
   - *Fix*: Added `expand_image_tokens_text_only_and_vision_paths` testing text-only passthrough and image token handling.
5. **Multimodal Scatter Empty-List / Null DoS & Audio Zero-Length Div-by-Zero (High)**:
   - *Issue*: In `forward_multimodal`, `concat_many(&feat_list, 1)` asserted `!arrays.is_empty()`, resulting in a server panic if multimodal preprocessing emitted 0 items or null pixel values. An empty audio sequence also caused a divide-by-zero panic in `masked_scatter`'s remainder calculation.
   - *Fix*: Added an explicit check `!feat_list.is_empty()` before `concat_many`, guarded `audio_size > 0` before scattering, and validated soft token counts in `expand_image_tokens`.
6. **Mixed Quantization Boolean `false` Override Ignored (High)**:
   - *Issue*: `quant_params_for_prefix` only checked `override_val.as_object()`, ignoring boolean `false` overrides (e.g., `"language_model.ple": false`). It fell through to the default group-64 4-bit quantization `(64, 4)`, treating unquantized modules as quantized and corrupting weights during load.
   - *Fix*: Handled `override_val.as_bool() == Some(false)` to return `(0, 0)` (unquantized), inspected both `text_config` and root `config`, and mapped `"ple"` / `"language_model.ple"` to `.ple_block`.
7. **Quadratic Window Mask Memory Allocation on Short Sequences (High)**:
   - *Issue*: `sliding_window = 512` means for any sequence with `L <= 513`, every token pair satisfies `|q - k| <= 512`. Generating an `[B, 1, L, L]` float32 tensor (up to 268 MB) for short sequences caused significant unnecessary VRAM allocation and graph overhead.
   - *Fix*: Checked `has_sliding && l > sliding_window + 1`, falling back to broadcasting the `[B, 1, 1, L]` full bidirectional mask when `L <= sliding_window + 1`.
8. **In-Loop Scalar Transfers & Cloning Overhead (Medium)**:
   - *Issue*: `layer_scalar` created temporary host vectors and performed redundant dtype conversions via `from_slice_f32` and `astype`.
   - *Fix*: Replaced host conversions with `mlxcel_core::multiply_scalar`, eliminated redundant `inputs_embeds` cloning, and avoided `astype` when scalar dtype already matches activations.
9. **Malformed Configuration Geometry Bounds & Architecture Detection (Low)**:
   - *Issue*: Non-positive or zero configuration dimensions could cause zero-division panics. Nested `text_config.architectures` was not checked in detection or sanitization.
   - *Fix*: Added `.max(1)` lower bounds to all geometry parameters, checked `text_config.architectures` in `detection.rs` and `sanitize.rs`, and stripped `model.` / `language_model.` prefixes before dropping `lm_head.*`.

---

## 3. Technical Decisions & Rationale

- **Reusing `mlxcel_core::compiled_geglu_approx_activation`**: EmbeddingGemma 2's MLP and PLE gating block both use `gelu_pytorch_tanh` (GeGLU approx). Reusing the fused C++ / HIP kernel avoids graph compilation overhead and matches Gemma 4 generative inference.
- **Inclusive Sliding Window Mask**: `create_bidirectional_window_mask(attention_mask, sliding_window + 1)` precisely maps onto EmbeddingGemma 2's inclusive `|q - k| <= sliding_window` definition.
- **In-model Projection vs External Modules**: EmbeddingGemma 2 executes `language_model.embedding_projection` inside the forward pass before pooling, avoiding external `Dense` modules while preserving standard `pool(&hidden, mask, PoolingMode::Mean)` and Matryoshka truncation downstream.

---

## 4. Verification

- `cargo test --lib embedding_gemma2`: 8 passed, 0 failed, 1 ignored (real checkpoint test requiring external model weights).
- `cargo test --lib embeddings::`: 85 passed, 0 failed.
- `cargo clippy --lib --tests -- -D warnings`: 0 warnings.
- `cargo fmt --check`: passed.
