# Technical Report: PR #1735 - feat(vlm): add the Cohere Compass (North-Micro-Vision) VLM

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (validated on GB10 / CUDA against a transformers oracle; the metal+accelerate acceptance command is not runnable on this host and was not run)
**Languages**: Rust
**Risk Level**: Medium (one new VLM family; no shared code path changes behavior for existing families)

---

## Executive Summary

`model_type: "cohere_compass"` (CohereLabs North-Micro-Vision-Instruct, 2.5B) is a Qwen3-VL deepstack vision tower in front of a Command-style parallel text decoder. Almost everything on the vision side is reuse: the encoder, the image processor, the `<|IMAGE_PAD|>` expansion, the MRoPE position-id rule and the weight-prefix remap are the Qwen3-VL ones, unchanged.

The decoder is new, and one detail in it is the reason this report exists. Issue #1354 specified Qwen3-VL's `InterleavedMRoPE` on the sliding layers. That is wrong. The checkpoint does carry `mrope_interleaved: true`, but upstream lists that key under `ignore_keys_at_rope_validation` and never reads it: `CohereCompassRotaryEmbedding` pre-permutes `inv_freq` and then assigns axes in contiguous `[H, W, T]` sections. Measured against the transformers class itself, the specified interleave is off by up to 1.96 in `cos` where the implemented table is off by 2.2e-7. Building to the issue would have shipped a model that loads, runs, and produces plausible-looking garbage.

---

## 1. What the family is

### 1.1 Text decoder (`cohere_compass_text`)

A Command-family decoder, 28 layers, hidden 2048, 16 Q / 8 KV heads at `head_dim` 128, SwiGLU `intermediate_size` 6144, vocab 262144, tied embeddings.

```
h = embed_tokens(ids)
for i in 0..28:
    n = LayerNorm(h)                        # weight only, eps 1e-5
    h = h + Attn_i(n) + SwiGLU_i(n)         # parallel block, one norm feeds both
    if i < len(deepstack) and image present:
        h[visual positions] += deepstack[i]
h = LayerNorm(h)
logits = embed_tokens.as_linear(h) * 0.25   # logit_scale
```

There is no `post_attention_layernorm` anywhere in the checkpoint. `layer_types` alternates three `sliding_attention` layers to one `full_attention`, so the full layers sit at indices 3, 7, 11, 15, 19, 23 and 27.

### 1.2 Positional encoding is per layer type, and one of them has none

`rope_parameters` is a per-layer-type map:

```json
{
  "sliding_attention": {"mrope_interleaved": true, "mrope_section": [24, 20, 20],
                        "rope_type": "default", "rope_theta": 50000},
  "full_attention": null,
  "rope_theta": 10000.0, "rope_type": "default"
}
```

The JSON `null` is load-bearing. It means the seven full-attention layers get **no positional encoding at all**, not "fall back to the block-level `rope_theta`". Reading it the second way rotates seven NoPE layers that were trained without any rotation, and nothing about the resulting model fails loudly. `CompassTextConfig::rope_for_layer_type` returns `Ok(None)` for an entry that is `null` or absent while other layer types are present, and `CompassAttention::rope` is `Option<CompassMRoPE>` so a NoPE layer carries no table at all rather than a disabled one.

### 1.3 Vision tower

`cohere_compass_vision` is `Qwen3VLVisionConfig` key for key: depth 27, hidden 1152, 16 heads, patch 16, temporal patch 2, spatial merge 2, `out_hidden_size` 2048, `num_position_embeddings` 2304 (a 48x48 grid interpolated per image), `deepstack_visual_indexes` `[8, 16, 24]`. `Qwen3VLVisionEncoder` parses and runs it unchanged, and `forward_with_grid` already returns exactly the `(features, deepstack_features)` pair the decoder needs.

---

## 2. The MRoPE the issue got wrong

### 2.1 What upstream actually computes

`CohereCompassRotaryEmbedding.compute_default_rope_parameters` does something Qwen3-VL does not: it **pre-rotates** the frequency list before any axis is chosen.

```python
inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))     # natural order, 64 entries
hw_dim   = mrope_section[0] + mrope_section[1]           # 24 + 20 = 44
t_dim    = mrope_section[2]                              # 20
inv_freq_3d[:hw_dim] = cat([inv_freq[:-t_dim][0::2], inv_freq[:-t_dim][1::2]])
inv_freq_3d[-t_dim:] = inv_freq[-t_dim:]
```

`recomposition_frequencies` then splits the `[3, B, L, 64]` frequency tensor by `mrope_section` and takes chunk `i` from axis `(i + 1) % 3`, which is H, then W, then T. So the sections are **contiguous and ordered `[H, W, T]`**, and `cos`/`sin` are the duplicate-halves (split) layout that `rotate_half` expects.

Written out per channel for the published `[24, 20, 20]`:

| channels | axis | frequencies |
|---|---|---|
| 0..23 | H | `inv_freq[0, 2, ..., 42]` then `inv_freq[1], inv_freq[3]` |
| 24..43 | W | `inv_freq[5, 7, ..., 43]` |
| 44..63 | T | `inv_freq[44..63]` |

The H/W boundary lands mid-permutation because the published section is `[24, 20, 20]` while the code's own default is `[22, 22, 20]`, where the split would be clean. That asymmetry is upstream's, and it is reproduced rather than tidied up.

### 2.2 What the issue specified instead

Qwen3-VL's interleave over the same `[24, 20, 20]` assigns, in natural frequency order, T to channels `{0, 3, ..., 57} + {60..63}` (24 of them), H to `{1, 4, ..., 58}` and W to `{2, 5, ..., 59}`. Different partition, different frequencies, different axis for the wide section. It is not a relabeling of the Compass table.

### 2.3 The measurement

`rope_check.py` builds `CohereCompassRotaryEmbedding` from the real `config.json` and compares its `cos`/`sin` at `(t, h, w) = (7, 11, 13)` against both formulations:

```
mlxcel formulation  max|cos diff|: 2.2290754853049322e-07
mlxcel formulation  max|sin diff|: 3.8017985570792945e-07
qwen3-vl interleave max|cos diff|: 1.9554328814065203
```

2.2e-7 is f32 rounding. 1.96 is a different function.

### 2.4 The consequence for the text-only path

`qwen3_vl.rs` has a fast path that swaps the MRoPE table for `fast_rope` when a sequence has no image, on the argument that all three axes then carry the same position and the interleaved table collapses to plain 1-D RoPE. That argument does **not** transfer. Compass's `inv_freq_3d` is a permutation of the natural order, so with equal positions channel `j` still uses `inv_freq_3d[j]` while `fast_rope` would use `inv_freq[j]`. The fast path is therefore absent here on purpose, and the module header says so; a text-only sequence simply reaches the same table with all three rows equal.

---

## 3. Sliding window: masks, not a rotating buffer

The issue asked for `RotatingKVCache(max_size = 4096)` on the sliding layers. The repository does not do sliding windows that way. `cohere2.rs`, `cohere2_moe.rs`, `olmo3.rs` and `gemma3n.rs` all keep a dense `KVCache` and enforce the window with two prefill masks plus the `window_size` argument to the fused SDPA, and `LanguageModel::make_caches` returns `Vec<KVCache>` so a rotating cache is not expressible there anyway.

Compass follows the shipped contract. `prefill_masks` builds `create_causal_mask(L, live_len)` for the full layers and `create_sliding_window_prefill_mask_dense(L, live_len, 4096)` for the sliding ones, sized from `live_len()` rather than the monotonic `offset` so a `--max-kv-size` trim cannot make the mask wider than the returned K/V (issue #419). `CompassAttention::attend` slices K/V down to the mask's key axis when the window clamps it, the same handling as `cohere2.rs` (issues #408 / #419). At decode width both masks are `None` and the window comes from `causal_attention`'s `window_size`.

The caller-supplied `mask` is deliberately ignored: this model needs two differently shaped masks per step and always builds them from its own caches, which is what `cohere2.rs` and `olmo3.rs` do.

---

## 4. Server path

The repository has a specific VLM trap: a family can pass every CLI test while the server runs zero decoder layers, because the CLI builds its own caches and an unoverridden `sequence_state_layout()` plus `supports_batching() == false` hands the scheduler an empty cache vector. `CohereCompassTextModel` keeps the default `supports_batching()` (true) and therefore the default `dense_kv_cache` layout, and `CohereCompassModel` overrides `forward_batched_with_context_and_ids` to the shared `forward_batched_with_seq_ids_dispatch` so each row in a mixed VL + text batch resolves its own MRoPE entry. The `QwenVlRuntime` impl carries the DeepStack-shaped vision cache and the per-sequence MRoPE bind / take / install hooks.

This is validated on `mlxcel-server`, not inferred: section 5 records a `/v1/chat/completions` run with an `image_url` part.

---

## 5. Validation

### 5.1 Oracle construction

There is no published bf16 checkpoint on this host and no mlx-vlm implementation of the family, so the oracle is transformers `5.18.0.dev0` (which carries `cohere_compass`) reading the **same weights** mlxcel reads. `dequantize.py` unpacks the MLX 4-bit affine tensors with the repository's own formula (`w = q * scale + bias`, values packed low-first in each u32, group 64) and writes one float32 checkpoint in the CohereLabs key layout. mlxcel's `remap_qwen3_vl_weights` maps that layout onto its own, so both sides load one file and any divergence is the forward pass rather than the quantizer.

### 5.2 Results

All numbers below are from GB10 (sm_121) under `--profile test-fast --features cuda`, against `mlx-community/North-Micro-Vision-Instruct-4bit` and the float32 checkpoint dequantized from it.

**Both published weight layouts load.** The 4-bit conversion is `language_model.model.*` / `vision_tower.*`; the dequantized file is written in the CohereLabs `model.language_model.*` / `model.visual.*` layout. Every run below was done on both, so the two prefixes are covered by construction rather than by a unit fixture.

**The rotary table matches upstream's class, and the issue's specification does not.** `rope_check.py` builds `CohereCompassRotaryEmbedding` from the real `config.json` and evaluates it at `(t, h, w) = (7, 11, 13)`:

```
mlxcel formulation  max|cos diff|: 2.2290754853049322e-07
mlxcel formulation  max|sin diff|: 3.8017985570792945e-07
qwen3-vl interleave max|cos diff|: 1.9554328814065203
```

**MRoPE position ids are bit-identical to `get_rope_index`.** `rope_index_check.py` transcribes `qwen_vl_mrope_positions_from_tokens` into Python and compares against the model's own method on the real image prompt:

| prompt | grids | seq | identical | max | rope_delta |
|---|---|---|---|---|---|
| one image | `[1, 28, 28]` | 210 | yes | 27 | -182 (both) |
| two images | `[1, 28, 28]`, `[1, 14, 14]` | 261 | yes | 36 | -224 (both) |

**Text-only is token-exact.** Server `/v1/chat/completions` with `logprobs: true` gives the actual emitted tokens, so this is a token comparison and not a decoded-string one:

| run | tokens | identical to oracle |
|---|---|---|
| f32 server, "Summarize the history of the Apollo program..." | 32 | yes, 32/32 |
| 4-bit server, same prompt | 32 | yes, 32/32 |

The CLI produces the identical string on three prompts (one of them a 79-token prompt), on both checkpoints, and the server text matches the CLI.

**The server really runs the decoder.** `prompt_tokens` is 18 for the text request and 210 for the image request, both exactly the oracle's counts, `completion_tokens` is non-zero, and the image request stops on EOS after 25 tokens. That rules out the family's known failure mode where an unoverridden `sequence_state_layout()` hands the scheduler an empty cache vector and the model silently emits nothing.

**The image path is correct but not token-exact.** It agrees with the oracle on everything up to the decoder input: 196 image tokens from grid `[1, 28, 28]`, a 210-token prompt, bit-identical MRoPE positions. The generated tokens then agree for two steps and diverge at step 2:

```
f32 server image1:  mlxcel=25 oracle=32 identical=False common_prefix=2
   first divergence: ' features' vs ' displays'
4bit server image1: identical result, byte for byte
```

In both flips mlxcel takes the oracle's rank-2 token rather than an unrelated one, but the margin is not negligible. Single image, step 2: `' displays'` at logprob -1.5065 against `' features'` at -1.7449, a 0.238-nat gap. Two images ("How many shapes are in each image?"), step 0: `'3'` at -0.511 against `"Let's"` at -1.3771, a 0.866-nat gap. So the vision-conditioned logits are shifted by up to roughly 0.9 nats at the first generated token, which is a real parity gap and not rounding. Both continuations still describe the same picture, and mlxcel's single-image one is the more specific: it names the red square, the blue circle and the green triangle of `tests/fixtures/test_image_shapes.png`, which the oracle's does not.

The two-image prompt is otherwise exact after the loader fix below: 245 image tokens (196 + 49) and a 263-token prompt, both matching the oracle, with bit-identical MRoPE positions.

Ruled out as the cause: `MLX_ENABLE_TF32=0` changes nothing (byte-identical output); both vision activations match the reference (block MLP tanh-GELU against `ACT2FN["gelu_pytorch_tanh"]`, patch merger exact GELU against `nn.GELU()`); the DeepStack branch order and injection depth match; quantization is not involved, since the 4-bit and float32 checkpoints produce byte-identical output on every prompt tested. What remains is accumulated numeric difference across the 27-block tower, which is shared Qwen3-VL code this PR does not touch. Filed as #1738 rather than papered over.

**One real bug the oracle caught.** The first version of the loader read the resize bounds from `preprocessor_config.json` (`size.shortest_edge` 65536), as the issue specified. HF's `AutoProcessor` reads the nested `image_processor` block of `processor_config.json` instead (`min_pixels` 16384). On a 224x224 input the stale bound upscales to a 16x16 merged grid, 64 image tokens against the reference's 49, so a two-image prompt carried 260 image tokens against the oracle's 245 and the two runs were not comparing the same prompt. Nothing failed; the counts silently disagreed. Fixed, with the precedence pinned by unit tests.

**Family surface.** `mlxcel arch` lists `Cohere Compass / North-Micro-Vision`; `mlxcel list` shows the checkpoint; `mlxcel inspect` reports the layer split it derives from the config as `sliding-window: 21 layer(s) capped at 4096 tokens, 7 global`, which is the expected 28-layer 3-to-1 pattern.

**Gates.**

| gate | result |
|---|---|
| `cargo test --profile test-fast --features cuda --lib cohere_compass` | `test result: ok. 23 passed; 0 failed` |
| `cargo clippy --workspace --all-targets -- -D warnings` (CI) | pass |
| `cargo fmt --all -- --check` (CI) | pass |
| `cargo test --workspace --profile test-fast --features metal,accelerate` | **not runnable on this host** (Linux + CUDA, no Metal, no Accelerate); unrun |


---

## 6. What was not validated

- `cargo test --workspace --profile test-fast --features metal,accelerate` from the issue's acceptance criteria **cannot run on this host**: it is Linux with CUDA, with no Metal and no Accelerate. The CUDA equivalent was run for the scoped modules instead. The metal+accelerate criterion is unrun.
- Video input (`<|VIDEO_PAD|>`, `video_token_id` 255032) is out of scope per the issue and is deliberately not wired: the family is absent from `qwen_video_runtime` and from the server's video-capable list, so a video request is refused rather than mis-served.
- The `CohereCompassTextForSequenceClassification` reranker head (`score_shift_a/b`, `pooling`) is out of scope; no published checkpoint uses it.
- Tensor-parallel sharding is out of scope. `fallback_architecture` returns `"cohere_compass"` for the family so the generic transformer plan does not claim it.
- `norm_type: "rms_norm"` and `transformer_block_type` are implemented / validated as config surface only. Neither key appears in `CohereCompassTextConfig` upstream or in the published checkpoint, so they are accepted-and-checked rather than exercised: an unknown `norm_type` and a `sequential` block type both fail at load with a named error instead of running as something else.

---

## 7. Files

| File | Role |
|---|---|
| `src/models/cohere_compass.rs` | decoder: state, masks, DeepStack injection, `logit_scale`, `LanguageModel` impl |
| `src/models/cohere_compass_config.rs` | `text_config` parsing and the per-layer-type RoPE resolution |
| `src/models/cohere_compass_layers.rs` | norm dispatch, attention with the sliding/full split, SwiGLU, parallel block |
| `src/models/cohere_compass_rope.rs` | the Compass MRoPE table (pre-permuted `inv_freq`, `[H, W, T]` sections) |
| `src/vision/cohere_compass.rs` | VLM wrapper: vision features, visual position mask, MRoPE state |
| `src/loading/vlm_cohere_compass.rs` | loader, including the checkpoint's own resize bounds |
| `src/multimodal/qwen_vl.rs` | `QwenVlRuntime` impl (DeepStack cache path) |
| `docs/supported-models.md` | family entry |
