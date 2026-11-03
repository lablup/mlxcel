# PR #2844: MiMo-V2.6 multimodal support

**Date**: 2026-11-03
**Status**: Partial (open PR; closes #2409)
**Risk Level**: High

## Executive Summary

PR #2844 adds the `mimo_v2` family (Xiaomi MiMo-V2, V2.5, V2.6) on top of the `mimo_v2_flash` decoder that #2829 landed for #842. The decoder is reused unchanged; the new work is a checkpoint loader that undoes the vendor's tensor-parallel fused QKV with its padded per-shard FP8 scale grids, repacks MXFP4 routed experts, stacks the per-expert layout into the Flash `switch_mlp` planes, and a `mimovl` vision tower that is wired into the shared Qwen2.5-VL image and video runtime. 28 files, +12,869/-4. Two committed JSON fixtures (the V2.6 `config.json` and a parity dump) account for 9,548 of the added lines, leaving roughly 3,300 lines of Rust, tests and docs.

The change is validated against synthetic tensors and one tiny fixture dumped from upstream mlx-vlm. The 173.8 GB real checkpoint is not present on the validation host, so no loader arithmetic has been checked against real weights and no CLI or server generation has run. The PR is also currently red in CI for a signature drift against `main`, described in section 5.

## 1. Problem Statement

Issue #2409 ports `mimo_v2`, an omni family whose upstream `LanguageModel` subclasses the `mimo_v2_flash` one and overrides only `sanitize` ("The decoder stack is unchanged from MiMo-V2-Flash. V2.6 differs only in how the checkpoint stores its weights"). The port therefore must build on the #842 decoder rather than fork a second one. That decoder already covers the hybrid sliding/full attention schedule, the split Q/K and V/O head widths, partial RoPE with per-layer bases, the `noaux_tc` router and 256 routed experts.

Three checkpoint layouts block a direct load:

- **Flat config.** The text hyperparameters sit at the top level of `config.json`, with no `text_config` object. `vision_config.in_chans` is the released spelling of `in_channels`, and `audio_config.speech_vocab_size` / `speech_zeroemb_idx` ship as JSON strings, not integers.
- **Fused, tensor-parallel QKV.** `attention_projection_layout: fused_qkv` replaces `q_proj` / `k_proj` / `v_proj` with one `self_attn.qkv_proj.weight` whose rows are the concatenation of TP shards, each holding its own q, then k, then v rows. The FP8 `weight_scale_inv` grid is per shard and padded to whole 128-row blocks, so the grid row count does not follow from the weight row count alone.
- **Native MXFP4 experts.** Routed experts ship as E2M1 codes with one E8M0 scale per 32 values under `<plane>.weight_scale`, which MLX's `gather_qmm` cannot consume as-is.

The vision side is a new tower, not a reuse: `mimovl` is Qwen2.5-VL-shaped but differs in GQA width, attention sinks, window shape and token ordering. Audio, the MTP layers and the bundled drafter are explicitly out of scope for the issue.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/models/mimo_v2.rs`, `src/models/mimo_v2_weights.rs` | Family entry point plus the remap: `candidate_degrees`, `tensor_parallel_degree`, `unfuse_qkv`, `dequantize_fp8_weights`, `convert_mxfp4_experts`, `stack_experts`, `sanitize_text_weights`, `take_vision_weights` |
| `src/loading/vlm_mimo_v2.rs` | Flat-config builder, token-id validation, audio config parse, text/vision weight split, encoder and processor construction, `skip_vision` handling |
| `src/vision/encoders/mimovl.rs`, `mimovl_layers.rs` | Conv3d patch embed, 28-block ViT with GQA, sinks, banded windows and the column-major schedule, patch merger |
| `src/vision/mimo_v2_vl.rs` | `MimoV2VlModel`: `QwenVlRuntime` implementation, feature scatter, `LanguageModel` forwarding to the Flash decoder, audio refusal |
| `src/models/mimo_v2_flash.rs` | `forward_prepared`, `input_embeddings`, `forward_prepared_embeddings_with_sequence_id` so a wrapper can decode pre-embedded rows |
| Detection, registry, metadata, capabilities, TP and memory tables | `ModelType::MiMoV2VLM`, `mimo_v2` detection arm, `mimo_v2_vlm` registry id, `TEXT_IMAGE_VIDEO` modalities, native video, VLM kind, adapter rejection |
| `src/server/model_worker.rs` | MiMo-specific audio rejection message |
| Tests and fixtures | 991 lines of loader/forward/parity tests plus the two JSON fixtures |
| Docs | README and `docs/supported-models.md` family entries |

## 3. Architecture as Implemented

### 3.1 Decoder reuse

`MiMoV2VlModel` owns a `MiMoV2FlashModel` and delegates every `LanguageModel` method to it: caches, sequence state, batching flags, snapshots and trimming. The only decoder additions are three entry points. `input_embeddings` exposes the token table so the wrapper can scatter vision features before decode. `forward_prepared` runs the decoder over already-embedded rows, which is the fixed multi-token prefill path. `forward_prepared_embeddings_with_sequence_id` resets the model-owned caches on a fresh multi-token prompt without a sequence id and otherwise routes through the per-sequence state. Because MiMo-V2 text positions are 1-D RoPE, the wrapper returns a `mrope_only(None)` snapshot and clears no media state.

### 3.2 Loader pipeline

The order matters and is encoded in `sanitize_text_weights`:

1. Drop `audio_encoder.*` and `speech_embeddings.*`; drop the vision tower into a separate map, normalizing the native `visual.` prefix to `vision_tower.`; strip `language_model.`; drop `.mtp.` and `model.mtp` keys.
2. If any key ends in `weight_scale`, run the MXFP4 repack first, because those sidecars must become `.scales` before anything looks for `_scale_inv`.
3. If any `self_attn.qkv_proj.weight` survives, unfuse it. This must precede the general FP8 pass because the unfuse consumes each fused tensor's grid per shard.
4. Dequantize every remaining `_scale_inv` projection to BF16.
5. Stack `mlp.experts.{e}.{gate,up,down}_proj.{weight,scales,biases}` into `mlp.switch_mlp.*` for each layer, then drop any leftover `mlp.experts.*` key.

The general FP8 pass and the expert repack are no-ops on the converted mlx-vlm release, which is already unfused, dequantized and stacked.

### 3.3 Fused QKV unfuse

`candidate_degrees` enumerates powers of two up to `num_key_value_heads` and keeps those where `degree * (heads/degree * head_dim + kv_heads/degree * (head_dim + v_head_dim))` equals the weight row count and `degree * ceil(shard/128)` equals the grid row count. `tensor_parallel_degree` intersects the candidate sets over every fused tensor and takes `max(common)`, matching upstream. On the published full-attention layer the weight has 13,568 rows. A degree-4 shard is 3,392 rows, which pads to 27 blocks, so the grid has 108 rows. That padded 108 cannot be explained by degrees 1 or 2, which is what pins degree 4; a 106-row grid alone would admit both 1 and 2. For each rank the loader slices the weight rows and the matching grid rows, reconstructs block FP8 for that shard with the shared LUT, cuts q/k/v by row count, and concatenates the shard pieces across ranks before writing `q_proj` / `k_proj` / `v_proj`.

### 3.4 MXFP4 experts

`convert_mxfp4_experts` moves every `<plane>.weight_scale` to a side table, views the packed `<plane>.weight` as `uint32` (four E2M1 codes per word), and re-emits the sidecar as `<plane>.scales` with the `.weight` suffix trimmed. The Flash loader then infers `mode: "mxfp4"`, `group_size: 32`, `bits: 4` from the bias-free `uint8` scales. The native release is the only path that exercises this; the converted release already ships `scales`.

### 3.5 Vision tower

The tower is 28 blocks of hidden 1280 with 32 query heads and 8 KV heads, but `qk_channels` is 64 rather than `1280 / 32 = 40`. The fused QKV output is therefore 3,072 wide (`(32 + 2*8) * 64`) and the attention output 2,048 (`32 * 64`), projected back to 1280. Four port-specific behaviors:

- **Sinks.** On windowed blocks, `use_sink` adds a learned per-head logit to the attention logit of key position 0. It is an additive mask entry, not a denominator-only sink, so softmax sees it exactly as upstream does.
- **Banded windows.** Inside each frame's `cu_seqlens` segment, positions with `|i - j| > visual_token_window_size` (64) are masked. Full blocks, listed in `fullatt_block_indexes` `[0, 9, 18, 27]`, are unmasked.
- **Column-major schedule.** `vit_window_attn_types` is a per-block list (`1` column-major with column rotary, `0` row-major, `-1` full attention). The encoder permutes whole spatial-merge blocks on entry to a `1` run and restores the inverse before leaving it. The published config alternates runs of `[-1, 0, 0, 0, 0, 1, 1, 1, 1]`.
- **Patch embed and merger.** The Conv3d patch embedding is accepted in 2-D `[out, C*T*P*P]` or either 5-D layout and reordered once into the temporal-major `T, C, P, P` row order that the shared Qwen2.5-VL processor emits. The merger is a weight-only LayerNorm, two linears and GELU, not Qwen's RMSNorm merger.

### 3.6 Media path

`MimoV2VlModel` implements `QwenVlRuntime`, so the CLI's `--image` / `--video`, the server's native video path, prompt expansion and feature scatter all run the shared Qwen2.5-VL code. `LoadedModel::MiMoV2VLM` is added to the capabilities table as `VlmRuntimeRef::Qwen`, to `qwen_video_runtime`, and to `model_type_has_native_video`. `get_input_embeddings` counts image and video placeholders, runs the tower, rejects a feature/placeholder mismatch, checks the hidden width against the text embedding width, then merges one-for-one. `skip_vision: true` or a checkpoint with no tower produces a text-only wrapper that still owns a processor for prompt info.

## 4. Technical Decisions

### 4.1 Reuse the #842 decoder instead of forking

The upstream relationship is a subclass with an overridden sanitizer, so the port keeps that shape: the decoder is untouched except for the three pre-embedded entry points the wrapper needs. A separate `mimo_v2` decoder would have duplicated 1,500 lines of hybrid-cache, router and partial-RoPE logic and let the two drift. The cost is that the decoder's assumptions (model-owned heterogeneous caches, batching policy) are inherited unchanged, which is correct here because the weights are the same family.

### 4.2 Implement `QwenVlRuntime` rather than a private media path

MiMo-V2 uses the Qwen2.5-VL processor: the same `<|vision_start|><|image_pad|><|vision_end|>` expansion and `grid_thw`. Copying that path into a MiMo-specific module would have duplicated prompt expansion, placeholder counting and the video frame path in three places (CLI, server, tests). Implementing the trait keeps one implementation and makes the server's video wiring a one-line capability addition. The trade-off is that any future MiMo-specific media behavior must be expressed through the shared trait or it will not reach both entry points.

### 4.3 Recover the TP degree from the padded grid

Assuming degree 4 would match the published checkpoints but would silently mis-slice any checkpoint converted at another degree. Reading the degree from the shapes, and intersecting over every fused tensor, makes an inconsistent checkpoint a load error instead of a plausible but wrong model. The padded-grid rule is the only thing that distinguishes degree 4 from degree 2 on the full-attention layers, so the docstring example became a test rather than a comment.

### 4.4 Move weights through the remap instead of copying

The native checkpoint is 173.8 GB. Every remap function takes and returns the `WeightMap` by value and removes entries as it goes, so an untouched plane is moved, and the converted release, where most passes are no-ops, pays no copy at all. This is the reason the API reads as a chain of owned transforms rather than a set of in-place mutators.

### 4.5 Reject audio explicitly

The checkpoint ships a 20-channel codec tokenizer and a small `AudioEncoder`, neither of which mlxcel has. The loader drops those key families and the server returns a message naming the missing port, rather than loading a half-model or silently ignoring the clip. The parser still reads `audio_config`, including its two string-typed scalars, so a later audio port does not inherit a stringly typed config.

### 4.6 Validate token ids against the vocabulary

The image, video, audio and vision boundary token ids are configurable and default to the published values. The builder rejects an id outside `vocab_size` because an out-of-range vision token id would place every image feature at the wrong row while still producing fluent text. It also accepts `text_config` when present and otherwise treats the flat root as the text config, matching upstream `ModelConfig.from_dict`.

## 5. Verification

The PR reports these targeted runs:

| Command | Reported evidence |
|---|---|
| `cargo test --profile test-fast -p mlxcel --lib loading::vlm::mimo_v2::` | 13 tests: config builder, detection, both unfuse degrees including the padded grid, MXFP4 remap, native layout dequant/stack, a tiny end-to-end image forward |
| `cargo test --profile test-fast -p mlxcel --lib vision::encoders::mimovl::` | fixture parity against tensors dumped from upstream mlx-vlm on a tiny tower |
| `models::mimo_v2_flash::` | 18 tests |
| `models::detection_tests::` | 85 tests |
| `models::registry::` | 20 tests |
| `models::metadata_tests::` | 4 tests |
| `model_metadata_tests::` | 7 tests |
| `models::vlm_text_only_detection_tests::` | 9 tests |
| `multimodal::qwen_vl::` | 10 tests |
| `cargo clippy --lib --tests -- -D warnings` | pass |
| `cargo fmt --check` | pass |

What the tests actually pin, read from the code:

- The 13 loader tests build the V2.6 fixture at flat top level and assert hidden 4096, 48 layers, 64/4 heads, head widths 192/128, 8 sliding KV heads, 256 experts top-8, the two RoPE bases, `in_chans`, `qk_channels` 64, `fullatt_block_indexes` `[0, 9, 18, 27]`, window 64, the coerced `speech_vocab_size` 1280 and `speech_zeroemb_idx` 1024, all seven token ids, and `skip_vision` false.
- Detection routes the real V2.6 config to `MiMoV2VLM`, and the registration assertions cover `is_vlm_model_type`, native video, registry id `mimo_v2_vlm`, family `Xiaomi` and `ALL_MODEL_TYPES`.
- The unfuse tests build constant FP8 planes (q = 1.0, k = 2.0, v = 4.0) and assert the recovered shapes `[64*192, 32]`, `[4*192, 32]`, `[4*128, 32]` and the values after dequantization, at degree 1 with no grid and at degree 4 with the 108-row padded grid. A 107-row grid errors with "no tensor-parallel degree".
- The native-layout test covers a layer 0 dense FP8 projection, layer 1 V2.6 MXFP4 experts and layer 2 V2.5 FP8 experts, and asserts the resulting `switch_mlp` dtypes (`uint32` and BF16), shapes, scales, and that no `.experts.` key survives.
- The MXFP4 test asserts a `[4, 8, 16]` uint8 plane becomes `[4, 8, 4]` uint32 with a `.scales` sidecar and no `weight_scale` key.
- The vision fixture is a 3-block tiny tower (full block 0, window blocks 1 and 2, sinks on 1 and 2, `vit_window_attn_types` `[-1, 1, 0]`) with grid `(1, 4, 4)`, merged output `[4, 6]`, compared at `2e-4 * max(|expected|, 1)`. The test names the failure mode: dropping the sinks or the banded mask changes the expected tensor.
- The forward test builds a 4-layer random `mimo_v2` on the Flash decoder plus the tiny tower, scatters four image features into a six-token prompt, checks the merged rows equal the tower output, and checks wrapper logits equal a direct `forward_prepared_embeddings_with_sequence_id` run. A placeholder/feature mismatch panics with "placeholders".

### Current gate status

The tree as committed no longer compiles against current `main`. `630aaeba` (#2842, this branch's parent) widened `reconstruct_block_fp8_tensor` to six arguments, adding `block_cols`, but `src/models/mimo_v2_weights.rs` still calls it with five at line 156 and line 293. CI run `41651251817` fails on exactly this, in `cargo-clippy`, `OpenXLA feature compile` and `WebUI installed artifact` (E0061, "this function takes 6 arguments but 5 arguments were supplied"). `cargo-fmt` and `CPU-only link` pass, and the PR is a draft with `mergeStateStatus: UNSTABLE`. The local clippy and fmt results above are what the PR reports; they do not describe the current tree.

## 6. Unverified

- **Real checkpoint.** `nativ-community/MiMo-V2.6-Flash-RL-MLX-4bit` is 173.8 GB and is not present on the validation host, so the CLI and server coherence run has not happened. No text, image or video prompt has been generated from real weights, and no teacher-forced logit trace against mlx-vlm has been produced. Every loader arithmetic path (degree recovery, per-shard FP8 reconstruction, MXFP4 scales, expert stacking) is synthetic-only. The native `XiaomiMiMo/MiMo-V2.6-Flash-RL` release (177.7 GB, FP8 plus MXFP4) was not downloaded either, so the "both layouts load" claim is untested at full scale.
- **Audio.** Explicitly out of scope. The 20-channel codec tokenizer has no mlxcel port; the server returns a MiMo-specific error and the loader drops the weights.
- **MTP layers and the bundled drafter.** `num_nextn_predict_layers` is 3 and the checkpoint ships `dflash/`. Both are dropped at load and not exercised.
- **CUDA and ROCm.** No run; this host is Apple Silicon.
- **Tensor and pipeline parallel.** `ModelType::MiMoV2VLM` is wired into the TP fallback architecture table so the dispatch is total, but no multi-device run was performed.
- **Batched decode.** The wrapper forwards the Flash decoder's batching flags unchanged; no concurrent server request was run against this family.

## 7. Learning Points

A padded quantization grid is a signal, not noise. The 108-row grid for a 106-block weight is what makes degree 4 distinguishable from degrees 1 and 2; treating the extra rows as padding to trim would have made the loader accept a degree-2 interpretation and slice the weight incorrectly. Reconstructing the degree from the checkpoint, and intersecting it over every fused tensor, is cheaper than trusting the issue prose.

Order of passes in a checkpoint remap is part of the contract. MXFP4 sidecars must be renamed before the FP8 pass looks for `_scale_inv`, and fused QKV must be unfused before the general FP8 pass because the grids are consumed per shard. Encoding the order in one function with comments, rather than as independent helpers the loader calls in some order, is what makes the converted and native releases both no-ops where they should be.

Multimodal wrappers need a pre-embedded decode entry point. The decoder previously only accepted token ids, so a wrapper that scattered features had no way back in without re-embedding. Adding `forward_prepared` and exposing the token table is a small change that keeps the media and text paths sharing one decoder.

## 8. Risks

High. A 173.8 GB family with a fully new vision tower and a multi-stage weight remap has no real-weight validation, so a loader error would appear as plausible but wrong logits rather than a crash. The synthetic tests are strong on shapes, dtypes and value constants, but they cannot catch a misinterpreted real FP8 grid, a wrong E8M0 scale convention, or an expert ordering mismatch. The branch also does not currently build against `main`, so the reported local gate results predate the tree that CI sees. The decoder itself is the #2829 port, which was validated on the Flash checkpoint, but the V2.6 checkpoint's text weights have never been through this path.
