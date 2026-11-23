# PR #3288: Add Clef and Clef-Flash (clef) decision models

**Date**: 2026-11-23
**Status**: PR open (draft, closes #2365; part of #2375). Unit, sequence-fixture and real-checkpoint parity tests pass; CLI and server checked on `models/clef-flash-mlx-8bit`, re-run after the security fixes
**Languages**: Rust, Python (fixture generator)
**Risk Level**: Low (new decision family; one additive Qwen3.5 hidden-state entry point and one additive video helper, generation paths untouched)

---

## Executive Summary

Clef and Clef-Flash (`model_type: clef`, `nativ-community/clef-*`) are Qwen3.5 vision-language backbones with a joint schema head that scores every option of every question in one forward pass over text, images and videos. This PR ports mlx-vlm's `models/clef` behind the #2343 decision API, so `mlxcel decide -m <path>` (with `--image` / `--video`) and `mlxcel-server --decision-model <path>` (`POST /v1/decisions`) serve it. On Clef-Flash 8-bit, token ids and spans match upstream exactly, probabilities are within 1.1e-3 of mlx-vlm on an image request and 2.3e-3 (score value 2.8e-3) on a text request, and an f32 run of both sides agrees to four places. Review found no correctness defects. A security pass then bounded the work done before the token cap, refused literal pad tokens and scrubbed server paths from video errors (section 2.5), and the real-checkpoint outputs were re-run unchanged afterward.

## 1. Problem Statement

### 1.1 Background

#2343 added the decision layer and #3282 / #3284 added the d1-3B and Decider-2b families. Clef was still refused with `Unsupported model type: clef` (#2365).

### 1.2 Existing Issues

- **No loader arm**: no `DecisionKind` recognized `model_type: clef`.
- **No post-norm hidden over merged embeddings**: Decider-2b's `hidden_from_embeddings` took token ids only; Clef needs `inputs_embeds` with media merged and MRoPE positions.
- **No arbitrary-index video sampling**: Clef samples `linspace(0, total - 1, n)` over every source frame, which no `FrameSamplingPolicy` expresses.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| MRoPE state leaking into a later request on the same instance | Medium | Low (cleared before and after every forward) |
| Generation output change from the Qwen3.5 hook | High | Low (additive function, no shared code edited) |
| One request making the single decision worker decode unbounded video | Medium | Low (4 GiB decode budget charged at source resolution, videos decoded one at a time only while the sequence fits the cap, 768-frame cap, server media count limits) |
| A literal pad token in the state panicking the MRoPE position builder (a 500) | Medium | Low (refused with a 400 when media is present) |

## 2. Technical Review

### 2.1 Head fidelity

Every op of `JointSchemaHead`, `EvidenceRoutingLayer` and `FieldDecoderLayer` was checked against `clef.py` and MLX's `nn` modules: LayerNorm eps 1e-5 through `fast::layer_norm`; MHA with biases on all four projections, `reshape(B, L, heads, -1)` head split and `sqrt(1 / head_dim)` scale; `nn.Linear` as `addmm(bias, x, W.T)`; GELU as the erf form in the activation dtype; span means as an f32 matmul cast back to the hidden dtype; precise softmax for option summaries; memory normalized per routing layer but raw for the field decoder; the prior and residual formula with both `exp(min(scale, ln 100))` clamps; type ids `noul 0, choice 1, score 2`; lexical vectors from dequantized `lm_head` rows averaged in f32 and cast to the hidden dtype.

### 2.2 Sequence and options

`bool` lists `true` then `false` with defaults overridden by criteria (a null description is omitted, as upstream). `choice` labels are sorted by code point for the schema and reported in request order with Python's first-maximum argmax. Compact sorted JSON matches `json.dumps(..., ensure_ascii=False, separators=(",", ":"), sort_keys=True)`. The 16384 cap refuses an oversized fixed part with upstream's message and truncates the state otherwise. Media blocks (image runs, timestamped video groups inside outer vision tokens) and video sampling are pinned by the committed fixture.

### 2.3 Backbone hook and state isolation

`Qwen35Model::forward_final_hidden_with_embeddings` resolves positions from the fallback MRoPE slot exactly as a prefill at cache offset 0 does, uses `0..L` on all three axes for text-only input (upstream's `get_rope_index`), runs the shared `hidden_from_embeddings` on fresh caches and clears the fallback slot before returning. `predict` also clears it on entry. The `ClefModel` owns its own `Qwen35VLModel`, separate from any generation model, and is driven from one thread, so no state reaches generation requests. `forward_internal`, `forward_internal_at` and the token path are unchanged.

### 2.4 Parity gap

The text request is 3e-4 over the issue's 2e-3 bound. No cast-point difference was found in the head, and the developer's f32 comparison (both sides identical to four places) rules out a structural port error. mlx-vlm's own bf16 and f32 runs differ by 7.9e-3 on the same request, so the residual is bf16 rounding order in the backbone kernels shared with generation. The test commits 3e-3 for text and 2e-3 for images.

### 2.5 Security fixes (commit `6a0cb27a`)

Before the fix, Clef decoded every requested video before checking the 16384-token cap and charged the shared 4 GiB budget with resized frames, so a request with many videos could decode up to 4 GiB per video on the single decision worker before being refused. Videos are now decoded one at a time, each only while the sequence so far (text, images, earlier videos) still fits, and the budget is charged with source-resolution bytes. The schema is encoded only up to the cap; an early refusal says `schema requires at least N tokens`, and a complete count keeps upstream's `schema requires N tokens before state; maximum is 16384`.

A literal `<|image_pad|>` or `<|video_pad|>` in the state or questions parsed as a placeholder, and with media present the MRoPE position builder panicked on the extra run. Such requests are now a 400. Video errors no longer echo the server-side path of an allowlisted file or a download's temp file. `decision::clef_limits_tests` covers the cap on schema encoding, the never-decoded videos past the cap, the pad-token refusal and the path scrub.

## 3. Technical Decisions

### 3.1 Extend `hidden_from_embeddings` instead of copying the layer loop

**Context:** Decider-2b added a `position_ids` parameter for this use. **Chosen:** a thin entry point that resolves positions and calls the shared loop, so the layer loop exists once. **Trade-off:** the entry point owns the fallback-slot clearing contract.

### 3.2 Explicit MRoPE positions for text-only input

The cache-offset rope path is mathematically identical but rounds differently in bf16; upstream hands `get_rope_index` positions to the layers, so the port does too.

### 3.3 Sampler callback in `video.rs`

`load_video_source_frames_at` takes a sampler over the probed frame count and keeps the existing probe caps and decoded-bytes budget, instead of adding a policy variant used by one caller.

## 4. Implementation Details

- `src/decision/clef.rs`: loading, media preparation, lexical vectors, readout.
- `src/decision/clef_head.rs`, `clef_layers.rs`: the head port.
- `src/decision/clef_sequence.rs`, `clef_media.rs`: sequence building and video geometry, including the capped schema encoding.
- `src/decision/clef_limits_tests.rs`: the security tests above.
- `src/multimodal/video_tests.rs`: `load_video_source_frames_at` tests (needs ffmpeg).
- `src/decision/loader.rs`, `src/models/detection.rs`: the `Clef` arm and the `generate` refusal.

## 5. Learning Points

### 5.1 Port error vs arithmetic noise

When a bf16 gap exceeds a bound, run both sides in f32 first. Agreement to four places there, plus upstream's own bf16/f32 spread, separates rounding order from a port error.

### 5.2 Spans come from separate encodes

Upstream concatenates separate tokenizer calls, so span positions depend on piece boundaries. Encoding each piece separately in the port is what makes the spans match exactly.

## 6. Related PRs and Issues

- Issue #2365: Clef / Clef-Flash decision model.
- Issue #2375: decision model epic.
- PR #3284: Decider-2b, which introduced `hidden_from_embeddings`.
- PR #3282: LiquidAI d1-3B decisions.

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 25 (vs. the Decider-2b base) |
| Lines added | +5500 |
| Lines deleted | -13 |
| Tests added | 20 (13 Clef unit, 4 limits, 1 parity, 1 detection, 1 video sampler) |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `661634e7` | feat | add Clef and Clef-Flash decision models (clef) |
| `c15c8d68` | fix | mark the Clef fixture generator executable |
| `6a0cb27a` | fix | bound Clef work before the token cap and refuse pad tokens |
| (final) | test/docs | cover the video sampler, document the security changes, re-validate |

## 8. Verification and Unverified Items

### Verified (final run, after the security commit)

- Release build on `models/clef-flash-mlx-8bit` through `mlxcel decide`, outputs identical to the pre-security run: text request `billing` 0.9757 with usage 135; image request `two` 0.9686 (remote true 0.9515) with usage 521; video request pattern true 0.9586 and kind `color bars test pattern` 0.9859 with usage 578. Audio is refused (`This model does not support audio input`), `mlxcel generate` is refused with the pointer to `mlxcel decide`, and a state containing `<|image_pad|>` with an image is refused with a 400-class message.
- `decision::` 95 passed (including the real-checkpoint parity test and `clef_limits_tests`), `models::detection_tests` 53 passed, `coverage_tests` 18 passed, `multimodal::video` 86 passed, plus `source_frames_at_decodes_the_sampler_picks_and_rejects_bad_ones` with `--include-ignored` (needs ffmpeg).
- clippy `-D warnings` on lib and tests, fmt, and the `scripts/ci` contract checks pass.
- `forward_final_hidden_with_embeddings` is covered through the real-checkpoint parity test, which runs the text, image and fixture paths.

### Unverified or Remaining

- `multimodal::video::tests::bench_single_pass_768_frames` (ignored, ffmpeg) fails on this host because ffmpeg rejects the 768-term `select` expression (`Cannot allocate memory`). The test is not part of this PR and the failure exists on the shared single-pass extractor, so a Clef clip long enough to sample hundreds of frames may hit it too. Not fixed here.

- NVFP4 / MXFP4 builds were not run; `clef-MLX-8bit` was load-and-smoke only.
- Video has no committed parity test (checked by hand at 7e-4).
- An unquantized Clef checkpoint (none is published) would load its backbone as f16 on Apple Silicon while the head stays bf16.
