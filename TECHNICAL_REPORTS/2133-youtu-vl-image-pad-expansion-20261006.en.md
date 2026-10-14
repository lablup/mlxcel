# Technical Report: PR #2133 - fix(youtu_vl): expand the template image placeholder per merged feature

**Date**: 2026-10-06
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (validated on the real checkpoint on CUDA through the CLI and mlxcel-server)
**Languages**: Rust
**Risk Level**: Low (one family's prompt preparation; the new mismatch check turns a silent truncation into a request error)

---

## Executive Summary

After #1610 fixed the patch emission order, Youtu-VL still described both three-shape fixtures as "a single circle" while the solid orange control read correctly. Issue #1618 pointed at the windowed-attention path because the only correct fixture was the only single-window one. The real cause was upstream of the vision tower: the chat template emits one `<|image_pad|>` per image, mlxcel never expanded it to one token per merged feature, and the masked scatter therefore put only the first feature into the prompt. PR #2133 expands the placeholder the way the checkpoint's processor does and makes a count mismatch an error. CLI and server output now match the transformers greedy output word for word on all three fixtures.

---

## 1. Problem Statement

### 1.1 Background

`YoutuVLProcessor.__call__` in the checkpoint's `processing_youtu_vl.py` replaces each `<|image_pad|>` in the rendered text with `h * w / 4` copies, one per merged vision feature. mlxcel's equivalent, `insert_youtu_vl_image_tokens`, was written for callers that pass no placeholder at all and splices a framed run after BOS. Its guard returned early whenever the prompt already contained an image token, on the assumption that such a prompt was already expanded.

### 1.2 Existing Issues

The chat template always renders the placeholder, so the guard fired on every templated request. The prompt reached the tower with one image token (`128262, 128264, 128263`) against 49, 121 or 196 features. `merge_llava`'s masked scatter fills placeholders in order and drops what does not fit, so the language model saw exactly one feature: the top-left merged patch.

### 1.3 Why it looked like a window defect

A uniform image produces the same content in every merged token, so the 224 orange fixture read correctly from its one surviving token. The two shape fixtures both have light grey corners, which the model turned into "a single circle on a plain background" in two polarities. Grid size and window count correlated with correctness only because the uniform fixture happened to be the small one.

---

## 2. Technical Review

### 2.1 Stage-by-stage diff

A transformers 4.56.0 oracle (the checkpoint's remote code, f32 on CPU) produced the reference processor output and the tower plus merger output. mlxcel's processor matched to 3e-8 at 224 and 448 (336 differs only by Lanczos versus bilinear resampling). Feeding the reference pixels through mlxcel's bf16 tower gave per-merged-token cosine mean 0.9999, 0.998 and 0.997 at 14x14, 22x22 and 28x28 patch grids. That rules out `get_window_index`, the `cu_window_seqlens` padding and deduplication, the full-attention boundaries, and the window inverse, which were the issue's candidates.

### 2.2 Locating the defect

The reference generated the correct answers in bf16, and the vision features matched, so the divergence was after the tower. Dumping the prompt ids showed one image token where the reference has 196.

---

## 3. Technical Decisions

### 3.1 Mirror the processor's replace loop

One placeholder per image is expanded in place, keeping the template's own start/end framing. A prompt that already carries one token per feature is left alone, checked first so a grid whose runs are a single token is not expanded twice. No placeholder keeps the existing splice-after-BOS fallback. Any other count is an error rather than a guess.

### 3.2 Fail on a placeholder/feature mismatch

Upstream raises when image tokens and features disagree. `YoutuVLModel::get_input_embeddings` now does the same, so this class of defect surfaces as a request error instead of a fluent description of one patch.

---

## 4. Validation

| fixture | before | after (CLI and server) |
|---|---|---|
| 224 orange | "a solid, uniform orange color" | unchanged |
| 336 shapes | "a single white circle on a black background" | red square, blue circle, green triangle (identical to transformers) |
| 448 shapes | "a single, solid black circle on a plain white background" | "- Red square - Blue circle - Green triangle" (identical to transformers) |

The two formerly known-failing tests in `tests/youtu_vl_parity.rs` pass on the real checkpoint. Server `prompt_tokens` are 82, 154 and 229, equal to the reference processor's counts. New unit tests cover expansion on a 28x28 grid, multi-image ordering, the mismatch error, and `get_window_index` against reference values on multi-window grids.

---

## 5. Change Summary

- `src/multimodal/youtu_vl_prompt.rs`, `youtu_vl_prompt_tests.rs`: placeholder expansion and tests.
- `src/vision/youtu_vl.rs`: count check before the scatter.
- `src/multimodal/vlm_runtime.rs`: both errors propagate as request errors.
- `src/vision/encoders/youtu_vl_tests.rs`: window-index reference test.
- `tests/youtu_vl_parity.rs`: un-gated, comments corrected.

Related: #1618 (closed by this PR), #1610, #1600, #1611.

---

## 6. Follow-up Actions

### Transferable lesson

A solid-color fixture cannot detect a defect in how features reach prompt slots: any single feature carries the whole image. Content tests need a spatially varied fixture, and a VLM port should assert placeholder count equals feature count rather than relying on a scatter that truncates.

### Open

The processor's `smart_resize` rounds edges to a multiple of 32 where the reference rounds up, so some input sizes other than these fixtures get a different patch grid. #1611's premise (cap at `max_num_patches` = 256) is contradicted by `YoutuVLProcessor.__call__`, which passes `max_image_patches=36864` and overrides the preprocessor config.
