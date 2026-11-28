# PR #3490: Close the remaining vision resize and frame-sampling parity gaps (issue #3478)

**Date**: 2026-11-27

**Status**: Merged as `e705ac818`. 8 files, +312 / -65. Validated locally on an M3 Ultra; remote CI was cancelled by maintainer decision. No ERNIE or Hunyuan-VL checkpoint was run end to end.

**Risk Level**: Low to Medium (patch grids and sampled frames change for thin images, extreme aspect ratios and exact `.5` frame positions; everything else is unchanged)

## Executive Summary

Hardening #3464 and #3466 left four places where mlxcel's image resize or video frame selection still differed from the Python reference. This PR closes them: frame indices round half to even like `np.round`, the Qwen2-VL family no longer floors the first rounding step, ERNIE 4.5 VL follows the `processing_ernie4_5_vl.py` its checkpoints ship (including the 200:1 ratio clamp), and Hunyuan-VL no longer overflows on a zero edge. A different grid changes the number of vision tokens and their positions, so these gaps produced different model input, not just a different pixel count.

## 1. Problem Statement

| Gap | Before | Reference |
|---|---|---|
| Frame sampling (`uniform_indices`, `evenly_spaced_indices`, Gemma 4 `uniform_sample_frames`, Inkling `pair_adjacent_frame_indices`) | `f64::round`, half away from zero: `uniform_indices(6, 3)` = `[0, 3, 5]` | `np.linspace(...).round()` and Python `round`, half to even: `[0, 2, 5]` |
| Qwen2-VL, dots.ocr, Hunyuan-VL thin edge | first step `max(1, round(e / f)) * f`: 10x1000 at factor 28 kept 28x1008 | transformers `round(e / f) * f` reaches 0, drops under `min_pixels`, rescales: 28x560 |
| ERNIE 4.5 VL past 200:1 | no clamp: 10x3000 gave 28x2996 | checkpoint code clamps the ratio: 28x5600 |
| Hunyuan-VL zero edge | u32 arithmetic overflowed in `smart_resize(0, 10)` | reference raises; mlxcel needs a finite answer |

## 2. Technical Decisions

### 2.1 Which reference ERNIE follows

ERNIE has three candidate references and they disagree.

| Option | Thin edge 10x1000 | 4x15 under `min_pixels` | 10x3000 past 200:1 |
|---|---|---|---|
| transformers 5.16.1 `ernie4_5_vl_moe` | 28x560 (no floor) | 56x112 | raises |
| mlx-vlm port | 28x1008 | 28x112 (`int()` before `ceil`) | 28x2996 |
| **Chosen: checkpoint `processing_ernie4_5_vl.py`** | 28x1008 | 56x112 | 28x5600 |

Every ERNIE 4.5 VL checkpoint (`baidu/ERNIE-4.5-VL-28B-A3B-PT`, `-Thinking`, and the validated `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`) names `Ernie4_5_VLImageProcessor` through `auto_map`, so the file they ship is what produced the training-time grids. transformers' class is loaded by no ERNIE checkpoint. The consequence is that item 2 (dropping the first-step floor) deliberately does not apply to ERNIE.

### 2.2 Where the one-factor floor still applies

Removing the first-step floor lets an edge reach 0 before the bounds check, which is what the reference does. The final result is still floored at one factor for the `min_pixels == 0` case, where the reference would return a zero edge and the vision encoder would get an empty grid. That deviation is commented in each function.

### 2.3 The 200:1 trim moved after both branches in Qwen2-VL

transformers raises past 200:1; mlxcel trims the long side so `max_image_tokens` still holds. Before this PR the trim ran only in the `max_pixels` branch. With the floor gone, a thin edge can reach the `min_pixels` branch with a long side over a small cap, so the trim now runs after either branch. `smart_resize_extreme_aspect_under_min_pixels_keeps_the_cap` covers it.

### 2.4 Arithmetic width

ERNIE and Hunyuan moved to u64 with `saturating_mul`, and a zero edge is treated as one pixel so `beta` stays finite. `smart_resize(0, 10)` on Hunyuan now returns 192x1632 instead of overflowing. Results are narrowed back to u32 with `try_from(..).unwrap_or(u32::MAX)`.

### 2.5 MiniMax-M3 left alone

Its mlx-vlm reference also has `max(factor, round_by_factor(..))`, so the floor there is parity, not a gap.

## 3. Change Summary

| File | Change |
|---|---|
| `src/multimodal/video.rs` | `uniform_indices`, `evenly_spaced_indices`, `pair_adjacent_frame_indices` round half to even |
| `src/vision/processors/gemma4.rs` | `uniform_sample_frames` rounds half to even |
| `src/vision/processors/qwen2_vl.rs` | first-step floor removed, 200:1 trim after both branches |
| `src/vision/processors/dots_ocr.rs` | first-step floor removed |
| `src/vision/processors/hunyuan_vl.rs` | first-step floor removed, zero edge as one pixel, saturating u64 |
| `src/vision/processors/ernie4_5_vl.rs` | checkpoint-code resize with ratio clamp, u64 |
| `video_tests.rs`, `gemma4_tests.rs` | expected-index helper rounds half to even; new tie cases |

No existing expected value changed. The only edit to an existing test is the `load_video` decode helper, where index 12 becomes 24 and stays inside the color tolerance. Every new value was computed with the Python reference (transformers 5.16.1 `smart_resize`, the checkpoint ERNIE file, numpy) and each test's doc comment shows the arithmetic.

| Commit | Message |
|---|---|
| `80d8f7c93` | fix: close remaining vision resize and frame-sampling parity gaps |

## 4. Validation

| Check | Result |
|---|---|
| `vision::processors::qwen2_vl` | 14 passed |
| `vision::processors::dots_ocr` | 6 passed |
| `vision::processors::ernie4_5_vl` | 8 passed (ratio-clamp test fails on main) |
| `vision::processors::hunyuan_vl` | 4 passed |
| `vision::processors::gemma4` | 21 passed, 1 ignored |
| `multimodal::video` | 57 passed, 11 ignored; 68 passed with the ffmpeg fixture tests included (`MLXCEL_TEST_VIDEO=1 ... --include-ignored`) |
| clippy `-D warnings`, fmt | clean |

## 5. Follow-up

- No ERNIE 4.5 VL or Hunyuan-VL checkpoint was available locally, so the new grids were not checked against generated output.
- The ERNIE thin-image and 4x15 tests pass on main by design; they pin the checkpoint behaviour so a later "parity with transformers" change does not silently move ERNIE.
- Any new Qwen2-VL-style processor should copy the floor-free first step, not the pre-#3490 code.
