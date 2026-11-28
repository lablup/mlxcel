# Technical Report: PR #1601 - fix(qwen2_5_vl): f32 tower norm and a correct window inverse

**Date**: 2026-09-03
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (localized against a transformers float32 oracle at two image sizes; validated end to end on both local Qwen2.5-VL checkpoints; the full workspace gate runs centrally)
**Languages**: Rust, Python, Markdown
**Risk Level**: Medium (changes the numerics of every Qwen2.5-VL and ColQwen2.5 image forward, and the token order of the merged vision tokens at any non-involution grid)

---

## Executive Summary

Every image mlxcel sent through the Qwen2.5-VL vision tower came back partly erased. The tower's RMSNorm reduces `sum(x * x)` over a 1280-wide feature axis, and Qwen2.5-VL carries massive activations: one channel of the residual stream reaches 5.4e2 by block 15 and 2.3e4 by block 31. In float16 that sum saturates at 65504, the norm returns `rsqrt(inf) = 0`, and the whole token row leaves the block as zeros. On a 448x448 image, 15 of 1024 tower tokens were erased at block 16 and 21 at block 17, and the damage compounded through the remaining blocks.

The reported symptom was mild: a three-shape image described as "The image contains a red square and a green triangle.", dropping the blue circle. The real blast radius was not. The repository's own fixture, a 224x224 solid orange square, was described as "a person wearing a white shirt and black pants, standing in front of a white wall with a black and white patterned rug on the floor". Every VLM test that drives that fixture asserts only that tokens were produced, so the defect had no coverage at all.

A second, independent defect sat after the patch merger. The un-reorder that returns merged tokens from window order to raster order built `reverse_indices[orig_idx] = rank`, which for a permutation of `0..N` reproduces `window_index` itself rather than `argsort(window_index)`. Applying a permutation twice restores the original order only when it is an involution, which a 16-wide merged grid happens to satisfy, so the 448x448 case in the report was unaffected while 120 of 144 merged tokens were misplaced at 336x336.

`VisionRMSNorm::forward` now promotes a float16 input to float32 for the reduction only, and `invert_window_index` builds the inverse directly. The tower's projections and attention still run in float16; the cost of the promotion is inside run-to-run noise.

---

## 1. Problem Statement

### 1.1 Background

Issue #1596 reported that a synthetic three-object image (red square, blue circle, green triangle on light gray) was described as "The image contains a red square and a green triangle." under greedy decoding, on both the mlx-community 4-bit conversion and the raw bf16 export of `Qwen/Qwen2.5-VL-3B-Instruct`. transformers named all three objects for the same PNG and prompt.

Two facts in the report narrowed the search before any code was read. The raw bf16 export and the 4-bit conversion produced the identical sentence, so quantized weight numerics were not the cause; and the output was byte-identical before and after PR #1582, so the raw-HF loader fix was not either.

A third fact came out of the investigation and explains the first: the mlx-community 4-bit conversion **does not quantize the vision tower**. Its `model.safetensors.index.json` carries `vision_tower.blocks.N.*.weight` with no matching `.scales`, so both checkpoints run the same bf16 tower, which `finish_vlm_weights_common` then converts to float16 on Apple Silicon. The two dumps of block 16 were byte-identical, which is why the two checkpoints agreed with each other and disagreed with transformers.

### 1.2 Localization

The stage-by-stage diff the issue asked for was run against a transformers **float32** oracle so the reference's own bf16 rounding could not mask a real gap. Taps were placed on both sides at the processor output, the patch embedding, the window bookkeeping and rotary table, every vision block, the patch merger before and after the un-reorder, the merged input embeddings, the MRoPE position ids, and the prefill logits. Two image sizes were used: 448x448 (16x16 merged grid, an involution) and 336x336 (12x12 merged grid, not an involution).

`rel` below is mean absolute difference over the oracle's mean absolute value, at 448x448 before the fix:

| Stage | Result |
|---|---|
| Processor `pixel_values`, `grid_thw`, image-token count | rel 6.8e-8, `(1, 32, 32)`, 256 tokens: identical |
| `PatchEmbed` `[1024, 1280]` | rel 2.2e-4: float16 rounding floor |
| `window_index`, `cu_window_seqlens`, `cu_seqlens` | integer-identical |
| `rot_pos_emb` `[1024, 40]` | exactly 0.0 |
| Vision blocks 0 to 14 | rel 4.6e-4 rising to 3.3e-3, flat |
| Vision block 15 | rel 3.5e-3, max absolute 4.06 (the massive activation first appears here, 536) |
| **Vision block 16** | **rel 3.8e-2, max absolute 94.2** |
| **Vision block 17** | **rel 2.8e-1, max absolute 6.4e3** |
| Vision block 31 | rel 8.0e-1, max absolute 1.9e4 |
| `PatchMerger` output | rel 9.9e-1 |
| MRoPE position ids, `rope_deltas` | integer-identical, -240 |

The first diverging stage is **block 16**. Everything upstream of it, including all the index arithmetic the issue flagged as a suspect, was already correct.

Running the identical tower with float32 activations (float16 weights, MLX type promotion) brought every block to **rel 8e-6**, flat from block 0 to block 31, with `max|.|` agreeing to 23410.50 against 23410.75. That answers the precision-versus-structure question the issue posed: precision.

### 1.3 The overflowing operation

Intra-block taps on blocks 15, 16 and 17 put the first divergence inside `norm1`, not in attention or the MLP. At block 16, `max|diff|` on the norm output was 11.56 against a `max|.|` of 12.10, which is a full-magnitude error on some rows rather than a rounding error on all of them.

Inspecting those rows in the float16 dump: **15 rows of 1024 came out of block 16's `norm1` as exact zeros, and 21 at block 17.** Their sum of squares at the norm's input separates cleanly on the float16 ceiling:

| | `sum(x * x)` over the 1280-wide axis |
|---|---|
| Smallest erased row | 67 332 |
| Largest erased row | 348 750 |
| Largest surviving row | 54 696 |
| float16 maximum | 65 504 |

No surviving row exceeds 65504 and no erased row falls below it. The reduction saturates to infinity in float16 and the norm returns `rsqrt(inf) = 0`.

Nothing else in the block overflows. The norm outputs themselves stay under 17, so the QKV and MLP projections see small inputs, and MLX accumulates matmuls in float32. The SwiGLU `down_proj` is what *creates* the massive activation (its output reaches 6359 at block 17), but it creates it correctly; only the next norm that has to reduce it fails.

### 1.4 The second defect

`forward_with_grid` un-reordered the merged tokens by sorting `(value, index)` pairs and writing `reverse_indices[orig_idx] = rank`. For a permutation of `0..N` the element with value `v` always lands at rank `v`, so the loop degenerated to `reverse_indices[i] = window_index[i]`: it reproduced the permutation instead of inverting it. Upstream builds `reverse_indices = mx.argsort(window_index, axis=0)` (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen2_5_vl/vision.py).

Applying a permutation twice is the identity only for an involution. Measured from the dumped merger output:

| Merged grid | Image | Involution | Tokens misplaced | Merger tap versus the oracle |
|---|---|---|---|---|
| 16x16 | 448x448 | yes | 0 of 256 | rel 3.3e-3 either way |
| 12x12 | 336x336 | no | 120 of 144 | rel 8.5e-1 old, rel 3.2e-3 with `argsort` |

That is why the size in the report could not expose it, and why the issue was right to require the stage diff at a second size.

### 1.5 Consequences

- **Dropped and mislabelled objects.** The reported symptom.
- **Fabricated content.** With enough rows erased the language model has no grounded signal and invents a scene. The solid orange fixture produced a person, a shirt, a wall and a rug.
- **Silent.** Output stayed finite and fluent throughout. No NaN, no error, no log line.
- **Uncovered.** Every existing VLM test asserts non-empty or finite output against a flat-color fixture, which passes on a fully hallucinated description.

---

## 2. Change Summary

| File | Change |
|---|---|
| `src/vision/encoders/qwen2_5_vl.rs` | `VisionRMSNorm::forward` promotes a float16 input to float32 for the reduction and returns float16. `invert_window_index` replaces the incorrect inverse construction, bounds-checked. `get_window_index`'s body is extracted as the free function `window_index_for_grid`. |
| `src/vision/encoders/qwen2_5_vl_tests.rs` | Four checkpoint-free tests for the window inverse and one for the float16 reduction. All five fail on the pre-fix code. |
| `tests/qwen2_5_vl_parity.rs` (new) | Three `#[ignore]` real-checkpoint tests asserting on description content. All three fail on the pre-fix code. |
| `tests/fixtures/test_image_shapes.png`, `test_image_shapes_336.png`, `generate_test_image_shapes.py` (new) | A three-object fixture at an involution size and a non-involution size, reproducible from the committed generator (Pillow 12.3.0). |

---

## 3. Technical Decisions

### 3.1 Promote the reduction, not the tower

The float32 diagnostic run is a correct fix and a bad one: it doubles activation memory for a 32-block 1280-wide tower and gives up float16 throughput on every projection, to solve a problem that occurs in one operation. The promotion is therefore scoped to `VisionRMSNorm::forward`, which is where the range is needed and nowhere else.

The cost was measured with both arms compiled into one binary, toggled by an environment variable, so no build-layout variance leaks into the comparison. Minimum over 30 tower forwards:

| Image | With promotion | Without |
|---|---|---|
| 448x448 | 144.525 ms | 144.201 ms |
| 336x336 | 95.185 ms | 95.196 ms |

Two casts of a `[N, 1280]` tensor are negligible next to 32 blocks of 1280-wide projections, and the difference is inside noise (the 336 arm is nominally faster with the promotion). Decode is untouched: the tower runs once per image at prefill. An earlier attempt to measure this from CLI wall-clock was discarded: the text-only baseline moved 77 ms between two builds of the same source, so process-level timing cannot resolve a sub-millisecond change.

### 3.2 Do not change `mlxcel_core::rms_norm`

The saturating reduction is a property of the shared `rms_norm` entry point, used by every family in the workspace. Changing its dtype behavior there would alter the numerics of every float16 text and vision model in one commit, on the strength of one family's measurement. This PR states the finding and fixes only Qwen2.5-VL's own private `VisionRMSNorm`, which is reachable from ColQwen2.5 through the same encoder and from nowhere else. Whether other float16 families reach the same ceiling is a real question this change deliberately does not answer.

### 3.3 Do not keep the tower in bf16

Loading the tower's weights as bf16 instead of float16 would also fix it, since bf16 has float32's exponent range. It was rejected: it edits `finish_vlm_weights_common` in the shared VLM loader, changes the dtype of the merger output crossing into the text embedding space, trades 11 mantissa bits for 8 across the whole tower, and is a larger behavior change than the defect requires. The RMSNorm promotion keeps float16's precision everywhere it is safe and buys range only where it is not.

### 3.4 Guard the inverse rather than assume the permutation

`window_index_for_grid` returns a permutation of `0..N` by construction, so an unchecked `inverse[window_index[rank]] = rank` would be correct. The index is bounds-checked anyway: an out-of-range value would otherwise panic inside the vision tower on a malformed grid, and the guard costs nothing on the real path.

### 3.5 Extract `window_index_for_grid`

The window permutation depends only on the grid and three configuration integers, but it lived as a method on an encoder that needs a checkpoint to construct. Lifting the body to a free function makes the property that matters, "the un-reorder is the inverse", testable with no weights, which is what the new unit tests exercise.

### 3.6 Commit two fixtures, not one

The issue asked for one 448x448 fixture. A second 336x336 rendering is committed because 448x448 is precisely the size at which the un-reorder defect is invisible: a regression test built only on it would not catch a reintroduction. Both come from one generator, and the 448 output is byte-identical to the image in the issue's reproduction snippet.

### 3.7 Assert on content, and allow synonyms

`tests/qwen2_5_vl_parity.rs` asserts that the description names each object and each color, matching case-insensitively against groups of interchangeable words: the red shape is legitimately either a "square" or a "rectangle" depending on the rendered size, and pinning one would make the test brittle for no gain. The solid-color test additionally asserts the *absence* of the specific words the pre-fix hallucination produced, because "mentions orange" alone would pass on a description that mentions orange inside an invented scene.

---

## 4. Validation

All on an Apple M1 Ultra, `metal,accelerate`, greedy (`-t 0`), 48 new tokens, prompt "What shapes and colors are in this image? Answer briefly." unless noted.

| Case | Before | After | Oracle |
|---|---|---|---|
| 448x448, bf16 export | The image contains a red square and a green triangle. | square, circle, triangle, red, blue, green | square, circle, triangle, red, blue, green |
| 448x448, 4-bit | same as above | The image contains a red square, a blue circle, and a green triangle. | as above |
| 336x336, bf16 export | The image contains a red circle and a blue triangle. | rectangle, circle, triangle, red, blue, green. | rectangle, circle, triangle; red, blue, green |
| 336x336, 4-bit | A red circle and a blue triangle. | The image contains a red square, a blue circle, and a green triangle. | as above |
| 1036x1036, 4-bit (37x37 merged, padded windows) | Green triangle and blue square. | Red square, blue circle, green triangle. | Red square, blue circle, green triangle. |
| Two images, 448x448 plus 224x448, 4-bit ("Describe each image briefly, in order.") | four invented images, the first repeated | names all four real objects across both | `grid_thw` `[[1, 32, 32], [1, 32, 16]]` matches |
| `tests/fixtures/test_image.png`, 4-bit ("Describe this image briefly.") | The image shows a person wearing a white shirt and black pants ... | The image is a solid orange color. | not applicable |
| `qwen2-vl-2b-4bit`, same image | rectangle, circle, triangle | rectangle, circle, triangle | unchanged, as expected for a separate encoder file |

The 448x448 bf16 result is **token-identical** to the transformers greedy output. The 1036x1036 4-bit result is byte-identical to the oracle. The 4-bit rows differ from the oracle in phrasing rather than content, which is expected: the language model is 4-bit quantized while the oracle is bf16.

After the fix, every tap (a) through (g) agrees at both sizes: processor rel 6.8e-8, patch embedding rel 2.2e-4, window and MRoPE indices integer-identical, worst vision block rel 5.4e-3, merger rel 3.3e-3 at 448 and 3.2e-3 at 336, merged image slots rel 3.3e-3 and 3.2e-3. Tap (h) is exact on the bf16 export.

Suites: `--lib vision::encoders::qwen2_5_vl` 9 passed, `--lib vision::processors::qwen2_vl` 4 passed, `--lib models::colqwen2_5` 6 passed, `--lib loading::vlm` 197 passed, `--test qwen2_5_vl_parity -- --ignored` 3 passed, `--test vlm_concurrency qwen2_5_vl -- --ignored` passed, `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` clean, `cargo fmt --all -- --check` clean. Regenerating the fixture and `cmp`-ing it against the committed PNG is byte-identical.

---

## 5. Learning Points

**A dtype that fits the value does not fit the reduction.** float16 represents 23410 fine. It does not represent 23410 squared, and an RMSNorm has to. When a family is known to carry massive activations, the operation to audit is not the one holding the large value but the next one that has to square or accumulate it.

**A saturating reduction fails loudly in the math and silently in the output.** `rsqrt(inf) = 0` is a finite, well-formed zero. There is no NaN to trip a guard, no exception, no non-finite check to fail. Fifteen erased rows out of 1024 still leave a fluent sentence.

**Two checkpoints agreeing is not two independent measurements.** The issue treated the 4-bit conversion and the bf16 export as independent evidence against a quantization cause. They are stronger than that and weaker: the mlx-community conversion leaves the vision tower unquantized, so both ran the same tower bit for bit. Checking the weight index for `.scales` under the tower prefix would have established that in one command.

**A permutation bug can be invisible at exactly the size you test at.** An involution is its own inverse, and the 16-wide merged grid of a 448x448 image is one. Any test for an inverse permutation has to run at a size where the permutation is not self-inverse, and the new unit tests keep both cases so the distinction stays documented.

**A fixture that cannot be wrong cannot catch anything.** A solid orange square admits exactly two failure modes a test can see: no output, and non-finite output. It cannot see a hallucination. The cost of that gap here was a defect that mangled every image the runtime processed for this family, sitting under a green suite.

**Measure a sub-millisecond change inside one binary.** Comparing two builds by CLI wall-clock produced a 77 ms swing on a text-only baseline that neither build should have changed. An environment-variable A/B inside a single binary resolved the same question to 0.3 ms.

---

## 6. Follow-ups and what remains unverified

- **#1600**: `reverse_window_indices` in `src/vision/encoders/youtu_vl_window.rs:147-159` carries the identical incorrect inverse, used by `youtu_vl.rs:425`. It is a separate family with its own checkpoint and validation path and was not fixed blind here.
- **Other float16 families.** This change does not survey which other models drive `mlxcel_core::rms_norm` with rows whose energy crosses 65504. The shared reduction is unchanged, so any such family is still exposed.
- **Video and multi-frame input.** Every measurement here is on still images with `t = 1`. The window permutation code paths for `grid_t > 1` are exercised only by the unit tests, not against an oracle.
- **The two divergences the issue scoped out** were confirmed inert at the sizes tested and left alone: the processor resamples with Lanczos3 where `Qwen2VLImageProcessor` uses bicubic, and the loader builds the processor from code defaults rather than `preprocessor_config.json`. Both 448 and 336 are multiples of 28, so `smart_resize` performs no resample, and this checkpoint's sidecar carries the same `min_pixels` and `max_pixels` as the defaults. A non-multiple-of-28 size would exercise the resampler, and that measurement has not been made.
