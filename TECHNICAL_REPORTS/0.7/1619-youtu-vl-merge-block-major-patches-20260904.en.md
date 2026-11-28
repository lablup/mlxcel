# Technical Report: PR #1619 - fix(youtu_vl): emit patches in merge-block-major order

**Date**: 2026-09-04
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (validated on the real checkpoint at three fixture sizes; one further defect in the same path remains open as #1618)
**Languages**: Rust
**Risk Level**: Low (single-family preprocessing path; the divergence was provable inside the repository without an external oracle)

---

## Executive Summary

`YoutuVLProcessor::try_preprocess_with_spatial` emitted one row per image patch in plain raster order over the patch grid, with the inner feature order `(c, dy, dx)`. Two independent consumers expect merge-block-major rows with channel-last `(dy, dx, c)` features, so the vision tower received patches that did not correspond to the positions and groupings it assumes. A 224x224 solid orange square was described as "completely black". PR #1619 rewrites the emission loop to `(block_y, block_x, inner_y, inner_x)` with `(dy, dx, c)` features, and the same fixture is now described as "a solid, uniform block of bright orange color".

The defect was provable without any external oracle, because the encoder's own two inputs disagreed with each other. That is the transferable part of this report.

---

## 1. Problem Statement

### 1.1 Background

Youtu-VL's processor was ported in the shape of `src/vision/processors/qwen2_vl.rs`, which emits patches with the inner order `(c, dy, dx)`. That order is correct for Qwen2-VL, whose upstream processor unfolds patches over the `(C, H, W)` image tensor. The port kept it and also dropped Qwen2-VL's outer merge-block loop, leaving plain raster rows. The stale comment above the loop was the fingerprint: it claimed to "match how upstream unfolds patches via `unfold`", and this checkpoint does not unfold.

### 1.2 Existing Issues

Two consumers disagree with what the loop produced.

The checkpoint's own processor, `convert_image_to_patches` in [image_processing_siglip2_fast.py](https://huggingface.co/tencent/Youtu-VL-4B-Instruct/blob/main/image_processing_siglip2_fast.py), reshapes to `(C, nh/m, m, ps, nw/m, m, ps)` and permutes `(1, 4, 2, 5, 3, 6, 0)`. That is merge-block-major rows with channel-last features. `Siglip2VisionEmbeddings.patch_embedding` is an `nn.Linear(num_channels * patch_size * patch_size, embed_dim)` applied directly to those rows, and `remap_youtu_vl_weights` in `src/loading/vlm_youtu_vl.rs` only renames that weight; it never permutes it. Any other emission order therefore feeds the trained projection a permuted vector.

Separately, mlxcel's own encoder assumes the same block-major grouping. `YoutuVLVisionEncoder::rot_pos_emb` builds position ids with `reshape(&[h / merge, merge, w / merge, merge])` followed by `transpose_axes(&[0, 2, 1, 3])`, and `forward_with_spatial` reshapes hidden states to `[n_groups, spatial_merge_unit, dim]` and gathers along the group axis, which requires each consecutive run of `spatial_merge_size ** 2` rows to be one 2x2 block. With raster rows, every token carried the rotary position of a different spatial location and the merger combined four patches that were not a block.

### 1.3 Risk Assessment

The failure mode is fluent, confident, wrong output rather than a crash or a NaN, so nothing in the build or the test suite flagged it. The repository's own Qwen2-VL processor has a test pinning the correct grouping (`owned_and_mlx_paths_share_spatial_merge_grouped_patch_order`); Youtu-VL had no equivalent, which is why the port's divergence survived review.

---

## 2. Technical Review

### 2.1 Root cause

The emission loop iterated a single flat patch index and reconstructed `(py, px)` from it:

```rust
for patch_idx in 0..total_patches_img {
    let py = patch_idx / w_patches as usize;
    let px = patch_idx % w_patches as usize;
    // ...
    for c in 0..in_channels {
        for dy in 0..self.patch_size {
            for dx in 0..self.patch_size {
```

Both the outer row order and the inner feature order are wrong for this checkpoint, and the two errors are independent: fixing only one still feeds the projection a permuted vector.

### 2.2 Why this was provable without an oracle

The encoder's two inputs contradicted each other inside this repository. `rot_pos_emb` and the merge-unit gather are block-major; the processor feeding them was raster. No reference implementation, no mlx-vlm venv and no logit trace was required to establish that one of them had to be wrong. Upstream's `convert_image_to_patches` then settled which one.

### 2.3 The fixture that isolates this defect from its neighbour

This family had a second, already-fixed defect in the same path: the window inverse, corrected to `argsort(window_index)` in #1600 / #1603. Attributing the orange-reads-black symptom required a fixture the window permutation provably cannot affect. At 224x224 the merged grid is 7x7, which fits inside a single 8x8 attention window, so `get_window_index` is the identity there and applying it twice is a no-op. A wrong description at that size therefore cannot be the window inverse's fault, and the patch order was the remaining candidate.

---

## 3. Technical Decisions

### 3.1 Mirror Qwen2-VL's loop shape, not its inner order

`src/vision/processors/qwen2_vl.rs` already has the correct four-level block loop, so the outer structure is copied from it. Its `(c, dy, dx)` inner order is deliberately not copied: that order is correct for Qwen2-VL and wrong here, and the difference is now stated in the comment so the next port does not repeat the substitution in either direction. `qwen2_vl.rs` was not modified.

### 3.2 Check the divisibility invariant instead of relying on it

`smart_resize` rounds both edges to a multiple of `patch_size * spatial_merge_size`, so `h_patches` and `w_patches` are always exact multiples of the merge factor and the block loops need no padding. Rather than rely on that silently, the loop now returns a new `YoutuVLPreprocessError::UnalignedPatchGrid` when it does not hold. A future change to the resize policy would otherwise drop the trailing partial block without any signal.

### 3.3 Write the known-failing assertions at the correct answer

`tests/youtu_vl_parity.rs` carries the two multi-window fixtures as `#[ignore]`d content assertions written at the full correct answer and labelled as known failing against #1618, rather than weakened to something this build satisfies. Weakening them would have recorded the current wrong output as the expected output, which is the specific anti-pattern that lets a defect become the specification.

---

## 4. Implementation Details

### 4.1 Key code change

```rust
let merge = self.spatial_merge_size;
// ... divisibility check, returning UnalignedPatchGrid ...
let mut row = 0usize;
for block_y in 0..hp / merge {
    for block_x in 0..wp / merge {
        for inner_y in 0..merge {
            for inner_x in 0..merge {
                let py = block_y * merge + inner_y;
                let px = block_x * merge + inner_x;
                // ... row_start from `row`, then:
                for dy in 0..self.patch_size {
                    for dx in 0..self.patch_size {
                        for c in 0..in_channels {
```

### 4.2 The unit test separates the two orderings

`patches_are_emitted_merge_block_major_with_channel_last_features` builds a 4x4 patch grid, which is 2x2 merge blocks and the minimum size that distinguishes block-major from raster: at 2x2 patches the two orders coincide. It asserts the row order is `0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15`, deliberately not `0..15`, so raster emission fails it.

The inner order is pinned in the same pass by encoding three independent values in the three channels: red carries the patch id, green carries `dy` and blue carries `dx`. Reading feature `(dy * patch_size + dx) * 3 + c` and checking all three therefore fails immediately under a `(c, dy, dx)` layout.

---

## 5. Validation

Measured on `models/mlx/youtu-vl-4b-instruct`, M1 Ultra, release build with `metal,accelerate`, greedy (`-t 0 -n 48`).

| Fixture | Before | After |
|---|---|---|
| `test_image.png`, 224x224 solid orange | "The image is completely black and contains no visible content, objects, text, or details." | "The image is a solid, uniform block of bright orange color." |
| `test_image_shapes.png`, 448x448 | "a single, solid black circle on a white background" | "a single, solid black circle on a plain white background" |
| `test_image_shapes_336.png`, 336x336 | "a single, solid black circle on a white background" | "a single white circle on a black background" |

The solid-colour fixture is the acceptance criterion. The two shape fixtures change and remain wrong, which is the expected outcome: at least one further defect remains in this family's vision path and is filed as #1618 rather than folded in here.

Gates: `cargo test --workspace --profile test-fast --features metal,accelerate` passed 10502 tests with 0 failures; `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`, `cargo fmt --all -- --check` and `scripts/ci/check_cross_repo_refs.py` all clean.

---

## 6. Change Summary

### Statistics

| Metric | Value |
|---|---|
| Files changed | 3 |
| Lines added | 364 |
| Lines removed | 19 |

### Changes by category

- `src/vision/processors/youtu_vl.rs`: emission loop rewritten, stale `unfold` comment replaced with one naming `convert_image_to_patches` and linking the upstream file, new `UnalignedPatchGrid` error variant.
- `src/vision/processors/youtu_vl_tests.rs`: checkpoint-free test pinning both the row order and the inner feature order.
- `tests/youtu_vl_parity.rs`: new, in the shape of `tests/qwen2_5_vl_parity.rs`; a passing 224 control and two `#[ignore]`d multi-window assertions tracked by #1618.

### Related issues

Closes #1610. Follows #1600 / #1603 (the window inverse in the same family, which this defect masked). Mirrors #1596 / #1601, the Qwen2.5-VL pair of defects. Opens #1618 for the residual multi-window defect. Related: #1611, the processor's patch-count cap.

---

## 7. Follow-up Actions

### Transferable lesson

When a VLM port produces fluent but wrong descriptions, check whether the port's own components already disagree before reaching for an external oracle. Here the processor's row order and the encoder's `rot_pos_emb` were derivable from the same repository and contradicted each other, which localises the defect at zero cost.

The second half is the control fixture. A change that produces a byte-identical result is a finding rather than a failed experiment, and the reverse holds too: a fix that corrects one fixture and not others is evidence about how many defects are present. Keeping the 224 single-window case alongside the multi-window ones is what makes the residual #1618 visible instead of leaving "the fix did not work" as the conclusion.

### Open

#1618 tracks the remaining defect. The evidence recorded there is that the only correct fixture is the only one whose merged grid fits in a single attention window, both incorrect ones span 2x2 windows with a partial row and column, and they are wrong differently from each other.
