# Technical Report: PR #1603 - fix: build the Youtu-VL window inverse as `argsort`, not the permutation itself

**Date**: 2026-09-03
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (arithmetic proven by checkpoint-free tests and by enumeration over every reachable grid; validated end to end on a local Youtu-VL checkpoint, which showed no content change because a second defect in the same family dominates, filed as #1610)
**Languages**: Rust, Markdown
**Risk Level**: Low (changes the token order of the merged vision tokens for Youtu-VL only, at any grid larger than a single attention window; no other family shares the helper)

---

## Executive Summary

`reverse_window_indices` in the Youtu-VL vision encoder was supposed to return `argsort(window_index)` so the encoder could put merged vision tokens back into raster order after the patch merger. It returned `window_index` itself. It sorted `(value, index)` pairs and wrote `reverse[orig_idx] = rank`, and for a permutation of `0..N` the element with value `v` always sorts to rank `v`, so the loop collapses to `reverse[i] = window_index[i]`. The call-site comment claimed the result was equivalent to `argsort(window_index)`, so the code and the comment disagreed and the comment was the one that was right about the intent.

Applying a permutation twice restores the original order only when the permutation is an involution. This is the same defect #1601 fixed for Qwen2.5-VL, but the two families are exposed very differently, and that difference is the part worth carrying forward. A Youtu-VL window is 8x8 merged tokens, and `build_processor` caps the processor at `vision_config.num_patches` (4096), so the merged grid runs from 1x1 to 32x32. Enumerating every grid in that range, the permutation is an involution only when the entire image fits in one window. There is no second safe size: every merged grid from 9x9 through 32x32 is affected, up to 960 of 1024 tokens misplaced at 1024x1024.

The helper now builds `inverse[window_index[rank]] = rank` directly, bounds-checked so a malformed grid cannot panic, matching the `invert_window_index` that #1601 landed for Qwen2.5-VL.

The end-to-end validation is the part a future maintainer should read first. On `tencent/Youtu-VL-4B-Instruct` the generated text is **byte-identical before and after this change** at both 448x448 and 336x336, and wrong in both. The reason is not that the fix is inert: it is that a second, larger defect upstream of the window reordering corrupts the vision path at every image size. That defect was root-caused during this work and filed as #1610.

---

## 1. Problem Statement

### 1.1 Background

Issue #1600 was filed from the #1596 / #1601 Qwen2.5-VL investigation, which found the same construction in `src/vision/encoders/qwen2_5_vl.rs`. It was deliberately left out of that PR because Youtu-VL is a separate family with its own encoder, checkpoint and validation path, and because the misplaced-token counts that motivate the fix depend on the family's own window width and processor caps. The issue asked for those numbers to be measured rather than assumed.

### 1.2 The defect

```rust
let mut indexed: Vec<(i32, usize)> = window_index.iter().enumerate().map(|(i, &v)| (v, i)).collect();
indexed.sort_by_key(|&(v, _)| v);
let mut reverse = vec![0i32; window_index.len()];
for (rank, &(_, orig_idx)) in indexed.iter().enumerate() {
    reverse[orig_idx] = rank as i32;
}
```

`get_window_index` returns a permutation of `0..N`. After the sort, `indexed[rank]` is `(rank, i)` where `window_index[i] == rank`, so the write is `reverse[i] = window_index[i]`. The function is an expensive identity on its input.

What the encoder needs is the inverse. `forward_with_spatial` gathers hidden states with `take(h_grouped, window_index, 0)`, so the row at position `j` after the gather holds the merged token whose raster index is `window_index[j]`. Restoring raster order means `out[i] = merged[j]` for the `j` with `window_index[j] == i`, which is exactly `argsort`.

### 1.3 Why this family is exposed differently

A window is `window_size / (patch_size * spatial_merge_size)` merged tokens, which is `256 / (16 * 2)` = 8 for this checkpoint. `build_processor` passes `vision_config.num_patches` (4096) as the processor's patch cap, so `effective_max_pixels` is 1,048,576 and the patch grid tops out at 64x64, the merged grid at 32x32.

Enumerating every merged grid the loader can produce:

| image | resized | patch grid | merged grid | windows | merged tokens misplaced |
|---|---|---|---|---|---|
| 224x224 | 224x224 | 14x14 | 7x7 | 1x1 | 0 of 49 |
| 256x256 | 256x256 | 16x16 | 8x8 | 1x1 | 0 of 64 |
| 336x336 | 352x352 | 22x22 | 11x11 | 2x2 padded | 99 of 121 |
| 384x384 | 384x384 | 24x24 | 12x12 | 2x2 padded | 120 of 144 |
| 448x448 | 448x448 | 28x28 | 14x14 | 2x2 padded | 160 of 196 |
| 512x512 | 512x512 | 32x32 | 16x16 | 2x2 | 192 of 256 |
| 768x768 | 768x768 | 48x48 | 24x24 | 3x3 | 528 of 576 |
| 1024x1024 and larger | 1024x1024 | 64x64 | 32x32 | 4x4 | 960 of 1024 |

Two conclusions follow, and both matter more than the individual numbers.

First, the permutation is an involution only for a single window. A grid whose window count per edge equals the window width would also be one, but for an 8-wide window that is a 64x64 merged grid, which needs 16384 patches and is past the 4096 cap. It is unreachable, so **one window is the only safe case**. Contrast Qwen2.5-VL, whose 4-wide window makes the common 16x16 merged grid an involution, which is why #1596 was invisible at the default 448x448 fixture size.

Second, the repository's only general image fixture, `tests/fixtures/test_image.png`, is 224x224. That is a 7x7 merged grid, a single window, the identity. No test driving that fixture could ever have failed on this defect, whatever it asserted.

---

## 2. Change Summary

| File | Change |
|---|---|
| `src/vision/encoders/youtu_vl_window.rs` | `reverse_window_indices` builds `inverse[window_index[rank]] = rank` directly, bounds-checked. Rustdoc records why the old construction degenerated. Name and signature unchanged. |
| `src/vision/encoders/youtu_vl.rs` | Call-site comment no longer claims the old code was equivalent to `argsort`. |
| `src/vision/encoders/youtu_vl_tests.rs` | Four checkpoint-free tests, all failing on the pre-fix code. |
| `CHANGELOG.md` | Entry under Unreleased / Fixed, scoped to the token order at that stage and pointing at #1610. |

The new tests call `get_window_index` at the checkpoint's real parameters rather than the 4-wide synthetic config the existing tests use, so the pinned counts are the ones a real image hits: the minimal non-involution `[2, 0, 1]` and an involution beside it, the 32x32 patch grid with its 192-of-256 count pinned, the single-window identity case kept so the case that hid the defect stays documented, and a two-image batch whose second grid needs padding on one edge, which exercises the per-image offset `get_window_index` adds.

---

## 3. Technical Decisions

### 3.1 Copy #1601's construction rather than invent one

`invert_window_index` in `src/vision/encoders/qwen2_5_vl.rs` had already been reviewed and merged for the identical problem. Using the same loop, the same bounds guard and the same comment shape keeps the two families diffable, which is the whole reason the repository mirrors the upstream directory layout.

### 3.2 Bounds-check rather than index directly

`get_window_index` emits a permutation of `0..len`, so the guard is unreachable in practice. It is there because the alternative on a malformed grid is a panic inside a vision forward, and because the helper is `pub(super)` and could acquire a second caller. The cost is one `try_from` and one `get_mut` per token, on a path that already allocates the vector.

### 3.3 Keep the name and signature

The single call site is untouched, so the diff stays inside the helper and the change cannot alter anything else by accident.

### 3.4 Merge despite no observable content change

The end-to-end run shows no difference (section 4). The argument for merging anyway is that the defect is arithmetic, not statistical: the tests reproduce the old construction's exact misplaced counts and prove the new inverse round-trips to the identity on every grid tried. Holding a proven-correct fix behind an unrelated defect would leave a known-wrong permutation in the tree and make #1610's own validation harder, since its author would have to reason about two defects at once.

### 3.5 Do not fold #1610 into this PR

The issue's Scope section rules it out, and the repository's one-issue-one-PR convention does too. There is a substantive reason as well: #1610 changes the processor's output layout, which affects every Youtu-VL run at every size, while this change affects only the merged token order above one window. They deserve separate bisect points.

### 3.6 Hold back the content-asserting parity test

A `tests/youtu_vl_parity.rs` in the shape of `tests/qwen2_5_vl_parity.rs` was written during this work. Both shape fixtures are non-involution for this family, so it would guard this fix well. It is not in this PR because it cannot pass until #1610 lands, and committing a known-failing test is worse than committing none. It is described in #1610 as the gate to add once that issue is fixed.

---

## 4. Validation

### 4.1 Checkpoint-free

`cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings` clean. `cargo test --workspace --profile test-fast --features metal,accelerate` green: 116 suites, 10499 passed, 0 failed, 332 ignored. `cargo fmt --all -- --check` clean. The `--workspace` scope matters here and was confirmed rather than assumed: the `mlxcel-core` suite appears in the log, so the run is not the root-only `-p mlxcel` shape that #1007 fixed.

The grid table in section 1.3 was produced by an independent reimplementation of `smart_resize` and `get_window_index`, written from the source and run outside the crate, and it reproduces the counts the new unit tests pin. That is a check on the tests, not a restatement of them.

### 4.2 Real checkpoint, and why it shows nothing

`models/mlx/youtu-vl-4b-instruct` (`tencent/Youtu-VL-4B-Instruct`), M1 Ultra, release build with `metal,accelerate`, greedy (`-t 0 -n 48`).

| fixture | merged grid | before | after |
|---|---|---|---|
| `test_image_shapes.png` (448) | 14x14, 160 of 196 misplaced | "The image contains a single, solid black circle on a white background." | identical |
| `test_image_shapes_336.png` (352) | 11x11, 99 of 121 misplaced | "The image contains a single, solid black circle on a white background." | identical |
| `test_image.png` (224) | 7x7, single window, 0 misplaced | "The image is completely black and contains no visible content, objects, text, or details." | identical |

The third row is the diagnostic one. Its permutation is the identity, so this change provably cannot alter it, yet a solid **orange** square is already described as black. Something upstream of the window reordering was corrupting the vision path at every size, which is why the two non-involution rows cannot show a difference either.

### 4.3 Root-causing the dominating defect

Two facts located it without needing an external oracle. `YoutuVLVisionEncoder::rot_pos_emb` builds position ids with `reshape(&[h / merge, merge, w / merge, merge])` then `transpose_axes(&[0, 2, 1, 3])`, which is merge-block-major, and `forward_with_spatial` reshapes to `[n_groups, spatial_merge_unit, dim]`, which assumes each consecutive group of four rows is one 2x2 patch block. But `YoutuVLProcessor::try_preprocess_with_spatial` emits rows in plain raster order. The encoder's own two inputs disagree. Separately, the checkpoint's `convert_image_to_patches` permutes `(1, 4, 2, 5, 3, 6, 0)`, giving merge-block-major rows and `(dy, dx, c)` inner features, while the processor emits `(c, dy, dx)` and the loader only renames `patch_embedding.weight` without permuting it.

Patching only that emission loop to the upstream ordering and rebuilding turned the 224x224 fixture from "completely black" into "a solid, uniform block of bright orange color", which confirms the mechanism. The three-shape fixtures changed but stayed wrong, so **the ordering fix is necessary but not sufficient** and at least one further defect remains in this family. That patch was reverted; it is not part of this PR. Filed as #1610, with the processor's patch-count cap divergence filed separately as #1611.

---

## 5. Learning Points

**A permutation bug can be invisible at exactly the size you test at.** Both this defect and #1596 survived because the fixture size happened to be an involution for that family. The involution set depends on the window width and the processor's caps, so it differs per family even when the buggy code is identical. Enumerate the reachable grids before deciding a fixture is representative.

**"No behavior change" is a finding, not a failure.** The byte-identical before and after here is what surfaced #1610. A run that shows nothing is worth doing precisely because the absence of a difference has to be explained, and the explanation was a larger defect nobody had filed.

**Prefer a control that cannot move.** The 224x224 single-window case is the reason this report can claim a second defect exists rather than guess. It isolates the change under test to zero effect by construction, so any wrongness it shows must come from somewhere else.

**Two components can be mutually inconsistent without either matching upstream.** The processor and `rot_pos_emb` disagreed with each other, which made the defect provable from inside the repository, before any reference implementation was consulted.

---

## 6. Follow-ups and What Remains Unverified

- **#1610** (priority:high): the processor's patch emission order. Root-caused and reproduced here, and it masks any user-visible benefit of this PR.
- **#1611**: `build_processor` uses `vision_config.num_patches` (4096) as the processor cap instead of `preprocessor_config.json`'s `max_num_patches` (256). Worth knowing that honoring the upstream cap would hold the merged grid at 8x8 or below, which would mask this PR's defect rather than fix it, so the two must not be treated as alternatives.
- **At least one further defect remains** in the Youtu-VL vision path beyond #1610, since the three-shape fixtures were still described wrong under the corrected ordering. Not yet localized.
- **No reference-implementation diff was run** for this family. The stage-by-stage transformers comparison that #1601 used is blocked until #1611 is resolved, because mlxcel and the HuggingFace processor currently produce different patch grids for the same image.
- **`docs/supported-models.md` lists Youtu-VL as supported.** That claim is optimistic while #1610 stands, but it was left alone rather than churned in an unrelated PR.
