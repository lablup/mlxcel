# Technical Report: PR #1405 - LFM2-VL Image Splitting

**Date**: 2026-08-25
**Author**: mlxcel contributors
**Status**: Completed with documented parity follow-up
**Languages**: Rust, Markdown
**Risk Level**: Medium

---

## Executive Summary

PR #1405 implements the LFM2-VL high-resolution `do_image_splitting` path requested by issue #1352. Large images now use the checkpoint's aspect-ratio tile policy, row-major views, and `<|img_row_r_col_c|>` or optional `<|img_thumbnail|>` prompt framing, while small images retain the previous single-view byte path.

Review and security hardening corrected the source of the per-view patch budget, rejected malformed or unbounded checkpoint metadata, and enforced layout and placeholder cardinality before image embeddings reach masked scatter. Focused tests, workspace clippy, formatting, release build, GitHub CI, and real-checkpoint inference passed; the full local workspace test gate remains blocked before execution by an unrelated ThinLTO linker defect in `qwen3_omni_moe_parity`.

---

## 1. Problem Statement

LFM2-VL checkpoints publish a split-image policy in `processor_config.json`, but mlxcel previously ignored that policy and smart-resized every image into one view. A 1920x1080 screenshot was therefore compressed into the single-view token budget instead of the multiple 512-pixel views used during reference preprocessing.

The missing feature also affected prompt construction and embedding projection: the runtime had no tile markers, no thumbnail marker, no per-image view layout, and an implicit one-view-per-image assumption. Implementing only the crop loop would have produced a prompt-to-feature cardinality mismatch.

| Risk | Impact | Likelihood before fix |
|------|--------|-----------------------|
| Large screenshots lose high-resolution detail | High | High |
| Tile prompt markers do not align with projected views | High | High without an end-to-end layout contract |
| Malformed checkpoint metadata drives oversized allocations | High | Medium |

---

## 2. Technical Decisions

### 2.1 Carry a per-image layout through the full VLM path

`Lfm2VlImageLayout` records each view's patch grid plus the logical tile rows and columns. The processor returns layouts in source-image order, the vision tower projects every flattened view in that order, and prompt expansion uses the same layouts to emit markers and the exact number of `<image>` placeholders.

This avoids reconstructing geometry from tensor shapes after preprocessing and keeps multi-image prompts unambiguous. It also gives the runtime a single boundary where malformed tile/view cardinality can be rejected before masked scatter.

### 2.2 Treat processor metadata as untrusted input

The loader reads and unwraps `processor_config.json`, with `config.json` fallback for compatible legacy fields, but it does not accept invalid values silently. Tolerance and downsample factors must be finite and positive, token ids must fit the runtime range, per-view patch counts are bounded, tile canvases use checked arithmetic, and malformed JSON is reported instead of replaced by defaults.

### 2.3 Use the processor patch budget, not the vision table length

The published checkpoint exposes `vision_config.num_patches=256` for the learned single-view position table and `max_num_patches=1024` for the padded processor row. A 512-pixel tile produces a 32x32 patch grid, so validating it against 256 incorrectly rejects the supported checkpoint. The final loader derives this limit from processor metadata and validates the 1024-patch view against that budget.

### 2.4 Preserve checkpoint policy instead of forcing a thumbnail

The local checkpoint declares `use_thumbnail=false`; mlxcel honors that value. The implementation still supports a thumbnail-last layout when a checkpoint enables it, but it does not override published metadata with a different reference-library default.

---

## 3. Implementation Details

| Area | Change |
|------|--------|
| `src/loading/vlm_lfm2_vl.rs` | Connects parsed tiling metadata and marker ids to processor and runtime construction. |
| `src/loading/vlm_lfm2_vl_metadata.rs` | Parses, validates, and bounds processor policy and tokenizer marker metadata. |
| `src/loading/vlm_lfm2_vl_tests.rs` | Tests defaults, nested configuration, invalid metadata, marker resolution, and patch-budget behavior. |
| `src/vision/processors/lfm2_vl.rs` | Selects the aspect-ratio grid, resizes once, crops row-major tiles, appends an optional thumbnail, packs every view, and returns layouts. |
| `src/multimodal/lfm2_vl_prompt.rs` | Expands logical image placeholders into framed per-view token runs and validates image/layout cardinality. |
| `src/vision/lfm2_vl.rs` | Projects all views and concatenates feature rows in prompt order. |
| `src/multimodal/vlm_runtime.rs` | Carries the layouts and resolved marker table into prompt expansion. |
| `docs/supported-models.md` | Records the implemented split-image capability. |

The expanded loader was refactored into metadata and test siblings so the implementation remains reviewable and every newly expanded source file stays below the repository's 500-line limit.

---

## 4. Review, Security, and Quality

### 4.1 Review correction

Review found that the first implementation validated a 512-pixel tile against `vision_config.num_patches`, which made the published checkpoint unloadable. The correction uses `max_num_patches` from processor metadata and added a regression test that distinguishes the two meanings.

### 4.2 Security hardening

| Finding | Severity | Resolution |
|---------|----------|------------|
| Malformed processor or tokenizer sidecars silently fell back | Medium | JSON parse errors now fail loading. |
| Non-finite tolerance or non-positive downsample values could corrupt bounds | Medium | Values are explicitly validated before arithmetic. |
| Tile dimensions and patch budgets could overflow or allocate excessive canvases | Medium | Checked arithmetic and explicit upper bounds reject unsafe metadata. |
| Prompt placeholders, layouts, and projected rows could diverge | Medium | Layout and logical-image cardinality are validated before merge. |

No critical or high-severity security or performance findings remained after hardening.

### 4.3 Test coverage

Nineteen focused tests were added across loader metadata, image processing, and prompt expansion. They cover reference defaults, nested metadata, invalid token and numeric fields, tile-ratio enumeration, area tie-breaking, row-major color order, optional thumbnail placement, multi-image expansion, malformed layouts, and byte-identical small-image preprocessing.

---

## 5. Real-Checkpoint Results

The local `models/lfm2-vl-450m-4bit` checkpoint loaded `tile_size=512`, `min_tiles=2`, `max_tiles=10`, `max_pixels_tolerance=2.0`, `use_thumbnail=false`, and `max_num_patches=1024`.

| Input | Rust layout | Reference layout | Prompt result |
|-------|-------------|------------------|---------------|
| 1920x1080 screenshot | 4x2 tiles, eight 32x32 patch views | 4x2 tiles, eight 32x32 patch views | 256 image tokens per tile, 2048 total, markers 397-400 and 407-410, no thumbnail |
| 640x480 image | One 26x36 patch view | One 26x36 patch view | 234 image tokens, unchanged single-view path |

The release binary produced finite 64-token Metal output for both inputs. Generated text was not token-exact against the Python reference for the large image; the unchanged small path also differed by one late token, which shows that a pre-existing runtime or preprocessing numerical difference remains outside the now-exact tile geometry and prompt framing. This report deliberately distinguishes generated-token parity from the structural and prompt-token parity established by this PR.

---

## 6. Validation Summary

| Validation | Result |
|------------|--------|
| `cargo test --profile test-fast lfm2_vl --lib` | Passed, 28 tests |
| `cargo clippy --workspace --all-targets -- -D warnings` | Passed |
| `cargo fmt --all -- --check` | Passed |
| `git diff --check` | Passed |
| `cargo build --release --features metal,accelerate --bin mlxcel` | Passed |
| GitHub CI | Passed all required checks |
| Full workspace tests | Blocked before execution by unrelated ThinLTO missing-symbol failure while linking `qwen3_omni_moe_parity`; reproduced with `CARGO_BUILD_JOBS=1` |

---

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed before this report | 8 |
| Lines added | 1,307 |
| Lines deleted | 145 |
| Focused tests added | 19 |

| Commit | Purpose |
|--------|---------|
| `c79d86255` | Add LFM2-VL image splitting. |
| `c1db2e3e4` | Validate split tolerance. |
| `11b40c73a` | Load the correct processor patch budget. |
| `426bd1b22` | Harden metadata and layout validation. |
| `5f3d2f1de` | Expand tiling test coverage. |
| `35e250fae` | Split loader metadata helpers and tests. |

---

## 8. Follow-up Actions

- Investigate the existing LFM2-VL numerical parity gap with controlled pixel-array and embedding comparisons; do not treat the exact geometry and prompt token sequence as proof of generated-token equality.
- Re-run the full workspace test gate after the macOS ThinLTO integration-test linker defect is corrected.

## References

- Issue #1352: LFM2-VL image splitting and marker framing.
- PR #1405: LFM2-VL image splitting implementation.
- `docs/supported-models.md`: Supported-model behavior.
