# Technical Report: PR #1402 - Align SmolVLM Split-Image Framing

**Date**: 2026-08-25
**Author**: mlxcel contributors
**Status**: Completed
**Languages**: Rust, Markdown
**Risk Level**: High

---

## Executive Summary

PR #1402 aligns SmolVLM and Idefics3 image splitting with checkpoint configuration and upstream prompt framing. It reads the actual preprocessing policy, emits row-major local tiles followed by a global thumbnail, generates tokenizer-derived row/column marker sequences, and fails safely when prompt placeholders and vision features cannot form a one-to-one mapping.

---

## 1. Problem Statement

### 1.1 Background

SmolVLM-family checkpoints describe image splitting in `preprocessor_config.json`. The previous loader primarily consumed `processor_config.json`, while the processor used original image dimensions to choose crops and the prompt expander treated every processed image as a single global block.

### 1.2 Existing Issues

- **Configuration drift**: `do_image_splitting`, resize limits, and normalization statistics could be ignored even though they were checkpoint-defined.
- **Geometry mismatch**: Split crops did not follow the configured resize, clamp, tile-multiple rounding, row-major local tile, and global-thumbnail order expected by the reference processor.
- **Prompt mismatch**: Local tiles lacked `<row_r_col_c>` framing, so the text token stream did not describe the feature order produced by preprocessing.
- **Unsafe cardinality boundary**: Extra or missing `<image>` placeholders could reach the masked feature merge with a different token count from the vision feature rows.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood before fix |
|------|--------|-----------------------|
| Split-image prompts associate features with the wrong spatial markers | High | High |
| Checkpoint-specific normalization and size policy are silently ignored | High | Medium |
| Malformed prompt/image combinations trigger an invalid feature scatter | High | Medium |

---

## 2. Change Summary

### 2.1 Checkpoint-Driven Configuration

The loader reads the splitting flag, longest-edge limits, and image normalization statistics from `preprocessor_config.json`, with `processor_config.json` retained only as a compatibility fallback. The vision configuration remains the source for the number of image tokens emitted per tile.

### 2.2 Reference-Compatible Tiling

The processor now applies the configured resize and clamp, rounds the resized dimensions to tile multiples, emits local tiles in row-major order, and appends the globally resized image last. Each image returns an explicit `TileLayout { rows, cols }`, allowing prompt construction to describe the exact local-tile arrangement rather than reconstructing it from pixel counts.

### 2.3 Tokenizer-Derived Framing

The SmolVLM prompt expander tokenizes the checkpoint's exact fake/image, row/column, and global marker strings with special-token insertion disabled. A split image receives one framed block per row plus a final global block, while a single image retains global-only framing. Idefics2 stays isolated on its fake-only single-tile contract instead of inheriting SmolVLM split markers.

### 2.4 Fail-Closed Cardinality Validation

Prompt expansion requires the placeholder count to equal the number of decoded images/layouts. Image-token block sizes use checked arithmetic, the runtime verifies that expanded image-token positions equal the encoded feature-row count, and zero-dimension images preserve the one-layout/one-tile invariant through a blank fallback tile. These checks prevent user-controlled prompt structure from reaching a masked scatter with incompatible shapes.

---

## 3. Technical Decisions

### 3.1 Carry Tile Layout as Structured Metadata

**Decision:** Return row and column counts next to processed pixel tensors.

**Rationale:** Prompt framing is a semantic description of preprocessing order. Explicit metadata keeps both paths synchronized and avoids fragile geometry inference after resize and rounding.

**Trade-off:** Callers and integration fixtures must adopt the richer processor result.

### 3.2 Tokenize Marker Strings Instead of Hard-Coding IDs

**Decision:** Cache marker token sequences produced by each model tokenizer.

**Rationale:** Token IDs are checkpoint vocabulary data, while the upstream contract is expressed as marker strings. Tokenizing with `add_special_tokens = false` preserves compatibility across related checkpoints without assuming one vocabulary layout.

**Trade-off:** Model initialization performs a small amount of one-time marker tokenization and retains a per-model cache.

### 3.3 Validate at Expansion and Merge Boundaries

**Decision:** Reject placeholder mismatches early and re-check token-to-feature cardinality immediately before the multimodal merge.

**Rationale:** The early check produces a clear input error, while the runtime check protects against future processor or prompt regressions. Defense at both boundaries is preferable to relying on masked-scatter behavior for malformed dimensions.

---

## 4. Review and Quality Findings

### 4.1 Implementation Review

The implementation review found no unresolved correctness issues after the integration fixture was updated for checkpoint normalization, `TileLayout`, and tokenizer-driven framing.

### 4.2 Security and Performance Review

The security review identified one HIGH cardinality mismatch and two MEDIUM robustness issues involving unchecked image-token arithmetic and zero-dimension inputs. Commit `ec5736feb` added exact placeholder validation, checked arithmetic, runtime feature-count verification, and the blank-tile invariant. No unresolved CRITICAL or HIGH security or performance findings remain.

### 4.3 Compatibility

- **Breaking changes**: None to CLI or HTTP interfaces.
- **New dependencies**: None.
- **Behavior change**: SmolVLM/Idefics3 split images now use checkpoint-defined geometry and marker framing; malformed prompt/image cardinality is rejected instead of reaching feature scattering.

---

## 5. Validation

- `cargo test --workspace --profile test-fast --features metal,accelerate` passed after the final hardening changes.
- `cargo clippy --workspace --all-targets -- -D warnings` passed.
- `cargo fmt --all -- --check` and `git diff --check` passed.
- The focused `smolvlm_parity` integration suite passed all six tests, including finite-logit, processor, normalization, detection, and prompt-token checks.
- A real local Idefics3 8B 4-bit checkpoint on Metal produced 2,197 image tokens for a 3x4 local split plus global thumbnail (`13 × 169`) and 169 image tokens for a single 256x256 image. The single-tile run matched the previous release binary's first greedy token.
- A local tokenizer reference check independently counted 2,197 `<image>` token IDs for the 3x4 split prompt and 169 for the single-image prompt.
- Full mlx-vlm token-exact generation could not start because its `AutoProcessor` rejected the local checkpoint's otherwise declared `Idefics3ImageProcessor`; this external reference-loader limitation did not prevent real-checkpoint mlxcel execution.

---

## 6. Change Statistics

| Item | Value |
|------|-------|
| Files changed | 7 |
| Lines added | 876 |
| Lines deleted | 235 |
| Implementation commits | 3 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `b64e0ce2a` | fix | Align SmolVLM split image framing |
| `18ebdb85e` | test | Update SmolVLM parity coverage for split framing |
| `ec5736feb` | fix | Guard SmolVLM image-token cardinality |

---

## 7. Follow-up Considerations

- Re-run token-exact reference generation when the local mlx-vlm/Transformers processor stack recognizes the checkpoint metadata.
- Preserve explicit tile-layout propagation when extending splitting to additional VLM families rather than inferring prompt geometry from tensor counts.
- Keep Idefics2 and SmolVLM framing paths separate unless checkpoint evidence establishes a shared marker contract.

---

## References

- Issue #1364: checkpoint splitting policy and row/column framing requirements
- PR #1402: implementation, parity updates, and cardinality hardening
- `docs/supported-models.md`: SmolVLM/Idefics3 split-image behavior
