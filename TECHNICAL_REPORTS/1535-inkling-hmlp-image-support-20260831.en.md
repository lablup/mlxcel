# Technical Report: PR #1535 - Inkling HMLP image support

**Date**: 2026-08-31
**Author**: mlxcel maintainers
**Status**: Completed with deterministic reference validation; real-checkpoint validation remains deferred
**Languages**: Rust, Markdown
**Risk Level**: High

---

## Executive Summary

PR #1535 adds native Inkling image ingestion on top of the text backbone delivered by issue #1318. It implements the reference hierarchical MLP vision tower, exact 40x40 tiling and normalization, dynamic placeholder expansion, ordered feature scatter, VLM detection and loading, and shared CLI/server runtime integration. Checkpoints without both a `vision_config` and `model.visual.*` tensors remain available through the text-only Inkling path.

Inkling's image stack differs from transformer-based vision towers. Every tile is duplicated into two temporal slots, progressively folded across time and space, and projected directly into one text-width soft token. Several errors can preserve plausible shapes while changing values or media order, so the implementation is anchored by deterministic tests for the selected scale rows, fold permutation, intentional trailing tile column, per-image placeholder cardinality, and raw checkpoint key mapping.

## 1. Problem Statement

mlxcel had the Inkling decoder but no HMLP encoder or VLM runtime. Existing image processors did not reproduce Inkling's unusual `columns = width / 40 + 1` rule, which creates a fully padded trailing column for exact-width images. Generic fixed-token prompt expansion also could not represent different tile counts for multiple images in one request.

The loader needed to preserve and rename `model.visual.*` tensors while reusing the existing Inkling text sanitizer, support dense and affine-quantized visual projections, and avoid classifying text-only exports as VLMs merely because they retain a vision config. The generation boundary also had to merge already normalized text embeddings with final-normalized visual features without applying the text embedding norm twice.

## 2. Change Summary

| Area | Result |
| --- | --- |
| HMLP graph | Added prime-factor scale planning, minimum-cost injective row selection, time-space-to-depth folding, bias-free projection, intermediate RMSNorm plus exact GELU, and final RMSNorm |
| Preprocessing | Added RGB conversion, optional upstream-compatible Lanczos resize, row-major 40x40 tiling, `-1` padding before rescale, CLIP normalization, temporal duplication, and per-image tile counts |
| Prompting | Added fail-closed expansion from one image marker per image to the exact number of tile tokens and a plain-prompt insertion fallback |
| Merge | Added ordered `merge_llava` scatter into normalized Inkling text embeddings and prepared-embedding decoder entry points that skip a second input norm |
| Loading | Added visual-key normalization, dense and affine projection validation, processor config parsing, and text-width compatibility checks |
| Detection | Added indexed and unindexed safetensors visual-weight detection while preserving the text-only model route |
| Runtime | Registered `InklingVLM` in loaded-model dispatch and the shared image path used by CLI `--image` and OpenAI-compatible `image_url` requests |
| Documentation | Updated the supported-model boundary and validation limits |

## 3. Technical Decisions

### 3.1 Preserve the reference graph exactly

The port follows the public mlx-vlm Inkling `vision.py` graph. For the four-layer checkpoint, the selected grids are rows `[0, 1, 2, 4, 5]`, yielding projection inputs 75, 512, 5120, and 9600. Folding reshapes and transposes in `t, row, column, channel` order before flattening. Intermediate layers use RMSNorm followed by exact erf GELU, while the final projection has no activation and is followed by the tower's final RMSNorm.

The one-layer 0.6B configuration selects the first and last grids directly and folds the complete `[2, 40, 40, 3]` tile into 9600 channels. Both plans are computed from config rather than hard-coded, but the loader rejects geometry outside the published `T=2, H=W=40, C=3` contract.

### 3.2 Reproduce padding and resize semantics before token counting

The processor initializes each channel-first tile to `-1`, copies available uint8 pixels, rescales by `1/255`, normalizes per channel, and then duplicates the channel-last result across time. Width uses floor division plus one rather than ceiling division. This distinction is observable for every exact multiple of 40 and directly determines prompt length.

Optional resizing matches the upstream long-edge fraction, its non-downscaling cap expression, half-up dimension rounding, and Lanczos filter. Tile counts are returned per image so placeholder expansion cannot accidentally apply the first image's count to every media block.

### 3.3 Separate normalized and raw embedding entry points

Inkling applies `embed_norm` before visual scatter. The VLM wrapper therefore requests normalized text embeddings, replaces image-marker rows with final-normalized HMLP features, and invokes prepared-embedding decoder entry points. Standard text calls continue through the existing raw-embedding path. This interface is also the extension point for the dependent audio implementation and prevents a visually silent double-normalization defect.

### 3.4 Treat checkpoint structure and media cardinality as untrusted input

The loader validates projection rows and packed widths, scale and bias shapes, normalization widths, the text hidden-size match, exact visual geometry, positive finite epsilon, and the MLX `i32` shape bound before inference. Detection reads only safetensors headers, caps unindexed headers at 128 MiB, and requires both config and visual weights.

The runtime counts actual image markers after expansion and requires an exact match with emitted HMLP features before scatter. Empty prompts, zero-size media blocks, arithmetic overflow, partially expanded marker layouts, and incompatible embedding widths return errors instead of silently shifting text and image positions.

## 4. Review and Hardening

Correctness, security, performance, and finalizer review produced the following hardening before merge:

- Added dense and quantized projection-shape checks before constructing infallible MLX layers.
- Rejected dense projection bias and malformed RMSNorm tensors.
- Added bounded, header-only fallback detection for unindexed safetensors files.
- Made tile allocation and count arithmetic checked and reported allocation failure.
- Rejected invalid resize, normalization, and text-width configuration values.
- Preserved prepared-embedding sequence-ID and last-logits paths for model-owned Inkling caches.
- Disabled chunked prefill and left image-feature caching out of scope rather than reusing one-shot visual embeddings under an unsafe cache contract.

No unresolved CRITICAL or HIGH correctness, security, or performance findings remained in the reviewed change.

## 5. Validation

| Gate | Result |
| --- | --- |
| `cargo test --lib inkling --profile test-fast --features metal,accelerate -- --test-threads=1` | Pass, 40/40 |
| `cargo check --lib --features metal,accelerate` | Pass |
| `cargo clippy --lib --features metal,accelerate -- -D warnings` | Pass |
| `cargo fmt --all -- --check` | Pass |
| `git diff --check` | Pass |

The focused suite covers both reference scale plans, channel fold order, tower output shape, malformed geometry and weights, exact-width trailing padding, per-image tile counts, one-marker-per-image expansion, plain prompt insertion, pre-expanded and ambiguous layouts, indexed and unindexed detection, and raw visual-key normalization. Existing Inkling text, cache, sanitizer, reasoning-marker, and chat-template tests also remain green in the same 40-test selection.

## 6. Validation Limits and Follow-up

The public Inkling-Small affine MLX checkpoint is approximately 153.5 GB and the native NVFP4 checkpoint is approximately 170.7 GB. Neither was available on the validation host. Real image-answer quality through CLI and server, feature parity against the one-layer 0.6B checkpoint, peak memory, and Apple GPU throughput therefore remain unverified and are not claimed by this report.

Issue #1323 adds native adjacent-frame video pairs on this shared visual shell. Audio, image-feature caching, tensor-parallel vision execution, and fused kernels remain separate work. The broad workspace test and all-target clippy gates are left to the epic-level final verification rather than duplicated in this focused issue worktree.

## References

- Epic #1313, issue #1327, and prerequisite issue #1318
- PR #1535
- Public mlx-vlm Inkling `vision.py`, `inkling.py`, and processing implementation
- `docs/supported-models.md`
