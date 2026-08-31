# Technical Report: PR #1546 - Inkling adjacent-frame video support

**Date**: 2026-08-31
**Author**: mlxcel maintainers
**Status**: Completed with deterministic reference validation; real-checkpoint validation remains deferred
**Languages**: Rust, Markdown
**Risk Level**: High

---

## Executive Summary

PR #1546 adds native Inkling video ingestion to the HMLP image shell delivered by PR #1535. It decodes video frames at the existing 2 fps default, chooses at most 16 evenly spaced adjacent pairs across the request, represents each first frame as one normal image entity, and replaces only temporal slot 1 of the resulting video-tile suffix with the matching second frame. CLI `--video` and OpenAI-compatible server `video_url` requests share the same prompt, tiling, splice, and embedding path.

The implementation follows the public mlx-vlm Inkling graph rather than treating video frames as independent still images. Timestamped prompt parts preserve chronological grounding, companion still images remain before video entities, and fail-closed shape, cardinality, FPS, and placeholder checks prevent media rows from being scattered into the wrong text positions.

## 1. Problem Statement

Inkling image patches have shape `[N, 2, 40, 40, 3]`. PR #1535 duplicated a still image into both temporal slots, which is correct for images but does not encode motion. A generic video fallback that sends every sampled frame as an independent image doubles the visual token count for the same two-frame evidence and cannot use the HMLP temporal axis learned by the checkpoint.

Native video therefore needs two synchronized representations. The first frame of every selected pair determines the image entity, tile count, prompt placeholder run, and slot 0 pixels. The immediately following sampled frame must use the same tiler and replace slot 1 only for those video tiles. Any non-adjacent pairing, tile-order drift, or incorrect suffix boundary can keep tensor shapes valid while producing double exposures, modifying companion still images, or silently scattering features into the wrong prompt rows.

## 2. Change Summary

| Area | Result |
| --- | --- |
| Pair selection | Added odd-frame padding, a minimum of two pairs, a request-wide maximum of 16 pairs, evenly spaced rounded anchors, and strict `anchor + 1` companions |
| Prompting | Added the chronological introduction, one `frame at t=<seconds>s:` text part and image marker per pair, then the original user question |
| Image processing | Reused the exact Inkling 40x40 tiler independently for first and second frames and verified identical per-pair tile layouts |
| Temporal splice | Rebuilt only the last video-tile rows as `[first_slot0, second_slot0]` while preserving both temporal planes of preceding still-image rows |
| Shared runtime | Added one host preparation function used by both CLI and server before the existing HMLP tower and normalized embedding scatter |
| CLI | Routed Inkling `--video`, including mixed companion `--image` input, through the native pair path |
| Server | Enabled Inkling `video_url`, retained fd-backed ffmpeg input, decoded optional `image_url` companions with configured limits, and required consistent per-request FPS |
| Capability detection | Added `InklingVLM` to server video admission with a synthetic indexed-checkpoint regression test |
| Documentation | Documented video behavior, adjacent-pair semantics, the 16-pair cap, and real-checkpoint validation limits |

## 3. Technical Decisions

### 3.1 Preserve adjacency and the reference anchor formula

After duplicating an odd final frame, the pair count is `max(2, min(max_pairs, len / 2))`. Anchor `i` is `min(round(i * (len - 2) / max(n_pairs - 1, 1)), len - 2)`, and its second frame is always `anchor + 1`. This preserves local motion while distributing evidence across the full sampled clip. The minimum of two pairs intentionally permits repeated anchors for a one-frame input, matching the reference rather than inventing a separate short-clip rule.

The pair budget is request-wide. Multiple CLI paths or server video parts are flattened in declared order before selection, so no request can exceed 16 first-frame image entities. Server parts with different FPS values are rejected because one anchor index cannot be converted to an honest timestamp under multiple time bases.

### 3.2 Treat every pair as one prompt image entity

The prompt contains companion still-image markers first, then the exact introduction `Here is a video as a sequence of frames in chronological order.`, followed by timestamp text and one image marker per pair. The already-rendered user question remains after those additions. Existing template markers are accepted only when there are none or exactly one per companion still; other counts fail instead of consuming a reserved image token that did not come from admitted media.

After first-frame preprocessing, each marker expands to that entity's actual tile count. The expanded marker total must still equal the HMLP feature count before `merge_llava` replaces normalized text embeddings. This keeps prompt and feature cardinality coupled to preprocessing rather than to an assumed fixed token count.

### 3.3 Splice the temporal suffix before the tower

Companion stills and first frames are preprocessed together, in that order. Second frames are preprocessed separately and their slot-0 plane becomes `[M, 40, 40, 3]`. The model boundary validates the full `[N, 2, 40, 40, 3]` and replacement shapes, keeps `pixel_values[..N-M]` unchanged, takes slot 0 from the final `M` rows, concatenates the replacement as slot 1, and then runs the same HMLP tower as an image request.

This boundary mirrors the public mlx-vlm `get_input_embeddings` implementation and avoids a second video-specific vision graph. It also leaves PR #1535's prepared-embedding sequence-ID and last-logit lifecycle unchanged, so later audio and MTP work can compose with the public `InklingVlModel.text` wrapper without bypassing visual prefill.

### 3.4 Retain server media security properties

Server video decoding continues through `ResolvedVideo.source`. On Unix this is the fd-backed handle acquired after allowlist and canonical-path validation, so ffmpeg cannot reopen a swapped path between admission and decode. Companion image bytes use the existing limit-aware decoder. Empty frame sets, non-finite or non-positive FPS, inconsistent multi-video FPS, mismatched adjacent-frame tile layouts, invalid tensor ranks or dimensions, suffix sizes outside `1..=N`, arithmetic overflow, and unmatched image placeholders return errors before vision execution.

## 4. Review and Hardening

Correctness, security, performance, and finalizer review produced the following hardening before merge:

- Added deterministic tests for the exact 10-frame anchors `[0, 3, 5, 8]`, odd final-frame duplication, the two-pair minimum, and strict adjacency.
- Tested that three leading still tiles retain two zero-valued planes while only slot 1 of the final two video rows receives replacement values.
- Validated second-frame tile counts against the corresponding first-frame suffix before any MLX splice.
- Rejected untrusted or ambiguous image-marker cardinality before reconstructing the timestamped prompt.
- Kept the server's fd-backed decode path and configured image limits instead of introducing a path-based shortcut.
- Required a single FPS time base for multiple server video parts, while retaining the standard 2 fps default and CLI override.
- Capped first-frame visual entities at 16 request-wide; second frames reuse the same visual token rows rather than adding another marker run.
- Added an explicit startup capability test using the same indexed visual-weight evidence that distinguishes public Inkling VLM checkpoints from text-only exports.
- Kept audio, MTP, fused kernels, padded batching, image-feature caching, and broad epic-level verification outside this issue.

No unresolved CRITICAL or HIGH correctness, security, or performance findings remained in the reviewed change.

## 5. Validation

| Gate | Result |
| --- | --- |
| `cargo test -p mlxcel inkling --lib` | Pass, 45/45 |
| `cargo test -p mlxcel pair_adjacent_frames --lib` | Pass, 1/1 |
| `cargo test -p mlxcel timestamped_pair_messages --lib` | Pass, 1/1 |
| `cargo test -p mlxcel slot1_overwrite_touches_only_the_tail --lib` | Pass, 1/1 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | Pass |
| `cargo check -p mlxcel --lib --features metal,accelerate` | Pass |
| `cargo check -p mlxcel --bin mlxcel --bin mlxcel-server` | Pass |
| `cargo fmt --all -- --check` | Pass |
| `git diff --check` | Pass |

The focused tests cover pair spacing and adjacency, single/odd/short frame behavior, chronological prompt part layout, mixed still/video marker order, reserved-token rejection, suffix-only slot replacement, invalid replacement cardinality, Inkling VLM media admission, and the existing image, HMLP, text, cache, sanitizer, reasoning-marker, and chat-template regressions.

The OpenXLA feature gate could not be reproduced locally because this host does not provide `IREE_DIST`; PR CI owns that compile check. Broad workspace tests and all-target clippy remain the epic-level final verification responsibility.

## 6. Validation Limits and Follow-up

The public Inkling-Small affine MLX checkpoint is approximately 153.5 GB and the native NVFP4 checkpoint is approximately 170.7 GB. Neither checkpoint nor the intended real bouncing-ball fixture was available on the validation host. Real CLI and server video-answer quality, motion-direction accuracy, peak memory, and Apple GPU throughput therefore remain unverified and are not claimed by this report.

Audio and MTP are independent epic work. The video method is additive and delegates to the same normalized image prefill after the temporal splice; it does not change the public text-wrapper or prepared-embedding contracts those changes use. Future real-checkpoint validation should compare a synthetic adjacent-motion clip against public mlx-vlm at 2 fps and confirm both answer direction and expanded visual-token cardinality.

## References

- Epic #1313, issue #1323, and prerequisite issue #1327
- PR #1546 and prerequisite PR #1535
- Public mlx-vlm Inkling `inkling.py`, `vision.py`, image processing, and generic video helper implementation
- `docs/supported-models.md`
