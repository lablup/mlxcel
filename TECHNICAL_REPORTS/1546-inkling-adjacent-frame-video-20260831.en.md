# Technical Report: PR #1546 - Inkling adjacent-frame video support

**Date**: 2026-08-31
**Author**: mlxcel maintainers
**Status**: In review; all reported HIGH findings addressed with deterministic validation
**Languages**: Rust, Markdown
**Risk Level**: High

---

## Executive Summary

PR #1546 adds native Inkling video ingestion to the HMLP image shell delivered by PR #1535. CLI `--video` and OpenAI-compatible server `video_url` requests now share one bounded host-preparation path that selects adjacent frame pairs per clip, preserves independent clip chronologies, decodes only selected source frames, and uses the HMLP temporal axis instead of representing both frames as separate image entities.

The final design addresses three independent-review HIGH findings. Structured timestamp and image parts are inserted at a validated boundary inside the current user turn rather than after the global BOS token. Multiple videos retain clip boundaries and share one 16-pair request budget without forming cross-clip pairs. Probe-derived frame and pixel budgets are checked before the first decode, and preprocessing borrows compact selected frames without deep-cloning the full sampled sequence. Timestamps use actual selected source indices and source FPS, eliminating drift after the generic 768-frame sampling ceiling.

The branch also normally merges the native Inkling MTP implementation from main commit `092d3dd0`. The combined wrapper preserves HMLP prepared-embedding prefill for image and video requests while exposing the same public text backbone and state lifecycle used by MTP.

## 1. Problem Statement

Inkling visual inputs have shape `[N, 2, 40, 40, 3]`. Still images duplicate the same pixels into both temporal slots. Video must instead place the first selected frame in slot 0 and the immediately adjacent frame in slot 1 while retaining one prompt image entity and one HMLP feature row per tile.

A correct host path must also preserve conversation structure and bound untrusted media work. Flattening clips before pairing can join the final frame of one video to the first frame of another and gives later clips the wrong timestamp origin. Decoding every sampled frame before applying a 16-pair cap leaves memory proportional to the input rather than the admitted work. Moving rendered image placeholders after BOS can place media outside the current user turn when system messages or history are present.

These failures can preserve valid tensor shapes while changing chronology, modifying the wrong temporal rows, scattering features into unrelated prompt positions, or exhausting memory before the nominal pair limit takes effect.

## 2. Change Summary

| Area | Result |
| --- | --- |
| Clip-local planning | Computes adjacent-pair indices independently per video and never pairs across media boundaries |
| Request-wide allocation | Gives every admitted clip the reference minimum of two pairs, then distributes remaining capacity round-robin within a 16-pair total; at most eight clips are admitted |
| Timestamp accuracy | Maps sampled anchors back to actual source-frame indices and divides by probed source FPS, including after the 768-frame sampling ceiling |
| Resource admission | Probes all clips, then checks at most 32 unique selected frames and 512 MiB of worst-case RGBA storage before the first ffmpeg decode |
| Selective decode | Decodes only unique selected source indices through the existing fd-safe single-pass extractor and stores compact per-clip pair indices |
| Prompt safety | Inserts complete Inkling text/image content parts immediately before the final user text part; system messages, history, companion still parts, and generation tails remain in place |
| Plain CLI mode | Preserves the explicit `--no-chat-template` behavior by prepending media after BOS without pretending that a structured turn exists |
| Temporal splice | Replaces slot 1 only in the suffix belonging to video tiles; companion still rows retain duplicated temporal planes |
| Borrowed preprocessing | Processes references to compact decoded frames, avoiding deep image-buffer clones for first- and second-frame lists |
| MTP integration | Preserves the public `InklingVlModel.text` target and prepared-embedding sequence/last-logit entry points merged from main |
| Capability and docs | Keeps Inkling server video admission and documents clip, pair, frame, byte, prompt, and checkpoint limits |

## 3. Architecture and Data Flow

1. CLI paths become `VideoSource` values and server parts retain their already-admitted fd-backed `ResolvedVideo.source` handles.
2. Every clip is probed under the existing `VideoLimits`; the requested FPS is validated and the generic sampled-frame count is computed.
3. A request-wide allocator reserves two pairs per clip and distributes the remaining 16-pair budget without crossing clip boundaries.
4. Per-clip sampled anchors are mapped through uniform source indices. The compact set of unique source frames, actual anchor timestamps, decoded-frame total, and worst-case decoded bytes are known before decoding starts.
5. Only those unique indices are decoded. Pair indices refer to the compact per-clip vectors, so repeated short-clip anchors do not duplicate image buffers.
6. Structured prompts receive complete Inkling content parts before the final current-user text part. Each clip gets its own chronological introduction and timestamps beginning at that clip's source time.
7. Companion stills and pair-first frames are tiled together. Pair-second frames are tiled through borrowed references, and per-pair tile counts must match.
8. Only slot 1 of the final video-tile rows is replaced. The shared HMLP tower produces visual features, placeholder runs expand to actual tile counts, and `merge_llava` scatters features into normalized text embeddings.
9. The combined `InklingVlModel` prepared-embedding entry points forward those embeddings directly into the public text backbone, preserving the MTP-compatible sequence-state lifecycle without normalizing twice.

## 4. Technical Decisions

### 4.1 Preserve the public adjacent-pair formula within each clip

For a clip with an odd sampled-frame count, the final frame is conceptually repeated. The pair count is `max(2, min(clip_budget, padded_len / 2))`. Anchor `i` is `min(round(i * (padded_len - 2) / max(pair_count - 1, 1)), padded_len - 2)`, and the second position is `anchor + 1`, clamped only for the conceptual repeated final frame. A one-frame clip therefore produces two identical `[0, 0]` pairs, matching the reference minimum rather than inventing a different short-clip graph.

The 16-pair limit is request-wide, but the formula is evaluated per clip. Each clip receives two pairs first; remaining pairs are assigned round-robin up to its capacity. This makes the limit deterministic, prevents starvation, and makes eight clips the maximum representable request.

### 4.2 Use source indices as the timestamp authority

The generic sampler may cap a long input at 768 uniformly spaced frames. Computing time as `sampled_anchor / requested_fps` after that cap can be hundreds of seconds early. The hardened path maps every selected sampled position back to its actual source index and computes `source_index / probed_source_fps`. Each video owns its own mapping and starts an independent chronological prompt sequence.

### 4.3 Validate bounded work before decoding

The loader probes every admitted source and plans all pairs before invoking the frame extractor. It rejects requests over eight clips, 16 pairs, 32 unique selected frames, or 512 MiB of `width * height * 4 * selected_frames` storage. Checked arithmetic converts overflow into an error. Existing duration, resolution, and source limits continue to apply during probing.

The extractor receives only sorted unique source indices. A per-clip map converts those source indices into compact vector offsets. Runtime preparation then borrows `&DynamicImage` values for first and second entities, so neither the full sampled sequence nor selected frame buffers are cloned to build pair lists.

### 4.4 Keep media inside the current user turn

The public Inkling template emits structural tokens for user messages, text content, image content, and end-of-message boundaries. The runtime resolves those exact tokenizer IDs and finds the final `[message_user, content_text]` boundary. It requires a following end marker, rejects any later user content, requires companion image placeholders to precede the question, and verifies exact placeholder cardinality before and after insertion.

For every clip, the runtime inserts complete text and image content parts immediately before that final question. System messages, previous user/model turns, existing companion still-image parts, and the model-generation suffix retain their positions. Explicit CLI `--no-chat-template` requests use a separate plain layout because no trustworthy conversation boundary exists.

### 4.5 Compose video prefill with the merged MTP wrapper

Main commit `092d3dd0` made `InklingVlModel.text` the speculative target and added prepared-embedding sequence-aware and last-logit forwarding. The video branch adds only borrowed image preprocessing and temporal slot replacement around the existing image embedding preparation. Merge commit `2cade8d1` preserves both contracts and their tests. Image- and video-bearing requests therefore remain on classic HMLP prepared prefill before continuing through the shared text target; the video path does not bypass or re-normalize the merged embeddings.

## 5. Review and Hardening

Independent review reported three HIGH findings, all addressed on the feature branch:

- Prompt insertion now targets a validated current-user boundary. A regression includes a system message, prior user/model history, a companion still-image part, two video clips, the current question, and a generation tail.
- Pair allocation now preserves clip boundaries and independent timestamp origins while enforcing one request-wide budget.
- Video admission now caps clips, pairs, selected frames, and decoded bytes before decode; the decoder and processor operate only on compact selected frames without deep clones.
- Actual selected source indices fix timestamp drift after uniform sampling is capped at 768 frames.
- Shape, tile-layout, placeholder, FPS, arithmetic, and decoded-frame cardinality mismatches continue to fail before HMLP execution.
- Server decoding retains the fd-backed source handle established by media admission, avoiding a path reopen race.
- The normal main merge retains native MTP wrapper dispatch, exact state lifecycle, and prepared-image prefill behavior alongside video slot-1 preparation.

Independent re-review remains required before merge. This report does not treat green focused tests as review clearance.

## 6. Validation

| Gate | Result |
| --- | --- |
| `cargo test -p mlxcel inkling --lib` | Pass, 57/57 |
| `cargo test -p mlxcel-core drafter::inkling_mtp --lib` | Pass, 7/7 |
| `cargo test -p mlxcel inkling_ --lib` before the main merge | Pass, 28/28 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | Pass after the main merge |
| `cargo check -p mlxcel --bin mlxcel --bin mlxcel-server` | Pass after the main merge |
| `cargo fmt --all -- --check` | Pass after the main merge |
| `git diff --check` and `git diff --cached --check` | Pass |

The combined 57-test filter covers the text graph, HMLP image path, clip-local video planning, current-user prompt insertion, temporal suffix replacement, Inkling VLM MTP adapter, prepared-embedding prefill preservation, target verification, and exact KV plus four-convolution state restore/replay. The seven core tests additionally cover MTP config, detection, shard filtering, sanitization, forward shape, and flat snapshot restoration.

PR CI remains responsible for the repository's OpenXLA feature compile and broader platform matrix. The host does not provide the real checkpoint artifacts or Apple GPU hardware needed for end-to-end throughput and answer-quality validation.

## 7. Validation Limits and Follow-up

The public Inkling-Small affine MLX checkpoint is approximately 153.5 GB, the native NVFP4 checkpoint is approximately 170.7 GB, and the native MTP shard is approximately 4.5 GB. These artifacts and the intended real bouncing-ball fixture were unavailable on the validation host. Real CLI/server video-answer quality, motion-direction accuracy, peak unified memory, Apple GPU throughput, MTP throughput, and MTP acceptance length are not claimed.

Deterministic tests establish host semantics, reference pair indices, temporal row replacement, prompt placement, bounded planning, timestamp mapping, wrapper composition, and state restoration. A future checkpoint-backed validation should compare the same short motion clips against public mlx-vlm at 2 fps, verify expanded visual-token cardinality, and separately measure classic multimodal prefill followed by MTP-capable text decode.

## References

- Epic #1313, issue #1323, and prerequisite issue #1327
- PR #1546, prerequisite PR #1535, and merged MTP PR #1540
- Main MTP commit `092d3dd0` and feature integration merge `2cade8d1`
- Public mlx-vlm Inkling `inkling.py`, `vision.py`, image processing, processor, and video helper implementations
- `docs/supported-models.md`
