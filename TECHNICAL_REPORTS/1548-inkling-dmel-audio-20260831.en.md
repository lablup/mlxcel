# Technical Report: PR #1548 - Inkling dMel audio input

**Date**: 2026-08-31
**Author**: mlxcel maintainers
**Status**: In review; deterministic frontend and integration validation complete
**Languages**: Rust, Markdown
**Risk Level**: High

---

## Executive Summary

PR #1548 adds the Inkling dMel audio frontend and connects it to the HMLP multimodal shell delivered by PR #1535. CLI `--audio` and OpenAI-compatible server `input_audio` requests now share the bounded WAV decode, mono downmix, 16 kHz resampling, dMel feature extraction, strict 16-bin quantization, compact valid-row preparation, summed-channel embedding tower, prompt expansion, and prepared-embedding merge. Audio may be used alone or with still images.

The embedding order is explicit: token embeddings receive Inkling's optional input RMS normalization once, HMLP image features are scattered first, and audio features are scattered second. The resulting prepared tensor enters the decoder through the wrapper's prepared-embedding methods, so it is neither replaced by fresh token embeddings nor normalized twice. Audio-bearing requests remain on the classic multimodal path; the text-only native MTP path delivered by PR #1540 declines both raw audio and already-prepared audio tensors.

The implementation is validated with deterministic f64/CPU reference calculations and synthetic MLX graphs. The approximately 153.5 GB public affine checkpoint and roughly 170.7 GB native NVFP4 checkpoint were unavailable on the validation host, so this report does not claim a real-checkpoint transcription, quality, memory, or throughput result.

## 1. Problem Statement

Inkling does not use the Whisper-style frontend already present in mlxcel. It consumes 20 frames per second, with each frame represented by 80 log-mel channels quantized into 16 discrete bins. Each channel selects a row from its own 16-row segment of one 1,280-row embedding table; the 80 selected vectors are summed and RMS-normalized into one text-width feature row.

Several details are numerically load-bearing. The STFT uses an uncentered 1,600-sample periodic Hann window every 800 samples. The Slaney filterbank is derived in f64 and cast to f32, while the mel accumulation uses f32 row-wise dot products. Quantizer boundaries are f64 midpoints rounded downward onto the f32 lattice, and values use strict greater-than comparisons so a boundary tie stays in the lower bin. Small changes can move values across neighboring bins while preserving valid tensor shapes.

The control plane also needs exact cardinality and ordering. One prompt placeholder must expand to one token per valid audio frame. Right-padding rows must never reach the tower. In mixed image/audio requests, image and audio rows must scatter in the same order used by the public model. Server prompt synthesis must stay inside the current user turn rather than selecting an end marker from system messages or history. Prepared multimodal inputs must not enter the text-only speculative path.

## 2. Change Summary

| Area | Result |
| --- | --- |
| Host preprocessing | Reuses the bounded WAV boundary for decode, stereo averaging, linear 16 kHz resampling, cancellation checks, 16-clip admission, and five aggregate minutes |
| dMel extraction | Implements periodic Hann, uncentered RFFT, librosa-compatible Slaney filters, f32 mel accumulation, `log10`, batching, masks, and exact frame counts |
| Quantization | Builds downward-rounded f32 boundaries from f64 centers and applies strict lower-bin tie behavior |
| Compact bridge | Removes every padded row and concatenates only valid frames in request order before MLX allocation |
| Audio tower | Offsets each channel into its 16-row segment, gathers dense or affine-quantized embeddings, sums 80 channels, RMS-normalizes, and chunks at no more than 256 frames |
| Weight loading | Renames `model.audio.encoder.*` and `model.audio.final_norm.weight` into the reusable audio tower and validates dense and quantized layouts |
| VLM integration | Stores processor metadata in `InklingVlModel` and implements normalized text -> image scatter -> audio scatter |
| Prompt integration | Expands template placeholders in place or synthesizes complete wrappers at a validated current-user end boundary; plain CLI mode appends to the explicit raw prompt |
| CLI and server | Adds Inkling dispatch for CLI `--audio` and server `input_audio`, with optional still images and shared preparation statistics |
| Capability and detection | Exposes loaded-runtime audio capability while preserving the visual-shell detection rule: InklingVLM still requires both `vision_config` and `model.visual.*` weights |
| Speculative safety | Declines MTP burst for raw audio and for prepared embeddings after the raw payload has been consumed |
| Documentation | Records frontend formulas, limits, supported surfaces, scatter order, and real-model validation limits |

## 3. Architecture and Data Flow

1. The CLI provides one WAV path; the server supplies one or more already-decoded `input_audio` byte payloads. Both routes enter the shared `AudioFamilyPolicy::inkling()` boundary.
2. The boundary validates request limits, decodes WAV, averages channels, linearly resamples to 16 kHz mono, and reports source and normalized work metrics.
3. The Inkling feature extractor pads the batch to its longest clip, adds the left STFT context, computes 80 log-mel values per 50 ms frame, masks padded rows, and reports each clip's valid frame count.
4. The quantizer clips values to `[-7, 2]`, counts strict comparisons against 15 downward-rounded boundaries, and emits int32 IDs in `[0, 15]`.
5. The host bridge compacts valid `[frame, 80]` rows in clip order. Prompt preparation expands one audio placeholder per clip to exactly the corresponding valid-frame count.
6. Optional images are processed through the existing Inkling tiler and HMLP tower. Image placeholders expand to their actual tile counts.
7. The model normalizes text embeddings once, scatters HMLP rows first, runs compact dMel IDs through the summed-channel audio tower, and scatters audio rows second.
8. The wrapper's prepared-embedding prefill methods feed the merged tensor directly to the Inkling decoder. Server scheduling, chunked-prefill, Neural Accelerator alignment, and MTP burst gates observe the prepared tensor and keep the request on the classic path.

## 4. Technical Decisions

### 4.1 Preserve reference bin decisions rather than approximate values

The frontend computes the filterbank and bin centers in f64 because their derivation is stable there, then follows the public implementation's f32 execution boundary. Every quantizer midpoint is adjusted to the greatest representable f32 not above the f64 midpoint. The comparison is `value > boundary`, not `>=`, which makes exact ties deterministic and keeps them in the lower bin.

Mel multiplication accumulates one f32 dot product per output row. This deliberately avoids a wider or differently associated matrix multiplication that can perturb values near a bin boundary. Tests compare random-noise output against an independent f64 reference and explicitly exercise every boundary.

### 4.2 Compact padding before the tower

Batch extraction uses right padding for efficient host work, but the model contract contains no padding mask. The bridge therefore walks feature rows and the boolean mask together, retains only valid rows, verifies exact allocation cardinality, and constructs the MLX array as `[total_valid_frames, 80]`. Placeholder counts and tower output rows must match exactly before scatter.

### 4.3 Compose with HMLP without double normalization

The text backbone exposes normalized input embeddings and prepared-embedding forward methods. `InklingVlModel` uses those APIs rather than reimplementing decoder behavior. Image preparation returns an already-normalized and image-scattered tensor; audio merge accepts that tensor and changes only audio placeholder rows. Audio-only requests start from the same normalized text tensor. A synthetic regression compares the combined helper with an explicit image-first then audio merge and verifies special-token suppression.

### 4.4 Keep synthesized server audio in the current user turn

The server's normalized text view may omit `input_audio` parts. When a template placeholder survives, expansion happens in place and exact media cardinality is required. Otherwise, the runtime resolves the public Inkling structural tokens, finds the final `[message_user, content_text]` boundary, requires its following end marker, rejects later user content, and inserts audio wrappers immediately before that marker. A regression includes system/history content, a companion image part, a current question, and a model-generation tail.

### 4.5 Exclude multimodal prefill from text-only MTP

InklingVLM remains an MTP target for text-only requests. Raw image/audio payloads and `vlm_embeddings` are independent gates because the raw media can be consumed before speculative dispatch sees the sequence. Tests cover raw audio and the post-preparation state where only the merged tensor remains.

## 5. Review and Hardening

The inline correctness, security, and performance review found no unresolved CRITICAL or HIGH issue. The final implementation includes the following hardening:

- checked arithmetic bounds feature, token, and allocation counts before host or MLX construction;
- WAV and aggregate-duration admission occurs before model-specific feature work;
- non-finite input, invalid sample rates, zero frames, malformed shapes, out-of-range bins, and placeholder mismatches fail with errors rather than reaching unchecked MLX operations;
- audio special-token IDs are resolved from `processor_config.json` and the placeholder ID must match `config.json`;
- visual-shell detection remains dependent on both visual config and visual weights, so audio weights cannot fabricate a VLM runtime;
- server audio admission consults the loaded model capability before preprocessing;
- prompt synthesis uses a structural current-user boundary rather than a global end-token search;
- prepared audio tensors are excluded from chunked/speculative paths that would discard or misalign them.

Independent PR review and CI remain required before merge. This report does not treat local green tests as merge approval.

## 6. Validation

| Gate | Result |
| --- | --- |
| `cargo test --lib inkling -- --nocapture` | Pass, 77/77 |
| `cargo test --lib inkling_audio -- --nocapture` | Pass, 6/6 |
| `cargo test --lib mtp_burst_declines -- --nocapture` | Pass, 3/3 |
| `cargo test --lib detect_model_media_support_recognises_inkling_video -- --nocapture` | Pass, 1/1 |
| `cargo check --lib` | Pass |
| `cargo check --bin mlxcel` | Pass |
| `cargo clippy --lib -- -D warnings` | Pass |
| `cargo clippy --bin mlxcel -- -D warnings` | Pass |
| `cargo fmt --all -- --check` | Pass |
| `git diff --check` and `git diff --cached --check` | Pass |

The focused Inkling filter covers the f64 reference frontend, frame masks, strict quantizer boundaries, compact row ordering, summed-channel tower, processor config, weight normalization, visual detection, current-user prompt insertion, mixed image/audio scatter, prepared prefill, MTP state behavior, and raw/prepared speculative gates. PR CI remains responsible for the broader workspace, feature, and platform matrix.

## 7. Validation Limits and Follow-up

The requested `models/Inkling-Small-mlx-4bit` checkpoint is approximately 153.5 GB, and the native NVFP4 checkpoint is roughly 170.7 GB. Neither artifact was present on the validation host. Apple GPU generation, real CLI/server transcription, word error rate, peak unified memory, and throughput were not measured. No real-model claim is inferred from the deterministic frontend or tiny synthetic graphs.

A checkpoint-backed follow-up should compare the exact host dMel IDs with public mlx-vlm, compare tower rows within the issue's relative tolerance, run the clean speech fixture through both CLI and `/v1/chat/completions`, verify placeholder count equals `ceil(samples / 800)`, and measure mixed image/audio prefill memory separately from subsequent text decode.

## References

- Epic #1313 and issue #1311
- PR #1548 and prerequisites PR #1532, PR #1535, PR #1540, and PR #1546
- Public mlx-vlm Inkling audio, processor, feature-extractor, and model implementations
- `docs/supported-models.md`
