# PR #2575: Add SAM 3.1 multiplex tracking

**Date**: 2026-10-23
**Reviewer**: -
**Status**: Partial (library implementation, ROCm parity, CLI validation, clippy, and repository contracts complete; central synchronization and final CI pending)
**Languages**: Rust, Python, Markdown
**Risk Level**: High

## Executive Summary

PR #2575 adds native SAM 3.1 open-vocabulary detection, prompted image segmentation, and Object Multiplex video tracking to mlxcel. It reuses the SAM 3 ViT backbone, CLIP text path, DETR detector, geometry and mask components, shared FPN implementation, prompt handling, and instance-mask output contract while adding the SAM 3.1 three-branch neck, multiplex mask decoder, decoupled memory attention, and bounded tracker state.

Library and real-checkpoint validation is complete on the pinned Linux ROCm gfx1151 environment. The released `mlx-community/sam3.1-bf16` checkpoint contains 1,961 F32 tensors rather than BF16 tensors, and mlxcel preserves those F32 parameters and F32 inputs. The real three-frame traffic test tracks nine cars as 27 non-empty masks, keeps stable IDs, reaches mask IoU of at least 0.95 for every object and frame, and invokes `track_step` exactly twice regardless of object count. Current-head CLI image and video artifacts, architecture dispatch, generation redirects, clippy with warnings denied, the eight repository contracts, formatting, and diff checks pass. Synchronization with the VDA and MoGe work and final PR CI remain pending.

## 1. Problem Statement

SAM 3.1 retains SAM 3's detector and image-segmentation stack but replaces its tracking path with Object Multiplex. A faithful port therefore needs to share the large detector implementation without copying it, load three FPN branches for detection, interaction, and propagation, map an arbitrary object count into 16-slot tracker buckets, and preserve the shared source-pixel instance and mask output contract.

The public checkpoint also creates a precision-policy trap: its repository name contains `bf16`, but the pinned safetensor contains only F32 parameters. Treating the repository label as the tensor dtype would change inference. Video tracking adds a second risk because source-resolution masks, retained spatial memory, object pointers, conditioning concatenations, and attention projections can create a large working set before output serialization.

## 2. Change Summary

| Area | Change |
|---|---|
| Shared SAM 3 reuse | Exposed reusable detector, prompt, mask-decoder, memory, and FPN hooks. The SAM 3.1 model composes the existing ViT, CLIP, DETR, geometry, segmentation, and preprocessing implementations instead of duplicating them. |
| Three-branch neck | Added detection, interactive, and propagation FPN heads at the SAM 3.1 scales while keeping FPN resize and projection math in the shared SAM 3 module. |
| Object Multiplex | Added gather-based multiplex and demultiplex state, valid-slot padding, a 16-object bucket size, multiplex mask decoding, decoupled memory attention, temporal position encoding, and bounded conditioning history. |
| Video tracking | Detects objects on frame 0, applies optional pixel boxes as region filters, initializes one multiplex session, and performs one `track_step` for every later frame for all objects together. |
| Precision and loading | Preserves the checkpoint's actual F32 tensors and F32 model inputs, reuses the idempotent SAM 3 checkpoint sanitizer, and validates both supported quantization metadata locations. |
| Safety and resources | Rejects malformed shape-critical configuration and non-finite raw tensors, checks detector/tracker tensor shapes, limits objects and prompt expansion, and applies checked retained-plus-transient tracker memory accounting before source-mask resize. |
| Registry and CLI integration | Registers `sam3.1` and `sam3.1_video`, advertises `detect`, `segment`, and `track`, preserves all video frames for explicit tracking without a target FPS, and documents the shared prompt/output surface. |
| Reproducible evidence | Adds pinned two-cat and three-frame traffic references, 27 binary mask PNGs, portable reference scripts, source and output hashes, exact runtime provenance, and binary-asset budget coverage. |

## 3. Technical Decisions

### 3.1 Reuse the SAM 3 detector and FPN implementation

SAM 3.1 shares the expensive image and text encoders and detector stack with SAM 3. The implementation extracts reusable detector hooks and moves FPN construction into a shared module. `TriFpnNeck` loads three checkpoint roots but delegates each branch to the same FPN layer code. Frame 0 computes the backbone once and reuses it for detection and tracker initialization; every later frame computes one backbone and one tracker feature path.

This approach keeps SAM 3 checkpoint behavior covered by its existing parity suite and avoids parallel detector implementations that could diverge in preprocessing, dtype, box conversion, NMS, or mask interpolation.

### 3.2 Preserve object order across multiplex buckets

`MultiplexState` assigns objects to fixed 16-slot buckets, pads unused slots, and records the inverse object-to-slot mapping. Decoder and memory operations run in bucket space, then demultiplex back to the frame-0 object order. The 1-based `object_id` is therefore stable across frames and across the transition from one bucket to two buckets.

Unit coverage exercises 3 objects with padding, a full 16-object bucket, and 17 objects spilling into two buckets. The video acceptance test verifies that nine objects still require only one `track_step` call per later frame.

### 3.3 Trust tensor metadata rather than the checkpoint name

The pinned `mlx-community/sam3.1-bf16` revision contains 1,961 F32 tensors. Runtime probes confirm that all 1,961 loaded parameters remain F32, preprocessed inputs are F32, and tracker logits and scores are F32. The loader therefore preserves the data actually stored in the checkpoint instead of coercing it based on the repository name.

This differs from the SAM 3 BF16 acceptance path, which remains covered independently. The PR also reruns real affine 4-bit SAM 3 inference to ensure that the shared loader and detector refactor did not break quantized weights.

### 3.4 Bound failure before allocation-heavy transforms

Configuration validation runs before model construction. It validates the shared detector heads and dimensions, both quantization blocks, and the exact published SAM 3.1 decoder, memory-attention, downsampler, fuser, RoPE, multiplex, and history constants. Raw detector and tracker logits, boxes, presence values, and selected masks must be finite before sigmoid, interpolation, or thresholding. Tensor ranks and query/object cardinalities are checked before indexing.

Video initialization filters selected query-instance pairs by region box and checks the prospective object count and working-set budget before source-resolution resize. The 512 MiB guard uses checked arithmetic for retained output, tracker and resize work, retained frame features, pointers, conditioning concatenations and projections, and current-frame attention activations. Tests cover overflow-amplifying dimensions and a 12-object, 32-frame case that exceeds the bound.

### 3.5 Keep the shared prompt and output contract

Image `detect` and `segment` use the shared repeatable text and pixel-box prompt surface. Video `track` requires text prompts; supplied boxes act as source-pixel regions that retain text detections whose IoU exceeds the upstream threshold. Frame-0 instances define stable labels and object IDs. Later scores are `sigmoid(object_score_logits)`, boxes come from nonzero mask extents, masks use nearest-neighbor source resize and threshold `> 0`, and rows use the shared `Instance` and `DenseOutput` contract.

A video with no passing frame-0 detections writes an empty result with frame metadata, and a single-frame video writes frame 0 without calling `track_step`.

## 4. Provenance

| Component | Pinned value |
|---|---|
| Upstream implementation | `Blaizzy/mlx-vlm` commit `abd3b2d7b728cb5372d4e092260da325e5ded209` |
| SAM 3.1 checkpoint | `mlx-community/sam3.1-bf16` revision `a992e302ea9b0f03f41dfd93414a4fd0e818f65b` |
| Checkpoint safetensor | SHA-256 `a1b1c19dcc9bdd68438bcd74433fadc90740e73c37a1f386872672d134879c42`; 1,961 F32 tensors |
| Processor assets | `mlx-community/sam3-bf16` revision `dfe573c3171dbcfda8399c650d9135afa7e94592` |
| Reference runtime | MLX 0.32.5 from the exact mlxcel ROCm archive |
| Native library | SHA-256 `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094` |
| Python binding | SHA-256 `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc` |
| Validation device | ROCm gfx1151, AMD Radeon 8060S Graphics, 20 compute units, warp size 32 |
| Traffic source | `opengeos/datasets` cars video, source SHA-256 `fdfe751efd5423cb70be1440bb2e9850d63715bf3ccb67fd2743f1bb514b687f` |

`tests/fixtures/sam3_1/provenance.json` records the source checkout command, checkpoint and processor hashes, exact native paths and hashes, device selection, TF32 control statement, clip and decode commands, portable reference scripts, committed fixture hashes, and captured output hashes. The three-frame committed MP4 has SHA-256 `115950dbc5d9572e6103d2e9649ab4dbc7e73952696195cb0e1380ecd4ea9e7a`.

## 5. Runtime Validation

### 5.1 Unit, registry, and bucket behavior

| Case | Result |
|---|---|
| SAM 3.1 focused unit suite | 14 passed, 0 failed, 2 real-checkpoint tests ignored |
| Registry focus | 3 passed, including both model aliases and mask capabilities |
| Multiplex sizes | Round-trip and padding checks passed for 3, 16, and 17 objects |
| Conditioning | Closest-frame selection and pinned temporal sine-position literals passed |
| Edge cases | Empty detection, single-frame tracking, non-finite output rejection, Pillow mode-F mask resize, and memory-bound checks passed |

### 5.2 Real F32 cat segmentation

The real pinned checkpoint loaded with F32 parameters and F32 inputs and completed prompted segmentation in 8.14 seconds. It returned the same two detections as the upstream reference. Scores are within 0.01, every box coordinate is within 2 pixels, and both source-resolution masks meet the required IoU of at least 0.98. Output scores remain F32 and masks are u8.

### 5.3 Nine-object video tracking

The three-frame 640x360 traffic fixture initializes nine real cars and produces 27 instance rows and 27 non-empty masks. Every frame contains all nine objects; stable `object_id` values, scores within 0.01, boxes within 2 pixels, and per-mask IoU of at least 0.95 match the pinned upstream reference. Every reference foreground count is greater than zero, and the actual masks are non-empty because each has positive overlap with its reference.

The tracker records exactly two `track_step` calls, one for each frame after frame 0, rather than one call per object. The same real test also verifies the single-frame contract and the empty-threshold contract. It passed in 29.44 seconds.

### 5.4 Existing SAM 3 regressions

The final binary passed 15 focused SAM 3 unit tests with six accelerator-backed cases ignored in the focused run. It then passed all four existing real BF16 reference cases in 33.50 seconds: 24-layer text parity, text-only cat segmentation, box-prompt segmentation, and three-frame video tracking. Cat mask IoUs remain 1.0 and 1.0; box-prompt mask IoUs remain 0.9999825 and 1.0; the text projection retains BF16 behavior.

The real affine 4-bit checkpoint also loaded and completed prompted segmentation in 6.15 seconds. This confirms that the shared detector, FPN, validation, and loader changes preserve both BF16 and quantized SAM 3 paths.

### 5.5 Validation logs

| Log | Evidence |
|---|---|
| `issue-2430-compile-v3.log` | Final V6 test-fast library test binary built successfully in 10 minutes |
| `issue-2430-sam31-units.log` | SAM 3.1 unit and edge-case coverage; 14 passed |
| `issue-2430-sam31-registry.log` | SAM 3.1 aliases and architecture capabilities; 3 passed |
| `issue-2430-sam3-units.log` | Existing SAM 3 focused regressions; 15 passed |
| `issue-2430-sam31-real-cat.log` | Real F32 two-cat segmentation parity; 1 passed |
| `issue-2430-sam31-real-video.log` | Nine-car, three-frame multiplex parity and edge contracts; 1 passed |
| `issue-2430-sam3-real-bf16.log` | All four existing real SAM 3 BF16 cases; 4 passed |
| `issue-2430-sam3-real-4bit.log` | Real affine 4-bit SAM 3 inference; 1 passed |
| `issue-2430-gpu-batch.log` | Exclusive ROCm guard held continuously; every target exited 0 |
| `issue-2430-validation-summary.md` | Frozen-source, binary hash, validation results, reviews, and pending gates |
| `issue-2430-clippy-v2.log` | Current-head `cargo clippy --lib --tests -- -D warnings`; passed in 21 minutes 4 seconds |
| `issue-2430-cli-build-v2.log` | Current-head ROCm CLI binary; built in 9 minutes 19 seconds |
| `issue-2430-fast-contracts-final.log` | Eight repository contract checks; all passed |
| `issue-2430-cli-gpu-batch.log` | Guarded image and video CLI predictions; both exited 0 |
| `issue-2430-cli-metrics.json` | JSON, PNG, tensor dtype, shape, instance, and numerical artifact validation; exited 0 |
| `issue-2430-cli-arch.json` | Architecture family, aliases, runtime, and output-kind dispatch validation |
| `issue-2430-cli-redirect-alias.log`, `issue-2430-cli-redirect-video.log` | Both generation commands rejected with the expected `use mlxcel predict` message and exit 1 |

The compiled V6 test binary is `target/test-fast/deps/mlxcel-d5e762440845331a` with SHA-256 `a7af693e828828b5f7b5d32b2949f9881c061aec9729d9711b077cec2ce7718c`. Its library runtime and real-checkpoint parity runs preceded the later mechanical CI fixes: removing unused private tracker-ID storage, replacing divisibility expressions with `is_multiple_of`, and keeping the binary-asset budget expression on one parseable line. These changes do not alter model arithmetic or the public output contract. Current-head clippy and the actual CLI run cover the post-fix source; that CLI binary is `target/test-fast/mlxcel` with SHA-256 `c8dd6bb4d56a7fc447895ebfbe93790476c10e7d4a0ba4ae6be815b1a81a4bfb`. `cargo fmt --all -- --check`, `git diff --check`, and all eight fast repository contracts passed. Independent implementation review and independent security/performance review closed with no findings on the final reviewed snapshot.

### 5.6 CLI artifact and dispatch validation

The current-head CLI binary completed real image and video prediction under the ROCm guard. Image prediction intentionally omitted `--task`, exercising the default `segment` path. It wrote `result.safetensors`, two stable PNG masks, `instances.json`, and `meta.json`. Both mask IoUs are 1.0; the maximum absolute box error is `0.000064765625` pixels and the maximum absolute score error is `0.0000000205734253`.

Video prediction wrote `result.safetensors`, 27 stable `fNNNNNN_oNNN.png` files, `instances.json`, and `meta.json`. The validator found 27 non-empty masks with stable IDs across three frames and a minimum mask IoU of `0.999113475177305`. It also checked that tensor masks are u8 while boxes and scores are F32, and cross-checked the JSON rows, tensor rows, and PNG artifacts.

`mlxcel arch --json` reports family `sam3_1`, model types and aliases `sam3.1` and `sam3.1_video`, runtime `predict`, and output kind `masks`. Calling `mlxcel generate` with either model type returns the expected exit code 1 and directs the caller to `mlxcel predict`.

### 5.7 Pending integrated evidence

The following items are not yet acceptance evidence and must be updated after central synchronization:

- Central integration of the preceding Video Depth Anything and MoGe work, followed by focused SAM 3.1 and shared SAM 3 regression runs.
- Final PR #2575 CI on the integrated head, including repository asset/provenance and required platform gates.

## 6. Risk and Compatibility

This PR adds no server endpoint and retains the shared prompt, instance, mask, and artifact contracts introduced by SAM 3. Its checkpoint sanitizer remains idempotent and rejects mixed convolution layouts. Region boxes filter text detections rather than replacing text prompts, and video defaults remain explicit: callers must request `track`.

The runtime parity evidence is limited to Linux ROCm on gfx1151 with the pinned native MLX library. This report makes no full-checkpoint Metal or CUDA parity claim. Platform CI can establish compilation and ordinary test compatibility, but it is not model parity evidence unless it runs the pinned real-checkpoint tests.

Realtime webcam caching, mid-video object addition and reconditioning, interactive point refinement, conversion from the gated Meta `.pt` checkpoint, and annotators remain outside issue #2430.

## 7. Follow-up Actions

- Synchronize the VDA and MoGe work into the central integration branch and rerun focused SAM 3.1 and SAM 3 regressions.
- Run final PR CI on the integrated head and update this report with the resulting evidence.
- Mark PR #2575 ready only after the pending draft gates pass.

## References

- PR #2575: `feat: add SAM 3.1 multiplex tracking`
- Issue #2430: Port SAM 3.1 with Object Multiplex video tracking
- PR #2551 / issue #2429: shared SAM 3 detector, prompts, instances, masks, and output artifacts
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: shared CLI and output contract
- `tests/fixtures/sam3_1/provenance.json`: source, checkpoint, runtime, and fixture provenance
