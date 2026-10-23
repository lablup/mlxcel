# PR #2576: Add Video Depth Anything Dense Prediction

**Date**: 2026-10-23

**Status**: Implementation and local validation complete on final integration head `45e895ab`; current CI pending

**Languages**: Rust, Python, Markdown

**Risk**: Medium

## Executive Summary

PR #2576 adds offline Video Depth Anything inference for still images and videos through the dense-prediction API and CLI. It ports the shared-DINOv2 temporal DPT graph, OpenCV-compatible preprocessing, 32-frame overlapping-window inference, relative scale/shift alignment, metric-scale bypass, source-resolution F32 depth output, per-frame 16-bit previews, registry metadata, and documentation for all six published Small, Base, and Large relative/metric checkpoint families.

The implementation has completed its local validation gates on final integration head `45e895ab`. A final library/test build, 14 focused unit tests, focused clippy, formatting, all six real-checkpoint load-only cases, consecutive-prefix regression coverage, Small relative and metric 40-frame parity on both ROCm and CPU, and actual 64-frame ROCm CLI inference passed. The complete authentic fixture and provenance set is committed, the exact binary inventory passes, and all eight repository contract groups pass after rebasing onto `f69dd301`. Independent implementation and security/performance reviews have no remaining findings. All non-skipped CI checks passed before the rebase; CI for the rebased final head is pending.

Runtime results remain bound to the original validated implementation commit `3f01fb6a` and its exact source and binary hashes. Later commits contain independently reviewed equivalent lint rewrites, test extraction, a comment-only license header, and fixture/provenance files. The reports do not relabel the earlier executions as runs of the later source. Range-diff confirms that all six issue commits are patch-equivalent after the conflict-free rebase onto exact integration base `f69dd301`; the rebased implementation commit is `bcf6e2a7`.

## 1. Problem Statement

mlxcel could decode videos and had a shared DINOv2 backbone, but it could not dispatch `video_depth_anything` checkpoints or produce temporally consistent depth. Supporting the published checkpoints requires more than a per-frame depth head: preprocessing, temporal motion modules, overlapping-window conditioning, relative-window alignment, metric bypass, padding, interpolation, source-resolution restoration, and output metadata must all match the upstream offline predictor.

The main correctness risks are temporal drift at a window boundary, incorrect preprocessing geometry, checkpoint layout or dtype coercion, and sampling a sparse subset when `--max-frames` is intended to keep a consecutive prefix. Long clips and malformed configurations also require checked allocation bounds because each 32-frame window retains backbone, decoder, resized depth, and host alignment state.

## 2. Change Summary

| Area | Change |
|---|---|
| Model graph | Added the shared-DINOv2 VDA wrapper, four-scale DPT decoder, AnimateDiff-style temporal attention and GEGLU blocks, feature fusion, and depth head. |
| Checkpoint loading | Added published Small, Base, and Large presets, relative and metric modes, convolution layout sanitization, one model-wide floating-dtype policy, uniform dtype checks, and exact linear, normalization, convolution, and temporal-head shape checks. |
| Preprocessing | Reproduced the upstream 518-pixel lower-bound rule, 14-pixel multiples, wide-aspect adjustment, OpenCV cubic sampling with border replication, ImageNet normalization, and source-size validation. |
| Temporal inference | Added exact 32-frame conditioning, 10-frame overlap, upstream keyframes, last-frame padding and trimming, relative least-squares scale/shift alignment, metric bypass, and eight-frame overlap blending. |
| Dense and CLI integration | Registered `video_depth_anything` for image/video `depth`, returns F32 `depth` with shape `(T,H,W)`, preserves source resolution, writes 16-bit frame previews, and records metric mode and units. |
| Video sampling | Added a keep-all/consecutive-prefix policy for dense temporal predictors while preserving existing decoded-byte and frame limits; default multimodal sampling behavior remains unchanged. |
| Safety and provenance | Added source, preprocessing, output, model-workset, token, window, dtype, layout, and finite-value checks; authentic reference generation pins source, checkpoint, input, decoded frames, native runtime, Python binding, and TF32 state. |
| Documentation and tests | Added operator documentation, architecture capabilities, environment variables, focused unit coverage, genuine 40-frame reference harnesses, six-checkpoint load tests, CLI output validation, and authenticated recovery records. |

From final integration base `f69dd301` through current head `45e895ab`, the branch changes 71 tracked files with 9,259 insertions and 7 deletions. The six rebased VDA commits are:

- `bcf6e2a7` — implementation and integration; patch-equivalent to runtime-validated `3f01fb6a`
- `a00a9c0e` — three lint-only equivalent rewrites
- `0fa70ed6` — extraction of eight byte-equivalent predictor tests
- `dc670ca3` — comment-only test license header
- `e52a3ee2` — authentic fixtures, provenance, and binary policy updates
- `45e895ab` — archived CPU CLI parity evidence and checksums

## 3. Technical Decisions

### 3.1 Reuse DINOv2 without changing checkpoint precision

VDA uses the shared DINOv2 implementation and selected normalized intermediate layers, but applies one checkpoint-wide backend dtype policy before constructing either the backbone or temporal head. A scoped preserve-dtype constructor prevents the legacy standalone DINO load path from converting only the backbone while the VDA head remains in another dtype. Every loaded tensor must then share the same supported floating dtype, and every consumed layer validates its exact configured shape.

### 3.2 Preserve the upstream offline window contract

Each model call receives 32 frames. New windows advance by 22 frames, condition on the upstream keyframe set, repeat the final source frame when necessary, and trim padding after stitching. Relative checkpoints fit scale and shift against the overlap before blending; metric checkpoints bypass alignment to preserve metres. Alignment is performed as each window completes so the predictor does not retain all raw 32-frame outputs for the full video.

### 3.3 Match preprocessing before numerical comparison

The processor derives a patch-aligned target with the upstream lower-bound and wide-aspect rules. It applies OpenCV-style cubic coefficients to F32 RGB values with replicated edge indices, followed by ImageNet normalization. This keeps processor differences from being misattributed to the transformer or temporal head during parity analysis.

### 3.4 Bound live memory across host and model work

The predictor checks source dimensions before RGB conversion, caps frames and processed pixels, and uses checked arithmetic for host outputs, overlap references, conditioned frames, and model intermediates. The DPT high-resolution refinenet and output head run in groups of four frames. The combined preflight covers retained DINO features, projected pyramid levels, temporal paths, resized head tensors, source-resolution output, and host alignment buffers before inference begins.

### 3.5 Authenticate genuine reference evidence

Reference tooling validates revision-pinned source files and checkpoints before importing MLX, rejects `MLX_ENABLE_TF32` values other than `0`, and records the exact mapped native library and Python binding hashes. Fixtures store complete 40-frame outputs instead of sparse probes. Rust comparisons require exact shape, F32 dtype, finite values, checkpoint identity, decoded input identity, processor identity, and device-specific RMS thresholds.

## 4. Architecture and Data Flow

```text
image or consecutive video frames
    -> source/frame/workset validation
    -> OpenCV-compatible resize + ImageNet normalization
    -> 32-frame conditioned windows
    -> shared DINOv2 selected intermediate features
    -> four-scale DPT projections and temporal motion modules
    -> chunked high-resolution depth head
    -> source-resolution bilinear depth
    -> relative overlap alignment or metric bypass
    -> eight-frame blend + padding trim
    -> F32 depth [T,H,W] + metadata + 16-bit previews
```

A still image uses the same temporal model path by repeating its single frame for the conditioned window and returning one trimmed output frame. When the CLI receives `--max-frames` without `--target-fps`, the shared video loader keeps the first requested consecutive frames so temporal adjacency is preserved.

## 5. Security and Performance Review

Independent review findings are closed in the current source. The loader rejects malformed model types, unsupported encoder presets, incompatible temporal dimensions, mixed or non-floating checkpoint dtypes, non-positive tensor shapes, unexpected biases, and exact layer-shape mismatches. Inference rejects inconsistent frame dimensions, excessive source or processed sizes, arithmetic overflow, excessive combined worksets, malformed model output lengths, and non-finite model or alignment values.

The implementation bounds work and does not claim a throughput improvement. Full 32-frame temporal attention remains the intended upstream algorithm, while high-resolution decoder work is chunked to keep peak allocations within the validated budget. No performance benchmark was run.

## 6. Verification

### 6.1 Source and binary identity

The runtime validation is bound to historical implementation commit `3f01fb6a3d2d20c366a4673c0da3cb3d130961a7` and source-manifest SHA-256 `fc330220267a3968ee6aacb72a6c8c0716e0109789bd060fb7bdab7f869f389a`. The rebased implementation commit `bcf6e2a7` is patch-equivalent. The final library/test binary SHA-256 is `19a3f527055d41e45063c47f493d59e7831879cda6b9b7658ac53ee8995b6d1b`; the CPU CLI captures use binary SHA-256 `dcc303d1e3a3c62ec3aa004ce71bd68f2f1c2160983620c8175cfa31ee41809b`.

The later lint commit has source-manifest SHA-256 `7273c98d90b33290d47d75012cb3c2bc0783b01306cbe0918589e39d0de43f51`. Its exact diff SHA-256 is `de2c434000682db93b26ebc22b3b4deb91f382fe348d6ce4272d264c73f5aadd`: two integer `max(1).min(16)` expressions became `clamp(1,16)`, and one needless shared-slice borrow was removed. Independent review found no math, control-flow, or output change. Test extraction preserved all eight test bodies after de-indentation, the license addition is comment-only, and the fixture commit does not change production Rust model code. Earlier runtime artifacts remain labeled with the original source and binary.

### 6.2 Build, unit, load, lint, and repository gates

- Final library/test build: exit `0`.
- Focused unit tests: 14 passed, covering target sizing, 40/100-frame window plans, alignment, interpolation, processor behavior, allocation guards, and video prefix sampling.
- Real checkpoint matrix: all six Small/Base/Large relative/metric checkpoints passed sequential load-only validation with expected configurations, layouts, and F32 backbone, head, and normalization tensors.
- Focused `cargo clippy --lib --tests -- -D warnings`: exit `0` in 21 minutes 43 seconds; formatting passed.
- Authentic fixture inventory: 49 tracked files, including the decoded 40-frame input, both complete CPU outputs and provenance records, recovery audit material, CPU CLI acceptance receipts and checksums, generator/downloader, native-runtime provenance, checkpoint matrix, and the 64-frame clip.
- Post-rebase contracts: formatting, diff, license, binary policy, CPU evidence checksums, and all eight final repository contract groups exited `0`. The contract log SHA-256 is `d287974b158243a1a4a9b6820fd0b002cbc144326699d3b4c07af77e79d2ef59`; the exact binary-assets check reported 40 declared binary files using 10,295.8 KiB of the 10,505 KiB budget.
- CI: every completed non-skipped check passed before the rebase. Checks for final head `45e895ab` are pending.

Base and Large evidence is load-only. No Base or Large forward inference is claimed.

### 6.3 Genuine ROCm 40-frame parity

The fixed 40-frame DAVIS clip crosses one window boundary and exercises conditioning, overlap handling, alignment, blending, and trimming. Both Small checkpoints passed the `1e-2` ROCm threshold on gfx1151.

| Checkpoint | Frames | Relative RMS | Threshold | Result |
|---|---:|---:|---:|---|
| Small relative | 40 | `4.912878453e-6` | `1e-2` | Passed |
| Small metric | 40 | `7.973837467e-7` | `1e-2` | Passed |

### 6.4 Genuine CPU 40-frame parity

Both Small CPU results compare complete finite F32 `[40,96,128]` tensors with the independently reviewed fail-closed comparator, SHA-256 `37c3616b4d6ce4087ba14020839fa7fa6f8db2255203d4d84e68f40e21a30386`, under `python -O`. Both pass the strict `1e-4` threshold.

| Checkpoint | Relative RMS | Threshold | Result |
|---|---:|---:|---|
| Small relative | `5.142513694620207e-6` | `1e-4` | Passed |
| Small metric | `7.445840126949833e-7` | `1e-4` | Passed |

The metric upstream oracle is a clean run: Python and its wrapper exited `0`, the complete two-window log is committed under `metric_cpu_run/`, and the output and provenance hashes match the fixture inventory.

The relative upstream oracle has a documented recovery boundary. Its wrapper exited `127` after a shell command-not-found error, the Python exit code was not independently captured, and the final detailed generator log was lost. The complete output and legacy provenance were accepted only after an independent full audit authenticated their hashes and invariants and compared the recovered CPU tensor with the pinned GPU output. The CPU and GPU outputs are distinct, with CPU-to-GPU relative RMS `3.4021821538643326e-7` and maximum absolute difference `5.793571472167969e-5`. The exact review packet is committed under `relative_cpu_recovery/`; no clean completion line or Python exit is reconstructed.

For both Rust CPU captures, the actual CLI exited `0` and an independent saved-output validator exited `0`. The surrounding wrappers exited `1` afterward because of a case-only device-display assertion. Those wrappers did not persist a `finished_unix` value, so the reports do not invent one. The capture script changed while the runs were active: the exact pre-edit executed bytes and SHA were not preserved, and the authenticated archived script is explicitly labeled as a post-edit snapshot rather than execution proof. These limitations are recorded in the parity receipts; inference was not rerun or relabeled.

### 6.5 Real ROCm 64-frame CLI output

Both Small checkpoints completed actual CLI inference and wrote F32 depth tensors with shape `[64,96,128]`, 64 valid 16-bit preview PNGs, and metadata identifying the family, task, input size, frame count, metric mode, and units. Relative output reports `relative_disparity`; metric output reports `metres`.

The relative CLI command exited `0`; its original external wrapper later exited `1` only because it compared the runtime device display with the wrong letter case. An independent corrected validator checked the saved output without rerunning inference. The metric wrapper completed with exit `0` and a complete log.

## 7. Limits

- Base and Large relative/metric checkpoints have genuine load-only coverage; this report makes no Base or Large forward, parity, or memory claim.
- Metal runtime validation was not performed.
- Streaming, webcam input, and a server route remain outside scope.
- No throughput, latency, or memory benchmark was performed.
- Runtime parity is proven for the Small checkpoints on CPU and ROCm gfx1151 with the stated provenance boundaries.
- CI for final rebased head `45e895ab` remains pending.

## 8. Learning Points

- Temporal model parity must cross a window boundary; a clip shorter than 32 frames cannot verify conditioning, alignment, and overlap blending.
- Relative and metric checkpoints share the graph but require different stitching semantics because scale/shift correction would corrupt metric units.
- Keeping a consecutive prefix is part of model correctness for temporal predictors, not only a video-loader preference.
- Checkpoint provenance must bind the actual tensors, preprocessing input, runtime library, device controls, and full output artifact to make numerical parity auditable.
- Recovery evidence can support acceptance only when its missing facts remain explicit and independent checks authenticate the complete artifacts without reconstructing unavailable run state.

## 9. Remaining Work

- [ ] Let repository CI complete on final rebased head `45e895ab`.

## References

- PR #2576: Video Depth Anything dense prediction
- Issue #2352: Video Depth Anything temporal monocular depth
- Epic #2382: dense-prediction model delivery
- Upstream `Blaizzy/mlx-vlm` source revision `abd3b2d7b728cb5372d4e092260da325e5ded209`
- `docs/dense-prediction.md`: VDA behavior and output contract
- `tests/fixtures/video_depth_anything/README.md`: authentic fixture provenance and regeneration
