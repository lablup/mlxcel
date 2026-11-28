# PR #2545: Add dense prediction output foundation

**Date**: 2026-10-23
**Status**: Partial (PR open; full ROCm workspace gate and CI rerun pending)
**Languages**: Rust, C++, Python, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2545 establishes the shared infrastructure for dense vision models that return maps and keypoints instead of text or bounding boxes. It adds the `mlxcel predict` CLI, an object-safe predictor and family registry, bounded image and video input, upstream-compatible interpolation and preprocessing, and a file contract built around f32 safetensors, JSON metadata, optional keypoints, and PNG previews. The family registry is intentionally empty in this foundation PR; later issues in epic #2382 add the model networks and post-processing.

The implementation also fixes a pinned MLX ROCm overlay prerequisite: `AsStrided` now selects the HIP device header and encoder instead of reaching the Metal path on Linux. Focused gfx1151 tests, a real ffmpeg MP4 path, package-scoped clippy, formatting, and repository contract gates passed. Metal validation, the new CI run, and the full ROCm workspace regression gate were not complete when this pre-merge report was written.

## 1. Problem Statement

mlxcel previously had token generation and an object-detection CLI, but no common output surface for per-pixel depth, normals, masks, segmentation, point maps, or human keypoints. Each planned dense model would otherwise need to invent its own input validation, interpolation, serialization, previews, and CLI behavior. This would make formats inconsistent and duplicate resource-safety work across Video Depth Anything, MoGe-3, Sapiens2, SAM 3D Objects, and their shared DINOv2 backbone.

The foundation also needed to compile on the project’s ROCm development path. The pinned MLX overlay retained an upstream CUDA-or-Metal dispatch in `AsStrided`, causing Linux ROCm builds to include a Metal header and preventing the new Rust paths from being compiled and tested on gfx1151.

Without this change, model ports would have no stable contract for operators or downstream tooling, and video/image inputs or repeated output writes could consume unbounded resources or overwrite unrelated files.

## 2. Change Summary

| Area | Change |
|---|---|
| CLI and input | Added `mlxcel predict` with exactly one of `--image` or `--video`, model task selection, JSON object options, frame/FPS controls, bounded payload/decode/allocation checks, and stage-specific errors. |
| Predictor architecture | Added `DenseInput`, `DenseOutput`, `PersonKeypoints`, the object-safe `DensePredictor` trait, and a `model_type` constructor registry. The production family table remains empty until model PRs land. |
| Tensor processing | Ported NHWC bilinear, bicubic, and nearest interpolation behavior; added ImageNet normalization, integer scaling, patch-multiple resize, and guarded OIHW-to-OHWI convolution conversion. |
| Output contract | Writes contiguous f32 `result.safetensors`, `meta.json`, optional `keypoints.json`, and depth/normal/mask/segmentation previews. Video previews use numbered directories. |
| Safe reruns | Materializes one tensor on the host at a time and replaces only writer-owned artifacts; stale preview paths and directory symlinks are handled without following an external target. |
| Registry integration | Added `Runtime::Predict`, `OutputKind::DenseMap`, and `OutputKind::Keypoints`; known dense checkpoints used with `generate` receive a redirect to `mlxcel predict`. |
| ROCm prerequisite | Corrected the pinned MLX `AsStrided` overlay to use the HIP encoder and refreshed overlay provenance. |
| Documentation | Added the dense CLI, output formats, preprocessing API, registration guide, scope limits, and README indexes. |

Implementation diff before this report: 29 files, +2,823/-8. Related commits are `65cb1fdd` (ROCm prerequisite), `aee9d056` (foundation), and `41775825` (documentation clarification).

## 3. Technical Decisions

### 3.1 A separate `predict` surface

Dense results do not fit the token stream from `generate` or the box-row contract from `detect`. A separate command keeps those interfaces stable while giving dense models an output directory, task selection, and file-oriented lifecycle. Extending `detect` was rejected because maps, intrinsics, and keypoints require different serialization and can be video sequences.

### 3.2 An empty registry before model ports

This PR defines the interface and validates it through a test-only stub family, but does not pretend that any dense checkpoint is already supported. A recognized-but-unported `model_type` produces a specific unsupported-family error, while accidental `generate` use points to `predict`. Model PRs therefore add only a predictor and one registry row against an already tested contract.

### 3.3 Safetensors as the lossless source, PNG as a preview

All tensor outputs are stored as contiguous f32 with their upstream keys and shapes. JSON records family, task, source size, elapsed time, finite ranges, model metadata, and flattened intrinsics. PNG files are derived previews: depth uses 16-bit normalization, normals map from `[-1, 1]`, masks threshold at 0.5, and segmentation stores 8-bit class IDs. Consumers that require numeric fidelity read safetensors rather than reverse-engineering a visualization.

### 3.4 Bounded input and conservative output replacement

Image loading enforces payload, decoder, and prospective RGB-f32 allocation limits. Video decoding reuses the shared ffmpeg loader, defaults to 32 sampled frames, and applies a 512 MiB decoded-frame budget. Output reruns delete only known artifacts created by the writer and retain unrelated files, which makes an existing directory usable without turning it into a recursive deletion boundary.

### 3.5 Shared interpolation follows the upstream numerical contract

The resize helpers use NHWC tensors and fixed upstream-generated PyTorch literals for bilinear and bicubic antialiasing, size and scale-factor modes, and nearest sampling in both upsample and downsample directions. Same-size and dtype-preserving behavior is documented because later model ports depend on these details for parity.

## 4. Architecture and Integration Contract

```text
image / video
      |
bounded loader and sampler
      |
model_type registry -> DensePredictor::predict(task, opts)
      |
DenseOutput { tensors, keypoints, meta }
      |
result.safetensors + meta.json + keypoints.json + PNG previews
```

Model implementations must expose an object-safe predictor, preserve upstream tensor names, declare supported tasks, attach preprocessing context to internal errors, and register their lowercase top-level `model_type`. The CLI provides the outer load, predict, and write contexts. Mesh and Gaussian export, streaming/webcam input, and a server route remain outside this contract.

## 5. Verification and Limits

| Check | Result at report time |
|---|---|
| Dense module tests on gfx1151 | 17 passed, covering interpolation, serialization/previews, registry behavior, guarded layout conversion, and preprocessing. |
| `predict` command tests | Five focused tests passed; the actual ffmpeg MP4 test was rerun with ffmpeg available and passed. |
| Registry and exact contract filters | 12 registry tests passed; parser, architecture JSON, generate redirect, catalog projection, and video-budget filters passed. |
| Output safety | Synthetic round-trip, numbered video previews, stale writer-owned cleanup, and directory-symlink replacement tests passed. |
| Lint and format | Package library and `mlxcel` binary clippy passed with `-D warnings`; `cargo fmt --all -- --check` and `git diff --check` passed. |
| Repository fast contracts | The orchestrated fast contract make gates passed before report generation. |
| Full ROCm workspace regression | Running when this report was written; no pass is claimed here. |
| CI after report commit | Pending; the report commit triggers a fresh PR run. |
| Metal | Not run on the ROCm Linux host; delegated to CI. |

No real dense model inference is claimed by this PR because the production family table is empty by design. No throughput or memory benchmark was performed; the resource work here consists of hard limits and one-host-tensor-at-a-time serialization rather than a measured performance optimization.

## 6. Learning Points

- Infrastructure can be tested before a model lands by registering a test-only predictor that exercises the real CLI, input decoding, task selection, and writer.
- A preview is not a numerical interchange format. Keeping lossless f32 tensors and recording the exact normalization range prevents visualization choices from becoming an implicit API.
- Resource budgets must include representation expansion. A compressed image that passes a payload limit can still exceed memory limits when decoded and converted to RGB f32.
- Safe reuse of an output directory requires an explicit ownership list. Removing only standard artifacts avoids recursive deletion of user files and makes symlink behavior testable.
- Numerical ports need fixtures produced by the reference implementation. Fixed PyTorch probes make interpolation drift visible without requiring Python or PyTorch in the Rust test environment.

## 7. Follow-up Actions

- Complete the full ROCm workspace regression gate and the fresh PR CI run before merge.
- Confirm Metal compilation and tests in CI.
- Add DINOv2 and the dense model families through the dependent issues in epic #2382.
- Add mesh/Gaussian export with the SAM 3D Objects implementation; define a server wire contract only when a client requirement exists.

## References

- PR #2545: `feat: add dense prediction output foundation`
- Issue #2349: dense-prediction output surface
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: operator and model-integration contract
