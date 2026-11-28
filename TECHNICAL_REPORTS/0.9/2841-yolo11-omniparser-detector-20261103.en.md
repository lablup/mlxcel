# PR #2841: YOLO11 detector for OmniParser v2 icon detection

**Date**: 2026-11-03
**Status**: Completed (merged; closes #2413)

## Executive Summary

PR #2841 adds native YOLO11 inference for the OmniParser v2 `icon_detect` checkpoint, which completes the detection half of OmniParser. `mlxcel detect` now dispatches `model_type: yolo11` to a port of mlx-vlm's graph: C3k2 backbone, SPPF, C2PSA attention neck, and an anchor-free DFL head at strides 8, 16, and 32. Pre- and post-processing follow mlx-vlm's `prepare_image` and `non_max_suppression`: a centred stride-32 letterbox on a 114-grey canvas, class-aware greedy NMS, then the inverse transform to original pixels.

The shared dispatch contract also changed: `load_detector` takes `Option<f32>` so an unset `--threshold` uses each family's own default, and `mlxcel detect` gained `--size` (longest side, native when unset) and `--iou` (NMS, default 0.1). Both flags are YOLO11-only and warn when another detector receives them. Ten files, +2114/-19, of which about 600 lines are tests and docs.

## 1. Change Summary

| Area | Change |
|---|---|
| `src/vision/detection/yolo11/` (new) | `config.rs`, `layers.rs`, `model.rs`, `weights.rs`, `predictor.rs`, and unit tests; exact tensor schema with unconsumed-key reporting |
| `src/vision/detection/mod.rs` | `yolo11` in `detector_info` / `load_detector`; threshold is now `Option<f32>`; `DetectorOptions` and `predict_path_with_options` |
| `src/commands/detect.rs` | `--size` / `--iou` validation before the runtime starts, ignored-flag warnings, options passed to the detector |
| `src/main.rs` | `DetectArgs` flags and help text |
| `docs/supported-models.md` | YOLO11 row plus the OmniParser detect-then-caption recipe |

## 2. Technical Decisions

### 2.1 Port the OmniParser topology as a fixed table

The checkpoint is one fixed topology, not a family. The layer table from mlx-vlm is encoded directly, and `Yolo11Config::validate` rejects any `ch` or `stride` that differs from `[256, 512, 512]` / `[8, 16, 32]`. A different depth or width cannot be expressed by this graph, so failing at load is correct instead of producing a plausible but wrong forward pass. A different `nc` and `reg_max` are supported, because only the head output width changes.

### 2.2 Load weights with an exact schema

`weights::Loader` takes every expected tensor once with its expected shape and reports any key the graph never consumed. This makes a weight for a parameter-free layer (concat, upsample) or an out-of-range layer index fail with the offending key. The alternative, loading by prefix and ignoring leftovers, would have let a checkpoint with an extra stage load and silently drop it.

### 2.3 Decode on the host

The model returns `(1, 4 + nc, N)` in MLX. NMS, score filtering, class offsets, and the inverse letterbox run on the host over the read-back `f32` buffer. The arrays are small after the candidate cap (30k candidates, 300 kept), and keeping the greedy loop in Rust makes the class-aware suppression directly testable against synthetic boxes. Upstream's sequential comparison (the kept box against the remaining list) is preserved.

### 2.4 Threshold defaults move to the family

`--threshold` was already optional at the CLI, but `load_detector` required a resolved value. Moving `Option<f32>` into the loader means the family default lives with the family: RT-DETRv2 0.3, PP-DocLayoutV3 0.3, RF-DETR 0.5, YOLO11 0.05. An explicit value is validated before `detector_info` runs, so a bad `--threshold` still fails before any model file is opened.

## 3. Verification

- 17 `yolo11` unit tests: letterbox geometry and centring, inverse mapping, class-aware NMS, IoU threshold, max-det and max-NMS caps, malformed predictions, config and routing.
- `cargo test --profile test-fast --lib vision::detection --features metal,accelerate`: 106 passed.
- `cargo test --profile test-fast --bin mlxcel commands::detect`: 8 passed (RT-DETRv2 ordering and masks unchanged).
- Smoke run against `axiom-of-choice/OmniParser-v2-icon-detect` at `--size 1280` on a synthetic 1000x600 screenshot: five `icon` boxes at the drawn rectangles, `--threshold 0.95` keeps two, `--iou 0` keeps one, native resolution keeps five.
- `cargo clippy --lib --tests -- -D warnings` and `cargo fmt --check` pass locally. CI's `cargo-clippy` and `CPU-only link` jobs failed in the mlxcel-core CMake build (`Could NOT find BLAS`), unrelated to this diff; `mergeStateStatus` was `UNSTABLE` and the repository allowed the merge.

## 4. Unverified

The mlx-vlm raw-output parity fixture and the end-to-end box-parity comparison were not produced: this host has no Python mlx-vlm installation. The detector was validated against the real checkpoint structurally and by output inspection, not numerically against `mlx_vlm.models.yolo11`.
