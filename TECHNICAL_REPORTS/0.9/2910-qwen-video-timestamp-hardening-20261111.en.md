# PR #2910: Qwen3-VL video timestamp hardening

**Date**: 2026-11-11
**Issue**: #2402

## Summary

Reviewed temporal sampling metadata, timestamp expansion, mixed image/video ordering, MRoPE group positioning, family gating, and the real CLI/server preparation path. Qwen3-VL, Qwen3-VL-MoE, and Qwen3.5 dense/MoE VLMs use timestamps; Qwen2/2.5-VL retain the flat layout.

## Findings and corrections

- Avoid multiplication overflow and a potentially patch-sized padding allocation in timestamp calculation: compare group counts and compute first/last times from unpadded chunks. Reject unordered indices and non-finite computed times. Add boundary regressions, including `usize::MAX` and subnormal fps.
- Reject non-finite or negative externally supplied group seconds before tokenization.
- Add the mlx-vlm derivation notice and correct the scope documentation. Contrary to the original PR note, native Qwen video does not enter the CLI's image-frame fallback; `generate --video` already delegates to the same server preparation path.
- No new URL resolution, process execution, shared mutable state, or unbounded per-request cache is introduced. Existing video frame limits remain in effect. The timestamp helper now uses O(frame count) work without an extra padded copy.

## Real-checkpoint evidence

Apple M1 Ultra, 128 GiB, macOS 27.2, Metal+Accelerate release builds. Baseline is main `261080ba`; the after arm includes the original PR, main through `1b63a947`, and the hardening changes. Both use the same local `mlx-community/Qwen3-VL-4B-Instruct-4bit` checkpoint and greedy decoding, `--seed 1 -n 96 -p "Describe the motion."`.

The fixture is 40 RGB frames, 224×224 at 10 fps: black background and one 32×32 white square at `(12 + 4*i, 96)` in frame `i`, encoded with ffmpeg/libx264 to yuv420p. MP4 SHA-256: `ab433dbcf0ab530dbc9e0fe9b45b6bf46d67848bd8a66037f829cbfb56a240b3`. Default sampling is 2 fps. The fixture is generated locally, not committed as a binary artifact.

**Before (96-token cap):**

> The image depicts a **motion blur effect**.
>
> Here’s a breakdown of the motion:
>
> - **Subject**: A white square or rectangle is shown in motion.
> - **Effect**: The square is blurred horizontally, suggesting it is moving from left to right across the frame.
> - **Direction**: The blur trails to the right, indicating the object is moving rightward.
> - **Style**: The blur is a radial or linear blur, commonly used in animation or photography to represent speed or

**After (EOS at 24 tokens):**

> A white square moves from the left side of the screen to the center, then to the right side of the screen.

A second-family smoke used `llama-3.2-11b-vision-instruct-4bit`, the same clip, `--video-max-frames 2`, and `-n 48 --seed 1`. Both arms produced identical text without a crash, confirming the frame fallback remains unchanged; both hallucinated a pen/pencil and hand, so this is regression evidence, not a quality pass for that model.

## Validation and limits

The focused Metal-enabled test-fast suite passed 126 tests (13 ignored) across `multimodal::qwen_vl`, `multimodal::vlm_runtime`, and `multimodal::video`, including the new boundary tests. Real checkpoint output is correctness evidence only: no throughput claim is made on a shared machine. Qwen3-VL-MoE and Qwen3.5 VLM checkpoints were not run; family flags and prompt arithmetic are covered structurally and by tests. Full workspace tests were not run.

The ffmpeg-enabled `multimodal::video` run passed 53 tests with `--include-ignored --skip bench_single_pass_768_frames --test-threads=1`; the skipped test is an explicitly non-gating timing benchmark. Metal+Accelerate Clippy (`--profile test-fast -p mlxcel --lib --tests -- -D warnings`), formatting, and diff whitespace checks passed. Existing native C++ unused-capture warnings remain.
