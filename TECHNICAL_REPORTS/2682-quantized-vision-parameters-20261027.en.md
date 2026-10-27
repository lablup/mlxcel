# Technical Report: PR #2682 — quantized vision parameters

**Date**: 2026-10-27
**Status**: Implementation complete; verification results below
**Risk Level**: Medium

## Executive Summary

Vision/VLM embedding tables and learned parameters now respect MLX quantization sidecars instead of interpreting packed integers as dense values. Every Group A/B/C site in #2676 is integrated into the real loading path, with malformed inputs rejected before MLX FFI calls.

## Problem Statement

Some checkpoints quantize position tables, class/query tokens, separators, latents, and text-extension embeddings. Existing loaders copied packed U32 arrays and inferred logical widths from packed dimensions, leading to incorrect embeddings or uncatchable C++ reshape/broadcast errors. Rejecting all quantized vision tensors would unnecessarily exclude valid vendor exports.

## Technical Decisions

Row-lookup tables retain `UnifiedEmbedding`; whole-array consumers dequantize once at construction through `load_dense_param`. Expected logical dimensions and stored packed/scales geometry determine group size, bit width, and affine/MXFP4/MXFP8/NVFP4 mode. Validation covers rank, leading dimensions, floating dense dtypes, packed and sidecar dtypes, bias consistency, checked arithmetic, supported backend modes, and final shapes. Available configuration metadata is threaded through constructors, including InternVL and GOT-OCR. Sidecars survive key remapping and selective weight filtering.

Dense arrays remain unchanged copies. Learned grids that are resampled become dense once at load. SAM3 block-float prompt embeddings use BF16 activation dtype rather than UINT8 scale-storage dtype. The real Moondream2 checkpoint stores `[1,729,1152]`, while existing fixtures use two-dimensional tables; both layouts are accepted and normalized to singleton-leading 3D. Synthetic tests alone initially missed that compatibility requirement, so the real-checkpoint sweep supplied the decisive evidence.

## Validation and Measurements

Measurements use Apple M5 Max 128 GiB, release `metal,accelerate`, baseline `ffea91111977751e409a8cb03ae1945ef372b327` and final integrated code `46b344b6` (main `b4cdf023`), `tests/fixtures/test_image.png`, and prompt `Describe the image.`. The fixed continuation corpus and top-eight logit traces are committed under `benchmarks/issue-2676/`. The image prompt runs at its native prefill width; the remaining 63 teacher-forced continuation tokens run together after that context.

| Checkpoint | Image prompt tokens | Positions | Top-1 disagreement | Decided disagreement, gap >= 2 | Reference-choice logit delta |
|---|---:|---:|---:|---:|---:|
| HunyuanOCR 4-bit | 282 | 64 | 0/64 | 0/16 | 0 |
| Gemma4 12B 4-bit | 275 | 64 | 0/64 | 0/28 | 0 |

All recorded trace data agree exactly. Perplexity is unchanged at 15.7281818418 and 14.6555180299. Greedy generation requested up to 64 tokens; HunyuanOCR produced 64 and Gemma4 ended after 9, with identical decoded outputs before and after. Thirteen checkpoints load and generate successfully: the six requested local equivalents, GLM-4.1V, InternVL3, LFM2-VL, Kimi-VL, Moondream2, Qwen2.5, and Gemma3. Gemma4 assistant is validated as a text MTP drafter alongside its target; standalone and offline image-MTP execution are unsupported by the existing CLI.

New tests cover each Group A loader, dense/quantized 1D/2D/3D parameters, whole-array add, slice/reshape, token concatenation, supported quantization modes, and malformed layouts. Temporarily restoring the old packed-as-dense behavior produced ten expected safe failures, covering all six Group A cases and the shared call patterns; the implementation was then restored. Full workspace/all-target clippy and formatting pass. The final integrated serial gate reports 14,193 passes, 455 ignored, and two unrelated failures: server read-only-field fixture coverage and Step3p7 text batching. Counts sum Cargo result summaries, including subprocess harnesses.

## Review Findings and Follow-up

Review corrected seed-based quantization-mode inference, missing InternVL/GOT metadata propagation, sidecar dtype validation, SAM3 activation dtype, dense integer acceptance, and Phi4 dimension overflow. The first un-serialized workspace run also showed GPU tests interfering through MLX's process-global default device; single-threaded reruns passed without changing numerical tolerances. Use `make verify-test`, which includes `--no-fail-fast -- --test-threads=1`.

The first serialized gate exposed an unchanged server-schema coverage failure (114 declared fields versus 113 expected). Concurrent PR #2681 fixed that count mismatch and was merged into this branch before final integration validation; its unchanged read-only-field fixture still needs synchronization and is the remaining server test failure. The unchanged Step3p7 text-only batching comparison also failed its 1e-5 bound; the same failure was reproduced in an isolated build of baseline `ffea9111` (0 passed, 1 failed). No numerical tolerance was changed.

## Change Summary

Centralized learned-parameter validation, all enumerated vision/VLM consumers and relevant sidecar remaps, focused regression suites, model-porting guidance, and reproducible parity evidence. No dependency or checkpoint conversion step is added. Related: #2676, #2459, #2659.
