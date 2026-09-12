# Technical Report: PR #1854 - Localize Qwen3-VL vision parity

**Date**: 2026-09-12
**Status**: Completed
**Languages**: Rust, Python, Markdown
**Risk Level**: Medium

## Executive Summary

PR #1854 continues the image-path parity investigation opened by #1738 after the Cohere Compass implementation in #1735 established token-exact text generation but a non-exact image-conditioned continuation. It adds a reproducible, stage-by-stage comparison between mlxcel and a pinned transformers oracle for `mlx-community/North-Micro-Vision-Instruct-4bit`, fixes premature precision loss in learned-position interpolation, and records the first remaining material divergence.

The corrected position embedding now agrees with the float32 oracle to RMSE `2.84e-07`, and vision RoPE is exact. The first residual beyond ordinary float32 rounding now appears in block 0 MLP at RMSE `0.00228`; the mismatch accumulates to post-merger RMSE `0.00810`, so this PR does not claim image-conditioned token exactness.

## 1. Problem Statement

PR #1735 narrowed the Cohere Compass mismatch to the shared 27-block Qwen3-VL vision tower. It had already ruled out TF32, both candidate vision activations, DeepStack order and injection depth, image-processor bounds, and quantization. The remaining evidence was only end-to-end: mlxcel generated `The image features...` where the transformers oracle generated `The image displays...`.

That evidence could not identify whether the first meaningful error arose in patch embedding, learned positions, vision RoPE, attention, MLP, DeepStack, or the merger. The existing production forward path also offered no safe way to extract intermediate tensors, and a synthetic-only comparison would not prove behavior on the published checkpoint.

The investigation therefore needed three properties:

1. A real-checkpoint oracle using the same processor and weights as the implementation under test.
2. Stage boundaries fine-grained enough to identify the first divergence rather than only its final consequence.
3. No production request-path overhead, decoder changes, video expansion, or assumption that Apple hardware generation explains the mismatch.

Without this localization, a future fix could alter the wrong subsystem, widen a tolerance over a semantic error, or repeat hypotheses that had already been falsified.

## 2. Technical Decisions

### 2.1 Observe the existing tower instead of creating a second implementation

The test-only `Qwen3VLVisionStage` observer reports input patches, patch embedding, learned positions, post-position hidden state, vision RoPE, attention/after-attention/MLP/output for every block, all three DeepStack tensors, and pre-/post-merger state. The ignored dump test serializes each observed tensor as little-endian float32 plus a JSON manifest.

The observer is gated by `#[cfg(any(test, feature = "test-utils"))]`. Production continues to call the original forward methods, so the diagnostic surface adds no callback dispatch, tensor copies, file I/O, or feature flag to normal inference.

### 2.2 Preserve float32 only across interpolation arithmetic

The previous path created bilinear weights in float32 and immediately cast them to the learned table's bf16/f16 dtype. Both the weights and the multiply/add therefore lost precision before the position result reached the hidden state.

```
Before: position table dtype + downcast weights -> low-precision interpolation -> residual add
After:  gathered rows cast to f32 + f32 weights -> f32 interpolation -> one cast to hidden dtype -> residual add
```

The fix promotes the four gathered embedding corners to float32, performs the weighted sum in float32, and casts the completed position embedding back to the patch hidden-state dtype immediately before addition. This matches transformers' arithmetic boundary without promoting all 27 vision blocks to float32.

### 2.3 Follow the pinned oracle's interpolation convention

The requested comparison with `torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)` was performed, but the pinned transformers Cohere Compass/Qwen3-VL implementation reports `bilinear` with `align_corners=True`. A regression test pins the true-corners result and separately proves that the half-pixel false-corners fixture differs by more than `2.0`, preventing a future refactor from silently adopting the wrong coordinate transform.

### 2.4 Make oracle provenance and dump inputs fail closed

The helper reuses the #1735 float32-copy approach rather than comparing unrelated converted weights. It loads the local checkpoint and `AutoProcessor`, reconstructs the transformers vision tower, and compares the same image and grid as the Rust manifest.

Review found that the first helper version printed a pinned revision but did not prove the installed package came from it. The final helper reads the installed distribution's `direct_url.json` and requires transformers commit `df04b012229d50d2b6dfba32c61c3057c3a40ea1`.

Security review then constrained model loading to local files with `trust_remote_code=False`, accepts only regular non-symlink files, requires stage paths to remain below the dump directory, validates positive shapes and float32 dtype, checks exact byte lengths, and caps each stage at 1 GiB. These checks keep an out-of-band diagnostic tool from becoming an arbitrary file-read or memory-exhaustion path.

## 3. Localization Results

The comparison used `tests/fixtures/test_image_shapes.png`, grid `[1, 28, 28]`, a float32 copy of `mlx-community/North-Micro-Vision-Instruct-4bit`, transformers `5.18.0.dev0` at the verified commit above, and torch `2.14.0`.

| Stage | Max absolute error | Mean absolute error | RMSE | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Processor input after Rust T,C to oracle C,T view | `2.98e-08` | `9.84e-10` | `5.42e-09` | Layout is semantically equivalent |
| Raw, unpermuted T,C comparison | `1.49` | `0.179` | `0.464` | Expected layout mismatch, not a patch bug |
| Patch embedding | `1.14e-05` | `1.25e-07` | `2.04e-07` | Float32 rounding scale |
| Learned position embedding | `5.72e-05` | `3.73e-08` | `2.84e-07` | Corrected interpolation boundary |
| Vision RoPE | `0` | `0` | `0` | Exact |
| Block 0 attention | `1.29e-05` | `2.37e-07` | `4.98e-07` | Float32 rounding scale |
| Block 0 MLP | `0.0311` | `0.00143` | `0.00228` | First remaining material divergence |
| DeepStack after block 8 | `0.0122` | `0.00148` | `0.00188` | Accumulated tower residual |
| DeepStack after block 16 | `0.0170` | `0.00154` | `0.00206` | Accumulated tower residual |
| DeepStack after block 24 | `0.0243` | `0.00151` | `0.00209` | Accumulated tower residual |
| Post-merger | `0.0961` | `0.00602` | `0.00810` | Final measured vision residual |

The input-layout comparison prevented a false patch-embedding fix. Rust stores raw patches in T,C order while transformers exposes a C,T view; applying the corresponding reshape and transpose reduces the input difference from RMSE `0.464` to `5.42e-09`. The existing MLX kernel weight arrangement is therefore retained.

The real greedy prompt remains non-exact after the interpolation fix: mlxcel begins `The image features...`, while the recorded oracle begins `The image displays...`. The report and supported-model documentation treat the block 0 MLP as the next localization boundary, not as a proven root cause.

## 4. Implementation Summary

| File | Responsibility |
| --- | --- |
| `src/vision/encoders/qwen3_vl.rs` | Float32 interpolation boundary and test-only stage observer |
| `src/vision/encoders/qwen3_vl_stage_observer_tests.rs` | Precision, interpolation convention, patch order, dtype, and stage-coverage regression tests |
| `src/vision/encoders/qwen3_vl_stage_dump_tests.rs` | Ignored real-checkpoint float32 stage dump with manifest |
| `scripts/tools/qwen3_vl_stage_oracle_compare.py` | Pinned transformers comparison, float32 checkpoint copy, provenance and input validation |
| `docs/supported-models.md` | Reproducible measurements and explicit non-exactness note |

No checkpoint tensors, generated dumps, model binaries, decoder changes, or video support are committed.

## 5. Review Findings

| Finding | Severity | Resolution |
| --- | --- | --- |
| Installed transformers provenance was asserted but not validated | High | Fixed in `55ccd5f7` by verifying `direct_url.json` commit metadata |
| Local manifest/checkpoint inputs could follow unsafe paths or request unbounded reads | Medium | Fixed in `07ac1462` with local-only loading, regular-file/path checks, exact sizes, and a 1 GiB per-stage cap |

No Critical or High findings remain. The production performance change is limited to float32 arithmetic for the learned-position interpolation itself; the completed embedding is cast back before the 27-block tower, avoiding persistent hidden-state promotion.

## 6. Validation

Local gates passed:

- `cargo test --profile test-fast --features metal,accelerate --lib vision::encoders::qwen3_vl::` — 8 passed, 1 ignored.
- The ignored `dump_north_micro_vision_stage_tensors` test ran against the real checkpoint and produced the manifest consumed by the oracle helper.
- `/private/tmp/qwen3vl-oracle-1738/bin/python scripts/tools/qwen3_vl_stage_oracle_compare.py compare --model models/North-Micro-Vision-Instruct-4bit-f32 --rust-dump /private/tmp/qwen3vl-stage-1738-final --model-dtype f32 --input-dtype f32` reproduced the measurements in Section 3 and verified oracle provenance.
- `cargo check --lib --tests --features metal,accelerate`.
- `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`.
- `cargo fmt --all -- --check`.
- Python bytecode compilation, cross-repository reference checking, and `git diff --check origin/main...HEAD`.

PR CI passed cargo-clippy, cargo-deny, cargo-fmt, OpenXLA feature compilation, crate-version, cross-repository-reference, kernel-dtype-key, license-header, WebUI-contract, llama-compatibility, and CLA checks. Platform-specific irrelevant jobs were skipped.

## 7. Change Summary

Implementation diff before these reports: 5 files changed, 1,042 insertions, and 5 deletions.

| Commit | Purpose |
| --- | --- |
| `5d5c9ef9` | Localize Qwen3-VL stages and fix position interpolation precision |
| `55ccd5f7` | Validate transformers oracle provenance |
| `07ac1462` | Harden local oracle inputs |

Issue #1738 is linked through `Closes #1738`. The PR is intentionally left open for the wave runner to merge.

## 8. Follow-up Actions and Learning Points

The next parity investigation should begin inside block 0 MLP and compare norm output, first projection, tanh-GELU, gate/product, and second projection before following the residual through later blocks. Decoder and video changes remain out of scope until that first MLP residual is explained.

Two reusable lessons follow from this work:

- Compare tensor semantics, not raw storage order. The T,C/C,T input result would otherwise have produced a plausible but incorrect patch-weight change.
- Pin and verify the oracle as part of the measurement. A version string or printed constant is not provenance, and #1769 already showed why unexplained failures should not be attributed to M1/M5 hardware generation when a shared software pin can produce byte-identical behavior.

Related context: [`1735-cohere-compass-vlm-20260910.en.md`](1735-cohere-compass-vlm-20260910.en.md), especially its oracle construction and image-path findings.
