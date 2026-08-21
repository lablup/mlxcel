# Technical Report: PR #916 - Molmo2 indexed attention pooling on OpenXLA

## Executive Summary

PR #916 (issue #871) brings the Molmo2 vision path to StableHLO/IREE: linear patch embedding, selected-layer concatenation, indexed attention pooling with signed sentinels, and a SwiGLU projector merged additively at scanned image-patch positions.

The implementation had been blocked for weeks on a reported numeric divergence at ViT layer 18. That blocker did not exist. It was an artifact of the MLX reference running in TF32 while the IREE side ran genuine F32, so the gate was comparing against a less accurate oracle. Correcting the oracle exposed a second, real problem: the gate had only ever been run on a 224x224 solid-color image, and every ordinary photograph failed on a tolerance that was wrong for the tensor it was applied to.

## 1. Problem Statement

Three separate things were wrong, discovered in order, each hidden behind the previous one.

**The recorded blocker was measurement error.** Earlier revisions reported a hard failure at selected layer 18, row 513, with `max_abs` around `0.272`, and proposed investigating whether attention, LayerNorm, or the MLP produced it.

**The passing gate proved almost nothing.** Once corrected, the gate passed at `max_abs = 0.049804688` against a limit of `0.05`, a margin of 0.4 percent, on the smallest and least textured input available. Its `1x1` crop tiling never reaches the high-resolution tiling, pooling offsets, or multi-crop merge.

**The server could not serve images at any default.** The OpenXLA context capacity defaults to 256 tokens. One Molmo2 image expands to between 424 and 1834. Worse, admission rejected the request only after the vision tower had already run.

## 2. Technical Decisions

### 2.1 Correct the oracle, do not touch the thresholds

MLX defaults `MLX_ENABLE_TF32` to 1, so `cublas_gemm.cpp` selects `CUBLAS_COMPUTE_32F_FAST_TF32`, a 10-bit mantissa compute type, for f32 contractions. `src/lib/mlxcel-xla/README.md` already declares that mode invalid as F32 reference evidence.

A controlled A/B changing only that variable:

| stage | `TF32=1` | `TF32=0` | ratio |
| --- | --- | --- | --- |
| `vit.patch_embedding` | 0.0003978014 | 0.00000059604645 | 667x |
| `vit.block.0` | 0.002866745 | 0.000008821487 | 325x |
| `vit.probe.18.row.513.input` | 0.27172852, FAIL | 0.00012207031 | 2226x |

The control reproduces the historically recorded value and stops in the same place, which is what makes this an attribution rather than a coincidence. No production attention or LayerNorm change was needed. The divergence also sits at the layer-18 input, meaning it accumulated through earlier blocks, contradicting the layer-local cause the earlier revisions hypothesized.

### 2.2 Make the tolerance scale-aware, and prove it did not go blunt

Four generated photographs covering `2x2`, `4x1`, `1x4` and `2x3` tilings all failed, every one at `projector.output_all` and nowhere earlier. That tensor carries values around 2e4, where one f32 ULP is already about 0.004, so a fixed `0.05` asked a 2000-term dot product to agree within tens of ULPs regardless of accumulation order. All four agreed to within 0.001 percent relatively and put 4 to 12 elements out of roughly 2e6 past the bound.

The contract became `|a - b| <= atol + rtol * |b|` with `atol` unchanged at `0.05` and `rtol` at `1e-5`, so the absolute term still floors the comparison near zero.

Widening a bound is exactly the change that can silently destroy a gate, so blunting is measured rather than asserted:

| case | violation ratio | verdict |
| --- | --- | --- |
| swapped selected layers | 1158x | rejected |
| 2 percent error on a large value | 1693x | rejected |
| 0.06 against 0.0 | 1.2x | rejected |
| worst real measurement | 0.41x | admitted |

Real passes and deliberate defects are separated by more than 2800x.

### 2.3 Reject an unusable capacity at startup, do not silently raise it

Raising the default was implemented first and then reverted on measurement. Capacity is the length every decode step attends over:

| capacity | decode | relative |
| --- | --- | --- |
| 256 | 3.18 tok/s | baseline |
| 1024 | 2.17 tok/s | 1.47x slower |
| 2048 | 1.41 tok/s | 2.26x slower |

No single value is both safe and cheap: covering the worst-case image would slow every text-only request on a VLM checkpoint by more than 2x, and it would still be a silent tax nobody asked for. `xla_image_context_floor` instead derives the worst case from config alone, walking every tiling with `rows * cols <= max_crops` rather than assuming the squarest or largest, because a column token per pooled row makes tall tilings the maximum. `ensure_xla_image_context_capacity` then refuses to start and names the requirement. An operator-pinned capacity passes through untouched, because text-only serving from a VLM checkpoint is a real workload.

### 2.4 Exit instead of parking on a startup failure

The OpenXLA worker logged and returned on any startup failure, leaving `loaded` false with nothing observing the dead thread, so the server stayed up answering nothing. `exit_on_worker_startup_failure` ends the process, extending the fail-fast posture `worker_failfast.rs` already documents for panics. It exits rather than aborts, because a misconfiguration is not a broken invariant.

## 3. Change Summary

| Area | Change |
| --- | --- |
| `src/lib/mlxcel-xla/src/emitter/molmo2_*` | Molmo2 vision graph emission: patch embedding, position policy, selected layers, indexed pooling, SwiGLU projector |
| `src/lib/mlxcel-xla/src/molmo2*.rs` | Runtime, artifact identity binding, geometry and cardinality preflight |
| `src/multimodal/molmo2_xla_preprocessor.rs` | Host preprocessing, prompt expansion, scatter-add at `image_patch_id` positions |
| `src/multimodal/host_preprocessor.rs` | `xla_image_context_floor`, `ensure_xla_image_context_capacity`, family-neutral error text |
| `src/worker_failfast.rs`, `src/server/model_worker.rs` | Startup failures end the process instead of parking it |
| `tests/molmo2_xla_vision_parity.rs` | Scale-aware tolerance, outlier-vs-drift reporting, blunting checks, tiling documentation |

## 4. Review Findings

The significant findings came from re-verification rather than a reviewer.

Rebasing onto the updated `main` and rechecking across feature combinations caught a real defect: the new synthetic tests called `assert_within`, which is gated on the diagnostics features along with its progress reporting and the `std::io` import behind that, so a `cuda,xla-iree` build failed to compile the test target. Earlier verification had used only `xla-diagnostics` and could not have seen it. The fix extracts the admission predicate without the reporting, which also means the tolerance shape is now verified in every build rather than only diagnostics ones.

One misattribution is worth recording: the re-export gate in `src/models/mod.rs` was initially described, in a commit message and an issue body, as fixing a break on `main`. It was not. `main` has no such re-export and the function is private there; both came from this branch. The commit was rewritten and the issue corrected.

## 5. Validation

Vision parity, all five tilings, GB10 with `MLX_ENABLE_TF32=0` and `MLX_CUDA_ARCHITECTURES=121`:

| image | tiling | crops | prompt tokens | tolerance used |
| --- | --- | --- | --- | --- |
| fixture 224x224 | 1x1 | 2 | 424 | 29.2 percent |
| 756x756 | 2x2 | 5 | 970 | 40.6 percent |
| 378x1512 | 4x1 | 5 | 1024 | 35.1 percent |
| 1512x378 | 1x4 | 5 | 984 | 36.1 percent |
| 1024x768 | 2x3 | 7 | 1348 | 31.2 percent |

Server path on `MLXCEL_BACKEND=xla mlxcel-server`: text-only fallback, single image, mixed continuous batch of two image and two text requests, client cancellation with the slot freed 16ms after the abort against 313.8s for the same request run to completion, slot reuse, and `/metrics` populated. The image answers are content-dependent rather than merely well-formed, which is the part that matters: the fixture is a solid RGB(255, 100, 50) square and the model describes it as a solid orange rectangle.

Greedy output is identical between the MLX and OpenXLA backends on four prompts, matching on both text and `completion_tokens`. This compares detokenized text plus token counts rather than token ids, because no CLI path emits ids, so it is not an id-level proof.

## 6. Related Work

Issue #871 is closed by this PR. Three follow-ups were filed rather than folded in: #1270 for the CI coverage gap that let an XLA compile break reach `main`, #1271 for capacity bucketing so text and image requests need not share one static graph shape, and #1272 for deriving worst-case image floors for the remaining qualified vision families, since only Molmo2 has one today and the others fall through the guard silently.

The recurring lesson across all three problems is the same. Each was hidden behind evidence that looked like a pass or looked like a specific failure, and each was only exposed by widening what was measured: the oracle's own precision mode, the range of inputs, and the set of feature combinations.
