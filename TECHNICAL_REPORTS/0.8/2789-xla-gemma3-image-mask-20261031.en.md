# PR #2789: Use the transformers Gemma 3 image mask on the OpenXLA path

**Date**: 2026-10-31 (generated before merge; the PR was an open draft at the time)
**Status**: In review
**Languages**: Rust
**Risk Level**: Medium (every Gemma 3 image request on `MLXCEL_BACKEND=xla` now prefills under a different attention mask, so its outputs change, and compiled bundles keyed on the old mask identity are rejected; the MLX path, token prefill and other OpenXLA families are unchanged)

## Executive Summary

#2773 moved the MLX Gemma 3 image prefill to the transformers mask: causal attention plus bidirectional attention inside each image block, intersected with the sliding window on sliding layers. The OpenXLA path kept the pre-#2773 mask, where every non-padding token sees every other non-padding token on every layer and no window band is applied. Since #2779 both backends build the same prompt ids for an image, so this mask was the remaining reason `MLXCEL_BACKEND=xla` and MLX gave different Gemma 3 image answers. Issue #2781 filed the residual that the #2773 and #2779 reports listed. This PR makes the host mask built by `prepare_gemma3_vlm_prefill` equal the MLX mask cell for cell, lets the emitter apply its existing sliding band to the caller's bias for Gemma 3, bumps the mask identity so stale bundles are rejected, and adds a final-position logits API so a tiny-model IREE test can compare OpenXLA against the MLX forward row by row. That test passes at a worst row error of 2.6e-6 (f32 on CUDA), and the real 4B checkpoint served over OpenXLA names the three shapes in the test image.

## 1. Problem Statement

### 1.1 What the OpenXLA path did

The Gemma 3 image prompt reaches IREE as a prepared embeddings prefill. `Gemma3IreeHostPreprocessor` (`src/multimodal/host_preprocessor.rs`) runs host image processing, and `prepare_gemma3_vlm_prefill` (`src/lib/mlxcel-xla/src/prepared_gemma3.rs`) assembles the post-scale embeddings and a `[1, 1, S, S]` additive bias. That bias was `0.0` iff both the query and the key were non-padding, otherwise `f32::MIN`, with `causal: false`. Its identity string was `gemma3-vlm-bidirectional-padding-f32-min-v1`.

In the graph, the embeddings-prefill arm of the emitter (`src/lib/mlxcel-xla/src/emitter/model.rs`) uses the caller's bias as `cmask` for global layers and builds `cmask_local` (the bias intersected with `i - j < W`) for sliding layers only when `embeddings_prefill_local_window` returns `Some`. For any config with `embeddings_prefill_uses_authoritative_mask` set, which is Gemma 3 VLM, it returned `None`, so the sliding layers also took the full bidirectional bias.

### 1.2 What the MLX path does after #2773

`vision::merge::attention_mask_for_image_runs` allows `(q, k)` iff `key_valid[k] && (k <= q || (block[q] > 0 && block[q] == block[k]))`, where an image block is a maximal run of `image_token_id` numbered from 1 and every other token has block 0. Query validity is not applied. On sliding layers, `gemma3::external_mask_with_window_band` additionally drops `q - k >= window`. The band is one-sided, so later members of a query's own image block stay visible.

### 1.3 Why it mattered

The mask is part of the training contract. A text token that reads later prompt tokens computes hidden states the checkpoint never produced, and this was the defect that made the MLX path describe `tests/fixtures/test_image_shapes.png` as "a white square-like shape" before #2773. Once #2779 made the prompts identical, the mask was the one remaining source of backend divergence for Gemma 3 image requests, and the OpenXLA path still carried the defect #2773 had fixed.

## 2. Technical Decisions

### 2.1 Host mask mirrors the MLX predicate, including its quirks

`prepare_gemma3_vlm_prefill` now computes a block id per position in the loop that already validates tokens: a counter is incremented when an image token follows a non-image token (or sits at position 0), and non-image positions keep 0. The bias cell is `0.0` iff `attention_mask[k] == 1 && (k <= q || same_block)`. This is the same predicate as `attention_mask_for_image_runs`, written on the host because the producer already works on host vectors.

The PR copies the MLX behavior where it is debatable rather than improving on it. Query validity is not applied, so a padding query row keeps whatever keys the predicate allows (its output is never read). A left-padded row at position 0 ends up with every key masked, which the finite `f32::MIN` keeps NaN-free. Copying exactly is what makes the cell-for-cell bit comparison in the root-crate test (2.4) possible; any "fix" here would have to land on both paths together.

### 2.2 The band stays in the graph

`embeddings_prefill_local_window` now returns `c.sliding_window` for every config, so the existing `PrefillInput::Embeddings` and `PrefillInput::DeepStack` arms band the caller's bias on sliding layers. Positions are sequential from 0, so buffer index equals position and the graph's `i - j < W` is the predicate MLX uses.

The rejected alternative was a second host-side mask for sliding layers. That would change the prepared-prefill payload schema and every consumer of it, while the graph already had the band code and only the gate needed to change. Configs that never set the authoritative flag already got the band, so this change only affects Gemma 3 VLM.

`embeddings_prefill_uses_authoritative_mask` stays, because it still selects the `f32::MIN` masked-value check (`model.rs`, the masked value for the caller's bias) and the Gemma 3 VLM gate in `iree.rs`. Its doc comment now says the caller supplies the authoritative base mask and the emitter still bands it. Out-of-band cells use the emitter's `neg_big` (-1e30) where MLX uses `f32::MIN`. Both give zero softmax weight; the PR does not claim bit equality for banded cells.

### 2.3 Identity bump, schema unchanged

`GEMMA3_VLM_MASK_MODE` becomes `gemma3-vlm-causal-image-blocks-sliding-band-f32-min-v2`. It is part of `Gemma3VlmCompatibilityContract::stable_identity`, so a bundle or compiled artifact built under the old behavior fails the identity check instead of being reused with the wrong mask semantics. The name records both halves of the change (host block mask, graph band). `GEMMA3_VLM_PREPARED_PREFILL_CONTRACT` is unchanged because the payload shape and dtype did not change.

### 2.4 A public logits entry point for the parity test

No API outside the `diagnostics` feature returned embeddings-prefill logits, and the parity test needs the graph the server actually runs. `XlaPreparedPrefillLogits` (`src/lib/mlxcel-xla/src/batch.rs`, exported under the `iree` feature) wraps `IreeRaggedLlama`, runs `PreparedIreePrefill::prepare_for_mode` with the engine's hidden size, capacity and position mode, and returns slot 0's last-position logits. Going through the ragged engine rather than a diagnostics-only path means the test exercises the production band code. The cost is a small public type that only tests use today.

### 2.5 Three layers of tests, each failing first on main

- `prepared_gemma3::tests::exports_causal_image_block_mask` spells out the full 7x7 matrix for `[pad, text, img, img, text, img, text]`, and `image_blocks_start_at_position_zero_and_text_only_prompts_stay_causal` covers a run at position 0 and a prompt with no image. On main: `query 1 key 2: left 0.0, right f32::MIN`.
- `emitter::model::gemma3_vlm_mask_tests::authoritative_embeddings_mask_keeps_the_sliding_window_band` asserts `Some(window)` with the flag set and `None` without a window. On main: `left: None, right: Some(4096)`.
- `src/loading/vlm_gemma_xla_mask_tests.rs` (root crate, `xla-backend`, no IREE) compares `prepare_gemma3_vlm_prefill` against `vision::merge::multimodal_attention_mask` bit for bit on the issue's prompt plus edge cases (run at 0, no image, two blocks split by one text token, padding inside the prompt). This pins the two implementations to each other, so a later change to either path fails here.
- `tests/xla_prepared_prefill.rs::local_task_gemma3_vlm_prefill_matches_mlx_mask` (ignored, real IREE) builds a tiny f32 Gemma 3 text checkpoint (2 layers, pattern 2, window 4) and a 12-token prompt with a 4-token image block at positions 3 to 6. Because the engine returns only final-position logits, each row is checked as the last row of its own prefix, with each side building its mask from that prefix. MLX runs `forward_with_embeddings_and_sequence_id` with the #2773 mask, image rows divided by `sqrt(hidden)` as `vision::merge` does. A `ctor` pins `MLX_ENABLE_TF32=0` so the MLX side is full f32. The device comes from `MLXCEL_XLA_DEVICE` (default `local-task`).

The failing-first runs separate the two fixes. With both reverted, row 0 matches and rows 1 to 11 are off by 0.06 to 4.8. With only the band reverted, rows 0 to 3 match and rows 4 to 11 fail (row 4 off by 1.26), starting at the first row past the 4-token window. So the band has its own failing evidence independent of the host mask.

## 3. Change Summary

12 files, +572/-45: two commits (dff56af0 fix, e3b03c28 docs) plus a merge of origin/main (196fae3f).

- `src/lib/mlxcel-xla/src/prepared_gemma3.rs`: per-position image block ids, the causal image-block mask, the v2 mask identity, the module doc, and the rewritten and new unit tests.
- `src/lib/mlxcel-xla/src/emitter/model.rs`: `embeddings_prefill_local_window` returns the window unconditionally; the gate test is rewritten.
- `src/lib/mlxcel-xla/src/emitter/config.rs`: the `embeddings_prefill_uses_authoritative_mask` doc comment.
- `src/lib/mlxcel-xla/src/batch.rs`, `src/lib/mlxcel-xla/src/lib.rs`: `XlaPreparedPrefillLogits` and its export.
- `src/lib/mlxcel-xla/src/iree.rs`: the compatibility-identity test expects the v2 mask mode.
- `src/loading/vlm_gemma.rs`, `src/loading/vlm_gemma_xla_mask_tests.rs` (new): the OpenXLA vs MLX mask test module.
- `src/loading/vlm_gemma_xla_tests.rs`: `expected_mask` in the `xla_gemma3_reference_check` boundary gate now builds the image-block mask from `image_token_index`, and the progress text names it.
- `src/multimodal/host_preprocessor.rs`: the `Gemma3IreeHostPreprocessor` doc says the mask is causal plus image blocks and the band is applied in the graph.
- `tests/xla_prepared_prefill.rs`: the tiny Gemma 3 fixture and the ignored IREE parity test.
- `docs/benchmarks.md`: the Gemma 3 mask paragraph says the OpenXLA path builds the same mask and bands it in the graph.

## 4. Results

- `cargo fmt --all -- --check` clean; mlxcel-xla lib, 365 tests (`--profile test-fast --features mlxcel-core/cuda`) pass; mlxcel-xla clippy `--all-targets -D warnings` clean.
- Root lib with `cuda,xla-backend`, `loading::` module: 383 passed.
- Release `cuda,xla-iree` clippy on `-p mlxcel-xla -p mlxcel --lib --test xla_prepared_prefill -D warnings` clean; release `cargo check --features cuda,xla-reference-diagnostics` clean (covers the boundary gate's expected-mask change).
- `xla_prepared_prefill --ignored` with `MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_PRECISION=f32`: the new test passes with worst row error 2.6e-6 against a 1e-3 tolerance.
- Real checkpoint: `mlxcel-server` (release `cuda,xla-iree`) on `gemma-3-4b-it-4bit` with `MLXCEL_BACKEND=xla MLXCEL_XLA_DEVICE=cuda MLXCEL_XLA_CONTEXT_CAPACITY=512` answered one image request for `test_image_shapes.png` (278 prompt tokens) with "A red square is located in the upper left corner. A blue circle is positioned in the upper right. A green triangle is placed below the circle."

## 5. Trade-offs and Residuals

- **The band is not exercised on a real checkpoint.** The 278-token smoke prompt is under the 1024 window, so the band is a no-op there. Only the tiny 4-token-window test covers it.
- **CUDA precision.** CUDA defaults to f16 contractions, which leave a 3e-3 gap at row 0 even before any masking difference, so the parity test passes only with `MLXCEL_XLA_PRECISION=f32` on CUDA. The default `local-task` device is the f32 path the test was designed for.
- **`local-task` could not run on this host.** IREE fails to create that device (`thread creation failed with 22`) on main and on this branch, so the two pre-existing tests in `xla_prepared_prefill` (which hardcode `local-task`) and the new test on its default device were not run here.
- **Pre-existing clippy warnings.** The root lib under `xla-backend` without `xla-iree` does not pass clippy because of `host_preprocessor_molmo_tests.rs` and `worker_failfast.rs`, which this PR does not touch.
- **Bundle invalidation.** The identity bump forces any cached Gemma 3 VLM bundle or compiled artifact to be rebuilt. That is intended; a deployment with a prebuilt bundle sees a rejection, not a silent mismatch.
- **Output change.** Gemma 3 image requests on the OpenXLA backend produce different text than earlier builds, now matching the MLX backend's mask.
- **New public surface.** `XlaPreparedPrefillLogits` is a public, non-diagnostics type with a single test caller and a fixed batch of 4.
- **Out of scope.** PaliGemma and Gemma3n have no OpenXLA Gemma 3 mask path. `spike/openxla/gemma3_vlm_mask_check.py` (the historical #869 spike, not in CI) still describes the old mask.

## 6. Learning Points

- When one backend changes semantics, its sibling backend's mirror is a residual by default. #2773 named the OpenXLA mirror as a follow-up but filed nothing; #2781 existed only because the next report listed it again. File the follow-up issue in the PR that creates it.
- A flag that once meant "the caller's mask is complete, skip the window" bundled two decisions: who owns the base mask and whether the graph bands it. Separating them (keep the flag for the value check and the gate, always band) was a one-line change because the band code was already shared.
- Pin two implementations of the same predicate to each other with a direct bit-for-bit test that needs no runtime. It is cheaper than an end-to-end parity run and fails on whichever side drifts.
- A final-position-only logits API can still give per-row parity by running every prefix. Each prefix builds its own mask, so the row-by-row result also localizes which mask component fails (here, row 4 is exactly the first row past the window).
- Mask-only parity on GPU needs the numeric noise floor below the tolerance. Pinning f32 on both sides (`MLX_ENABLE_TF32=0`, `MLXCEL_XLA_PRECISION=f32`) is what turned a 3e-3 baseline into 2.6e-6.
