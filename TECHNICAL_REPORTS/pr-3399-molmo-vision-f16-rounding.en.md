# PR #3399: Round the Molmo v1 vision tower at eager's F16 boundaries

**Date**: 2026-11-24
**Status**: PR open (closes #3390; part of #932 and epic #566). Unit suites pass; verified on GB10 CUDA against eager MLX on a real Molmo-7B checkpoint

## Executive Summary

The XLA Molmo v1 vision graph ran the CLIP tower in f32, while eager MLX stores an F16 tensor after every operator for the F16 checkpoint. With the reference harnesses repaired by #3365, `xla_molmo_reference_check` reached its vision stages for the first time since #2837 and failed 2 of 14 stages, identically on IREE `local-task` and `cuda`: `vision.selected_layer_21` (max_abs 2.80) and `vision.selected_layer_14` (max_abs 2.69), always the same channels in the duplicate crops. This PR rounds the graph to F16 at every point where eager materialises an F16 tensor, while keeping f32 accumulation inside contractions, softmax, and norm statistics. The gate passes 14/14 on both devices with the `VISION` bound unchanged.

## 1. Problem Statement

### 1.1 Background

#3365 repaired the double-hash bug in the reference harnesses (#2837), so the Molmo gate reached its vision stages. Two stages failed with 4 and 2 elements over the `VISION` bound. Molmo was qualified 14/14 in #870, so the question was regression versus numeric contract gap.

### 1.2 Not a CUDA regression

Building the gate at the #870 merge commit (`3ba57b30`; no digest patch needed, the double hash came in with #2837) shows `vision.selected_layer_21` already failing on this GB10 host (4 elements, max_abs 1.56). #870 was qualified with eager on ROCm (`gfx1151`), so the host difference was always there and only became visible once the harness reached the vision stages. Intermediate commits were not bisected further once the base itself failed.

### 1.3 Root cause: a rounding-boundary gap

The failing elements sit in channel 650, the tower's outlier channel (|x| up to about 200), plus channel 363 in layer 21, at rows 68/117/309 and their duplicate-crop rows. The outlier channel dominates each LayerNorm's variance, so an f32-versus-F16 difference in it propagates through every block. The divergence builds up across blocks rather than coming from one output rounding (about 90 F16 steps at that magnitude). Eager's result is the reference, and eager's result contains F16 rounding at each stored tensor, so a graph that skips those roundings is closer to the real-number answer and further from eager.

## 2. Change Summary

| Area | Change |
|---|---|
| `emitter/molmo_vision_f16.rs` (new) | `round_f16`, `linear_f16`, `layer_norm_f16`, `quick_gelu_f16` and 4 unit tests. |
| `emitter/molmo_vision.rs`, `emitter/mod.rs` | Builds the tower from the new ops; dequantized weight rounded; `MOLMO_VISION_PRECISION` stays F32 (contractions accumulate in f32). |
| `operator_numeric_contract/families.rs`, `family_tests.rs` | Declares f32-to-F16 output rounding on the operators that round; attention scores/softmax and the pool attention path stay f32. Molmo2 unchanged. |
| `src/multimodal/host_preprocessor_molmo_diagnostics_support.rs` | `compare_stage` now lists the first failing elements (capped at 16) so a failure can be traced to a row and channel. |
| `numeric_ops.rs`, `docs/gelu-call-sites.md` | Doc updates: Molmo uses `quick_gelu_f16`, the shared helper remains for LLaVA and Qwen2-VL. |
| `CHANGELOG.md` | Changed entry, including the one-time recompile. |

11 files, +279/-60. No tolerance changed; `VISION` stays at atol 0.25, rtol 0.05.

## 3. Technical Decisions

### 3.1 Rounding design, op by op

Each rule mirrors what eager MLX stores for this checkpoint, whose tower is quantized with F16 activations.

- **Dequantized weights** are rounded to F16, because eager's quantized matmul dequantizes into the F16 activation type.
- **Linear products and bias adds** round twice: the product, then `product + bias` (matching eager's `quantized_matmul` then `add`).
- **Pixels, patch embedding and the position add** round once.
- **LayerNorm** follows MLX's CUDA kernel: f32 statistics, the normalized value rounded to F16, then `w*n+b` rounded once (the kernel's half FMA). Rounding the product and the sum separately measured worse (layer 21 max_abs 2.81 against 2.50).
- **quick_gelu** evaluates in f32 with the F16 coefficient (`1.7021484375`) and three roundings, matching `gelu_sigmoid`'s per-op stores.
- **Attention** runs in f32 and casts the context back to F16 once; scores and softmax stay f32.
- **Both residual adds** round once.

Ablations on `local-task`: dropping any one of the linear, attention-context, residual, quick_gelu or pre-tower roundings fails the gate again (2 to 12 elements per stage). The dequant rounding was ablated against the two-rounding LayerNorm and also fails without it.

### 3.2 `round_f16` as a narrowing convert pair

Unlike the Qwen3-VL BF16 case (#3336), which had to round on the integer bit pattern because Metal has no BF16 type, F16 is representable on every target. `round_f16` is a stablehlo f32-to-f16-to-f32 narrowing pair, which rounds to nearest, ties to even, and matches MLX's overflow-to-inf behaviour. No NaN guard is needed; the convert pair preserves NaN payloads through the narrowing.

### 3.3 Contract and digest change

`molmo_vision_operators` now declares F16 output rounding on the rounding operators while exempting the four scores/softmax ids and `pool.attention-context`. Eager casts the pool attention context back to F16 before `wo`, but the graph keeps the whole pool path in f32; every observable stage passes with headroom (vision.projector max_abs 0.1875 local-task / 0.094 cuda against the envelope on refs up to about 200), so this stays a contract-declared choice. The contract flows into `production_artifact_contract`, so the Molmo manifest digest changes and cached Molmo vision artifacts recompile once. Molmo2's emitter and contract are untouched.

## 4. Results

GB10, `xla_molmo_reference_check`, pinned revision `5c04b3a4`, `test_image.png`, `VISION` bound unchanged, before and after.

| stage max_abs (failures) | main local-task | fix local-task | fix cuda |
|---|---|---|---|
| vision.patch_embedding | 1.34e-3 | 4.88e-4 | 4.88e-4 |
| vision.selected_layer_21 | 2.80 (8) | 2.50 (0) | 1.81 (0) |
| vision.selected_layer_14 | 2.69 (4) | 2.47 (0) | 1.78 (0) |
| vision.projector | 0.334 | 0.188 | 0.094 |
| prepared.sparse_add_visual_rows | 0.334 | 0.188 | 0.094 |
| decoder.prefill_logits | 2.07e-2 | 1.01e-2 | 1.10e-2 |

14/14 on both devices. After the fix the worst selected-layer element is 0.12 inside the bound on `local-task` (it was 0.69 over); the projector went from 0.004 inside the bound to 0.22 inside. Molmo CLI greedy through the XLA backend reports the same 16 token ids as eager.

## 5. Limits

- **Pool path stays f32 in the graph.** Eager materialises F16 tensors after the pooling 2D attention path (the pooling query mean, the pool context cast before `wo`, the pad-embed adds), but the graph keeps those f32. This is the contract-declared choice above, not reproduced rounding.
- **CUDA-kernel rounding on all targets.** LayerNorm and quick_gelu roundings follow MLX's CUDA kernels and are applied on every target including Metal and the CPU targets.
- **F16 checkpoint only.** The rounding points are chosen for the F16 checkpoint; an F32 or BF16 Molmo tower would need its own boundaries.
- **Summation order not matched.** IREE's contraction order differs from eager's; the fix matches rounding boundaries only.

## 6. Learning Points

- Same class as #3336: matching a lower-precision reference means matching where it rounds. Skipping the roundings moves away from the reference, not toward it.
- When a family's checkpoint is quantized with low-precision activations, the graph's numeric contract must name the activation dtype, not just the contraction precision.
- The failing channels (the outlier channel that dominates LayerNorm variance) identified the class before any fix: outlier-heavy channels amplify a precision gap through norms.
- A reference harness repair (#3365) can surface failures in stages that were silently skipped for months; treat the harness reachability fix as part of qualifying a family.

## Related Commits

| Hash | Type | Message |
|---|---|---|
| `db82c17` | fix | round the Molmo v1 vision tower at eager's F16 boundaries |
| `a246b06` | docs | correct the Molmo pool-path and quick_gelu user comments |
| `cdc649e` | docs | record the Molmo quick_gelu_f16 call site in the gelu inventory |

Related: #3390 (this change), #3365 (harness repair), #2837 (double hash), #870 (original qualification), #3336 (same class, Qwen3-VL BF16), #932, #566.

## Verification

- `cargo test -p mlxcel-xla --lib -- --test-threads=1`: 540 passed, 0 failed, 21 ignored. `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings` clean. `cargo fmt --check -p mlxcel-xla` clean.
- Gate 14/14 on `local-task` and on `cuda` (gpu-lock, bounded waits), independently re-run by the implementation reviewer.
- Molmo CLI greedy parity: identical 16 token ids, eager vs XLA.
