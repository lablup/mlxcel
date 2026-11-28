# PR #3336: Round the Qwen3-VL vision tower at eager's BF16 boundaries

**Date**: 2026-11-24
**Status**: PR open (closes #3291; blocked by #3290, found in #3212; part of #932 and epic #566). Unit suites pass; verified on GB10 CUDA against eager MLX on a real Qwen3-VL checkpoint

## Executive Summary

The XLA Qwen3-VL vision graph ran every operator in f32, while eager MLX runs the BF16 checkpoint's vision tower on BF16 tensors. Over 24 blocks the two drifted apart, and on the fixture image the greedy output split at generated index 8, a near-tie that XLA resolved the other way. #3290 had already fixed preprocessing, so the remaining cause was rounding. This PR keeps the f32 carrier but rounds to BF16 at every point where eager stores a BF16 tensor, and keeps f32 accumulation inside GEMMs, norm reductions, and softmax statistics.

The rounding is integer round-to-nearest-even on the f32 bit pattern with a NaN guard, so the same graph lowers on Metal, which has no BF16 type. The operator numeric contract declares the new roundings, so the contract digest and the artifact digest change and cached Qwen3-VL vision artifacts recompile once. On the fixture the first greedy divergence moves from index 8 to index 27, which is an exact eager tie. The maintainer revised two acceptance criteria on #3291: chained block-23 drift is recorded rather than gated, and the f32 host-dequant control is no longer a target.

## 1. Problem Statement

### 1.1 Background

#3212 compared XLA and eager Qwen3-VL and found a divergence on a 256x256 solid image. At generated index 8, XLA had 6437=22.8980 over 4285=22.8781 (margin 0.020). Eager had 4285=22.875 over 6437=22.75 (margin 0.125, one BF16 step at that magnitude). Feeding eager's language model the XLA vision features gave an exact BF16 tie, so the language side was not the cause. #3290 fixed image preprocessing and cleared that part of the gap. What remained was the vision tower itself: `QWEN3_VL_VISION_PRECISION = F32` against eager's BF16.

### 1.2 Why f32 is not simply more accurate

Eager's result is the reference, and eager's result contains BF16 rounding at each stored tensor. A graph that skips those roundings is closer to the real-number answer and further from eager. Activations in this tower reach magnitudes above 2500 in the last blocks, where one BF16 step is 16, so a skipped rounding at an early block changes which values later blocks see. Bit-identical output was never the goal (summation order inside GEMMs differs by backend). The goal was to match eager's rounding boundaries so the remaining difference is summation order only.

### 1.3 Risk of leaving it

| Risk | Impact | Likelihood |
|---|---|---|
| XLA Qwen3-VL output diverges from eager at near-ties on ordinary images | Medium: user-visible token differences between backends | High on images with large activations |
| #932's qualification of the Qwen3-VL family cannot cite a vision-tower oracle that passes | High for #932 | Certain without this PR |

## 2. Change Summary

| Area | Change |
|---|---|
| `emitter/qwen3_vl_ops.rs` | `round_bf16`, `linear_bf16`, `layer_norm_bf16`, `exact_gelu_bf16`, `tanh_gelu_bf16`, `apply_vision_rope_bf16`, `attention_bf16`, `add_bf16`. |
| `emitter/qwen3_vl.rs`, `emitter/mod.rs`, `emitter/builder.rs` | Builds the tower from the new ops, takes `Precision` as a parameter, adds the integer and bitcast helpers `round_bf16` needs. `QWEN3_VL_VISION_PRECISION` becomes `qwen3_vl_vision_precision(device)`. |
| `qwen3_vl_runtime.rs` | Resolves the precision per device and passes it to both the emitter and `ProductionNumerics`, so graph and contract read one value. |
| `operator_numeric_contract/families.rs`, `family_tests.rs` | Declares BF16 output rounding on every Qwen3-VL operator except the attention scores. Pins updated. |
| `CHANGELOG.md` | Changed entry, including the one-time recompile. |
| Tests | `qwen3_vl_vision_precision` pinned per target (F32 on Metal, BF16 on CUDA; it returns BF16 whenever `precision_lowers_on(device, Bf16)` holds). The full vision graph is emitted at the Metal precision for the smallest and largest patch bucket and asserted to contain no `bf16` type. |

8 files, +603/-229. No tolerance changed (atol/rtol 0.08).

## 3. Technical Decisions

### 3.1 Rounding design, op by op

Each rule mirrors what the eager MLX CUDA path stores for this checkpoint, whose vision tower is unquantized BF16.

- **Input patches** are cast to BF16 once.
- **Patch embedding and every linear.** MLX computes `matmul` and then a separate `add` of the bias. The product is stored in BF16 (f32 accumulation inside the GEMM), then the bias add is stored again. The graph rounds twice: `linear_bf16` rounds the product, adds the broadcast bias, rounds again.
- **Interpolated position embeddings.** The four corner terms are summed in f32 in the order `(c0+c1)+(c2+c3)` and rounded once. The fixture diff against eager is exactly 0 at the input, patch embedding, and position embedding stages.
- **LayerNorm** follows MLX's CUDA kernel (`layer_norm.cu`): f32 mean and variance, the normalized value rounded, `w * n` rounded, then `+ b` rounded. Three roundings after the f32 statistics. The main merger normalizes then reshapes; the DeepStack mergers reshape then normalize, and the graph keeps each order.
- **RoPE.** q and k are widened to f32, rotated, and rounded once.
- **Attention** follows cuDNN flash SDPA, which eager selects for unmasked BF16 attention at head_dim 64. Scores and the row sum stay f32, the unnormalized probabilities are rounded to BF16 before the value GEMM, and the normalized context is rounded. This is why the attention scores are the one operator without a declared output rounding.
- **GELU** (tanh form in the blocks, exact form in the merger) evaluates in f32 and rounds once.
- **Residual adds** round once. **Main and DeepStack mergers** use the same linear, norm, and GELU ops.

### 3.2 Integer round-to-nearest-even with a NaN guard

The obvious implementation is `convert f32 to bf16` then back. That needs a BF16 type in the lowered module, and the Metal target has none (#612). `round_bf16` instead works on the bit pattern:

```rust
let bits = bitcast_convert(value, ui32);
let lsb = (bits >> 16) & 1;
let rounded = (bits + 0x7fff + lsb) & 0xffff_0000;
select(value != value, value, bitcast_convert(rounded, f32))
```

Adding `0x7fff` plus the lowest kept bit rounds to nearest, ties to even, and the mask drops the low 16 bits. The carry can run into the exponent, which is correct for values that round up to the next power of two. NaN needs a guard: CUDA's canonical NaN is `0x7fffffff`, and the integer carry would turn it into `0x80000000`, which is `-0.0`. The `select` on `value != value` passes NaN through unchanged.

Contractions take BF16 inputs where `precision_lowers_on(device, Bf16)` holds (CUDA) and the BF16-exact f32 carrier on Metal. Because the carrier holds only BF16-representable values, the products are the same. `qwen3_vl_vision_precision(device)` makes that choice, and a test asserts that the Metal-precision graph names no `bf16` type, so Metal compilability does not rest on construction alone.

| Option | Pros | Cons |
|---|---|---|
| Convert to `bf16` and back | One op, simplest | Needs a BF16 type, fails on Metal |
| Run the tower in BF16 end to end | Fewer ops | Loses f32 accumulation and statistics that eager keeps |
| **Chosen: f32 carrier with integer rounding** | Lowers on every target, accumulation stays f32 | More elementwise ops (XLA fuses them), a fourth copy of a rounding helper |

### 3.3 Contract and digest change

The operator list for Qwen3-VL now declares BF16 output rounding on each operator except the attention scores, and `ProductionNumerics` receives the same `Precision` the emitter used. Two things change as a result. The contract digest in the v3 manifest goes from `ae5bfb35...` to `bdd1632d...`. The artifact digest goes from `18e85256...` (`defac5fb`) to `4d3934a5...` (`775dbb9f`). The VMFB cache path derives from the emitted MLIR text, so the new graph also gets a new path. A stored artifact whose contract differs is recompiled before load (the #3242 mechanism), so the old f32 artifact is never reused. The cost is one extra compile per patch bucket on first use after upgrade.

### 3.4 Acceptance criteria revised by the maintainer

Two criteria on #3291 were revised on 2026-11-23, with the reasons recorded on the issue.

- **Chained vision drift to or below Qwen2.5-VL's level.** Met for final logits (0.198 against 0.85). Not met for chained block 23 (184 against 133). Block 10 amplifies one-step summation-order differences about 40x on both test images, and matching rounding boundaries cannot remove a difference that comes from the order of f32 sums inside a GEMM. The number is recorded, not gated.
- **Every same-input control passes.** The report-only block-23 `host_dequant_dense_f32` control compares against an f32 reference. Matching eager's BF16 rounding is expected to widen the distance to f32, so it is no longer a target. On shapes224, 6 values exceed tolerance, all BF16 cancellation at magnitude 2545 where one step is 16. The fused control against eager BF16 keeps the 2 failures already present on main.

No tolerance was relaxed.

## 4. Results

GB10 CUDA, `xla_qwen3_vl_deepstack_check`, atol/rtol 0.08 unchanged, before and after on the same fixtures.

| Measurement | Before | After |
|---|---|---|
| shapes224 final-logit max_abs | 0.205 | 0.198 |
| shapes224 chained block 23 (reference 2528) | 230 | 184 |
| shapes224 main merger | 1.82 | 1.00 |
| fixture chained block 23 (reference 2656) | 1020 | 1100 |
| fixture main merger | 1.52 | 5.16 |
| fixture same-input block 23 control (fused) | 3 failures | 0 failures |
| Qwen3-VL greedy vs eager (fixture, 64 tokens) | diverges at 8 | diverges at 27 |
| Qwen2.5-VL greedy vs eager | exact (EOS at 8) | exact (EOS at 8) |
| warm `iree_vision_seconds` (single run) | 1.85 s | 1.67 s |

Same-input controls of the production eager block are at one BF16 step or less in every block except fixture block 10, which is at two steps (report-only, was 0.338). The fixture's main merger got worse (1.52 to 5.16) because its input, chained block 23, got worse (1020 to 1100). The merger's own rounding is not the cause: across 8 of 8 image and merger pairs, each merger's drift moves in the same direction as the chained drift of the block feeding it.

Index 27 is an exact eager tie: tokens 576 and 2619 are both 30.375 (margin 0.0). XLA has 2619 over 576 by 0.021, inside one BF16 step (0.125) at that magnitude. That satisfies the criterion that a divergence sits only where the margin is at most one step.

## 5. Limits

- **Summation-order amplification.** GEMM summation order differs between cuBLAS and IREE's kernels. Block 10 amplifies that difference about 40x on both images. Chained drift at block 23 (184 against Qwen2.5-VL's 133) is therefore above the earlier target and will stay there until the accumulation order itself matches.
- **Per-tile running maximum is not reproduced.** The flash kernel keeps a running maximum per key tile. The graph uses the row maximum. The probabilities are rounded after subtracting a different offset in some tiles, which can shift a rounding by one step. The code documents this.
- **CUDA-kernel rounding on all targets.** LayerNorm and attention roundings follow MLX's CUDA kernels and cuDNN flash SDPA, and are applied on every target including Metal and the CPU targets. Eager on Metal or CPU may round differently in those ops. No Metal or CPU eager comparison was run.
- **BF16-exact carrier assumption.** The Metal f32 carrier forms the same products only while the vision weights are BF16-exact. A quantized vision tower host-dequantizes to values that are not BF16-exact, so this holds for the BF16 checkpoint only.
- **No same-input merger control.** Merger evidence is indirect (direction agreement with chained drift).
- **Single timing run.** The vision time change (1.85 s to 1.67 s) is one warm run, not a benchmark.
- **Fourth rounding helper.** `round_bf16` is the fourth emitter copy of a BF16 rounding helper. The integer form is required by Metal; the other three convert through `bf16`.

## 6. Learning Points

- Matching a BF16 reference means matching where it rounds, not rounding more or less. Running in higher precision moves away from the reference.
- Rounding to BF16 on the bit pattern (add `0x7fff` plus the kept low bit, mask the low half) is a portable replacement for a BF16 convert and needs an explicit NaN guard.
- When a gap shrinks to summation order, a gate that demands it close is measuring the wrong thing. Record it and gate on the outcome that matters (final logits, greedy margin).
- Pass one `Precision` value to both the emitter and the contract so a graph and its recorded numerics cannot disagree.

## Related Commits

| Hash | Type | Message |
|---|---|---|
| `65ac67e2` | update | round the Qwen3-VL vision tower at eager's BF16 boundaries |
| `7dfef804` | test | pin the Qwen3-VL vision precision per target and add changelog |

Related: #3291 (this change), #3290 (preprocessing), #3212 (found the divergence), #3211 and PR #3242 (contract binding), #932, #566.

## Verification

- `cargo test -p mlxcel-xla --lib qwen3_vl`: 17 passed. `operator_numeric_contract`: 19 passed. `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings` clean.
- `cargo check --features cuda,xla-iree --all-targets` and `cargo check --no-default-features --features xla-diagnostics --all-targets`.
- Deepstack check on shapes224 and on the fixture with `--token-ids` at indexes 8 and 27, before and after.
- `mlxcel generate -n 64 --temp 0`, eager against `MLXCEL_BACKEND=xla`, for qwen3-vl-2b-instruct-4bit and qwen2.5-vl-3b-instruct-4bit.
