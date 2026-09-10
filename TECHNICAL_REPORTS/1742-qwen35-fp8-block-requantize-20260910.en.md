# Technical Report: PR #1742 - feat(qwen3_5): load vendor fine-grained FP8 checkpoints

**Date**: 2026-09-10
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed with one acceptance criterion deliberately unverified (the bf16 reference checkpoint was not fetched; the `metal,accelerate` acceptance command is not runnable on this host and was not run)
**Languages**: Rust
**Risk Level**: Medium (new load-time conversion on the Qwen3.5 paths; no behavior change for checkpoints that carry no `weight_scale_inv` sidecar)

---

## Executive Summary

The vendor FP8 releases of the Qwen3.5 family (`Qwen/Qwen3.8-27B-FP8` and siblings) store every converted projection as raw `E4M3` bytes in `<name>.weight` plus a `bfloat16` inverse scale per 128x128 block in `<name>.weight_scale_inv`. The issue that motivated this work assumed mlxcel would fail on the unexpected sidecar. It does not. MLX's safetensors reader maps `F8_E4M3` to `uint8` (`dtype_from_safetensor_str` in `mlx/io/safetensors.cpp`), so the bytes arrive intact, `UnifiedLinear` sees no `.scales` key and takes its dense path, and the model loads in a third of a second and generates. It is simply wrong by a per-block factor on every projection.

That is the whole reason this is a correctness fix rather than a capability gap, and it is what the fix has to be evidenced against: a load that succeeds proves nothing here, because a load already succeeded.

`models::fp8_block` reconstructs each pair on device and requantizes it to MLX-native `mxfp8` (group size 32, 8 bits). On the released 27B checkpoint the conversion adds about 30 seconds to load, holds peak allocator memory at 1.04x the final resident size, and produces coherent output.

---

## 1. What the checkpoint actually holds

Read directly from `models/mlx/qwen3.8-27b-fp8` rather than from the issue text:

```
config.json quantization_config:
  {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
   "weight_block_size": [128, 128], "modules_to_not_convert": [... 882 entries ...]}
```

| Property | Measured |
|---|---|
| Tensors in the index | 1606 |
| `F8_E4M3` tensors | 407 |
| `*.weight_scale_inv` sidecars | 407 |
| `F8_E4M3` tensors with no sidecar | 0 |
| Sidecar in the same shard as its weight | 407 / 407 |
| Sidecar dtype | `BF16`, all of them |
| Sidecars whose shape is not exactly `ceil(R/128) x ceil(C/128)` | 0 |
| Weights whose width is not a multiple of 32 | 0 |
| Weights not block-aligned on both axes | 0 |

The last row matters for testing. Every real tensor in this checkpoint is an exact multiple of 128 on both axes, so the real checkpoint never exercises the pad-and-slice branch. The unit test therefore uses a 130x160 tensor, which is aligned on neither.

Converted tensors are the ten projections per decoder layer (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`) plus the seven in the bundled `mtp.*` head. Everything else stays `BF16`: `embed_tokens`, `lm_head`, all norms, `conv1d`, `A_log`, `dt_bias`, the low-rank `linear_attn.in_proj_a` / `in_proj_b`, and the entire 27-block vision tower. The 882-entry `modules_to_not_convert` list is informational: only a tensor that actually has a sidecar is converted, so the list never has to be parsed.

### 1.1 The scale direction is the trap

`weight_scale_inv` names an inverse, which invites a divide. It is a multiply. Checked in numpy on three real tensors, independently of the Rust path:

| Tensor | `decode` alone | `decode * scale_inv` | `decode / scale_inv` |
|---|---|---|---|
| `layers.3.self_attn.q_proj` `[12288, 5120]` | max 448, std 92.7 | max 0.320, std 0.0172 | max 4.08e6 |
| `layers.1.mlp.down_proj` `[5120, 17408]` | max 448, std 76.8 | max 0.441, std 0.0108 | max 5.31e6 |
| `layers.1.linear_attn.in_proj_qkv` `[10240, 5120]` | max 448, std 90.9 | max 0.333, std 0.0154 | max 3.82e6 |

The multiply lands on a normal transformer projection distribution; the divide is off by seven orders of magnitude. Note also that the raw decode maxes out at exactly 448 in every tensor, which is the E4M3 maximum: the publisher chose each block scale to fill the E4M3 range, which is what makes the block scale load-bearing rather than cosmetic.

---

## 2. The conversion

`requantize_block_fp8_weights(weights, block)` walks the sidecar keys in sorted order and, per pair:

```
indices  = astype(raw_uint8, uint32)
decoded  = take(lut_256_f32, indices, axis=0)          # E4M3 decode, on device
padded   = pad(decoded, rows -> rb*128, cols -> cb*128, 0.0)
blocked  = reshape(padded, [rb, 128, cb, 128])
scaled   = blocked * reshape(astype(scale_inv, f32), [rb, 1, cb, 1])
restored = slice(reshape(scaled, [rb*128, cb*128]), [0,0], [R, C])
(q, s)   = quantize(restored, group_size=32, bits=8, mode="mxfp8")
```

`<name>.weight` becomes the packed `uint32` plane (`[R, C/4]`), `<name>.scales` is added (`uint8`, `[R, C/32]`), and `<name>.weight_scale_inv` is dropped. No `.biases` plane is produced, and the code refuses to continue if MLX ever returns one, because `mxfp8` is a block-float mode and an affine zero-point there would mean the mode was misrouted.

Three points are worth stating explicitly.

**The decode runs on device.** `sanitize.rs` already had a host-side `f8_e4m3_to_f32`, and the existing `F8_E4M3` branch in `tensor_view_to_array` uses it in a per-byte loop. At 26 GB of E4M3 bytes that loop would walk the whole checkpoint through the CPU and build an f32 `Vec` four times its size. Lifting the same function into a 256-entry `f32` table and gathering through it with `take` keeps the decode on the GPU and keeps the host copy at 1 KB.

**f32, not bf16.** The issue's pseudocode decodes to bf16. f32 is used instead: E4M3 and bf16 scales are both exact in f32, so the product is the exactly-rounded true weight, and `mxfp8`'s own group-maximum search then sees the true maximum rather than a bf16-rounded one. The cost is a transient that is 4x the E4M3 bytes instead of 2x, which the measurement below shows is affordable.

**Peak memory is bounded by construction, and measured.** Each pair is `eval`ed before the loop advances, and the E4M3 bytes and the sidecar are dropped before the quantize call. Without the `eval` MLX would keep every reconstruction lazy until the first forward pass, at which point the transient is not one tensor's f32 copy but the whole checkpoint's.

### 2.1 What is refused, and why by name

Each of these mis-scales a tensor without failing if it is accepted, which is precisely the failure mode this PR exists to remove:

| Condition | Result |
|---|---|
| `weight_block_size` other than `[128, 128]` | named error |
| `fmt` present and not `e4m3` | named error |
| `quant_method: "fp8"` with no `weight_block_size` (per-tensor FP8) | named error |
| Weight rank other than 2 | named error |
| Sidecar shape that does not tile the weight | named error |
| Sidecar with no matching `.weight` | named error |
| Width not a multiple of 32 | named error |
| Weight already decoded to a float dtype | named error |
| No sidecar at all | tensor passes through unchanged |

The last row is what lets the pre-pass run unconditionally on the Qwen3.5 paths.

---

## 3. Where it is wired

All three Qwen3.5 entry points, because all three reach `Qwen35Model::from_weights` and all three would otherwise build dense layers over unscaled bytes:

- `Qwen35Model::load` (text): detect from the full config, merge the effective `quantization` into `text_config` before it is deserialized, convert after `load_text_weights` and before `sanitize_moe_weights`.
- `load_qwen3_5_vlm_with_variant` (VLM, the path the released checkpoint takes): convert immediately after `load_vlm_weights_common` and **before** the weight map is split into text and vision, so every sidecar is still adjacent to its weight. The vision tower simply has none.
- `try_load_special_model_from_weights` / `qwen35_text_config` (adapter): same two steps on the owned copy.

The merged block is `{"group_size": 32, "bits": 8, "mode": "mxfp8"}`. `Qwen35Config` reads only `group_size` and `bits`; `UnifiedLinear::from_weights` then calls `infer_quantization_mode(has_biases: false, 32, 8)`, which returns `"mxfp8"`. Nothing had to be threaded through the model constructor, and the dense tensors keep taking the dense branch because that branch keys on the presence of `.scales` per tensor, not on the config.

---

## 4. Validation

Every command ran on Linux / aarch64 / GB10 (sm_121) with `--features cuda` and `MLX_ENABLE_TF32=0` set explicitly. MLX defaults `MLX_ENABLE_TF32=1`, which is a 10-bit mantissa and is not valid f32 evidence.

### 4.1 The unit-level proof

`cargo test --profile test-fast --features cuda --lib models::fp8_block` - 10 passed.

The central test, `fp8_block_requantize_matches_direct_path`, compares the device path **bitwise on both planes** against a reference built on the host: expand each block scale over its whole tile, multiply element by element, and quantize the result with the same `quantize_weights_with_mode(..., 32, 8, "mxfp8")`. The two routes share only the E4M3 decode table. Block pairing, the padding, the broadcast axis order, and the trailing slice are all wrong-answer-shaped rather than crash-shaped, and a bitwise comparison is the only thing that catches them. The tensor is 130x160, aligned on neither axis, so the padding and the slice are both live.

The same test also dequantizes the result and checks it against the reconstruction per group: E4M3 carries four significant bits, so an element sharing its group's binade with the group maximum rounds by at most `group_max / 16`. That bound holds for every element.

Supporting cases: a block-aligned 256x128 tensor (the no-padding branch, also bitwise); all 256 E4M3 encodings through the device LUT compared to the host table bit for bit, plus a round-trip through `f32_to_f8_e4m3`; the seven refusals in the table above; dense tensors left untouched; and an identity check on a map with no sidecars.

### 4.2 Real checkpoint

`Qwen/Qwen3.8-27B-FP8`, 29 GB:

```
mlxcel generate -m models/mlx/qwen3.8-27b-fp8 -p "Write a limerick about a Rust compiler." -n 220 --show-reasoning
```

```
Model loaded in 51.195s (resident: 23.36 GB, peak: 24.28 GB).
...
There once was a coder in Rust,
Whose compiler was grumpy and just,
It said, "You must borrow,
Or your code will not go,
'Cause lifetimes will make you be dust."
[Generated 207 tokens in 34.85s = 5.94 tok/s]
```

The model produced an on-topic reasoning trace that checked its own rhyme scheme, then a well-formed limerick, and stopped on its own at 207 of 220 tokens. A separate 64-token run loaded in 32.096 s at 23.36 GB resident and 24.37 GB peak.

**Peak memory**: 24.37 / 23.36 = **1.04x**, against the 1.5x the issue asks for. These are MLX allocator figures (`mlxcel_core::memory::snapshot`, active and peak), which is the right instrument here: process RSS peaked at 6.9 GB under `/usr/bin/time -v` because MLX's CUDA allocations on this host do not appear in RSS. The number is measured, not asserted.

### 4.3 Neighbouring suites

`--lib models::qwen3_5` (28 passed, 11 ignored), `--lib models::sanitize` (71 passed), `--lib loading::special` (6 passed). `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets --features cuda -- -D warnings` pass.

---

## 5. What is not verified

**Logit parity against the bf16 original.** The issue asks that top-5 logits agree with `Qwen/Qwen3.8-27B` within the mxfp8 tolerance. That checkpoint is 55.6 GB and was deliberately not fetched on the validation host. The criterion is left unticked, and no substitute is offered for it: comparing against a different quantization of the same architecture, or against this PR's own dequantization, would answer a different question. What is proven instead is the reconstruction formula itself, bitwise, at the unit level, plus the scale direction on real tensors in numpy, plus coherent generation at model scale.

**`cargo test --workspace --profile test-fast --features metal,accelerate`.** Not runnable on this host: Linux, aarch64, CUDA, no Metal and no Accelerate. Unrun, not passed. The CUDA equivalents above were run for the changed scope.

**`qwen3_5_moe` FP8.** No such release was available. Detection covers it because it keys on the config, and a fused 3-D `experts.gate_up_proj` carrying block scales would be refused by name (rank 2 is required) rather than mis-converted. Per-expert 2-D experts would convert and reach the existing quantized switch-layer loaders, but that is reasoning, not a measurement.

**MTP.** The drafter keeps its own weight loader inside `mlxcel-core`, which cannot reach `crate::models::fp8_block`, so the bundled `mtp.*` FP8 tensors are not converted for it. MTP speculative decoding is already unsupported for this family, and the VLM loader drops `mtp.*` entirely, so nothing regresses.

---

## 6. Follow-ups worth filing

- Move the FP8 block reconstruction into `mlxcel-core` so the MTP drafter and any other core-side loader can use it, if MTP support for this family is ever revisited.
- `requantize_block_fp8_weights` is family-agnostic already. DeepSeek V3 (UE8M0) and Mistral 4 FP8 could reuse it once their block geometry is confirmed; `mistral4.rs` currently drops `weight_scale_inv` keys outright, which is the same silent mis-scaling this PR removed from Qwen3.5.
- The conversion is about 30 s on a 27B checkpoint every load. A cached converted copy on disk would remove it, at the cost of a second full-size artifact.
