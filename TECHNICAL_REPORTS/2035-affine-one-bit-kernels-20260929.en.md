# Technical Report: PR #2035 - Run Affine 1-bit Checkpoints with Fused Metal Kernels

**Date**: 2026-09-29

**Status**: Implemented and validated on Apple M1 Ultra (Metal); pending merge.

**Languages**: C++ (Metal kernels, bridge routing), Rust (loader validation, layer routing, tests)

**Risk Level**: Medium

## Executive Summary

MLX ships affine quantized kernels for 2, 3, 4, 5, 6 and 8 bits. Public 1-bit checkpoints (`prism-ml/Bonsai-{1.7B,4B,8B,27B}-mlx-1bit`, plain `qwen3`) use the standard MLX packing at one bit per weight, passed every mlxcel load check, and aborted the process on the first token. This PR adds mlxcel's own 1-bit matvec and matmul kernels, routes every bridge-level quantized primitive to them, and keeps fused C++ helpers (which call MLX directly) off 1-bit weights.

## 1. Problem Statement

On main, `mlxcel generate -m Bonsai-1.7B-mlx-1bit` loaded in 0.12 s and then printed `libc++abi: terminating due to uncaught exception of type std::runtime_error: [metal::Device] Unable to load kernel affine_dequantize_float16_t_gs_128_b_1` (exit 134). The first failure was the embedding lookup's dequantize, not a matmul. Load succeeded because `validate_quantization_params` is a bounds check (1..=32) and `infer_quantization_bits` returned the declared width early whenever the shapes agreed, so its {2..8} allowlist was never consulted.

## 2. Layout and Math

For a linear with `N` rows and `K` columns, `weight` is `uint32 [N, K/32]` with bit `j` (LSB first) of word `c` holding column `32c + j`; `scales` and `biases` are `[N, K/G]`, `G` in {32, 64, 128}. Dequantized `w = bit * scale + bias`. Because the bias does not depend on the bit, a matvec needs only two sums per (row, group): the activations under the bit mask, and all activations. `y = sum_g scale * masked_sum + bias * total_sum`.

## 3. Change Summary

- `cpp/mlx_cxx_one_bit.cpp` (new): qmv (64-thread threadgroups, 2 simdgroups of R = 4 or 8 rows, 16 columns per lane per step, `simd_sum` reduction, `ALIGNED` template to drop the row guard), qmm (32x32 tiles, 128 threads, bits expanded to `bias` or `scale + bias` in threadgroup memory, 8x8 simdgroup fragments, f32 accumulation, `m_size` passed as a scalar input so a new prompt length does not compile a new kernel), and a dequantize graph that unpacks through a `uint8` view (a quarter of the temporary a `uint32` shift would need).
- `mlx_cxx_bridge.cpp`: `quantized_matmul` (both `transpose` layouts), `quantized_linear_forward`, `quantized_linear_forward_global_scale`, `dequantize` and `quantized_embedding` route an affine `bits == 1` triple with biases to the new file.
- `layers.rs`: `QuantizedWeight::is_one_bit`; the two `UnifiedLinear` accessors that feed fused helpers answer `None` for 1-bit; explicit declines in the su-scaled-rope path and compiled `SwiGLUMLP`; the #1994 dense-GEMM prefill is skipped for 1-bit so prefill uses qmm; `LOADABLE_AFFINE_BITS` and `validate_one_bit_layout`; `QuantizedMultiLinear` refuses 1-bit.
- `switch_layers.rs`: 1-bit expert stacks refused at load with the prefix.
- FFI: `one_bit_quantized_matmul`, `one_bit_dequantize` (both `Result`), `one_bit_kernel_available`.

## 4. Technical Decisions

### Routing at the bridge, declining at the accessor

Two kinds of callers reach MLX with a quantized triple. Bridge primitives (`ffi::quantized_matmul` and friends) are called from `UnifiedLinear`, `QuantizedEmbedding` and about thirty model files; routing inside them covers all of those at once, including callers the issue did not list. Fused C++ helpers call `mlx::core::quantized_matmul` themselves and cannot be routed without touching each one. Every such helper is gated on `quantized_weight()`, `as_quantized_weight()` or `fused_quantized_weight()`, and runtime LoRA (#1439) already made `None` mean "take the graph fallback" for all of them. Answering `None` for 1-bit reuses that contract, so no new fallback code was written.

### Two allowlists, not one

`SUPPORTED_AFFINE_BITS` answers what MLX's affine quantize can emit and is read by producers (`split-mtp`, gemma4 per-module overrides). Adding 1 there would let those produce a width nothing can quantize. The loader instead reads a new `LOADABLE_AFFINE_BITS`.

### The oracle

The kill switch runs `x @ dequant(W)^T` with the dequantized weight rounded once to the shipped dtype (`scale + bias` formed in f32). The unit tests do not compare the kernel to that path; both are compared to a host f64 computation over the exact device values, so a shared bug cannot hide.

### CUDA deferred

The issue asked for a CUDA qmv port or a recorded reason. No CUDA host was available (GB10 runner down). A JIT kernel that has never executed, enabled by default, risks silent wrong output, so the port table is Metal-only and CUDA/ROCm take the dequantize graph through `has_kernel_port`, never through a refusal.

## 5. Validation

- Step 0 on main: the abort message quoted in section 1.
- `one_bit_tests` (17): kernel and fallback each within 5e-3 (f16/f32) or 2e-2 (bf16) of `max|y|` against the f64 oracle across M 1 to 40, K 256 to 4096, N 64 to 1003 (unaligned included), G 32/64/128; dequantize exact against the host formula for f32/f16/bf16; kill switch bit-identical to the graph; `transpose=false`; embedding gather and tied `as_linear`; fused QKV declines; load rejections name the prefix.
- `Bonsai-1.7B-mlx-1bit`, 64 greedy tokens: kernel and fallback text identical, answer "Paris". `Bonsai-8B-mlx-1bit`, 128 greedy tokens: identical, fluent. `mlxcel-server` with two concurrent greedy chat requests on the 1.7B: both correct.
- Contract tests, `layers::tests`, `switch_layers::tests`, fmt and clippy (`-p mlxcel-core`, `-p mlxcel`) pass.

## 6. Deferred

- Decode throughput against `mlx-community/Qwen3-8B-4bit`: the host GPU was shared with other jobs during this run, so no timing is published.
- CUDA qmv port and its validation on GB10.
- 1-bit MoE expert stacks and MLA projections (refused at load; no public checkpoint uses them).
