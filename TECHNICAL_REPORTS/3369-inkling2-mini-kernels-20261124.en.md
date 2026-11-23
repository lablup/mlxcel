# PR #3369: Inkling 2 Mini Metal kernels

**Date**: 2026-11-24
**Status**: Draft PR open (closes #3359). Narrow suites pass; validated on an M3 Ultra (512 GB) on both Inkling 2 Mini checkpoints

## Executive Summary

The PR ports mlx-vlm's three Inkling 2 Mini Metal kernels: the fused short convolution, attention head preparation and the grouped NVFP4/MXFP8 expert matvec. Each kernel matches the ops path it replaces bit for bit and falls back to it outside its gate, and `MLXCEL_INKLING2_KERNELS=0` turns all three off. Decode goes from 32.1 to 35.3 tok/s on the native checkpoint and from 36.8 to 44.8 tok/s on the 4-bit export, both now ahead of mlx-vlm's 32.8 and 38.5. A verify forward of 8 tokens drops from 60.4 to 51.8 ms. Greedy output is byte-identical with and without the kernels. To make the grouped kernel reachable, Inkling 2 now supports prompt-lookup verify rounds.

## 1. Problem Statement

After #3367, Inkling 2 Mini decoded at 32.1 tok/s native and 32.0 tok/s 4-bit, against 32.8 and 38.5 for mlx-vlm. Upstream had three kernels the port left out. Every layer visit (48 per token) runs four short convolutions as about eight small ops each, plus four per-head normalizations of about seven ops each. During a short verify, every routed expert is decoded once per route.

| Risk | Impact | Likelihood |
|---|---|---|
| A kernel changes numerics | Greedy output drifts from the ops path and from mlx-vlm | Addressed by bitwise parity tests and equal greedy output |
| A kernel breaks the trim layout of #3367 | Speculative rewinds become inexact | Addressed: the kernel writes the same history |

## 2. Design

### 2.1 Kernels and gates

| Kernel | Gate | mlxcel difference from upstream |
|---|---|---|
| Short convolution (02053f5c) | bf16 `x`, kernel 4, `B <= 4`, `L <= 8`, `C % 128 == 0`, f32 history, Metal | Writes the next margin history (`min(W - 1 + margin, R + L)` rows) instead of the `W - 1 + L` window; no padding mask |
| Head preparation (de08fdf8) | Looped model, gamma-free value norm, `[1, 1..=8, H, 128]` bf16, `d_rel == 16` | Reads the f32 `q_scale` / `rel_scale` the model keeps |
| Grouped expert matvec (8e80d434) | Looped model, speculative verify forward, `[1, 2..=8, hidden]` bf16, top-k below 64, every plane native NVFP4 or MXFP8 without biases | Gated on `speculative_append_active()` rather than an upstream argument |

The launchers live in `src/lib/mlxcelverse/kernels/inkling2_kernels.cpp` and `inkling2_grouped_experts.cpp`. Each has a `KernelPorts` table with only a Metal entry and is chosen through `select_kernel_port`. The bridge declarations return `Result`, and `mlxcel_core::inkling2_kernels` turns a refusal into `None`. The grouped kernel inlines MLX's fp4/fp8 decoders, because the JIT preamble does not include them. It follows `fp_qmv_fast_impl` / `fp_qmv_impl` lane assignment, dot grouping, sum order and global-scale placement, so each route equals its `gather_qmm` row.

### 2.2 Reaching the grouped kernel

Plain decode never runs a multi-token forward, and Inkling 2 used to refuse verify rounds. The model now reports `supports_speculative_verify_rewind` and implements `rewind_decode_appends` through `trim_state`, which checks every visit before it mutates anything. A verify forward keeps `max(16, L)` extra convolution rows (`conv::history_margin`), and the sliding KV margin of 64 already covers the largest prompt-lookup block (65), so a rejected block of any allowed size can be rewound exactly.

## 3. Validation

| Check | Result |
|---|---|
| Short convolution parity: f32 and bf16 output, residual none/bf16/f32, strided `x`, history and residual, history of 0 to 19 rows | bitwise equal, next history equal |
| Short convolution rewind: speculative block of 5, keep 0/2/4/5, then a further step | bitwise equal history and output |
| Head preparation parity: `L` 1/4/8, scales 0.25/5/1, strided q and r, NaN injection at `L = 8` | bitwise equal (`u16` view) |
| Grouped experts: NVFP4 (with per-expert global scale) and MXFP8 at Mini sizes, expert 5 over two tiles, others in runs of 1 to 3 | bitwise equal to per-token and batched `SwitchGLU` |
| Gates | refused for CPU, batch 2, `L = 9`, head_dim 64, `d_rel` 8, f16, a non-loop config, an affine plane, top-k 64, and any non-verify forward |
| Greedy, 128 tokens, both checkpoints, 3 runs per arm | identical across arms |
| `--prompt-lookup` | runs on both arms with identical output |

Decode (`--profile`, A = kernels off): native 32.28 / 32.97 / 30.95 against 32.80 / 36.61 / 36.47 tok/s; 4-bit 35.94 / 37.21 / 37.28 against 44.75 / 44.72 / 44.80 tok/s. Verify forward medians on the native checkpoint: width 4 goes from 52.0 to 45.7 ms, width 8 from 60.4 to 51.8 ms. Without the speculative scope, only the grouped kernel is left out (46.9 / 54.5 ms), which puts the grouped kernel's share at 2.6% and 5.0%. The argmax digests agree in every mode.

Selectors, run serially under the GPU lock: `--lib inkling` (142 passed, which includes `models::inkling2::`) and `--lib coverage_tests` (18). Clippy, fmt and the CI contract scripts pass.

## 4. Limitations and Follow-ups

- The short convolution and head preparation kernels were measured together. The grouped kernel is isolated by the verify timing.
- Prompt-lookup accepts few proposals on the sample prompt, so there it decodes slower than plain greedy (27.4 / 31.7 tok/s). Its text agrees with plain greedy for 213 characters. After that, the multi-token verify forward picks a different phrasing, the same on both arms.
- The kernels are Metal-only. A CUDA or HIP port must add its file to `EXPECTED_IN_SCOPE` of `check_kernel_dtype_keys.py`.
- Inkling 2 Small does not take the loop kernels (no `loop_config`), which matches upstream. Its short convolution kernel is covered only by the unit tests.
