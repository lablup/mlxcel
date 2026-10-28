# PR #2707: Dequantize NVFP4 on the GPU and add a WMMA expert prefill (ROCm)

**Date**: 2026-10-28 (written before merge; PR open at `3cd30dbf`)
**Status**: Verified on gfx1151 only; wave64 and multi-arch fat builds not built
**Languages**: HIP/C++ (mlxcelverse ROCm overlay), Rust (model comment, tests), Markdown
**Risk Level**: Medium (default-on numerics change in MoE prefill; new device code compiled for architectures nobody built)

## Executive Summary

PR #2707 (closes #2698) roughly triples NVFP4 prefill throughput of `mlx-community/Nemotron-Cascade-2-30B-A3B-nvfp4` on gfx1151 and raises decode by about a third. Two changes do the work. First, an NVFP4 `dequantize` on a GPU stream now runs on the GPU instead of being redirected to the CPU stream; this removes a host gap that took about two thirds of a 512-token prefill (the embedding lookup) and several milliseconds of every decode step. Second, a rocWMMA tile variant of the expert-batched NVFP4 gather (`nvfp4_gather_expert_wmma_kernel`) takes the sorted MoE prefill when experts average at least 16 rows.

512-token prefill went from 329.3 to 1007.5 tok/s, 2048-token prefill from 330.9 to 1105.7 tok/s, and decode from 45.2 to 60.0 tok/s (512-token prompt), medians of three runs of one binary. A third candidate, NVFP4 support in the fused Mamba2 and fused MoE decode calls, was built, measured no faster, and not shipped; the measurement is recorded at the code site so nobody rebuilds it blind. The WMMA kernel's behavior on wave64 devices and in a multi-architecture build is unverified.

## 1. Problem Statement

Issue #2698 followed #2665, which made Nemotron-H load NVFP4 exports and added `nvfp4_gather_expert_batched_kernel` for the sorted MoE prefill. After #2665 a kernel trace of a 512-token prefill still showed the GPU idle for about half the wall time, and #2665's LOCAL_FIXES item 69 left that gap open. The issue also asked whether Nemotron-H's fused decode calls (`fused_mamba2_forward`, `fused_moe_forward`), which accept only affine weights, should learn NVFP4.

The trace localized the idle time: about 1 s of a 1.5 s prefill (2.1 s under the tracer) sat between the embedding row gather and the first layer, with no host API calls on the main thread. The cause was in the overlay's `fp_dequantize` (`mlx/ops.cpp`): on ROCm it redirected every block-float `dequantize` on a GPU stream to the CPU stream, because the generic `fp_dequantize` kernel (`quantized/fp_quantize.hip`) does not decode block-float scales. `quantized_embedding` is a row gather followed by `dequantize`, so each NVFP4 embedding lookup dequantized its rows on the CPU, reading them from fine-grained (uncached, device-coherent) memory at about 2 ms a token. Decode paid the same cost per token, about 4 ms of GPU idle per step.

## 2. Change Summary

| Area | Change |
|---|---|
| `mlx/ops.cpp` (`fp_dequantize`) | On a GPU stream, NVFP4 with no global scale and an f32, f16 or bf16 output builds a `fast::Quantize` primitive (dequantize direction) instead of falling through to the CPU route. `MLX_ROCM_NVFP4_GPU_DEQUANTIZE=0` (exactly `0`) restores the CPU route. MXFP4, MXFP8, global-scaled NVFP4 and other output types keep the CPU route. |
| `quantized/quantized.cpp`, `quantized.h` | `fast::Quantize::eval_gpu` dispatches `QuantizationMode::Nvfp4` to the new `nvfp4_dequantize`, declared in `quantized.h`. |
| `quantized/qmm.hip` (`nvfp4_dequantize`) | Checked wrapper (`check_nvfp4_planes`, encoder input/output registration) around the existing exact `nvfp4_dequantize_dense` that LOCAL_FIXES item 63 added for the GEMM route. No new dequantize kernel. |
| `quantized/qmm.hip` (`nvfp4_gather_expert_wmma_kernel`) | New 128-thread kernel, instantiated for `__half` and `hip_bfloat16` with two 16-row tiles a warp. Routing as in #2665: block z is expert z, `find_sorted_expert_run` finds its run, block z == E zeroes rows routed at or past E. Per 64-wide K step it gathers 32 activation rows into shared memory once per block; each of four 32-lane warps dequantizes a 32-column slab of the weight tile into the activation dtype (exact), then accumulates 2x2 16x16x16 rocWMMA tiles in f32. The f32 output tile aliases the A/B shared buffers; the global scale is applied once at the store. |
| `GatherQMM::eval_gpu` dispatch | The WMMA variant is taken inside #2665's NVFP4 expert-batched route when: f16 or bf16 activations, `K % 64 == 0`, `B / E >= MLX_ROCM_NVFP4_GATHER_WMMA_MIN_ROWS` (16), activation and weight bases 16-byte aligned and scales 4-byte aligned, `d.has_native_wmma()`, and `wmma_qmm_wavefront_ok()` (wave32). `MLX_ROCM_NVFP4_GATHER_WMMA=0` keeps the scalar kernel. Grid x is twice the average run's tile count, capped at the total tile count. |
| `src/models/nemotron_h.rs` | Comment only, at `is_affine_quantized`: records the rejected fused NVFP4 decode experiment and its numbers. No behavior change. |
| Docs | `docs/environment-variables.md` (three new variables), `docs/installation.md` (NVFP4 row: new numbers, WMMA output divergence, rejected fused decode), overlay `LOCAL_FIXES.md` item 70. |
| Tests | `tests/rocm_nvfp4_quant.rs`: GPU `dequantize` and the embedding lookup bit-exact against the CPU `dequantize`. `tests/rocm_nvfp4_quant/gather.rs`: WMMA route at 16 experts and 512 rows with multi-tile, single-row and empty runs, rows past the last expert, with and without the global scale, a partial 128-column block, f32 (scalar), f16 and bf16; plus a child-process rerun with the variant off. |

Statistics: 10 files, +616 / -8. Commits: `7e7378b7` (implementation), `3cd30dbf` (review fixes: alignment gate on the WMMA route, output-type gate on the GPU `dequantize` route).

## 3. Technical Decisions

### 3.1 Reuse the GEMM route's exact dequantize, behind `fast::Quantize`

The fix could have taught the generic `fp_dequantize` kernel to decode E4M3 scales, or added a new primitive. Instead the graph builds the existing `fast::Quantize` primitive in dequantize mode, and its `eval_gpu` calls `nvfp4_dequantize_dense`, which item 63 already uses on the dequantize-then-GEMM path and which is exact (E2M1 times E4M3 has at most six significant bits and fits fp16's normal range). This kept the change to a dozen lines of host code and gave a bit-exact result against the CPU route, which is why greedy output with the GPU `dequantize` alone is byte-identical on both test prompts.

The route is deliberately narrow. Global-scaled NVFP4 is excluded because `nvfp4_dequantize_dense` has no global-scale input; MXFP4 and MXFP8 still need a scale-aware kernel; and outputs other than f32/f16/bf16 keep the CPU route rather than throwing (the review fix in `3cd30dbf`). The kill switch compares the value to exactly `"0"`, unlike the WMMA switch, which uses `parse_warp_kernel_env`; the docs state each rule.

### 3.2 WMMA variant gated by rows per expert

The variant amortizes one shared-memory dequantize of a 64 x 128 weight tile over up to 32 rows. With few rows per expert that per-block dequantize costs more than the scalar kernel's FMAs. Measured on the target checkpoint (128 experts, top-6), prefill with and without the variant:

| Prompt tokens | Rows per expert | With WMMA | Scalar |
|---|---|---|---|
| 64 | 3 | 319 | 389 |
| 128 | 6 | 526 | 615 |
| 256 | 12 | 765 | 768 |
| 512 | 24 | 1017 | 877 |

The crossover sits between 12 and 24, so the default threshold is 16 average rows (`B / E`, integer division), adjustable through `MLX_ROCM_NVFP4_GATHER_WMMA_MIN_ROWS`. The average is a proxy: a skewed routing can still give some experts runs shorter than one tile, which the kernel handles (masked rows read the run's first row and are not stored) but does not optimize.

Tile shape was measured too. Two row tiles a warp beat one and four (single runs): 1115 against 936 and 1043 tok/s at 2048 tokens, 1021 against 928 and 801 at 512. Grid x at one, two and four times the average tile count measured within 2%, so two was kept as a bounded occupancy knob.

### 3.3 Rejected: fused NVFP4 decode (measured, not shipped)

`fused_mamba2_forward` multiplies only affine projections, so an NVFP4 export takes the unfused Mamba2 step and the unfused MoE path (`is_affine_quantized` returns false without biases). The issue asked whether extending the fused calls would pay. The PR implemented NVFP4 support in both `fused_mamba2_forward` and `fused_moe_forward` and measured it with a toggle on one binary, medians of three:

| Prompt | Fused | Unfused |
|---|---|---|
| 512 tokens | 61.0 tok/s | 61.9 tok/s |
| 2048 tokens | 59.8 tok/s | 60.3 tok/s |

The fused calls halve the per-step graph build (about 0.5 against 1.0 to 1.3 ms), but the engine's decode loop already overlaps graph construction with the previous step's GPU work, so the saving does not reach the wall clock and the fused path comes out slightly slower. The code was dropped; the numbers and the reason are recorded in the doc comment on `is_affine_quantized` in `src/models/nemotron_h.rs` and in the NVFP4 row of `docs/installation.md`. Anyone revisiting this should first check whether decode has become build-bound (for example after other per-step GPU time falls), since that is the condition under which the halved build time would show up.

### 3.4 Accepting a numerics change on a default-on path

The scalar kernel and the WMMA variant both accumulate in f32 but in different orders, so some bf16 outputs move by one rounding step. On a 600-token prompt the 160-token greedy output diverges about 17 tokens in ("repeats about" against "repeats the description of"); short prompts stay below the row gate and are unchanged. The PR accepts this as summation-order drift rather than a defect: the tests hold the variant to the same per-expert f32 reference and tolerance as the scalar kernel, and `MLX_ROCM_NVFP4_GATHER_WMMA=0` restores the previous output. Anyone comparing greedy transcripts across this change should expect the difference.

## 4. Validation

gfx1151, `Nemotron-Cascade-2-30B-A3B-nvfp4`, `scripts/bench_decode.sh --prompt-tokens N` under `scripts/rocm_gpu_guard.sh --hold`, medians of three, one binary with toggles at 0 and unset:

| | Before | GPU dequantize only | After |
|---|---|---|---|
| Prefill 512 | 329.3 | 876.3 | 1007.5 |
| Prefill 2048 | 330.9 | 900.5 | 1105.7 |
| Decode, 512 prompt | 45.2 | 59.9 | 60.0 |
| Decode, 2048 prompt | 43.5 | 59.9 | 58.7 |
| Prefill 64 | 220.6 | | 389.1 |

Most of the gain is the host gap (3.1); the WMMA variant adds about 15 to 23% on top at 512 and 2048 tokens and does not run in decode. `cargo test --features rocm --test rocm_nvfp4_quant` (11 passed, 1 ignored child) and `--test rocm_gather_qmm_expert_batched` (2 passed) pass on gfx1151.

## 5. Learning Points

- A CPU fallback in an op builder can hide as GPU idle time with no host API activity. The trace showed a gap with nothing on the host thread; the cause was a graph-level stream redirect, not a slow kernel. Grep the overlay's `ops.cpp` for `s.device == Device::gpu` redirects when a trace shows unexplained idle time around an op.
- Fine-grained memory makes CPU fallbacks much more expensive on an APU than on a discrete GPU with a fast host copy: the CPU reads uncached, coherent memory.
- Gate tensor-core variants on work per block, not only on shape legality. The WMMA kernel is legal at 3 rows per expert and 18% slower there.
- Record rejected optimizations with their numbers at the code site that would invite them. A future reader of `is_affine_quantized` sees why NVFP4 is excluded and under what condition the answer could change.

## 6. Follow-up and Limitations

**Unverified: wave64 and multi-architecture builds of the WMMA kernel.** The host builds gfx1151 only. `ROCM_HAS_WMMA` in `qmm.hip` is 1 for the CDNA wave64 targets (gfx908, gfx90a, gfx942, gfx950) as well as RDNA3/3.5/4, so in a fat build the kernel body, including its `__half` rocWMMA fragments, is compiled for wave64 devices too. Earlier `qmm.hip` fragments are `hip_bfloat16` only; `flash_attention_wmma.hip` already instantiates `__half` fragments, which lowers but does not remove the compile risk, since that file uses a different fragment configuration and gate. At run time a wave64 device never launches the kernel (`wmma_qmm_wavefront_ok()` requires a 32-lane wavefront), which matters because the kernel's layout (`warp = tid >> 5`, four 32-lane warps each owning a tile) assumes wave32 and would be wrong on a 64-lane wavefront. The remaining risk is a compile failure (rocWMMA `static_assert` or unsupported f16 fragment configuration) on a non-gfx1151 target, which would break the whole ROCm build for that target list. A `MLX_ROCM_ARCHITECTURES` build that includes at least one CDNA target and gfx1100 or gfx1201 would close this. Architectures outside the `ROCM_HAS_WMMA` list compile an empty body and are excluded at dispatch by `d.has_native_wmma()`.

Other limitations:

- Global-scaled NVFP4 `dequantize`, MXFP4 and MXFP8 `dequantize` still run on the CPU stream. A checkpoint whose embedding uses one of these keeps the host gap.
- The row gate uses the average rows per expert; heavily skewed routing is not modeled.
- Only one checkpoint (`Nemotron-Cascade-2-30B-A3B-nvfp4`) on one device was measured. The 16-row crossover may differ on other RDNA parts or other expert shapes.
- The WMMA path changes greedy output on long prompts (3.4).

## References

- Issue #2698; prior work #2665 (expert-batched NVFP4 gather, LOCAL_FIXES item 69), #2617 (native NVFP4 on ROCm), item 63 (`nvfp4_dequantize_dense`), item 52 (`wmma_qmm_wavefront_ok`)
- `src/lib/mlxcelverse/overlays/rocm/LOCAL_FIXES.md` item 70
- `src/models/nemotron_h.rs` (`is_affine_quantized`), `docs/environment-variables.md`, `docs/installation.md`
