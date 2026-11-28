# PR #2679: Expert-batched NVFP4 sorted-gather prefill on ROCm, validated on Nemotron-H

**Date**: 2026-10-27
**Status**: Complete on Linux gfx1151 (ROCm 7.15)
**Languages**: HIP, Rust, Markdown
**Risk Level**: Medium (new kernel on the default NVFP4 prefill path; Nemotron-H loader change)

## Executive Summary

Issue #2665 asked for an expert-batched prefill route for NVFP4 MoE gathers on ROCm, validated on a real checkpoint. The target checkpoint, `mlx-community/Nemotron-Cascade-2-30B-A3B-nvfp4`, could not load at all. PR #2679 fixes the Nemotron-H loader and adds the kernel. On gfx1151 the kernel raises prefill from 269 to 328 tok/s at 512 tokens and from 237 to 330 tok/s at 2048 tokens. Greedy output is byte-identical with the kernel on and off.

## 1. Problem Statement

- `GatherQMM::eval_gpu` sent every NVFP4 gather to `nvfp4_qmv_kernel`. Its multi-row arm shares a weight load across eight rows. Every 16-column block, however, reloads each activation row, so a prefill reads its activations about N / 16 times.
- No real NVFP4 checkpoint had run on ROCm. Nemotron-H's local `QuantizedSwitchLinear` required affine `.biases` and hardcoded `"affine"`. Loading failed with `Missing switch_mlp.fc1.biases for layer 1`.
- Nemotron-H's MoE never sorted its prefill. A sorted-gather kernel would therefore never have been reached on this model.

## 2. Change Summary

| Area | Change |
|---|---|
| Nemotron-H experts | The shared `SwitchLinear` replaces the local enum. It infers the mode from the planes and refuses an unflagged `global_scale`. The declared-pair bound of #958 is kept. |
| Nemotron-H MoE | Prefills of 64 or more (token, expert) pairs are sorted by expert, as in mlx-lm's `SwitchMLP`. |
| Fused paths | `fused_moe_forward` and `fused_mamba2_forward` multiply affine weights only, so they are now gated on affine planes. NVFP4 takes the unfused `gather_qmm` and SSM step. The fused MoE also requires fc1 and fc2 to share one packing, found in review. |
| Kernel | `nvfp4_gather_expert_batched_kernel<T, 8, 4>` in `qmm.hip` handles transposed, sorted, `M == 1`, `B >= 64` gathers, with no expert-count limit. Block E zeroes the rows routed past the last expert. `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0` keeps the old kernel. |
| Records | LOCAL_FIXES item 69, `docs/installation.md` (names the validated checkpoint), and `docs/environment-variables.md`. |

## 3. Technical Decisions

### 3.1 Kernel instead of per-run dequantize plus GEMM

The issue offered a second design: dequantize each expert's run and call the GEMM. That design needs the run boundaries on the host, which costs a stream synchronization per call. At 128 x 1856 x 2688 experts it would also write about 1.27 GB of bf16 per projection call. The GPU-only kernel needs neither.

### 3.2 Tile shape

Each block handles one expert. Each 16-lane group computes several columns from one activation load. Measured at 512 tokens:

| Rows per chunk | Columns per group | Prefill |
|---|---|---|
| 8 | 1 | 290.6 tok/s |
| 8 | 2 | 317.9 tok/s |
| 8 | 4 | 327.1 tok/s |
| 4 | 4 | 312.7 tok/s |
| 16 | 1 | 151.0 tok/s (register pressure) |

The shipped shape is 8 rows and 4 columns, which keeps three instantiations (f32, f16, bf16).

### 3.3 No expert-count gate

The affine route requires `E <= 64` and `B / E >= 4`. Those limits are not copied here: the target checkpoint has 128 experts, with about 24 rows each at 512 tokens.

## 4. Validation

- **Benchmark**: `scripts/bench_decode.sh` under the GPU guard, medians of three runs of the same binary, kernel off then on.
  - 512 tokens: prefill 269.2 to 328.1 tok/s (1.22x).
  - 2048 tokens: prefill 236.5 to 330.3 tok/s (1.40x).
  - Decode: about 44 to 48 tok/s with the kernel off and on. The decode gather has six rows and never takes the route, so the spread is noise.
  - Short prompts: 16 tokens 114.1 to 119.2 tok/s and 64 tokens 195.5 to 222.9 tok/s. Most of the 128 experts get zero to two rows here, and there is no regression.
- **Output**: a 160-token greedy generation is byte-identical with the kernel off and on, and the text is coherent.
- **Kernel tests**: `rocm_nvfp4_quant` passes all 8 tests, including a new 128-expert, `B = 512` sorted case in three dtypes, with and without the global scale. A variant with `N = 250` covers the partial four-column group.
- **Red, global scale**: reading the global scale of expert `e + 1` fails the global-scale cases (relative error 0.53 to 0.54).
- **Red, expert offset**: reading the weights of expert `e + 1` fails every sorted case (relative error 1.40 to 1.58).
- **Loader test**: a new Nemotron-H lib test loads NVFP4 planes and checks the sorted and unsorted MoE paths. It fails on `main`'s loader with the missing-biases error. All 74 nemotron_h lib tests pass.
- **Lint**: clippy with `-D warnings`, fmt, and overlay records pass.

## 5. Known Gaps

- **Remaining prefill time**: a kernel trace of a 512-token prefill puts the new kernel at 7.7 ms per call, about 23% of the wall time. The GPU is idle for roughly half of the wall time, a host-side cost this PR does not address. A WMMA tile kernel could cut the MoE share further.
- **Decode on NVFP4**: Nemotron-H decodes through the unfused Mamba2 step, because the fused step is affine-only.
- **Checkpoint coverage**: only one NVFP4 checkpoint was measured. The fallback `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` was not downloaded.
- **Other backends**: the Nemotron-H prefill sort also applies on Metal and CUDA, but only for non-fused paths (latent-projection models, or backends without the fused port), matching mlx-lm. It was not measured there.
