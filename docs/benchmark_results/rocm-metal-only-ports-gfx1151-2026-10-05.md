# HIP ports of four Metal-only kernels on gfx1151 (2026-10-05)

lablup/mlxcel#2069, the last of the #1814 ports. Four kernels had a Metal port only and sat behind gates that did not read their tables: the fused xIELU activation (Apertus), the fused add3 + LayerNorm (Cohere2), the Mamba1 selective scan (Mamba, Falcon-Mamba, Jamba) and the fc1 squared-ReLU MoE kernel (Nemotron-H's opt-in `MLXCEL_FUSED_MOE_RELU2`). Each now has a HIP source in `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`, a filled `.rocm` entry and a predicate that reads its table and requires the GPU as the default device, and each gate reads that predicate. CUDA keeps its fallbacks.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0 (HIP 7.15.26333), MLX pin `81ba1c6a`, ROCm overlay `75915908`, `cargo build --release --features rocm`. Before: `origin/main` `6668031c`. After: the PR branch at `800e23dc`.

## What each port matches

| Kernel | Reference it is held to on ROCm | Result on gfx1151 | Negative check |
|---|---|---|---|
| `fused_xielu` | the elementwise `apertus_xielu` graph on the same backend, f32 / f16 / bf16 | byte-identical (0 of 16410 elements differ in each dtype) | `alpha_n` in the positive branch fails it (nrms 0.30) |
| `fused_add3_layer_norm` | the unfused `compiled_add3` + `fast::layer_norm` pair on the same backend | byte-identical in all ten cases (f16, bf16, f32; widths 5, 1025, 4096 and the 6656 limit) | `rsqrtf` for `1.0f / sqrtf` fails it |
| `mamba1_selective_scan` (float32-state variant) | f32 scalar reference within 1e-5; bf16 no less accurate than the graph scan | passes, N up to 32 | a lane fold starting at 8 fails at N = 32 (relative error 0.87) |
| `moe_fc1_relu2` (+ the #2065 down kernel) | dense f32 reference; `gather_qmm` within its own jitter | 8.1e-6 nrms / 4.2e-4 nmax from the reference; 5.45e-3 to 5.70e-3 from `gather_qmm`, which is itself 5.08e-3 to 5.43e-3 from the reference | a lane fold starting at 8 fails it (nrms 0.76) |

The xIELU and add3 ports follow ROCm's own graph rather than the Metal kernels: the Metal kernels are byte-identical to Metal's graph by reproducing its rounding points and reduction order, and on ROCm the overlay's elementwise kernels (one rounding to `T` per op, the device `expm1f`) and its `layer_norm_kernel<T, 256, 4>` (256 threads per row, `__shfl_xor` folds of width 32, `1.0f / sqrtf`) set different ones. ROCm builds therefore launch add3 with 256 threads per row. The identity is pinned by the tests, not by matching compiler flags, so a ROCm compiler upgrade could break it and the tests would say so.

The Mamba1 port is the float32-state variant (the Metal one). The CUDA graph-exact variant cannot be reproduced on ROCm: the graph scan's `state @ C` has K = N (8 or 16), which fails the overlay's GEMV condition `K % 32 == 0`, so it runs through rocBLAS, whose reduction order a custom kernel cannot copy. The kernel reads the sequence length from `X_shape`, which needed the by-value shape arguments that #2100 added to the overlay (`LOCAL_FIXES.md` item 30).

## Decode throughput

Only Nemotron-H has a checkpoint on this host. Apertus, Cohere2, Mamba, Falcon-Mamba and Jamba have none, so their decode is not measured; their kernels are covered by the tests above. `scripts/bench_decode.sh` at pp512/tg128 with `MLXCEL_FUSED_MOE_RELU2=1` on both sides, before and after alternated run by run, every run through `scripts/rocm_gpu_guard.sh`. Before, the flag declined to `gather_qmm` (no fc1 port); after, it takes the HIP fc1 and down kernels. Raw rows: `benchmarks/rocm_strixhalo-gfx1151_2026-10-05_relu2-before.csv` and `..._relu2-after.csv`.

| Model | Before tok/s | After tok/s | Change (medians) |
|---|---|---|---|
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit, `MLXCEL_FUSED_MOE_RELU2=1` | 75.15 / 75.22 / 75.09, median 75.15 | 75.24 / 75.38 / 75.13, median 75.24 | +0.1%, within noise |

This matches the Metal note on the flag: it replaces only the routed fc1 and fc2 GEMVs, which `gather_qmm` already runs near bandwidth, so it is performance-neutral on Nemotron-H. It stays opt-in.

## Correctness of the relu2 path on a model

The issue asked for Nemotron-H's greedy 128-token output with the flag set to match the default path on gfx1151. It does not: on three prompts the texts agree for about 20 tokens and then part at a near-tie ("user query" against "user request", "checks if" against "checks whether"). The kernel keeps fc1, relu² and fc2 in f32 where `gather_qmm` rounds the intermediates to bf16, which is why it sits about a thousand times closer to the f32 reference in the parity test, and a free-running generation conditions everything after the first flip on different text. The teacher-forced traces show the flips are near-ties (`benchmarks/logit_traces/rocm_gfx1151_9186b075/`, `python3 scripts/compare_logit_traces.py <reference> <candidate> --decided 2.0`):

| Reference | Candidate | Top-1 disagreement | Decided mismatches | Largest gap | Perplexity ref / cand |
|---|---|---|---|---|---|
| ROCm default | ROCm `MLXCEL_FUSED_MOE_RELU2=1` | 1 / 128 | 0 / 66 | 0.500 | 7.410 / 7.436 |
| Metal M5 default (`metal_m5_d1128266`) | ROCm `MLXCEL_FUSED_MOE_RELU2=1` | 5 / 128 | 0 / 71 | 0.250 | 7.489 / 7.436 |

The default path is unchanged by this PR: its `w1ctx512` trace on the branch is byte-identical in every data row to one built on `main` `6668031c`.

## CPU device

Custom kernels run only on the GPU stream. Each new predicate, and the two fused MoE predicates from #2065, also requires the GPU as the default device, so under `MLXCEL_DEVICE=cpu` the models take their graph paths instead of throwing "Custom kernels only run on GPU". `tests/cpu_device_custom_kernel_gates.rs` (its own test binary, so moving the default device cannot affect other tests) checks all six predicates and runs `fused_xielu` and `residual_add3_layer_norm` on the CPU.

## Not measured

Decode and end-to-end output for Apertus, Cohere2, Mamba, Falcon-Mamba and Jamba (no checkpoint on this host). Metal and CUDA (not available on this host): the Metal and CUDA kernel sources and table entries are untouched; Metal now also runs the widened add3 and Mamba1 tests and the new relu2 test, and its predicates gained the GPU-device term. Wave64 (CDNA) parts.
