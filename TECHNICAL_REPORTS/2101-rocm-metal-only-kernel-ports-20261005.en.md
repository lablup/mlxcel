# Technical Report: PR #2101 - HIP ports of the four Metal-only kernels

**Date**: 2026-10-05

**Status**: Implemented and tested on the gfx1151 host; branch head `3833998b` (measured binary `800e23dc`) on origin/main `6668031c`, PR open, pending merge.

**Languages**: C++ (HIP kernel sources through `fast::hip_kernel`, port tables, predicates), Rust (model gates in `apertus.rs`, `layers.rs`, `mamba.rs`, `nemotron_h.rs`, FFI, parity tests, a new integration test binary), Markdown, TSV/CSV (logit traces, bench rows)

**Risk Level**: Medium (on ROCm, Apertus, Cohere2, Mamba, Falcon-Mamba and Jamba now run new GPU kernels by default, and none of those families has a checkpoint on the test host, so they are covered by kernel tests only. The Mamba1 port carries its state in float32, so Mamba-family output on ROCm changes relative to the graph scan. Nemotron-H's default path is byte-identical to main; its opt-in `MLXCEL_FUSED_MOE_RELU2` path is not byte-identical in greedy text. Metal and CUDA were not run; Metal's predicates gained a GPU-device term)

## Executive Summary

Issue #2069, the last of the #1814 port items, covered four kernels that had a Metal port only and sat behind gates that did not read their port tables: the fused xIELU activation (Apertus), the fused add3 + LayerNorm (Cohere2), the Mamba1 selective scan (Mamba, Falcon-Mamba, Jamba) and the fc1 squared-ReLU MoE kernel used by Nemotron-H's opt-in `MLXCEL_FUSED_MOE_RELU2` branch. On ROCm all four fell back to MLX graphs.

The PR adds a HIP source, a filled `.rocm` table entry and a table-reading predicate for each kernel, and points each model gate at its predicate. CUDA keeps its fallbacks. `grep -rn '\.rocm = nullptr' --include=*.cpp src/` goes from 16 to 12; the one left in `mlx_cxx_kernels.cpp` is the CUDA graph-exact Mamba1 table.

The central decision is what each port is held to. xIELU and add3 are byte-identical to ROCm's own unfused graph (0 of 16410 elements differ in f32, f16 and bf16 for xIELU; all ten add3 cases identical), because they reproduce the ROCm graph's rounding points rather than the Metal kernels'. The Mamba1 port is the float32-state variant, because the CUDA graph-exact variant cannot be reproduced on ROCm. The relu2 kernel stays in f32 and is within 8.1e-6 nrms of a dense f32 reference.

On Nemotron-3-Nano with `MLXCEL_FUSED_MOE_RELU2=1`, decode moves from a median of 75.15 to 75.24 tok/s (+0.1%, within noise), as expected for a kernel that replaces only bandwidth-bound GEMVs. The issue asked for the relu2 greedy text to match the default path; it does not, and the PR documents this as a deviation backed by teacher-forced traces with 0 decided-position mismatches.

A static review found that under `MLXCEL_DEVICE=cpu` all four new paths would have thrown "Custom kernels only run on GPU", because the predicates read only the port tables. The predicates now also require the GPU as the default device, and the same fix was applied to #2065's two fused MoE predicates, which had the same exposure.

## 1. Problem Statement

### 1.1 Four kernels unreachable on ROCm

Each kernel was blocked by a gate that named Metal rather than asking whether a port exists:

- `fused_xielu` returned its elementwise fallback whenever `metal::is_available()` was false.
- `residual_add3_layer_norm` required `ffi::metal_is_available()`.
- `mamba.rs` gated the scan on `gpu_backend_kind() == GpuBackendKind::Metal`. Jamba went through `mamba1_scan_kernel_accepts`, which was false on ROCm because neither Mamba1 table had a `.rocm` entry, so Jamba, Mamba and Falcon-Mamba all walked the per-step graph scan.
- `fused_moe_forward`'s relu2 branch declined to `gather_qmm` on ROCm since #2065, because `moe_fc1_relu2_ports()` had no `.rocm` entry (only the down kernel had been ported).

Filling the tables alone would not have opened the first three gates, which is the same lesson #2065 recorded for the fused MoE pair.

### 1.2 Contracts that differ per kernel

The four kernels do not share one correctness contract. xIELU and add3 are documented as byte-identical to the unfused graph they replace, so `MLXCEL_FUSED_XIELU` and `MLXCEL_FUSED_ADD_NORM` (both on by default) never perturb greedy decode. The Mamba1 scan has two variants with different contracts: Metal's carries the state in float32 and differs from the graph scan, while CUDA's (#1981) rounds every step exactly as the graph scan does. The relu2 kernel is held to a dense f32 reference. A port therefore has to pick its target per kernel before it can be tested.

## 2. Change Summary

Eight commits:

- **`64331d34`** `update(rocm): port the fc1 squared-ReLU MoE kernel to HIP`
- **`fdc613a5`** `update(rocm): port the fused xIELU kernel to HIP`
- **`38fd43f1`** `update(rocm): port the float32-state Mamba1 scan kernel to HIP`
- **`fc5170ca`** `update(rocm): port the fused add3 + LayerNorm kernel to HIP`
- **`9186b075`** `docs(rocm): list the newly ported kernels in the README`
- **`eb72b359`** `fix(rocm): gate #2069 kernel ports on the GPU device`: the review fixes (section 4).
- **`800e23dc`** `fix(rocm): gate the fused MoE predicates on the GPU device too`: the same fix for #2065's predicates.
- **`3833998b`** `docs(rocm): publish the #2069 kernel port results on gfx1151`: `docs/benchmark_results/rocm-metal-only-ports-gfx1151-2026-10-05.md`, bench CSVs, traces under `benchmarks/logit_traces/rocm_gfx1151_9186b075/`, `docs/installation.md` and `docs/environment-variables.md` rows.

27 files, +1584 / -124. All HIP sources live in `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`.

### 2.1 `moe_fc1_relu2`

`MOE_FC1_RELU2_HIP_SOURCE` is written from the Metal source in the shape of the #2065 gate-up port: one 32-lane wavefront per row, the 16..1 fold with `__shfl_down(v, o, 32)`, the wave32 guard, and `T` in the template args. A new bridge predicate, `fused_moe_relu2_kernels_available()`, reads both the fc1 and the down tables, and the branch reads it instead of checking the two tables inline. On ROCm the branch's rows per block default to 2, the geometry #2065 measured for `run_fused_moe_two_kernel`, so both kernels of the branch use the same launch shape. The flag stays opt-in.

### 2.2 `fused_xielu`

The early return now reads `fused_xielu_kernel_available()`. The HIP kernel targets byte identity with `apertus_xielu` on ROCm, whose graph is a chain of overlay elementwise kernels that each widen to float, compute once and round to `T`, with `Expm1` calling the device `expm1f`. The port rounds every intermediate through `T` the same way, calls the same `expm1f`, propagates NaN through `minimum` as the graph does, and sets `#pragma clang fp contract(off)` so the f32 path cannot fuse `x * beta` into the final add. On ROCm the element count is read from `x_shape[0]` instead of a template argument (section 4.2). `apertus_xielu` became `pub(crate)` so the new test can compare against it.

### 2.3 `mamba1_selective_scan`

`MAMBA1_SCAN_HIP_SOURCE` ports the Metal float32-state variant, with `simd_sum` replaced by the 16..1 `__shfl_down` fold. The CUDA graph-exact variant is not reachable on ROCm: the graph scan's `state @ C` has K = N (8 or 16), which fails the overlay GEMV condition `K % 32 == 0`, so the overlay sends it to rocBLAS, whose reduction order a custom kernel cannot copy. A new predicate, `mamba1_scan_float_state_kernel_available()`, reads the float32-state table; `mamba.rs` reads it and then `mamba1_scan_kernel_accepts`, as Jamba's gate already did, so a state wider than 32 takes the graph scan instead of leaving columns unwritten.

The kernel reads the sequence length from `X_shape`. The first run faulted (HSA memory fault at `0x4000000000`) because the fork's `hip_kernel` declared `<input>_shape` and `_strides` as pointers while the launch passes them by value. The implementer fixed this in the overlay's `custom_kernel.cpp`, but #2100, merged meanwhile, fixed the same bug as `LOCAL_FIXES.md` item 30. On rebase the branch dropped its copy and uses #2100's fix; the conflicts in `docs/environment-variables.md`, `custom_kernel.cpp` and `LOCAL_FIXES.md` were resolved to main's version.

### 2.4 `fused_add3_layer_norm`

`residual_add3_layer_norm` reads the new `fused_add3_layer_norm_available()`, and the `.expect` message names that predicate. The HIP kernel reproduces ROCm's unfused pair rather than the Metal kernel's reduction: the residual is rounded to `T` after each add, as the overlay's compiled `Add` does, and the norm follows the overlay's `layer_norm_kernel<T, 256, 4>` (256 threads per row, strided groups of 4, 16..1 `__shfl_xor` folds of width 32, the eight wavefront sums folded again, `1.0f / sqrtf`, `T(w * norm + b)` with the bias read from memory). ROCm builds launch 256 threads per row through `MLXCEL_BRIDGE_ROCM_BACKEND`. The row is held in registers sized from the 6656 limit (28 floats per thread).

### 2.5 Tests

- New `fused_moe_relu2_parity_tests` runs `fused_moe_forward` with the flag unset and set at Nemotron-3-Nano shapes, 4 and 8 bit, against a dense f32 reference and the `gather_qmm` branch. Three helpers in `fused_moe_parity_tests.rs` became `pub(crate)` for it.
- New `apertus_tests::fused_xielu_kernel_matches_graph_every_dtype` compares the kernel with `apertus_xielu` in f32, f16 and bf16, asserts the kernel path is taken on Metal and ROCm, and asserts zero differing bits on ROCm. The existing bf16 bit-for-bit test now exercises the HIP kernel.
- `residual_add3_layer_norm_matches_the_unfused_pair` was compiled only with the `metal` feature. It now builds on every backend, asserts the kernel path on Metal and ROCm, skips visibly when `MLXCEL_FUSED_ADD_NORM=0`, and adds f32, widths 1025 and 5, and the 6656 limit, for ten cases.
- The Mamba1 f32 parity test adds N = 32, and a new test asserts the predicate is true on Metal and ROCm so the parity tests cannot skip silently. The bf16 test now runs on ROCm too.
- New integration test binary `tests/cpu_device_custom_kernel_gates.rs` (section 4.1).

No tolerance was loosened.

## 3. Measured Results

### 3.1 Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0 (HIP 7.15.26333), MLX pin `81ba1c6a`, ROCm overlay `75915908`, `cargo build --release --features rocm`. Before: origin/main `6668031c`. After: the branch at `800e23dc`.

### 3.2 Kernel parity

| Kernel | Reference on ROCm | Result on gfx1151 | Negative check |
|---|---|---|---|
| `fused_xielu` | `apertus_xielu` graph, f32 / f16 / bf16 | byte-identical (0 of 16410 elements differ in each dtype) | `alpha_n` in the positive branch fails it (nrms 0.30) |
| `fused_add3_layer_norm` | unfused `compiled_add3` + `fast::layer_norm` | byte-identical in all ten cases (f16, bf16, f32; widths 5, 1025, 4096, 6656) | `rsqrtf` for `1.0f / sqrtf` fails it |
| `mamba1_selective_scan` (float32 state) | f32 scalar reference within 1e-5; bf16 no less accurate than the graph scan | passes, N up to 32; `mamba1_scan_parity_tests` 5 passed | a lane fold starting at 8 fails at N = 32 (relative error 0.87) |
| `moe_fc1_relu2` with the #2065 down kernel | dense f32 reference; `gather_qmm` within its own jitter | 8.1e-6 nrms / 4.2e-4 nmax from the reference; 5.45e-3 to 5.70e-3 from `gather_qmm`, which is itself 5.08e-3 to 5.43e-3 from the reference | a lane fold starting at 8 fails it (nrms 0.76) |

The add3 failure under `rsqrtf` confirms that the test detects a change of a single rounding step, which is the reason the contract is byte identity rather than a tolerance.

### 3.3 Decode throughput

Only Nemotron-H has a checkpoint on this host. `scripts/bench_decode.sh` at pp512/tg128, `MLXCEL_FUSED_MOE_RELU2=1` on both sides, before and after alternated run by run, every run through `scripts/rocm_gpu_guard.sh`. Before, the flag declined to `gather_qmm`; after, it takes the HIP fc1 and down kernels.

| Model | Before tok/s | After tok/s | Change (medians) |
|---|---|---|---|
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit, `MLXCEL_FUSED_MOE_RELU2=1` | 75.15 / 75.22 / 75.09 (median 75.15) | 75.24 / 75.38 / 75.13 (median 75.24) | +0.1%, within noise |

This matches the Metal note on the flag: it replaces only the routed fc1 and fc2 GEMVs, which `gather_qmm` already runs near bandwidth. Raw rows: `benchmarks/rocm_strixhalo-gfx1151_2026-10-05_relu2-before.csv` and `..._relu2-after.csv`.

### 3.4 The relu2 greedy deviation

The issue asked for Nemotron-H's greedy 128-token output with the flag set to match the default path. On three prompts the texts agree for about 20 tokens and then part at a near-tie ("user query" against "user request", "checks if" against "checks whether"). The kernel keeps fc1, relu² and fc2 in f32 where `gather_qmm` rounds the intermediates to bf16, which is why it sits about a thousand times closer to the f32 reference (8.1e-6 against 5.1e-3 to 5.4e-3 nrms). Once the first token flips, free-running generation conditions everything after it on different text, so text equality is not a useful measure here. Teacher-forced `w1ctx512` traces (`python3 scripts/compare_logit_traces.py <reference> <candidate> --decided 2.0`):

| Reference | Candidate | Top-1 disagreement | Decided mismatches | Largest gap | Perplexity ref / cand |
|---|---|---|---|---|---|
| ROCm default | ROCm `MLXCEL_FUSED_MOE_RELU2=1` | 1 / 128 | 0 / 66 | 0.500 | 7.410 / 7.436 |
| Metal M5 default (`metal_m5_d1128266`) | ROCm `MLXCEL_FUSED_MOE_RELU2=1` | 5 / 128 | 0 / 71 | 0.250 | 7.489 / 7.436 |

The default path is unchanged: its `w1ctx512` trace on the branch is byte-identical in every data row to one built on main `6668031c`.

## 4. Review Findings Fixed in the PR

### 4.1 HIGH: the CPU device

Custom kernels run only on the GPU stream; on a CPU stream `eval_cpu` throws "Custom kernels only run on GPU". A port table answers "does this backend have the kernel", not "is the default device the GPU", so under `MLXCEL_DEVICE=cpu` a ROCm build would have built a custom kernel on the CPU stream in all four new paths. Each predicate now also requires `default_device() == Device::gpu`, as `ssm_kernel_available` and `mamba1_scan_kernel_accepts` already did.

The orchestrator then found the same exposure in #2065's `fused_moe_kernels_available()` and `moe_down_kernel_available()`: under `MLXCEL_DEVICE=cpu` SwitchGLU decode went into the HIP launch, whose `eval_cpu` throws after the bridge call has already returned `Ok`. Both now carry the device term, and Nemotron-H takes `forward_nonfused` on the CPU. Metal had the same exposure through `metal::is_available()` and backend-kind checks; its GPU behavior does not change.

`tests/cpu_device_custom_kernel_gates.rs` asserts all six predicates are false on the CPU device and runs `fused_xielu` (against the scalar formula) and `residual_add3_layer_norm` (against the unfused pair) on the CPU. It is a separate test binary because it moves the process-global default device, and the shared lib test binaries must not see that move. It fails with the device term removed.

### 4.2 MEDIUM: one hipRTC compile per activation size

The xIELU port first passed the element count `n` as a template argument, as Metal does. hipRTC keys its cache on template arguments and has no eviction, so every distinct activation size would compile and keep another kernel. On ROCm the kernel now reads `x_shape[0]`, giving one kernel per dtype. Metal's template arguments are unchanged.

### 4.3 Other items

- Mamba's gate lacked the N <= 32 check; it now goes through `mamba1_scan_kernel_accepts`.
- Tests that could pass without the kernel were tightened (the kernel-path assertions and visible skip in section 2.5).
- Comments no longer call the ROCm relu2 branch byte-identical, scope xIELU's identity to the dtypes tested per backend (Metal bf16; ROCm f32, f16, bf16; Metal f32 and f16 to a tolerance only), and say add3 identity is pinned by the test rather than by matching compiler flags.

## 5. Technical Decisions

- **Hold xIELU and add3 to the same backend's graph, not to the Metal kernel.** The Metal kernels are byte-identical to Metal's graph because they copy Metal's rounding points and reduction order. ROCm's graph has different ones (one rounding per elementwise op, the device `expm1f`, a 256-thread norm with `__shfl_xor` folds), so a faithful translation of the Metal kernel would have been byte-identical to nothing on ROCm. The add3 constraint (byte identity, no loosened tolerance) was met in all ten cases.
- **Port the float32-state Mamba1 variant.** Graph-exact output is unreachable on ROCm because the graph's `state @ C` runs in rocBLAS. The float32-state variant has a known contract (Metal's), passes the same f32 and bf16 tests, and lets Mamba, Falcon-Mamba and Jamba share one port.
- **Keep relu2 in f32 and accept the greedy deviation.** Matching `gather_qmm`'s text would require reproducing its bf16 intermediate rounding, which would move the kernel away from the f32 reference. The traces show the flips are near-ties with 0 decided mismatches against both ROCm default and Metal. The flag stays opt-in and the deviation is stated in the PR body.
- **Gate on device as well as table.** The #1801 rule (gate and dispatch read the same table) stays; the device term is the one fact the table cannot carry.
- **Read shapes, not template args, for sizes that vary per call on ROCm.** It bounds the hipRTC cache to one entry per dtype.
- **Reuse #2065's ROCm geometry for the relu2 branch.** Rows per block default to 2 so both kernels of the branch match the measured choice for the down kernel.
- **Drop the duplicate overlay fix on rebase.** #2100 had fixed the same by-value shape bug; keeping one copy avoids two divergent entries in `LOCAL_FIXES.md`.

## 6. Validation

On gfx1151, `--release --features rocm`:

- `make verify-rocm` on the final head: 147 suites, 11867 passed, 0 failed, 378 ignored (code head `800e23dc`; the later commits are documentation only).
- Kernel parity with negative checks as in section 3.2: xIELU byte-identical in three dtypes; add3 byte-identical in ten cases; `mamba1_scan_parity_tests` 5 passed; `fused_moe_relu2_parity_tests` within bounds.
- `cpu_device_custom_kernel_gates` passes, and fails with the device term removed.
- `models::mamba::` and `models::jamba::`: 22 passed, 2 ignored. Cohere2 model tests: 13 passed, 7 ignored.
- `cargo clippy -D warnings` on `mlxcel-core` and `mlxcel`: clean.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: pass, with the dtype-key pin at 9 in scope.
- Nemotron-H default `w1ctx512` trace byte-identical to main.

## 7. Residual Risks and What Was Not Verified

- **Metal and CUDA not run.** No hardware on this host. Metal's kernel sources and table entries are untouched, but Metal now runs the widened add3 and Mamba1 tests and the new relu2 and CPU-device tests for the first time, and its predicates gained the GPU-device term (GPU behavior unchanged; `MLXCEL_DEVICE=cpu` now falls back instead of throwing). On CUDA the fc1_relu2, xIELU, add3 and float-state Mamba1 tables stay null, and the add3 test compares the unfused pair with itself.
- **No end-to-end run for Apertus, Cohere2, Mamba, Falcon-Mamba or Jamba.** No checkpoints on the host, so decode throughput and model output for these families are unmeasured. They now run the new kernels by default on ROCm. For the Mamba families that means float32-state output, which differs from the previous ROCm graph scan.
- **The byte identity of xIELU and add3 depends on the ROCm compiler.** It is pinned by tests, not by matching compiler flags. A ROCm compiler or overlay upgrade that changes `expm1f` or the norm kernel would break it, and the tests would report it.
- **Wave64 (CDNA) untested.** The wave32 `#error` guard is inert because AMD clang 23 defines neither `__AMDGCN_WAVEFRONT_SIZE` spelling (#2098); correctness there rests on the explicit shuffle widths. The add3 port folds with `__shfl_xor(..., 32)` and assumes eight 32-lane groups per 256-thread block.
- **One device, one model measured.** Throughput comes from Nemotron-H on gfx1151 only.
- **A C++ warning** about a temporary destroyed at the end of an expression appears near `mlx_cxx_kernels.cpp:3346`. Whether it predates this PR was not checked.

Recommended follow-ups: a Metal run of the widened add3 and Mamba1 tests and the new relu2 and CPU-device tests; a Metal greedy check of the relu2 near-tie claim; a wave-size check for ROCm ports that does not rely on the inert `__AMDGCN_WAVEFRONT_SIZE` macro; checkpoints for Apertus, Cohere2, Mamba, Falcon-Mamba and Jamba on a ROCm host; triage of the `mlx_cxx_kernels.cpp:3346` warning.

## 8. Learning Points

- **Byte identity is a property of a backend, not of a kernel.** The same contract ("identical to the unfused graph") required different code on Metal and ROCm, because each backend's graph rounds in different places. A port that copies the other backend's kernel faithfully can still miss the contract.
- **A negative check has to be able to fail.** With N <= 16, lanes 16..31 of the Mamba1 fold hold zeros, so a fold that started at 8 still passed. Adding N = 32 made the check real (relative error 0.87).
- **A port table describes the backend, not the device.** `MLXCEL_DEVICE=cpu` keeps a GPU backend but moves the stream; a predicate that reads only the table opens a path that throws. The same gap existed in a merged predicate (#2065) and on Metal.
- **hipRTC template arguments are cache keys.** Anything that varies per call (an element count) belongs in a shape argument, or the cache grows without bound.
- **Free-running greedy text is a weak equality test for numerics changes.** One near-tie flip changes everything after it. Teacher-forced traces with a decided-gap threshold separate a numerics improvement from a defect.
- **Parallel units can fix the same bug.** #2100 and this branch both fixed the fork's by-value shape declaration; resolving to main's version on rebase kept a single fix.

Refs: #2069, #1814, #2065, #2098, #2100, #1981, #2005, #2007, #1801, #1813.
