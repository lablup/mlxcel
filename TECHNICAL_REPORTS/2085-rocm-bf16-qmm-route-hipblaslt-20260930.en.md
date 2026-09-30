# Technical Report: PR #2085 - Route large bf16 qmm to hipBLASLt and gate dense prefill on ROCm

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; head `c15ff36d` on origin/main `7278397a`, pending merge.

**Languages**: C++/HIP (ROCm overlay: `QuantizedMatmul` dispatch, exported route predicate), C++ (cxx bridge), Rust (dense-prefill eligibility, test), Markdown

**Risk Level**: Medium (the default kernel for every bf16 affine GEMM of 128 rows or more changes on the RDNA 3.5 tier, which covers prefill of every bf16-scale 4-bit checkpoint there; other ROCm tiers keep their dispatch unless an environment variable sets a ceiling, and on Metal and CUDA the new bridge call answers true, so their eligibility is unchanged)

## Executive Summary

Issue #2081 (part of epic #1801) was the last failure in `make verify-rocm` on gfx1151: the bf16 case of `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`. The test asserts that mlxcel's dense prefill path (`dequantize` then `matmul`) returns the same bytes as `quantized_matmul` wherever `prefill_dense_gemm_eligible` accepts a projection. On ROCm, bf16 `quantized_matmul` ran the fork's fused `qmm_wmma_dense_kernel`, which accumulates through rocWMMA tiles in its own K order, while the dense side ran hipBLASLt. 499 of 1,048,576 outputs differed. f16 passed because its `quantized_matmul` also dequantizes and calls hipBLASLt.

The issue offered three options: make the two sides byte-identical, exclude the ROCm bf16 case from eligibility, or replace byte identity with a ULP bound. The PR takes option 1, and does it by routing rather than by changing either kernel. An op-level measurement showed that the WMMA kernel is also the slower of the two on every bf16 shape measured from 128 rows up, so sending those GEMMs to dequantize + hipBLASLt makes both sides identical and speeds up production prefill. With the default route against the old one (`MLX_ROCM_WMMA_QMM=1`), Gemma 3 4B 4-bit prefill goes from 961 to 2815 tok/s at 2048 tokens.

The dispatch decision lives in one overlay function, `select_qmm_route`, which feeds both `QuantizedMatmul::eval_gpu` and an exported predicate, `rocm::quantized_matmul_runs_dequant_gemm`. The Rust eligibility asks that predicate through the bridge, so on ROCm the dense path runs only where `quantized_matmul` already runs the same GEMM. The unit's full `make verify-rocm` on `814f9efb` was the first fully green ROCm gate of the epic #1801 run.

## 1. Problem Statement

### 1.1 The failing assertion

```
panicked at src/lib/mlxcel-core/src/layers.rs:6369:17:
assertion `left == right` failed: dtype 12 bias false: dense GEMM must match qmm bytes
```

The issue measured the first failing case (bf16, no bias, x `[1, 1024, 2048]`, 4-bit affine weight `[1024, 2048]`, group 64): 499 of 1,048,576 outputs differ (0.048%). 465 are 1 ULP, 10 are 2 ULP, and 24 are 3 to 35 ULP, all on outputs below 0.01 in magnitude, where cancellation makes bf16 ULP distance large. The largest absolute error is 0.25 (41.5 against 41.75, 1 ULP) against a maximum |out| of 76.5, and one near-zero element crosses sign (-8.5e-6 against 1.6e-5, 28308 ULP by bit distance). The bias case was never reached. The failure had been hidden behind the NVFP4 abort that #1806 (PR #2030) removed, and had failed in every full ROCm gate since.

### 1.2 Why the premise held on Metal and not on ROCm

The eligibility rule came from #1994/#2001 (PR #2002): affine mode, f16 or bf16 input with scales in the same dtype, a 2-D weight, at least `min_rows` rows, and more than 512 output tiles of 32 x 32. Its byte-identity claim rests on an in-tree sweep that ran on Metal, where both sides dequantize with the same rounding and differ only in tiling. On ROCm the two sides reach different kernels:

- Dense side: `affine_dequantize`, then `matmul`, which runs hipBLASLt.
- qmm side, bf16: `qmm_wmma_dense_kernel` whenever the device has native WMMA and is not a low-CU iGPU, for bf16 x/scales/biases, group 64, 4/6/8 bits, `N % 16 == 0`, `K % 64 == 0`. Its dequantization matches `affine_dequantize`, but it accumulates in f32 through rocWMMA 16 x 16 x 16 tiles in its own K order.
- qmm side, f16: no WMMA path, so `affine_dequantize` plus `dequant_rocblas_gemm`, which is hipBLASLt too.

`MLX_ROCM_WMMA_QMM=0` made the test pass, which confirmed a reduction-order difference and not a dequantization one. The issue also ruled out the CPU-stream OpenBLAS miswrite fixed in #2079: every op in the test runs on the default GPU stream.

### 1.3 Production exposure before the fix

`prefill_dense_gemm_min_rows_default` in `hardware.rs` returns a threshold only on Apple M1. On ROCm the dense path runs only when an operator sets `MLXCEL_PREFILL_DEQUANT_MIN_M`, so the mismatch was a latent correctness hole behind an opt-in, not a default-path bug.

## 2. Why Option 1, by Routing

The issue required a decision among three options and forbade simply `cfg`-gating the assertion off. It also attached a condition to option 1: a fix is acceptable only if it does not regress bf16 prefill throughput on gfx1151, compared at the op level and on a real model.

The unit measured the two kernels before choosing. bf16, 4-bit g64, one `[1, M, K]` input against an `[N, K]` weight, mean of 40 calls per arm in two alternated rounds (`dense` re-dequantizes on every call):

| M | K x N | qmm ms | dense ms | dense / qmm | differ |
|---:|---|---:|---:|---:|---:|
| 16 | 4096 x 4096 | 0.388 | 0.337 | 0.87 | 79 / 65,536 |
| 16 | 4096 x 14336 | 1.329 | 1.649 | 1.24 | 261 / 229,376 |
| 64 | 4096 x 4096 | 0.413 | 0.488 | 1.18 | 305 / 262,144 |
| 64 | 14336 x 4096 | 1.472 | 1.995 | 1.36 | 1,217 / 262,144 |
| 128 | 4096 x 4096 | 0.805 | 0.570 | 0.71 | 572 / 524,288 |
| 128 | 14336 x 4096 | 2.519 | 2.479 | 0.98 | 2,309 / 524,288 |
| 256 | 4096 x 14336 | 4.022 | 2.274 | 0.57 | 4,153 / 3,670,016 |
| 512 | 4096 x 14336 | 9.187 | 3.769 | 0.41 | 4,242 / 7,340,032 |
| 1024 | 4096 x 1024 | 1.403 | 0.320 | 0.23 | 1,164 / 1,048,576 |
| 1024 | 4096 x 14336 | 15.775 | 4.644 | 0.29 | 12,251 / 14,680,064 |
| 2048 | 14336 x 4096 | 51.491 | 10.046 | 0.20 | 29,524 / 8,388,608 |

Across the full sweep of 50 cells (M 2 to 2048; K x N of 4096 x 4096, 4096 x 1024, 4096 x 14336, 14336 x 4096, 1024 x 3072), the WMMA kernel won only on some shapes at 64 rows or fewer, and dense was faster on every shape from 128 rows up (1.0x to 3.1x at 128 rows, 2.4x to 5.1x at 1024 and 2048, per the PR). Every cell also differed in bytes, so the mismatch is not specific to the test shape.

That measurement settled the choice, because correctness and speed pointed the same way:

- **Option 1 by routing** sends large bf16 GEMMs to the route that is both faster and byte-identical to the dense path. With the change, `quantized_matmul` differs from dense in 0 outputs of every cell from 128 rows up, and below 128 rows it is unchanged.
- **Option 1 by changing the WMMA kernel** (matching hipBLASLt's reduction order) would have kept the slower kernel on exactly the shapes where it loses, and it would have tied the kernel to hipBLASLt's internal reduction order, which is not a stable target across library versions.
- **Option 2** (exclude ROCm bf16 on WMMA devices from eligibility) would have made the test honest but left bf16 prefill on the slow kernel, and the opt-in dense path would never help a bf16 model on ROCm.
- **Option 3** (a ULP bound) would have given up the guarantee the dense path is built on, and the measured distribution makes a clean bound hard to state: 24 outputs beyond 2 ULP and one sign crossing at 28308 ULP bit distance mean the bound would need an absolute term fitted to near-zero outputs. The issue specifically warned against a bound chosen to make the test pass. It would also have kept bf16 prefill on the slow kernel.

Below 128 rows the kernel still wins on some shapes, so it stays there, and the dense path is refused there instead (section 3.3). That part is option 2 applied only where routing does not reach.

## 3. Change Summary

Three commits on `fix/issue-2081-rocm-bf16-dense-gemm`:

- **`a73370f2`** `fix(rocm): route large bf16 qmm to hipBLASLt and gate dense prefill`: the route function, the ceiling, the exported predicate, the bridge call, the eligibility change, the ROCm test block, the results page and the docs.
- **`814f9efb`** `fix(rocm): tighten the dense-prefill route check and its test`: review follow-ups. The predicate refuses one-row GEMMs (which `matmul` sends to gemv, not hipBLASLt) and takes a device index so the bridge checks the default GPU instead of device 0. The route-guard test asserts the expected eligibility of each shape and that at least one differing shape is refused, runs the route-independent assertions before any skip, and treats `MLX_NO_HIPBLASLT` as a route override.
- **`c15ff36d`** `docs(rocm): note the dequant cache footprint and more route overrides`: security-review follow-ups. `LOCAL_FIXES.md` item 29 documents the dequantized-weight LRU footprint, and the test's override list gains `MLX_ROCM_FORCE_LOW_CU` and `MLX_ROCM_FORCE_WARP_SIZE`.

Files by area:

- Overlay: `patches-rocm/mlx/backend/rocm/quantized/qmm.hip` (`QmmRoute`, `QmmRouteInputs`, `wmma_qmm_env`, `wmma_qmm_max_m`, `select_qmm_route`, `quantized_matmul_runs_dequant_gemm`, dispatch rewired), `rocm.h` (declaration), `no_rocm.cpp` (stub returning false), `LOCAL_FIXES.md` item 29.
- Bridge: `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.{h,cpp}` (`quantized_matmul_matches_dense_gemm`), `src/lib/mlxcel-core/src/lib.rs` (ffi declaration).
- Rust: `src/lib/mlxcel-core/src/layers.rs` (eligibility, doc comment, test), `src/lib/mlxcel-core/src/hardware.rs` (doc comment only).
- Docs: `docs/benchmark_results/rocm-bf16-qmm-route-gfx1151-2026-09-30.md`, `docs/environment-variables.md`, `docs/mlxcelverse/upstream/README.md` (item 29 not packaged yet).

### 3.1 One route decision

Before the PR, `QuantizedMatmul::eval_gpu` decided its route inline across three separate conditions: a WMMA block, a dequant-GEMM block, and inside it an fp8 block. The PR moves that into `select_qmm_route(const QmmRouteInputs&, rocm::Device&)`, which returns one of `WmmaDense`, `DequantFp8Gemm`, `DequantGemm` or `Other`, and `eval_gpu` now branches on the returned value. The WMMA shape test is the same conjunction as before. The change in behavior is one condition inside the WMMA branch:

```cpp
const bool hipblaslt_is_faster =
    env != 1 && dequant && in.M >= wmma_qmm_max_m(d) && !fp8();
```

When it holds, a GEMM that fits the WMMA kernel goes to `DequantGemm` instead. `wmma_qmm_max_m` returns `MLX_ROCM_WMMA_QMM_MAX_M` when that is a positive integer, else 128 on the `Rdna35` tier and `INT_MAX` everywhere else. Three guards keep the ceiling from reaching a route it was not measured against:

- `dequant` must hold: the dequant route must be available and enabled (`MLX_ROCM_QMM_DEQUANT_GEMM` not `0`) and preferred or forced for the shape. The ceiling never sends a GEMM to `Other`.
- `!fp8()`: where the fallback would be the e4m3 path (RDNA 4 with an fp8-capable hipBLASLt), the fused kernel stays, as before. The fp8 lambda is evaluated only where it decides the route, so a device that never reaches it is not probed.
- `env != 1`: `MLX_ROCM_WMMA_QMM=1` removes the ceiling, which is the old dispatch on any device that is not a low-CU iGPU (there it forces the kernel on, as it always did).

### 3.2 The exported predicate

`rocm::quantized_matmul_runs_dequant_gemm(device_index, M, N, K, x_dtype, scales_dtype, biases_dtype, group_size, bits)` rebuilds the `QmmRouteInputs` that `eval_gpu` derives for one transposed affine GEMM with no batch dimensions, including `should_use_dequant_gemm_path` and the `force_dequant_gemm` term for bit widths qmv does not support, and calls the same `select_qmm_route`. It returns true only for `QmmRoute::DequantGemm` with `is_hipblaslt_available()`:

- `M < 2` is refused because `matmul` sends a single row with a transposed weight to gemv, not hipBLASLt.
- `DequantFp8Gemm` is refused because the fp8 GEMM rounds the activation and weight to e4m3 and cannot match a bf16 matmul.
- Without hipBLASLt, `dequant_rocblas_gemm` and `matmul`'s rocBLAS path fall back to different rocBLAS calls, so only the hipBLASLt case is claimed.

Because both callers go through the same function, the predicate cannot drift from the dispatch as long as the inputs are rebuilt the same way. The environment overrides (`MLX_ROCM_WMMA_QMM`, `MLX_ROCM_WMMA_QMM_MAX_M`, `MLX_ROCM_QMM_DEQUANT_GEMM` and the rest) move both answers together.

### 3.3 The bridge and the Rust eligibility

`quantized_matmul_matches_dense_gemm` in the bridge answers true unless the build is a ROCm bridge (`MLXCEL_BRIDGE_ROCM_BACKEND`), the runtime backend is ROCm and the default device is a GPU. On ROCm it refuses any x with an axis above 1 before the last two, because `QuantizedMatmul` batches over those axes and a batched GEMM is not the single GEMM `matmul` runs on the same rows, then asks the overlay predicate with the default GPU's index. `no_rocm.cpp` stubs the predicate as false for fork builds without ROCm, and those builds never reach it because of the backend check.

`prefill_dense_gemm_eligible` keeps its tile rule and adds the bridge call as the last condition, after the cheaper row and tile checks return early. On Metal and CUDA the call returns true, so eligibility there is exactly the old rule. The doc comment now carries the #2081 evidence (mismatch count, ULP distribution, kernel paths) that the acceptance criteria required, and states the invariant the change enforces: the dense path is an optimization, so wherever it runs it must return the bytes `quantized_matmul` would have.

### 3.4 The test

The Metal-era test body is unchanged in intent. The route-independent assertions (the narrow-N refusal at exactly 512 tiles, and the min-rows refusal) now run first. The byte check at `[1, 1024, 2048]` still asserts eligibility for both dtypes, except that on a ROCm device other than the measured gfx1151 route, a refused shape logs and skips instead of failing.

A ROCm-only helper, `prefill_dense_gemm_rocm_route_guard`, checks the guarantee itself on shapes that pass the tile rule but reach different routes:

| dtype | rows | N | route on gfx1151 | expected eligibility |
|---|---:|---:|---|---|
| bf16 | 64 | 8448 | WMMA kernel (below the ceiling) | refused |
| f16 | 64 | 8448 | dequantize + hipBLASLt | accepted |
| bf16 | 256 | 4096 | dequantize + hipBLASLt (above the ceiling) | accepted |

N 8448 at 64 rows gives 2 x 264 = 528 tiles, just above the 512-tile floor, so the tile rule does not hide the route check. On every ROCm device the helper asserts that an accepted shape matches in bytes. On the measured route (`rocm_measured_route()`: gfx1151 and none of seven route-moving variables set) it also asserts each expected eligibility and that at least one shape really differed, so the guard cannot pass with the route check removed. With the route check bypassed, it fails on bf16 at 64 rows. Finally, x `[2, 512, 2048]` must be refused on ROCm because of its batch axis.

## 4. Results

### 4.1 Model prefill

`mlxcel-bench-decode --prompt-tokens {512, 2048} -n 8 --warmup-tokens 4 --ignore-eos` on gfx1151, one binary, `MLX_ROCM_WMMA_QMM=1` ("before", the old dispatch on this device) against the default ("after"), three ABBA runs per cell, with a sampler confirming no other GPU process ran. Prefill tok/s, mean (range):

| Model | Scales | pp512 before | pp512 after | pp2048 before | pp2048 after |
|---|---|---:|---:|---:|---:|
| gemma-3-4b-it-4bit | bf16 | 1146 (1046-1199) | 2289 (2197-2359) | 961 (948-972) | 2815 (2805-2824) |
| Qwen3-0.6B-4bit | bf16 | 4581 (4230-5024) | 7633 (7147-8012) | 3059 (2894-3219) | 4236 (4027-4566) |
| Qwen3-30B-A3B-4bit | bf16 | 311 (305-315) | 326 (321-334) | 283 (282-284) | 297 (296-299) |
| Meta-Llama-3.1-8B-Instruct-4bit | f16 | 969 (879-1032) | 1008 (994-1026) | 1149 (1143-1153) | 1132 (1124-1137) |

Computed from the means: Gemma 3 4B gains 2.0x at 512 tokens and 2.9x at 2048; Qwen3-0.6B gains 67% and 38%; Qwen3-30B-A3B gains about 5% at both lengths, because only its attention projections take this path and its experts go through `gather_qmm`.

Llama 3.1 8B is the control. Its scales are f16, so none of its GEMMs reaches the changed branch (f16 never takes the WMMA kernel), and its movement (+4.0% at 512 with overlapping ranges, -1.5% at 2048) has no code path to explain it. It bounds the run-to-run noise the bf16 gains should be read against.

Decode is unchanged in every cell: M is 1, which never took the WMMA kernel. MLX peak memory is within 0.1 GB of before, except Qwen3-0.6B at 512 tokens (1.04 to 1.46 GB), because the dequantize route allocates a bf16 copy of each weight matrix, which `LOCAL_FIXES.md` item 28 (PR #2084) bounds.

### 4.2 Production exposure: `MLXCEL_PREFILL_DEQUANT_MIN_M`

On ROCm the dense path runs only when `MLXCEL_PREFILL_DEQUANT_MIN_M` is set. After this change, the only projections it accepts there are ones where `quantized_matmul` already runs dequantize + hipBLASLt, so turning it on cannot change bytes. It also cannot change speed in any meaningful way: both paths run the same GEMM, and `quantized_matmul` does it with its dequantized-weight cache. Measured with `MLXCEL_PREFILL_DEQUANT_MIN_M=1024` against unset, same binary, pp2048, three ABBA runs: gemma-3-4b-it-4bit 2871 against 2852 tok/s, Qwen3-0.6B-4bit 4174 against 4185.

So the production picture is:

- **Default configuration (variable unset)**: the dense path never runs on ROCm, before or after. The user-visible effect of the PR is the faster `quantized_matmul` route for bf16 prefill.
- **Variable set**: before the PR, an operator could get bf16 outputs different from `quantized_matmul` (the #2081 mismatch) on any accepted projection. After it, the variable is a no-op for both bytes and speed. `docs/environment-variables.md` and the `hardware.rs` doc comment now say that setting it on ROCm gains nothing, and the ROCm default stays off.

## 5. Technical Decisions

- **Measure the kernels before choosing between the three options.** The issue's throughput condition on option 1 turned into the deciding evidence: the kernel that broke byte identity was also the slower one from 128 rows up.
- **Route rather than rewrite a kernel.** Moving a shape between two existing, already-shipped routes needs no new numerical code and makes identity hold by construction (the same GEMM on both sides), instead of by matching an opaque library's reduction order.
- **One function for dispatch and predicate.** A separate predicate that restated the dispatch conditions would drift the first time someone edited one side. Sharing `select_qmm_route` makes the Rust eligibility a question to the dispatch itself.
- **Claim only the hipBLASLt case.** The predicate refuses one-row GEMMs, the fp8 route, the no-hipBLASLt fallback, and batched inputs, because in each of those the two sides reach different code even when the high-level route name matches.
- **Scope the ceiling to the measured tier.** The 128-row default applies only to `Rdna35`. RDNA 3, RDNA 4 and CDNA keep the kernel at every row count unless `MLX_ROCM_WMMA_QMM_MAX_M` is set, and the fp8 fallback is excluded, so an unmeasured device cannot lose throughput by default.
- **Keep an exact switch for the old dispatch.** `MLX_ROCM_WMMA_QMM=1` restores the pre-PR route on this device, which is how one binary produced both columns of the prefill table.
- **Leave the ROCm default of `MLXCEL_PREFILL_DEQUANT_MIN_M` off.** With identical routes the dense path has nothing to add on ROCm, as 4.2 measured.

## 6. Validation

From the PR body, on gfx1151 (Radeon 8060S):

- The target test passes by default and under `MLX_ROCM_WMMA_QMM=0`, `MLX_ROCM_WMMA_QMM=1`, `MLX_NO_HIPBLASLT=1` and `MLX_ROCM_FORCE_LOW_CU=1`.
- `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings` passes.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt` passes, and `cargo test --features rocm --test dead_doc_pointers` passes.

Orchestrator verification:

- The unit ran the full `make verify-rocm` on `814f9efb` (gfx1151): 146 test suites, 11,785 passed, 0 failed, 378 ignored, smoke OK. That is the first fully green ROCm gate of the epic #1801 run; the previous baseline failure was this test (PR #2084's gate on `3c9edea0` failed in exactly this target and nowhere else).
- The head commit `c15ff36d` only adds a `LOCAL_FIXES.md` sentence and two environment names to the test's override list. The target test, clippy, `verify-rocm-overlay` and `verify-fmt` passed on it.
- The orchestrator runs a final gate on a fresh build after merge.

## 7. Residual Risks and What Was Not Verified

- **The route is judged at graph build.** `prefill_dense_gemm_eligible` asks the predicate when the graph is built, and `dequant_rocblas_gemm` reads hipBLASLt availability again at eval. A stream capture starting in between would send both sides to different rocBLAS fallbacks. HIP graphs are off on this backend (`use_hip_graphs()` returns false), so this needs `MLXCEL_PREFILL_DEQUANT_MIN_M` set during a decode capture. The predicate also assumes the hipBLASLt launch itself does not throw; if it did, each side would take its own rocBLAS fallback.
- **Dequantized-weight cache footprint.** The route's LRU (8 matrices or 256 MB, `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE`, `MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES`) gets no hits across a model's projections in prefill, so each projection is dequantized again per prefill chunk, and the last entries (up to 256 MB) stay alive after prefill. f16 checkpoints already behaved this way; bf16 checkpoints on RDNA 3.5 now do too. `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=0` turns it off. The transient copies grow with projection size; no dense bf16-scale checkpoint above 4B was available on the host, so peak memory for one was not measured.
- **Other RDNA tiers were not measured.** The ceiling applies to the whole `Rdna35` tier (`gfx1150` to `gfx1152`) but was measured on gfx1151 only; low-CU gfx1152 parts skip the WMMA kernel anyway unless forced. RDNA 3, RDNA 4 and CDNA keep their previous dispatch. On a ROCm device other than gfx1151, the test's byte check skips a refused shape instead of failing, so coverage there is weaker.
- **Rows between 64 and 128.** The published table has no cell between those row counts; 128 is the first measured count at which dense won on every shape.
- **Metal and CUDA were not run on this host.** The Rust eligibility and test changed on those paths, but the bridge returns true there and the ROCm block is gated to ROCm.
- **Upstreaming.** Item 29 is not packaged for the fork: the ceiling was measured on one device, and the exported predicate exists for mlxcel's dense-prefill check. A fork PR would carry the ceiling alone, with measurements from at least one more RDNA 3 or RDNA 3.5 device.

## 8. Learning Points

- **A byte-identity claim is a claim about kernels, not about math.** The #2001 sweep was valid on Metal. On another backend the same two ops reached different kernels, and the premise needed a backend-level check rather than a tile rule.
- **When a correctness fix and a performance question share a knob, measure both first.** Here the evidence that fixed the test also closed a 2.9x prefill gap the issue had not asked about.
- **Make a predicate about dispatch ask the dispatch.** Exporting a function that runs the same route selection, instead of mirroring conditions in Rust, is what keeps the eligibility honest when environment overrides or future tiers move the route.
- **Give a guard test a mutation it must catch.** The route-guard asserts that at least one shape really differs on the measured route, so deleting the route check fails the test instead of silently passing it.

Refs: #2081, #1801, #1994, #2001, #2002, #1806, #2030, #2079, #2084.
