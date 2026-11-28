# Technical Report: PR #2057 - Committed mxfp4 Regression Tests and a Native-Route Load Log on ROCm

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: Rust (tests, loader helper), Markdown

**Risk Level**: Low (no kernel, overlay or build change; one new integration test gated on `feature = "rocm"`, one load-time helper that adds a stderr line on ROCm only, and a doc update. The loader call sites are shared with Metal and CUDA, which were not run)

## Executive Summary

Issue #1808 (Phase 2 of epic #1801) was filed for three mxfp4 failures on gfx1151: `quantized_matmul` hanging at 256x512 with M = 1, GPU `quantize` failing at 4096x4096 with `hipLaunchKernel(...) failed: invalid configuration argument`, and a GPU memory fault in `qmv_warp_shared_kernel`. It proposed a kernel fix or, failing that, a load-time conversion of mxfp4 to affine 4-bit.

The PR's audit found that none of these failures reproduces on main (`dfc59867`). They were fixed by PR #1818's overlay items 8, 10 and 11 in `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`, which landed a few hours after the issue was filed, but nothing committed checked them. The PR therefore changes no kernel. It commits `tests/rocm_mxfp4_quant.rs`, which checks GPU `quantize`, `quantized_matmul` and `gather_qmm` in mxfp4 against CPU references and pins the capability table's `Native` entry. Each numeric test was seen to fail with its overlay fix reverted: reverting item 8 gives NaN, reverting item 10 gives a GPU memory fault that is now reported as an error rather than a hang (since #1804), and reverting item 11 gives the original "invalid configuration argument". The affine 4-bit fallback the issue kept in reserve is not needed.

The PR also closes the issue's load-log criterion: on ROCm the first quantized layer of each natively run block-float mode prints `Quantization mode mxfp4: running on native ROCm kernels, no load-time conversion ...`, the native counterpart of the NVFP4 repack line. Metal, CUDA and CPU print nothing new.

## 1. Problem Statement

### The issue described failures that were already fixed

The issue was filed against the overlay before PR #1818 vendored it. By the time work started, its own refresh log (2026-09-29) recorded that PR #1818 reported mxfp4 correct, but also that no `LOCAL_FIXES.md` item named a hang fix, so which change removed the hang was not on record. The plan's first step was rewritten as "reproduce first": run the exact hang case under a timeout, and only root-cause it if it still hangs.

The audit on gfx1151 at `dfc59867`, each case under `timeout`:

- mxfp4 `quantized_matmul` at 256x512, M = 1: finishes and matches `x @ dequantize(w).T` computed in f32 on the CPU (relative error 1.8e-7 for f32, 2.8e-3 for bf16).
- GPU mxfp4 `quantize` at 4096x4096: launches in about 80 ms and matches the CPU quantizer bit for bit.
- mxfp4 `gather_qmm`: matches a per-expert dequantized reference, sorted and unsorted (1.6e-6 f32, 2.6e-3 bf16).

Mapped to the overlay:

| Issue symptom | Overlay fix | Root cause |
|---|---|---|
| qmm hang at 256x512, M = 1; memory fault in `qmv_warp_shared_kernel` | Item 8 | qmv dispatch used the activation dtype as the scale type, but mxfp4 stores one E8M0 byte per group, so kernels read scales at 2 to 4 times the real stride (NaN and out-of-bounds reads) |
| MoE experts through `gather_qmm` | Item 10 | Non-affine modes reached `gather_qmv_kernel` instantiated with `AFFINE=true` and T-typed scales |
| `quantize` "invalid configuration argument" at 4096x4096 | Item 11 | The failing launch was `arg_reduce`, not `fp_quantize.hip`: one 1024-thread block per output on a 1-D grid exceeded AMD's 2^32 - 1 threads per grid dimension for a 16.7M-element argmin |

### Nothing committed held the fix in place

PR #1818 ran its op checks ad hoc. `ffi_tests` had four mxfp4 group-32 cases, but at 128 and 256 wide they sit below the 256x512 hang shape, none covers `gather_qmm` numerically, and until #1806 two of them sorted after an NVFP4 abort and never ran in the gate. A fork sync that dropped any of items 8, 10 or 11 would have gone unnoticed until a user loaded gpt-oss.

### The load log did not say which route ran

After #1806 the backend capability table decides, per mode, whether a checkpoint runs natively, is converted, or is refused. A converted NVFP4 checkpoint logs its repack route. A native mxfp4 load on ROCm logged nothing, so a load log could not show whether gpt-oss ran natively or through a conversion. The issue's first acceptance criterion asks for exactly that line.

## 2. Change Summary

- **`tests/rocm_mxfp4_quant.rs`** (new, 421 lines, `#![cfg(feature = "rocm")]`, each test skips off ROCm and holds `streams::lock_default_device` for its body).
  - `capability_table_reports_mxfp4_native_on_rocm`: `quant_mode_support(Mxfp4) == Native`. Its doc says the entry must become `ConvertTo(Affine)` if the numeric tests fail for good.
  - `gpu_quantize_matches_cpu_bitwise`: 256x512 and 4096x4096, f32 and bf16. Checks packed and scale shapes, then compares GPU bytes against the CPU quantizer on row bands (every row at 256x512; the first and last 64 rows at 4096x4096, which catch both a failed launch and a grid that stops short of the tail).
  - `quantized_matmul_matches_dequantized_reference`: shapes 256x512 (the hang case), 512x256 and gpt-oss-20b's 2880x2880; f32, f16, bf16; M = 1 (qmv) and 8, 64 (qmm). Reference is `x @ dequantize(w).T` in f32 on the CPU.
  - `gather_qmm_matches_per_expert_reference`: 16 experts, N = 256, K = 2880 (gpt-oss's reduction width, 90 groups of 32), top 4; T = 1 and 8; f32 and bf16; both the unsorted (`[T,1,1,K]`, indices `[T, top_k]`) and sorted (`[T*top_k,1,K]`, indices ordered by expert) shapes that `SwitchLinear::forward` passes.
  - Every result goes through `try_eval`, so a GPU failure is a test failure with the HIP status. Tolerances are 1e-4 (f32), 5e-3 (f16), 2e-2 (bf16), relative to `max |want|`.
- **`src/lib/mlxcel-core/src/layers.rs`** (+114/-).
  - `native_quantization_route_line(mode, backend) -> Option<String>`: a line only for a parsed block-float mode that is `Native` on a backend where some other mode is not. Today that means mxfp4 and mxfp8 on ROCm.
  - `validate_quantization_mode_for_running_backend(mode)`: runs `validate_quantization_mode_runnable` against `gpu_backend_kind()`, then prints the route line to stderr once per mode per process, using a static `[AtomicBool; QuantMode::ALL.len()]`.
  - The dense and embedding loaders (`reconcile_quantization_layout_logged`) and `QuantizedMultiLinear::from_weights` call the new function.
  - Unit test `native_route_line_only_where_a_route_was_chosen` iterates every backend and mode and asserts no line on Metal, CUDA or CPU.
- **`src/models/{gpt_oss,kimi_linear,switch_layers}.rs`**: the three model-crate loaders (`ExpertLinear`, `MultiLinear`, `SwitchLinear`) switch to the same function.
- **`src/lib/mlxcel-core/src/hardware.rs`**: the `quant_mode_support` doc names the committed test as the evidence for the ROCm mxfp4 entry. The table itself is unchanged.
- **`docs/installation.md`**: the ROCm status table names the log line and the test. The gpt-oss decode figure goes from 3.6 tok/s (PR #1818) to about 8 tok/s (8.07 in the #2056 baseline).

Commits: `c273284c` adds the tests, the route line and the docs; `4785419d` is a review follow-up that removes an `expect` from the once-per-mode slot lookup and rewords the `hardware.rs` comment to say the revert checks were run while #1808 was worked, not that the repository reproduces them.

## 3. Technical Decisions

### Audit before fixing

The issue's two top-level remedies (root-cause the hang, or add an affine fallback) both assumed the failure still existed. Re-running the exact cases first turned a kernel investigation and a load-time repack with its memory cost into a test and a log line. The affine fallback in plan step 4 stays unbuilt, and the capability table keeps `Native`.

### Prove each test catches the defect it names

A test that passes on fixed code shows nothing unless it also fails on broken code. Each of items 8, 10 and 11 was reverted in turn, and the matching test failed: NaN for item 8, a GPU memory fault for item 10, and the launch error for item 11. The item 10 result differs from the failure recorded in `LOCAL_FIXES.md` (wrong numerics with relative error up to 1e34). The likely reason is that at these shapes the misread T-typed scale stride runs out of bounds before it can produce garbage; this was not traced further. Because #1804 makes a queue fault an error on the waiting event, this surfaces as a test failure instead of a hung process. These reverts were done during development and are not scripted in the repository. The review follow-up changed the `hardware.rs` wording so it does not imply otherwise.

### Keep CPU reference cost bounded

The CPU side is the slow half. MLX's CPU quantizer takes about 90 s for 4096x4096 on this host, and its CPU random generator about 16 s for 16M samples. The tests generate inputs on the GPU and compare the same evaluated bytes on both devices. They quantize only row bands of the large matrix on the CPU, which works because quantization is independent per row group. The `gather_qmm` output width and expert count are cut to 256 and 16 while K stays at gpt-oss's 2880. The full gpt-oss shape is covered end to end by the real checkpoint instead.

### Log the route only where a route was chosen

A line on every backend would change Metal and CUDA load output for no information, since they run every mode natively. The condition is structural rather than a `backend == Rocm` check: log a parsed non-affine mode that is `Native` on a backend where at least one mode is not. If a future backend gains a conversion entry, its native block-float loads start logging with no code change. Affine is every backend's baseline and never logs. Once per mode per process keeps a many-layer MoE load to one line.

### One entry point for all quantized loaders

Five call sites in two crates each called `validate_quantization_mode_runnable(mode, gpu_backend_kind())`. They now call one function that validates and logs. The runnable check still takes an explicit backend, so its unit tests do not depend on the host.

## 4. Validation

PR author (gfx1151):

- `cargo test --profile test-fast --features rocm --test rocm_mxfp4_quant -- --test-threads=1`: 4 passed, 26 s.
- `mlxcel generate -m models/mlx/gpt-oss-20b-MXFP4-Q4 --temp 0`: coherent answer ("The capital of France is Paris."), and the native-route line appears once. Qwen3-0.6B-4bit (affine) prints no line.
- mlxcel-core `ffi_tests::compiled_qgelu_mlp_global_scale` (5 passed, including the two mxfp4 cases the NVFP4 abort used to hide). `layers::tests`: the new test passes. The only failure is the known bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`. Root lib `models::{switch_layers,gpt_oss,kimi_linear}`: 40 passed. `dead_doc_pointers`: passed.
- clippy `-D warnings` on mlxcel-core and mlxcel, and `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: clean.

Orchestrator verification (gfx1151, branch on origin/main `dfc59867`):

- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke (32 tokens) passed.
- `verify-test-rocm` failed in three targets with exactly the 37 known baseline failures and nothing else. 35 are in `-p mlxcel-core --lib`: 34 fused paged-attention tests with no ROCm port yet (#1814), plus bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`. The other two come from #2037: `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit` and `tests::family_order_is_exhaustive`.
- `tests/rocm_mxfp4_quant.rs` ran inside the full suite: 4 passed in 26.0 s.
- mlxcel-core lib passed 1797, one more than before (the new route-line test).

## 5. Learning Points

- **Reproduce a stale issue before working it.** The issue carried three failures, a plan to bisect a hang, and a conversion fallback. All of it was already obsolete, and the only record of why was a PR that landed hours after the issue was filed. The reproduce-first step, which the issue refresh added, cost a few minutes and removed a kernel investigation. The remaining gap was different from what the issue named: the fix was real but unguarded.
- **A regression test is proven by reverting the fix.** Running each test against a reverted overlay item is what shows it guards that item. A shape or dtype that happens to miss the defect (the old `ffi_tests` cases at 128 and 256 wide) passes either way.
- **A symptom can change with the reporting path.** Reverting item 10 produced a memory fault, not the wrong numbers `LOCAL_FIXES.md` records, and after #1804 that fault surfaces as an error instead of a hang. When a revert check reports a different failure than the history, check which layer reports it before concluding the test is wrong.
- **A slow CPU reference can look like a hung GPU.** During the audit a debug-profile test build appeared to hang. The GPU work had finished; the time was going to the CPU reference code (CPU random generation, quantization, dequantization and the f32 reference matmul), which unoptimized builds make far slower. For an issue whose headline is a GPU hang, this is an easy false positive. The tests are run under `--profile test-fast` (release-derived, `opt-level = 3`), generate inputs on the GPU, and limit the CPU reference to row bands. The module doc still advises running under a timeout when bisecting, since a real hang blocks the test.
- **Log a decision where one was made.** A route line on backends with no choice adds noise. Deriving the condition from the capability table ties the log to the table.

## 6. Caveats and What Is Not Verified

- **Metal and CUDA** were not run. The five loader call sites are shared, but `native_quantization_route_line` returns `None` for every mode on those backends. A unit test covers this for every backend, so their behavior is unchanged by construction.
- **Revert evidence is not in the repository.** The three revert checks were done by hand during the work. A future overlay sync that drops an item is caught by the committed tests, but the claim that each test catches its item rests on the PR record.
- **Coverage bounds.** Only group size 32 is tested, which is the only fp group size `gather_qmm` accepts on ROCm; others throw. `gather_qmm` is tested at N = 256 and 16 experts, not gpt-oss's full expert shape. The expert-batched gather path (item 9, opt-in via `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=1`) is not exercised.
- **Only gfx1151** was run.
- **Speed is out of scope.** gpt-oss decodes at about 8 tok/s through the generic `gather_qmm` kernel, because the fused MoE launchers have no ROCm port. That is #1814's concern.

## 7. Remaining Work

- #1814: ROCm fused MoE and paged-attention ports. The gpt-oss decode rate and the 34 fused paged-attention baseline failures belong there.
- #1813: upstream items 8, 10 and 11 to the fork. Until then the committed tests are what keeps a fork sync from silently dropping them.

Refs: #1808 (closed by this PR), #1801, #1804, #1806, #1813, #1814, #2037, PR #1818, PR #2056.
