# Technical Report: PR #2030 - Backend Quantization Capability Table and NVFP4 Load Policy

**Date**: 2026-09-29

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: Rust

**Risk Level**: Medium (changes the NVFP4 load route selection on every backend; Metal and CUDA are held to their previous routes by unit tests, not by a run on those backends)

## Executive Summary

Issue #1806 (phase 1 of epic #1801) asked for one place that answers "can this backend run quantization mode M natively?" and a load-time policy that converts or rejects based on the answer. This PR adds that table to `mlxcel_core::hardware`, keyed on the runtime `GpuBackendKind` instead of build features, and routes the ModelOpt NVFP4 repack through it. On ROCm, a ModelOpt NVFP4 checkpoint now loads with no environment variables by converting to affine 4-bit, where before it produced a group-16 native layout that the ROCm dispatch cannot run.

The same table backs a per-layer guard in every quantized layer loader, so an MLX-native NVFP4 export on ROCm now fails at load with the layer name, instead of aborting the process at the first forward pass. The two native-NVFP4 FFI tests skip on a backend the table marks non-native. That clears the `terminate called` abort that ended the `mlxcel-core` lib test binary on ROCm and hid every test sorting after it.

## 1. Problem Statement

Three problems, all rooted in the loader assuming the backend could run whatever mode it produced.

**The NVFP4 route was chosen by a build flag.** `current_nvfp4_repack_strategy()` read `cfg!(feature = "cuda")`. A ROCm build is not a CUDA build, so it took the default `DirectTranscode` route, which emits MLX native NVFP4 (group 16, E4M3 block scales). The ROCm qmv dispatch (`patches-rocm/mlx/backend/rocm/quantized/qmm.hip`) implements group sizes 32, 64 and 128 only and throws for anything else. The `DenseAffine` route that would work already existed, but only behind `MLXCEL_NVFP4_DENSE_REPACK=1`.

**The throw was an abort, not an error.** `quantized_matmul`, `gather_qmm` and the compiled MLP helpers do not return `Result` across the cxx bridge, so a C++ exception from the ROCm backend terminates the process. In production that meant a crash at the first forward pass with no mention of the layer or the mode. In the test gate, `ffi_tests::compiled_qgelu_mlp_global_scale_native_nvfp4_prefill_matches_reference` aborted the single-threaded `-p mlxcel-core --lib` run. Because libtest runs tests in name order, no test sorting after it (the rest of `ffi_tests`, then `hardware`, `layers`, `mla`, `paged_v2`, `sampling*` and more) had reached the runner on ROCm since the abort appeared. The issue records this as the last red item in `make verify-rocm` and the blocker for #1807, #1808 and #1809.

**Unconvertible layers were skipped silently.** When a ModelOpt layer could not be repacked (a non-scalar `weight_scale_2`, missing scales, an in_dim with no matching group), the loader printed "Skipping NVFP4 repack" and moved on. On a backend with native NVFP4 that is tolerable, since the untouched tensors were never going to be run through a missing kernel. On a backend that must convert, a skipped layer is either left in a layout the backend cannot run or silently wrong.

## 2. Change Summary

- `src/lib/mlxcel-core/src/hardware.rs`: `QuantMode` (the four modes MLX parses, with `as_str` and an exact `from_mlx_name`), `QuantModeSupport` (`Native`, `ConvertTo(mode)`, `Unsupported`), `GpuBackendKind::display_name`, `GpuBackendKind::quant_mode_support` (the table), and a free `quant_mode_support` that reads the running backend. Metal, CUDA and `None` (CPU) report every mode native. ROCm reports affine, mxfp4 and mxfp8 native and NVFP4 as `ConvertTo(Affine)`.
- `src/models/sanitize.rs`: `nvfp4_repack_strategy` takes a `GpuBackendKind` instead of `cuda_build` and returns `Result<Nvfp4LoadRoute, String>`, a struct carrying the backend, the strategy, a human-readable reason and a `conversion_required` flag. On ROCm the route is always `DenseAffine` with `conversion_required = true`; `MLXCEL_NVFP4_NATIVE_REPACK` is ignored there and the reason says so. `repack_nvfp4_weights_to_quantized` became fallible and is split so a test can inject a route (`repack_nvfp4_weights_with_route`). Every former "Skipping NVFP4 repack" site now goes through `nvfp4_layer_not_repacked`, which fails the load with the layer name and reason when conversion is required and keeps the old skip otherwise. A ROCm-only pre-check rejects a non-2-D weight, a non-U8 weight dtype, a scale shape that is not `[out_dim, num_groups]` and an in_dim that does not fit `i32`. The affine group check moved before the dense reconstruction. The single load log line now states the source mode, the target and the reason.
- `src/models/sanitize.rs`: the FP4 dequantization loop is now `dequantize_modelopt_nvfp4_rows`, which splits rows across up to 16 scoped threads, and the host byte buffers and the f32 matrix are dropped as soon as the next stage no longer needs them.
- `src/lib/mlxcel-core/src/layers.rs`: `validate_quantization_mode_runnable(mode, backend)` refuses any mode the table does not mark `Native`, with a message naming the mode, backend and remedy (re-quantize to affine with `mlx_lm.convert -q`, or use a backend with native kernels). It is called from `reconcile_quantization_layout_logged` (the shared dense and embedding loaders) and `QuantizedMultiLinear::from_weights`, and in the binary crate from `SwitchLinear`, gpt-oss `ExpertLinear` and kimi_linear's `MultiLinear`.
- `src/models/gemma4.rs`, `src/loading/vlm_gemma.rs`: propagate the now-fallible `sanitize_gemma4_nvfp4_weights`.
- `src/lib/mlxcel-core/src/ffi_tests.rs`: `backend_runs_native_nvfp4` gates the two native-NVFP4 tests on the table and prints the reason when skipping.
- Docs: the NVFP4 row in `docs/installation.md`, both NVFP4 variables in `docs/environment-variables.md`, and the `verify-rocm` ordering comment in the `Makefile`, which no longer cites the #1806 abort.

## 3. Technical Decisions

### The table is keyed on the runtime backend, not on build features

The issue's original plan left this open ("the backend kind from #1803 or the build features"); its 2026-09-29 refresh decided it, and the PR follows. A `cfg!` chain cannot describe a multi-backend build, and the WebUI catalog already paid for that mistake once (#1886, where every ROCm build was reported unsupported). `gpu_backend_kind()` reports the backend MLX actually resolved in this process, which is what decides whether a kernel exists at the moment the model runs.

This choice has one visible side effect, which the PR body states: a CUDA build that sees no device now runs as `None`. With `MLXCEL_NVFP4_DENSE_REPACK=1`, the dense route there used to target native NVFP4 (because the build was CUDA) and now targets affine (because the running backend is the CPU). Both run on the CPU backend, so this changes the output format but does not break anything.

### Three verdicts, not a boolean

`ConvertTo(mode)` exists because the epic's direction is "convert where possible, reject only where conversion is impossible". A boolean would force every caller to re-derive the conversion target. `Unsupported` has no row today, but the loader and the NVFP4 route both handle it with a distinct message, so a future backend row can use it without a code change elsewhere. `quant_table_conversions_land_on_native_modes` walks `GpuBackendKind::ALL` and asserts two structural rules: affine is native on every backend (it is the last-resort conversion target), and a conversion target is itself native on the same backend, so conversions never chain.

The mxfp4 entry on ROCm is `Native`, not the `ConvertTo(affine)` the issue originally proposed. The refresh marked that proposal stale: PR #1818 measured mxfp4 `quantized_matmul`, `gather_qmm` and GPU `quantize` against a dequantized f32 reference and ran gpt-oss-20b-MXFP4-Q4 coherently. #1808 still owns confirming the original hang does not reproduce.

### Convert-or-reject lives in two places, deliberately

The NVFP4 repack in `sanitize.rs` is the conversion: it runs before the model is built, on checkpoints it recognizes (the ModelOpt triplet with `weight_scale_2`). The per-layer guard in `layers.rs` is the rejection: anything that reaches a layer loader in a mode the backend cannot run has, by construction, taken no conversion route. Its main real case is an MLX-native NVFP4 export (`"mode": "nvfp4"` in `config.json`), whose layout has no load-time conversion. Putting the guard at the layer loaders rather than in one model-level check means MoE experts (`SwitchLinear`, gpt-oss, kimi_linear) and MLA (`QuantizedMultiLinear`) are covered through the same function, and the error names the exact layer.

The guard is kept out of the pure `reconcile_quantization_layout` so its shape tests do not depend on the host's backend. An unparseable mode is left to `validate_quantization_mode`, so the two checks never report the same string twice.

### Fail only where conversion is required

`nvfp4_layer_not_repacked` keeps the old skip on backends where NVFP4 is native, because there a skipped layer stays in a layout the backend can run, and changing that would be a behavior change on Metal and CUDA that the issue explicitly ruled out. On ROCm the same condition fails the load. The ROCm-only pre-check adds conditions the old code never tested (dtype, scale shape, 3-D weights) because on the conversion path those would have produced wrong affine weights rather than a skip.

### The tests hold the old routes, and the conversion path runs on every host

`nvfp4_repack_strategy_keeps_pre_1806_routes_where_native` embeds the previous decision function and compares it with the new one for every backend and every combination of the two env vars, wherever the table says NVFP4 is native. That is the evidence for the acceptance criterion "Metal and CUDA choose the same routes as before" in the absence of those backends on this host. Splitting out `repack_nvfp4_weights_with_route` lets tests inject a ROCm route on a Metal or CUDA host, so the conversion and its load errors are exercised on every CI runner rather than only on gfx1151.

### Threaded reconstruction, bit-identical

The dense f32 reconstruction of a large ModelOpt checkpoint was single-threaded and, on ROCm, now runs on every load rather than only under an env var. It is split by row ranges across scoped threads, each thread writing a disjoint slice, so the arithmetic per element is unchanged and a test holds the output bit-identical to the old serial loop. Measured on Gemma-4-E2B-it-NVFP4: 225 s load before, 210 s after, and identical `--temp 0` text.

## 4. Validation

Author's runs, from the PR body (gfx1151, `--features rocm`):

- `cargo test -p mlxcel-core --lib --profile test-fast --features rocm -- --test-threads=1` ran to the end with no `terminate called`: 1770 passed, 35 failed, 1 ignored. The 35 are listed in the PR body.
- `-p mlxcel --lib` for sanitize, nvfp4, gpt_oss, switch_layers and kimi_linear: 132 passed; every other model test module mentioning nvfp4: 621 passed.
- `clippy -D warnings` with `--features rocm --lib --tests` (plus `--examples` for the root crate) clean; `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt` passed.
- `bg-digitalservices/Gemma-4-E2B-it-NVFP4` (ModelOpt), no env vars: the load log names the affine route and the reason; `--temp 0` answers "The capital of France is Paris." at 25 tok/s.
- `mlx-community/gemma-4-e2b-it-nvfp4` (MLX-native): fails at load with `language_model.model.embed_tokens: quantization mode nvfp4 has no native kernel on the ROCm backend ...`.

Orchestrator verification (gfx1151, branch rebased onto origin/main `bfc2bfd9`):

- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke (32 tokens) all passed.
- `verify-test-rocm` failed in exactly two targets:
  - `-p mlxcel-core --lib`: 1779 passed, 35 failed, 1 ignored. The 35 are exactly the list in the PR body: 34 fused paged-attention tests without a ROCm port (tracked by #1814) plus the bf16 `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`. All of them were previously hidden behind the NVFP4 abort. The pass count is higher than the author's 1770 because the rebase onto `bfc2bfd9` brought in more tests.
  - `-p mlxcel --lib`: 8555 passed, 3 failed. The 3 are VLM detection tests broken on main by #2031 and are unrelated to this PR.
- All other test binaries passed.

The issue's gate criterion (`-p mlxcel-core --lib` finishes with no `terminate called`, and the full pass/fail list is recorded) is met. The target is still red, but the failures are now visible and belong to #1814 and one bf16 byte-identity case, not to this change.

## 5. Learning Points

- **An abort hides tests; it does not fail them.** With `--test-threads=1` and name-ordered execution, one uncatchable throw removes every later test from the report. "Not in the failure list" meant "never ran". Clearing the abort surfaced 35 failures that had existed for some time. When a gate target aborts, count the tests that reached the runner before treating the failure list as complete.
- **The cxx bridge turns a backend throw into process termination.** Any FFI call that does not return `Result` must be guarded on the Rust side before it is made. The capability table is that guard for quantization modes, and the per-layer check moves the failure from the first forward pass to load time, where the layer name is still known.
- **Decide on the runtime backend.** Build features describe what was compiled, not what is running. The same lesson was recorded in #1886 for the WebUI; this PR applies it to the load path.
- **Keep the old function in the test.** Embedding the pre-change decision function in a test and comparing it across the full input space is a cheap, strong way to prove "no behavior change" on backends the author cannot run.

## 6. What Is Not Verified

- **Metal and CUDA were not run.** Their NVFP4 route selection moved from a build flag to the runtime backend, and their layer loaders gained the mode check. The table reports every mode native on both, and unit tests hold their routes equal to the previous function, but no model was loaded on either backend for this PR.
- **The conversion-loss logit trace (acceptance criterion 4) was not measured.** The method needs a native-NVFP4 reference, which means Metal or CUDA. A CPU-device substitute did not finish its first 32-token chunk in 40 minutes, and a CPU-only build does not link on Linux (`copy_gpu_inplace` is undefined). The ROCm dense-affine trace was captured and can be compared once a Metal trace of the same checkpoint exists. Until then the only quality evidence for the conversion is greedy-decoding agreement on one prompt.
- **The bf16 byte-identity failure** in `prefill_dense_gemm_matches_qmm_bytes_where_eligible` was surfaced, not investigated. f16 passes. Whether it is a ROCm bf16 GEMM rounding difference or a real mismatch is open.

## 7. Remaining Work

- Record the conversion-loss numbers on a Metal host (native NVFP4 versus `DenseAffine` on Metal, then `DenseAffine` on ROCm), per the issue's step 5.
- #1807 (mxfp8) and #1808 (mxfp4) build on this table for their conversions; #1808 should confirm the mxfp4 `Native` entry on ROCm.
- #1814: ROCm ports of the fused paged-attention kernels, which account for 34 of the 35 remaining `mlxcel-core` failures.
- Triage the bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` failure on ROCm.
