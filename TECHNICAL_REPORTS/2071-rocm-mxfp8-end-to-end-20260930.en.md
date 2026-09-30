# Technical Report: PR #2071 - mxfp8 End to End on ROCm, Including FP8 Block Checkpoints

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: Rust (tests, example), C++ (ROCm overlay), Markdown

**Risk Level**: Low (one four-line guard in the ROCm overlay's SDPA fallback decision that only affects CPU streams, one example binary change, new tests, doc comments, benchmark traces and docs; no library code path changes on Metal or CUDA, which were not run)

## Executive Summary

Issue #1807 (Phase 2 of epic #1801) asked for proof that mxfp8 works on ROCm on the paths mlxcel uses. mxfp8 matters beyond native mxfp8 checkpoints: every vendor FP8 block checkpoint is requantized to mxfp8 at load by `src/models/fp8_block.rs`. The kernel fixes were already in place (PR #1818, `LOCAL_FIXES.md` items 8 and 10), but nothing committed checked `gather_qmm` numerically in mxfp8, no FP8 block checkpoint had been run end to end on ROCm, and the question of whether ROCm's non-bit-identical `quantize` matters at load was open.

The PR closes all three. `src/models/switch_layers_mxfp_tests.rs` checks mxfp8 and mxfp4 `gather_qmm` (through `SwitchLinear::forward`) and `quantized_matmul` (through `UnifiedLinear::forward`) against a reference decoded and accumulated in f64 on the host. The gather cases reach every branch a block-float mode can take in the ROCm `GatherQMM::eval_gpu`, and both gather tests fail with item 10 reverted. A dense vendor FP8 block checkpoint (`Lazarus-Ai/ReAligned-Qwen3.5-0.8B-FP8`) was traced on ROCm against the CPU device, standing in for Metal, which this host lacks: 0 of 50 decided positions disagree. GPU-quantized and CPU-quantized weights disagree at 0 of 346 decided positions, so load-time quantization stays on the GPU.

Building the CPU reference exposed two defects, both fixed here. `examples/logit_trace` ignored `MLXCEL_DEVICE`, so a "CPU" trace silently ran on the GPU. And on a ROCm build, `MLXCEL_DEVICE=cpu` aborted every attention model with `NYI` because the overlay's `ScaledDotProductAttention::use_fallback` never looked at the stream device (new `LOCAL_FIXES.md` item 23, guarded by `tests/cpu_device_sdpa.rs`).

## 1. Problem Statement

### The MoE path had no committed numeric check

The feasibility spike found mxfp8 `quantized_matmul` returning NaN on ROCm, caused by the qmv dispatch reading one-byte E8M0 scales at the activation dtype's stride (item 8). While validating, PR #1818 found a second, different bug on the gather path: non-affine modes reached a `gather_qmv_kernel` instantiated with `AFFINE=true`, so fp weights were dequantized with the affine formula (relative error 1.0 to 1e34, item 10). PR #1818 checked both with an ad-hoc probe. A search of `src/`, `tests/` and `examples/` found no committed test that checked `gather_qmm` numerically in mxfp8; the mxfp8 cases in `switch_layers.rs` and the model tests only validated loader rejection. PR #2057 had since committed mxfp4 checks in `tests/rocm_mxfp4_quant.rs`, but not mxfp8, and not through the production layers.

Fused MoE launchers have no ROCm port (`.rocm = nullptr` in the port tables) and `fused_moe_enabled()` requires custom kernels, which ROCm lacks, so the MLX graph path through `gather_qmm` is what an mxfp8 MoE actually runs on ROCm. That is the path the check had to cover.

### No FP8 block checkpoint had run on ROCm

The gfx1151 correctness matrix (PR #1826) traced affine 4-bit only and assigned mxfp8 coverage to this issue. The issue's acceptance criterion asked for a decided-position mismatch rate against Metal within #1809's `--decided 2.0` threshold.

### GPU `quantize` is not bit-identical to the CPU

ROCm `quantize` produces the same E8M0 scales as the CPU but rounds ties differently, changing about 3.3% of weight bytes by one step with the same RMS error. The issue asked whether that moves decided positions: if it did, requantization at load should move to the CPU stream; if not, the difference should be documented.

## 2. Change Summary

- **`src/models/switch_layers_mxfp_tests.rs`** (new, 540 lines, included from `switch_layers.rs` as `#[cfg(test)] mod mxfp_tests`, so it can reach the crate-private `SwitchLinear` and `gather_sort`).
  - Host reference: decodes packed codes (E4M3 for mxfp8, E2M1 for mxfp4) and E8M0 scales (`2^(e - 127)`) in Rust and accumulates in f64 from the activations as the device holds them. It shares no kernel with the backend under test.
  - `mxfp_host_decode_matches_mlx_dequantize`: pins the host decoder bit for bit to MLX `dequantize`, so a wrong reference cannot pass silently.
  - `mxfp8_gather_qmm_matches_host_reference`, `mxfp4_gather_qmm_matches_host_reference`: five gather cases (unsorted decode `B = 4`, unsorted prefill `B = 24`, sorted prefill `B = 128`, sorted shared-activation `B = 32` with activation stride 0, and multi-row `M = 4`) over bf16, f16 and f32, with `N = 320` so the second column block runs its bound check.
  - `mxfp8_quantized_matmul_matches_host_reference`: dense rows `M = 1, 4, 64`.
  - `mxfp_matmuls_match_host_reference_on_cpu_device`: a reduced matrix on the MLX CPU device (both modes, unsorted prefill, sorted shared-activation and multi-row gather, dense `M = 1` and `4`, bf16 and f16, 64-wide output), in about 2.5 s.
  - Bounds (relative L2): 2e-2 (bf16), 4e-3 (f16), 2e-3 (f32). The tests are not backend-gated and hold `lock_default_device`.
- **`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/scaled_dot_product_attention.cpp`** (+12): `use_fallback` returns `true` for any non-GPU stream before its shape checks, and throws `invalid_argument` if `force_fused` is set there. `LOCAL_FIXES.md` item 23 records it.
- **`tests/cpu_device_sdpa.rs`** (new, 138 lines): runs attention on the CPU device, including a decode-shaped query, against a host softmax reference. It aborts on the pre-fix overlay.
- **`examples/logit_trace.rs`** (+10): calls `mlxcel::initialize_runtime_checked()` so `MLXCEL_DEVICE` is honoured, warns on an invalid override, and writes a `# device` header line into every trace.
- **`src/models/fp8_block.rs`** (doc only): explains why requantization stays on the default device, with the measurements.
- **`src/lib/mlxcel-core/src/hardware.rs`** (doc only): the `quant_mode_support` evidence list cites the mxfp8 tests and the checkpoint run. The table already had ROCm mxfp8 as `Native`; nothing in it changes.
- **`docs/benchmark_results/rocm-fp8-block-gfx1151-2026-09-30.md`** (new) and **`benchmarks/logit_traces/fp8_block_gfx1151/`** (new): the write-up, eight trace files, `METADATA.txt`, `RUNS.txt` and `SHA256SUMS`.
- **`docs/installation.md`**: the ROCm status table adds FP8 block checkpoints to the mxfp8 row and a new row for the CPU device on a ROCm build ("runs, but slowly"). **`rocm-correctness-gfx1151-2026-09-12.md`**: its stale mxfp8 gap line points to the new run.

Commits: `a9a05ac5` adds the tests, the SDPA guard and the `logit_trace` fix; `35375780` adds the benchmark results and traces; `c26086b2` makes the sorted shared-activation case actually reach the stride-0 schedule and widens the f32 bound for TF32; `9d1f570e` shrinks the CPU arm from 38 s to 2.5 s and adds SAFETY comments.

## 3. Technical Decisions

### Test through the production layers, in the lib target

The tests call `SwitchLinear::forward` and `UnifiedLinear::forward` rather than the bare `gather_qmm` and `quantized_matmul` FFI. The gather inputs come from `gather_sort`, the same helper `SwitchGLU` uses, so the sorted and unsorted shapes are the ones the model code actually produces. That needs crate-private items, which is why the file lives in the `mlxcel` lib target as a `#[path]` module instead of under `tests/`. A side effect worth knowing: the existing `fp8_block` round-trip test in the same target already runs on the ROCm default device, which is why plan step 4 needed no separate ROCm round-trip test.

### A host-decoded f64 reference, pinned to MLX

PR #2057 compared against MLX's CPU backend. Here the reference is decoded in Rust. MLX's CPU `fp_qmm_t` accumulates in the activation dtype, so in bf16 it is the least accurate backend measured (8.2e-3 against ROCm's 1.8e-3), and it is scalar and slow. A host f64 reference is exact up to the device's input rounding and independent of every kernel under test. The risk of a hand-written decoder is that it is wrong in the same way as the kernel; the separate bit-for-bit test against MLX `dequantize` closes that.

### Enumerate the branches the test reaches, then check the claim

`GatherQMM::eval_gpu` gates every specialised path (grouped WMMA prefill, expert-batched, tiled, wide, idot, warp-shared) on `mode_ == Affine`, so a block-float mode reaches exactly one launch, the item 10 `gather_qmv_kernel<T, uint8_t, BITS, 32, false>`. The case list was built to hit every arm inside it: three `T`, two `BITS`, `implicit_lhs` false and true, and for the sorted-rhs schedule both activation strides (`K` and 0). The first version's shared-activation case used 16 slots over 8 experts, but `use_sorted_rhs_schedule` needs `B / E >= 4`, so the stride-0 branch the module doc claimed was never exercised. Commit `c26086b2` raised it to 32 slots. The opt-in expert-batched kernel (item 9) is affine-only and unreachable from these modes, which answers the issue's "default or opt-in path" question: the default.

### Bounds that hold on every backend

The tests are not gated on ROCm. The bf16 and f16 bounds come from activation-dtype rounding, about 2.4 times above the least accurate correct backend seen (the MLX CPU backend). The f32 bound started at 1e-5 but would fail on CUDA sm80 and later, where the sorted prefill case (`B >= 8 * E`) takes MLX's grouped GEMM and `MLX_ENABLE_TF32` (on by default) rounds f32 activations to a 10-bit mantissa. It is now 2e-3, still three orders of magnitude below the item 10 defect (1.0 and above). The CPU arm exists to show the bounds are not fitted to one GPU; it was cut to a reduced matrix because the full one held the global default-device lock for 38 s on every backend.

### The CPU device as the reference, since Metal is not available

The acceptance criterion names Metal. The host has no Metal, and the official Qwen3.5 FP8 releases (27B at 31 GB, 35B-A3B MoE at 37.5 GB) leave no room for a CPU reference on a 30 GiB host. The community `ReAligned-Qwen3.5-0.8B-FP8` checkpoint uses exactly the vendor layout the loader converts (`quant_method: fp8`, 128x128 blocks, `*.weight_scale_inv`), so it exercises the same load path. It is dense, so it covers requantization and dense mxfp8 `quantized_matmul`, while the op tests cover the MoE gather path. The CPU backend runs this checkpoint at about 2.7 minutes per traced token, so the CPU arm is one width (`w32`, 256 positions, 50 decided), assembled from eight parallel processes. The CPU is a weaker reference than Metal because it accumulates in bf16.

### Keep load-time quantization on the GPU

The effect of tie rounding was isolated by tracing on the GPU twice: once as shipped, and once with `requantize_block_fp8_weights` temporarily quantizing on the CPU stream (a local switch, not committed). Both arms compute on the GPU, and the ROCm traces are byte-identical on rerun, so every difference comes from the packed weights. Across `w1`, `w8` and `w256`, 0 of 346 decided positions disagree; all 42 top-1 disagreements sit at reference gaps of 0.375 or less. Moving quantization to the CPU would take the 0.8B load from under a second to about four minutes (the `w1` trace went from 6 s to 250 s). The difference is documented in `fp8_block.rs` rather than engineered away, and plan step 4 needs no new test because the quantize stream did not change.

### Guard SDPA on the stream device, as upstream CUDA does

The overlay's `use_fallback` decided from shapes alone, so a decode-shaped query on a CPU stream built the fused primitive, whose `eval_cpu` throws `NYI`, which across the cxx bridge is `std::terminate`. Upstream CUDA's `has_fused_kernel` checks the stream device before shapes; the fix copies that order. `force_fused` on a CPU stream now throws a catchable `invalid_argument` rather than silently falling back, so a caller that demands the fused kernel learns it cannot have it.

## 4. Validation

PR author (gfx1151):

- `cargo test --features rocm --lib models::switch_layers::mxfp_tests models::fp8_block -- --test-threads=1`: 16 passed, including the existing round trip. Worst relative L2 error on ROCm: 1.8e-3 (bf16), 2.2e-4 (f16), 2.9e-7 (f32). On the CPU device, reduced matrix: 8.2e-3 (bf16), 1.0e-3 (f16); the earlier full-matrix run gave 7.9e-3, 1.0e-3 and 1.2e-7.
- Revert checks: with item 10 reverted and the overlay rebuilt, both gather tests fail at their first case with non-finite output. With the SDPA guard reverted, `cargo test --features rocm --test cpu_device_sdpa` aborts with `NYI`. Both pass restored.
- Checkpoint `Lazarus-Ai/ReAligned-Qwen3.5-0.8B-FP8` @ `db97e6a7`: CPU against ROCm at `w32` (`32 8 8 0`), 14 top-1 disagreements, 0 of 50 decided, largest gap at a disagreement 0.125, perplexity 100.80 (CPU) and 100.95 (ROCm). In every disagreement the CPU's token is ROCm's second choice. #1809's largest gap across twelve Metal and ROCm pairs was 1.125.
- `mlxcel generate` on ROCm: 132 tensors requantized in 565 ms, 56 to 62 tok/s, coherent text.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`, `cargo test --test dead_doc_pointers`, and clippy `-D warnings` on `mlxcel` (lib, tests, examples) and `mlxcel-core` (lib, tests): pass.

Orchestrator verification (gfx1151, branch rebased onto origin/main `9484ffc2`):

- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke (32 tokens) passed.
- `verify-test-rocm` failed in four targets. Three are the known baseline failures: bf16 `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, and #2037's `gelu_approx_matches_mlx_nn_bit_for_bit` and `tests::family_order_is_exhaustive`. The fourth is `rocm_mxfp4_quant::quantized_matmul_matches_dequantized_reference` at `qmm 2880x2880 M=1`, which is pre-existing and not caused by this PR: it fails 7 of 10 runs on main `9484ffc2` without this PR and 6 of 10 at `4595b06f` (the #2057 merge commit), across f32, f16 and bf16. It is tracked in a separate issue filed from this gate.
- Every other target passed, including the PR's new mxfp tests and `tests/cpu_device_sdpa.rs`.

## 5. Learning Points

- **A reference run must record where it ran.** `examples/logit_trace` never read `MLXCEL_DEVICE`, so the first "CPU" traces were GPU traces and would have agreed with ROCm perfectly for the wrong reason. The fix resolves the device like the CLI and writes it into the trace header, so a trace file now says which device produced it.
- **A test's coverage claim has to be checked against the dispatch gates.** The module doc claimed the stride-0 sorted schedule was covered, but the case missed the schedule's `B / E >= 4` gate. Reading the conditions in `qmm.hip` and sizing each case against them is what found it. The same reading showed that item 9's path is unreachable from block-float modes, which settled a question the issue left open.
- **Bounds for untested backends need the backend's defaults in mind.** An f32 bound tuned to ROCm (2.9e-7 measured) would fail on CUDA sm80+ because TF32 is on by default for grouped GEMM. Setting the bound from each backend's rounding model, then checking it still sits far below the defect, keeps a backend-neutral test from being fitted to the host.
- **Isolate one variable when judging a numeric difference.** Comparing GPU-quantized against CPU-quantized weights with both computed on the GPU, and confirming the traces are deterministic, attributes every difference to tie rounding in `quantize`. Comparing ROCm against the CPU device would have mixed that with the CPU's bf16 accumulation.
- **An escape hatch that aborts is worse than none.** `MLXCEL_DEVICE=cpu` is the natural fallback for a mismatched `gfx` build or a suspect kernel, and on ROCm it crashed on the first attention call. It now runs, slowly, and the installation doc describes it as a correctness reference and escape hatch, not a serving mode.

## 6. Caveats and What Is Not Verified

- **Metal and CUDA** were not run. They are touched by `examples/logit_trace` (which now also applies the default wired limit and arch-compatibility refusal the CLI already applies there) and by the new tests, which run on every backend. No library code path changes there; the SDPA edit is ROCm overlay only.
- **The reference is the CPU device, not Metal.** The criterion was met against a weaker reference (bf16 accumulation). A Metal trace of the same checkpoint would close it as written.
- **The CPU arm covers one width** (`w32`, 50 decided positions). `w1`, `w8` and `w256` were traced on ROCm only, for the quantization comparison.
- **MoE FP8 checkpoints were not run.** `Qwen/Qwen3.5-35B-A3B-FP8` and `Qwen/Qwen3.5-27B-FP8` are too large for this host and for a CPU reference. The mxfp8 MoE path is verified at op level only.
- **Revert evidence is not in the repository.** The item 10 and SDPA revert checks, and the CPU-quantize experiment switch, were run by hand.
- **Only gfx1151** was run, and only group size 32, the only fp group size `gather_qmm` accepts on ROCm.
- **Issue plan step 5** (flip mxfp8 to `Native`) needed no code: the capability table from #1806 already lists ROCm mxfp8 as `Native`. This PR adds the committed evidence behind that entry.

## 7. Remaining Work

- The flaky `rocm_mxfp4_quant::quantized_matmul_matches_dequantized_reference` at `qmm 2880x2880 M=1` (a separate issue filed from this gate). It predates this PR and reproduces at #2057's merge commit.
- #1809: a Metal trace of an FP8 block checkpoint would replace the CPU-device reference.
- #1813: propose item 23 (the SDPA stream-device guard) to the fork, alongside items 8 and 10.
- #1814: ROCm fused MoE ports. mxfp8 MoE runs through the generic `gather_qmm` kernel until then.

Refs: #1807 (closed by this PR), #1801, #1806, #1808, #1809, #1813, #1814, #2037, PR #1818, PR #1826, PR #2057.
