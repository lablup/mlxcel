# Technical Report: PR #2079 - Run CPU-stream BLAS single-threaded over fine-grained memory

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: C++ (ROCm overlay allocator), Rust (test), Markdown

**Risk Level**: Low for correctness, medium for CPU-stream speed (one allocator hook; GPU kernels are unchanged. CPU-stream BLAS becomes about 7x slower on the measured gemv. Metal and CUDA builds never copy `patches-rocm/`)

## Executive Summary

Issue #2072 (part of epic #1801) reported that `quantized_matmul_matches_dequantized_reference` in `tests/rocm_mxfp4_quant.rs` failed intermittently on gfx1151, always at `qmm 2880x2880 M=1`, the gpt-oss-20b decode shape. The issue suspected `qmv_warp_shared_kernel`, and if the kernel had been at fault, gpt-oss decode on ROCm could have been producing wrong logits silently.

The kernel was right. The test's CPU reference was wrong. On an APU the ROCm allocator hands out fine-grained device memory for every array, CPU-stream arrays included, and multithreaded OpenBLAS (Debian 0.3.29, 32 threads) writing a `cblas_sgemm` result into that memory returns wrong output columns, differently from call to call. The wrong columns are each the last column of one OpenBLAS thread's share of the output, whose 64-byte cache line the next thread also writes. A standalone C program reproduced it with no MLX involved.

The fix is in the overlay allocator: `unified_malloc` calls `openblas_set_num_threads(1)` once, on its first fine-grained allocation, through a weak symbol so a build against another BLAS links and changes nothing. Tolerances are unchanged. The cost is CPU-stream BLAS speed: the reproducing gemv went from about 0.2 s to about 1.4 s per call. The change is `LOCAL_FIXES.md` item 27, an upstreaming candidate for #1813.

## 1. Problem Statement

The failing case quantizes a random `[2880, 2880]` weight to mxfp4 on the GPU, runs `quantized_matmul` on the GPU, and compares it against a reference built on the CPU stream: `dequantize_cpu` of the same packed bytes and scales, then `want = astype(x, f32) @ dense.T`. Every dtype failed (`f32: 0.17341012 exceeds 0.0001`, `f16: 0.11339103`, `bf16: 0.19927761`), at rates of 7 of 10 runs at main `9484ffc2` and 6 of 10 at the #2057 merge, with the same error values recurring across runs. The 2880x2880 M=8 and M=64 cases and the smaller shapes never failed.

Two properties made the kernel look guilty. K=2880 is the only tested K above the kernel's 2048-element shared-memory chunk, so it is the only shape that takes two chunks and a tail. And the test seeds once and draws inputs in a fixed order, so nominally fixed inputs appeared to produce varying output. Until it was fixed, `make verify-rocm` was nondeterministic, and with hosted CI down that gate is what every epic PR merges on.

## 2. Investigation

The issue's plan was to prove the inputs fixed, build an in-process reproduction, and decide kernel vs reference. The investigation followed that plan and ended on the side the issue ranked less likely.

### 2.1 Hypotheses ruled out

| Hypothesis | How it was checked | Result |
|---|---|---|
| Input nondeterminism | Hashed `w`, `packed`, `scales` and `x` across runs | Identical in every run |
| Race at the chunk boundary in `shared_x` | Read the chunked qmv kernels (`qmv_warp_shared_kernel` and its batched and gather variants) | `__syncthreads()` both after each chunk load and before the next load |
| Tail chunk (2880 - 2048 = 832 elements) | Same reading | The tail loads and reads only `k < chunk_end` |
| Wave32 mismatch | Same reading | `THREADS_PER_COL=16` fits in a wave32 |
| GPU quantizer disagrees with the CPU path | Compared GPU and CPU `dequantize`, and GPU packed/scales against the CPU quantizer | Bit for bit equal |

### 2.2 The reference was the wrong side

In a failing process the same wrong columns repeat on every repetition, because `want` is computed once and reused. That was the first sign that the varying quantity across runs was the reference, not the GPU output. With tensors dumped from a failing run, the GPU output matched an independent numpy dequantize-and-dot of the packed bytes exactly. The CPU `matmul` was off in 4 columns.

It also explains the shape of the symptom. The reference for every dtype is an f32 matmul on the CPU stream, so it goes through `cblas_sgemm` regardless of the tested dtype, which is why f32, f16 and bf16 all failed. Recurring identical error values fit a fault that lands on deterministic column positions.

### 2.3 Isolating it in the CPU stream

A CPU-only loop of `matmul(x, w.T)` on the CPU stream, no GPU kernel involved:

- wrong in 35 of 60 calls with the host otherwise idle;
- wrong in 12 of 200 while another GPU process was running, so concurrent GPU load lowers the rate rather than causing it;
- wrong in 0 of 300 with `OPENBLAS_NUM_THREADS=1`.

### 2.4 Standalone C reproduction

A C program calling `cblas_sgemm` for `[1, 2880] x [2880, 2880]^T` directly, varying where the buffers live and how many threads OpenBLAS uses (gfx1151 host, Ryzen AI MAX+ 395, Debian OpenBLAS 0.3.29 pthread):

| Inputs | Output | OpenBLAS threads | Wrong calls |
|---|---|---|---|
| fine-grained (`hipExtMallocWithFlags(hipDeviceMallocFinegrained)`) | fine-grained | 32 (default) | 37 of 50 |
| malloc | fine-grained | 32 | 10 of 300 |
| malloc | malloc | 32 | 0 |
| `hipHostMalloc` | `hipHostMalloc` | 32 | 0 |
| fine-grained | fine-grained | 1, 2 or 4 | 0 |

This removes MLX, the GPU kernel and the test from the picture. The defect needs two ingredients: multithreaded OpenBLAS and an output in fine-grained memory. Inputs in fine-grained memory raise the rate but are not required.

### 2.5 Where the wrong columns sit

Through MLX, each wrong column was the last column of one OpenBLAS thread's share of the output. Shares were 93 columns, so the wrong columns were 650, 1022, 1859 and similar (650 = 7 x 93 - 1). An f32 output column is 4 bytes, so a 64-byte line holds 16 columns, and column 650's line (columns 640 to 655) is also written by the thread whose share starts at 651. Every wrong column found sat at such a shared line.

## 3. Change Summary

- **`patches-rocm/mlx/backend/rocm/allocator.cpp`**: declares `extern "C" void openblas_set_num_threads(int) __attribute__((weak))` and adds `single_thread_cpu_blas_for_finegrained()`, which calls it once under a `std::once_flag` when the symbol resolves. `unified_malloc` calls it right after a successful `hipExtMallocWithFlags(..., hipDeviceMallocFinegrained)`. A comment records the measurements and the reasoning.
- **`tests/rocm_cpu_blas_finegrained.rs`** (new): generates `w` and `x` on the GPU, computes an f64 host reference, then runs the `[1, 2880] x [2880, 2880]^T` matmul 16 times on the CPU stream and fails on any column off by more than 1e-4 relative to the output scale. It holds `lock_default_device` for its whole body and skips on non-ROCm backends.
- **`patches-rocm/LOCAL_FIXES.md`**: item 27, placed in the runtime list after item 25, with the mechanism, the measurements, the cost, the rejected alternative and the #1813 upstreaming note.
- **`docs/installation.md`** (ROCm support table, CPU device row): BLAS work on the CPU device runs on one OpenBLAS thread, and why.

`tests/rocm_mxfp4_quant.rs` and `qmm.hip` are not changed. The test that exposed the bug now passes because its reference is right.

## 4. Technical Decisions

### Fix the reference path, not the tolerance

The issue's acceptance criteria forbade loosening tolerances. The errors were 0.06 to 0.2 relative, far outside any rounding tolerance, so a looser bound would have hidden a real wrong-result bug in the CPU backend rather than noise.

### One BLAS thread instead of a cacheable scratch output

The C table shows a cacheable output keeps multithreaded OpenBLAS correct, so the alternative was to give BLAS a malloc'd output and copy the result into the fine-grained buffer. MLX's CPU backend writes BLAS and LAPACK results straight into array buffers at many call sites (`cblas.cpp`, `conv.cpp`, `masked_mm.cpp`, the LAPACK primitives), so that route would mean a change at every one of them in the fork. One thread fixes all of them from a single place. Two and four threads were also exact in the C reproduction, but the PR did not measure them through MLX or at other shapes, so one thread is the setting with evidence behind it.

### Set it in the allocator, on the first fine-grained allocation

The hook sits where fine-grained memory is created. On an APU every array comes from this path, so the first fine-grained allocation precedes any CPU-stream BLAS call that could write such a buffer. A discrete-GPU configuration that does not take the fine-grained path never calls it and keeps OpenBLAS's thread count.

### Weak symbol

`openblas_set_num_threads` is OpenBLAS-specific. Declaring it weak lets a build linked against another BLAS still link; the pointer is null there and the hook does nothing.

## 5. Cost

- The reproducing gemv takes about 1.4 s per call on the CPU stream, up from about 0.2 s. The CPU reads fine-grained memory slowly, and one thread no longer hides that.
- The full `rocm_mxfp4_quant` target takes 19 s instead of 8 s.
- Not affected: GPU work, quantized CPU matmuls, and bf16/f16 CPU matmuls, which use MLX's own SIMD kernels rather than BLAS. Model inference under `MLXCEL_DEVICE=cpu` therefore barely touches BLAS.
- The setting is process-global. Any other OpenBLAS user in the same process after the first fine-grained allocation also runs single-threaded.

## 6. Validation

PR author (gfx1151):

- A forced clean HIP rebuild (trashed `hip_objs/`, reconfigured) before any measurement. Baseline at main: 3 of 8 runs failed (errors 0.104, 0.173, 0.063).
- `tests/rocm_cpu_blas_finegrained.rs` failed in 5 of 5 runs with the allocator change reverted, confirmed by the `allocator.cpp.o` timestamp, each within the first 8 calls, and passed in 6 of 6 runs with it.
- `cargo test --profile test-fast --features rocm --test rocm_mxfp4_quant -- --test-threads=1 quantized_matmul` passed 50 of 50 in a row. Another KFD process was listed at the start of 49 runs, a new PID each time and most likely the previous run's process exiting, so the runs were not strictly isolated.
- The full `rocm_mxfp4_quant` target (4 tests) passed. Clippy on the new test with `-D warnings`, `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`, and `dead_doc_pointers` passed.
- No gpt-oss-20b logit trace: the issue required one only if the kernel was at fault, and the kernel's output did not change.

Orchestrator verification (gfx1151, branch rebased onto `bf5bf525`):

- The rebase conflicted in `LOCAL_FIXES.md`. The orchestrator kept #2076's edited item 25, then this PR's item 27 in the runtime list, then #2076's build section with item 26.
- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke run (32 tokens) passed.
- `verify-test-rocm` failed in exactly the three known baseline targets: `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, and #2037's `gelu_approx_matches_mlx_nn_bit_for_bit` and `tests::family_order_is_exhaustive`. In #2076's gate the same step also failed `rocm_mxfp4_quant`; here it did not.
- Inside the full suite, `tests/rocm_cpu_blas_finegrained.rs` passed and `tests/rocm_mxfp4_quant.rs` passed 4 of 4. The mlxcel-core library tests passed 1835.

The branch has since been rebased onto `929c80ab` (#2077), which touches only the paged KV cache and the server scheduler; the gate above was not re-run on that base.

## 7. Learning Points

- **Check the reference before the kernel.** A reference computed once and reused turns a flaky reference into what looks like a flaky kernel: the reference is the same wrong value on every repetition inside a run, and a different wrong value across runs. The decisive test was a third, independent computation (numpy over the dumped packed bytes), which agreed with the GPU and not with the CPU.
- **Fine-grained memory is not ordinary memory for the CPU.** On this APU the allocator's fine-grained memory is what every array gets, including arrays the CPU stream writes. Code that is correct on malloc'd memory, here multithreaded OpenBLAS with adjacent threads writing the same 64-byte line, returned wrong results on it. Any CPU code writing these buffers in parallel is suspect in the same way.
- **Strip the system down to a standalone reproduction.** The C `cblas_sgemm` table moved the question from "which layer of MLX" to "which buffer placement and thread count", and its rows directly justify both the fix (one thread) and the rejected alternative (cacheable output).
- **The orchestrator's own wrong turns.** The orchestrator first took the failure as input-dependent, then as a race in the kernel's chunked shared-memory loop. It also tried a warm-tree bisect across commits, which was invalid: before #2076 fixed #2075, a warm ROCm tree did not recompile HIP objects when only a header changed, so bisect steps could run stale kernels. Both hypotheses fell once inputs were hashed and the GPU output was checked against an independent decode, and all reproduction work behind this PR was done on a forced clean HIP rebuild. The lesson for the epic: before bisecting a GPU symptom, confirm the build compiles what each step contains and confirm which side of the comparison is wrong.
- **Concurrent load changed the rate, not the cause.** The CPU loop failed less often with another GPU process running (12 of 200 vs 35 of 60). A rate that moves with background load is easy to misread as a contention bug; here it was a property of the idle case.

## 8. Caveats and What Is Not Verified

- **The hardware mechanism** is not established. The evidence is positional (wrong columns sit at thread-share boundaries sharing a 64-byte line) and conditional (it needs a fine-grained output and appeared at 32 threads, not at 1, 2 or 4). Why concurrent writes to one line of fine-grained memory lose data on this APU, and whether it depends on the OpenBLAS version, the kernel driver or the CPU, was not investigated.
- **Why only M=1 failed** in `rocm_mxfp4_quant` was not examined; M=8 and M=64 at the same K and N never failed.
- **Two or four threads** were exact in the C reproduction but were not tried through MLX, so a less costly thread count is untested.
- **Other BLAS and LAPACK paths** (convolution, linear algebra) are covered by the same process-wide setting but were not tested individually.
- **Metal and CUDA** were not run (not available on this host). The change is confined to `patches-rocm`, which those builds never compile.
- **The 50-run acceptance loop** ran with another KFD process present at the start of most runs.

## 9. Remaining Work

- #1813: propose item 27 to the fork with the other upstreaming candidates.
- If CPU-stream f32 BLAS speed ever matters on the APU, the cacheable scratch output (or a measured higher thread count) is the recorded alternative.
- The three baseline `verify-test-rocm` failures (`prefill_dense_gemm_matches_qmm_bytes_where_eligible`, and #2037's two) are tracked outside this PR.

Refs: #2072 (closed by this PR), #1801, #2075, PR #2076, #1808, PR #2057, PR #2071, #1813, #2037.
