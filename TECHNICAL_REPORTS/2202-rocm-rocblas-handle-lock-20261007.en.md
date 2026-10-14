# Technical Report: PR #2202 - Initialize the ROCm Device's rocBLAS Handle Once, Under a Lock

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); code head `e337e6bf` on origin/main `33f60aa9`, PR open, pending merge. Closes #2198 (part of #1801).

**Languages**: C++ and HIP (ROCm overlay `device.h`, `device.cpp`, `matmul.cpp`, `gemms/rocblas_gemm.cpp`, `quantized/qmm.hip`), Rust (new `tests/rocm_rocblas_handle_concurrency.rs`), Markdown (`patches-rocm/LOCAL_FIXES.md`)

**Risk Level**: Low to medium. The change does not alter any kernel, GEMM argument or result. It adds a per-device mutex that every rocBLAS call now holds from the stream bind through the GEMM enqueue, so host-side enqueue of rocBLAS GEMMs is serialized across threads. That cost was not measured. Metal and CUDA code paths are untouched.

## Executive Summary

Issue #2198 reported two unlocked windows in the ROCm overlay's `Device`. First, `get_rocblas_handle()` set its initialized flag before it called `rocblas_create_handle`, so a second thread arriving in that window got a null handle. Second, every GEMM called `set_rocblas_stream()` on the one shared handle with no lock, so thread B's rebind between thread A's rebind and A's GEMM put A's GEMM on B's stream. The server runs forwards on several worker threads that share one `Device`, each on its own stream, so both windows are reachable on a single GPU.

The PR makes the rocBLAS handle reachable only through a move-only `RocblasLease`. `Device::acquire_rocblas(stream)` locks `rocblas_mtx_`, creates the handle on first use, binds the stream only when it differs from the last bound one (throwing on failure), and returns a lease that holds the mutex until it is destroyed. An atomic `rocblas_ready_` flag is stored with release only after a successful create. A failed create warns once, throws, and is retried by the next call. All eight rocBLAS call sites take a lease; `get_rocblas_handle()` is private and `set_rocblas_stream()` is gone, so an unguarded call no longer compiles.

The new test reproduces the failure deterministically. With the fix removed, 20 of 20 runs failed (187 of 200 child processes). With the fix, 0 of 20 runs and 0 of 200 child processes failed. This differs from #2196's JIT-cache test, which was only a crash guard. The orchestrator's `make verify-rocm` on `e337e6bf` passed: 157 suites, 12,050 passed, 0 failed, 385 ignored.

## 1. Problem Statement

### 1.1 The handle init race

On main at `33f60aa9`, `Device::get_rocblas_handle()` set `rocblas_initialized_ = true` first, then ran the architecture check and `rocblas_create_handle`. A second thread that arrived in that window saw the flag, saw `rocblas_available_` still at its default `true`, and returned `rocblas_`, which was still `nullptr`. If two threads both read the flag before either wrote it, both created a handle and one leaked. Several call sites discard the GEMM status (the f32 `rocblas_sgemm` sites, for example), so a null handle left the output unwritten with no error.

`is_rocblas_available()` runs on the first f32 GEMM that hipBLASLt does not take and on every affine quantized matmul route decision, so the first quantized forward in a process creates the handle. With `-m <chat> --embedding-model <emb>`, first requests to both models that arrive together race the init, and no startup warmup forward creates the handle first.

### 1.2 The stream rebind race

Every rocBLAS call site called `set_rocblas_stream(stream)` and then issued its GEMM. rocBLAS binds the stream on the handle, and the overlay keeps one handle per device. Thread B's rebind between thread A's rebind and A's GEMM enqueued A's GEMM on B's stream, and A's stream does not wait for it, so A can read the output before the GEMM ran. This is open on every rocBLAS GEMM, not only the first, whenever two worker threads run rocBLAS GEMMs.

### 1.3 The probed flags

`has_native_wmma()` and `supports_cdna_mfma_gemm()` set their probed flags before computing the result, so a concurrent caller read `false` once and took the non-WMMA route. `is_rocblas_bf16_available()` had no caller and its recovery path called `hipDeviceReset()` and recreated the shared handle under other threads. Both are folded into the fix (section 3.3).

## 2. Change Summary

| Area | Change |
|---|---|
| `patches-rocm/mlx/backend/rocm/device.h` | New move-only `RocblasLease` (a `std::unique_lock<std::mutex>` plus the handle, `handle()` accessor). `Device::acquire_rocblas(hipStream_t)` added. `get_rocblas_handle()` made private with a "caller holds `rocblas_mtx_`" contract. New members `std::mutex rocblas_mtx_`, `std::atomic<bool> rocblas_ready_`, `rocblas_warned_`, `rocblas_arch_supported_`, `arch_name_`. `has_native_wmma()` and `supports_cdna_mfma_gemm()` are `const` inline getters. `set_rocblas_stream()`, `is_rocblas_bf16_available()` and the probed and bf16 members removed. |
| `patches-rocm/mlx/backend/rocm/device.cpp` | Architecture facts computed once in `Device::Device` from the `hipDeviceProp_t` it already reads, with the same arch lists, now as file-local helpers. Handle created under the mutex; `rocblas_ready_` stored with release after a successful create. `acquire_rocblas` binds the stream only when changed. `is_rocblas_available()` answers without the lock once ready. |
| `matmul.cpp` (3 lambdas) | Each starts with `auto lease = device.acquire_rocblas(stream); rocblas_handle handle = lease.handle();`. The two outer handle fetches are deleted. |
| `gemms/rocblas_gemm.cpp` (3 lambdas), `quantized/qmm.hip` (2 lambdas) | Same two lines. In `dequant_rocblas_gemm` the lease is taken after the hipBLASLt attempt, so hipBLASLt GEMMs never hold it. |
| `tests/rocm_rocblas_handle_concurrency.rs` (new) | Fresh-process concurrency test (section 4). |
| `patches-rocm/LOCAL_FIXES.md` | Item 37. |

One commit, 7 files, 506 insertions, 269 deletions. The `handle` local alias in each lambda keeps the argument lists of the rocBLAS calls unchanged; the lease outlives every call in the lambda.

## 3. Design

### 3.1 The lease

One handle per `Device`, created once under `rocblas_mtx_`, and every rocBLAS call made while holding that mutex through a lease that has already bound the stream.

- **Bind and enqueue are one critical section.** The lease holds the mutex from `acquire_rocblas` to scope exit, so another thread cannot rebind the handle between this thread's bind and its GEMM. This closes the rebind race.
- **Rebind only on change.** `acquire_rocblas` compares against `rocblas_stream_` and calls `rocblas_set_stream` only when the stream differs. A failure throws `std::runtime_error` with the status instead of running the GEMM on the wrong stream.
- **Release on unwind.** The lock is a `std::unique_lock` inside the lease, so an exception in the lambda releases the mutex.
- **Compile-time enforcement.** `get_rocblas_handle()` is private and `set_rocblas_stream()` is removed, so a new call site cannot reach the handle without `acquire_rocblas`. A repository grep for `get_rocblas_handle`, `rocblas_set_stream` and `set_rocblas_stream` under `patches-rocm` matches only `device.cpp` and `device.h`.
- **Only host-side enqueue is serialized.** The GEMMs still overlap on their own streams on the GPU.

### 3.2 The init protocol

`get_rocblas_handle()` runs under the mutex. If `rocblas_ready_` is already set it returns the handle. On an unsupported architecture it throws. Otherwise it calls `rocblas_create_handle`; on failure it prints one warning (guarded by `rocblas_warned_`), caches nothing and throws, and the next call retries. On success it stores the handle, resets `rocblas_stream_` to null (a fresh handle is bound to the null stream, so the first lease rebinds) and stores `rocblas_ready_` with release.

Before this change a failed create disabled rocBLAS for the whole process. Now a transient failure is retried, and the failure is an exception rather than a null handle returned to the caller.

`is_rocblas_available()` loads `rocblas_ready_` with acquire and returns true without the mutex once the handle exists, so a thread that holds a lease can call it without deadlocking. Otherwise it locks, attempts the create, and reports whether it worked.

### 3.3 Architecture facts at construction

`rocblas_arch_supported_`, `has_native_wmma_` and `cdna_mfma_ok_` are computed once in `Device::Device` from the `hipDeviceProp_t` the constructor already reads, using the same arch lists as before. `has_native_wmma()` and `supports_cdna_mfma_gemm()` become `const` getters, which removes the probed flags and the race on them.

This moves the unsupported-architecture warning to construction. The issue text had the warning printing on first use; with the support flag known at construction, `is_rocblas_available()` no longer needs to lock on an unsupported device, so the warning prints once when the `Device` is built. That is a deliberate deviation from the issue text, and it changes when the message appears, not whether.

`is_rocblas_bf16_available()` is removed with its members. It had no caller in the tree outside `device.cpp` and `device.h`, and its failure path called `hipDeviceReset()` and recreated the shared handle, which is unsafe with other threads running.

### 3.4 Rejected: per-thread handles

Upstream CUDA at the pin (MLX `81ba1c6a`, `mlx/backend/cuda/cublas_utils.cpp`) keeps cuBLASLt handles in `static thread_local std::vector<CublasHandles> cache(gpu::device_count())`, creates each with `CHECK_CUBLAS_ERROR`, and passes the stream per call to `cublasLtMatmul` (`encoder.stream()`). Nothing is shared there. The PR does not copy it:

- rocBLAS binds the stream on the handle, not per call, so per-thread handles would still need a bind, and the fork is built around one handle per device.
- A per-thread handle would be created on each worker's first GEMM, which may fall inside a decode-step stream capture.

The lease trades serialization for a single shared handle. A per-thread handle with an eager creation point is the alternative if a profile ever shows contention on `rocblas_mtx_`.

## 4. Verification

### 4.1 The test

`tests/rocm_rocblas_handle_concurrency.rs` (`#![cfg(feature = "rocm")]`) runs `first_gemm_from_many_threads_in_fresh_processes`, which spawns the `#[ignore]` child test 10 times in fresh processes with `MLX_NO_HIPBLASLT=1` (so GEMMs stay on rocBLAS; #2200 will drop this). Each child starts 8 threads on their own streams behind a barrier. Each thread runs 33 f32 `[64, 128] x [128, 64]` matmuls with integer inputs in `-4..=4`, and every output is checked exactly against a host `i32` reference. A fresh process matters because the init window is once per process.

### 4.2 The 20-run table

On gfx1151, 20 runs of the test per build, 10 child processes each:

| Build | Runs failed | Child processes failed |
|---|---:|---:|
| Overlay from main (fix removed), new test only | 20 of 20 | 187 of 200 |
| This PR | 0 of 20 | 0 of 200 |

Across the failing threads without the fix, 354 reported outputs left at 0 (a GEMM issued on the null handle, status discarded) and 86 reported a sum one partial GEMM short (a GEMM enqueued on another thread's stream and read before it ran). Those two signatures match the two races in sections 1.1 and 1.2.

Unlike #2196's JIT-cache test, which was only a crash guard, this test reproduces the failure on every run without the fix, so a regression in the lease should be caught on this host.

### 4.3 Gates

- `cargo clippy --features rocm --test rocm_rocblas_handle_concurrency -- -D warnings`: clean.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`: pass.
- Orchestrator's `make verify-rocm` on `e337e6bf` (on main `33f60aa9`): 157 test suites, 12,050 passed, 0 failed, 385 ignored; clippy, smoke, overlay and script gates OK.

Not verified: Metal and CUDA (not available on this host; the change touches only files under `patches-rocm/` and a `rocm`-gated test), and the performance cost of the lease.

## 5. LOCAL_FIXES Item 37

The entry goes under "Fixes to the fork's kernels" in `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`. It records the flag-before-work init, the rebind race and its null-handle and wrong-stream consequences, the probed flags, the construction-time arch facts and the warning move, the init protocol, the lease and its serialization, the private accessor, the removed bf16 probe, the upstream comparison and why per-thread handles were rejected, and the test with its 20-of-20 count. It ends with the fork-policy wording: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there (lablup/mlxcel#2198)." Nothing is added under `docs/mlxcelverse/upstream/`.

## 6. Technical Decisions

- **One lease, one mutex, not a lock around `set_rocblas_stream` alone.** The race is between bind and GEMM, so the lock must span both.
- **Make the unguarded path impossible.** A private getter and no public rebind turn the next unguarded call into a compile error instead of a review item.
- **Throw on failure, retry on the next call.** A null handle returned to a call site that discards status produces silent wrong output; an exception does not. Retrying avoids disabling rocBLAS for the process after one transient failure.
- **Compute arch facts at construction.** They are immutable properties of the device; removing the lazy flags removes their race without a lock.
- **Take the lease after the hipBLASLt attempt in `dequant_rocblas_gemm`.** hipBLASLt GEMMs never touch the rocBLAS handle, so they should not queue behind it.
- **Lease over per-thread handles.** Section 3.4.
- **Remove the uncalled bf16 probe.** Dead code with an unsafe recovery path.

## 7. Follow-ups

- **#2200, hipBLASLt shared state.** Next in the series. One 32 MB workspace per device is shared by every stream, `ensure_workspace` frees and reallocates it unlocked, init publishes `initialized` before the handle and workspace exist, and a failed cached-pipe matmul destroys a `GemmPipe` another thread may still be using. It also removes `MLX_NO_HIPBLASLT=1` from this test, so the test then covers the default bf16 and fp16 route too.
- **#2197, the `rocm::device()` map.** An unlocked `find`/`try_emplace` on a process-global map, latent and multi-GPU only.
- **Stale LOCAL_FIXES intro wording.** The intro text of `LOCAL_FIXES.md` is out of date; #2144 retires it, and this PR does not touch it.

## 8. Residual Risks

- **Lock cost unmeasured.** Every rocBLAS GEMM now takes `rocblas_mtx_` for the host-side enqueue, so concurrent rocBLAS enqueues from several workers serialize. It is expected to be small next to GEMM execution, which still overlaps on the GPU, but that is an expectation, not a measurement.
- **Lease scope is the lambda.** A future call site that stores a handle past the lease, or calls `lease.handle()` after the lease is destroyed, would reintroduce the race. The header comment states the contract.
- **hipBLASLt and the device map are still open.** #2200 and #2197 remain until they land; multi-model ROCm serving can still race on hipBLASLt state.
- **One host.** All runs are on one gfx1151.

## 9. Learning Points

- **Locking the stream bind alone would not have fixed it.** The failure is the gap between bind and use, so the unit to protect is the pair, which is what the lease encodes.
- **A fresh-process test is needed for once-per-process init.** Running the threads in a long-lived test binary would hit the init window once, at best. Ten children per run turn a rare window into a count.
- **Separate signatures in the failure count.** Zeros and sums one partial GEMM short map to the null handle and the wrong stream, which is how the test shows both races were real and both are closed.
- **Discarded GEMM status hides bad handles.** The old null handle produced no error at several sites. Throwing from `acquire_rocblas` moves the failure to where it can be seen.
- **Make the safe path the only path.** Hiding `get_rocblas_handle()` and deleting `set_rocblas_stream()` means the next rocBLAS call site cannot skip the lock.
