// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Stream-selection wrappers for generation-time pipelining.
//!
//! Generation owners ([`crate::generate::CxxGenerator`],
//! [`crate::speculative::SpeculativeGenerator`], and the server-side
//! `BatchScheduler`) create a dedicated MLX stream up front and install
//! it as the default for the worker thread that drives the generation
//! loop. Until that stream was a plain
//! `mlx::core::new_stream(Device::gpu)` instance, which means every
//! owner had to be constructed on the same thread that ran the loop —
//! otherwise the stream would be "owned" by the wrong thread and any
//! `synchronize()` call would target a different physical stream than
//! the one that dispatched the work.
//!
//! upstream `mlx-vlm` PR #1050 (commit `728fab1`)
//! introduces `mlx::core::ThreadLocalStream`: a TLS-backed handle that
//! resolves to a per-thread `Stream` on demand. The generation owners
//! now hold a `ThreadLocalStream` and resolve it on the worker thread
//! at install time, so dispatch and synchronization always pair up on
//! the same per-thread stream regardless of which thread constructed
//! the generator.
//!
//! See `docs/bridge-overhead-microbench.md` for the per-op cost
//! microbench used to validate that this change is at worst a no-op
//! and at best a small per-step latency win on multi-worker
//! deployments where construction and execution can happen on
//! different threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::UniquePtr;
use crate::ffi;
use crate::ffi::{MlxStream, MlxThreadLocalStream};

/// Create a thread-local generation stream bound to the GPU device.
///
/// Returns `None` when the MLX default device is not the GPU (CPU-only
/// builds, or a default moved to the CPU with `set_default_device(false)`;
/// `default_device_is_gpu()` is false in both cases) so callers can fall
/// back to the MLX default stream.
///
/// The returned [`MlxThreadLocalStream`] handle is independent of the
/// thread it was created on — pass it to
/// [`install_thread_local_default_stream`] from any thread that wants
/// to bind its dedicated per-thread stream as the default for
/// subsequent MLX dispatches on that thread.
///
/// Used by: CxxGenerator, SpeculativeGenerator, BatchScheduler, AudioWorker
pub fn new_thread_local_generation_stream() -> Option<UniquePtr<MlxThreadLocalStream>> {
    if ffi::default_device_is_gpu() {
        Some(ffi::new_thread_local_stream_gpu())
    } else {
        None
    }
}

/// Resolve the calling thread's `MlxStream` from a thread-local handle
/// and install it as the default stream for that thread.
///
/// This is the worker-thread-side counterpart of
/// [`new_thread_local_generation_stream`]: the generator owner is
/// typically constructed on a control thread, then later runs on a
/// dedicated worker thread. Calling this method **on the worker thread**
/// (e.g. at the top of the generation loop) ensures every subsequent
/// MLX op on that thread is dispatched on the same per-thread stream
/// that synchronization will target.
///
/// `None` is a safe no-op for CPU-only builds.
///
/// Used by: CxxGenerator, SpeculativeGenerator, BatchScheduler, AudioWorker
pub fn install_thread_local_default_stream(tls: Option<&UniquePtr<MlxThreadLocalStream>>) {
    if let Some(tls) = tls {
        let stream = ffi::stream_from_thread_local_stream(tls);
        ffi::set_default_stream(&stream);
    }
}

/// Synchronize the calling thread's stream associated with a
/// thread-local handle.
///
/// Equivalent to resolving the handle on the calling thread and
/// calling `synchronize_stream`, but uses MLX's
/// `synchronize(ThreadLocalStream)` overload so the synchronization is
/// guaranteed to target the same per-thread stream that previously
/// dispatched work via this handle.
///
/// `None` is a safe no-op (matches the CPU-only build path of
/// [`new_thread_local_generation_stream`]).
///
/// Used by: AudioWorker (after each request); tests. Generation-loop callers
/// rely on the default stream installed by
/// [`install_thread_local_default_stream`] and on per-op `eval` for
/// synchronization.
pub fn synchronize_thread_local_stream(tls: Option<&UniquePtr<MlxThreadLocalStream>>) {
    if let Some(tls) = tls {
        ffi::synchronize_thread_local_stream(tls);
    }
}

/// RAII guard that restores the calling thread's MLX default stream on drop.
///
/// Constructed by capturing the current default stream before installing a
/// new one. When the guard is dropped the previous stream is restored,
/// leaving the thread in exactly the state it was in before the installation.
///
/// This is primarily useful in tests that call
/// [`install_thread_local_default_stream`] and must not leak the mutated
/// per-thread state into subsequent test cases that run on the same thread.
///
/// # Example
///
/// ```ignore
/// let _guard = DefaultStreamGuard::capture();
/// install_thread_local_default_stream(Some(&tls));
/// // … test body …
/// // guard restores previous default stream here
/// ```
#[must_use]
pub struct DefaultStreamGuard {
    previous: UniquePtr<MlxStream>,
}

impl DefaultStreamGuard {
    /// Capture the calling thread's current default stream.
    ///
    /// The returned guard will restore that stream when dropped.
    pub fn capture() -> Self {
        Self {
            previous: ffi::default_stream(),
        }
    }
}

impl Drop for DefaultStreamGuard {
    fn drop(&mut self) {
        ffi::set_default_stream(&self.previous);
    }
}

/// RAII guard that restores MLX's process-wide default device on drop.
///
/// [`DefaultDeviceGuard::cpu`] and [`DefaultDeviceGuard::gpu`] record whether
/// the default device is currently the GPU (`default_device_is_gpu`), move
/// it, and move it back when the guard drops. [`DefaultDeviceGuard::capture`]
/// records without moving, for code that changes the device through another
/// path such as [`set_default_gpu_device`].
///
/// The default device is process-global and every MLX op dispatched without
/// an explicit stream lands on it, so a test that moves it and never moves it
/// back changes what every later test in the same process measures: under
/// `--test-threads=1` the real-checkpoint gates that sorted after
/// `multimodal::host_preprocessor_tests` scored a reranker on the CPU backend
/// instead of the GPU (issue #1421). A `Once` cannot restore anything; this
/// guard can, and `Drop` runs during unwinding, so a failing test does not
/// poison the modules after it either.
///
/// Guards nest: an inner guard restores the value the outer one installed,
/// and the outer guard then restores the original. Restoration goes through
/// the boolean `set_default_device`, so a GPU default comes back as GPU
/// index 0 even if it had been moved to another index with
/// [`set_default_gpu_device`].
///
/// [`DefaultDeviceGuard::gpu`] on a build without a GPU backend
/// (`gpu_backend_available()` false) leaves the default device where it is:
/// MLX rejects `set_default_device(gpu)` there, and the guard must never ask
/// for a device the backend cannot provide.
///
/// The guard restores, it does not exclude. Creating one does not stop
/// another thread from moving the same process-global device in the middle
/// of the span, so every caller that moves the default device, or that
/// measures something which depends on where it points, holds
/// [`lock_default_device`] for that span as well.
///
/// # Example
///
/// ```ignore
/// let _device = DefaultDeviceGuard::cpu();
/// // … test body runs on the CPU …
/// // guard restores the previous default device here
/// ```
#[must_use = "the previous default device is restored when the guard drops; bind it to a named local, not `_`"]
pub struct DefaultDeviceGuard {
    previous_is_gpu: bool,
}

/// How many [`DefaultDeviceGuard`]s exist right now, process-wide. See
/// [`default_device_guards_held`].
static LIVE_DEVICE_GUARDS: AtomicUsize = AtomicUsize::new(0);

/// The number of [`DefaultDeviceGuard`]s currently held anywhere in the
/// process.
///
/// A test helper that asserts nobody leaked the default device reads this to
/// tell a leak from a guard another test is legitimately holding at this
/// instant. Under `--test-threads=1`, which every gate this repository
/// defines passes, no guard can be alive while an unrelated test runs, so a
/// real leak still fails loudly. A parallel run has no such guarantee:
/// `scripts/run_quality_gate.sh` runs `cargo test --lib` without
/// `--test-threads=1`, and there a concurrent `DefaultDeviceGuard::cpu()` is
/// correct code doing its job, not the leak the assertion is looking for
/// (issue #1421).
#[must_use]
pub fn default_device_guards_held() -> usize {
    LIVE_DEVICE_GUARDS.load(Ordering::Acquire)
}

/// The one process-wide lock for code that moves MLX's default device, or
/// that reads it expecting nobody else to move it.
///
/// [`DefaultDeviceGuard`] puts the device back, which is necessary and not
/// sufficient: the device is process-global while libtest runs one binary's
/// tests in parallel, so a lock private to one module only excludes that
/// module's own tests. Two modules holding two different locks still
/// interleave, and the second one then reads a device the first is
/// legitimately holding, either as the baseline it will compare against later
/// or as the backend its measurement lands on. Both happened here:
/// `multimodal::host_preprocessor`'s restore check read the CPU default that
/// `vision::merge` was holding and reported it as a leak in 4 of 300 runs of
/// `cargo test --lib` over the two modules, and any gate holding
/// `mlx_test_guard` could have scored a real checkpoint on the CPU for the
/// same reason without saying so. One lock for the one device removes both
/// (issue #1421).
///
/// The lock is not reentrant, so take it once per span. Creating a
/// [`DefaultDeviceGuard`] deliberately does not take it, so guards still nest
/// inside a span that holds it. A poisoned lock is recovered rather than
/// propagated: one panicking test must fail alone, not cascade into every
/// test after it.
#[must_use = "the lock is released when the returned value drops; bind it to a named local, not `_`"]
pub fn lock_default_device() -> DefaultDeviceLock {
    DefaultDeviceLock {
        _guard: DEFAULT_DEVICE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    }
}

/// The lock behind [`lock_default_device`].
static DEFAULT_DEVICE_LOCK: Mutex<()> = Mutex::new(());

/// The span held by [`lock_default_device`], released on drop, including
/// while a panic unwinds.
#[derive(Debug)]
pub struct DefaultDeviceLock {
    _guard: MutexGuard<'static, ()>,
}

impl DefaultDeviceGuard {
    /// Record the current default device without moving it.
    pub fn capture() -> Self {
        LIVE_DEVICE_GUARDS.fetch_add(1, Ordering::AcqRel);
        Self {
            previous_is_gpu: ffi::default_device_is_gpu(),
        }
    }

    /// Move the default device to the CPU until the guard drops.
    pub fn cpu() -> Self {
        let guard = Self::capture();
        ffi::set_default_device(false);
        guard
    }

    /// Move the default device to the GPU until the guard drops. A no-op on
    /// a build without a GPU backend, where the default device stays put.
    pub fn gpu() -> Self {
        let guard = Self::capture();
        if ffi::gpu_backend_available() {
            ffi::set_default_device(true);
        }
        guard
    }
}

impl Drop for DefaultDeviceGuard {
    fn drop(&mut self) {
        ffi::set_default_device(self.previous_is_gpu);
        // Decrement after the restore, so an observer that sees zero live
        // guards also sees the restored device rather than a half-dropped one.
        LIVE_DEVICE_GUARDS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Legacy helper — create a non-thread-local GPU stream.
///
/// Kept for backward compatibility with any external user of the
/// `mlxcel_core::streams` module that has not yet migrated to the
/// thread-local API. New code should call
/// [`new_thread_local_generation_stream`] instead.
///
/// Used by: external crates only; in-tree generation owners now use
/// [`new_thread_local_generation_stream`].
#[deprecated(
    since = "26.5.9",
    note = "Use `new_thread_local_generation_stream` so dispatch and synchronization stay on the same per-thread stream."
)]
pub fn new_generation_stream() -> Option<UniquePtr<MlxStream>> {
    if ffi::default_device_is_gpu() {
        Some(ffi::new_gpu_stream())
    } else {
        None
    }
}

/// Legacy helper — install a previously created `MlxStream` as the
/// default stream.
///
/// Used together with the deprecated
/// [`new_generation_stream`]. Prefer
/// [`install_thread_local_default_stream`] in new code.
#[deprecated(
    since = "26.5.9",
    note = "Use `install_thread_local_default_stream` to bind the per-thread MLX stream."
)]
pub fn install_default_stream(stream: Option<&UniquePtr<MlxStream>>) {
    if let Some(stream) = stream {
        ffi::set_default_stream(stream);
    }
}

// --- Index-aware multi-GPU device/stream helpers (epic #486, sub-issue #487) ---
//
// The boolean device API targets GPU index 0 only. The helpers below let a
// caller place a stream or the default device on a specific GPU, validating
// the index against the backend's reported device count. They are the
// foundation for per-rank device placement in single-node tensor
// parallelism; the consuming TP runtime lands in sub-issue #488.

/// Error returned by the index-aware GPU helpers when the requested GPU
/// index is outside `0..gpu_device_count()`.
///
/// On a single-GPU backend (Metal/Apple, CPU-only) only index 0 is valid, so
/// any `index > 0` produces this error. On a multi-GPU CUDA host the valid
/// range widens to the real adapter count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidGpuIndex {
    /// The out-of-range index that was requested.
    pub requested: i32,
    /// The number of usable GPUs reported by the active backend.
    pub device_count: i32,
}

impl std::fmt::Display for InvalidGpuIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU index {} is out of range: {} GPU(s) available (valid indices 0..{})",
            self.requested, self.device_count, self.device_count
        )
    }
}

impl std::error::Error for InvalidGpuIndex {}

/// Number of usable GPUs reported by the active MLX backend.
///
/// Portable across backends via MLX's `device_count(DeviceType::gpu)`: Metal
/// reports 1 (single unified-memory GPU), a CUDA build reports the real
/// adapter count, and a CPU-only build clamps to 1. Always `>= 1`.
pub fn gpu_device_count() -> i32 {
    ffi::gpu_device_count()
}

/// Validate that `index` names a usable GPU for the active backend.
///
/// Returns [`InvalidGpuIndex`] for a negative index or one at/above
/// [`gpu_device_count`].
fn check_gpu_index(index: i32) -> Result<(), InvalidGpuIndex> {
    let device_count = ffi::gpu_device_count();
    if index < 0 || index >= device_count {
        return Err(InvalidGpuIndex {
            requested: index,
            device_count,
        });
    }
    Ok(())
}

/// Create a new MLX stream pinned to GPU `index` (0-based).
///
/// Index-aware sibling of the boolean `new_stream_on_device`. On a
/// single-GPU backend only index 0 is valid; `index > 0` returns
/// [`InvalidGpuIndex`]. On a multi-GPU CUDA host any index in
/// `0..gpu_device_count()` is valid. This is the foundation for placing each
/// tensor-parallel rank's compute on its own GPU (epic #486).
pub fn new_stream_on_gpu(index: i32) -> Result<UniquePtr<MlxStream>, InvalidGpuIndex> {
    check_gpu_index(index)?;
    Ok(ffi::new_stream_on_gpu_index(index))
}

/// Make GPU `index` the default device for subsequent MLX ops.
///
/// Validates `index` against [`gpu_device_count`]; index 0 is always valid.
/// Mirrors the boolean `set_default_device` but targets a specific GPU.
pub fn set_default_gpu_device(index: i32) -> Result<(), InvalidGpuIndex> {
    check_gpu_index(index)?;
    ffi::set_default_device_index(index);
    Ok(())
}

/// Create a thread-local MLX stream pinned to GPU `index`.
///
/// Index-aware sibling of [`new_thread_local_generation_stream`]; validates
/// `index` against [`gpu_device_count`]. Intended for the multi-GPU TP
/// runtime (sub-issue #488) so each rank's worker thread dispatches on its
/// own GPU.
pub fn new_thread_local_stream_on_gpu(
    index: i32,
) -> Result<UniquePtr<MlxThreadLocalStream>, InvalidGpuIndex> {
    check_gpu_index(index)?;
    Ok(ffi::new_thread_local_stream_gpu_index(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Index-aware multi-GPU helpers (epic #486, sub-issue #487) ---

    /// The portable device count is always at least one, on every backend.
    #[test]
    fn gpu_device_count_is_at_least_one() {
        assert!(
            gpu_device_count() >= 1,
            "gpu_device_count must report at least one usable GPU"
        );
    }

    /// A stream can be created on every valid GPU index `0..count`. On Metal
    /// this exercises only index 0; on a multi-GPU CUDA host it covers all.
    #[test]
    fn new_stream_on_gpu_accepts_every_valid_index() {
        let count = gpu_device_count();
        for index in 0..count {
            let stream = new_stream_on_gpu(index)
                .unwrap_or_else(|e| panic!("index {index} should be valid: {e}"));
            assert!(
                !stream.is_null(),
                "stream for index {index} must be non-null"
            );
        }
    }

    /// Out-of-range indices (at/above the count, and negative) return the
    /// typed error carrying the offending index and the device count.
    #[test]
    fn new_stream_on_gpu_rejects_out_of_range() {
        let count = gpu_device_count();
        match new_stream_on_gpu(count) {
            Ok(_) => panic!("index == count ({count}) must be rejected"),
            Err(e) => {
                assert_eq!(e.requested, count);
                assert_eq!(e.device_count, count);
            }
        }
        match new_stream_on_gpu(-1) {
            Ok(_) => panic!("negative index must be rejected"),
            Err(e) => {
                assert_eq!(e.requested, -1);
                assert_eq!(e.device_count, count);
            }
        }
    }

    /// `set_default_gpu_device` validates the index: index 0 always succeeds,
    /// an at-count index errors. A [`DefaultDeviceGuard`] puts the previous
    /// default device back so the process-global default is not left mutated
    /// for sibling tests.
    #[test]
    fn set_default_gpu_device_validates_index() {
        let _lock = lock_default_device();
        let _device = DefaultDeviceGuard::capture();
        let count = gpu_device_count();
        assert!(
            set_default_gpu_device(count).is_err(),
            "index == count ({count}) must be rejected"
        );
        // Index 0 is the GPU that is already the default on a single-GPU
        // backend, so installing it leaves the process as-is; the guard
        // restores the previous default either way.
        set_default_gpu_device(0).expect("index 0 is always valid");
    }

    /// The thread-local stream factory validates indices the same way.
    #[test]
    fn new_thread_local_stream_on_gpu_validates_index() {
        let count = gpu_device_count();
        let tls = new_thread_local_stream_on_gpu(0).expect("index 0 is always valid");
        assert!(!tls.is_null(), "TLS handle for index 0 must be non-null");
        assert!(
            new_thread_local_stream_on_gpu(count).is_err(),
            "index == count ({count}) must be rejected"
        );
    }

    /// The typed error renders a human-readable, actionable message naming
    /// both the bad index and the available device count.
    #[test]
    fn invalid_gpu_index_display_is_actionable() {
        let err = InvalidGpuIndex {
            requested: 3,
            device_count: 1,
        };
        let msg = err.to_string();
        assert!(
            msg.contains('3'),
            "message should name the bad index: {msg}"
        );
        assert!(
            msg.contains('1'),
            "message should name the device count: {msg}"
        );
    }

    /// CUDA-gated: a trivial op runs on a non-default GPU index and produces
    /// the correct result. Compiled only under the `cuda` feature (no Metal
    /// build risk) and skipped on a single-GPU CUDA host. Authored for a
    /// multi-GPU CUDA box; not exercised on this Apple/Metal machine.
    #[cfg(feature = "cuda")]
    #[test]
    fn trivial_op_runs_on_non_default_gpu_index() {
        let _lock = lock_default_device();
        let count = gpu_device_count();
        if count < 2 {
            // Single-GPU CUDA host: no second device to target.
            return;
        }
        // Pin the default device to GPU 1, compute 1 + 1 there, then restore.
        set_default_gpu_device(1).expect("index 1 valid on a multi-GPU host");
        let a = ffi::ones(&[1], crate::dtype::FLOAT32);
        let b = ffi::ones(&[1], crate::dtype::FLOAT32);
        let c = ffi::add(&a, &b);
        ffi::eval(&c);
        let value = ffi::item_f32(&c);
        set_default_gpu_device(0).expect("index 0 is always valid");
        assert_eq!(value, 2.0, "1 + 1 on GPU 1 should equal 2");
    }

    // --- Backend availability vs. the default device (issue #1421) ---

    /// `gpu_backend_available` answers whether a GPU backend exists, and that
    /// answer does not move with the default device: it is the same before,
    /// during and after a CPU guard. On Apple Silicon (Metal) it is `true`;
    /// on a build with neither Metal nor CUDA it is `false`.
    #[test]
    fn gpu_backend_available_does_not_track_the_default_device() {
        let _lock = lock_default_device();
        let available = ffi::gpu_backend_available();
        if crate::metal_is_available() {
            assert!(
                available,
                "Metal is a GPU backend, so Apple Silicon must report one"
            );
        }
        if !crate::metal_is_available() && !crate::cuda_is_available() {
            assert!(!available, "a CPU-only build has no GPU backend to report");
        }
        let baseline = ffi::default_device_is_gpu();
        {
            let _cpu = DefaultDeviceGuard::cpu();
            assert!(
                !ffi::default_device_is_gpu(),
                "the guard moves the default device to the CPU"
            );
            assert_eq!(
                ffi::gpu_backend_available(),
                available,
                "moving the default device must not change whether a backend exists"
            );
        }
        assert_eq!(
            ffi::default_device_is_gpu(),
            baseline,
            "the guard restores the default device when it drops"
        );
        assert_eq!(ffi::gpu_backend_available(), available);
    }

    /// Nested guards restore in reverse order: the inner guard returns to the
    /// value the outer one installed, and the outer returns to the original.
    #[test]
    fn default_device_guards_nest_and_restore_in_order() {
        let _lock = lock_default_device();
        let baseline = ffi::default_device_is_gpu();
        {
            let _outer = DefaultDeviceGuard::cpu();
            assert!(!ffi::default_device_is_gpu());
            {
                let _inner = DefaultDeviceGuard::gpu();
                // `gpu()` moves to the GPU when a backend exists and is a
                // no-op otherwise, so the default tracks availability here.
                assert_eq!(ffi::default_device_is_gpu(), ffi::gpu_backend_available());
            }
            assert!(
                !ffi::default_device_is_gpu(),
                "the inner guard restores the outer's CPU default"
            );
        }
        assert_eq!(ffi::default_device_is_gpu(), baseline);
    }

    /// `capture` records without moving, so code that changes the device
    /// through another path (here the raw boolean setter) is still put back.
    #[test]
    fn default_device_guard_capture_restores_without_moving() {
        let _lock = lock_default_device();
        let baseline = ffi::default_device_is_gpu();
        {
            let _device = DefaultDeviceGuard::capture();
            assert_eq!(
                ffi::default_device_is_gpu(),
                baseline,
                "capture must not move the device"
            );
            ffi::set_default_device(false);
            assert!(!ffi::default_device_is_gpu());
        }
        assert_eq!(ffi::default_device_is_gpu(), baseline);
    }

    /// `Drop` runs while a panic unwinds, so a failing test that holds a
    /// guard does not leave the CPU default behind for the modules after it.
    #[test]
    fn default_device_guard_restores_while_unwinding() {
        let _lock = lock_default_device();
        let baseline = ffi::default_device_is_gpu();
        let outcome = std::panic::catch_unwind(|| {
            let _cpu = DefaultDeviceGuard::cpu();
            assert!(!ffi::default_device_is_gpu());
            panic!("simulated test failure while the guard is held");
        });
        assert!(outcome.is_err(), "the closure must have panicked");
        assert_eq!(
            ffi::default_device_is_gpu(),
            baseline,
            "the guard must restore the device during unwinding"
        );
    }

    /// The live-guard count rises and falls with the guards themselves, and a
    /// guard's restore has already run by the time the count reaches zero, so
    /// a reader that sees no live guard also sees the restored device. This is
    /// what lets a leak assertion elsewhere tell a leak from a guard that a
    /// concurrent test is legitimately holding.
    #[test]
    fn live_device_guards_are_counted_until_the_device_is_restored() {
        let _lock = lock_default_device();
        let baseline = default_device_guards_held();
        {
            let _outer = DefaultDeviceGuard::capture();
            assert_eq!(default_device_guards_held(), baseline + 1);
            {
                let _inner = DefaultDeviceGuard::cpu();
                assert_eq!(default_device_guards_held(), baseline + 2);
            }
            assert_eq!(default_device_guards_held(), baseline + 1);
        }
        assert_eq!(
            default_device_guards_held(),
            baseline,
            "every guard must be accounted for once it has dropped"
        );
    }

    /// The deprecated name still compiles (with a warning, silenced here on
    /// purpose) and keeps its original default-device meaning.
    #[test]
    #[allow(deprecated)]
    fn deprecated_is_gpu_available_keeps_the_default_device_meaning() {
        let _lock = lock_default_device();
        assert_eq!(crate::is_gpu_available(), ffi::default_device_is_gpu());
        let _cpu = DefaultDeviceGuard::cpu();
        assert!(!crate::is_gpu_available());
    }

    /// Smoke test: the TLS handle factory either succeeds (GPU build)
    /// or returns `None` cleanly (CPU-only build).
    #[test]
    fn new_thread_local_generation_stream_is_total() {
        let _ = new_thread_local_generation_stream();
    }

    /// `new_generation_stream` (the deprecated, non-thread-local
    /// helper) keeps working. We intentionally still exercise it so
    /// that any external consumer that imports it continues to build.
    #[test]
    #[allow(deprecated)]
    fn legacy_new_generation_stream_is_total() {
        let _ = new_generation_stream();
    }

    /// Resolving the same TLS handle twice on the same thread returns
    /// `MlxStream` wrappers that are distinct allocations (each call
    /// produces a fresh `unique_ptr`) but represent the same underlying
    /// per-thread MLX stream. We assert the allocations are independent
    /// by pointer comparison; MLX's per-thread invariant is provided by
    /// upstream and validated by upstream's own test suite.
    ///
    /// Skipped on CPU-only builds (no TLS handle to resolve).
    #[test]
    fn resolved_stream_wrappers_are_independent_allocations() {
        let Some(tls) = new_thread_local_generation_stream() else {
            // CPU-only build: nothing to verify.
            return;
        };
        let stream_a = ffi::stream_from_thread_local_stream(&tls);
        let stream_b = ffi::stream_from_thread_local_stream(&tls);

        // Two `make_unique` calls in the C++ resolver always produce
        // distinct heap allocations for the wrapper struct, so a
        // different address is the cheapest structural check that the
        // bridge is genuinely returning two separately owned wrappers
        // (rather than, say, alias-aliasing a single static handle).
        let ptr_a: *const MlxStream = stream_a.as_ref().expect("stream A non-null");
        let ptr_b: *const MlxStream = stream_b.as_ref().expect("stream B non-null");
        assert_ne!(
            ptr_a, ptr_b,
            "stream_from_thread_local_stream must produce independent wrappers per call"
        );
    }

    /// Round-trip: resolving the handle on the main thread, installing
    /// it as default, dispatching a tiny op, and synchronizing through
    /// the same handle does not panic and leaves MLX in a usable state
    /// for subsequent dispatches.
    ///
    /// A [`DefaultStreamGuard`] captures the previous default stream
    /// before installation so that the per-thread state is restored on
    /// exit, regardless of whether the test panics. This prevents state
    /// from leaking into other tests that run on the same thread.
    ///
    /// Skipped on CPU-only builds.
    #[test]
    fn install_and_synchronize_round_trip_works() {
        let Some(tls) = new_thread_local_generation_stream() else {
            return;
        };
        // Capture previous default stream; restored on drop.
        let _guard = DefaultStreamGuard::capture();
        install_thread_local_default_stream(Some(&tls));
        // A trivial op on the now-installed default stream — verifies
        // that the resolved stream is wired correctly into MLX's
        // dispatch system.
        let arr = ffi::zeros(&[1, 1], crate::dtype::FLOAT32);
        ffi::eval(&arr);
        synchronize_thread_local_stream(Some(&tls));
    }

    /// The TLS-backed install path is a no-op when handed `None` (the
    /// CPU-only build case). It must not panic, and must not leave
    /// MLX's default stream in a bad state — a follow-up trivial op
    /// must still succeed.
    #[test]
    fn install_thread_local_default_stream_is_noop_on_none() {
        install_thread_local_default_stream(None);
        synchronize_thread_local_stream(None);
        // Sanity: MLX is still usable.
        let arr = ffi::zeros(&[1, 1], crate::dtype::FLOAT32);
        ffi::eval(&arr);
    }
}
