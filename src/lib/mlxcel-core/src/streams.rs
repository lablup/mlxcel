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

use crate::ffi;
use crate::ffi::{MlxStream, MlxThreadLocalStream};
use crate::UniquePtr;

/// Create a thread-local generation stream bound to the GPU device.
///
/// Returns `None` on CPU-only builds (where `is_gpu_available()` is
/// false) so callers can fall back to the MLX default stream.
///
/// The returned [`MlxThreadLocalStream`] handle is independent of the
/// thread it was created on — pass it to
/// [`install_thread_local_default_stream`] from any thread that wants
/// to bind its dedicated per-thread stream as the default for
/// subsequent MLX dispatches on that thread.
///
/// Used by: CxxGenerator, SpeculativeGenerator, BatchScheduler, AudioWorker
pub fn new_thread_local_generation_stream() -> Option<UniquePtr<MlxThreadLocalStream>> {
    if ffi::is_gpu_available() {
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
    if ffi::is_gpu_available() {
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

#[cfg(test)]
mod tests {
    use super::*;

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
