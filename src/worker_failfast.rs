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

//! Fail-fast wrapper shared by every core inference worker thread.
//!
//! Lives at the crate root so both the server worker threads
//! (`crate::server::model_provider::model_worker`) and the remote pipeline
//! stage service thread (`crate::distributed::pipeline::remote_service`) can
//! re-impose the same fail-fast posture under the release `panic = "unwind"`
//! policy (issue #375). See [ADR 0003](../docs/adr/0003-release-panic-unwind-with-core-thread-abort.md).

/// Run a core inference worker thread body, converting an uncaught panic into a
/// clean process abort while forwarding the body's normal return value.
///
/// The release profile uses `panic = "unwind"` (issue #375) so the deliberate
/// `catch_unwind` request-isolation backstop works in production: a synthesis
/// panic on the audio worker becomes a per-request error, not a process abort.
/// That same unwind policy would let a panic on a core generation thread unwind
/// out of the worker loop and silently kill the thread, leaving the process
/// alive but unable to generate. A panic in a core worker means an invariant is
/// broken, so we re-impose fail-fast here: log without request content, then
/// `abort` so a supervisor restarts into fresh state.
///
/// Every core inference thread routes through this helper: the local batched
/// worker, the local legacy `--no-batch` worker, and the remote pipeline stage
/// service thread (which runs core forward passes for a remote stage and would
/// otherwise unwind into a zombie stage process while `serve_remote_pipeline_stage`
/// stays parked on `ctrl_c`). The catch is about PANICS only: a body that
/// returns normally, including a recoverable `Err`, has its value forwarded
/// unchanged to the caller, so an `Err` does not abort.
///
/// The only deliberately CONTAINED boundary is the audio worker `run_guarded`,
/// whose own `catch_unwind` turns a synthesis or transcription panic into a
/// per-request error; it is NOT wrapped here. There is deliberately no global
/// abort panic hook, which would run before unwinding and defeat that backstop.
/// `AssertUnwindSafe` is correct precisely because we abort (never continue) on
/// a caught panic, so no torn state is ever observed.
pub(crate) fn run_core_thread_or_abort<F, R>(label: &str, body: F) -> R
where
    F: FnOnce() -> R,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            tracing::error!(
                target: "mlxcel::worker",
                "core worker thread '{label}' panicked; aborting to preserve fail-fast"
            );
            std::process::abort();
        }
    }
}
