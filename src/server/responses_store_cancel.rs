// Copyright 2025-2026 Lablup Inc.
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

//! Cancellation of background jobs for `POST /v1/responses/{id}/cancel`
//! (issue #2899).
//!
//! The cancel route answers with the job's final object, so it needs more
//! than [`ResponsesStore::request_background_cancel`], which only flips the
//! token:
//!
//! - [`ResponsesStore::cancel_background`] flips the token, acknowledges the
//!   cancel, and finishes a job that is still `queued` in the same critical
//!   section, so a job cannot start generating between the status check and
//!   the terminal write.
//! - [`ResponsesStore::background_cancel_requested`] tells the job's
//!   recorder that a failure followed a cancel.
//! - [`ResponsesStore::force_background_cancel`] is the fallback for a job
//!   that did not record a terminal object in time.
//!
//! - [`ResponsesStore::fail_active_background`] ends every active job as
//!   `failed` when the model that runs them is unloaded (issue #2901).
//!
//! All of them write through the same path as
//! [`ResponsesStore::finish_background`], so the first terminal object wins
//! and a later write by the job is a no-op.

use std::sync::atomic::Ordering;
use std::time::Instant;

use super::{BackgroundCancel, BackgroundState, ResponsesStore};
use crate::server::store_budget::serialized_json_len_saturating;
use crate::server::types::responses_response::{ResponseErrorBody, ResponseObject, ResponseStatus};

impl ResponsesStore {
    /// Cancel a job on behalf of the cancel route.
    ///
    /// - An active job gets its token flipped and the cancel acknowledged in
    ///   the in-flight registry (when the job is still registered there), so
    ///   a generation the cancel stops before its first token ends
    ///   `cancelled`, not with an error.
    /// - A `queued` job is finished at once as `cancelled` with no output:
    ///   the result is [`BackgroundCancel::Terminal`] with the recorded
    ///   object. Its task skips generation, or stops at its next token check.
    /// - An `in_progress` job stays active:
    ///   [`BackgroundCancel::Requested`]. The caller waits for the job's own
    ///   terminal object with [`Self::wait_background_terminal`], which keeps
    ///   the partial output, and falls back to
    ///   [`Self::force_background_cancel`].
    /// - A finished response returns its retained object, whatever its
    ///   status; an unknown id returns [`BackgroundCancel::NotFound`].
    ///
    /// Used by: `cancel_background_response` (the cancel route).
    pub fn cancel_background(&self, id: &str) -> BackgroundCancel {
        let now = Instant::now();
        let mut jobs = self.lock_jobs();
        let Some(job) = jobs.jobs.get_mut(id) else {
            let mut state = self.write_inner();
            return match Self::retained_response_locked(&mut state, &self.config, id, now) {
                Some(response) => BackgroundCancel::Terminal(response),
                None => BackgroundCancel::NotFound,
            };
        };
        job.cancel_requested = true;
        job.token.store(true, Ordering::Release);
        // Lock order: `jobs`, then `in_flight` (read).
        self.cancel_in_flight(id);
        if job.snapshot.status != ResponseStatus::Queued {
            return BackgroundCancel::Requested;
        }
        match self.record_cancelled_locked(jobs, id, now) {
            Some(recorded) => BackgroundCancel::Terminal(recorded),
            None => BackgroundCancel::NotFound,
        }
    }

    /// Whether a cancel was requested for the active job `id` through
    /// [`Self::request_background_cancel`] or [`Self::cancel_background`].
    /// `false` once the job is terminal or for an unknown id.
    ///
    /// The job's token is not a substitute: a refused admission sets it too.
    ///
    /// Used by: `JobRecorder::fail`, so a failure the cancel itself caused is
    /// recorded `cancelled`.
    pub fn background_cancel_requested(&self, id: &str) -> bool {
        self.lock_jobs()
            .jobs
            .get(id)
            .is_some_and(|job| job.cancel_requested)
    }

    /// Finish an active job as `cancelled` with the output of its snapshot
    /// (none), for a job that ignored its token past the cancel route's wait.
    /// Returns the recorded object: this `cancelled` one, or the object that
    /// was recorded first when the job finished in the meantime. `None` when
    /// neither an active job nor a retained object has this id.
    ///
    /// The job's later terminal write is a no-op.
    ///
    /// Used by: `cancel_background_response` (the cancel route).
    pub fn force_background_cancel(&self, id: &str) -> Option<ResponseObject> {
        let now = Instant::now();
        let jobs = self.lock_jobs();
        if jobs.jobs.contains_key(id) {
            return self.record_cancelled_locked(jobs, id, now);
        }
        let mut state = self.write_inner();
        Self::retained_response_locked(&mut state, &self.config, id, now)
    }

    /// End every active job as `failed` with `code` and `message`, and
    /// return how many were ended. Each job's token is set, so a generation
    /// still running stops at its next token check, and its later terminal
    /// write is a no-op. The recorded object is the job's snapshot, without
    /// the output generated so far.
    ///
    /// After this returns, [`Self::active_background_jobs`] reads 0 unless a
    /// job was admitted concurrently; the caller stops admission first.
    ///
    /// Used by: the router pool's unload, before it stops the model worker
    /// (issue #2901).
    pub fn fail_active_background(&self, code: &str, message: &str) -> usize {
        let mut failed = 0usize;
        loop {
            let now = Instant::now();
            let mut jobs = self.lock_jobs();
            let Some(id) = jobs.jobs.keys().next().cloned() else {
                return failed;
            };
            let Some(job) = jobs.jobs.remove(&id) else {
                return failed;
            };
            job.token.store(true, Ordering::Release);
            let mut terminal = job.snapshot.clone();
            terminal.status = ResponseStatus::Failed;
            terminal.completed_at = None;
            terminal.error = Some(ResponseErrorBody {
                code: code.to_string(),
                message: message.to_string(),
            });
            // The snapshot carries no output, so sizing it under the lock is
            // cheap. The guard is consumed by the terminal write.
            let terminal_bytes = serialized_json_len_saturating(&terminal);
            self.record_terminal_locked(jobs, &id, job, terminal, terminal_bytes, None, now);
            failed += 1;
        }
    }

    /// Remove the active job `id` and record a `cancelled` copy of its
    /// snapshot as its terminal object.
    fn record_cancelled_locked(
        &self,
        mut jobs: std::sync::MutexGuard<'_, BackgroundState>,
        id: &str,
        now: Instant,
    ) -> Option<ResponseObject> {
        let job = jobs.jobs.remove(id)?;
        job.token.store(true, Ordering::Release);
        let mut terminal = job.snapshot.clone();
        terminal.status = ResponseStatus::Cancelled;
        terminal.completed_at = None;
        // The snapshot is the `queued` object without output, so this is
        // cheap enough to size under the lock.
        let terminal_bytes = serialized_json_len_saturating(&terminal);
        // No prepared stream event: the terminal event of a streamed job is
        // serialized from the recorded object under the log lock.
        Some(self.record_terminal_locked(jobs, id, job, terminal, terminal_bytes, None, now))
    }
}
