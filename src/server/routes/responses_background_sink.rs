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

//! Terminal recording for background Responses jobs (issue #2898):
//! [`JobRecorder`] writes a job's one terminal object into the response
//! store, and [`BackgroundSink`] is the
//! [`ResponseEventSink`] a background generation runs with. The job
//! lifecycle is described in [`super::responses_background`].

use std::sync::Arc;

use crate::server::AppState;
use crate::server::responses_store::ResponsesStore;
use crate::server::responses_translator::TranslatedRequest;
use crate::server::types::ErrorResponse;
use crate::server::types::responses_request::CreateResponseRequest;
use crate::server::types::responses_response::{ResponseErrorBody, ResponseObject, ResponseStatus};
use crate::server::types::responses_stream::ResponseStreamEvent;

use super::responses::append_conversation_turn;
use super::responses_run::{PreparedRun, ResponseEventSink};

/// `error.code` of a failed job whose cause is the server.
const CODE_SERVER_ERROR: &str = "server_error";
/// `error.code` of a failed job refused because the queue was full.
const CODE_SERVER_OVERLOADED: &str = "server_overloaded";
/// `error.code` of a failed job whose prompt could not be built: a media
/// read, a render or a compaction refused the request.
const CODE_INVALID_PROMPT: &str = "invalid_prompt";

/// The `error.code` a failed job reports for a refusal the foreground would
/// have answered with `error`.
fn failure_code(error: &ErrorResponse) -> &'static str {
    if error.status.is_client_error() {
        CODE_INVALID_PROMPT
    } else if error.status == axum::http::StatusCode::SERVICE_UNAVAILABLE
        && error.error.error_type != "server_error"
    {
        CODE_SERVER_OVERLOADED
    } else {
        CODE_SERVER_ERROR
    }
}

/// Records a job's terminal object, at most once.
///
/// Shared by the job task and its [`BackgroundSink`]. When the last handle
/// is dropped while the job is still active (a panic in the job task or in
/// the driver, or a runtime shutting down), `Drop` records `failed`, so no
/// job stays active forever.
pub(crate) struct JobRecorder {
    store: Arc<ResponsesStore>,
    response_id: String,
    /// The `queued` object: a terminal object the driver did not build
    /// (`failed`, or `cancelled` before generation) is this with its status
    /// replaced.
    template: ResponseObject,
}

impl JobRecorder {
    /// The recorder of the job admitted as `template` (its `queued` object).
    ///
    /// Used by: `spawn_background_job_with`.
    pub(super) fn new(
        store: Arc<ResponsesStore>,
        response_id: String,
        template: ResponseObject,
    ) -> Self {
        Self {
            store,
            response_id,
            template,
        }
    }

    pub(super) fn response_id(&self) -> &str {
        &self.response_id
    }

    /// Whether no terminal object has been recorded yet.
    pub(super) fn is_active(&self) -> bool {
        self.store.background_status(&self.response_id).is_some()
    }

    /// Replace the input items the terminal object is stored with, after
    /// automatic compaction rewrote them.
    pub(super) fn set_input_items(
        &self,
        input_items: Vec<crate::server::types::responses_request::ResponseInputItem>,
    ) {
        self.store
            .set_background_input_items(&self.response_id, input_items);
    }

    /// Record `failed` with `error: { code, message }`. A cancel wins: when
    /// one was acknowledged, or requested through
    /// `request_background_cancel` or `cancel_background` (a generation
    /// stopped before its first token reports that as an error), the job is
    /// recorded `cancelled`, so a cancel never ends a job `failed`. The
    /// job's token alone does not count: a refused admission sets it too.
    ///
    /// A no-op once a terminal object is recorded.
    pub(super) fn fail(&self, code: &str, message: &str) {
        let cancel_acknowledged = self.store.finish_in_flight(&self.response_id);
        if !self.is_active() {
            return;
        }
        let cancel_requested = self.store.background_cancel_requested(&self.response_id);
        let mut terminal = self.template.clone();
        if cancel_acknowledged || cancel_requested {
            terminal.status = ResponseStatus::Cancelled;
        } else {
            tracing::warn!(
                response_id = %self.response_id,
                code,
                "background response failed: {message}"
            );
            terminal.status = ResponseStatus::Failed;
            terminal.error = Some(ResponseErrorBody {
                code: code.to_string(),
                message: message.to_string(),
            });
        }
        self.store.finish_background(&self.response_id, terminal);
    }

    /// Record `failed` for a refusal the foreground would have answered
    /// with `error`.
    pub(super) fn fail_with(&self, error: &ErrorResponse) {
        self.fail(failure_code(error), &error.error.message);
    }

    /// Record `failed` with `error.code: "server_error"`.
    pub(super) fn fail_server_error(&self, message: &str) {
        self.fail(CODE_SERVER_ERROR, message);
    }

    /// Record `cancelled` for a job cancelled before generation started.
    pub(super) fn cancel_before_start(&self) {
        self.store.finish_in_flight(&self.response_id);
        let mut terminal = self.template.clone();
        terminal.status = ResponseStatus::Cancelled;
        self.store.finish_background(&self.response_id, terminal);
    }
}

impl Drop for JobRecorder {
    fn drop(&mut self) {
        if self.is_active() {
            self.fail_server_error(
                "The background response ended unexpectedly; check the server log.",
            );
        }
        // The cancel route can record the terminal object while the job
        // still runs; a job that then panics would otherwise never retire
        // its cancel registration.
        self.store.finish_in_flight(&self.response_id);
    }
}

/// The sink of a background generation: events are dropped, the first
/// progress callback moves the job to `in_progress`, and the terminal
/// response is recorded in the response store.
///
/// It holds no client channel, so nothing a client does to its connection
/// can cancel the generation.
pub(crate) struct BackgroundSink {
    recorder: Arc<JobRecorder>,
    state: AppState,
    request: Arc<CreateResponseRequest>,
    translated: Arc<TranslatedRequest>,
}

impl BackgroundSink {
    /// The sink for `run`, recording through `recorder`.
    ///
    /// Used by: the background job task.
    pub(crate) fn new(run: &PreparedRun, recorder: Arc<JobRecorder>) -> Self {
        Self {
            recorder,
            state: run.state.clone(),
            request: Arc::clone(&run.request),
            translated: Arc::clone(&run.translated),
        }
    }
}

impl ResponseEventSink for BackgroundSink {
    fn emit(&self, event: &ResponseStreamEvent) {
        // The driver reports a failed generation only as this event.
        if let ResponseStreamEvent::Error { code, message, .. } = event {
            self.recorder.fail(code, message);
        }
    }

    fn on_started(&self) {
        self.recorder
            .store
            .mark_background_in_progress(&self.recorder.response_id);
    }

    fn record_terminal(&self, response: &ResponseObject) {
        let recorded = self
            .recorder
            .store
            .finish_background(&self.recorder.response_id, response.clone());
        // The conversation gets this turn unless nothing was generated (a
        // cancel before the first token), the store kept a `failed` object
        // in place of an oversized one, or the cancel route recorded the
        // terminal object first (`cancelled` without output; this write was
        // then a no-op).
        let generated =
            !(recorded.status == ResponseStatus::Cancelled && recorded.output.is_empty());
        if self.translated.effective_store && generated && recorded.status == response.status {
            append_conversation_turn(&self.state, &self.request, &self.translated, &recorded);
        }
    }

    fn finish_registration(&self) -> bool {
        self.recorder
            .store
            .finish_in_flight(&self.recorder.response_id)
    }

    fn generation_finished(&self) {}
}
