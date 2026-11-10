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

//! Background mode for `POST /v1/responses` (issue #2898).
//!
//! A `background: true` create returns a `queued` response at once and runs
//! the generation as a detached job; `GET /v1/responses/{id}` polls it.
//!
//! ## Lifecycle
//!
//! 1. [`create_background_response`] refuses a full queue, then
//!    [`spawn_background_job`] admits the job into the
//!    [`ResponsesStore`](crate::server::responses_store::ResponsesStore)
//!    (`queued`), registers its cancellation token and spawns the job task.
//! 2. The job task runs the work the foreground handler does before it
//!    answers but that can take long: the video-frames fallback, automatic
//!    compaction and [`prepare_response_run`]. A failure there ends the job
//!    as `failed`.
//! 3. [`drive_response_run`] runs on the blocking pool with a
//!    [`BackgroundSink`]. The first prefill-progress or token callback moves
//!    the job to `in_progress`.
//! 4. Exactly one terminal object is recorded through
//!    `ResponsesStore::finish_background`: the driver's final response
//!    (`completed`, `incomplete`, `cancelled`) or a `failed` object.
//!
//! ## Streaming
//!
//! A job created with `stream: true` (issue #2900) gets a
//! [`BackgroundEventLog`] at admission, already holding `response.created`
//! (status `queued`) and `response.queued`. The driver appends its events
//! through the [`BackgroundSink`], starting with `response.in_progress` at
//! the first progress callback, and `ResponsesStore::finish_background`
//! appends the one terminal event. The create call and every
//! `GET /v1/responses/{id}?stream=true` read that log through
//! [`super::responses_background_stream`].
//!
//! ## Cancellation token
//!
//! The token `admit_background` returns is the job's only cancellation flag.
//! It is registered in the store's in-flight registry under the response id
//! before the create call returns, and it is the token handed to
//! [`drive_response_run`]. `POST /v1/responses/{id}/cancel` flips it through
//! `ResponsesStore::cancel_background`. Nothing ties it to the client connection: dropping
//! the create response or a poll cancels nothing.
//!
//! ## Cancel
//!
//! `POST /v1/responses/{id}/cancel` on a background response answers with
//! the final object, the one a following `GET` returns
//! ([`cancel_background_response`], issue #2899):
//!
//! - `queued`: the route records `cancelled` with no output at once. The job
//!   task drops its pre-generation work (video decode, compaction, request
//!   preparation) as a foreground client disconnect does (a summary pass or
//!   clip decode already running finishes that step), and skips generation
//!   or stops at its next token check.
//! - `in_progress`: the route flips the token and waits up to
//!   [`BACKGROUND_CANCEL_WAIT`] for the driver to record its terminal
//!   object, which keeps the partial output. A job that does not answer in
//!   time is recorded `cancelled` with no output by the route.
//! - The first terminal object wins in every case. When the job finished
//!   `completed`, `incomplete` or `failed` first, the cancel is refused with
//!   400, as for any finished response; an already `cancelled` response is
//!   returned unchanged.
//!
//! The terminal write frees the job slot. The blocking generation may run on
//! until it next checks the token; its late terminal write is a no-op, and it
//! holds no job slot meanwhile.
//!
//! ## No job stays active
//!
//! Every exit records a terminal object. A generation error (including a
//! worker that exited) reaches [`BackgroundSink::emit`] as an `error` event;
//! a panic in the job task or in the driver drops the last [`JobRecorder`]
//! handle, whose `Drop` records `failed` when the job is still active. The
//! recorder and the sink live in [`super::responses_background_sink`].
//!
//! Jobs live in memory only: a process exit loses them.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::server::config::ReasoningBudgetOverride;
use crate::server::responses_background_events::BackgroundEventLog;
use crate::server::responses_store::{
    BackgroundAdmitError, BackgroundCancel, InFlightToken, ResponsesStore,
};
use crate::server::responses_translator::{
    OutboundContext, TranslatedRequest, build_response_object,
};
use crate::server::streaming_responses::CancellationToken;
use crate::server::types::ErrorResponse;
use crate::server::types::responses_request::CreateResponseRequest;
use crate::server::types::responses_response::{ResponseObject, ResponseStatus};
use crate::server::types::responses_stream::ResponseStreamEvent;
use crate::server::{AppState, LiveSettings};

use super::responses_auto_compact::{AutoCompaction, apply_auto_compaction};
pub(crate) use super::responses_background_sink::{BackgroundSink, JobRecorder};
use super::responses_run::{
    PreparedRun, ResponseEventSink, drive_response_run, prepare_response_run,
};

/// Everything a background job needs, built by the create handler once the
/// request passed its synchronous validation.
pub(crate) struct BackgroundJob {
    pub(crate) state: AppState,
    pub(crate) live: Arc<LiveSettings>,
    pub(crate) request: CreateResponseRequest,
    pub(crate) translated: TranslatedRequest,
    pub(crate) response_id: String,
    pub(crate) created_at: f64,
    pub(crate) priority: crate::server::batch::RequestPriority,
    pub(crate) budget_override: ReasoningBudgetOverride,
    pub(crate) structured:
        Option<Arc<std::sync::Mutex<crate::server::structured::StructuredOutputConstraint>>>,
    /// The parsed `context_management` threshold. Automatic compaction is a
    /// full generation pass, so it runs inside the job.
    pub(crate) compact_threshold: Option<usize>,
}

/// Checks a `background: true` request needs beyond the foreground ones,
/// before any other work: nothing here depends on the request content.
///
/// Used by: `create_response`.
pub(crate) fn validate_background_request(
    state: &AppState,
    request: &CreateResponseRequest,
) -> Result<(), ErrorResponse> {
    let Some(store) = state.responses_store.as_ref() else {
        return Err(ErrorResponse::new(
            "background mode needs the response store, which is disabled on this server; \
             restart with --responses-store-max-entries > 0",
            "invalid_request_error",
        ));
    };
    // The same three conditions `admit_background` refuses with `Disabled`,
    // checked here so a full queue cannot turn a 400 into a 503.
    let config = store.config();
    if config.background_max_jobs == 0 || config.max_entries == 0 || config.max_bytes == 0 {
        return Err(background_disabled());
    }
    if request.stream && config.background_event_log_bytes == 0 {
        return Err(ErrorResponse::new(
            "streaming a background response is disabled on this server; restart with \
             --responses-background-event-log-bytes > 0, or set stream to false and poll \
             GET /v1/responses/{id}",
            "invalid_request_error",
        ));
    }
    Ok(())
}

fn background_disabled() -> ErrorResponse {
    ErrorResponse::new(
        "background mode is disabled on this server; restart with \
         --responses-background-max-jobs > 0 and a non-zero response store budget",
        "invalid_request_error",
    )
}

/// Answer a validated `background: true` create: refuse a full queue with
/// the foreground's 503, admit and spawn the job, and return its `queued`
/// object with HTTP 200.
///
/// Used by: `create_response`.
pub(crate) fn create_background_response(job: BackgroundJob) -> Response {
    if !job.state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }
    // The race-safe form of the check above. The slot is released at once:
    // the job reserves its own after the prelude, as the foreground does
    // after compaction, and a queue that filled up meanwhile fails the job.
    if let Err(err) = job.state.model_provider.reserve_single_stream_queue_slot() {
        return super::generation_error_to_response(err).into_response();
    }
    let streamed = job.request.stream.then(|| job.state.clone());
    match spawn_background_job(job) {
        Ok(queued) => match streamed {
            // `stream: true` (issue #2900): the create call is the first
            // reader of the job's event log. `X-Conversation-Id` is not
            // read: the resume cursor is `starting_after`.
            Some(state) => super::responses_background_stream::stream_from_start(&state, queued),
            None => Json(queued).into_response(),
        },
        Err(BackgroundAdmitError::Disabled) => background_disabled().into_response(),
        Err(BackgroundAdmitError::LimitReached { limit }) => rate_limited(format!(
            "{limit} background responses are already active \
             (--responses-background-max-jobs); retry after one finishes"
        ))
        .into_response(),
        Err(BackgroundAdmitError::OverBudget) => rate_limited(
            "the active background responses fill the response store budget \
             (--responses-store-max-bytes); retry after one finishes"
                .to_string(),
        )
        .into_response(),
    }
}

fn rate_limited(message: String) -> ErrorResponse {
    let mut response = ErrorResponse::new(message, "rate_limit_exceeded");
    response.error.code = Some("rate_limit_exceeded".to_string());
    response.status = StatusCode::TOO_MANY_REQUESTS;
    response
}

/// How long a cancel waits for an `in_progress` job to record its own
/// terminal object (with the partial output) before the route records
/// `cancelled` itself. A generation checks its token once per token, so the
/// wait only runs out for a job stuck in one long step.
pub(crate) const BACKGROUND_CANCEL_WAIT: Duration = Duration::from_secs(5);

/// Cancel the background response `id` and answer with its final object, or
/// return `None` when `id` is not a background response (the caller then
/// applies the foreground rules). `wait` bounds the wait for an
/// `in_progress` job; the route passes [`BACKGROUND_CANCEL_WAIT`].
///
/// The reply is always the object a following `GET` returns: the recorded
/// `cancelled` object, or 400 when the response finished with another status
/// first. See the module documentation for each case.
///
/// Used by: `cancel_response`.
pub(crate) async fn cancel_background_response(
    store: &ResponsesStore,
    id: &str,
    wait: Duration,
) -> Option<Response> {
    let recorded = match store.cancel_background(id) {
        BackgroundCancel::NotFound => return None,
        // A stored foreground response keeps the foreground rules.
        BackgroundCancel::Terminal(response) if response.background != Some(true) => return None,
        BackgroundCancel::Terminal(response) => Some(response),
        BackgroundCancel::Requested => match store.wait_background_terminal(id, wait).await {
            Some(response) => Some(response),
            None => {
                tracing::warn!(
                    response_id = %id,
                    "background response did not stop within {wait:?} of its cancel; \
                     recording it as cancelled"
                );
                store.force_background_cancel(id)
            }
        },
    };
    Some(match recorded {
        Some(response) if response.status == ResponseStatus::Cancelled => {
            Json(response).into_response()
        }
        Some(_) => ErrorResponse::new(
            format!("response '{id}' has already finished and cannot be cancelled"),
            "invalid_request_error",
        )
        .into_response(),
        // The job finished and its object was evicted while the cancel
        // waited.
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                format!("response '{id}' not found"),
                "not_found",
            )),
        )
            .into_response(),
    })
}

/// Admit `job` as a `queued` background response and spawn its task.
/// Returns the `queued` object the create call answers with.
///
/// Must be called inside a Tokio runtime.
///
/// Used by: [`create_background_response`].
pub(crate) fn spawn_background_job(
    job: BackgroundJob,
) -> Result<ResponseObject, BackgroundAdmitError> {
    spawn_background_job_with(job, drive_response_run)
}

/// [`spawn_background_job`] with the blocking driver supplied by the caller,
/// so a test can replace [`drive_response_run`].
pub(crate) fn spawn_background_job_with<D>(
    job: BackgroundJob,
    driver: D,
) -> Result<ResponseObject, BackgroundAdmitError>
where
    D: FnOnce(PreparedRun, CancellationToken, Arc<dyn ResponseEventSink>) + Send + 'static,
{
    let Some(store) = job.state.responses_store.clone() else {
        return Err(BackgroundAdmitError::Disabled);
    };
    let queued = queued_response(&job);
    let cancelled = store.admit_background(
        queued.clone(),
        job.translated.canonical_input_items.clone(),
        job.translated.effective_store,
    )?;
    // Registered before the create call answers, so a cancel sent right
    // after it finds the job.
    match store.in_flight_write() {
        Ok(mut registry) => {
            registry.insert(job.response_id.clone(), cancelled.clone());
        }
        Err(poisoned) => {
            poisoned
                .into_inner()
                .insert(job.response_id.clone(), cancelled.clone());
        }
    }
    // `stream: true` (issue #2900): the log exists, with its two opening
    // events, before the create call answers.
    if job.request.stream {
        let log = Arc::new(BackgroundEventLog::new(
            store.config().background_event_log_bytes,
        ));
        log.append(&ResponseStreamEvent::Created {
            sequence_number: 0,
            response: queued.clone(),
        });
        log.append(&ResponseStreamEvent::Queued {
            sequence_number: 1,
            response: queued.clone(),
        });
        store.attach_background_event_log(&job.response_id, log);
    }
    let recorder = Arc::new(JobRecorder::new(
        store,
        job.response_id.clone(),
        queued.clone(),
    ));
    tokio::spawn(run_background_job(job, cancelled, recorder, driver));
    Ok(queued)
}

/// The object a background response shows until it is terminal: no output
/// and zero usage. Partial output is not exposed by polling.
fn queued_response(job: &BackgroundJob) -> ResponseObject {
    let mut response = build_response_object(OutboundContext {
        response_id: job.response_id.clone(),
        model_id: job.state.display_model_id().to_string(),
        created_at: job.created_at,
        completed_at: job.created_at,
        status: ResponseStatus::Queued,
        prompt_tokens: 0,
        completion_tokens: 0,
        cached_tokens: 0,
        reasoning_tokens: 0,
        text: String::new(),
        reasoning_text: None,
        parsed_tool_calls: None,
        max_tool_calls: job.request.max_tool_calls,
        request: &job.request,
        error: None,
        incomplete_reason: None,
        finish_reason: "queued".to_string(),
    });
    response.completed_at = None;
    // The assembly adds a message item even for empty text.
    response.output.clear();
    response
}

/// The job task: the prelude, then the blocking generation.
async fn run_background_job<D>(
    job: BackgroundJob,
    cancelled: InFlightToken,
    recorder: Arc<JobRecorder>,
    driver: D,
) where
    D: FnOnce(PreparedRun, CancellationToken, Arc<dyn ResponseEventSink>) + Send + 'static,
{
    let BackgroundJob {
        state,
        live,
        request,
        mut translated,
        response_id,
        created_at,
        priority,
        budget_override,
        structured,
        compact_threshold,
    } = job;

    let prelude = run_prelude(
        &state,
        &live,
        &request,
        &mut translated,
        compact_threshold,
        &recorder,
    );
    let auto = match unless_cancelled(&recorder, prelude).await {
        Some(Ok(auto)) => auto,
        Some(Err(err)) => {
            recorder.fail_with(&err);
            return;
        }
        None => return,
    };
    // A cancel that arrived while the job was still queued: skip generation.
    if cancelled.load(Ordering::Acquire) {
        recorder.cancel_before_start();
        return;
    }
    let prepare = prepare_response_run(
        state,
        live,
        request,
        translated,
        response_id,
        created_at,
        priority,
        budget_override,
        structured,
        auto,
    );
    let mut run = match unless_cancelled(&recorder, prepare).await {
        Some(Ok(run)) => run,
        Some(Err(err)) => {
            recorder.fail_with(&err);
            return;
        }
        None => return,
    };

    let sink = BackgroundSink::new(&run, Arc::clone(&recorder));
    run.background_opening = Some(sink.next_sequence_number());
    let sink: Arc<dyn ResponseEventSink> = Arc::new(sink);
    let admission = run.arm_admission(cancelled.clone());
    let generation = tokio::task::spawn_blocking(move || driver(run, cancelled, sink));
    // The waiter is held until the provider answers: dropping it unanswered
    // would cancel the generation. A refusal also fails the generation call,
    // so the driver reports it; the result is kept only as a fallback.
    let admission = admission.wait().await;
    if let Err(join_error) = generation.await {
        tracing::error!(
            response_id = %recorder.response_id(),
            "background response generation task failed: {join_error}"
        );
        // A panic after the terminal was recorded must not log a failure.
        if recorder.is_active() {
            recorder.fail_server_error(
                "The background response generation ended unexpectedly; check the server log.",
            );
        }
        return;
    }
    if recorder.is_active() {
        // The driver returned without a terminal outcome.
        match admission {
            Err(err) => recorder.fail_with(&super::generation_error_to_response(err)),
            Ok(()) => recorder.fail_server_error(
                "The background response ended without a result; check the server log.",
            ),
        }
    }
}

/// Run `work`, the job's pre-generation work, unless the job is cancelled
/// first: a cancel of a `queued` job records its terminal object and frees
/// its slot at once, so the work is dropped (stopping a queued video decode,
/// a compaction pass, or a media fetch, as a foreground client disconnect
/// does) instead of running on with no slot bounding it. `None` when the job
/// was cancelled; its cancel registration is retired.
async fn unless_cancelled<F: std::future::Future>(
    recorder: &JobRecorder,
    work: F,
) -> Option<F::Output> {
    tokio::select! {
        output = work => Some(output),
        () = recorder.terminal_recorded() => {
            recorder.cancel_before_start();
            None
        }
    }
}

/// The work the foreground handler does before it answers that can take
/// long: the video-frames fallback (issue #1322) and automatic compaction
/// (issue #2397), in the foreground's order.
async fn run_prelude(
    state: &AppState,
    live: &Arc<LiveSettings>,
    request: &CreateResponseRequest,
    translated: &mut TranslatedRequest,
    compact_threshold: Option<usize>,
    recorder: &JobRecorder,
) -> Result<Option<AutoCompaction>, ErrorResponse> {
    crate::server::chat_request::expand_request_video_parts(state, &mut translated.chat_request)
        .await
        .map_err(|err| err.into_error_response())?;
    let Some(threshold) = compact_threshold else {
        return Ok(None);
    };
    let auto = apply_auto_compaction(state, live, request, translated, threshold).await?;
    if auto.compacted() {
        // The stored response carries the capsule in its output, so it must
        // be stored with the compacted input items, not the admitted ones.
        recorder.set_input_items(translated.canonical_input_items.clone());
        // A rebuilt request repeats the video substitution.
        crate::server::chat_request::expand_request_video_parts(
            state,
            &mut translated.chat_request,
        )
        .await
        .map_err(|err| err.into_error_response())?;
    }
    Ok(Some(auto))
}

#[cfg(test)]
#[path = "responses_background_route_tests.rs"]
mod responses_background_route_tests;
