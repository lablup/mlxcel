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

//! `POST /v1/responses` with the built-in `mcp` tool: request setup and the
//! HTTP side of the server-side MCP tool loop (issue #1457).
//!
//! [`prepare_mcp`] decides whether a request uses the loop and validates
//! everything about it that can be refused with a 400 before any model or
//! tool work: the tool declarations, the named servers, tool-name collisions,
//! and the approval answers. [`create_mcp_response`] then runs the loop
//! ([`super::responses_mcp_loop::drive_mcp_run`]) on a blocking thread and
//! answers either with the SSE stream or with the collected response object.
//! Both shapes come from the same event sequence, so they list the same
//! output items.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};

use crate::server::config::ReasoningBudgetOverride;
use crate::server::mcp::McpRuntime;
use crate::server::mcp::approvals::ApprovalAnswer;
use crate::server::mcp::request::{McpRequestPlan, plan_request};
use crate::server::responses_translator::TranslatedRequest;
use crate::server::streaming::sse_response;
use crate::server::streaming_responses::{CancellationToken, responses_sse_channel_resumable};
use crate::server::types::ErrorResponse;
use crate::server::types::responses_request::{
    CreateResponseRequest, ResponseInput, ResponseInputItem, ResponseTool,
};
use crate::server::types::responses_response::ResponseObject;
use crate::server::types::responses_stream::ResponseStreamEvent;
use crate::server::types::{ChatCompletionRequest, ResponseStatus};
use crate::server::{AppState, LiveSettings};

use super::chat::SharedConstraint;
use super::responses::persist_response;
use super::responses_mcp_loop::{McpRun, drive_mcp_run};
use super::responses_run::{ResponseEventSink, SseResponseSink};

/// The validated MCP side of one request.
pub(super) struct McpSetup {
    pub(super) runtime: Arc<McpRuntime>,
    pub(super) plan: McpRequestPlan,
    /// The request's approval answers, checked against the pending requests
    /// of `previous_response_id` but not yet consumed.
    pub(super) answers: Vec<ApprovalAnswer>,
}

impl McpSetup {
    /// Offer the planned MCP tools to the model as function tools.
    pub(super) fn add_tools(&self, chat_request: &mut ChatCompletionRequest) {
        if self.plan.function_tools.is_empty() {
            return;
        }
        chat_request
            .tools
            .get_or_insert_with(Vec::new)
            .extend(self.plan.function_tools.iter().cloned());
    }
}

fn bad_request(message: impl Into<String>) -> ErrorResponse {
    ErrorResponse::new(message, "invalid_request_error")
}

/// Resolve the MCP side of `request`, or `None` when it has none.
///
/// With the tool loop disabled this always answers `None`; the translator
/// then refuses an `mcp` tool or approval answer with the message that names
/// the opt-in.
pub(super) async fn prepare_mcp(
    state: &AppState,
    request: &CreateResponseRequest,
) -> Result<Option<McpSetup>, ErrorResponse> {
    let Some(runtime) = state.mcp.as_ref() else {
        return Ok(None);
    };
    let declarations: Vec<_> = request
        .tools
        .iter()
        .flatten()
        .filter_map(|tool| match tool {
            ResponseTool::Mcp(definition) => Some(definition),
            _ => None,
        })
        .collect();
    let answers: Vec<ApprovalAnswer> = match &request.input {
        ResponseInput::Text(_) => Vec::new(),
        ResponseInput::Items(items) => items
            .iter()
            .filter_map(|item| match item {
                ResponseInputItem::McpApprovalResponse {
                    approval_request_id,
                    approve,
                    ..
                } => Some(ApprovalAnswer {
                    approval_request_id: approval_request_id.clone(),
                    approve: *approve,
                }),
                _ => None,
            })
            .collect(),
    };
    let previous = request.previous_response_id.as_deref();
    if answers.is_empty() {
        // A response that waits on approvals can only be continued by
        // answering them: the model's turn has calls with no result yet.
        if let Some(previous) = previous
            && let Err(error) = runtime.approvals().check(previous, &[])
        {
            return Err(bad_request(error.to_string()));
        }
        if declarations.is_empty() {
            return Ok(None);
        }
    }
    if request.background == Some(true) {
        return Err(bad_request(
            "background mode is not supported together with mcp tools; send the request \
             without 'background'",
        ));
    }
    let client_functions: HashSet<String> = request
        .tools
        .iter()
        .flatten()
        .filter_map(|tool| match tool {
            ResponseTool::Function(function) => Some(function.name.clone()),
            _ => None,
        })
        .collect();
    let plan = plan_request(runtime, &declarations, &client_functions)
        .await
        .map_err(bad_request)?;
    if !answers.is_empty() {
        let Some(previous) = previous else {
            return Err(bad_request(
                "mcp_approval_response needs previous_response_id naming the stored response \
                 that requested the approval",
            ));
        };
        let pending = runtime
            .approvals()
            .check(previous, &answers)
            .map_err(|error| bad_request(error.to_string()))?;
        for approval in &pending {
            let allowed = plan
                .server(&approval.server_label)
                .is_some_and(|server| server.allows(&approval.tool_name));
            if !allowed {
                return Err(bad_request(format!(
                    "mcp_approval_response '{}' answers a call to tool '{}' of MCP server \
                     '{}', which this request's tools do not declare or allow; send the same \
                     mcp tool declaration as the request that was paused",
                    approval.id, approval.tool_name, approval.server_label
                )));
            }
        }
    }
    Ok(Some(McpSetup {
        runtime: Arc::clone(runtime),
        plan,
        answers,
    }))
}

/// Everything [`create_mcp_response`] needs from the create handler.
pub(super) struct McpResponseArgs {
    pub(super) state: AppState,
    pub(super) live: Arc<LiveSettings>,
    pub(super) request: CreateResponseRequest,
    pub(super) translated: TranslatedRequest,
    pub(super) response_id: String,
    pub(super) created_at: f64,
    pub(super) priority: crate::server::batch::RequestPriority,
    pub(super) budget_override: ReasoningBudgetOverride,
    pub(super) structured: Option<SharedConstraint>,
    pub(super) setup: McpSetup,
}

/// Run the tool loop for one request and answer it, streamed or not.
pub(super) async fn create_mcp_response(args: McpResponseArgs, headers: &HeaderMap) -> Response {
    let McpResponseArgs {
        state,
        live,
        request,
        translated,
        response_id,
        created_at,
        priority,
        budget_override,
        structured,
        setup,
    } = args;
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }
    // A tool-loop response holds a blocking thread while it waits on tools,
    // outside the model queue, so the number running at once has its own
    // cap: the same --max-queue-depth that bounds queued model work.
    let Some(permit) = setup.runtime.admit_run(state.config.max_queue_depth) else {
        return ErrorResponse::service_unavailable(
            "Too many MCP tool-loop responses are running. Please try again later.",
        )
        .into_response();
    };
    let stream = request.stream;
    let request = Arc::new(request);
    let translated = Arc::new(translated);
    let run = McpRun {
        state: state.clone(),
        live,
        request: Arc::clone(&request),
        translated: Arc::clone(&translated),
        response_id: response_id.clone(),
        created_at,
        priority,
        budget_override,
        structured,
        setup,
    };
    let runtime_handle = tokio::runtime::Handle::current();

    if stream {
        // b10621 resumable stream (#1444), as on the plain streaming path.
        let session = super::stream::conversation_id_from_headers(headers).map(|cid| {
            state
                .stream_sessions
                .create_or_replace(&cid, super::stream::request_stream_owner(&state, headers))
        });
        let (sender, stream, cancelled, keepalive) =
            responses_sse_channel_resumable(128, state.config.sse_ping_interval, session);
        let sink: Arc<dyn ResponseEventSink> = Arc::new(SseResponseSink::from_parts(
            &state,
            request,
            translated,
            &response_id,
            sender,
            &cancelled,
        ));
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            drive_mcp_run(run, cancelled, sink, runtime_handle);
        });
        return sse_response(stream, keepalive);
    }

    let cancelled: CancellationToken = Arc::default();
    let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel();
    let sink: Arc<dyn ResponseEventSink> = Arc::new(CollectingSink {
        state,
        request,
        translated,
        outcome: Mutex::new(Some(outcome_tx)),
    });
    // A caller that hangs up drops this future: stop the loop, so no further
    // model turn or tool call runs for a response nobody reads.
    let mut hangup = CancelOnDrop(Some(cancelled.clone()));
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        drive_mcp_run(run, cancelled, sink, runtime_handle);
    });
    let outcome = outcome_rx.await;
    hangup.0 = None;
    match outcome {
        Ok(Ok(response)) => Json(*response).into_response(),
        Ok(Err(error)) => error.into_response(),
        Err(_) => ErrorResponse::internal_server_error(
            "the MCP tool loop ended without producing a response",
        )
        .into_response(),
    }
}

struct CancelOnDrop(Option<CancellationToken>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(cancelled) = self.0.take() {
            cancelled.store(true, Ordering::Relaxed);
        }
    }
}

type Outcome = Result<Box<ResponseObject>, ErrorResponse>;

/// The sink of a non-streamed tool-loop response: events are dropped, the
/// terminal response is persisted and handed to the waiting handler.
struct CollectingSink {
    state: AppState,
    request: Arc<CreateResponseRequest>,
    translated: Arc<TranslatedRequest>,
    outcome: Mutex<Option<tokio::sync::oneshot::Sender<Outcome>>>,
}

impl CollectingSink {
    fn finish(&self, outcome: Outcome) {
        if let Some(sender) = self.outcome.lock().ok().and_then(|mut slot| slot.take()) {
            let _ = sender.send(outcome);
        }
    }
}

impl ResponseEventSink for CollectingSink {
    fn emit(&self, event: &ResponseStreamEvent) {
        match event {
            ResponseStreamEvent::Completed { response, .. }
            | ResponseStreamEvent::Incomplete { response, .. }
            | ResponseStreamEvent::Failed { response, .. } => {
                self.finish(Ok(Box::new(response.clone())));
            }
            ResponseStreamEvent::Error { code, message, .. } => {
                let error = if code == "server_overloaded" {
                    ErrorResponse::service_unavailable(message.clone())
                } else if code == "invalid_request_error" {
                    ErrorResponse::new(message.clone(), "invalid_request_error")
                } else {
                    ErrorResponse::internal_server_error(message.clone())
                };
                self.finish(Err(error));
            }
            _ => {}
        }
    }

    fn on_started(&self) {}

    fn record_terminal(&self, response: &ResponseObject) {
        // A cancelled turn with no output adds nothing to a conversation.
        if response.status == ResponseStatus::Cancelled && response.output.is_empty() {
            return;
        }
        persist_response(&self.state, &self.request, &self.translated, response);
    }

    fn finish_registration(&self) -> bool {
        false
    }

    fn generation_finished(&self) {}
}

#[cfg(all(test, unix))]
#[path = "responses_mcp_tests.rs"]
mod tests;
