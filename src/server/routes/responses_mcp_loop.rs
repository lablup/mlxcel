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

//! The server-side MCP tool loop of one `/v1/responses` request (issue
//! #1457): model turn, tool calls, tool results, next model turn, until the
//! model answers without calling a tool or a bound stops it.
//!
//! A model turn's tool calls are handled in the order the model made them:
//!
//! - A call to an MCP tool that needs no approval runs at once and becomes an
//!   `mcp_call` item; its result is the tool message of the next turn.
//! - A call that needs approval becomes an `mcp_approval_request` item and
//!   ends the response. The arguments are kept server-side
//!   ([`crate::server::mcp::approvals`]); the follow-up request's answer runs
//!   exactly those. When the response is not stored there is nowhere to
//!   collect the answer, so the call is refused instead of run.
//! - A call to a client-declared function tool becomes a `function_call`
//!   item and ends the response, as it does without the loop.
//!
//! Tool output is untrusted text: it is capped, handed to the model as a tool
//! message, and never parsed or acted on here. Arguments and outputs are not
//! logged.
//!
//! Three bounds stop a response: `--mcp-max-tool-rounds` (and the request's
//! `max_tool_calls`, or, when the request sets none, the fixed per-turn cap
//! [`DEFAULT_MAX_CALLS_PER_TURN`]) end it as `incomplete`, `--mcp-tool-timeout` and
//! `--mcp-max-output-bytes` fail or truncate a single call. `usage.mcp`
//! reports the counts and the bounds in force.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::server::config::ReasoningBudgetOverride;
use crate::server::mcp::approvals::PendingApproval;
use crate::server::mcp::config::DEFAULT_MAX_CALLS_PER_TURN;
use crate::server::model_provider::QueueFullError;
use crate::server::responses_translator::{
    OutboundContext, TranslatedRequest, build_response_object, short_uuid,
};
use crate::server::streaming_responses::CancellationToken;
use crate::server::tool_calls;
use crate::server::types::request::{
    Message, MessageContent, Role, ToolCallFunction, ToolCallInMessage,
};
use crate::server::types::responses_mcp::{McpApprovalRequestItem, McpCallStatus, McpUsage};
use crate::server::types::responses_request::CreateResponseRequest;
use crate::server::types::responses_response::{
    ResponseIncompleteDetails, ResponseObject, ResponseOutputItem, ResponseStatus,
};
use crate::server::types::responses_stream::ResponseStreamEvent;
use crate::server::{AppState, LiveSettings};

use super::chat::SharedConstraint;
use super::responses::count_reasoning_tokens;
use super::responses_mcp::McpSetup;
use super::responses_mcp_round::{LoopEvents, run_round};
use super::responses_run::{ResponseEventSink, prepare_generation, streamed_output_text};

/// `incomplete_details.reason` when the tool-round bound ended the response.
pub(crate) const INCOMPLETE_MAX_TOOL_ROUNDS: &str = "max_tool_rounds";
/// `incomplete_details.reason` when the request's `max_tool_calls` did.
pub(crate) const INCOMPLETE_MAX_TOOL_CALLS: &str = "max_tool_calls";
/// `incomplete_details.reason` when a model turn hit its output limit.
const INCOMPLETE_MAX_OUTPUT_TOKENS: &str = "max_output_tokens";

/// One tool-loop request, ready to run.
pub(super) struct McpRun {
    pub(super) state: AppState,
    pub(super) live: Arc<LiveSettings>,
    pub(super) request: Arc<CreateResponseRequest>,
    pub(super) translated: Arc<TranslatedRequest>,
    pub(super) response_id: String,
    pub(super) created_at: f64,
    pub(super) priority: crate::server::batch::RequestPriority,
    pub(super) budget_override: ReasoningBudgetOverride,
    /// The constraint the create handler built for the first model turn.
    pub(super) structured: Option<SharedConstraint>,
    pub(super) setup: McpSetup,
}

/// Token counts summed over every model turn of the response.
#[derive(Default)]
struct Totals {
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
}

/// How a tool call that is not run is reported.
pub(super) struct Refusal {
    pub(super) error: String,
    pub(super) status: McpCallStatus,
}

/// Why the loop stopped.
enum Stop {
    /// The model answered, or handed calls back to the caller.
    Finished,
    /// A bound ended the response; the reason goes to `incomplete_details`.
    Bound(&'static str),
    Cancelled,
    /// A model turn failed; the response has no terminal object.
    Failed {
        code: &'static str,
        message: String,
    },
}

pub(super) struct ToolLoop<'a> {
    pub(super) run: &'a McpRun,
    pub(super) handle: &'a tokio::runtime::Handle,
    pub(super) cancelled: &'a CancellationToken,
    pub(super) events: LoopEvents<'a>,
    pub(super) usage: McpUsage,
    totals: Totals,
    /// Approval requests this response ends with.
    pending: Vec<PendingApproval>,
}

/// Run the tool loop to its terminal event on the calling thread.
///
/// Blocking: call it from `spawn_blocking`. `handle` runs the async MCP and
/// preparation steps; `cancelled` stops the loop between and during steps.
///
/// Used by: [`super::responses_mcp::create_mcp_response`].
pub(super) fn drive_mcp_run(
    run: McpRun,
    cancelled: CancellationToken,
    sink: Arc<dyn ResponseEventSink>,
    handle: tokio::runtime::Handle,
) {
    let limits = run.setup.runtime.limits();
    let mut tool_loop = ToolLoop {
        run: &run,
        handle: &handle,
        cancelled: &cancelled,
        events: LoopEvents::new(sink.as_ref()),
        usage: McpUsage {
            max_tool_rounds: limits.max_tool_rounds,
            tool_timeout_ms: u64::try_from(limits.tool_timeout.as_millis()).unwrap_or(u64::MAX),
            max_output_bytes: limits.max_output_bytes,
            ..McpUsage::default()
        },
        totals: Totals::default(),
        pending: Vec::new(),
    };
    tool_loop.open();
    let stop = tool_loop.rounds();
    tool_loop.finish(stop, sink.as_ref());
}

impl ToolLoop<'_> {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// The response object as of now, with `status` and the items delivered
    /// so far.
    fn response(&self, status: ResponseStatus, completed_at: f64) -> ResponseObject {
        let run = self.run;
        let reasoning_text = self.events.reasoning_text();
        let mut response = build_response_object(OutboundContext {
            response_id: run.response_id.clone(),
            model_id: run.state.display_model_id().to_string(),
            created_at: run.created_at,
            completed_at,
            status,
            prompt_tokens: self.totals.prompt_tokens,
            completion_tokens: self.totals.completion_tokens,
            cached_tokens: self.totals.cached_tokens,
            reasoning_tokens: count_reasoning_tokens(
                &run.state.tokenizer,
                reasoning_text.as_deref(),
            ),
            text: String::new(),
            reasoning_text: None,
            parsed_tool_calls: None,
            max_tool_calls: run.request.max_tool_calls,
            request: &run.request,
            error: None,
            incomplete_reason: None,
            finish_reason: "stop".to_string(),
        });
        // The assembly adds a message item even for empty text; the items of
        // this response are the ones the loop delivered.
        response.output = self.events.output.clone();
        response.output_text = streamed_output_text(&response.output);
        response.usage.mcp = Some(self.usage.clone());
        response
    }

    /// `response.created`, `response.in_progress`, then one `mcp_list_tools`
    /// item per server this chain has not listed yet.
    fn open(&mut self) {
        let mut initial = self.response(ResponseStatus::InProgress, self.run.created_at);
        initial.completed_at = None;
        let created = initial.clone();
        self.events
            .event(|sequence_number, _| ResponseStreamEvent::Created {
                sequence_number,
                response: created,
            });
        self.events
            .event(|sequence_number, _| ResponseStreamEvent::InProgress {
                sequence_number,
                response: initial,
            });
        let run = self.run;
        for server in &run.setup.plan.servers {
            if run
                .translated
                .mcp_listed_servers
                .contains(&server.request.server_label)
            {
                continue;
            }
            self.list_tools(server);
        }
    }

    /// Consume the request's approval answers and run the approved calls
    /// with the arguments stored when they were requested.
    ///
    /// The request's `max_tool_calls` caps these calls too: an approved call
    /// past it is reported as not run, and the response ends `incomplete`.
    fn resume_approvals(&mut self, messages: &mut [Message]) -> Result<(), Stop> {
        let run = self.run;
        if run.setup.answers.is_empty() {
            return Ok(());
        }
        let mut bound = None;
        let previous = run.request.previous_response_id.as_deref().unwrap_or("");
        let approvals = run
            .setup
            .runtime
            .approvals()
            .consume(previous, &run.setup.answers)
            .map_err(|error| Stop::Failed {
                code: "invalid_request_error",
                message: error.to_string(),
            })?;
        for (answer, approval) in run.setup.answers.iter().zip(approvals) {
            // A denial already reads as the call's result in the transcript.
            if !answer.approve {
                continue;
            }
            let calls_spent = run
                .request
                .max_tool_calls
                .is_some_and(|max| self.usage.tool_calls >= max);
            let refusal = calls_spent.then(|| {
                bound = Some(INCOMPLETE_MAX_TOOL_CALLS);
                Refusal {
                    error: "not run: the response reached its max_tool_calls limit".to_string(),
                    status: McpCallStatus::Incomplete,
                }
            });
            let (_, text) = self.mcp_call(
                &approval.server_label,
                &approval.tool_name,
                &approval.arguments,
                Some(approval.id.clone()),
                refusal,
            );
            if let Some(message) = messages.iter_mut().rev().find(|message| {
                message.role == Role::Tool
                    && message.tool_call_id.as_deref() == Some(approval.id.as_str())
            }) {
                message.content = MessageContent::Text(text);
            }
        }
        match bound {
            Some(reason) => Err(Stop::Bound(reason)),
            None => Ok(()),
        }
    }

    /// The model turns and the tool work between them.
    fn rounds(&mut self) -> Stop {
        let run = self.run;
        let limits = run.setup.runtime.limits();
        let mut chat = run.translated.chat_request.clone();
        if let Err(stop) = self.resume_approvals(&mut chat.messages) {
            return stop;
        }
        // An approval request needs the stored response to be answered.
        let can_collect_approval = run.translated.effective_store
            && run.state.responses_store.is_some()
            && run.translated.conversation_id.is_none();
        let mut structured = run.structured.clone();
        let mut first_turn = true;
        loop {
            if self.is_cancelled() {
                return Stop::Cancelled;
            }
            if !first_turn {
                // `tool_choice` steers the first turn only: a forced call on
                // every turn could never reach an answer.
                chat.tool_choice = None;
                structured = match self
                    .handle
                    .block_on(super::chat::build_chat_constraint(&run.state, &chat))
                {
                    Ok(structured) => structured,
                    Err(error) => {
                        return Stop::Failed {
                            code: "server_error",
                            message: error.error.message,
                        };
                    }
                };
            }
            first_turn = false;
            let generation = match self.handle.block_on(prepare_generation(
                &run.state,
                &run.live,
                &chat,
                run.priority,
                run.budget_override,
                structured.take(),
                None,
            )) {
                Ok(generation) => generation,
                Err(error) => {
                    let code = if error.status == axum::http::StatusCode::SERVICE_UNAVAILABLE {
                        "server_overloaded"
                    } else if error.status == axum::http::StatusCode::BAD_REQUEST {
                        "invalid_request_error"
                    } else {
                        "server_error"
                    };
                    return Stop::Failed {
                        code,
                        message: error.error.message,
                    };
                }
            };
            let round = match run_round(
                &run.state,
                &run.live,
                generation,
                self.cancelled.clone(),
                &mut self.events,
            ) {
                Ok(round) => round,
                Err(_) if self.is_cancelled() => return Stop::Cancelled,
                Err(error) => {
                    return if error.downcast_ref::<QueueFullError>().is_some() {
                        Stop::Failed {
                            code: "server_overloaded",
                            message: "All slots are busy. Please try again later.".to_string(),
                        }
                    } else {
                        Stop::Failed {
                            code: "server_error",
                            message: error.to_string(),
                        }
                    };
                }
            };
            self.totals.prompt_tokens += round.result.prompt_tokens;
            self.totals.completion_tokens += round.result.completion_tokens;
            self.totals.cached_tokens += round.result.cached_tokens;
            if self.is_cancelled() {
                return Stop::Cancelled;
            }
            // A turn cut off at its output limit is not acted on: a call it
            // was still writing would run with half its arguments.
            if round.result.finish_reason == "length" {
                return Stop::Bound(INCOMPLETE_MAX_OUTPUT_TOKENS);
            }
            let calls = if tool_calls::should_parse_tool_calls(&chat) {
                tool_calls::parse_tool_calls_with_context(
                    &round.raw,
                    chat.tools.as_deref(),
                    round.primed_open_thinking,
                )
                .tool_calls
            } else {
                Vec::new()
            };
            if calls.is_empty() {
                return Stop::Finished;
            }

            let rounds_spent = self.usage.tool_rounds >= limits.max_tool_rounds;
            let mut bound = None;
            let mut hands_back = false;
            let mut turn_calls = Vec::with_capacity(calls.len());
            let mut results = Vec::new();
            let mut turn_mcp_calls = 0usize;
            for call in calls {
                let Some(tool) = run.setup.plan.tools.get(&call.name) else {
                    // A client-declared function tool: the caller runs it.
                    let call_id = call
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("call_{}", short_uuid()));
                    self.function_call(&call_id, &call.name, &call.arguments);
                    turn_calls.push(assistant_call(&call_id, &call.name, &call.arguments));
                    hands_back = true;
                    continue;
                };
                let calls_spent = run
                    .request
                    .max_tool_calls
                    .is_some_and(|max| self.usage.tool_calls >= max);
                // Without a request `max_tool_calls`, one model turn may not
                // fan out into an unbounded number of calls.
                turn_mcp_calls += 1;
                let turn_spent = run.request.max_tool_calls.is_none()
                    && turn_mcp_calls > DEFAULT_MAX_CALLS_PER_TURN;
                let refusal = if rounds_spent {
                    bound = Some(INCOMPLETE_MAX_TOOL_ROUNDS);
                    Some(Refusal {
                        error: format!(
                            "not run: the response reached its limit of {} tool rounds",
                            limits.max_tool_rounds
                        ),
                        status: McpCallStatus::Incomplete,
                    })
                } else if calls_spent {
                    bound.get_or_insert(INCOMPLETE_MAX_TOOL_CALLS);
                    Some(Refusal {
                        error: "not run: the response reached its max_tool_calls limit".to_string(),
                        status: McpCallStatus::Incomplete,
                    })
                } else if turn_spent {
                    bound.get_or_insert(INCOMPLETE_MAX_TOOL_CALLS);
                    Some(Refusal {
                        error: format!(
                            "not run: one model turn asked for more than {DEFAULT_MAX_CALLS_PER_TURN} \
                             MCP calls and the request sets no max_tool_calls"
                        ),
                        status: McpCallStatus::Incomplete,
                    })
                } else if tool.requires_approval && can_collect_approval {
                    let item = McpApprovalRequestItem {
                        id: format!("mcpr_{}", short_uuid()),
                        server_label: tool.server_label.clone(),
                        name: call.name.clone(),
                        arguments: if call.arguments.trim().is_empty() {
                            "{}".to_string()
                        } else {
                            call.arguments.clone()
                        },
                    };
                    self.pending.push(PendingApproval {
                        id: item.id.clone(),
                        response_id: run.response_id.clone(),
                        server_label: item.server_label.clone(),
                        tool_name: item.name.clone(),
                        arguments: item.arguments.clone(),
                    });
                    turn_calls.push(assistant_call(&item.id, &item.name, &item.arguments));
                    self.events
                        .item_added(ResponseOutputItem::McpApprovalRequest(item.clone()));
                    self.events
                        .item_done(ResponseOutputItem::McpApprovalRequest(item));
                    hands_back = true;
                    continue;
                } else if tool.requires_approval {
                    Some(Refusal {
                        error: "not run: this tool requires approval, and an approval can only \
                                be collected on a stored response chained with \
                                previous_response_id (store is off, or the request uses a \
                                conversation). Send the request with store enabled, or set \
                                require_approval to \"never\" for this tool"
                            .to_string(),
                        status: McpCallStatus::Failed,
                    })
                } else {
                    None
                };
                let (item_id, text) = self.mcp_call(
                    &tool.server_label,
                    &call.name,
                    &call.arguments,
                    None,
                    refusal,
                );
                turn_calls.push(assistant_call(&item_id, &call.name, &call.arguments));
                results.push(tool_result(&item_id, text));
                if self.is_cancelled() {
                    return Stop::Cancelled;
                }
            }
            if let Some(reason) = bound {
                return Stop::Bound(reason);
            }
            if hands_back {
                return Stop::Finished;
            }
            // A round counts whether its calls ran or were refused: a model
            // that keeps calling a tool it may not run (approval that cannot
            // be collected) must still reach the round bound.
            if !results.is_empty() {
                self.usage.tool_rounds += 1;
            }
            chat.messages.push(Message {
                role: Role::Assistant,
                content: MessageContent::Text(round.message_text),
                name: None,
                tool_call_id: None,
                reasoning: round.reasoning_text,
                tool_calls: Some(turn_calls),
            });
            chat.messages.extend(results);
        }
    }

    /// Decide the final status, store the response and its approval
    /// requests, and send the terminal event.
    fn finish(mut self, stop: Stop, sink: &dyn ResponseEventSink) {
        sink.generation_finished();
        let cancel_acknowledged = sink.finish_registration();
        let (status, incomplete_reason) = match stop {
            Stop::Failed { code, message } if !cancel_acknowledged => {
                self.events
                    .event(|sequence_number, _| ResponseStreamEvent::Error {
                        sequence_number,
                        code: code.to_string(),
                        message,
                    });
                return;
            }
            Stop::Cancelled | Stop::Failed { .. } => (ResponseStatus::Cancelled, None),
            _ if cancel_acknowledged || self.is_cancelled() => (ResponseStatus::Cancelled, None),
            Stop::Bound(reason) => (ResponseStatus::Incomplete, Some(reason)),
            Stop::Finished => (ResponseStatus::Completed, None),
        };
        let mut response = self.response(status, crate::server::clock::unix_seconds_now() as f64);
        if status != ResponseStatus::Completed {
            response.completed_at = None;
        }
        response.incomplete_details = incomplete_reason.map(|reason| ResponseIncompleteDetails {
            reason: reason.to_string(),
        });
        // Registered before the response can be read, so its approval
        // requests are answerable as soon as the caller sees them.
        let pending = std::mem::take(&mut self.pending);
        if !pending.is_empty() {
            self.run.setup.runtime.approvals().register(pending);
        }
        sink.record_terminal(&response);
        self.events.event(|sequence_number, _| match status {
            ResponseStatus::Incomplete => ResponseStreamEvent::Incomplete {
                sequence_number,
                response,
            },
            _ => ResponseStreamEvent::Completed {
                sequence_number,
                response,
            },
        });
    }
}

fn assistant_call(call_id: &str, name: &str, arguments: &str) -> ToolCallInMessage {
    ToolCallInMessage {
        id: call_id.to_string(),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    }
}

fn tool_result(call_id: &str, text: String) -> Message {
    Message {
        role: Role::Tool,
        content: MessageContent::Text(text),
        name: None,
        tool_call_id: Some(call_id.to_string()),
        reasoning: None,
        tool_calls: None,
    }
}
