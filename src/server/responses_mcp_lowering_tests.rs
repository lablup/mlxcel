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

use super::*;
use crate::server::types::responses_request::{
    ResponseInputContent, ResponseInputItem, ResponseInputRole,
};

fn user(text: &str) -> ResponseInputItem {
    ResponseInputItem::Message {
        role: ResponseInputRole::User,
        content: ResponseInputContent::Text(text.to_string()),
        name: None,
    }
}

fn call(
    id: &str,
    output: Option<&str>,
    error: Option<&str>,
    approval: Option<&str>,
) -> ResponseInputItem {
    ResponseInputItem::McpCall {
        id: id.to_string(),
        server_label: "fake".to_string(),
        name: "add".to_string(),
        arguments: r#"{"a":1,"b":2}"#.to_string(),
        output: output.map(str::to_string),
        error: error.map(str::to_string),
        approval_request_id: approval.map(str::to_string),
    }
}

fn approval_request(id: &str) -> ResponseInputItem {
    ResponseInputItem::McpApprovalRequest {
        id: id.to_string(),
        server_label: "fake".to_string(),
        name: "add".to_string(),
        arguments: r#"{"a":1,"b":2}"#.to_string(),
    }
}

fn approval_response(id: &str, approve: bool) -> ResponseInputItem {
    ResponseInputItem::McpApprovalResponse {
        approval_request_id: id.to_string(),
        approve,
        reason: None,
    }
}

/// `C:<id>` for a call, `O:<id>=<text>` for an output, `M` for a message.
fn shape(items: &[ResponseInputItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| match item {
            ResponseInputItem::FunctionCall { call_id, .. } => format!("C:{call_id}"),
            ResponseInputItem::FunctionCallOutput { call_id, output } => match output {
                ResponseToolOutput::Text(text) => format!("O:{call_id}={text}"),
                ResponseToolOutput::Parts(_) => format!("O:{call_id}=<parts>"),
            },
            ResponseInputItem::Message { .. } => "M".to_string(),
            other => panic!("not lowered: {other:?}"),
        })
        .collect()
}

#[test]
fn a_transcript_without_mcp_items_is_untouched() {
    let items = vec![
        user("hi"),
        ResponseInputItem::FunctionCall {
            call_id: "c1".to_string(),
            name: "f".to_string(),
            arguments: "{}".to_string(),
        },
    ];
    assert!(!has_mcp_items(&items));
    assert_eq!(shape(&lower_mcp_items(items)), ["M", "C:c1"]);
}

#[test]
fn consecutive_calls_render_as_one_turn_then_its_results() {
    let items = vec![
        user("go"),
        ResponseInputItem::McpListTools {
            server_label: "fake".to_string(),
        },
        call("mcp_1", Some("3"), None, None),
        call("mcp_2", None, Some("boom"), None),
        user("next"),
    ];
    assert_eq!(listed_mcp_servers(&items), ["fake"]);
    assert_eq!(
        shape(&lower_mcp_items(items)),
        [
            "M",
            "C:mcp_1",
            "C:mcp_2",
            "O:mcp_1=3",
            "O:mcp_2=Tool call failed: boom",
            "M"
        ]
    );
}

#[test]
fn an_approved_request_is_filled_by_the_call_that_resolved_it() {
    let items = vec![
        user("go"),
        approval_request("mcpr_1"),
        approval_response("mcpr_1", true),
        call("mcp_9", Some("3"), None, Some("mcpr_1")),
    ];
    assert_eq!(
        shape(&lower_mcp_items(items)),
        ["M", "C:mcpr_1", "O:mcpr_1=3"]
    );
}

#[test]
fn an_approved_request_not_yet_run_keeps_an_empty_output_for_the_loop_to_fill() {
    let items = vec![
        user("go"),
        call("mcp_1", Some("7"), None, None),
        approval_request("mcpr_1"),
        approval_response("mcpr_1", true),
    ];
    assert_eq!(
        shape(&lower_mcp_items(items)),
        ["M", "C:mcp_1", "C:mcpr_1", "O:mcp_1=7", "O:mcpr_1="]
    );
}

#[test]
fn a_denied_request_reads_as_a_refused_call() {
    let items = vec![
        approval_request("mcpr_1"),
        ResponseInputItem::McpApprovalResponse {
            approval_request_id: "mcpr_1".to_string(),
            approve: false,
            reason: Some("too risky".to_string()),
        },
    ];
    let lowered = shape(&lower_mcp_items(items));
    assert_eq!(lowered[0], "C:mcpr_1");
    assert!(
        lowered[1].starts_with("O:mcpr_1=The user denied this tool call"),
        "{lowered:?}"
    );
    assert!(lowered[1].ends_with("Reason: too risky"), "{lowered:?}");
}

#[test]
fn an_answer_to_an_unknown_request_closes_nothing() {
    let items = vec![user("go"), approval_response("mcpr_missing", true)];
    assert_eq!(shape(&lower_mcp_items(items)), ["M"]);
}

#[test]
fn a_resolving_call_without_a_recorded_answer_still_closes_its_request() {
    let items = vec![
        approval_request("mcpr_1"),
        call("mcp_9", Some("3"), None, Some("mcpr_1")),
    ];
    assert_eq!(shape(&lower_mcp_items(items)), ["C:mcpr_1", "O:mcpr_1=3"]);
}

#[test]
fn mcp_and_function_calls_of_one_turn_share_the_run() {
    let items = vec![
        call("mcp_1", Some("3"), None, None),
        ResponseInputItem::FunctionCall {
            call_id: "c1".to_string(),
            name: "f".to_string(),
            arguments: "{}".to_string(),
        },
        ResponseInputItem::FunctionCallOutput {
            call_id: "c1".to_string(),
            output: ResponseToolOutput::Text("done".to_string()),
        },
    ];
    assert_eq!(
        shape(&lower_mcp_items(items)),
        ["C:mcp_1", "C:c1", "O:mcp_1=3", "O:c1=done"]
    );
}
