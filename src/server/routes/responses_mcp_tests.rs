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

//! The server-side MCP tool loop through the real router (issue #1457): a
//! scripted model on one side, the scripted stdio MCP server
//! (`tests/fixtures/mcp/fake_mcp_server.py`) on the other.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::server::mcp::McpRuntime;
use crate::server::mcp::test_support::{fake_config, fake_spec};
use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::responses_store::{ResponsesStore, ResponsesStoreConfig};
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

struct Harness {
    app: Router,
    handle: ScriptedStreamHandle,
    prompts: mpsc::Receiver<String>,
    _options: mpsc::Receiver<crate::server::ServerGenerateOptions>,
    runtime: Option<Arc<McpRuntime>>,
}

async fn harness() -> Harness {
    harness_with(Some(fake_config())).await
}

async fn harness_with(mcp: Option<crate::server::mcp::McpStartupConfig>) -> Harness {
    let (options_tx, options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let runtime = match mcp {
        Some(config) => Some(McpRuntime::start(config).await),
        None => None,
    };
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("mcp-test-model"),
        batch_metrics,
    )
    .with_responses_store(Some(Arc::new(ResponsesStore::new(
        ResponsesStoreConfig::default(),
    ))))
    .with_mcp(runtime.clone());
    Harness {
        app: create_app(state),
        handle,
        prompts: prompt_rx,
        _options: options_rx,
        runtime,
    }
}

fn tool_call(name: &str, arguments: Value) -> String {
    format!(
        "<tool_call>\n{}\n</tool_call>",
        json!({"name": name, "arguments": arguments})
    )
}

fn mcp_tool(extra: Value) -> Value {
    let mut tool = json!({"type": "mcp", "server_label": "fake", "require_approval": "never"});
    tool.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    tool
}

fn body(tools: Value) -> Value {
    json!({"model": "mcp-test-model", "input": "What is alpha plus beta?", "tools": tools})
}

impl Harness {
    /// Script one model turn: its tokens, then the end of the turn.
    fn turn(&self, tokens: &[&str]) {
        for token in tokens {
            self.handle.token(token);
        }
        self.handle.finish();
    }

    async fn post(&self, body: Value) -> (StatusCode, String) {
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/responses")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request builds");
        let response =
            tokio::time::timeout(Duration::from_secs(60), self.app.clone().oneshot(request))
                .await
                .expect("the response arrives")
                .expect("router responds");
        let status = response.status();
        let bytes = tokio::time::timeout(
            Duration::from_secs(60),
            axum::body::to_bytes(response.into_body(), 16 * 1024 * 1024),
        )
        .await
        .expect("the body ends")
        .expect("body collects");
        (
            status,
            String::from_utf8(bytes.to_vec()).expect("utf-8 body"),
        )
    }

    async fn create(&self, body: Value) -> (StatusCode, Value) {
        let (status, text) = self.post(body).await;
        (
            status,
            serde_json::from_str(&text).expect("response is JSON"),
        )
    }

    /// POST a streamed request and return its events as `(name, data)`.
    async fn stream(&self, mut body: Value) -> Vec<(String, Value)> {
        body["stream"] = json!(true);
        let (status, text) = self.post(body).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        text.split("\n\n")
            .filter_map(|frame| {
                let name = frame
                    .lines()
                    .find_map(|line| line.strip_prefix("event: "))?;
                let data = frame.lines().find_map(|line| line.strip_prefix("data: "))?;
                Some((
                    name.to_string(),
                    serde_json::from_str(data).expect("event data is JSON"),
                ))
            })
            .collect()
    }

    fn prompts(&self) -> Vec<String> {
        self.prompts.try_iter().collect()
    }
}

fn item_types(response: &Value) -> Vec<&str> {
    response["output"]
        .as_array()
        .expect("output array")
        .iter()
        .map(|item| item["type"].as_str().expect("item type"))
        .collect()
}

fn error_message(response: &Value) -> &str {
    response["error"]["message"].as_str().unwrap_or_default()
}

/// The two-round task: look up two numbers, then add them.
fn script_two_round_task(h: &Harness) {
    h.turn(&[
        "Let me look.",
        &tool_call("lookup", json!({"key": "alpha"})),
        &tool_call("lookup", json!({"key": "beta"})),
    ]);
    h.turn(&[&tool_call("add", json!({"a": 17, "b": 25}))]);
    h.turn(&["The answer ", "is 42."]);
}

#[tokio::test]
async fn mcp_tool_is_refused_with_the_opt_in_named_when_the_loop_is_off() {
    let h = harness_with(None).await;
    let (status, response) = h.create(body(json!([mcp_tool(json!({}))]))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = error_message(&response);
    assert!(message.contains("tool type 'mcp'"), "{message}");
    assert!(message.contains("--mcp-tools"), "{message}");

    let (status, response) = h
        .create(json!({
            "model": "mcp-test-model",
            "previous_response_id": "resp_x",
            "input": [{"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": true}]
        }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        error_message(&response).contains("--mcp-tools"),
        "{response}"
    );
}

#[tokio::test]
async fn non_streamed_two_round_task_lists_tools_calls_them_and_answers() {
    let h = harness().await;
    script_two_round_task(&h);
    let (status, response) = h.create(body(json!([mcp_tool(json!({}))]))).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed");
    assert_eq!(
        item_types(&response),
        [
            "mcp_list_tools",
            "message",
            "mcp_call",
            "mcp_call",
            "mcp_call",
            "message"
        ]
    );
    let output = response["output"].as_array().unwrap();
    assert_eq!(output[0]["server_label"], "fake");
    assert!(
        output[0]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "add"
                && tool["input_schema"]["required"] == json!(["a", "b"]))
    );
    assert_eq!(output[2]["name"], "lookup");
    assert_eq!(output[2]["output"], "17");
    assert_eq!(output[2]["error"], Value::Null);
    assert_eq!(output[2]["status"], "completed");
    assert_eq!(output[3]["output"], "25");
    assert_eq!(output[4]["name"], "add");
    assert_eq!(output[4]["arguments"], r#"{"a":17,"b":25}"#);
    assert_eq!(output[4]["output"], "42");
    assert_eq!(response["output_text"], "Let me look.The answer is 42.");

    // Usage adds up over the three model turns, and reports the loop.
    assert_eq!(response["usage"]["input_tokens"], 3);
    assert_eq!(response["usage"]["output_tokens"], 6);
    assert_eq!(response["usage"]["mcp"]["tool_rounds"], 2);
    assert_eq!(response["usage"]["mcp"]["tool_calls"], 3);
    assert_eq!(response["usage"]["mcp"]["failed_tool_calls"], 0);
    assert_eq!(response["usage"]["mcp"]["max_tool_rounds"], 4);
    assert_eq!(response["usage"]["mcp"]["tool_timeout_ms"], 3000);
    assert_eq!(response["usage"]["mcp"]["max_output_bytes"], 4096);
    assert_eq!(response["tools"][0]["type"], "mcp");

    // Each later turn saw the earlier calls' results as tool messages.
    let prompts = h.prompts();
    assert_eq!(prompts.len(), 3);
    assert!(!prompts[0].contains("<|tool|>"), "{}", prompts[0]);
    assert!(
        prompts[1].ends_with("<|tool|>17<|tool|>25<|assistant|>"),
        "{}",
        prompts[1]
    );
    assert!(
        prompts[2].ends_with("<|tool|>42<|assistant|>"),
        "{}",
        prompts[2]
    );
}

#[tokio::test]
async fn streamed_two_round_task_emits_the_mcp_events_in_order() {
    let h = harness().await;
    script_two_round_task(&h);
    let events = h.stream(body(json!([mcp_tool(json!({}))]))).await;
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    for expected in [
        "response.created",
        "response.in_progress",
        "response.mcp_list_tools.in_progress",
        "response.mcp_list_tools.completed",
        "response.mcp_call.in_progress",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.mcp_call.completed",
        "response.output_item.added",
        "response.output_item.done",
        "response.output_text.delta",
        "response.completed",
    ] {
        assert!(names.contains(&expected), "missing {expected} in {names:?}");
    }
    assert_eq!(names.last(), Some(&"response.completed"));
    // Sequence numbers are gapless and every event names its own type.
    for (index, (name, data)) in events.iter().enumerate() {
        assert_eq!(data["sequence_number"], index as u64, "{name}");
        assert_eq!(data["type"], name.as_str());
    }
    // One mcp_call item: added, in_progress, arguments, completed, done, all
    // under the same item id and output index.
    let added = events
        .iter()
        .find(|(name, data)| {
            name == "response.output_item.added" && data["item"]["type"] == "mcp_call"
        })
        .map(|(_, data)| data)
        .expect("an mcp_call item is added");
    let item_id = added["item"]["id"].as_str().unwrap();
    let index = &added["output_index"];
    assert_eq!(added["item"]["status"], "in_progress");
    let of_item: Vec<&str> = events
        .iter()
        .filter(|(_, data)| data["item_id"] == item_id || data["item"]["id"] == item_id)
        .map(|(name, data)| {
            assert_eq!(&data["output_index"], index, "{name}");
            name.as_str()
        })
        .collect();
    assert_eq!(
        of_item,
        [
            "response.output_item.added",
            "response.mcp_call.in_progress",
            "response.mcp_call_arguments.delta",
            "response.mcp_call_arguments.done",
            "response.mcp_call.completed",
            "response.output_item.done",
        ]
    );
    // The final object lists exactly the items the stream completed.
    let completed = &events.last().unwrap().1["response"];
    let done: Vec<&Value> = events
        .iter()
        .filter(|(name, _)| name == "response.output_item.done")
        .map(|(_, data)| &data["item"])
        .collect();
    assert_eq!(completed["output"].as_array().unwrap().len(), done.len());
    for (listed, streamed) in completed["output"].as_array().unwrap().iter().zip(done) {
        assert_eq!(listed, streamed);
    }
    assert_eq!(
        item_types(completed),
        [
            "mcp_list_tools",
            "message",
            "mcp_call",
            "mcp_call",
            "mcp_call",
            "message"
        ]
    );
    assert_eq!(completed["usage"]["mcp"]["tool_rounds"], 2);
    assert_eq!(completed["output_text"], "Let me look.The answer is 42.");
}

/// Hang, crash, oversized output and invalid JSON-RPC each fail only their
/// own call: the item carries `error`, and the model still answers.
#[tokio::test]
async fn a_misbehaving_tool_fails_its_own_call_and_the_response_completes() {
    let h = harness().await;
    for (tool, needle) in [
        ("hang", "did not answer within 3000 ms"),
        ("crash", "exited or closed its output"),
        ("big", "larger than the"),
        ("garbage", "invalid JSON-RPC"),
        ("fail", "the tool could not do that"),
    ] {
        h.turn(&[&tool_call(tool, json!({}))]);
        h.turn(&[&tool_call("add", json!({"a": 1, "b": 2}))]);
        h.turn(&["Recovered."]);
        let (status, response) = h.create(body(json!([mcp_tool(json!({}))]))).await;
        assert_eq!(status, StatusCode::OK, "{tool}: {response}");
        assert_eq!(response["status"], "completed", "{tool}");
        let output = response["output"].as_array().unwrap();
        let failed = &output[1];
        assert_eq!(failed["type"], "mcp_call", "{tool}");
        assert_eq!(failed["status"], "failed", "{tool}");
        assert_eq!(failed["output"], Value::Null, "{tool}");
        let error = failed["error"].as_str().unwrap();
        assert!(error.contains(needle), "{tool}: {error}");
        // The same server serves the next call after its child was replaced.
        assert_eq!(output[2]["output"], "3", "{tool}");
        assert_eq!(response["output_text"], "Recovered.", "{tool}");
        assert_eq!(response["usage"]["mcp"]["failed_tool_calls"], 1, "{tool}");
        // The model was told the call failed.
        let prompts = h.prompts();
        assert!(
            prompts[1].contains("<|tool|>Tool call failed: "),
            "{tool}: {}",
            prompts[1]
        );
    }
}

#[tokio::test]
async fn a_streamed_failing_call_reports_mcp_call_failed() {
    let h = harness().await;
    h.turn(&[&tool_call("crash", json!({}))]);
    h.turn(&["Sorry."]);
    let events = h.stream(body(json!([mcp_tool(json!({}))]))).await;
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert!(names.contains(&"response.mcp_call.failed"), "{names:?}");
    assert!(!names.contains(&"response.mcp_call.completed"), "{names:?}");
    assert_eq!(names.last(), Some(&"response.completed"));
}

#[tokio::test]
async fn long_tool_output_is_truncated_and_counted() {
    let h = harness().await;
    h.turn(&[&tool_call("wordy", json!({"size": 6000}))]);
    h.turn(&["Done."]);
    let (_, response) = h.create(body(json!([mcp_tool(json!({}))]))).await;
    let call = &response["output"][1];
    assert_eq!(call["status"], "completed");
    assert!(
        call["output"]
            .as_str()
            .unwrap()
            .ends_with("[output truncated at 4096 bytes]")
    );
    assert_eq!(response["usage"]["mcp"]["truncated_outputs"], 1);
}

#[tokio::test]
async fn the_tool_round_bound_ends_the_response_as_incomplete() {
    let mut config = fake_config();
    config.limits.max_tool_rounds = 1;
    let h = harness_with(Some(config)).await;
    h.turn(&[&tool_call("lookup", json!({"key": "alpha"}))]);
    h.turn(&[&tool_call("lookup", json!({"key": "beta"}))]);
    let (status, response) = h.create(body(json!([mcp_tool(json!({}))]))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["incomplete_details"]["reason"], "max_tool_rounds");
    let output = response["output"].as_array().unwrap();
    assert_eq!(output[1]["output"], "17");
    assert_eq!(output[2]["status"], "incomplete");
    assert_eq!(output[2]["output"], Value::Null);
    assert!(
        output[2]["error"]
            .as_str()
            .unwrap()
            .contains("limit of 1 tool rounds")
    );
    assert_eq!(response["usage"]["mcp"]["tool_rounds"], 1);
    assert_eq!(response["usage"]["mcp"]["tool_calls"], 1);
    // The refused call never reached the model as a third turn.
    assert_eq!(h.prompts().len(), 2);
}

#[tokio::test]
async fn max_tool_calls_caps_the_calls_that_run() {
    let h = harness().await;
    h.turn(&[
        &tool_call("lookup", json!({"key": "alpha"})),
        &tool_call("lookup", json!({"key": "beta"})),
    ]);
    let mut request = body(json!([mcp_tool(json!({}))]));
    request["max_tool_calls"] = json!(1);
    let (_, response) = h.create(request).await;
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["incomplete_details"]["reason"], "max_tool_calls");
    assert_eq!(response["output"][1]["output"], "17");
    assert_eq!(response["output"][2]["status"], "incomplete");
}

#[tokio::test]
async fn a_truncated_model_turn_ends_the_loop_without_running_its_calls() {
    let h = harness().await;
    h.handle
        .token(&tool_call("lookup", json!({"key": "alpha"})));
    h.handle.finish_length();
    let (_, response) = h.create(body(json!([mcp_tool(json!({}))]))).await;
    assert_eq!(response["status"], "incomplete");
    assert_eq!(
        response["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(item_types(&response), ["mcp_list_tools"]);
    assert_eq!(response["usage"]["mcp"]["tool_calls"], 0);
}

#[tokio::test]
async fn a_client_function_call_is_handed_back_and_ends_the_loop() {
    let h = harness().await;
    h.turn(&[
        &tool_call("lookup", json!({"key": "alpha"})),
        &tool_call("get_weather", json!({"city": "Seoul"})),
    ]);
    let tools = json!([
        mcp_tool(json!({})),
        {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}
    ]);
    let (status, response) = h.create(body(tools.clone())).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed");
    assert_eq!(
        item_types(&response),
        ["mcp_list_tools", "mcp_call", "function_call"]
    );
    let function_call = &response["output"][2];
    assert_eq!(function_call["name"], "get_weather");
    assert_eq!(function_call["arguments"], r#"{"city":"Seoul"}"#);
    let call_id = function_call["call_id"].as_str().unwrap().to_string();
    assert_eq!(h.prompts().len(), 1);

    // The caller answers its function call; the chain replays the MCP call
    // as history and does not list the server's tools again.
    h.turn(&["Sunny, and alpha is 17."]);
    let (status, next) = h
        .create(json!({
            "model": "mcp-test-model",
            "previous_response_id": response["id"],
            "tools": tools,
            "input": [{"type": "function_call_output", "call_id": call_id, "output": "sunny"}]
        }))
        .await;
    assert_eq!(status, StatusCode::OK, "{next}");
    assert_eq!(item_types(&next), ["message"]);
    let prompts = h.prompts();
    assert!(
        prompts[0].ends_with("<|tool|>17<|tool|>sunny<|assistant|>"),
        "{}",
        prompts[0]
    );
}

#[tokio::test]
async fn approval_is_required_by_default_and_runs_the_stored_arguments_once() {
    let h = harness().await;
    h.turn(&[&tool_call("add", json!({"a": 17, "b": 25}))]);
    let tools = json!([{"type": "mcp", "server_label": "fake"}]);
    let (status, paused) = h.create(body(tools.clone())).await;
    assert_eq!(status, StatusCode::OK, "{paused}");
    assert_eq!(paused["status"], "completed");
    assert_eq!(
        item_types(&paused),
        ["mcp_list_tools", "mcp_approval_request"]
    );
    let request_item = &paused["output"][1];
    assert_eq!(request_item["server_label"], "fake");
    assert_eq!(request_item["name"], "add");
    assert_eq!(request_item["arguments"], r#"{"a":17,"b":25}"#);
    assert_eq!(paused["usage"]["mcp"]["tool_calls"], 0, "nothing ran yet");
    let approval_id = request_item["id"].as_str().unwrap().to_string();
    let resume = |approve: bool| {
        json!({
            "model": "mcp-test-model",
            "previous_response_id": paused["id"],
            "tools": tools,
            "input": [{
                "type": "mcp_approval_response",
                "approval_request_id": approval_id,
                "approve": approve,
                // Not a field of the item: a client cannot swap the arguments.
                "arguments": "{\"a\":1000,\"b\":1000}"
            }]
        })
    };

    // Continuing without answering, or answering from the wrong response,
    // is refused and consumes nothing.
    let (status, response) = h
        .create(json!({
            "model": "mcp-test-model", "previous_response_id": paused["id"],
            "tools": tools, "input": "never mind"
        }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        error_message(&response).contains("pending MCP approval"),
        "{response}"
    );
    let mut wrong_response = resume(true);
    wrong_response["previous_response_id"] = json!("resp_other");
    let (status, _) = h.create(wrong_response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let mut no_previous = resume(true);
    no_previous
        .as_object_mut()
        .unwrap()
        .remove("previous_response_id");
    let (status, response) = h.create(no_previous).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        error_message(&response).contains("previous_response_id"),
        "{response}"
    );
    let mut undeclared = resume(true);
    undeclared["tools"] = json!([mcp_tool(json!({"allowed_tools": ["lookup"]}))]);
    let (status, response) = h.create(undeclared).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        error_message(&response).contains("do not declare or allow"),
        "{response}"
    );
    assert!(
        h.prompts().len() == 1,
        "no refused request reached the model"
    );

    // Approved: the call runs with the arguments the model produced.
    h.turn(&["It is 42."]);
    let (status, resumed) = h.create(resume(true)).await;
    assert_eq!(status, StatusCode::OK, "{resumed}");
    assert_eq!(item_types(&resumed), ["mcp_call", "message"]);
    let call = &resumed["output"][0];
    assert_eq!(call["approval_request_id"], approval_id.as_str());
    assert_eq!(call["arguments"], r#"{"a":17,"b":25}"#);
    assert_eq!(call["output"], "42");
    let prompts = h.prompts();
    assert!(
        prompts[0].ends_with("<|tool|>42<|assistant|>"),
        "{}",
        prompts[0]
    );

    // The same approval cannot run a second call.
    let (status, response) = h.create(resume(true)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        error_message(&response).contains("does not match a pending approval"),
        "{response}"
    );

    // A later turn chained from the resumed response replays the call once.
    h.turn(&["Still 42."]);
    let (status, _) = h
        .create(json!({
            "model": "mcp-test-model", "previous_response_id": resumed["id"],
            "tools": tools, "input": "Are you sure?"
        }))
        .await;
    assert_eq!(status, StatusCode::OK);
    let prompts = h.prompts();
    assert_eq!(
        prompts[0].matches("<|tool|>42").count(),
        1,
        "{}",
        prompts[0]
    );
}

#[tokio::test]
async fn a_denied_approval_runs_nothing_and_tells_the_model() {
    let h = harness().await;
    h.turn(&[&tool_call("add", json!({"a": 17, "b": 25}))]);
    let tools = json!([{"type": "mcp", "server_label": "fake", "require_approval": "always"}]);
    let (_, paused) = h.create(body(tools.clone())).await;
    let approval_id = paused["output"][1]["id"].as_str().unwrap();
    h.turn(&["I will not add them."]);
    let (status, resumed) = h
        .create(json!({
            "model": "mcp-test-model",
            "previous_response_id": paused["id"],
            "tools": tools,
            "input": [{
                "type": "mcp_approval_response", "approval_request_id": approval_id,
                "approve": false, "reason": "not now"
            }]
        }))
        .await;
    assert_eq!(status, StatusCode::OK, "{resumed}");
    assert_eq!(item_types(&resumed), ["message"]);
    assert_eq!(resumed["usage"]["mcp"]["tool_calls"], 0);
    let prompts = h.prompts();
    assert!(
        prompts[1]
            .contains("<|tool|>The user denied this tool call, so it was not run. Reason: not now"),
        "{}",
        prompts[1]
    );
}

#[tokio::test]
async fn an_approval_that_cannot_be_collected_refuses_the_call() {
    let h = harness().await;
    h.turn(&[&tool_call("add", json!({"a": 17, "b": 25}))]);
    h.turn(&["I could not add them."]);
    let mut request = body(json!([{"type": "mcp", "server_label": "fake"}]));
    request["store"] = json!(false);
    let (status, response) = h.create(request).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(
        item_types(&response),
        ["mcp_list_tools", "mcp_call", "message"]
    );
    let call = &response["output"][1];
    assert_eq!(call["status"], "failed");
    assert_eq!(call["output"], Value::Null);
    assert!(
        call["error"]
            .as_str()
            .unwrap()
            .contains("requires approval"),
        "{call}"
    );
    assert_eq!(
        response["usage"]["mcp"]["tool_calls"], 0,
        "the tool never ran"
    );
}

#[tokio::test]
async fn refused_calls_count_toward_the_tool_round_bound() {
    // A model that keeps calling a tool it may not run must still stop.
    let mut config = fake_config();
    config.limits.max_tool_rounds = 1;
    let h = harness_with(Some(config)).await;
    h.turn(&[&tool_call("add", json!({"a": 1, "b": 2}))]);
    h.turn(&[&tool_call("add", json!({"a": 1, "b": 2}))]);
    let mut request = body(json!([{"type": "mcp", "server_label": "fake"}]));
    request["store"] = json!(false);
    let (status, response) = h.create(request).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["incomplete_details"]["reason"], "max_tool_rounds");
    assert_eq!(response["usage"]["mcp"]["tool_rounds"], 1);
    assert_eq!(response["usage"]["mcp"]["tool_calls"], 0);
    assert_eq!(h.prompts().len(), 2);
}

#[tokio::test]
async fn max_tool_calls_caps_approved_calls_on_resume() {
    let h = harness().await;
    h.turn(&[
        &tool_call("add", json!({"a": 17, "b": 25})),
        &tool_call("add", json!({"a": 1, "b": 2})),
    ]);
    let tools = json!([{"type": "mcp", "server_label": "fake"}]);
    let (_, paused) = h.create(body(tools.clone())).await;
    assert_eq!(
        item_types(&paused),
        [
            "mcp_list_tools",
            "mcp_approval_request",
            "mcp_approval_request"
        ]
    );
    let answers: Vec<Value> = paused["output"].as_array().unwrap()[1..]
        .iter()
        .map(|item| {
            json!({"type": "mcp_approval_response", "approval_request_id": item["id"], "approve": true})
        })
        .collect();
    let _ = h.prompts();
    let (status, resumed) = h
        .create(json!({
            "model": "mcp-test-model", "previous_response_id": paused["id"],
            "tools": tools, "max_tool_calls": 1, "input": answers
        }))
        .await;
    assert_eq!(status, StatusCode::OK, "{resumed}");
    assert_eq!(resumed["status"], "incomplete");
    assert_eq!(resumed["incomplete_details"]["reason"], "max_tool_calls");
    assert_eq!(item_types(&resumed), ["mcp_call", "mcp_call"]);
    assert_eq!(resumed["output"][0]["output"], "42");
    assert_eq!(resumed["output"][1]["status"], "incomplete");
    assert_eq!(resumed["usage"]["mcp"]["tool_calls"], 1);
    assert!(h.prompts().is_empty(), "no model turn after the bound");
}

#[tokio::test]
async fn per_tool_approval_runs_exempt_tools_and_pauses_on_the_rest() {
    let h = harness().await;
    h.turn(&[
        &tool_call("lookup", json!({"key": "alpha"})),
        &tool_call("add", json!({"a": 1, "b": 2})),
    ]);
    let tools = json!([{
        "type": "mcp", "server_label": "fake",
        "require_approval": {"never": {"tool_names": ["lookup"]}}
    }]);
    let (_, response) = h.create(body(tools)).await;
    assert_eq!(
        item_types(&response),
        ["mcp_list_tools", "mcp_call", "mcp_approval_request"]
    );
    assert_eq!(response["output"][1]["output"], "17");
    assert_eq!(response["output"][2]["name"], "add");
}

#[tokio::test]
async fn request_shapes_the_loop_does_not_serve_are_refused() {
    let h = harness().await;
    for (request, needle) in [
        (
            body(json!([{"type": "mcp", "server_label": "nope"}])),
            "not a configured MCP server",
        ),
        (
            body(
                json!([{"type": "mcp", "server_label": "fake", "server_url": "https://example.com/mcp"}]),
            ),
            "'server_url' is not supported",
        ),
        (
            body(json!([mcp_tool(
                json!({"headers": {"Authorization": "Bearer x"}})
            )])),
            "'headers'",
        ),
        (
            body(json!([mcp_tool(json!({})), {"type": "function", "name": "add"}])),
            "tool name 'add'",
        ),
        (
            body(json!([mcp_tool(json!({})), mcp_tool(json!({}))])),
            "more than once",
        ),
        (
            {
                let mut request = body(json!([mcp_tool(json!({}))]));
                request["background"] = json!(true);
                request
            },
            "background mode is not supported together with mcp tools",
        ),
        (
            {
                let mut request = body(json!([mcp_tool(json!({}))]));
                request["context_management"] =
                    json!([{"type": "compaction", "compact_threshold": 1000}]);
                request
            },
            "context_management is not supported together with mcp tools",
        ),
    ] {
        let (status, response) = h.create(request.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{request}: {response}");
        assert!(
            error_message(&response).contains(needle),
            "{request}: {response}"
        );
    }
    assert!(
        h.prompts().is_empty(),
        "no refused request reached the model"
    );
}

#[tokio::test]
async fn allowed_tools_limits_what_the_model_can_reach() {
    let h = harness().await;
    // The model asks for a tool the request did not allow: it is not a
    // declared tool, so the parser drops the call and nothing runs.
    h.turn(&[&tool_call("env_names", json!({})), "No tool for that."]);
    let (status, response) = h
        .create(body(json!([mcp_tool(json!({"allowed_tools": ["add"]}))])))
        .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let listed: Vec<&str> = response["output"][0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(listed, ["add"]);
    assert_eq!(response["usage"]["mcp"]["tool_calls"], 0);
    assert!(!item_types(&response).contains(&"mcp_call"));
}

#[tokio::test]
async fn an_unavailable_server_is_reported_and_the_response_still_completes() {
    let mut config = fake_config();
    config.limits.tool_timeout = Duration::from_secs(2);
    config
        .servers
        .push(fake_spec("broken", &["--startup", "crash"]));
    let h = harness_with(Some(config)).await;
    h.turn(&["I have no tools from that server."]);
    let events = h
        .stream(body(json!([
            {"type": "mcp", "server_label": "broken", "require_approval": "never"}
        ])))
        .await;
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert!(
        names.contains(&"response.mcp_list_tools.failed"),
        "{names:?}"
    );
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["status"], "completed");
    assert_eq!(item_types(completed), ["mcp_list_tools", "message"]);
    assert!(
        completed["output"][0]["error"]
            .as_str()
            .unwrap()
            .contains("exited or closed its output")
    );
    assert_eq!(completed["output"][0]["tools"], json!([]));
}

#[tokio::test]
async fn chat_completions_ignores_the_loop() {
    let h = harness().await;
    h.turn(&[&tool_call("add", json!({"a": 1, "b": 2}))]);
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "mcp-test-model",
                "messages": [{"role": "user", "content": "add"}],
                "tools": [{"type": "function", "function": {"name": "add"}}]
            })
            .to_string(),
        ))
        .unwrap();
    let response = h.app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let chat: Value = serde_json::from_slice(&bytes).unwrap();
    // The call goes back to the client; nothing was run server-side.
    assert_eq!(chat["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(h.prompts().len(), 1);
}

/// Tool-loop responses running at once are capped at --max-queue-depth: one
/// more is refused with a 503 before any model turn or tool call.
#[tokio::test]
async fn tool_loop_responses_past_the_admission_cap_get_503() {
    let h = harness().await;
    let runtime = h.runtime.as_ref().expect("loop enabled");
    let cap = ServerConfig::default().max_queue_depth;
    let permits: Vec<_> = (0..cap)
        .map(|_| runtime.admit_run(cap).expect("under the cap"))
        .collect();
    let (status, response) = h.create(body(json!([mcp_tool(json!({}))]))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{response}");
    assert!(
        error_message(&response).contains("MCP tool-loop responses"),
        "{response}"
    );
    assert!(h.prompts().is_empty(), "no model turn ran");
    drop(permits);
    assert!(runtime.admit_run(cap).is_some(), "places free on drop");
}
