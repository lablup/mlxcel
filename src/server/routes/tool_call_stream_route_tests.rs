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

//! Route-level incremental tool-call streaming (#2396): the scripted provider
//! generates a tool call one character per token through the real
//! `/v1/chat/completions` and `/v1/responses` streaming handlers, and the
//! tests read the SSE body a client would.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const HERMES: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>";
const QWEN: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

fn tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}
            }}
        }},
        {"type": "function", "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}
        }}
    ])
}

/// An app whose model generates `raw` one character per token.
fn app_generating(raw: &str) -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let (provider, handle) = ModelProvider::scripted_streaming_for_route_tests(options_tx);
    for ch in raw.chars() {
        handle.token(&ch.to_string());
    }
    handle.finish();
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    create_app(AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("ok".to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("tool-stream-test-model"),
        batch_metrics,
    ))
}

/// The `data:` payloads of an SSE response, with their `event:` names.
async fn post_sse(app: Router, path: &str, body: Value) -> Vec<(Option<String>, Value)> {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app.oneshot(request).await.expect("router responds");
    assert_eq!(response.status(), StatusCode::OK, "{path}");
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body collects");
    String::from_utf8_lossy(&bytes)
        .split("\n\n")
        .filter_map(|frame| {
            let mut event = None;
            let mut data = None;
            for line in frame.lines() {
                if let Some(name) = line.strip_prefix("event:") {
                    event = Some(name.trim().to_string());
                } else if let Some(payload) = line.strip_prefix("data:") {
                    data = serde_json::from_str(payload.trim()).ok();
                }
            }
            Some((event, data?))
        })
        .collect()
}

#[derive(Debug, Default)]
struct ChatCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    argument_chunks: usize,
}

async fn chat_stream(raw: &str, extra: Value) -> (BTreeMap<u64, ChatCall>, Option<String>) {
    let mut body = json!({
        "model": "tool-stream-test-model",
        "stream": true,
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": tools()
    });
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        body.extend(extra.clone());
    }
    let frames = post_sse(app_generating(raw), "/v1/chat/completions", body).await;
    let mut calls: BTreeMap<u64, ChatCall> = BTreeMap::new();
    let mut finish_reason = None;
    for (_, chunk) in frames {
        let Some(choice) = chunk["choices"].get(0) else {
            continue;
        };
        if let Some(reason) = choice["finish_reason"].as_str() {
            finish_reason = Some(reason.to_string());
        }
        let Some(deltas) = choice["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for delta in deltas {
            let call = calls
                .entry(delta["index"].as_u64().expect("tool call index"))
                .or_default();
            if let Some(id) = delta["id"].as_str() {
                assert!(call.id.is_none(), "a call is started once");
                call.id = Some(id.to_string());
            }
            if let Some(name) = delta["function"]["name"].as_str() {
                call.name = Some(name.to_string());
            }
            if let Some(args) = delta["function"]["arguments"].as_str()
                && !args.is_empty()
            {
                call.arguments.push_str(args);
                call.argument_chunks += 1;
            }
        }
    }
    (calls, finish_reason)
}

#[tokio::test]
async fn chat_streams_tool_call_arguments_in_several_chunks() {
    let (calls, finish_reason) = chat_stream(HERMES, json!({})).await;
    assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(calls.len(), 1, "{calls:?}");
    let call = &calls[&0];
    assert_eq!(call.name.as_deref(), Some("get_weather"));
    assert!(call.id.as_deref().is_some_and(|id| id.starts_with("call_")));
    // The concatenated deltas are the model's own argument text.
    assert_eq!(call.arguments, r#"{"city": "Paris", "days": 3}"#);
    assert!(call.argument_chunks > 1, "{call:?}");
}

#[tokio::test]
async fn chat_named_tool_choice_streams_only_the_chosen_call() {
    let raw = format!(
        "<tool_call>{{\"name\": \"get_time\", \"arguments\": {{\"tz\": \"UTC\"}}}}</tool_call>{HERMES}"
    );
    let choice = json!({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}});
    let (calls, finish_reason) = chat_stream(&raw, choice).await;
    assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(
        calls.keys().copied().collect::<Vec<_>>(),
        vec![1],
        "{calls:?}"
    );
    assert_eq!(calls[&1].name.as_deref(), Some("get_weather"));
    assert!(calls[&1].argument_chunks > 1);
}

#[tokio::test]
async fn responses_streams_function_call_argument_deltas() {
    let body = json!({
        "model": "tool-stream-test-model",
        "stream": true,
        "input": "weather in Paris?",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}
            }}
        }]
    });
    let frames = post_sse(app_generating(QWEN), "/v1/responses", body).await;
    let named = |name: &str| -> Vec<&Value> {
        frames
            .iter()
            .filter(|(event, data)| {
                event.as_deref() == Some(name) || data["type"].as_str() == Some(name)
            })
            .map(|(_, data)| data)
            .collect()
    };

    let added = named("response.output_item.added");
    let added: Vec<&Value> = added
        .into_iter()
        .filter(|e| e["item"]["type"] == "function_call")
        .collect();
    assert_eq!(added.len(), 1, "{frames:?}");
    let item_id = added[0]["item"]["id"]
        .as_str()
        .expect("item id")
        .to_string();
    let call_id = added[0]["item"]["call_id"]
        .as_str()
        .expect("call id")
        .to_string();
    assert_eq!(added[0]["item"]["status"], "in_progress");

    let deltas: Vec<&str> = named("response.function_call_arguments.delta")
        .into_iter()
        .filter(|e| e["item_id"] == item_id.as_str())
        .filter_map(|e| e["delta"].as_str())
        .collect();
    assert!(deltas.len() > 1, "{deltas:?}");
    let expected = r#"{"city":"Paris","days":3}"#;
    assert_eq!(deltas.concat(), expected);

    let done = named("response.function_call_arguments.done");
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["arguments"], expected);
    let item_done: Vec<&Value> = named("response.output_item.done")
        .into_iter()
        .filter(|e| e["item"]["type"] == "function_call")
        .collect();
    assert_eq!(item_done[0]["item"]["arguments"], expected);
    assert_eq!(item_done[0]["item"]["call_id"], call_id.as_str());

    // The completed response carries the call under the streamed call id.
    let completed = named("response.completed");
    let output = completed[0]["response"]["output"]
        .as_array()
        .expect("output items");
    let call = output
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function_call item");
    assert_eq!(call["call_id"], call_id.as_str());
    assert_eq!(call["arguments"], expected);
}
