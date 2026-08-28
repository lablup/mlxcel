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

//! `/apply-template` and the input-token-count routes (#1442).
//!
//! The counting routes are checked against the prompt the generation path
//! really builds rather than against a fixed number: the test renders through
//! `/apply-template`, tokenizes that prompt through `/tokenize`, and asserts the
//! count matches. A drift between the counter and the renderer therefore fails
//! here instead of being discovered by a client whose context budget is wrong.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tower::ServiceExt;

use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

/// A template that renders roles and content, so the endpoint's answer depends
/// on the request rather than being a constant.
const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if tools %}<|tools|>{{ tools | length }}{% endif %}\
{% if add_generation_prompt %}<|assistant|>{% endif %}";

fn app() -> Router {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        MlxcelTokenizer::stub_with_byte_fallback(),
        PathBuf::from("prompt-inspection-test-model"),
        batch_metrics,
    );
    create_app(state)
}

async fn post(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app().oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, parsed)
}

fn chat_body() -> serde_json::Value {
    serde_json::json!({
        "model": "prompt-inspection-test-model",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "Hello"}
        ]
    })
}

#[tokio::test]
async fn apply_template_answers_the_rendered_prompt() {
    let (status, body) = post("/apply-template", chat_body()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["prompt"],
        "<|system|>be brief<|user|>Hello<|assistant|>"
    );
    assert!(
        body.get("choices").is_none(),
        "/apply-template must not answer a completion, got {body}"
    );
}

#[tokio::test]
async fn apply_template_reflects_tools_in_the_render() {
    let mut with_tools = chat_body();
    with_tools["tools"] = serde_json::json!([
        {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    ]);
    let (status, body) = post("/apply-template", with_tools).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body["prompt"]
            .as_str()
            .expect("prompt string")
            .contains("<|tools|>1"),
        "the tools array must reach the template, got {body}"
    );
}

#[tokio::test]
async fn apply_template_refuses_an_oversized_tool_array() {
    let mut too_many = chat_body();
    too_many["tools"] = serde_json::Value::Array(
        (0..129)
            .map(|i| {
                serde_json::json!({
                    "type": "function",
                    "function": {"name": format!("f{i}"), "parameters": {"type": "object"}}
                })
            })
            .collect(),
    );
    let (status, body) = post("/apply-template", too_many).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(
        body["error"]["message"],
        "Too many tools: 129. Maximum allowed is 128."
    );
}

#[tokio::test]
async fn both_chat_input_token_paths_answer_the_same_count() {
    let mut counts = Vec::new();
    for path in [
        "/chat/completions/input_tokens",
        "/v1/chat/completions/input_tokens",
    ] {
        let (status, body) = post(path, chat_body()).await;
        assert_eq!(status, StatusCode::OK, "{path}: {body}");
        counts.push(body["input_tokens"].as_u64().expect("a count"));
    }
    assert_eq!(counts[0], counts[1]);
}

#[tokio::test]
async fn the_chat_input_token_count_matches_the_rendered_prompt() {
    let (_, rendered) = post("/apply-template", chat_body()).await;
    let prompt = rendered["prompt"].as_str().expect("prompt string");

    let (_, tokenized) = post(
        "/tokenize",
        serde_json::json!({"content": prompt, "add_special": true}),
    )
    .await;
    let expected = tokenized["tokens"].as_array().expect("token array").len() as u64;

    let (_, counted) = post("/chat/completions/input_tokens", chat_body()).await;
    assert_eq!(
        counted["input_tokens"].as_u64().expect("a count"),
        expected,
        "the count must be of the prompt the server would actually prefill"
    );
}

#[tokio::test]
async fn the_count_grows_with_the_conversation() {
    let (_, short) = post("/chat/completions/input_tokens", chat_body()).await;
    let mut longer = chat_body();
    longer["messages"] = serde_json::json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Hello"}
    ]);
    let (_, long) = post("/chat/completions/input_tokens", longer).await;
    assert!(
        long["input_tokens"].as_u64().expect("a count")
            > short["input_tokens"].as_u64().expect("a count"),
        "a longer conversation must count more tokens: {short} vs {long}"
    );
}

#[tokio::test]
async fn both_responses_input_token_paths_answer_the_same_count() {
    let body = serde_json::json!({
        "model": "prompt-inspection-test-model",
        "input": "Hello"
    });
    let mut counts = Vec::new();
    for path in ["/responses/input_tokens", "/v1/responses/input_tokens"] {
        let (status, response) = post(path, body.clone()).await;
        assert_eq!(status, StatusCode::OK, "{path}: {response}");
        counts.push(response["input_tokens"].as_u64().expect("a count"));
    }
    assert_eq!(counts[0], counts[1]);
    assert!(counts[0] > 0);
}

#[tokio::test]
async fn the_anthropic_count_route_answers_the_same_key() {
    // `/v1/messages/count_tokens` predates this issue and already answers
    // `input_tokens`; the three surfaces must not drift apart on the key name.
    let (status, body) = post(
        "/v1/messages/count_tokens",
        serde_json::json!({
            "model": "prompt-inspection-test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hello"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["input_tokens"].as_u64().is_some(), "{body}");
}

/// A template that also branches on the reasoning kwargs, so the reasoning
/// modes are observable in the rendered prompt.
const REASONING_TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if enable_thinking %}<|think|>{% endif %}\
{% if reasoning_effort %}<|effort:{{ reasoning_effort }}|>{% endif %}\
{% if add_generation_prompt %}<|assistant|>{% endif %}";

async fn post_with_template(
    template: &str,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let (options_tx, _options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::recording_for_route_tests(options_tx));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(template.to_string()),
        MlxcelTokenizer::stub_with_byte_fallback(),
        PathBuf::from("prompt-inspection-test-model"),
        batch_metrics,
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = create_app(state)
        .oneshot(request)
        .await
        .expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("body collects");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, parsed)
}

#[tokio::test]
async fn apply_template_reflects_the_reasoning_modes() {
    // `chat_template_kwargs.enable_thinking` and `reasoning_effort` both reach
    // the template through the same resolution the generating routes use, so
    // the rendered prompt differs by mode rather than being constant.
    let mut thinking = chat_body();
    thinking["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
    let (status, on) = post_with_template(REASONING_TEMPLATE, "/apply-template", thinking).await;
    assert_eq!(status, StatusCode::OK, "{on}");
    assert!(
        on["prompt"].as_str().expect("prompt").contains("<|think|>"),
        "enable_thinking must reach the template, got {on}"
    );

    let mut not_thinking = chat_body();
    not_thinking["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
    let (_, off) = post_with_template(REASONING_TEMPLATE, "/apply-template", not_thinking).await;
    assert!(
        !off["prompt"]
            .as_str()
            .expect("prompt")
            .contains("<|think|>"),
        "enable_thinking:false must not prime the reasoning block, got {off}"
    );

    let mut effort = chat_body();
    effort["reasoning_effort"] = serde_json::json!("high");
    let (_, rendered) = post_with_template(REASONING_TEMPLATE, "/apply-template", effort).await;
    assert!(
        rendered["prompt"]
            .as_str()
            .expect("prompt")
            .contains("<|effort:high|>"),
        "reasoning_effort must reach a template that references it, got {rendered}"
    );
}

#[tokio::test]
async fn the_input_token_count_follows_the_reasoning_mode() {
    // The count is of the prompt the mode really produces, so each mode's count
    // must equal the tokenization of that mode's own rendered prompt. Asserting
    // the relationship rather than a fixed number keeps this independent of the
    // stub vocabulary, whose tiny alphabet can collapse two different prompts
    // onto the same token count.
    for enable in [true, false] {
        let mut body = chat_body();
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": enable});

        let (_, rendered) =
            post_with_template(REASONING_TEMPLATE, "/apply-template", body.clone()).await;
        let prompt = rendered["prompt"].as_str().expect("prompt").to_string();
        assert_eq!(
            prompt.contains("<|think|>"),
            enable,
            "enable_thinking={enable} must decide the reasoning block: {prompt}"
        );

        let (_, tokenized) = post(
            "/tokenize",
            serde_json::json!({"content": prompt, "add_special": true}),
        )
        .await;
        let expected = tokenized["tokens"].as_array().expect("token array").len() as u64;

        let (_, counted) =
            post_with_template(REASONING_TEMPLATE, "/chat/completions/input_tokens", body).await;
        assert_eq!(
            counted["input_tokens"].as_u64().expect("a count"),
            expected,
            "enable_thinking={enable}: the count must be of the prompt that mode renders"
        );
    }
}

#[tokio::test]
async fn a_body_without_a_model_is_served_the_way_b10621_serves_it() {
    // b10621's own routes take messages or input and nothing else, so a client
    // written against llama-server sends no `model`. The OpenAI generating
    // routes still require one; these three do not.
    let (status, body) = post(
        "/apply-template",
        serde_json::json!({"messages": [{"role": "user", "content": "Hello"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["prompt"], "<|user|>Hello<|assistant|>");

    for path in [
        "/chat/completions/input_tokens",
        "/v1/chat/completions/input_tokens",
    ] {
        let (status, body) = post(
            path,
            serde_json::json!({"messages": [{"role": "user", "content": "Hello"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{path}: {body}");
        assert!(body["input_tokens"].as_u64().is_some(), "{path}: {body}");
    }

    for path in ["/responses/input_tokens", "/v1/responses/input_tokens"] {
        let (status, body) = post(path, serde_json::json!({"input": "Hello"})).await;
        assert_eq!(status, StatusCode::OK, "{path}: {body}");
        assert!(body["input_tokens"].as_u64().is_some(), "{path}: {body}");
    }
}

#[tokio::test]
async fn a_malformed_body_is_a_400_not_a_422() {
    let (status, body) = post("/apply-template", serde_json::json!({"messages": 7})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}
