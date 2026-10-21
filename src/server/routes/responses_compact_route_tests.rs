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

//! `POST /v1/responses/compact` and capsule replay on `/v1/responses`
//! (issue #2397), through the real router against a scripted model and a
//! one-token-per-byte tokenizer.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

use crate::server::conversation_store::{ConversationStore, ConversationStoreConfig};
use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::responses_compaction::SUMMARY_INSTRUCTION;
use crate::server::responses_store::{ResponsesStore, ResponsesStoreConfig};
use crate::server::state::ModelMediaSupport;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

/// Marks every turn compaction should summarize away.
const OLD: &str = "OLDTURN";
const SUMMARY: &str = "HANDOFF-SUMMARY";

struct Harness {
    app: Router,
    conversations: Arc<ConversationStore>,
    handle: ScriptedStreamHandle,
    prompts: mpsc::Receiver<String>,
    _options: mpsc::Receiver<crate::server::ServerGenerateOptions>,
}

fn harness(config: ServerConfig) -> Harness {
    let (options_tx, options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let conversations = Arc::new(ConversationStore::new(ConversationStoreConfig::default()));
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        MlxcelTokenizer::stub_all_byte_fallback(),
        PathBuf::from("compact-test-model"),
        batch_metrics,
    )
    .with_conversation_store(Some(conversations.clone()))
    .with_responses_store(Some(Arc::new(ResponsesStore::new(
        ResponsesStoreConfig::default(),
    ))));
    Harness {
        app: create_app(state),
        conversations,
        handle,
        prompts: prompt_rx,
        _options: options_rx,
    }
}

impl Harness {
    /// POST `body`; `tokens`, when given, script the one generation it runs.
    async fn post(&self, path: &str, body: &Value, tokens: Option<&[&str]>) -> (StatusCode, Value) {
        if let Some(tokens) = tokens {
            for token in tokens {
                self.handle.token(token);
            }
            self.handle.finish();
        }
        let request = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request builds");
        let response = self
            .app
            .clone()
            .oneshot(request)
            .await
            .expect("router responds");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .expect("body collects");
        (status, serde_json::from_slice(&bytes).unwrap_or_default())
    }
}

fn history() -> Value {
    let filler = "x".repeat(300);
    json!([
        {"type": "message", "role": "system", "content": "SYS"},
        {"type": "message", "role": "user", "content": format!("{OLD} q1 {filler}")},
        {"type": "message", "role": "assistant", "content": format!("{OLD} a1 {filler}")},
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": format!("{OLD} result {filler}")},
        {"type": "message", "role": "user", "content": "latest question"},
    ])
}

fn compact_body(input: Value) -> Value {
    json!({"model": "compact-test-model", "input": input, "keep_tokens": 32})
}

fn error_code(body: &Value) -> Option<&str> {
    body["error"]["code"].as_str()
}

#[tokio::test]
async fn compact_returns_retained_items_and_one_capsule_that_replays() {
    let h = harness(ServerConfig::default());
    let (status, compacted) = h
        .post(
            "/v1/responses/compact",
            &compact_body(history()),
            Some(&[SUMMARY]),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{compacted}");
    assert_eq!(compacted["object"], "response.compaction");
    let output = compacted["output"].as_array().expect("output array");
    let capsules: Vec<&Value> = output
        .iter()
        .filter(|i| i["type"] == "compaction")
        .collect();
    assert_eq!(capsules.len(), 1, "{output:?}");
    assert_eq!(
        output.last(),
        Some(capsules[0]),
        "the capsule ends the output"
    );
    assert!(
        capsules[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("cmp_"))
    );
    assert_eq!(output[0]["role"], "system", "instructions keep their role");
    assert_eq!(output[1]["content"], "latest question");
    assert_eq!(output.len(), 3, "{output:?}");
    let usage = &compacted["usage"];
    assert_eq!(
        usage["total_tokens"].as_u64(),
        Some(usage["input_tokens"].as_u64().unwrap() + usage["output_tokens"].as_u64().unwrap())
    );
    assert_eq!(usage["output_tokens"], 1, "usage counts the summary pass");

    // One summary pass, over the discarded turns plus the handoff instruction.
    let summary_prompt = h.prompts.try_recv().expect("summary generation ran");
    assert!(summary_prompt.contains(OLD) && summary_prompt.contains(&SUMMARY_INSTRUCTION[..40]));
    assert!(h.prompts.try_recv().is_err(), "exactly one generation");

    // Replay: `[...output, new user message]` renders the summary, not the
    // discarded turns.
    let mut input = output.clone();
    input.push(json!({"type": "message", "role": "user", "content": "follow up"}));
    let body = json!({"model": "compact-test-model", "input": input, "store": true});
    let (status, response) = h.post("/v1/responses", &body, Some(&["ok"])).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let prompt = h.prompts.try_recv().expect("generation ran");
    for kept in ["SYS", SUMMARY, "latest question", "follow up"] {
        assert!(prompt.contains(kept), "{kept} missing from {prompt}");
    }
    assert!(!prompt.contains(OLD), "discarded turns replayed: {prompt}");

    // A chained turn rehydrates the stored capsule and expands it again.
    let chained = json!({
        "model": "compact-test-model",
        "previous_response_id": response["id"],
        "input": "and then",
        "store": true,
    });
    let (status, body) = h.post("/v1/responses", &chained, Some(&["ok"])).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let prompt = h.prompts.try_recv().expect("generation ran");
    assert!(
        prompt.contains(SUMMARY) && prompt.contains("and then"),
        "{prompt}"
    );
    assert!(!prompt.contains(OLD), "{prompt}");
    let next = json!({"model": "compact-test-model", "previous_response_id": body["id"],
        "input": "third continuation", "store": false});
    let (status, body) = h.post("/v1/responses", &next, Some(&["ok"])).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let prompt = h.prompts.try_recv().expect("third generation ran");
    for kept in [
        "SYS",
        SUMMARY,
        "latest question",
        "follow up",
        "and then",
        "third continuation",
    ] {
        assert!(prompt.contains(kept), "{kept} missing from {prompt}");
    }
    assert!(!prompt.contains(OLD), "{prompt}");

    // A chained turn that resends the system message renders it once: the
    // stored input already carries it in front of the capsule.
    let resent = json!({
        "model": "compact-test-model",
        "previous_response_id": response["id"],
        "input": [
            {"type": "message", "role": "system", "content": "SYS"},
            {"type": "message", "role": "user", "content": "resent"},
        ],
        "store": false,
    });
    let (status, body) = h.post("/v1/responses", &resent, Some(&["ok"])).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let prompt = h.prompts.try_recv().expect("generation ran");
    assert_eq!(prompt.matches("<|system|>SYS").count(), 1, "{prompt}");
    assert!(
        prompt.contains(SUMMARY) && prompt.contains("resent"),
        "{prompt}"
    );
}

/// A 1x1 PNG served by a loopback origin that counts its connections.
const PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2N1ekAAAAASUVORK5CYII=";

async fn png_origin() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("loopback listener binds");
    let addr = listener.local_addr().expect("listener address");
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    let png = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, PNG_BASE64)
        .expect("png decodes");
    let server = tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            counter.fetch_add(1, Ordering::SeqCst);
            let mut request = [0u8; 2048];
            let _ = socket.read(&mut request).await;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                png.len()
            );
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(&png).await;
        }
    });
    (format!("http://{addr}/cat.png"), hits, server)
}

#[tokio::test]
async fn a_compaction_reads_each_image_url_once() {
    let _private = crate::server::media_net::allow_private_media_urls_in_tests();
    let (url, hits, server) = png_origin().await;
    let (options_tx, _options_rx) = mpsc::channel();
    // Counts media on the "worker" (image soft tokens included) and answers
    // every generation with one token, which serves as the summary.
    let provider = Arc::new(ModelProvider::media_expanding_for_route_tests(
        options_tx, 50,
    ));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        MlxcelTokenizer::stub_all_byte_fallback(),
        PathBuf::from("compact-test-model"),
        batch_metrics,
    )
    .with_media_support(ModelMediaSupport {
        image: true,
        ..ModelMediaSupport::default()
    });
    let app = create_app(state);
    let input = json!([
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": format!("{OLD} look {}", "x".repeat(300))},
            {"type": "input_image", "image_url": url},
        ]},
        {"type": "message", "role": "assistant", "content": format!("{OLD} {}", "y".repeat(300))},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "and again"},
            {"type": "input_image", "image_url": url},
        ]},
        {"type": "message", "role": "assistant", "content": "seen"},
        {"type": "message", "role": "user", "content": "latest"},
    ]);
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses/compact")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(compact_body(input).to_string()))
        .expect("request builds");
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body collects");
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "one fetch for the whole compaction"
    );
    server.abort();
}

#[tokio::test]
async fn unversioned_alias_compacts_and_counts_the_replayed_prompt() {
    let h = harness(ServerConfig::default());
    let (status, compacted) = h
        .post(
            "/responses/compact",
            &compact_body(history()),
            Some(&[SUMMARY]),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{compacted}");
    let _ = h.prompts.try_recv();
    let input = compacted["output"].clone();
    let (status, original) = h
        .post(
            "/v1/responses/input_tokens",
            &json!({"model": "m", "input": history()}),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{original}");
    let (status, replayed) = h
        .post(
            "/v1/responses/input_tokens",
            &json!({"model": "m", "input": input}),
            None,
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{replayed}");
    assert!(replayed["input_tokens"].as_u64() < original["input_tokens"].as_u64());
}

#[tokio::test]
async fn a_summary_that_is_not_shorter_fails_with_context_length_exceeded() {
    let h = harness(ServerConfig::default());
    let long = "z".repeat(3000);
    let (status, body) = h
        .post(
            "/v1/responses/compact",
            &compact_body(history()),
            Some(&[long.as_str()]),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(error_code(&body), Some("context_length_exceeded"), "{body}");
    assert!(
        body.get("output").is_none(),
        "no compacted context returned"
    );
}

#[tokio::test]
async fn history_over_one_window_fails_without_generating() {
    let h = harness(ServerConfig {
        context_size: 256,
        ..ServerConfig::default()
    });
    let (status, body) = h
        .post("/v1/responses/compact", &compact_body(history()), None)
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(error_code(&body), Some("context_length_exceeded"), "{body}");
    assert!(h.prompts.try_recv().is_err(), "no summary generation ran");
}

#[tokio::test]
async fn bad_capsules_and_budgets_are_invalid_requests() {
    let h = harness(ServerConfig::default());
    let v2 = base64::Engine::encode(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
        json!({"v": 2, "summary": "s", "retained": []}).to_string(),
    );
    for capsule in ["%%%not-base64", v2.as_str()] {
        let input = json!([
            {"type": "compaction", "id": "cmp_x", "encrypted_content": capsule},
            {"type": "message", "role": "user", "content": "hi"},
        ]);
        for path in ["/v1/responses", "/v1/responses/compact"] {
            let body = json!({"model": "compact-test-model", "input": input, "store": false});
            let (status, body) = h.post(path, &body, None).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
            assert_eq!(
                body["error"]["type"], "invalid_request_error",
                "{path}: {body}"
            );
        }
    }
    let mut body = compact_body(history());
    body["max_output_tokens"] = json!(16385);
    let (status, _) = h.post("/v1/responses/compact", &body, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(h.prompts.try_recv().is_err(), "no generation ran");
}

#[test]
fn compact_output_must_fit_the_replay_body_limit() {
    let output = vec![json!({"type":"message", "role":"user", "content":"retained"})];
    let size = serde_json::to_vec(&json!({"model":"m", "input":output}))
        .unwrap()
        .len();
    assert!(super::validate_replay_size(&output, "m", size).is_ok());
    let error = super::validate_replay_size(&output, "m", size - 1)
        .unwrap_err()
        .into_error_response();
    assert_eq!(error.status, StatusCode::BAD_REQUEST);
    assert!(error.error.message.contains("above the"));
}

#[tokio::test]
async fn conversation_compaction_deduplicates_instructions_and_appends_only_new_items() {
    let h = harness(ServerConfig::default());
    let (status, compacted) = h
        .post(
            "/v1/responses/compact",
            &compact_body(history()),
            Some(&[SUMMARY]),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let _ = h.prompts.try_recv();
    let initial = json!({"model":"compact-test-model", "conversation":"conv_compact", "input":history(), "store":true});
    assert_eq!(
        h.post("/v1/responses", &initial, Some(&["old answer"]))
            .await
            .0,
        StatusCode::OK
    );
    let _ = h.prompts.try_recv();
    let before = h.conversations.get("conv_compact").unwrap().items.len();
    let replay = json!({"model":"compact-test-model", "conversation":"conv_compact", "input":compacted["output"], "store":true});
    let (status, body) = h.post("/v1/responses", &replay, Some(&["ok"])).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let prompt = h.prompts.try_recv().unwrap();
    assert_eq!(prompt.matches("SYS").count(), 1, "{prompt}");
    assert!(!prompt.contains(OLD), "{prompt}");
    let after = h.conversations.get("conv_compact").unwrap().items.len();
    assert_eq!(
        after - before,
        compacted["output"].as_array().unwrap().len() + body["output"].as_array().unwrap().len()
    );
}
