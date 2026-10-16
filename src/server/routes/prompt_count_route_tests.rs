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

//! The counting routes against the usage their generating siblings report
//! (issue #1788).
//!
//! The provider here answers as if the worker expanded every media item into
//! [`TOKENS_PER_MEDIA`] soft tokens, for a generation and a count-only request
//! alike. A route that counts by encoding the rendered text sees one
//! placeholder per image instead and lands far below that, so each test both
//! holds `input_tokens` to `usage` and proves the count went through the
//! worker.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::Router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use tower::ServiceExt;

use crate::server::state::ModelMediaSupport;
use crate::server::{
    AppState, ChatTemplateProcessor, ModelProvider, ServerConfig, ServerGenerateOptions, create_app,
};
use crate::tokenizer::MlxcelTokenizer;

/// Soft tokens the fake worker adds per media item. Far above any rendered
/// prompt in these tests, so a text-only encode can never reach it.
const TOKENS_PER_MEDIA: usize = 1000;

const TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\
{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

/// A 1x1 PNG, inline, so no test touches the network.
const PNG_BASE64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2N1ekAAAAASUVORK5CYII=";

fn png_data_uri() -> String {
    format!("data:image/png;base64,{PNG_BASE64}")
}

fn app_with(
    support: ModelMediaSupport,
    config: ServerConfig,
    allowlist: Vec<PathBuf>,
) -> (Router, mpsc::Receiver<ServerGenerateOptions>) {
    let (options_tx, options_rx) = mpsc::channel();
    let provider = Arc::new(ModelProvider::media_expanding_for_route_tests(
        options_tx,
        TOKENS_PER_MEDIA,
    ));
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        config,
        ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
        // One token per byte, which is how the fake provider counts a prompt,
        // so a text-only count can be held to `usage` as well as to the
        // worker-side expansion (issue #2337).
        MlxcelTokenizer::stub_all_byte_fallback(),
        PathBuf::from("prompt-count-test-model"),
        batch_metrics,
    )
    .with_media_support(support)
    .with_video_dir_allowlist(Arc::new(allowlist));
    (create_app(state), options_rx)
}

fn image_app() -> (Router, mpsc::Receiver<ServerGenerateOptions>) {
    app_with(
        ModelMediaSupport {
            image: true,
            ..ModelMediaSupport::default()
        },
        ServerConfig::default(),
        Vec::new(),
    )
}

async fn post(app: &Router, path: &str, body: &serde_json::Value) -> serde_json::Value {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app.clone().oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .expect("body collects");
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
    assert_eq!(status, StatusCode::OK, "{path}: {parsed}");
    parsed
}

/// POST a streaming body and return its SSE events as `(event, data)` pairs,
/// in wire order.
async fn post_sse(
    app: &Router,
    path: &str,
    body: &serde_json::Value,
) -> Vec<(String, serde_json::Value)> {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");
    let response = app.clone().oneshot(request).await.expect("router responds");
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
            Some((event?, data?))
        })
        .collect()
}

fn count(body: &serde_json::Value) -> u64 {
    body["input_tokens"]
        .as_u64()
        .expect("an input_tokens count")
}

/// The options the worker received for the next dispatched request.
fn dispatched(options_rx: &mpsc::Receiver<ServerGenerateOptions>) -> ServerGenerateOptions {
    options_rx
        .try_recv()
        .expect("the request reached the worker")
}

fn chat_image_body() -> serde_json::Value {
    serde_json::json!({
        "model": "prompt-count-test-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": png_data_uri()}}
        ]}]
    })
}

#[tokio::test]
async fn chat_image_count_equals_the_usage_of_the_same_body() {
    let (app, options_rx) = image_app();
    let body = chat_image_body();
    let usage = post(&app, "/v1/chat/completions", &body).await["usage"]["prompt_tokens"]
        .as_u64()
        .expect("usage.prompt_tokens");
    let generation = dispatched(&options_rx);
    assert!(!generation.count_prompt_only, "a generation is not a count");
    assert!(
        usage > TOKENS_PER_MEDIA as u64,
        "the fake worker expanded the image: {usage}"
    );

    for path in [
        "/chat/completions/input_tokens",
        "/v1/chat/completions/input_tokens",
    ] {
        assert_eq!(count(&post(&app, path, &body).await), usage, "{path}");
        let counted = dispatched(&options_rx);
        assert!(counted.count_prompt_only, "{path} counts on the worker");
        assert_eq!(counted.max_tokens, 1, "{path}");
        assert!(counted.prompt_cache_ctx.is_none(), "{path}");
    }
}

#[tokio::test]
async fn responses_image_count_equals_the_usage_of_the_same_body() {
    let (app, options_rx) = image_app();
    let body = serde_json::json!({
        "model": "prompt-count-test-model",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is this"},
                {"type": "input_image", "image_url": png_data_uri()}
            ]
        }]
    });
    let usage = post(&app, "/v1/responses", &body).await["usage"]["input_tokens"]
        .as_u64()
        .expect("usage.input_tokens");
    let _ = dispatched(&options_rx);
    assert!(usage > TOKENS_PER_MEDIA as u64, "{usage}");

    for path in ["/responses/input_tokens", "/v1/responses/input_tokens"] {
        assert_eq!(count(&post(&app, path, &body).await), usage, "{path}");
        assert!(dispatched(&options_rx).count_prompt_only, "{path}");
    }
}

#[tokio::test]
async fn anthropic_image_count_equals_the_usage_of_the_same_body() {
    let (app, options_rx) = image_app();
    let body = serde_json::json!({
        "model": "prompt-count-test-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": PNG_BASE64
            }}
        ]}]
    });
    let usage = post(&app, "/v1/messages", &body).await["usage"]["input_tokens"]
        .as_u64()
        .expect("usage.input_tokens");
    let _ = dispatched(&options_rx);
    assert!(usage > TOKENS_PER_MEDIA as u64, "{usage}");

    for path in ["/v1/messages/count_tokens", "/messages/count_tokens"] {
        assert_eq!(count(&post(&app, path, &body).await), usage, "{path}");
        assert!(dispatched(&options_rx).count_prompt_only, "{path}");
    }
}

#[tokio::test]
async fn a_text_only_count_stays_local_and_unchanged() {
    // Text has no expansion to wait for, so it is still encoded on the request
    // thread and never queued behind generations on the worker.
    let (app, options_rx) = image_app();
    let chat = serde_json::json!({
        "model": "prompt-count-test-model",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let rendered = post(&app, "/apply-template", &chat).await["prompt"]
        .as_str()
        .expect("prompt")
        .to_string();
    let tokenized = post(
        &app,
        "/tokenize",
        &serde_json::json!({"content": rendered, "add_special": true}),
    )
    .await["tokens"]
        .as_array()
        .expect("tokens")
        .len() as u64;

    let anthropic = serde_json::json!({
        "model": "prompt-count-test-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hello"}]
    });
    for (path, body) in [
        ("/v1/chat/completions/input_tokens", &chat),
        ("/v1/messages/count_tokens", &anthropic),
    ] {
        assert_eq!(count(&post(&app, path, body).await), tokenized, "{path}");
    }
    assert!(
        options_rx.try_recv().is_err(),
        "a text-only count must not reach the worker"
    );
}

/// The PR #1777 shape on the frame fallback: two clips, each read as ordered
/// stills, so the worker sees eight images.
///
/// `#[ignore]` because it decodes real clips, the convention
/// `chat_request_tests::video_part_expands_through_a_real_clip` documents. Run
/// with `-- --ignored` on a host with ffmpeg 5.0+.
#[tokio::test]
#[ignore = "needs ffmpeg 5.0+ on PATH"]
async fn video_fallback_count_equals_the_usage_of_the_same_body() {
    assert!(
        crate::multimodal::video::ffmpeg_available(),
        "this test was run explicitly, so a missing ffmpeg is a failure and not a skip"
    );
    let dir = std::env::temp_dir().join(format!("mlxcel-count-clips-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("clip dir");
    let mut clips = Vec::new();
    for (index, source) in ["testsrc2", "smptebars"].into_iter().enumerate() {
        let clip = dir.join(format!("clip-{index}.mp4"));
        let status = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
            .arg(format!("{source}=size=64x48:rate=10"))
            .args(["-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
            .arg(&clip)
            .status()
            .expect("ffmpeg runs");
        assert!(
            status.success(),
            "ffmpeg could not build {}",
            clip.display()
        );
        clips.push(clip);
    }
    let (app, options_rx) = app_with(
        ModelMediaSupport {
            image: true,
            video_frames_fallback: true,
            ..ModelMediaSupport::default()
        },
        ServerConfig {
            video_max_frames: 4,
            ..ServerConfig::default()
        },
        vec![dir.canonicalize().expect("canonical clip dir")],
    );
    let body = serde_json::json!({
        "model": "prompt-count-test-model",
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": format!("file://{}", clips[0].display())}},
            {"type": "video_url", "video_url": {"url": format!("file://{}", clips[1].display())}},
            {"type": "text", "text": "Compare the two clips."}
        ]}]
    });

    let usage = post(&app, "/v1/chat/completions", &body).await["usage"]["prompt_tokens"]
        .as_u64()
        .expect("usage.prompt_tokens");
    let _ = dispatched(&options_rx);
    let counted = count(&post(&app, "/v1/chat/completions/input_tokens", &body).await);
    assert!(dispatched(&options_rx).count_prompt_only);
    std::fs::remove_dir_all(&dir).expect("clip dir removed");

    assert_eq!(counted, usage);
    assert!(
        usage > 8 * TOKENS_PER_MEDIA as u64,
        "two clips of four frames each reach the worker as eight images: {usage}"
    );
}

/// `message_start.usage.input_tokens` of a streamed `/v1/messages` reply,
/// after checking that `message_start` is the stream's first event.
async fn streamed_input_tokens(app: &Router, body: &serde_json::Value) -> u64 {
    let mut streaming = body.clone();
    streaming["stream"] = serde_json::Value::Bool(true);
    let events = post_sse(app, "/v1/messages", &streaming).await;
    let (first, data) = events.first().expect("the stream carries events");
    assert_eq!(
        first, "message_start",
        "message_start opens the stream: {events:?}"
    );
    assert!(
        events
            .iter()
            .skip(1)
            .all(|(name, _)| name != "message_start"),
        "message_start is sent once: {events:?}"
    );
    assert!(
        events.iter().any(|(name, _)| name == "content_block_delta"),
        "the generated token streamed after message_start: {events:?}"
    );
    data["message"]["usage"]["input_tokens"]
        .as_u64()
        .expect("message_start.message.usage.input_tokens")
}

fn anthropic_image_body() -> serde_json::Value {
    serde_json::json!({
        "model": "prompt-count-test-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": PNG_BASE64
            }}
        ]}]
    })
}

/// Issue #2337: the streaming `message_start` used to encode the placeholder
/// text (16 tokens for one image on gemma-3-4b-it-4bit, against 275 in
/// `usage`). It now waits for the worker's prefill report, which carries the
/// expanded length, so all three figures agree.
#[tokio::test]
async fn anthropic_streaming_image_input_tokens_equal_count_and_usage() {
    let (app, options_rx) = image_app();
    let body = anthropic_image_body();

    let usage = post(&app, "/v1/messages", &body).await["usage"]["input_tokens"]
        .as_u64()
        .expect("usage.input_tokens");
    let _ = dispatched(&options_rx);
    assert!(usage > TOKENS_PER_MEDIA as u64, "{usage}");

    let counted = count(&post(&app, "/v1/messages/count_tokens", &body).await);
    assert!(dispatched(&options_rx).count_prompt_only);

    let streamed = streamed_input_tokens(&app, &body).await;
    assert!(!dispatched(&options_rx).count_prompt_only);

    assert_eq!(counted, usage, "count_tokens equals usage");
    assert_eq!(streamed, usage, "message_start equals usage");
}

/// The text-only half of #2337: `message_start` goes out before the worker
/// answers, from the same local count `count_tokens` takes, and it agrees
/// with `usage` too.
#[tokio::test]
async fn anthropic_streaming_text_input_tokens_equal_count_and_usage() {
    let (app, options_rx) = image_app();
    let body = serde_json::json!({
        "model": "prompt-count-test-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hello there"}]
    });

    let usage = post(&app, "/v1/messages", &body).await["usage"]["input_tokens"]
        .as_u64()
        .expect("usage.input_tokens");
    let _ = dispatched(&options_rx);
    let counted = count(&post(&app, "/v1/messages/count_tokens", &body).await);
    assert!(
        options_rx.try_recv().is_err(),
        "a text-only count stays local"
    );
    let streamed = streamed_input_tokens(&app, &body).await;
    let _ = dispatched(&options_rx);

    assert!(usage > 0, "{usage}");
    assert_eq!(counted, usage);
    assert_eq!(streamed, usage);
}

// ── reasoning re-echo (issue #2337, item 2) ─────────────────────────────────

/// Renders an assistant turn's reasoning when the request (or the #2110
/// re-echo) supplies one, so a filled turn renders longer than the body as
/// sent.
const ECHO_TEMPLATE: &str = "{% for m in messages %}<|{{ m['role'] }}|>\
{% if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}\
{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

/// Turn 1's generation: a reasoning block, then the answer.
const TURN1_TOKENS: &[&str] = &["<think>", "the capital", "</think>", "\n\n", "Paris."];

struct EchoHarness {
    app: Router,
    handle: crate::server::model_provider::ScriptedStreamHandle,
    prompts: mpsc::Receiver<String>,
    _options: mpsc::Receiver<ServerGenerateOptions>,
}

/// A scripted model behind the given route surface with the prompt cache on,
/// which is what enables the reasoning re-echo store (#2110), and a
/// byte-per-token tokenizer so a count is the rendered prompt's length.
fn echo_harness() -> EchoHarness {
    let (options_tx, options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(ECHO_TEMPLATE.to_string()),
        MlxcelTokenizer::stub_all_byte_fallback(),
        PathBuf::from("prompt-count-test-model"),
        batch_metrics,
    )
    .with_prompt_cache(Some(Arc::new(
        crate::server::prompt_cache::PromptCacheStore::new(),
    )));
    EchoHarness {
        app: create_app(state),
        handle,
        prompts: prompt_rx,
        _options: options_rx,
    }
}

impl EchoHarness {
    /// Generate `body` on `path` as a scripted reply of `tokens`; return the
    /// prompt the model was handed.
    async fn generate(&self, path: &str, body: &serde_json::Value, tokens: &[&str]) -> String {
        for token in tokens {
            self.handle.token(token);
        }
        self.handle.finish();
        let _ = post(&self.app, path, body).await;
        self.prompts.recv().expect("generation dispatched")
    }
}

/// Count the rendered prompt by the rule the generation tokenizes with.
fn prefilled_tokens(prompt: &str) -> u64 {
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    tokenizer
        .encode(prompt, !tokenizer.prompt_carries_bos(prompt))
        .expect("prompt encodes")
        .len() as u64
}

/// Prime the store with turn 1 on `generate_path`, then hold `count_path`'s
/// answer for the content-only follow-up to the length of the prompt the
/// follow-up generation is prefilled with.
async fn assert_count_includes_reasoning_echo(
    generate_path: &str,
    count_path: &str,
    turn1: serde_json::Value,
    turn2: serde_json::Value,
) {
    let harness = echo_harness();
    harness.generate(generate_path, &turn1, TURN1_TOKENS).await;

    let counted = count(&post(&harness.app, count_path, &turn2).await);
    let prompt2 = harness.generate(generate_path, &turn2, &["ok"]).await;
    assert!(
        prompt2.contains("<think>the capital</think>Paris."),
        "{generate_path}: the generation re-injected turn 1's reasoning: {prompt2:?}"
    );
    assert_eq!(
        counted,
        prefilled_tokens(&prompt2),
        "{count_path} counts the prompt {generate_path} prefills: {prompt2:?}"
    );
}

#[tokio::test]
async fn chat_count_includes_the_reasoning_echo() {
    assert_count_includes_reasoning_echo(
        "/v1/chat/completions",
        "/v1/chat/completions/input_tokens",
        serde_json::json!({
            "model": "prompt-count-test-model",
            "messages": [{"role": "user", "content": "q1"}]
        }),
        serde_json::json!({
            "model": "prompt-count-test-model",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "q2"}
            ]
        }),
    )
    .await;
}

#[tokio::test]
async fn responses_count_includes_the_reasoning_echo() {
    assert_count_includes_reasoning_echo(
        "/v1/responses",
        "/v1/responses/input_tokens",
        serde_json::json!({
            "model": "prompt-count-test-model",
            "store": false,
            "input": [{"type": "message", "role": "user", "content": "q1"}]
        }),
        serde_json::json!({
            "model": "prompt-count-test-model",
            "store": false,
            "input": [
                {"type": "message", "role": "user", "content": "q1"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Paris."}
                ]},
                {"type": "message", "role": "user", "content": "q2"}
            ]
        }),
    )
    .await;
}

#[tokio::test]
async fn anthropic_count_includes_the_reasoning_echo() {
    // The Anthropic route records only a `thinking` block the client was
    // sent, so both turns ask for extended thinking.
    let thinking = serde_json::json!({"type": "enabled", "budget_tokens": 256});
    assert_count_includes_reasoning_echo(
        "/v1/messages",
        "/v1/messages/count_tokens",
        serde_json::json!({
            "model": "prompt-count-test-model",
            "max_tokens": 512,
            "thinking": thinking,
            "messages": [{"role": "user", "content": "q1"}]
        }),
        serde_json::json!({
            "model": "prompt-count-test-model",
            "max_tokens": 512,
            "thinking": thinking,
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "q2"}
            ]
        }),
    )
    .await;
}

/// A backend that streams a token without a prefill report first (the XLA
/// serve loop, the disaggregated coordinator) still gets `message_start` as
/// the stream's first event, exactly once, carrying the local fallback count
/// (issue #2337).
#[tokio::test]
async fn anthropic_streaming_media_without_a_prefill_report_still_opens_with_message_start() {
    let (options_tx, _options_rx) = mpsc::channel();
    let (provider, handle) = ModelProvider::scripted_streaming_for_route_tests(options_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let app = create_app(
        AppState::new(
            provider,
            ServerConfig::default(),
            ChatTemplateProcessor::with_template(TEMPLATE.to_string()),
            MlxcelTokenizer::stub_all_byte_fallback(),
            PathBuf::from("prompt-count-test-model"),
            batch_metrics,
        )
        .with_media_support(ModelMediaSupport {
            image: true,
            ..ModelMediaSupport::default()
        }),
    );
    handle.token("hi");
    handle.finish();
    let streamed = streamed_input_tokens(&app, &anthropic_image_body()).await;
    assert!(streamed > 0, "the local count stands in: {streamed}");
}
