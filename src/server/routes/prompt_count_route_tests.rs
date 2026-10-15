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
        MlxcelTokenizer::stub_with_byte_fallback(),
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
