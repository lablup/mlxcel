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

//! Route-level reasoning re-injection (issue #2110): the real
//! `/v1/chat/completions` handler, streaming and non-streaming, against a
//! scripted model and the Jamba-Reasoning template.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use super::JAMBA_TEMPLATE;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::model_provider::ModelProvider;
use crate::server::prompt_cache::PromptCacheStore;
use crate::server::{AppState, ServerConfig, create_app};
use crate::tokenizer::MlxcelTokenizer;

struct RouteHarness {
    app: axum::Router,
    handle: crate::server::model_provider::ScriptedStreamHandle,
    prompts: mpsc::Receiver<String>,
    _options: mpsc::Receiver<crate::server::ServerGenerateOptions>,
}

fn route_harness(prompt_cache: bool) -> RouteHarness {
    let (options_tx, options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(JAMBA_TEMPLATE.to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    )
    .with_prompt_cache(prompt_cache.then(|| Arc::new(PromptCacheStore::new())));
    RouteHarness {
        app: create_app(state),
        handle,
        prompts: prompt_rx,
        _options: options_rx,
    }
}

impl RouteHarness {
    /// Post one chat turn whose generation is `tokens`; return the rendered
    /// prompt the model saw and the response body.
    async fn turn(
        &self,
        messages: serde_json::Value,
        stream: bool,
        tokens: &[&str],
    ) -> (String, String) {
        for token in tokens {
            self.handle.token(token);
        }
        self.handle.finish();
        let body = serde_json::json!({
            "model": "route-test-model",
            "stream": stream,
            "messages": messages,
        });
        let response = self
            .app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .expect("request builds"),
            )
            .await
            .expect("route responds");
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body collects")
            .to_bytes();
        let prompt = self.prompts.recv().expect("generation dispatched");
        (prompt, String::from_utf8_lossy(&bytes).into_owned())
    }
}

const TURN1_TOKENS: &[&str] = &["<think>", "the capital", "</think>", "\n\n", "Paris."];

/// The issue's failure on the real route: a content-only turn 2 must keep the
/// thinking instruction on the earlier user turn, so turn 1's history boundary
/// (`user\nTHINK_PREFIX\nq1`) still prefixes it. Fails without re-injection.
#[tokio::test]
async fn content_only_follow_up_keeps_the_generated_rendering_on_both_paths() {
    for stream in [false, true] {
        let harness = route_harness(true);
        let (prompt1, body1) = harness
            .turn(
                serde_json::json!([{"role": "user", "content": "q1"}]),
                stream,
                TURN1_TOKENS,
            )
            .await;
        assert!(prompt1.starts_with("<|im_start|>user\nTHINK_PREFIX\nq1"));
        assert!(body1.contains("the capital"), "reasoning surfaced: {body1}");

        let (prompt2, _) = harness
            .turn(
                serde_json::json!([
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "Paris."},
                    {"role": "user", "content": "q2"},
                ]),
                stream,
                &["ok"],
            )
            .await;
        assert!(
            prompt2.starts_with("<|im_start|>user\nTHINK_PREFIX\nq1<|im_end|>\n"),
            "stream={stream}: the earlier turn must render as generated: {prompt2:?}"
        );
    }
}

/// With the prompt cache off nothing is recorded or re-injected: the turn keeps
/// the template's content-only rendering.
#[tokio::test]
async fn prompt_cache_off_leaves_content_only_history_untouched() {
    let harness = route_harness(false);
    harness
        .turn(
            serde_json::json!([{"role": "user", "content": "q1"}]),
            false,
            TURN1_TOKENS,
        )
        .await;
    let (prompt2, _) = harness
        .turn(
            serde_json::json!([
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "q2"},
            ]),
            false,
            &["ok"],
        )
        .await;
    assert!(
        prompt2.starts_with("<|im_start|>user\nq1<|im_end|>\n"),
        "{prompt2:?}"
    );
}
