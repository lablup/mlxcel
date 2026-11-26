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

//! Tencent Hy4 reasoning split (#2354).
//!
//! Every structural tag of the Hy4 chat template carries an `:opensource`
//! suffix, so its think pair is `<think:opensource>` / `</think:opensource>`.
//! The template primes the generation prompt with the open tag, and on the
//! real checkpoint the model emitted the open tag again before its reasoning:
//! with no marker table knowing the pair, the first server smoke returned
//! `content: "<think:opensource>We need to respond ..."` and no
//! `reasoning_content`.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer};
use tower::ServiceExt;

use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::model_provider::ModelProvider;
use crate::server::routes::chat::{
    extract_reasoning_content, is_prompt_primed_open_thinking, primed_open_thinking_close_marker,
};
use crate::server::thinking_budget::{ThinkingBudget, ThinkingState, resolve_thinking_token_ids};
use crate::server::tool_calls::clean_structural_tokens;
use crate::server::tool_calls::stream_filter::StreamFilter;
use crate::server::{AppState, ServerConfig, create_app};
use crate::tokenizer::{MlxcelTokenizer, ThinkingMarkers};

const OPEN: &str = "<think:opensource>";
const CLOSE: &str = "</think:opensource>";

/// The Hy4 checkpoint's think markers, as non-special added tokens (the
/// shipped `tokenizer.json` marks both `special: false`).
fn hy4_tokenizer() -> MlxcelTokenizer {
    let mut hf = Tokenizer::new(BPE::default());
    let added: Vec<AddedToken> = [OPEN, CLOSE]
        .iter()
        .map(|s| AddedToken::from(*s, false))
        .collect();
    hf.add_tokens(added).expect("added tokens register");
    MlxcelTokenizer::HuggingFace(hf)
}

fn hy4_markers() -> ThinkingMarkers {
    hy4_tokenizer().infer_thinking_markers()
}

/// The generation-prompt branch of the Hy4 `chat_template.jinja` (the
/// default, thinking-on arm) over a minimal message loop.
const HY4_TEMPLATE: &str = concat!(
    "{%- for message in messages -%}",
    "{{- '<｜hy_start:opensource｜>' + message.role + '<｜hy_middle:opensource｜>' + message.content + '<｜hy_end:opensource｜>' -}}",
    "{%- endfor -%}",
    "{%- if add_generation_prompt -%}",
    "{{- '<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource>' }}",
    "{%- endif -%}",
);

/// The outputs a primed Hy4 generation can take: the reasoning and the close
/// alone, or with the open tag emitted again first, as the real checkpoint did.
fn generations() -> [(&'static str, String); 2] {
    [
        ("close only", format!("abc{CLOSE}answer")),
        ("open repeated", format!("{OPEN}abc{CLOSE}answer")),
    ]
}

#[test]
fn the_tokenizer_resolves_the_hy4_pair() {
    let markers = hy4_markers();
    assert_eq!(markers.think_start.as_deref(), Some(OPEN));
    assert_eq!(markers.think_end.as_deref(), Some(CLOSE));
    assert_eq!(markers.primeable_pairs(), vec![(OPEN, CLOSE)]);
}

#[test]
fn the_generation_prompt_is_primed_and_closes_with_the_hy4_marker() {
    let markers = hy4_markers();
    let prompt = format!("<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜>{OPEN}");
    assert!(is_prompt_primed_open_thinking(&markers, &prompt));
    assert_eq!(
        primed_open_thinking_close_marker(&markers, &prompt).as_deref(),
        Some(CLOSE)
    );
    // The template's `no_think` arm closes the block in the prompt.
    assert!(!is_prompt_primed_open_thinking(
        &markers,
        &format!("{prompt}{CLOSE}")
    ));
}

#[test]
fn non_streaming_split_moves_reasoning_out_of_content() {
    for (case, raw) in generations() {
        assert_eq!(clean_structural_tokens(&raw), "answer", "{case}");
        assert_eq!(
            extract_reasoning_content(&raw, true)
                .as_deref()
                .map(str::trim),
            Some("abc"),
            "{case}"
        );
    }
}

#[test]
fn streaming_split_moves_reasoning_out_of_content() {
    for (case, raw) in generations() {
        // One char at a time, so each marker straddles every fragment boundary.
        let mut filter = StreamFilter::new_primed_open_thinking();
        let (mut content, mut reasoning) = (String::new(), String::new());
        let mut take = |out: crate::server::tool_calls::stream_filter::FilterOutput| {
            content.push_str(out.content.as_deref().unwrap_or(""));
            reasoning.push_str(out.reasoning.as_deref().unwrap_or(""));
        };
        for ch in raw.chars() {
            take(filter.feed(&ch.to_string()));
        }
        take(filter.flush());
        assert_eq!(reasoning.trim(), "abc", "{case}");
        assert_eq!(content.trim(), "answer", "{case}");
        assert!(!content.contains("opensource"), "{case}: {content:?}");
        assert!(!reasoning.contains("opensource"), "{case}: {reasoning:?}");
    }
}

#[test]
fn the_thinking_budget_resolves_the_hy4_ids() {
    let tokenizer = hy4_tokenizer();
    let hf = tokenizer.hf_tokenizer().expect("HF backend");
    let ids = resolve_thinking_token_ids(&tokenizer).expect("Hy4 ids resolve");
    assert_eq!(ids.open as u32, hf.token_to_id(OPEN).unwrap());
    assert_eq!(ids.close as u32, hf.token_to_id(CLOSE).unwrap());
    let budget = ThinkingBudget::Limited(std::num::NonZeroU32::new(64).unwrap());
    let mut state = ThinkingState::new(Some(ids), Some(budget), true);
    state.observe(7);
    state.observe(ids.close);
    assert_eq!(
        state.phase,
        crate::server::thinking_budget::ThinkingPhase::Closed
    );
}

/// The real `/v1/chat/completions` handler against a scripted model.
struct Harness {
    app: axum::Router,
    handle: crate::server::model_provider::ScriptedStreamHandle,
    prompts: mpsc::Receiver<String>,
}

fn harness() -> Harness {
    let (options_tx, _options_rx) = mpsc::channel();
    let (prompt_tx, prompt_rx) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let provider = Arc::new(provider);
    let batch_metrics = provider.batch_metrics().clone();
    let mut state = AppState::new(
        provider,
        ServerConfig::default(),
        ChatTemplateProcessor::with_template(HY4_TEMPLATE.to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    );
    state.thinking_markers = Arc::new(hy4_markers());
    Harness {
        app: create_app(state),
        handle,
        prompts: prompt_rx,
    }
}

impl Harness {
    async fn chat(&self, stream: bool, tokens: &[&str]) -> (String, String) {
        for token in tokens {
            self.handle.token(token);
        }
        self.handle.finish();
        let body = json!({
            "model": "route-test-model",
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
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
        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body collects")
            .to_bytes();
        let text = String::from_utf8_lossy(&bytes).into_owned();
        assert_eq!(status, StatusCode::OK, "{text}");
        let prompt = self.prompts.recv().expect("generation dispatched");
        (prompt, text)
    }
}

/// `(content, reasoning_content)` concatenated over an SSE body's deltas.
fn streamed_fields(body: &str) -> (String, String) {
    let (mut content, mut reasoning) = (String::new(), String::new());
    for event in body
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<Value>(data.trim()).ok())
    {
        let delta = &event["choices"][0]["delta"];
        content.push_str(delta["content"].as_str().unwrap_or(""));
        reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or(""));
    }
    (content, reasoning)
}

fn fields(stream: bool, body: &str) -> (String, String) {
    if stream {
        return streamed_fields(body);
    }
    let v: Value = serde_json::from_str(body).expect("JSON body");
    let message = &v["choices"][0]["message"];
    let field = |name: &str| message[name].as_str().unwrap_or("").to_string();
    (field("content"), field("reasoning_content"))
}

#[tokio::test]
async fn chat_completions_split_hy4_reasoning_streaming_and_not() {
    let scripts: [(&str, &[&str]); 2] = [
        ("close only", &["abc", CLOSE, "answer"]),
        ("open repeated", &[OPEN, "abc", CLOSE, "answer"]),
    ];
    for (name, tokens) in scripts {
        for stream in [false, true] {
            let h = harness();
            let (prompt, body) = h.chat(stream, tokens).await;
            assert!(prompt.ends_with(OPEN), "{prompt:?}");
            let (content, reasoning) = fields(stream, &body);
            let case = format!("{name}, stream {stream}: {body}");
            assert_eq!(reasoning.trim(), "abc", "{case}");
            assert_eq!(content.trim(), "answer", "{case}");
            assert!(!content.contains("opensource"), "{case}");
        }
    }
}

/// The shape of the first real server smoke: the model repeated the open tag
/// and ran out of tokens inside the block. Nothing belongs in `content`.
#[tokio::test]
async fn a_truncated_hy4_reasoning_block_stays_out_of_content() {
    for stream in [false, true] {
        let h = harness();
        let (_, body) = h.chat(stream, &[OPEN, "We need to respond"]).await;
        let (content, reasoning) = fields(stream, &body);
        let case = format!("stream {stream}: {body}");
        assert_eq!(reasoning.trim(), "We need to respond", "{case}");
        assert_eq!(content.trim(), "", "{case}");
    }
}
