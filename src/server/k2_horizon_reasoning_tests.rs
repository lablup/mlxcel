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

//! IFM K2-Horizon reasoning split (#3299).
//!
//! The K2-Horizon template primes the generation prompt with `<ifm|think>`,
//! `<ifm|think_fast>` or `<ifm|think_faster>` (by `reasoning_effort`), so the
//! model emits only the reasoning and the matching close tag. Before #3299 no
//! marker table knew these pairs and the reasoning, close tag included, stayed
//! in `content`.

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

/// (`reasoning_effort`, primed open, close) for each template branch.
const EFFORTS: [(&str, &str, &str); 3] = [
    ("high", "<ifm|think>", "</ifm|think>"),
    ("medium", "<ifm|think_fast>", "</ifm|think_fast>"),
    ("low", "<ifm|think_faster>", "</ifm|think_faster>"),
];

/// The K2-Horizon checkpoint's think markers, as non-special added tokens
/// (the shipped `tokenizer.json` marks all six `special: false`).
fn k2_tokenizer() -> MlxcelTokenizer {
    let mut hf = Tokenizer::new(BPE::default());
    let added: Vec<AddedToken> = EFFORTS
        .iter()
        .flat_map(|(_, open, close)| [*open, *close])
        .map(|s| AddedToken::from(s, false))
        .collect();
    hf.add_tokens(added).expect("added tokens register");
    MlxcelTokenizer::HuggingFace(hf)
}

fn k2_markers() -> ThinkingMarkers {
    k2_tokenizer().infer_thinking_markers()
}

/// The generation-prompt branch of `models/k2-horizon-0.9b/chat_template.jinja`
/// over a minimal message loop.
const K2_TEMPLATE: &str = concat!(
    "{%- for message in messages -%}",
    "{{- '<|ifm|im_start|>' + message.role + '\\n' + message.content + '<|ifm|im_end|>\\n' -}}",
    "{%- endfor -%}",
    "{%- if add_generation_prompt -%}",
    "{%- set effort = reasoning_effort | default('high') -%}",
    "{%- if effort == 'high' -%}{{- '<|ifm|im_start|>assistant\\n<ifm|think>\\n' }}",
    "{%- elif effort == 'medium' -%}{{- '<|ifm|im_start|>assistant\\n<ifm|think_fast>\\n' }}",
    "{%- elif effort == 'low' -%}{{- '<|ifm|im_start|>assistant\\n<ifm|think_faster>\\n' }}",
    "{%- endif -%}",
    "{%- endif -%}",
);

fn generation(close: &str) -> String {
    format!("abc\n{close}\nanswer")
}

#[test]
fn the_tokenizer_resolves_the_k2_horizon_pair_and_every_effort_variant() {
    let markers = k2_markers();
    assert_eq!(markers.think_start.as_deref(), Some("<ifm|think>"));
    assert_eq!(markers.think_end.as_deref(), Some("</ifm|think>"));
    let pairs: Vec<(&str, &str)> = EFFORTS.iter().map(|(_, o, c)| (*o, *c)).collect();
    assert_eq!(markers.primeable_pairs(), pairs);

    // A family without variants keeps exactly its one pair.
    let qwen = ThinkingMarkers {
        think_start: Some("<think>".into()),
        think_end: Some("</think>".into()),
        ..ThinkingMarkers::default()
    };
    assert_eq!(qwen.primeable_pairs(), vec![("<think>", "</think>")]);
}

#[test]
fn each_effort_prompt_is_primed_and_closes_with_its_own_marker() {
    let markers = k2_markers();
    for (effort, open, close) in EFFORTS {
        let prompt = format!("<|ifm|im_start|>assistant\n{open}\n");
        assert!(
            is_prompt_primed_open_thinking(&markers, &prompt),
            "{effort}"
        );
        assert_eq!(
            primed_open_thinking_close_marker(&markers, &prompt).as_deref(),
            Some(close),
            "{effort}"
        );
        // A closed block at the end of the prompt is not primed.
        let closed = format!("{prompt}{close}\n");
        assert!(
            !is_prompt_primed_open_thinking(&markers, &closed),
            "{effort}"
        );
    }
}

#[test]
fn non_streaming_split_moves_reasoning_out_of_content_for_every_effort() {
    for (effort, _, close) in EFFORTS {
        let raw = generation(close);
        assert_eq!(clean_structural_tokens(&raw), "answer", "{effort}");
        assert_eq!(
            extract_reasoning_content(&raw, true)
                .as_deref()
                .map(str::trim),
            Some("abc"),
            "{effort}"
        );
    }
}

#[test]
fn streaming_split_moves_reasoning_out_of_content_for_every_effort() {
    for (effort, _, close) in EFFORTS {
        let raw = generation(close);
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
        assert_eq!(reasoning.trim(), "abc", "{effort}");
        assert_eq!(content.trim(), "answer", "{effort}");
        assert!(!content.contains("ifm|think"), "{effort}: {content:?}");
    }
}

#[test]
fn the_thinking_budget_sees_every_variant_close() {
    let tokenizer = k2_tokenizer();
    let hf = tokenizer.hf_tokenizer().expect("HF backend");
    let ids = resolve_thinking_token_ids(&tokenizer).expect("K2-Horizon ids resolve");
    assert_eq!(ids.open as u32, hf.token_to_id("<ifm|think>").unwrap());
    assert_eq!(ids.close as u32, hf.token_to_id("</ifm|think>").unwrap());
    for (effort, _, close) in EFFORTS {
        let close_id = hf.token_to_id(close).unwrap() as i32;
        assert!(ids.is_close(close_id), "{effort}");
        // A primed block with a budget closes on the variant's own close
        // instead of counting the answer as reasoning.
        let budget = ThinkingBudget::Limited(std::num::NonZeroU32::new(64).unwrap());
        let mut state = ThinkingState::new(Some(ids), Some(budget), true);
        state.observe(7);
        state.observe(close_id);
        assert_eq!(
            state.phase,
            crate::server::thinking_budget::ThinkingPhase::Closed,
            "{effort}"
        );
    }
    assert!(!ids.is_close(ids.open));
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
        ChatTemplateProcessor::with_template(K2_TEMPLATE.to_string()),
        MlxcelTokenizer::stub(),
        PathBuf::from("route-test-model"),
        batch_metrics,
    );
    state.thinking_markers = Arc::new(k2_markers());
    Harness {
        app: create_app(state),
        handle,
        prompts: prompt_rx,
    }
}

impl Harness {
    async fn chat(&self, stream: bool, effort: &str, tokens: &[&str]) -> (String, String) {
        for token in tokens {
            self.handle.token(token);
        }
        self.handle.finish();
        let body = json!({
            "model": "route-test-model",
            "stream": stream,
            "reasoning_effort": effort,
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

#[tokio::test]
async fn chat_completions_split_k2_horizon_reasoning_streaming_and_not() {
    for (effort, open, close) in EFFORTS {
        for stream in [false, true] {
            let h = harness();
            let tokens = ["abc", "\n", close, "\n", "answer"];
            let (prompt, body) = h.chat(stream, effort, &tokens).await;
            assert!(prompt.ends_with(&format!("{open}\n")), "{prompt:?}");
            let (content, reasoning) = if stream {
                streamed_fields(&body)
            } else {
                let v: Value = serde_json::from_str(&body).expect("JSON body");
                let message = &v["choices"][0]["message"];
                (
                    message["content"].as_str().unwrap_or("").to_string(),
                    message["reasoning_content"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                )
            };
            let case = format!("effort {effort}, stream {stream}: {body}");
            assert_eq!(reasoning.trim(), "abc", "{case}");
            assert_eq!(content.trim(), "answer", "{case}");
            assert!(!content.contains("ifm|think"), "{case}");
        }
    }
}
