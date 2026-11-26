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

//! Request-to-model reasoning effort mapping for a steering model (#3435).

use serde_json::{Value, json};

use super::resolve_model_reasoning_effort;
use crate::models::reasoning_effort_steering::ReasoningEffortSteering;
use crate::server::chat_request::prepare_chat_request_with_cache;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::server::prompt_cache::key::fold_model_reasoning_effort;
use crate::server::types::request::ChatCompletionRequest;

/// Reads only `enable_thinking`, as Raon-1 mini's template does.
const TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n\
{% endfor %}<|im_start|>assistant\n{% if enable_thinking is defined and not enable_thinking %}\
<think></think>{% endif %}";

fn steering() -> ReasoningEffortSteering {
    ReasoningEffortSteering::from_config_value(&json!({
        "reasoning_effort_steering": {
            "alpha": {"none": 0, "low": 0.3, "medium": 0.1, "high": 0.05, "xhigh": 0.02, "max": 0},
            "default": "high",
            "layers": [12, 14, 16, 18, 20]
        }
    }))
    .unwrap()
    .unwrap()
}

fn processor(steered: bool) -> ChatTemplateProcessor {
    let mut processor = ChatTemplateProcessor::with_template(TEMPLATE.to_string());
    // Raon's tokenizer has `<think>`, so its processor thinks by default.
    processor.set_default_enable_thinking(true);
    processor.set_reasoning_effort_steering(steered.then(steering));
    processor
}

fn request(extra: Value) -> ChatCompletionRequest {
    let mut body = json!({
        "model": "raon",
        "messages": [{"role": "user", "content": "What is 17 + 25?"}],
    });
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    serde_json::from_value(body).unwrap()
}

fn resolve(extra: Value) -> Result<Option<String>, String> {
    resolve_with_default(extra, None)
}

fn resolve_with_default(
    extra: Value,
    server_default: Option<&ChatTemplateKwargs>,
) -> Result<Option<String>, String> {
    let request = request(extra);
    resolve_model_reasoning_effort(
        &processor(true),
        &request,
        server_default,
        &request.merged_extra_body(),
    )
}

#[test]
fn every_request_channel_reaches_the_model() {
    assert_eq!(
        resolve(json!({"reasoning_effort": "low"})),
        Ok(Some("low".into()))
    );
    assert_eq!(
        resolve(json!({"chat_template_kwargs": {"reasoning_effort": "xhigh"}})),
        Ok(Some("xhigh".into()))
    );
    assert_eq!(
        resolve(json!({"extra_body": {"reasoning_effort": "medium"}})),
        Ok(Some("medium".into()))
    );
    assert_eq!(
        resolve(json!({"reasoning": {"effort": "MAX"}})),
        Ok(Some("max".into()))
    );
    // The explicit template kwarg wins over the portable field.
    assert_eq!(
        resolve(json!({
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "max"}
        })),
        Ok(Some("max".into()))
    );
}

#[test]
fn the_responses_api_reasoning_effort_reaches_the_model() {
    let body = json!({
        "model": "raon",
        "input": "What is 17 + 25?",
        "reasoning": {"effort": "low"},
    });
    let responses: crate::server::types::responses_request::CreateResponseRequest =
        serde_json::from_value(body).unwrap();
    let translated =
        crate::server::responses_translator::responses_request_to_chat(&responses, None, None)
            .unwrap();
    let chat = translated.chat_request;
    assert_eq!(
        resolve_model_reasoning_effort(&processor(true), &chat, None, &chat.merged_extra_body()),
        Ok(Some("low".into()))
    );
}

#[test]
fn no_level_is_the_config_default_and_an_unsteered_model_gets_none() {
    assert_eq!(resolve(json!({})), Ok(Some("high".into())));
    let request = request(json!({"reasoning_effort": "minimal"}));
    assert_eq!(
        resolve_model_reasoning_effort(
            &processor(false),
            &request,
            None,
            &request.merged_extra_body()
        ),
        Ok(None),
        "a model without steering ignores the level (the template decides)"
    );
}

#[test]
fn an_unsupported_level_is_refused_with_the_accepted_set() {
    for bad in [
        json!({"reasoning_effort": "minimal"}),
        json!({"reasoning_effort": "off"}),
        json!({"chat_template_kwargs": {"reasoning_effort": 0.5}}),
    ] {
        let err = resolve(bad.clone()).unwrap_err();
        assert!(
            err.contains("none, low, medium, high, xhigh, max"),
            "{bad}: {err}"
        );
    }
}

#[test]
fn thinking_off_alone_selects_none() {
    assert_eq!(
        resolve(json!({"chat_template_kwargs": {"enable_thinking": false}})),
        Ok(Some("none".into()))
    );
    assert_eq!(
        resolve(json!({"reasoning": false})),
        Ok(Some("none".into()))
    );
    let server_off = ChatTemplateKwargs::from_json_object(
        json!({"enable_thinking": false})
            .as_object()
            .unwrap()
            .clone(),
    );
    assert_eq!(
        resolve_with_default(json!({}), Some(&server_off)),
        Ok(Some("none".into()))
    );
    // The request's own thinking-on beats the server default.
    assert_eq!(
        resolve_with_default(
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            Some(&server_off)
        ),
        Ok(Some("high".into()))
    );
    // So does a portable `reasoning: true`, which the render maps onto
    // `enable_thinking: true` over the server default.
    assert_eq!(
        resolve_with_default(json!({"reasoning": true}), Some(&server_off)),
        Ok(Some("high".into()))
    );
}

#[test]
fn the_server_default_level_applies_only_when_the_request_sets_none() {
    let server_low = ChatTemplateKwargs::from_json_object(
        json!({"reasoning_effort": "low"})
            .as_object()
            .unwrap()
            .clone(),
    );
    assert_eq!(
        resolve_with_default(json!({}), Some(&server_low)),
        Ok(Some("low".into()))
    );
    assert_eq!(
        resolve_with_default(json!({"reasoning_effort": "max"}), Some(&server_low)),
        Ok(Some("max".into()))
    );
}

#[test]
fn a_request_turning_thinking_off_beats_the_server_default_level() {
    let server_low = ChatTemplateKwargs::from_json_object(
        json!({"reasoning_effort": "low"})
            .as_object()
            .unwrap()
            .clone(),
    );
    for thinking_off in [
        json!({"chat_template_kwargs": {"enable_thinking": false}}),
        json!({"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}),
        json!({"reasoning": false}),
    ] {
        assert_eq!(
            resolve_with_default(thinking_off.clone(), Some(&server_low)),
            Ok(Some("none".into())),
            "{thinking_off}"
        );
    }
    // An explicit per-request level still wins over the request's own
    // thinking-off and over the server default.
    assert_eq!(
        resolve_with_default(
            json!({
                "reasoning_effort": "medium",
                "chat_template_kwargs": {"enable_thinking": false}
            }),
            Some(&server_low)
        ),
        Ok(Some("medium".into()))
    );
    assert_eq!(
        resolve_with_default(
            json!({"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "xhigh"}}),
            Some(&server_low)
        ),
        Ok(Some("xhigh".into()))
    );
    // A request that leaves thinking on keeps the server level.
    assert_eq!(
        resolve_with_default(
            json!({"chat_template_kwargs": {"enable_thinking": true}}),
            Some(&server_low)
        ),
        Ok(Some("low".into()))
    );
}

#[tokio::test]
async fn a_thinking_off_request_renders_and_steers_the_same_way_over_a_server_level() {
    let server_low = ChatTemplateKwargs::from_json_object(
        json!({"reasoning_effort": "low"})
            .as_object()
            .unwrap()
            .clone(),
    );
    let prepared = prepare_chat_request_with_cache(
        &processor(true),
        &request(json!({"chat_template_kwargs": {"enable_thinking": false}})),
        Some(&server_low),
        false,
        false,
        false,
        &crate::tokenizer::ThinkingMarkers::default(),
    )
    .await
    .unwrap();
    assert!(
        prepared.prompt.trim_end().ends_with("<think></think>"),
        "{}",
        prepared.prompt
    );
    assert_eq!(prepared.model_reasoning_effort.as_deref(), Some("none"));
}

async fn render(extra: Value) -> anyhow::Result<crate::server::chat_request::PreparedChatRequest> {
    prepare_chat_request_with_cache(
        &processor(true),
        &request(extra),
        None,
        false,
        false,
        false,
        &crate::tokenizer::ThinkingMarkers::default(),
    )
    .await
}

#[tokio::test]
async fn a_portable_none_renders_thinking_off_and_reaches_the_model() {
    let prepared = render(json!({"reasoning_effort": "none"})).await.unwrap();
    assert!(
        prepared.prompt.trim_end().ends_with("<think></think>"),
        "{}",
        prepared.prompt
    );
    assert_eq!(prepared.model_reasoning_effort.as_deref(), Some("none"));

    // A template-kwarg `none` changes only the alpha: thinking stays on, as
    // the model card documents.
    let prepared = render(json!({"chat_template_kwargs": {"reasoning_effort": "none"}}))
        .await
        .unwrap();
    assert!(
        !prepared.prompt.contains("<think></think>"),
        "{}",
        prepared.prompt
    );
    assert_eq!(prepared.model_reasoning_effort.as_deref(), Some("none"));

    // Any other level keeps thinking on.
    let prepared = render(json!({"reasoning_effort": "low"})).await.unwrap();
    assert!(!prepared.prompt.contains("<think></think>"));
    assert_eq!(prepared.model_reasoning_effort.as_deref(), Some("low"));
}

#[tokio::test]
async fn request_preparation_refuses_minimal_before_rendering() {
    let err = render(json!({"reasoning_effort": "minimal"}))
        .await
        .err()
        .expect("minimal is refused");
    let message = err.to_string();
    assert!(
        message.contains("reasoning_effort must be one of none, low, medium, high, xhigh, max"),
        "{message}"
    );
}

#[test]
fn the_cache_signature_separates_efforts_and_leaves_other_models_alone() {
    let base = "template-sig".to_string();
    assert_eq!(fold_model_reasoning_effort(base.clone(), None), base);
    let high = fold_model_reasoning_effort(base.clone(), Some("high"));
    let low = fold_model_reasoning_effort(base.clone(), Some("low"));
    let max = fold_model_reasoning_effort(base.clone(), Some("max"));
    assert_ne!(high, base);
    assert_ne!(high, low);
    assert_ne!(high, max);
    assert_ne!(low, max);
    assert_eq!(high, fold_model_reasoning_effort(base, Some("high")));
}
