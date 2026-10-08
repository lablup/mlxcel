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

//! Model-free request routing regressions for typed-only checkpoint templates.
//!
//! Embedded data URI resolves local bytes only, as in the existing request tests.

use std::path::PathBuf;

use serde_json::{Value, json};

use super::prepare_chat_request_with_cache;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::types::ChatCompletionRequest;
use crate::tokenizer::ThinkingMarkers;

const JSON_OBSERVER: &str = concat!(
    "{% for m in messages %}",
    "{% if m.content is not string %}{{ m | tojson }}{{ '\\n' }}{% endif %}",
    "{% endfor %}",
    "{% if add_generation_prompt %}GENERATE{% endif %}",
);

fn checkpoint(template: &str) -> ChatTemplateProcessor {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"chat_template": template}).to_string(),
    )
    .unwrap();
    ChatTemplateProcessor::from_model_path(dir.path())
        .unwrap()
        .unwrap()
}

fn observed_messages(rendered: &str) -> Vec<Value> {
    rendered
        .strip_suffix("GENERATE")
        .unwrap_or(rendered)
        .lines()
        .map(|line| serde_json::from_str(line).expect("template, not fallback, must render"))
        .collect()
}

fn request(messages: Value) -> ChatCompletionRequest {
    serde_json::from_value(json!({"model": "test", "messages": messages})).unwrap()
}

#[tokio::test]
async fn image_routing_keeps_system_assistant_and_plain_followup_text() {
    let processor = ChatTemplateProcessor::from_model_path(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/granite_vision_3_2"),
    )
    .unwrap()
    .unwrap();
    let req = request(json!([
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": [
            {"type": "text", "text": "IMAGE_QUESTION"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ]},
        {"role": "assistant", "content": "ANSWER"},
        {"role": "user", "content": "FOLLOWUP"},
    ]));
    let prepared = prepare_chat_request_with_cache(
        &processor,
        &req,
        None,
        true,
        true,
        false,
        &ThinkingMarkers::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        prepared.prompt,
        concat!(
            "<|system|>\nSYSTEM\n",
            "<|user|>\n<image>\nIMAGE_QUESTION\n",
            "<|assistant|>\nANSWER<|end_of_text|>",
            "<|user|>\nFOLLOWUP\n<|assistant|>\n",
        )
    );
    assert_eq!(prepared.image_data, vec![b"hello".to_vec()]);
    assert!(
        prepared.history_prompt.is_none(),
        "media history snapshot stays disabled"
    );
}

#[tokio::test]
async fn reasoning_routing_preserves_strings_and_both_reasoning_spellings() {
    let processor = checkpoint(JSON_OBSERVER);
    let req = request(json!([
        {"role": "user", "content": "QUESTION"},
        {"role": "assistant", "content": "ANSWER", "reasoning_content": "TRACE"},
        {"role": "user", "content": "FOLLOWUP"},
    ]));
    let prepared = prepare_chat_request_with_cache(
        &processor,
        &req,
        None,
        true,
        true,
        false,
        &ThinkingMarkers::default(),
    )
    .await
    .unwrap();
    let history = prepared
        .history_prompt
        .as_ref()
        .expect("raw text history boundary");
    assert_eq!(prepared.prompt, format!("{history}GENERATE"));
    let rows = observed_messages(&prepared.prompt);
    assert_eq!(
        rows,
        vec![
            json!({"role": "user", "content": [{"type": "text", "text": "QUESTION"}]}),
            json!({
                "role": "assistant", "content": [{"type": "text", "text": "ANSWER"}],
                "reasoning": "TRACE", "reasoning_content": "TRACE",
            }),
            json!({"role": "user", "content": [{"type": "text", "text": "FOLLOWUP"}]}),
        ]
    );
}

#[tokio::test]
async fn raw_normalization_does_not_reintroduce_stripped_old_thinking() {
    let processor = checkpoint(JSON_OBSERVER);
    let mut req = request(json!([
        {"role": "user", "content": "QUESTION"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<think>PRIVATE_INLINE</think>ANSWER"}],
            "reasoning_content": "PRIVATE_PARALLEL",
        },
        {"role": "user", "content": "FOLLOWUP"},
    ]));
    req.chat_template_kwargs =
        Some(serde_json::from_value(json!({"preserve_thinking": false})).unwrap());
    let prepared = prepare_chat_request_with_cache(
        &processor,
        &req,
        None,
        true,
        true,
        false,
        &ThinkingMarkers::default(),
    )
    .await
    .unwrap();
    let rows = observed_messages(&prepared.prompt);
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows[1],
        json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "ANSWER"}],
        })
    );
    assert!(!prepared.prompt.contains("PRIVATE_"));
    assert!(!prepared.history_prompt.unwrap().contains("PRIVATE_"));
}

#[tokio::test]
async fn tool_routing_preserves_call_ids_names_arguments_and_empty_content() {
    let processor = checkpoint(JSON_OBSERVER);
    let req = request(json!([
        {"role": "user", "content": "QUESTION"},
        {
            "role": "assistant",
            "content": null,
            "name": "planner",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"Seoul\"}"},
            }],
        },
        {
            "role": "tool",
            "content": "SUNNY",
            "tool_call_id": "call-1",
            "name": "weather",
        },
        {"role": "user", "content": "FOLLOWUP"},
    ]));
    let prepared = prepare_chat_request_with_cache(
        &processor,
        &req,
        None,
        true,
        true,
        false,
        &ThinkingMarkers::default(),
    )
    .await
    .unwrap();
    let rows = observed_messages(&prepared.prompt);
    assert_eq!(rows.len(), 4);
    assert_eq!(
        rows[0]["content"],
        json!([{"type": "text", "text": "QUESTION"}])
    );
    assert_eq!(
        rows[1],
        json!({
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "name": "planner",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "weather", "arguments": {"city": "Seoul"}},
            }],
        })
    );
    assert_eq!(
        rows[2],
        json!({
            "role": "tool",
            "content": [{"type": "text", "text": "SUNNY"}],
            "tool_call_id": "call-1",
            "name": "weather",
        })
    );
    assert_eq!(
        rows[3]["content"],
        json!([{"type": "text", "text": "FOLLOWUP"}])
    );
    let history = prepared
        .history_prompt
        .expect("tool-history boundary retained");
    assert_eq!(prepared.prompt, format!("{history}GENERATE"));
}
