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

//! Automatic compaction on the `conversation` path, and the unchanged
//! streamed shape without `context_management` (issue #2397).

use axum::http::StatusCode;
use serde_json::json;

use super::{CONTEXT, OLD, SUMMARY, Script, body, harness, harness_with, history, sse_events};
use crate::server::ServerConfig;
use crate::server::config::ReasoningBudgetOverride;
use crate::server::thinking_budget::ThinkingBudget;

#[tokio::test]
async fn a_conversation_compacts_once_and_replays_the_summary() {
    let h = harness(CONTEXT);
    let mut first = body(history(), Some(200));
    first["conversation"] = json!("conv_auto");
    let (status, response) = h
        .post(&first, &[Script(&[SUMMARY], false), Script(&["ok"], false)])
        .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["output"][0]["type"], "compaction", "{response}");
    assert_eq!(
        h.drain_prompts().len(),
        2,
        "one summary pass, one generation"
    );

    let mut next = body(json!("next step"), Some(200));
    next["conversation"] = json!("conv_auto");
    let (status, second) = h.post(&next, &[Script(&["done"], false)]).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let prompts = h.drain_prompts();
    assert_eq!(prompts.len(), 1, "below threshold: no second summary pass");
    for kept in ["SYS", SUMMARY, "latest question", "ok", "next step"] {
        assert!(prompts[0].contains(kept), "{kept} missing: {}", prompts[0]);
    }
    assert!(!prompts[0].contains(OLD), "{}", prompts[0]);
    assert_eq!(
        prompts[0].matches("<|system|>SYS").count(),
        1,
        "{}",
        prompts[0]
    );
    assert_eq!(second["output"][0]["type"], "message", "{second}");
}

#[tokio::test]
async fn a_stream_without_context_management_keeps_its_shape() {
    let h = harness(CONTEXT);
    let mut request = body(history(), None);
    request["stream"] = json!(true);
    request["max_output_tokens"] = json!(64);
    let (status, text) = h.post_raw(&request, &[Script(&["an", "swer"], true)]).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let events = sse_events(&text);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ],
        "{text}"
    );
    let sequence: Vec<u64> = events
        .iter()
        .map(|(_, data)| data["sequence_number"].as_u64().expect("sequence_number"))
        .collect();
    assert_eq!(sequence, (0..events.len() as u64).collect::<Vec<_>>());
    assert!(
        events
            .iter()
            .filter_map(|(_, data)| data["output_index"].as_u64())
            .all(|index| index == 0)
    );
    let (_, done) = &events[8];
    assert_eq!(done["item"]["status"], "completed");

    // A `length` stop still reports `completed`, with no capsule or cap.
    let response = &events[9].1["response"];
    assert_eq!(response["status"], "completed");
    assert!(response["incomplete_details"].is_null());
    assert!(response["completed_at"].is_number());
    let output = response["output"].as_array().expect("output");
    assert_eq!(output.len(), 1, "{output:?}");
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["status"], "completed");
    assert_eq!(response["output_text"], "answer");
    let prompts = h.drain_prompts();
    assert_eq!(prompts.len(), 1, "no summary pass");
    assert!(prompts[0].contains(OLD));
    assert_eq!(h.options.try_recv().expect("options").max_tokens, 64);
}

#[tokio::test]
async fn the_reasoning_budget_stays_below_the_capped_output() {
    let context = 512;
    let h = harness(context);
    let mut request = body(json!("short question"), Some(10_000));
    request["max_output_tokens"] = json!(100_000);
    // Valid against the uncapped limit (the 512-token context), above the cap.
    request["thinking_budget_tokens"] = json!(500);
    let (status, response) = h.post(&request, &[Script(&["ok"], false)]).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let options = h.options.try_recv().expect("generation options");
    assert!(options.max_tokens < 500, "cap {}", options.max_tokens);
    let expected = u32::try_from(options.max_tokens - 1)
        .ok()
        .and_then(std::num::NonZeroU32::new)
        .map(ThinkingBudget::Limited);
    assert_eq!(
        options.reasoning_budget,
        ReasoningBudgetOverride::Explicit(expected),
        "budget leaves room for visible output under max_tokens {}",
        options.max_tokens
    );
}

#[tokio::test]
async fn a_busy_server_refuses_before_counting() {
    let h = harness_with(ServerConfig {
        context_size: CONTEXT,
        max_queue_depth: 0,
        ..ServerConfig::default()
    });
    for stream in [false, true] {
        let mut request = body(history(), Some(1));
        request["stream"] = json!(stream);
        let (status, response) = h.post(&request, &[]).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{response}");
        assert_eq!(response["error"]["type"], "server_busy", "{response}");
    }
    assert!(
        h.drain_prompts().is_empty(),
        "no summary pass, no generation"
    );
}
