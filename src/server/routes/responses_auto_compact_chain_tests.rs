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

use super::{CONTEXT, OLD, SUMMARY, Script, body, harness, history, sse_events};

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
