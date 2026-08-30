// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

use std::sync::Arc;

use super::chat_request::request_has_effective_input;
use super::responses_store::{ResponsesStore, ResponsesStoreConfig, StoredResponse};
use super::responses_translator::{
    OutboundContext, build_response_object, responses_request_to_chat,
};
use super::types::request::{ContentPart, MessageContent, Role};
use super::types::responses_request::{
    CreateResponseRequest, INPUT_FILE_UNSUPPORTED, INPUT_IMAGE_FILE_ID_UNSUPPORTED,
};
use super::types::responses_response::{ResponseObject, ResponseStatus};

fn request(body: &str) -> CreateResponseRequest {
    serde_json::from_str(body).expect("request should deserialize")
}

fn translated(body: &str) -> super::types::request::ChatCompletionRequest {
    responses_request_to_chat(&request(body), None, None)
        .expect("request should translate")
        .chat_request
}

#[test]
fn input_text_and_input_image_parts_lower_to_chat_parts() {
    let chat = translated(
        r#"{
            "model":"m",
            "input":[{
                "type":"message",
                "role":"user",
                "content":[
                    {"type":"input_text","text":"describe"},
                    {"type":"input_image","image_url":"data:image/png;base64,aA==","detail":"high"}
                ]
            }]
        }"#,
    );

    assert_eq!(chat.messages.len(), 1);
    assert_eq!(chat.messages[0].role, Role::User);
    let MessageContent::Parts(parts) = &chat.messages[0].content else {
        panic!("expected content parts");
    };
    assert!(matches!(
        &parts[0],
        ContentPart::Text { text } if text == "describe"
    ));
    let ContentPart::ImageUrl { image_url } = &parts[1] else {
        panic!("expected image part");
    };
    assert_eq!(image_url.url, "data:image/png;base64,aA==");
    assert_eq!(image_url.detail.as_deref(), Some("high"));
    assert!(request_has_effective_input(&chat));
}

#[test]
fn input_image_file_id_only_is_rejected_with_named_error() {
    let request = request(
        r#"{
            "model":"m",
            "input":[{
                "type":"message",
                "role":"user",
                "content":[{"type":"input_image","file_id":"file_1"}]
            }]
        }"#,
    );
    let error = responses_request_to_chat(&request, None, None)
        .unwrap_err()
        .to_string();
    assert_eq!(error, INPUT_IMAGE_FILE_ID_UNSUPPORTED);
}

#[test]
fn input_file_part_is_rejected() {
    let request = request(
        r#"{
            "model":"m",
            "input":[{
                "type":"message",
                "role":"user",
                "content":[{"type":"input_file","file_id":"file_1"}]
            }]
        }"#,
    );
    let error = responses_request_to_chat(&request, None, None)
        .unwrap_err()
        .to_string();
    assert_eq!(error, INPUT_FILE_UNSUPPORTED);
}

#[test]
fn function_output_image_stays_after_tool_result() {
    let chat = translated(
        r#"{
            "model":"m",
            "input":[
                {"type":"function_call","call_id":"call_1","name":"screenshot","arguments":"{}"},
                {
                    "type":"function_call_output",
                    "call_id":"call_1",
                    "output":[
                        {"type":"input_text","text":"done"},
                        {"type":"input_image","image_url":"data:image/png;base64,aA=="}
                    ]
                }
            ]
        }"#,
    );

    assert_eq!(chat.messages.len(), 3);
    assert_eq!(chat.messages[0].role, Role::Assistant);
    assert!(chat.messages[0].tool_calls.is_some());
    assert_eq!(chat.messages[1].role, Role::Tool);
    assert_eq!(
        chat.messages[1].content.text(),
        "done\n[Image output attached in the next message]"
    );
    assert_eq!(chat.messages[2].role, Role::User);
    let MessageContent::Parts(parts) = &chat.messages[2].content else {
        panic!("expected image follow-up parts");
    };
    assert_eq!(parts.len(), 1);
    assert!(matches!(parts[0], ContentPart::ImageUrl { .. }));
    assert!(request_has_effective_input(&chat));
}

#[test]
fn function_output_preserves_image_order() {
    let chat = translated(
        r#"{
            "model":"m",
            "input":[{
                "type":"function_call_output",
                "call_id":"call_1",
                "output":[
                    {"type":"input_image","image_url":"data:image/png;base64,Zmlyc3Q="},
                    {"type":"input_image","image_url":"data:image/png;base64,c2Vjb25k"}
                ]
            }]
        }"#,
    );

    assert_eq!(
        chat.messages[1].content.image_urls(),
        vec![
            "data:image/png;base64,Zmlyc3Q=".to_string(),
            "data:image/png;base64,c2Vjb25k".to_string(),
        ]
    );
}

#[test]
fn function_output_preserves_text_alongside_visual_input() {
    let chat = translated(
        r#"{
            "model":"m",
            "input":[{
                "type":"function_call_output",
                "call_id":"call_1",
                "output":[
                    {"type":"input_text","text":"captured"},
                    {"type":"text","text":"red square"},
                    {"type":"input_image","image_url":"data:image/png;base64,aA=="}
                ]
            }]
        }"#,
    );

    assert_eq!(
        chat.messages[0].content.text(),
        "captured\nred square\n[Image output attached in the next message]"
    );
    assert_eq!(chat.messages[1].content.image_urls().len(), 1);
}

#[test]
fn function_output_string_form_is_unchanged() {
    let chat = translated(
        r#"{
            "model":"m",
            "input":[{
                "type":"function_call_output",
                "call_id":"call_1",
                "output":"plain result"
            }]
        }"#,
    );

    assert_eq!(chat.messages.len(), 1);
    assert_eq!(chat.messages[0].role, Role::Tool);
    assert_eq!(chat.messages[0].content.text(), "plain result");
}

#[test]
fn input_file_inside_function_output_is_rejected() {
    let request = request(
        r#"{
            "model":"m",
            "input":[{
                "type":"function_call_output",
                "call_id":"call_1",
                "output":[{"type":"input_file","file_id":"file_1"}]
            }]
        }"#,
    );
    let error = responses_request_to_chat(&request, None, None)
        .unwrap_err()
        .to_string();
    assert_eq!(error, INPUT_FILE_UNSUPPORTED);
}

#[test]
fn unknown_function_output_parts_remain_as_json_text() {
    let chat = translated(
        r#"{
            "model":"m",
            "input":[{
                "type":"function_call_output",
                "call_id":"call_1",
                "output":[
                    {"type":"input_text","text":"done"},
                    {"type":"foo","answer":42}
                ]
            }]
        }"#,
    );

    let tool_text = chat.messages[0].content.text();
    let (text, raw_json) = tool_text
        .split_once('\n')
        .expect("text and leftover JSON should be separate lines");
    assert_eq!(text, "done");
    let leftovers: serde_json::Value =
        serde_json::from_str(raw_json).expect("leftovers should be valid JSON");
    assert_eq!(leftovers, serde_json::json!([{"type":"foo","answer":42}]));
}

#[test]
fn known_non_image_tool_parts_preserve_original_json() {
    let chat = translated(
        r#"{
            "model":"m",
            "input":[{
                "type":"function_call_output",
                "call_id":"call_1",
                "output":[
                    {
                        "type":"input_audio",
                        "input_audio":{"data":"aGVsbG8="},
                        "extension":"keep-me"
                    },
                    {
                        "type":"video_url",
                        "video_url":{"url":"file:///tmp/clip.mp4"},
                        "vendor_option":7
                    }
                ]
            }]
        }"#,
    );

    let leftovers: serde_json::Value =
        serde_json::from_str(&chat.messages[0].content.text()).expect("valid leftover JSON");
    assert_eq!(
        leftovers,
        serde_json::json!([
            {
                "type":"input_audio",
                "input_audio":{"data":"aGVsbG8="},
                "extension":"keep-me"
            },
            {
                "type":"video_url",
                "video_url":{"url":"file:///tmp/clip.mp4"},
                "vendor_option":7
            }
        ])
    );
}

#[test]
fn assistant_message_image_moves_to_following_user_turn() {
    let chat = translated(
        r#"{
            "model":"m",
            "input":[{
                "type":"message",
                "role":"assistant",
                "content":[
                    {"type":"input_text","text":"visual result"},
                    {"type":"input_image","image_url":"data:image/png;base64,aA=="}
                ]
            }]
        }"#,
    );

    assert_eq!(chat.messages.len(), 2);
    assert_eq!(chat.messages[0].role, Role::Assistant);
    assert_eq!(chat.messages[0].content.text(), "visual result");
    assert!(chat.messages[0].content.image_urls().is_empty());
    assert_eq!(chat.messages[1].role, Role::User);
    assert_eq!(chat.messages[1].content.image_urls().len(), 1);
}

#[test]
fn non_user_message_preserves_video_and_audio_when_moving_images() {
    let chat = translated(
        r#"{
            "model":"m",
            "input":[{
                "type":"message",
                "role":"assistant",
                "content":[
                    {"type":"text","text":"mixed media"},
                    {"type":"video_url","video_url":{"url":"file:///tmp/clip.mp4","fps":1.0}},
                    {"type":"input_audio","input_audio":{"data":"aGVsbG8=","format":"wav"}},
                    {"type":"input_image","image_url":"data:image/png;base64,aA=="}
                ]
            }]
        }"#,
    );

    let MessageContent::Parts(retained) = &chat.messages[0].content else {
        panic!("non-image media should keep content in part form");
    };
    assert_eq!(retained.len(), 3);
    assert!(matches!(retained[0], ContentPart::Text { .. }));
    assert!(matches!(retained[1], ContentPart::VideoUrl { .. }));
    assert!(matches!(retained[2], ContentPart::InputAudio { .. }));
    assert!(chat.messages[0].content.image_urls().is_empty());
    assert_eq!(chat.messages[1].content.image_urls().len(), 1);
}

fn completed_response(request: &CreateResponseRequest) -> ResponseObject {
    build_response_object(OutboundContext {
        response_id: "resp_image_tool".to_string(),
        model_id: "m".to_string(),
        created_at: 1.0,
        completed_at: 2.0,
        status: ResponseStatus::Completed,
        prompt_tokens: 1,
        completion_tokens: 1,
        cached_tokens: 0,
        reasoning_tokens: 0,
        text: "stored answer".to_string(),
        reasoning_text: None,
        parsed_tool_calls: None,
        max_tool_calls: None,
        request,
        error: None,
        incomplete_reason: None,
        finish_reason: "stop".to_string(),
    })
}

#[test]
fn stored_response_with_image_tool_output_replays_identically() {
    let original = request(
        r#"{
            "model":"m",
            "input":[
                {"type":"function_call","call_id":"call_1","name":"screenshot","arguments":"{}"},
                {
                    "type":"function_call_output",
                    "call_id":"call_1",
                    "output":[
                        {"type":"input_text","text":"captured"},
                        {"type":"input_image","image_url":"data:image/png;base64,aA=="}
                    ]
                }
            ]
        }"#,
    );
    let store = Arc::new(ResponsesStore::new(ResponsesStoreConfig::default()));
    store.insert(
        "resp_image_tool".to_string(),
        StoredResponse {
            response: completed_response(&original),
            input_items: original.input.clone().into_items(),
        },
    );

    let next = request(
        r#"{
            "model":"m",
            "previous_response_id":"resp_image_tool",
            "input":"continue"
        }"#,
    );
    let replayed = responses_request_to_chat(&next, Some(&store), None)
        .expect("stored response should replay")
        .chat_request;

    assert_eq!(replayed.messages[1].role, Role::Tool);
    assert_eq!(
        replayed.messages[1].content.text(),
        "captured\n[Image output attached in the next message]"
    );
    assert_eq!(replayed.messages[2].role, Role::User);
    assert_eq!(replayed.messages[2].content.image_urls().len(), 1);
    assert_eq!(replayed.messages[3].role, Role::Assistant);
    assert_eq!(replayed.messages[3].content.text(), "stored answer");
    assert_eq!(replayed.messages[4].role, Role::User);
    assert_eq!(replayed.messages[4].content.text(), "continue");
}
