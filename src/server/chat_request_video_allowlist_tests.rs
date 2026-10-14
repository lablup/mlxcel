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

use std::fs;

use super::prepare_chat_request_with_cache_and_video_allowlist;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::types::request::VideoUrl;
use crate::server::types::{
    ChatCompletionRequest, ContentPart, Message, MessageContent, Role, SamplingParams,
};

fn video_request(path: &std::path::Path) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Parts(vec![ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: path.to_string_lossy().into_owned(),
                    fps: None,
                },
            }]),
            name: None,
            tool_call_id: None,
            reasoning: None,
            tool_calls: None,
        }],
        stream: false,
        stream_options: None,
        logprobs: None,
        top_logprobs: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        chat_template_kwargs: None,
        extra_body: None,
        prompt_cache_key: None,
        cache_prompt: None,
        user: None,
        reasoning_effort: None,
        extra_body_fields: serde_json::Map::new(),
        response_format: None,
        params: SamplingParams::default(),
    }
}

#[tokio::test]
async fn native_video_preparation_uses_startup_allowlist_snapshot() {
    let startup_dir =
        std::env::temp_dir().join(format!("mlxcel-video-startup-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&startup_dir).unwrap();
    let startup_dir = fs::canonicalize(startup_dir).unwrap();
    let startup_video = startup_dir.join("startup.mp4");
    fs::write(&startup_video, b"startup video").unwrap();
    let startup_allowlist = vec![startup_dir.clone()];

    let later_dir =
        std::env::temp_dir().join(format!("mlxcel-video-later-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&later_dir).unwrap();
    let later_video = later_dir.join("later.mp4");
    fs::write(&later_video, b"later video").unwrap();

    let processor = ChatTemplateProcessor::with_template("ok".to_string());
    let markers = crate::tokenizer::ThinkingMarkers::default();
    let prepared_at_startup = prepare_chat_request_with_cache_and_video_allowlist(
        &processor,
        &video_request(&startup_video),
        None,
        false,
        false,
        false,
        &markers,
        &startup_allowlist,
    )
    .await
    .unwrap();
    assert_eq!(prepared_at_startup.videos.len(), 1);
    drop(prepared_at_startup);

    let prepared_after_new_directory = prepare_chat_request_with_cache_and_video_allowlist(
        &processor,
        &video_request(&later_video),
        None,
        false,
        false,
        false,
        &markers,
        &startup_allowlist,
    )
    .await
    .unwrap();
    assert!(
        prepared_after_new_directory.videos.is_empty(),
        "a directory created after startup is not accepted until restart"
    );
    drop(prepared_after_new_directory);

    fs::remove_dir_all(startup_dir).unwrap();
    fs::remove_dir_all(later_dir).unwrap();
}
