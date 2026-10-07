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

//! The transcript's `messages` form (issue #2173).

use super::*;

fn turn(role: &str, content: &str, images: Vec<PathBuf>) -> Turn {
    Turn {
        message: ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        },
        images,
    }
}

#[test]
fn text_turns_are_plain_string_messages() {
    let transcript = vec![
        turn("user", "hi", Vec::new()),
        turn("assistant", "hello", Vec::new()),
    ];
    assert_eq!(
        messages_json(&transcript).expect("json"),
        json!([
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": "hello" },
        ])
    );
}

#[test]
fn image_turns_carry_data_uri_parts_before_the_text() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("pixel.jpg");
    std::fs::write(&path, [0xffu8, 0xd8, 0xff]).expect("write");
    let messages = messages_json(&[turn("user", "what is this?", vec![path])]).expect("json");
    let parts = messages[0]["content"].as_array().expect("parts");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "image_url");
    assert_eq!(parts[0]["image_url"]["url"], "data:image/jpeg;base64,/9j/");
    assert_eq!(parts[1], json!({ "type": "text", "text": "what is this?" }));
    let request = mlxcel::server::in_process::chat::chat_request_from_json(json!({
        "model": "m",
        "messages": messages,
    }))
    .expect("the server parses the parts");
    assert_eq!(request.image_urls().len(), 1);
}

#[test]
fn a_missing_image_is_an_error_naming_the_file() {
    let err = image_data_uri(Path::new("/nonexistent/x.png")).expect_err("missing");
    assert!(err.to_string().contains("/nonexistent/x.png"), "{err}");
}
