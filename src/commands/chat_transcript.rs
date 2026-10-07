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

//! The chat transcript and its `/v1/chat/completions` `messages` form, shared
//! by the chat REPL and `mlxcel run -p` (issue #2173).

use std::path::Path;

use anyhow::{Context, Result};
use base64::Engine as _;
use mlxcel::server::chat_template::ChatMessage;
use serde_json::{Value, json};

/// One message of the transcript, with the images it carries.
#[derive(Debug, Clone)]
pub(crate) struct Turn {
    pub(crate) message: ChatMessage,
    /// The message's images as `data:` URIs ([`image_data_uri`]). Each file
    /// is read once, when it is attached, so the transcript re-sent on every
    /// turn never touches the disk again and a file moved or deleted after
    /// its turn cannot break the later turns.
    pub(crate) images: Vec<String>,
}

/// The transcript's messages without their attachments.
pub(crate) fn transcript_messages(transcript: &[Turn]) -> Vec<ChatMessage> {
    transcript.iter().map(|turn| turn.message.clone()).collect()
}

/// The transcript as a `messages` array. A message with images carries them
/// as `image_url` parts with inline `data:` URIs, which every media-capable
/// route accepts without a `--media-path` root. `image_soft_tokens`
/// (`--image-soft-tokens`) rides on every part as the server's
/// `max_soft_tokens` extension, which the server validates against the Gemma 4
/// budget ladder.
pub(crate) fn messages_json(transcript: &[Turn], image_soft_tokens: Option<usize>) -> Value {
    let mut messages = Vec::with_capacity(transcript.len());
    for turn in transcript {
        let content = if turn.images.is_empty() {
            json!(turn.message.content)
        } else {
            let mut parts = Vec::with_capacity(turn.images.len() + 1);
            for uri in &turn.images {
                let mut image_url = json!({ "url": uri });
                if let Some(budget) = image_soft_tokens {
                    image_url["max_soft_tokens"] = json!(budget);
                }
                parts.push(json!({ "type": "image_url", "image_url": image_url }));
            }
            parts.push(json!({ "type": "text", "text": turn.message.content }));
            Value::Array(parts)
        };
        messages.push(json!({ "role": turn.message.role, "content": content }));
    }
    Value::Array(messages)
}

/// How many images the transcript carries. Every turn re-sends the whole
/// transcript, so this is the image count of the next request before the
/// message being composed.
pub(crate) fn transcript_image_count(transcript: &[Turn]) -> usize {
    transcript.iter().map(|turn| turn.images.len()).sum()
}

/// The server's per-request image cap, read from the server itself
/// ([`mlxcel::current_image_input_limits`]) so a configured override and the
/// default both apply and no number is duplicated here.
pub(crate) fn server_image_cap() -> usize {
    mlxcel::current_image_input_limits().max_images_per_request
}

/// Refuse an attachment that would push a request over the server's image
/// cap. `already` is the images the request carries without this attachment
/// (the transcript plus the images waiting for the next message) and `adding`
/// the images being attached. The REPL re-sends the whole transcript, so once
/// the conversation is over the cap every later turn would fail until
/// `/clear`; refusing at attach time keeps the conversation usable.
pub(crate) fn check_image_budget(already: usize, adding: usize, cap: usize) -> Result<(), String> {
    if already + adding <= cap {
        return Ok(());
    }
    Err(format!(
        "the server accepts at most {cap} images per request and each turn re-sends the whole conversation, \
         which already holds {already}; /clear starts a new conversation"
    ))
}

/// Read each image file once into a `data:` URI, naming the file that fails.
pub(crate) fn image_data_uris<P: AsRef<Path>>(paths: &[P]) -> Result<Vec<String>> {
    paths
        .iter()
        .map(|path| image_data_uri(path.as_ref()))
        .collect()
}

/// A `data:` URI for an image file, typed by its extension.
pub(crate) fn image_data_uri(path: &Path) -> Result<String> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read image {}", path.display()))?;
    let mime = match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        Some("bmp") => "image/bmp",
        _ => "image/png",
    };
    Ok(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

#[cfg(test)]
#[path = "chat_transcript_tests.rs"]
mod tests;
