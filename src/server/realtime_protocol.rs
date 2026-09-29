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

//! Wire types of the `/v1/realtime` VoiceChat WebSocket (issue #1376).
//!
//! Port of the message handling in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/server/realtime.py:
//! JSON text frames in both directions, base64 little-endian PCM16 audio,
//! `event_<16 hex>` event ids and `sess_<16 hex>` session ids.
//!
//! Used by: [`crate::server::realtime_engine`] (event serialization on the
//! worker thread), `server/routes/realtime.rs` (the socket loop), the
//! `voicechat_file_client` example.

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::models::nemotron_voicechat::VoiceChatEvent;

/// Samples per native frame (80 ms at 16 kHz); the slice size before a
/// session reports its own.
pub const DEFAULT_FRAME_SAMPLES: usize = 1280;
/// Input rate the protocol advertises before configuration.
pub const INPUT_SAMPLE_RATE: u32 = 16_000;
/// Output rate the protocol advertises before configuration.
pub const OUTPUT_SAMPLE_RATE: u32 = 22_050;
/// WebSocket close code sent with `server_busy` ("Try Again Later").
pub const CLOSE_CODE_TRY_AGAIN_LATER: u16 = 1013;

/// `error.code` values.
pub const CODE_INVALID_REQUEST: &str = "invalid_request";
pub const CODE_SERVER_BUSY: &str = "server_busy";
pub const CODE_SESSION_INITIALIZATION_FAILED: &str = "session_initialization_failed";
pub const CODE_INFERENCE_ERROR: &str = "inference_error";

/// `{type: "pcm16", sample_rate}` of `input_audio_format` / `output_audio_format`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFormat {
    #[serde(rename = "type")]
    pub kind: String,
    pub sample_rate: u32,
}

impl AudioFormat {
    pub fn pcm16(sample_rate: u32) -> Self {
        Self {
            kind: "pcm16".to_string(),
            sample_rate,
        }
    }
}

/// The `session` object of `session.created` / `session.updated`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionObject {
    pub id: String,
    /// `configuring` before `session.update`, `ready` after.
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_samples: Option<usize>,
    pub input_audio_format: AudioFormat,
    pub output_audio_format: AudioFormat,
}

/// `{code, message}` of an `error` event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

/// One server -> client event, without its `event_id` (added by
/// [`to_wire_json`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerEvent {
    #[serde(rename = "session.created")]
    SessionCreated { session: SessionObject },
    #[serde(rename = "session.updated")]
    SessionUpdated { session: SessionObject },
    #[serde(rename = "session.pong")]
    SessionPong,
    #[serde(rename = "input_audio_buffer.committed")]
    InputAudioBufferCommitted,
    #[serde(rename = "error")]
    Error { error: ErrorBody },
    #[serde(rename = "response.text.delta")]
    TextDelta {
        frame_index: u64,
        token_id: i32,
        delta: String,
        text: String,
    },
    #[serde(rename = "response.function.delta")]
    FunctionDelta {
        frame_index: u64,
        token_id: i32,
        delta: String,
        text: String,
    },
    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    TranscriptDelta {
        frame_index: u64,
        delta: String,
        transcript: String,
    },
    #[serde(rename = "response.audio.delta")]
    AudioDelta {
        frame_index: u64,
        /// Base64 little-endian PCM16 of `round(clip(x, -1, 1) * 32767)`.
        delta: String,
        format: String,
        sample_rate: u32,
        channels: u32,
        audio_codes: Vec<i32>,
    },
    #[serde(rename = "response.done")]
    Done { frame_index: u64 },
    #[serde(rename = "response.cancelled")]
    Cancelled { frame_index: u64 },
}

impl ServerEvent {
    /// An `error` event.
    pub fn error(code: &str, message: impl Into<String>) -> Self {
        Self::Error {
            error: ErrorBody {
                code: code.to_string(),
                message: message.into(),
            },
        }
    }
}

/// Map one session event to its wire event (upstream `_serialize_model_event`).
pub fn serialize_event(event: VoiceChatEvent) -> ServerEvent {
    match event {
        VoiceChatEvent::AssistantTextDelta {
            frame_index,
            token_id,
            delta,
            text,
        } => ServerEvent::TextDelta {
            frame_index,
            token_id,
            delta,
            text,
        },
        VoiceChatEvent::FunctionDelta {
            frame_index,
            token_id,
            delta,
            text,
        } => ServerEvent::FunctionDelta {
            frame_index,
            token_id,
            delta,
            text,
        },
        VoiceChatEvent::UserTranscriptDelta {
            frame_index,
            delta,
            text,
        } => ServerEvent::TranscriptDelta {
            frame_index,
            delta,
            transcript: text,
        },
        VoiceChatEvent::Audio {
            frame_index,
            samples,
            sample_rate,
            audio_codes,
        } => ServerEvent::AudioDelta {
            frame_index,
            delta: audio_to_base64(&samples),
            format: "pcm16".to_string(),
            sample_rate,
            channels: 1,
            audio_codes,
        },
        VoiceChatEvent::Done { frame_index } => ServerEvent::Done { frame_index },
        VoiceChatEvent::Cancelled { frame_index } => ServerEvent::Cancelled { frame_index },
    }
}

/// Serialize `event` as one JSON text frame with a fresh `event_id`.
pub fn to_wire_json(event: &ServerEvent) -> String {
    let mut value = serde_json::to_value(event).unwrap_or_else(|err| {
        serde_json::json!({
            "type": "error",
            "error": {"code": CODE_INFERENCE_ERROR, "message": err.to_string()},
        })
    });
    if let Value::Object(map) = &mut value {
        map.insert("event_id".to_string(), Value::String(new_event_id()));
    }
    value.to_string()
}

/// `<prefix>_<16 lowercase hex>` from a v4 UUID, as upstream's
/// `uuid.uuid4().hex[:16]`.
fn short_id(prefix: &str) -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    format!("{prefix}_{}", &hex[..16])
}

/// A fresh `event_<16 hex>`.
pub fn new_event_id() -> String {
    short_id("event")
}

/// A fresh `sess_<16 hex>`.
pub fn new_session_id() -> String {
    short_id("sess")
}

/// Decode base64 little-endian PCM16 to f32 in `[-1, 1)` (`/ 32768`).
/// Rejects invalid base64 and an odd byte count.
pub fn pcm16_from_base64(value: &str) -> Result<Vec<f32>, String> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|_| "audio must be valid base64 PCM16".to_string())?;
    if raw.len() % 2 != 0 {
        return Err("PCM16 audio must contain an even number of bytes".to_string());
    }
    Ok(raw
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0)
        .collect())
}

/// `round(clip(x, -1, 1) * 32767)` as little-endian PCM16 bytes.
pub fn pcm16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        // NaN clamps to NaN and casts to 0, matching numpy's behavior closely
        // enough for a value the session never emits.
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Base64 of [`pcm16_bytes`].
pub fn audio_to_base64(samples: &[f32]) -> String {
    base64::engine::general_purpose::STANDARD.encode(pcm16_bytes(samples))
}

/// A parsed client message.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientMessage {
    /// `session.update`; `session` is the raw `session` object (fields are
    /// validated when the session is opened, so a bad value is a
    /// `session_initialization_failed`).
    SessionUpdate { session: Value },
    /// `input_audio_buffer.append` with its raw fields.
    Append {
        audio: Option<Value>,
        sample_rate: Option<Value>,
    },
    /// `input_audio_buffer.commit`.
    Commit { pad_partial: bool },
    /// `session.cancel` or `response.cancel`.
    Cancel,
    /// `session.ping`.
    Ping,
    /// Any other `type` (an absent `type` is the empty string).
    Other(String),
}

/// Parse one text frame. `Err` means it is not a JSON object.
pub fn parse_client_message(text: &str) -> Result<ClientMessage, String> {
    const NOT_OBJECT: &str = "message must be a JSON object";
    let value: Value = serde_json::from_str(text).map_err(|_| NOT_OBJECT.to_string())?;
    let Value::Object(mut map) = value else {
        return Err(NOT_OBJECT.to_string());
    };
    let kind = map
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok(match kind.as_str() {
        "session.update" => ClientMessage::SessionUpdate {
            session: map.remove("session").unwrap_or(Value::Null),
        },
        "input_audio_buffer.append" => ClientMessage::Append {
            audio: map.remove("audio"),
            sample_rate: map.remove("sample_rate"),
        },
        "input_audio_buffer.commit" => ClientMessage::Commit {
            // Upstream `bool(message.get("pad_partial", True))`.
            pad_partial: map.get("pad_partial").is_none_or(truthy),
        },
        "session.cancel" | "response.cancel" => ClientMessage::Cancel,
        "session.ping" => ClientMessage::Ping,
        _ => ClientMessage::Other(kind),
    })
}

/// Python truthiness of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Session options requested by `session.update`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionRequest {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub seed: u64,
    pub max_streaming_seconds: Option<f32>,
}

/// Validate the `session` object of `session.update`.
pub fn parse_session_request(session: &Value) -> Result<SessionRequest, String> {
    let get = |key: &str| session.get(key).filter(|v| !v.is_null());
    let model = match get("model") {
        None => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err("session.model must be a string".to_string()),
    };
    let system_prompt = match get("system_prompt") {
        None => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err("session.system_prompt must be a string".to_string()),
    };
    let seed = match get("seed") {
        None => 0,
        Some(value) => value
            .as_u64()
            .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
            .ok_or_else(|| "session.seed must be a non-negative integer".to_string())?,
    };
    let max_streaming_seconds = match get("max_streaming_seconds") {
        None => None,
        Some(value) => Some(
            value
                .as_f64()
                .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
                .ok_or_else(|| "session.max_streaming_seconds must be a number".to_string())?
                as f32,
        ),
    };
    Ok(SessionRequest {
        model,
        system_prompt,
        seed,
        max_streaming_seconds,
    })
}

/// Decode the `audio` and `sample_rate` fields of an append (absent audio is
/// empty, absent rate is 16000).
pub fn parse_append(
    audio: Option<&Value>,
    sample_rate: Option<&Value>,
) -> Result<(Vec<f32>, u32), String> {
    let samples = match audio {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => pcm16_from_base64(s)?,
        Some(_) => return Err("audio must be valid base64 PCM16".to_string()),
    };
    let rate = match sample_rate {
        None | Some(Value::Null) => INPUT_SAMPLE_RATE,
        Some(value) => value
            .as_u64()
            .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
            .and_then(|r| u32::try_from(r).ok())
            .ok_or_else(|| "sample_rate must be an integer".to_string())?,
    };
    Ok((samples, rate))
}

#[cfg(test)]
#[path = "realtime_protocol_tests.rs"]
mod tests;
