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

//! Events, options, errors and the text-delta rule of the streaming
//! VoiceChat session.
//!
//! Port of `VoiceChatEvent` and `_TokenAccumulator` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/streaming.py.
//!
//! Used by: [`super::VoiceChatStreamingSession`], the `/v1/realtime` wire
//! mapping

use std::fmt;

/// One aligned output of a streaming session. `frame_index` counts 80 ms
/// audio frames (the system-prompt prefix does not advance it).
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceChatEvent {
    /// An assistant text token that changed the decoded text.
    AssistantTextDelta {
        frame_index: u64,
        token_id: i32,
        delta: String,
        /// Cumulative assistant text (authoritative).
        text: String,
    },
    /// A function-channel token that changed the decoded function text.
    FunctionDelta {
        frame_index: u64,
        token_id: i32,
        delta: String,
        text: String,
    },
    /// The RNNT transcript of the user audio grew.
    UserTranscriptDelta {
        frame_index: u64,
        delta: String,
        text: String,
    },
    /// 80 ms of assistant speech: exactly 1764 samples at 22.05 kHz.
    Audio {
        frame_index: u64,
        samples: Vec<f32>,
        sample_rate: u32,
        /// The 31 codec codes of this frame after control-code replacement.
        audio_codes: Vec<i32>,
    },
    /// `flush` finished the stream.
    Done { frame_index: u64 },
    /// `cancel` stopped the stream.
    Cancelled { frame_index: u64 },
}

impl VoiceChatEvent {
    /// The frame this event belongs to.
    pub fn frame_index(&self) -> u64 {
        match self {
            Self::AssistantTextDelta { frame_index, .. }
            | Self::FunctionDelta { frame_index, .. }
            | Self::UserTranscriptDelta { frame_index, .. }
            | Self::Audio { frame_index, .. }
            | Self::Done { frame_index }
            | Self::Cancelled { frame_index } => *frame_index,
        }
    }
}

/// Options of [`super::VoiceChatStreamingSession`].
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingOptions {
    /// System prompt; `None` uses the checkpoint default (empty on the
    /// released checkpoints), an empty string means none.
    pub system_prompt: Option<String>,
    /// Seed of MLX's global RNG (EAR-TTS sampling noise).
    pub seed: u64,
    /// Refuse audio past this many seconds (`ContextLimit`).
    pub max_streaming_seconds: Option<f32>,
    /// Keep persistent Nemotron-H caches (default). `false` recomputes the
    /// full input history every frame (diagnostic).
    pub use_language_cache: bool,
    /// Use the cache-aware frontend and encoder (default). `false`
    /// recomputes log-mel and the encoder over a sliding sample window.
    pub use_perception_cache: bool,
    /// Record per-frame stage timings.
    pub profile: bool,
    /// With `profile`, also time named sub-stages inside perception,
    /// language, EAR-TTS and the codec. This forces extra evaluations, so
    /// the frame totals of such a run are for attribution, not for the
    /// real-time factor.
    pub profile_stages: bool,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            system_prompt: None,
            seed: 0,
            max_streaming_seconds: None,
            use_language_cache: true,
            use_perception_cache: true,
            profile: false,
            profile_stages: false,
        }
    }
}

/// Errors of a streaming session.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum VoiceChatError {
    /// The stream exceeded `max_streaming_seconds`.
    ContextLimit { max_frames: u64 },
    /// `push_audio` on a flushed or cancelled session.
    Closed,
    /// Input that is not 16 kHz mono PCM, or an invalid option.
    InvalidInput(String),
    /// A model stage failed.
    Inference(String),
}

impl fmt::Display for VoiceChatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextLimit { max_frames } => {
                write!(
                    f,
                    "stream exceeded the configured {max_frames}-frame context"
                )
            }
            Self::Closed => write!(f, "streaming session is closed"),
            Self::InvalidInput(msg) | Self::Inference(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for VoiceChatError {}

impl From<String> for VoiceChatError {
    fn from(msg: String) -> Self {
        Self::Inference(msg)
    }
}

/// Cumulative text of one token channel and the delta rule.
///
/// `decode` maps the kept token ids to text. The delta is the new suffix
/// when the new cumulative decode extends the old one; otherwise (a
/// tokenizer revising a partial multibyte sequence) it is the single-token
/// decode, and the cumulative text stays authoritative.
#[derive(Debug, Clone, Default)]
pub struct TokenAccumulator {
    special_ids: Vec<i32>,
    tokens: Vec<i32>,
    text: String,
}

impl TokenAccumulator {
    pub fn new(special_ids: &[i32]) -> Self {
        Self {
            special_ids: special_ids.to_vec(),
            tokens: Vec::new(),
            text: String::new(),
        }
    }

    /// Cumulative text so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Kept token ids so far.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// Append `token_id`; `None` for special ids, else `(delta, text)`.
    pub fn append<F>(
        &mut self,
        token_id: i32,
        decode: F,
    ) -> Result<Option<(String, String)>, String>
    where
        F: Fn(&[i32]) -> Result<String, String>,
    {
        if self.special_ids.contains(&token_id) {
            return Ok(None);
        }
        self.tokens.push(token_id);
        let updated = decode(&self.tokens)?;
        let delta = match updated.strip_prefix(self.text.as_str()) {
            Some(rest) => rest.to_string(),
            None => decode(&[token_id])?,
        };
        self.text = updated.clone();
        Ok(Some((delta, updated)))
    }
}
