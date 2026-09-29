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

//! Session backend of the realtime engine (issue #1376): the two traits the
//! engine thread drives, and their implementation for Nemotron VoiceChat.
//!
//! The traits exist so the engine and the `/v1/realtime` socket loop can be
//! tested with a fake model and no checkpoint. Neither trait requires `Send`:
//! the model is loaded on the engine thread and never leaves it, and a
//! session borrows the model for its lifetime, exactly like
//! [`VoiceChatStreamingSession`].
//!
//! Used by: [`crate::server::realtime_engine`]

use crate::models::NemotronVoiceChatModel;
use crate::models::nemotron_voicechat::streaming::VoiceChatError;
use crate::models::nemotron_voicechat::{
    StreamingOptions, VoiceChatEvent, VoiceChatStreamingSession,
};

/// What `session.update` asks for.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RealtimeSessionConfig {
    /// `None` uses the checkpoint default; an empty string means none.
    pub system_prompt: Option<String>,
    pub seed: u64,
    pub max_streaming_seconds: Option<f32>,
}

/// Audio geometry of an opened session, reported in `session.updated`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealtimeSessionInfo {
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    /// Samples per native frame; `input_audio_buffer.append` is pushed in
    /// slices of this size.
    pub frame_samples: usize,
}

/// A loaded model that can open streaming sessions. Lives on the engine
/// thread.
pub trait RealtimeModel {
    /// Open a new session borrowing the model.
    fn open_session(
        &self,
        config: &RealtimeSessionConfig,
    ) -> Result<Box<dyn RealtimeSession + '_>, VoiceChatError>;
}

/// One streaming timeline. Mirrors [`VoiceChatStreamingSession`]'s
/// `push_audio` / `flush` / `cancel`.
pub trait RealtimeSession {
    fn info(&self) -> RealtimeSessionInfo;
    fn push_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<VoiceChatEvent>, VoiceChatError>;
    fn flush(&mut self, pad_partial: bool) -> Result<Vec<VoiceChatEvent>, VoiceChatError>;
    fn cancel(&mut self) -> Vec<VoiceChatEvent>;
    /// Whether `flush` or `cancel` already closed the session.
    fn is_closed(&self) -> bool;
}

/// [`VoiceChatStreamingSession`] plus the audio geometry of its model.
struct VoiceChatRealtimeSession<'m> {
    inner: VoiceChatStreamingSession<'m>,
    info: RealtimeSessionInfo,
}

impl RealtimeModel for NemotronVoiceChatModel {
    fn open_session(
        &self,
        config: &RealtimeSessionConfig,
    ) -> Result<Box<dyn RealtimeSession + '_>, VoiceChatError> {
        let options = StreamingOptions {
            system_prompt: config.system_prompt.clone(),
            seed: config.seed,
            max_streaming_seconds: config.max_streaming_seconds,
            ..StreamingOptions::default()
        };
        let inner = self.create_streaming_session(options)?;
        let info = RealtimeSessionInfo {
            input_sample_rate: self.config().input_sample_rate,
            output_sample_rate: self.config().output_sample_rate,
            frame_samples: inner.frame_samples(),
        };
        Ok(Box::new(VoiceChatRealtimeSession { inner, info }))
    }
}

impl RealtimeSession for VoiceChatRealtimeSession<'_> {
    fn info(&self) -> RealtimeSessionInfo {
        self.info
    }

    fn push_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<VoiceChatEvent>, VoiceChatError> {
        self.inner.push_audio(samples, sample_rate)
    }

    fn flush(&mut self, pad_partial: bool) -> Result<Vec<VoiceChatEvent>, VoiceChatError> {
        self.inner.flush(pad_partial)
    }

    fn cancel(&mut self) -> Vec<VoiceChatEvent> {
        self.inner.cancel()
    }

    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}
