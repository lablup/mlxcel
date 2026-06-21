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

//! Audio-model provider interface.
//!
//! This is the transport-agnostic seam between the HTTP audio routes and the
//! speech models. The server surface (routes, request parsing, binary
//! responses) depends only on the types defined here, so a speech-to-text
//! provider and a text-to-speech provider can be wired in later without
//! reworking the route layer.
//!
//! The two directions are intentionally split into separate methods with
//! default "kind not loaded" implementations: a provider that only does one
//! direction overrides just that method and reports its capability through
//! [`AudioModelProvider::supports`]. Until a provider is registered the
//! [`AppState`](crate::server::AppState) slot stays `None` and every audio
//! route returns a structured `501 Not Implemented`.

use thiserror::Error;

/// The two audio-model directions the OpenAI audio API exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AudioModelKind {
    /// Speech-to-text: audio bytes in, text out
    /// (`/audio/transcriptions`, `/audio/translations`).
    Stt,
    /// Text-to-speech: text in, waveform out (`/audio/speech`).
    Tts,
}

impl AudioModelKind {
    /// Stable lowercase identifier, used in error messages and logs.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioModelKind::Stt => "stt",
            AudioModelKind::Tts => "tts",
        }
    }
}

impl std::fmt::Display for AudioModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Decoded request for the speech-to-text direction.
///
/// The audio arrives as the raw uploaded bytes (any container the provider
/// understands, typically WAV); the provider owns decoding via the shared
/// reader in [`crate::audio`].
#[derive(Debug, Clone)]
pub struct AudioTranscribeInput {
    /// Raw bytes of the uploaded audio file.
    pub audio: Vec<u8>,
    /// Original upload filename, when the client supplied one.
    pub filename: Option<String>,
    /// Optional ISO-639-1 source-language hint.
    pub language: Option<String>,
    /// Optional sampling temperature forwarded to the model.
    pub temperature: Option<f32>,
    /// When `true`, translate the source audio to English
    /// (`/audio/translations`) instead of transcribing in place
    /// (`/audio/transcriptions`).
    pub translate: bool,
}

/// Result of a speech-to-text request.
#[derive(Debug, Clone)]
pub struct AudioTranscribeOutput {
    /// Recognized text.
    pub text: String,
    /// Detected or supplied language, surfaced in verbose responses.
    pub language: Option<String>,
    /// Source audio duration in seconds, surfaced in verbose responses.
    pub duration_seconds: Option<f32>,
}

/// Decoded request for the text-to-speech direction.
#[derive(Debug, Clone)]
pub struct AudioSynthesizeInput {
    /// Text to synthesize.
    pub input: String,
    /// Optional named voice.
    pub voice: Option<String>,
    /// Optional playback-speed multiplier.
    pub speed: Option<f32>,
}

/// Result of a text-to-speech request: an `f32` PCM waveform the route encodes
/// into the client's requested container (WAV today).
#[derive(Debug, Clone)]
pub struct AudioSynthesizeOutput {
    /// Interleaved PCM samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count (`samples` is interleaved when this is `> 1`).
    pub channels: u16,
}

/// Failure modes shared by both audio directions.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AudioModelError {
    /// No model is loaded for the requested direction. Routes map this to a
    /// `501 Not Implemented` response.
    #[error("audio model kind not loaded: {0}")]
    KindNotLoaded(AudioModelKind),
    /// The loaded model failed while processing the request.
    #[error("audio model inference failed: {0}")]
    Inference(String),
    /// The bounded audio worker queue was full at admission time. Routes map
    /// this to the shared `503` "all slots are busy" envelope so a burst sheds
    /// load instead of growing memory without bound.
    #[error("audio worker queue is full")]
    QueueFull,
    /// The worker did not reply within the per-request timeout. Routes map this
    /// to a `504 Gateway Timeout`: the upstream worker did not answer in time.
    /// The in-flight MLX work is not cancelled; only the caller's blocking
    /// thread is freed.
    #[error("audio request timed out")]
    Timeout,
}

/// Provider-facing interface implemented by the speech models.
///
/// A provider advertises the directions it can service through
/// [`supports`](AudioModelProvider::supports) and overrides only the matching
/// method. The unimplemented direction keeps the default body, which reports
/// [`AudioModelError::KindNotLoaded`].
pub trait AudioModelProvider: Send + Sync {
    /// Whether this provider can service the given direction.
    fn supports(&self, kind: AudioModelKind) -> bool;

    /// Run speech-to-text. Default: the direction is not loaded.
    fn transcribe(
        &self,
        _input: AudioTranscribeInput,
    ) -> Result<AudioTranscribeOutput, AudioModelError> {
        Err(AudioModelError::KindNotLoaded(AudioModelKind::Stt))
    }

    /// Run text-to-speech. Default: the direction is not loaded.
    fn synthesize(
        &self,
        _input: AudioSynthesizeInput,
    ) -> Result<AudioSynthesizeOutput, AudioModelError> {
        Err(AudioModelError::KindNotLoaded(AudioModelKind::Tts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_str_and_display_match() {
        assert_eq!(AudioModelKind::Stt.as_str(), "stt");
        assert_eq!(AudioModelKind::Tts.as_str(), "tts");
        assert_eq!(AudioModelKind::Tts.to_string(), "tts");
    }

    #[test]
    fn kind_not_loaded_error_names_the_kind() {
        let err = AudioModelError::KindNotLoaded(AudioModelKind::Stt);
        assert!(err.to_string().contains("audio model kind not loaded"));
        assert!(err.to_string().contains("stt"));
    }

    /// A provider that only implements one direction keeps the default
    /// "not loaded" body for the other, proving the two directions plug in
    /// independently.
    struct TtsOnly;
    impl AudioModelProvider for TtsOnly {
        fn supports(&self, kind: AudioModelKind) -> bool {
            kind == AudioModelKind::Tts
        }
        fn synthesize(
            &self,
            input: AudioSynthesizeInput,
        ) -> Result<AudioSynthesizeOutput, AudioModelError> {
            Ok(AudioSynthesizeOutput {
                samples: vec![0.0; input.input.len()],
                sample_rate: 24_000,
                channels: 1,
            })
        }
    }

    #[test]
    fn default_transcribe_reports_kind_not_loaded() {
        let provider = TtsOnly;
        assert!(provider.supports(AudioModelKind::Tts));
        assert!(!provider.supports(AudioModelKind::Stt));

        let synth = provider
            .synthesize(AudioSynthesizeInput {
                input: "abc".to_string(),
                voice: None,
                speed: None,
            })
            .expect("tts-only provider synthesizes");
        assert_eq!(synth.samples.len(), 3);

        let err = provider
            .transcribe(AudioTranscribeInput {
                audio: vec![],
                filename: None,
                language: None,
                temperature: None,
                translate: false,
            })
            .expect_err("tts-only provider cannot transcribe");
        assert!(matches!(
            err,
            AudioModelError::KindNotLoaded(AudioModelKind::Stt)
        ));
    }
}
