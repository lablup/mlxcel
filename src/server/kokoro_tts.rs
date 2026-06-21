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

//! Text-to-speech [`AudioModelProvider`] backed by the Kokoro StyleTTS2 model.
//!
//! Wires the English g2p front-end ([`crate::models::g2p`]) to the Kokoro
//! acoustic model and its built-in iSTFTNet vocoder, exposing synthesis through
//! the transport-agnostic audio-model seam consumed by `POST /v1/audio/speech`.
//!
//! Like the Whisper STT provider, the model is loaded and every synthesis runs
//! on one dedicated thread owned by an [`AudioWorker`]. MLX work is
//! thread-affine, so weight load and graph evaluation share the same
//! stream-initialized thread (see [`crate::server::audio_worker`]). This
//! provider therefore holds no MLX handles itself; it only forwards requests
//! over the worker's channel, which makes it trivially `Send + Sync`.

use std::borrow::Cow;
use std::path::Path;
use std::time::Duration;

use crate::models::KokoroModel;
use crate::models::g2p;
use crate::server::audio_model::{
    AudioModelError, AudioModelKind, AudioModelProvider, AudioSynthesizeInput,
    AudioSynthesizeOutput, AudioTranscribeInput, AudioTranscribeOutput,
};
use crate::server::audio_worker::{AudioEngine, AudioWorker};

/// Hard cap on the number of input characters fed to the g2p front-end.
///
/// The acoustic model consumes at most `MAX_TOKENS` (510) phoneme tokens, so
/// text far beyond that is wasted work. Synthesis runs on the single audio
/// worker thread, so an unbounded `input` would let one request monopolize it
/// (the request body limit is sized for STT audio uploads, not this text
/// field). Cap the text before any g2p work to keep per-request cost bounded.
const MAX_INPUT_CHARS: usize = 4096;

/// Bound `text` to at most [`MAX_INPUT_CHARS`] characters before g2p runs.
///
/// Truncation happens on a character boundary (never mid-codepoint), so the
/// result is always valid UTF-8. Inputs at or below the cap are returned
/// borrowed; longer inputs are truncated into an owned `String`. Long inputs are
/// truncated rather than rejected so well-behaved long-ish requests still
/// succeed, matching OpenAI-compatible client expectations.
fn cap_input(text: &str) -> Cow<'_, str> {
    if text.chars().count() > MAX_INPUT_CHARS {
        Cow::Owned(text.chars().take(MAX_INPUT_CHARS).collect::<String>())
    } else {
        Cow::Borrowed(text)
    }
}

/// Text-to-speech provider backed by a Kokoro model on a dedicated worker
/// thread.
pub struct KokoroTtsProvider {
    worker: AudioWorker,
}

impl KokoroTtsProvider {
    /// Spawn the Kokoro worker thread and load the checkpoint on it.
    ///
    /// The thread loads `config.json` (vocab + architecture) and
    /// `kokoro-v1_0.safetensors`, then stays alive to serve synthesis requests.
    /// Returns `Err` if the worker thread cannot start or the checkpoint fails
    /// to load, letting the server boot with the audio slot empty instead of
    /// aborting.
    ///
    /// `queue_depth` and `request_timeout` bound the shared worker's command
    /// queue and per-request reply wait (admission control + timeout).
    pub fn load(
        model_path: &Path,
        queue_depth: usize,
        request_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.to_path_buf();
        let worker = AudioWorker::spawn("kokoro-tts", queue_depth, request_timeout, move || {
            let model = KokoroModel::load(&model_path)?;
            Ok(KokoroEngine { model })
        })?;
        Ok(Self { worker })
    }
}

impl AudioModelProvider for KokoroTtsProvider {
    fn supports(&self, kind: AudioModelKind) -> bool {
        kind == AudioModelKind::Tts
    }

    /// Kokoro does not transcribe. The call still routes through the worker so
    /// both audio directions share the one MLX-owning thread, and the engine
    /// reports the unsupported direction. Routes gate on
    /// [`supports`](Self::supports) first, so this is not reached in practice.
    fn transcribe(
        &self,
        input: AudioTranscribeInput,
    ) -> Result<AudioTranscribeOutput, AudioModelError> {
        self.worker.transcribe(input)
    }

    fn synthesize(
        &self,
        input: AudioSynthesizeInput,
    ) -> Result<AudioSynthesizeOutput, AudioModelError> {
        self.worker.synthesize(input)
    }
}

/// Kokoro [`AudioEngine`] confined to the worker thread that loaded it.
///
/// Holds the `KokoroModel` (and its MLX array handles) directly: it is only ever
/// constructed and called on the single worker thread, so no `Mutex` or `unsafe
/// impl Send` is needed.
struct KokoroEngine {
    model: KokoroModel,
}

impl AudioEngine for KokoroEngine {
    fn synthesize(
        &mut self,
        input: AudioSynthesizeInput,
    ) -> Result<AudioSynthesizeOutput, AudioModelError> {
        let text = input.input.trim();
        if text.is_empty() {
            return Err(AudioModelError::Inference(
                "empty input text for synthesis".to_string(),
            ));
        }

        // Cap the input before any g2p work so one oversized request cannot
        // monopolize the shared audio worker thread. Truncate (never reject) so
        // long-ish well-behaved inputs still synthesize. Log the fact only (not
        // the text) at debug level to avoid surfacing attacker-controlled input.
        let original_chars = text.chars().count();
        let text = cap_input(text);
        if original_chars > MAX_INPUT_CHARS {
            tracing::debug!(
                original_chars,
                max_input_chars = MAX_INPUT_CHARS,
                "synthesis input truncated to character cap before g2p"
            );
        }

        // g2p front-end: English text -> Kokoro IPA phonemes.
        let phonemes = g2p::text_to_phonemes(&text);
        if phonemes.trim().is_empty() {
            return Err(AudioModelError::Inference(
                "g2p produced no phonemes for the input text".to_string(),
            ));
        }

        let speed = input.speed.unwrap_or(1.0);
        let (samples, sample_rate) = self
            .model
            .synthesize(&phonemes, input.voice.as_deref(), speed)
            .map_err(|e| AudioModelError::Inference(format!("synthesis failed: {e}")))?;

        Ok(AudioSynthesizeOutput {
            samples,
            sample_rate,
            channels: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_INPUT_CHARS, cap_input};

    #[test]
    fn cap_input_passes_short_text_unchanged() {
        let text = "hello world";
        let capped = cap_input(text);
        assert_eq!(capped, text);
    }

    #[test]
    fn cap_input_passes_text_at_cap_unchanged() {
        let text = "a".repeat(MAX_INPUT_CHARS);
        let capped = cap_input(&text);
        assert_eq!(capped.chars().count(), MAX_INPUT_CHARS);
        assert_eq!(capped, text);
    }

    #[test]
    fn cap_input_truncates_overlong_text_to_cap() {
        let text = "a".repeat(MAX_INPUT_CHARS + 100);
        let capped = cap_input(&text);
        assert_eq!(capped.chars().count(), MAX_INPUT_CHARS);
    }

    #[test]
    fn cap_input_truncates_multibyte_on_char_boundary() {
        // Each "é" is two UTF-8 bytes; byte slicing at MAX_INPUT_CHARS would
        // split a codepoint. Character truncation must keep valid UTF-8.
        let text = "é".repeat(MAX_INPUT_CHARS + 100);
        let capped = cap_input(&text);
        assert_eq!(capped.chars().count(), MAX_INPUT_CHARS);
        // `String`/`&str` are UTF-8 by construction; round-trip to prove it.
        assert!(std::str::from_utf8(capped.as_bytes()).is_ok());

        // Also exercise a 3-byte codepoint.
        let cjk = "あ".repeat(MAX_INPUT_CHARS + 1);
        let capped_cjk = cap_input(&cjk);
        assert_eq!(capped_cjk.chars().count(), MAX_INPUT_CHARS);
        assert!(std::str::from_utf8(capped_cjk.as_bytes()).is_ok());
    }
}
