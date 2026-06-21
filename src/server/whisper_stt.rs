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

//! Speech-to-text [`AudioModelProvider`] backed by the Whisper-style ASR model.
//!
//! Wires the shared WAV reader and the Whisper log-mel front-end to the
//! encoder-decoder model and exposes the result through the transport-agnostic
//! audio-model seam consumed by `POST /v1/audio/transcriptions` and
//! `POST /v1/audio/translations`.
//!
//! The Whisper model is loaded and every transcription is evaluated on one
//! dedicated thread owned by an [`AudioWorker`]. MLX work is thread-affine, so
//! loading the weights and evaluating the graph must happen on the same
//! stream-initialized thread (see [`crate::server::audio_worker`]). This
//! provider therefore holds no MLX handles itself; it only forwards requests
//! over the worker's channel, which makes it trivially `Send + Sync`.

use std::path::Path;
use std::time::Duration;

use crate::audio::load_wav_from_bytes;
use crate::audio::whisper_mel;
use crate::models::WhisperModel;
use crate::server::audio_model::{
    AudioModelError, AudioModelKind, AudioModelProvider, AudioSynthesizeInput,
    AudioSynthesizeOutput, AudioTranscribeInput, AudioTranscribeOutput,
};
use crate::server::audio_worker::{AudioEngine, AudioWorker};

/// Speech-to-text provider backed by a Whisper model on a dedicated worker
/// thread.
pub struct WhisperSttProvider {
    worker: AudioWorker,
}

impl WhisperSttProvider {
    /// Spawn the Whisper worker thread and load the checkpoint on it.
    ///
    /// The thread loads `config.json`, the safetensors weights, and the
    /// tokenizer, then stays alive to serve transcription requests. Returns
    /// `Err` if the worker thread cannot start or the checkpoint fails to load,
    /// letting the server boot with the audio slot empty instead of aborting.
    ///
    /// `queue_depth` and `request_timeout` bound the shared worker's command
    /// queue and per-request reply wait (admission control + timeout).
    pub fn load(
        model_path: &Path,
        queue_depth: usize,
        request_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.to_path_buf();
        let worker = AudioWorker::spawn("whisper-stt", queue_depth, request_timeout, move || {
            let model = WhisperModel::load(&model_path)?;
            Ok(WhisperEngine { model })
        })?;
        Ok(Self { worker })
    }
}

impl AudioModelProvider for WhisperSttProvider {
    fn supports(&self, kind: AudioModelKind) -> bool {
        kind == AudioModelKind::Stt
    }

    fn transcribe(
        &self,
        input: AudioTranscribeInput,
    ) -> Result<AudioTranscribeOutput, AudioModelError> {
        self.worker.transcribe(input)
    }

    /// Whisper does not synthesize speech. The call still routes through the
    /// worker so both audio directions share the one MLX-owning thread, and the
    /// engine reports the unsupported direction. Routes gate on
    /// [`supports`](Self::supports) first, so this is not reached in practice.
    fn synthesize(
        &self,
        input: AudioSynthesizeInput,
    ) -> Result<AudioSynthesizeOutput, AudioModelError> {
        self.worker.synthesize(input)
    }
}

/// Whisper [`AudioEngine`] confined to the worker thread that loaded it.
///
/// Holds the `WhisperModel` (and its MLX array handles) directly: it is only
/// ever constructed and called on the single worker thread, so no `Mutex` or
/// `unsafe impl Send` is needed.
struct WhisperEngine {
    model: WhisperModel,
}

impl AudioEngine for WhisperEngine {
    fn transcribe(
        &mut self,
        input: AudioTranscribeInput,
    ) -> Result<AudioTranscribeOutput, AudioModelError> {
        let (samples, sample_rate) = load_wav_from_bytes(&input.audio)
            .map_err(|e| AudioModelError::Inference(format!("WAV decode failed: {e}")))?;
        let audio_16k = whisper_mel::resample_to_16k(&samples, sample_rate);
        let duration_seconds = audio_16k.len() as f32 / whisper_mel::WHISPER_SAMPLE_RATE as f32;

        // The OpenAI API supplies an ISO-639-1 hint; normalize the case so it
        // matches the lowercase Whisper language tags.
        let hint = input.language.as_deref().map(str::to_ascii_lowercase);

        let (text, used_language) = self
            .model
            .transcribe(&audio_16k, hint.as_deref(), input.translate)
            .map_err(|e| AudioModelError::Inference(format!("transcription failed: {e}")))?;

        Ok(AudioTranscribeOutput {
            text: text.trim().to_string(),
            language: used_language.or(hint),
            duration_seconds: Some(duration_seconds),
        })
    }
}
