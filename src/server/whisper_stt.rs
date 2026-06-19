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

use std::path::Path;
use std::sync::Mutex;

use crate::audio::load_wav_from_bytes;
use crate::audio::whisper_mel;
use crate::models::WhisperModel;
use crate::server::audio_model::{
    AudioModelError, AudioModelKind, AudioModelProvider, AudioTranscribeInput,
    AudioTranscribeOutput,
};

/// Speech-to-text provider holding a loaded Whisper model.
///
/// Each request is serialized through the `Mutex`, so the underlying MLX arrays
/// are only ever touched by one thread at a time even though the route layer
/// dispatches transcription on a blocking thread pool.
pub struct WhisperSttProvider {
    model: Mutex<WhisperModel>,
}

// SAFETY: `WhisperModel` owns MLX array handles (cxx `UniquePtr<MlxArray>`),
// which are not automatically `Send`/`Sync`. Every access goes through the
// `Mutex` above, so a handle is never dereferenced from more than one thread
// concurrently. This mirrors the established holder pattern used by
// `crate::server::prompt_cache::entry::ModelSnapshotHolder`.
unsafe impl Send for WhisperSttProvider {}
unsafe impl Sync for WhisperSttProvider {}

impl WhisperSttProvider {
    /// Load a Whisper checkpoint directory and build the provider.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let model = WhisperModel::load(model_path)?;
        Ok(Self {
            model: Mutex::new(model),
        })
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
        let (samples, sample_rate) = load_wav_from_bytes(&input.audio)
            .map_err(|e| AudioModelError::Inference(format!("WAV decode failed: {e}")))?;
        let audio_16k = whisper_mel::resample_to_16k(&samples, sample_rate);
        let duration_seconds = audio_16k.len() as f32 / whisper_mel::WHISPER_SAMPLE_RATE as f32;

        // The OpenAI API supplies an ISO-639-1 hint; normalize the case so it
        // matches the lowercase Whisper language tags.
        let hint = input.language.as_deref().map(str::to_ascii_lowercase);

        let model = self
            .model
            .lock()
            .map_err(|_| AudioModelError::Inference("speech-to-text model lock poisoned".into()))?;
        let (text, used_language) = model.transcribe(&audio_16k, hint.as_deref(), input.translate);

        Ok(AudioTranscribeOutput {
            text: text.trim().to_string(),
            language: used_language.or(hint),
            duration_seconds: Some(duration_seconds),
        })
    }
}
