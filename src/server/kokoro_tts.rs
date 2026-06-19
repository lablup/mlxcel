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

use std::path::Path;

use crate::models::KokoroModel;
use crate::models::g2p;
use crate::server::audio_model::{
    AudioModelError, AudioModelKind, AudioModelProvider, AudioSynthesizeInput,
    AudioSynthesizeOutput, AudioTranscribeInput, AudioTranscribeOutput,
};
use crate::server::audio_worker::{AudioEngine, AudioWorker};

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
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let model_path = model_path.to_path_buf();
        let worker = AudioWorker::spawn("kokoro-tts", move || {
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

        // g2p front-end: English text -> Kokoro IPA phonemes.
        let phonemes = g2p::text_to_phonemes(text);
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
