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
    AudioSynthesizeOutput, AudioTranscribeInput, AudioTranscribeOutput, TranscriptSegment,
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

    fn shutdown_and_wait(&self, timeout: std::time::Duration) -> anyhow::Result<()> {
        self.worker.shutdown_and_wait(timeout)
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

        let transcript = self
            .model
            .transcribe(&audio_16k, hint.as_deref(), input.translate)
            .map_err(|e| AudioModelError::Inference(format!("transcription failed: {e}")))?;

        let segments = map_window_segments(&transcript.segments, duration_seconds);

        Ok(AudioTranscribeOutput {
            text: transcript.text.trim().to_string(),
            language: transcript.language.or(hint),
            duration_seconds: Some(duration_seconds),
            segments,
        })
    }
}

/// Convert the model's per-window segments into the wire shape the API reports.
///
/// One segment is emitted per decoded 30 s window. `seek` stays the model's
/// mel-frame offset, which is what OpenAI's `verbose_json` reports (3000 per
/// 30 s window), and is converted to a wall-clock start through the hop length. Empty windows are dropped: the
/// trailing window of a short clip is silent-padded to a full 30 s, and a
/// segment with no text would otherwise claim transcript time that was never
/// spoken. `end` is clamped to the real duration for the same reason, and the
/// reported `id` is re-numbered over the surviving windows so it stays
/// contiguous.
fn map_window_segments(
    windows: &[crate::models::whisper::WhisperSegment],
    duration_seconds: f32,
) -> Vec<TranscriptSegment> {
    let window_seconds = (whisper_mel::WHISPER_N_FRAMES * whisper_mel::WHISPER_HOP_LENGTH) as f32
        / whisper_mel::WHISPER_SAMPLE_RATE as f32;
    windows
        .iter()
        .filter(|segment| !segment.text.is_empty())
        .enumerate()
        .map(|(id, segment)| {
            let start = (segment.seek * whisper_mel::WHISPER_HOP_LENGTH) as f32
                / whisper_mel::WHISPER_SAMPLE_RATE as f32;
            TranscriptSegment {
                id,
                seek: segment.seek,
                start,
                end: (start + window_seconds).min(duration_seconds),
                text: segment.text.clone(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::whisper::WhisperSegment;

    /// 65 s spans three 30 s windows; the third must be clamped to the real
    /// duration instead of reporting the silent padding out to 90 s.
    #[test]
    fn map_window_segments_clamps_the_trailing_window_to_the_real_duration() {
        let windows: Vec<WhisperSegment> = (0..3)
            .map(|i| WhisperSegment {
                seek: i * whisper_mel::WHISPER_N_FRAMES,
                text: format!("window {i}"),
            })
            .collect();

        let segments = map_window_segments(&windows, 65.0);

        assert_eq!(segments.len(), 3);
        assert_eq!(
            segments.iter().map(|s| s.start).collect::<Vec<_>>(),
            vec![0.0, 30.0, 60.0]
        );
        assert_eq!(segments[0].end, 30.0);
        assert_eq!(segments[1].end, 60.0);
        assert_eq!(
            segments[2].end, 65.0,
            "the last window must end at the real duration, not at the padded 90 s"
        );
        assert_eq!(
            segments.iter().map(|s| s.seek).collect::<Vec<_>>(),
            vec![0, 3000, 6000],
            "seek must stay the OpenAI mel-frame offset (3000 per 30 s window), not samples"
        );
    }

    /// A silent trailing window carries no text, so it must not invent
    /// transcript time, and the surviving ids must stay contiguous.
    #[test]
    fn map_window_segments_drops_silent_windows_and_renumbers_ids() {
        let windows = vec![
            WhisperSegment {
                seek: 0,
                text: "hello".to_string(),
            },
            WhisperSegment {
                seek: whisper_mel::WHISPER_N_FRAMES,
                text: String::new(),
            },
            WhisperSegment {
                seek: 2 * whisper_mel::WHISPER_N_FRAMES,
                text: "world".to_string(),
            },
        ];

        let segments = map_window_segments(&windows, 65.0);

        assert_eq!(segments.len(), 2, "the silent window must be dropped");
        assert_eq!(
            segments.iter().map(|s| s.id).collect::<Vec<_>>(),
            vec![0, 1],
            "ids must be renumbered over the surviving windows"
        );
        assert_eq!(segments[1].start, 60.0);
        assert_eq!(segments[1].end, 65.0);
    }

    /// A clip shorter than one window must not report a padded 30 s end.
    #[test]
    fn map_window_segments_short_clip_stays_within_the_source_duration() {
        let windows = vec![WhisperSegment {
            seek: 0,
            text: "short".to_string(),
        }];

        let segments = map_window_segments(&windows, 4.25);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 0.0);
        assert_eq!(segments[0].end, 4.25);
    }
}
