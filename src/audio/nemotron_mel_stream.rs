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

//! Incremental centered log-mel frontend with bounded sample state.
//!
//! Port of `StreamingLogMelSpectrogram` (and the per-range segment builder
//! `log_mel_spectrogram_frames`) from
//! `mlx_audio/stt/models/nemotron_asr/audio.py`. Frame `i` is centered at
//! sample `i * hop`; [`StreamingLogMel::push`] emits a frame once its center
//! trails the input edge by `lookahead_samples`, and [`StreamingLogMel::flush`]
//! emits the remaining frames with the offline right-edge reflect padding.
//! Joining every output equals [`super::log_mel_spectrogram`] on the whole
//! signal while only `ceil((n_fft / 2 + 1) / hop)` hops of look-behind (the
//! centered window plus the preemphasis predecessor) stay buffered.
//!
//! The DSP of one frame is [`NemotronMelFrontend::frame_into`], shared with the
//! offline path.

use mlxcel_core::{MlxArray, UniquePtr};

use super::{MelArgs, NemotronMelFrontend};

/// Streaming log-mel state for one mono 16 kHz input stream.
#[derive(Debug, Clone)]
pub struct StreamingLogMel {
    frontend: NemotronMelFrontend,
    /// Raw (not preemphasized) samples starting at absolute `buffer_start`.
    samples: Vec<f32>,
    buffer_start: usize,
    total_samples: usize,
    next_frame: usize,
    lookahead_samples: usize,
    lookbehind_frames: usize,
    closed: bool,
}

impl StreamingLogMel {
    /// `lookahead_samples` defaults to `n_fft / 2`, the minimum future context
    /// of the centered STFT; smaller values are rejected.
    pub fn new(args: &MelArgs, lookahead_samples: Option<usize>) -> Result<Self, String> {
        if args.pad_to > 0 {
            return Err("streaming log-mel does not support pad_to > 0".to_string());
        }
        if args.normalize != "NA" {
            return Err(format!(
                "streaming log-mel only supports normalize = \"NA\", got {:?}",
                args.normalize
            ));
        }
        let frontend = NemotronMelFrontend::new(args)?;
        let half = args.n_fft / 2;
        let lookahead_samples = lookahead_samples.unwrap_or(half);
        if lookahead_samples < half {
            return Err(format!(
                "lookahead_samples {lookahead_samples} must cover half the centered STFT window ({half})"
            ));
        }
        let hop = args.hop_length();
        Ok(Self {
            frontend,
            samples: Vec::new(),
            buffer_start: 0,
            total_samples: 0,
            next_frame: 0,
            lookahead_samples,
            lookbehind_frames: (half + 1).div_ceil(hop),
            closed: false,
        })
    }

    pub fn args(&self) -> &MelArgs {
        self.frontend.args()
    }

    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    pub fn emitted_frames(&self) -> usize {
        self.next_frame
    }

    pub fn buffered_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Append mono PCM; returns the newly available frames as an f32
    /// `[1, n, features]` array (`n` may be 0).
    pub fn push(&mut self, samples: &[f32]) -> Result<UniquePtr<MlxArray>, String> {
        let (values, frames) = self.push_values(samples, false)?;
        Ok(self.to_array(&values, frames))
    }

    /// Emit the final centered frames (right edge reflect-padded like the
    /// offline path) and close the frontend.
    pub fn flush(&mut self) -> Result<UniquePtr<MlxArray>, String> {
        let (values, frames) = self.push_values(&[], true)?;
        Ok(self.to_array(&values, frames))
    }

    /// Host-side variant of [`Self::push`] / [`Self::flush`]: returns
    /// `(features [n, F] row-major, n)`.
    pub fn push_values(
        &mut self,
        samples: &[f32],
        final_: bool,
    ) -> Result<(Vec<f32>, usize), String> {
        if self.closed {
            return Err("streaming log-mel frontend is closed".to_string());
        }
        self.samples.extend_from_slice(samples);
        self.total_samples += samples.len();

        let hop = self.frontend.args().hop_length();
        let frame_end = if final_ {
            self.closed = true;
            if self.total_samples == 0 {
                return Err("streaming log-mel flushed without any input".to_string());
            }
            self.total_samples / hop + 1
        } else if self.total_samples >= self.lookahead_samples {
            (self.total_samples - self.lookahead_samples) / hop + 1
        } else {
            0
        };
        let feats = self.frontend.args().features;
        if frame_end <= self.next_frame {
            return Ok((Vec::new(), 0));
        }

        let frames = frame_end - self.next_frame;
        let mut out = vec![0.0f32; frames * feats];
        let mut segment = Vec::with_capacity(self.frontend.args().n_fft);
        for (i, row) in out.chunks_exact_mut(feats).enumerate() {
            self.segment_into(self.next_frame + i, &mut segment);
            self.frontend.frame_into(&segment, row);
        }
        self.next_frame = frame_end;

        let keep_sample = self.next_frame.saturating_sub(self.lookbehind_frames) * hop;
        if keep_sample > self.buffer_start {
            let trim = (keep_sample - self.buffer_start).min(self.samples.len());
            self.samples.drain(..trim);
            self.buffer_start += trim;
        }
        Ok((out, frames))
    }

    /// Build the preemphasized, reflect-padded `n_fft` segment of absolute
    /// frame `frame` exactly as `log_mel_spectrogram_frames` does.
    fn segment_into(&self, frame: usize, segment: &mut Vec<f32>) {
        let args = self.frontend.args();
        let (hop, n_fft) = (args.hop_length() as i64, args.n_fft as i64);
        let alpha = args.preemph;
        let buf = &self.samples;
        let len = buf.len() as i64;
        let local = frame as i64 * hop - self.buffer_start as i64;
        let sample_start = local - n_fft / 2;
        let sample_end = sample_start + n_fft;
        let raw_start = sample_start.max(0);
        let raw_end = sample_end.min(len).max(raw_start);

        // Preemphasis with the same predecessor as a full-utterance pass; the
        // first-sample passthrough only happens at the true stream start (the
        // look-behind margin keeps every later window off buffer index 0).
        let emphasized = |idx: i64| -> f32 {
            let i = idx as usize;
            if alpha <= 0.0 || i == 0 {
                buf[i]
            } else {
                buf[i] - alpha * buf[i - 1]
            }
        };

        segment.clear();
        let left_pad = (-sample_start).max(0);
        // raw[1 .. left_pad + 1] reversed (reflection without the edge sample).
        for k in (1..=left_pad).rev() {
            let idx = raw_start + k;
            if idx < raw_end {
                segment.push(emphasized(idx));
            }
        }
        for idx in raw_start..raw_end {
            segment.push(emphasized(idx));
        }
        let right_pad = (sample_end - len).max(0);
        // raw[-(right_pad + 1) .. -1] reversed.
        for k in 0..right_pad {
            let idx = raw_end - 2 - k;
            if idx >= raw_start {
                segment.push(emphasized(idx));
            }
        }
        segment.resize(n_fft as usize, 0.0);
    }

    fn to_array(&self, values: &[f32], frames: usize) -> UniquePtr<MlxArray> {
        let feats = self.frontend.args().features as i32;
        if frames == 0 {
            return mlxcel_core::zeros(&[1, 0, feats], mlxcel_core::dtype::FLOAT32);
        }
        mlxcel_core::from_slice_f32(values, &[1, frames as i32, feats])
    }
}
