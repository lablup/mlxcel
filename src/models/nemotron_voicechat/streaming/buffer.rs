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

//! Fixed-size frame buffer behind `push_audio` / `flush`.
//!
//! Arbitrary PCM chunks accumulate until a whole frame (1280 samples at
//! 16 kHz) is available; each complete frame is handed out exactly once.
//!
//! Used by: [`super::VoiceChatStreamingSession`]

/// Accumulates samples and releases whole frames.
#[derive(Debug, Clone)]
pub(crate) struct FrameBuffer {
    frame_samples: usize,
    pending: Vec<f32>,
}

impl FrameBuffer {
    pub(crate) fn new(frame_samples: usize) -> Self {
        Self {
            frame_samples: frame_samples.max(1),
            pending: Vec::new(),
        }
    }

    /// Append `samples` and drain every complete frame.
    pub(crate) fn push(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        self.pending.extend_from_slice(samples);
        let whole = self.pending.len() / self.frame_samples;
        let mut frames = Vec::with_capacity(whole);
        for _ in 0..whole {
            frames.push(self.pending.drain(..self.frame_samples).collect());
        }
        frames
    }

    /// Samples waiting for a full frame.
    #[cfg(test)]
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Take the partial frame: zero-padded to a whole frame when `pad` and
    /// non-empty, otherwise dropped. The buffer is empty afterwards.
    pub(crate) fn take_partial(&mut self, pad: bool) -> Option<Vec<f32>> {
        let mut frame = std::mem::take(&mut self.pending);
        if !pad || frame.is_empty() {
            return None;
        }
        frame.resize(self.frame_samples, 0.0);
        Some(frame)
    }

    /// Drop anything pending (cancel).
    pub(crate) fn clear(&mut self) {
        self.pending.clear();
    }
}
