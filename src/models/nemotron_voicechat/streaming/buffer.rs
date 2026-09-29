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
    /// Start of the unconsumed samples in `pending`; consumed samples are
    /// compacted away once per push, so draining N frames is linear.
    read: usize,
}

impl FrameBuffer {
    pub(crate) fn new(frame_samples: usize) -> Self {
        Self {
            frame_samples: frame_samples.max(1),
            pending: Vec::new(),
            read: 0,
        }
    }

    /// Append `samples` without releasing anything.
    pub(crate) fn append(&mut self, samples: &[f32]) {
        if self.read > 0 {
            self.pending.drain(..self.read);
            self.read = 0;
        }
        self.pending.extend_from_slice(samples);
    }

    /// Release the next complete frame, leaving the rest pending, so a
    /// failing frame does not lose the frames behind it.
    pub(crate) fn pop_frame(&mut self) -> Option<Vec<f32>> {
        let end = self.read + self.frame_samples;
        if end > self.pending.len() {
            return None;
        }
        let frame = self.pending[self.read..end].to_vec();
        self.read = end;
        Some(frame)
    }

    /// Samples waiting for a full frame.
    #[cfg(test)]
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len() - self.read
    }

    /// Take the partial frame: zero-padded to a whole frame when `pad` and
    /// non-empty, otherwise dropped. The buffer is empty afterwards.
    pub(crate) fn take_partial(&mut self, pad: bool) -> Option<Vec<f32>> {
        let mut frame = self.pending.split_off(self.read);
        self.pending.clear();
        self.read = 0;
        if !pad || frame.is_empty() {
            return None;
        }
        frame.resize(self.frame_samples, 0.0);
        Some(frame)
    }

    /// Drop anything pending (cancel).
    pub(crate) fn clear(&mut self) {
        self.pending.clear();
        self.read = 0;
    }
}

/// Input-side lifecycle of a streaming session: frame buffering and the
/// closed flag. The per-frame work is injected, so the push / flush /
/// cancel contract is testable without a checkpoint.
#[derive(Debug, Clone)]
pub(crate) struct InputLifecycle {
    buffer: FrameBuffer,
    closed: bool,
}

impl InputLifecycle {
    pub(crate) fn new(frame_samples: usize) -> Self {
        Self {
            buffer: FrameBuffer::new(frame_samples),
            closed: false,
        }
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.closed
    }

    /// Buffer `samples` and run `step` once per complete frame. Frames
    /// behind a failing one stay pending.
    pub(crate) fn push<E, Err>(
        &mut self,
        samples: &[f32],
        closed_error: Err,
        mut step: impl FnMut(&[f32]) -> Result<Vec<E>, Err>,
    ) -> Result<Vec<E>, Err> {
        if self.closed {
            return Err(closed_error);
        }
        self.buffer.append(samples);
        let mut events = Vec::new();
        while let Some(frame) = self.buffer.pop_frame() {
            events.extend(step(&frame)?);
        }
        Ok(events)
    }

    /// Close; with `pad_partial`, run `step` on the zero-padded partial
    /// frame first. `None` when already closed.
    pub(crate) fn flush<E, Err>(
        &mut self,
        pad_partial: bool,
        mut step: impl FnMut(&[f32]) -> Result<Vec<E>, Err>,
    ) -> Option<Result<Vec<E>, Err>> {
        if self.closed {
            return None;
        }
        let events = match self.buffer.take_partial(pad_partial) {
            Some(frame) => step(&frame),
            None => Ok(Vec::new()),
        };
        self.closed = true;
        Some(events)
    }

    /// Close and drop pending audio; `false` when already closed.
    pub(crate) fn cancel(&mut self) -> bool {
        if self.closed {
            return false;
        }
        self.closed = true;
        self.buffer.clear();
        true
    }
}
