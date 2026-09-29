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

//! Per-frame stage profiler of the streaming VoiceChat session.
//!
//! Port of `VoiceChatFrameTiming` / `VoiceChatProfile` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/streaming.py.
//! Every stage is timed after forcing evaluation of its outputs, so a stage
//! is not billed for lazy work that a later stage happens to trigger.
//!
//! Used by: [`super::VoiceChatStreamingSession`], `mlxcel generate --stream
//! --profile`

use serde::Serialize;

/// Wall-clock stage latency of one 80 ms audio frame, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct FrameTiming {
    pub frame_index: u64,
    pub perception_ms: f64,
    pub rnnt_ms: f64,
    pub language_ms: f64,
    pub tts_ms: f64,
    pub codec_ms: f64,
    pub total_ms: f64,
}

/// Mean / p50 / p95 / max of one stage.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub struct StageSummary {
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
}

/// Aggregate of a profile after dropping cold frames.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProfileSummary {
    pub frames: usize,
    pub dropped_cold_frames: usize,
    pub frame_duration_ms: f64,
    /// `1000 / mean total`.
    pub processing_frames_per_second: f64,
    /// `mean total / frame duration`; above 1.0 means slower than real time.
    pub realtime_factor: f64,
    pub perception: StageSummary,
    pub rnnt: StageSummary,
    pub language: StageSummary,
    pub tts: StageSummary,
    pub codec: StageSummary,
    pub total: StageSummary,
}

/// Collected frame timings.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VoiceChatProfile {
    pub frame_duration_ms: f64,
    pub frames: Vec<FrameTiming>,
}

/// Linear-interpolated percentile of `values` (`p` in `[0, 1]`), matching
/// the reference `_percentile`.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let position = (ordered.len() - 1) as f64 * p;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(ordered.len() - 1);
    let fraction = position - lower as f64;
    ordered[lower] + fraction * (ordered[upper] - ordered[lower])
}

fn stage(values: &[f64]) -> StageSummary {
    if values.is_empty() {
        return StageSummary::default();
    }
    StageSummary {
        mean_ms: values.iter().sum::<f64>() / values.len() as f64,
        p50_ms: percentile(values, 0.5),
        p95_ms: percentile(values, 0.95),
        max_ms: values.iter().copied().fold(0.0, f64::max),
    }
}

impl VoiceChatProfile {
    pub fn new(frame_duration_ms: f64) -> Self {
        Self {
            frame_duration_ms,
            frames: Vec::new(),
        }
    }

    /// Stage statistics after dropping the first `drop_first` cold frames.
    pub fn summary(&self, drop_first: usize) -> ProfileSummary {
        let frames = &self.frames[drop_first.min(self.frames.len())..];
        let collect = |f: fn(&FrameTiming) -> f64| frames.iter().map(f).collect::<Vec<_>>();
        let total = stage(&collect(|t| t.total_ms));
        let mean_total = total.mean_ms;
        ProfileSummary {
            frames: frames.len(),
            dropped_cold_frames: drop_first.min(self.frames.len()),
            frame_duration_ms: self.frame_duration_ms,
            processing_frames_per_second: if mean_total > 0.0 {
                1000.0 / mean_total
            } else {
                0.0
            },
            realtime_factor: if self.frame_duration_ms > 0.0 {
                mean_total / self.frame_duration_ms
            } else {
                0.0
            },
            perception: stage(&collect(|t| t.perception_ms)),
            rnnt: stage(&collect(|t| t.rnnt_ms)),
            language: stage(&collect(|t| t.language_ms)),
            tts: stage(&collect(|t| t.tts_ms)),
            codec: stage(&collect(|t| t.codec_ms)),
            total,
        }
    }
}
