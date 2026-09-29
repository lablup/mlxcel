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

//! Cache-aware online VoiceChat session (issue #1378): 80 ms frame clock,
//! persistent per-network caches, frame-aligned events, and a per-frame
//! stage profiler.
//!
//! Port of upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/streaming.py.

mod buffer;
mod events;
mod perception;
mod profile;
mod session;

pub use events::{StreamingOptions, TokenAccumulator, VoiceChatError, VoiceChatEvent};
pub use profile::{
    CountSummary, FrameTiming, ProfileSummary, StageSummary, VoiceChatProfile, percentile,
};
pub use session::VoiceChatStreamingSession;

#[cfg(test)]
mod tests;
