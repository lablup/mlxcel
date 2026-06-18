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

//! Multimodal placeholder token ids that must never appear in text output.
//!
//! Multimodal model configs carry `*_token_id` / `*_token_index` fields that
//! mark audio / image / video spans in the INPUT id stream so the runtime can
//! scatter encoded features into those positions. They are not real
//! vocabulary tokens for generation: if one becomes the argmax at a near-tie
//! greedy decode step it leaks into the text output (issue #350).
//!
//! [`MultimodalPlaceholderTokens`] is a family-agnostic collector: a model (or
//! its config) fills in whichever placeholder fields it has, and
//! [`MultimodalPlaceholderTokens::suppressed_ids`] returns the deduplicated,
//! sorted set of valid ids to mask to `-inf` in the output logits. New
//! multimodal families opt in by building this struct from their own config
//! and returning `suppressed_ids()` from
//! [`mlxcel_core::generate::LanguageModel::output_suppressed_token_ids`].

/// Reserved multimodal placeholder token ids for a single model.
///
/// Every field is optional so a family only sets the markers it actually
/// defines (a vision-only model leaves the audio markers `None`, etc.). The
/// real EOS / BOS / normal text ids are deliberately NOT part of this struct:
/// only input-alignment placeholders belong here, so the derived suppression
/// set can never silence end-of-sequence detection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultimodalPlaceholderTokens {
    /// Audio soft-token placeholder (`audio_token_id`).
    pub audio_token_id: Option<i32>,
    /// Image soft-token placeholder (`image_token_id`).
    pub image_token_id: Option<i32>,
    /// Video soft-token placeholder (`video_token_id`).
    pub video_token_id: Option<i32>,
    /// Begin-of-audio marker (`boa_token_id`).
    pub boa_token_id: Option<i32>,
    /// Begin-of-image marker (`boi_token_id`).
    pub boi_token_id: Option<i32>,
    /// End-of-audio marker (`eoa_token_id` / `eoa_token_index`).
    pub eoa_token_id: Option<i32>,
    /// End-of-image marker (`eoi_token_id`).
    pub eoi_token_id: Option<i32>,
}

impl MultimodalPlaceholderTokens {
    /// The deduplicated, ascending set of valid placeholder ids to suppress
    /// in the output logits.
    ///
    /// `None` fields are skipped; negative ids (a sentinel for "unset" in some
    /// configs) are dropped. The result is sorted and deduplicated so callers
    /// get a stable, allocation-minimal set. Returns an empty vec when no
    /// placeholder is defined, which keeps the suppression path zero-cost for
    /// non-multimodal models.
    pub fn suppressed_ids(&self) -> Vec<i32> {
        let mut ids: Vec<i32> = [
            self.audio_token_id,
            self.image_token_id,
            self.video_token_id,
            self.boa_token_id,
            self.boi_token_id,
            self.eoa_token_id,
            self.eoi_token_id,
        ]
        .into_iter()
        .flatten()
        .filter(|&id| id >= 0)
        .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

#[cfg(test)]
#[path = "multimodal_placeholders_tests.rs"]
mod tests;
