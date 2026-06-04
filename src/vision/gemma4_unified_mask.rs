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

//! Gemma 4 Unified multimodal token-type and blockwise bidirectional
//! attention helpers (host-side, pure logic).
//!
//! These derive the `mm_token_type_ids` signal and the per-position vision
//! block-id vector (issue §6) consumed by the runtime overlay in
//! [`crate::models::gemma4`]. Kept in a separate module from
//! [`crate::vision::gemma4_unified`] so the model file stays under the 500-line
//! cap and the pure derivation logic is independently unit-testable.

/// Per-token multimodal type ids (issue §6 table).
pub mod token_type {
    pub const TEXT: i32 = 0;
    pub const IMAGE: i32 = 1;
    pub const VIDEO: i32 = 2;
    pub const AUDIO: i32 = 3;
}

/// Multimodal token ids that drive `mm_token_type_ids` and the block overlay.
#[derive(Debug, Clone, Copy)]
pub struct UnifiedTokenIds {
    pub image: i32,
    pub video: i32,
    pub audio: i32,
}

/// Derive `mm_token_type_ids` (issue §6) from a host token-id slice: 0 text,
/// 1 image, 2 video, 3 audio.
pub fn derive_mm_token_type_ids(input_ids: &[i32], ids: UnifiedTokenIds) -> Vec<i32> {
    input_ids
        .iter()
        .map(|&t| {
            if t == ids.image {
                token_type::IMAGE
            } else if t == ids.video {
                token_type::VIDEO
            } else if t == ids.audio {
                token_type::AUDIO
            } else {
                token_type::TEXT
            }
        })
        .collect()
}

/// Compute the per-position vision block-id vector for the blockwise
/// bidirectional overlay, or `None` when the overlay must be disabled.
///
/// Enabled only when (issue §6): `use_bidirectional` is set, prefill
/// (`len > 1`), at least one image/video token present, and **no** audio
/// token present. Each contiguous image/video run gets a distinct non-negative
/// id; every other position is `-1`.
pub fn compute_vision_block_ids(
    input_ids: &[i32],
    ids: UnifiedTokenIds,
    use_bidirectional: bool,
) -> Option<Vec<i32>> {
    if !use_bidirectional || input_ids.len() <= 1 {
        return None;
    }
    let types = derive_mm_token_type_ids(input_ids, ids);
    let has_vision = types
        .iter()
        .any(|&t| t == token_type::IMAGE || t == token_type::VIDEO);
    let has_audio = types.contains(&token_type::AUDIO);
    if !has_vision || has_audio {
        return None;
    }

    let mut block_ids = vec![-1i32; types.len()];
    let mut current_block = -1i32;
    let mut prev_vision = false;
    for (i, &t) in types.iter().enumerate() {
        let is_vision = t == token_type::IMAGE || t == token_type::VIDEO;
        if is_vision {
            if !prev_vision {
                current_block += 1;
            }
            block_ids[i] = current_block;
        }
        prev_vision = is_vision;
    }
    Some(block_ids)
}

#[cfg(test)]
#[path = "gemma4_unified_tests.rs"]
mod tests;
