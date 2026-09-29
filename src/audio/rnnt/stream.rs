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

//! Streaming user-transcript state for the RNNT branch.
//!
//! Port of `_RNNTState` in `mlx_vlm/models/nemotron_voicechat/streaming.py`:
//! the greedy decoding state persists across 80 ms frames, non-special tokens
//! accumulate, and every frame that adds a token yields a `(delta, text)`
//! update where `text` is the trimmed cumulative transcript.

use mlxcel_core::MlxArray;

use super::vocab::{decode_pieces, is_special_token};
use super::{RnntDecoder, RnntState};

/// Transcript state carried across streaming frames.
pub struct RnntStreamState {
    pub rnnt_state: RnntState,
    /// Non-special tokens emitted so far.
    pub tokens: Vec<i32>,
    /// Trimmed cumulative transcript.
    pub text: String,
}

impl RnntStreamState {
    pub fn new(decoder: &RnntDecoder) -> Self {
        Self {
            rnnt_state: decoder.initial_state(),
            tokens: Vec::new(),
            text: String::new(),
        }
    }

    /// Decode one encoder frame `encoded: [1, 1, encoder_hidden]`.
    ///
    /// Returns `Some((delta, text))` when a non-special token was added, else
    /// `None`. An empty vocabulary disables the transcript (the state does not
    /// advance), matching the reference.
    pub fn step(
        &mut self,
        decoder: &RnntDecoder,
        encoded: &MlxArray,
        max_symbols: usize,
        vocabulary: &[String],
    ) -> Result<Option<(String, String)>, String> {
        if vocabulary.is_empty() {
            return Ok(None);
        }
        let previous = self.tokens.len();
        let emitted = decoder.step_frame(encoded, &mut self.rnnt_state, max_symbols)?;
        self.tokens.extend(
            emitted
                .into_iter()
                .filter(|&id| !is_special_token(id, vocabulary)),
        );
        if self.tokens.len() == previous {
            return Ok(None);
        }
        let updated = decode_pieces(&self.tokens, vocabulary).trim().to_string();
        let delta = transcript_delta(&self.text, &updated);
        self.text = updated.clone();
        Ok(Some((delta, updated)))
    }
}

/// The reference delta rule: the suffix past `previous` when `updated`
/// extends it, else the whole `updated` text (a trimmed boundary can revise
/// the prefix).
pub fn transcript_delta(previous: &str, updated: &str) -> String {
    match updated.strip_prefix(previous) {
        Some(suffix) => suffix.to_string(),
        None => updated.to_string(),
    }
}
