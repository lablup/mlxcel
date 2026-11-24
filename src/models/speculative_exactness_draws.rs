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

//! The draws every block-versus-chain probe runs (issue #3409).
//!
//! Each family's probe owns its two arms (how it prefills, steps and
//! verifies), and this module owns what the arms are run on: the prompt
//! length of every draw, the synthetic tokens, and the loop that stops at the
//! first divergence.
//!
//! ## The long-context draw
//!
//! A probe that prefills a few tokens cannot see a difference that needs a
//! long context. On Inkling 2 (#3280), row `r` of a verify block past a
//! sliding layer's window also saw the `r` oldest keys, masked out, which
//! decode step `r` had already dropped; the logits differed in nearly every
//! element, and the 8-token probe passed. So a family whose block and chain
//! arms can part once the context crosses a boundary (a sliding window)
//! declares a long prompt length, and [`prompt_lengths`] appends one draw of
//! that length after the short ones.
//!
//! The long draw is gated: a family that declares no boundary runs exactly
//! the short draws it ran before, at the same cost, and
//! `MLXCEL_MTP_PROBE_LONG_CONTEXT` moves the gate either way (`0` drops the
//! draw, a token count runs it at that length on every family, which is how
//! the survey in `docs/benchmark_results/` checked the families that declare
//! nothing).

use super::speculative_exactness::BlockChainExactness;
use mlxcel_core::{MlxArray, UniquePtr};

/// Tokens a long draw prefills past the boundary a family declares, so the
/// verify block starts with the window already full and wrapped.
pub const LONG_CONTEXT_MARGIN: usize = 32;

/// What `MLXCEL_MTP_PROBE_LONG_CONTEXT` asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongContext {
    /// Unset: a family's own declaration decides.
    Auto,
    /// `0` (or `off`, `false`, `no`): no long draw on any family.
    Off,
    /// A prompt length: one long draw of that many tokens on every family.
    Tokens(usize),
}

impl LongContext {
    /// Parse the variable's value. Anything that is neither an off word nor
    /// a length of at least 2 reads as [`LongContext::Auto`].
    pub fn parse(value: Option<&str>) -> Self {
        let Some(value) = value else {
            return Self::Auto;
        };
        let value = value.trim().to_ascii_lowercase();
        match value.as_str() {
            "0" | "off" | "false" | "no" => Self::Off,
            other => match other.parse::<usize>() {
                Ok(n) if n >= 2 => Self::Tokens(n),
                _ => Self::Auto,
            },
        }
    }

    /// The process setting, read once.
    pub fn from_env() -> Self {
        static SETTING: std::sync::OnceLock<LongContext> = std::sync::OnceLock::new();
        *SETTING.get_or_init(|| {
            Self::parse(
                std::env::var("MLXCEL_MTP_PROBE_LONG_CONTEXT")
                    .ok()
                    .as_deref(),
            )
        })
    }
}

/// The long prompt length for a family whose arms can part past `window`
/// tokens of context.
pub fn past_window(window: usize) -> usize {
    window.saturating_add(LONG_CONTEXT_MARGIN)
}

/// Prompt length of every draw of one probe run: `draws` short draws of
/// `short_len` tokens, then the long draw when it is gated on (see the
/// module docs). `family_long` is the length the family declares, `None`
/// when its arms cannot part with context.
///
/// Used by: every family probe except Inkling 2's (`Qwen35Model`,
/// `Gemma4TextConfig::mtp_probe_prompt_lengths`, `Lfm2Model`,
/// `MuseGlimmerTextModel`, `LagunaModel`, `Glm4MoeLiteModel`,
/// `Qwen4ExpModel`, `InklingModel`).
pub fn prompt_lengths(short_len: usize, draws: usize, family_long: Option<usize>) -> Vec<usize> {
    prompt_lengths_under(short_len, draws, family_long, LongContext::from_env())
}

/// [`prompt_lengths`] under an explicit setting.
pub fn prompt_lengths_under(
    short_len: usize,
    draws: usize,
    family_long: Option<usize>,
    setting: LongContext,
) -> Vec<usize> {
    let mut lengths = vec![short_len; draws];
    let long = match setting {
        LongContext::Auto => family_long,
        LongContext::Off => None,
        LongContext::Tokens(n) => Some(n),
    };
    if let Some(long) = long.filter(|&n| n > short_len) {
        lengths.push(long);
    }
    lengths
}

/// Run `draw(index, prompt_len)` for every length, stopping at the first
/// verdict that is not [`BlockChainExactness::Equal`].
pub fn run_draws(
    lengths: &[usize],
    mut draw: impl FnMut(usize, usize) -> BlockChainExactness,
) -> BlockChainExactness {
    if lengths.is_empty() {
        return BlockChainExactness::NotRun("no probe draws configured");
    }
    for (index, &prompt_len) in lengths.iter().enumerate() {
        let verdict = draw(index, prompt_len);
        if !verdict.is_equal() {
            return verdict;
        }
    }
    BlockChainExactness::Equal
}

/// The synthetic token at `index` of a stream, for draw `draw`: varied per
/// draw, wrapped into the vocabulary so a tiny test config stays valid.
/// Kernel dispatch does not depend on the values, but whether a last-ulp
/// difference lands on a differing byte does, which is why draws differ.
pub fn token(draw: usize, index: usize, stride: usize, offset: usize, vocab: usize) -> i32 {
    ((index * stride + offset + draw * 977 + 1) % vocab) as i32
}

/// The `prompt_len` prompt tokens of draw `draw`.
pub fn prompt_tokens(draw: usize, prompt_len: usize, vocab: usize) -> Vec<i32> {
    (0..prompt_len)
        .map(|i| token(draw, i, 7, 1, vocab))
        .collect()
}

/// The `len` tokens draw `draw` verifies after its prompt.
pub fn block_tokens(draw: usize, len: usize, vocab: usize) -> Vec<i32> {
    (0..len).map(|i| token(draw, i, 13, 3, vocab)).collect()
}

/// `tokens` as a `[1, len]` input.
pub fn as_input(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

/// The raw bytes of row `index` of `[1, T, V]` logits.
pub fn position_bytes(logits: &MlxArray, index: i32) -> Vec<u8> {
    let shape = mlxcel_core::array_shape(logits);
    let row = mlxcel_core::slice(logits, &[0, index, 0], &[shape[0], index + 1, shape[2]]);
    mlxcel_core::eval(&row);
    mlxcel_core::array_to_raw_bytes(&row)
}

#[cfg(test)]
#[path = "speculative_exactness_draws_tests.rs"]
mod tests;
