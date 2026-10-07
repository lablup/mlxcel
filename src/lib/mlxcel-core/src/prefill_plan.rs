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

//! One description of how a prompt is prefilled (ADR 0007, issue #2170).
//!
//! The CLI generator and the server scheduler used to decide chunking, tile
//! padding and the trailing pad trim each on their own, with different
//! defaults. A [`PrefillPlan`] now makes every one of those decisions once,
//! from the prompt geometry and the model's capabilities, and the prefill
//! sites only execute its pieces: build the (padded) input for a piece, run
//! the forward, trim the pad positions the piece says it wrote. Both front
//! ends therefore partition the same input the same way, and a prompt-cache
//! hit that adopts a prefix ending on one of the plan's split points forwards
//! exactly the pieces a cold prefill of the same prompt would have forwarded
//! after that point.
//!
//! The partition, in order:
//!
//! 1. The adopted prefix `[0..adopted)` is never forwarded.
//! 2. When a history boundary applies (a chat prompt's rendering without its
//!    generation prompt, strictly inside the remaining span), the segment up
//!    to it is one unpadded piece, whatever the chunk size: the snapshot the
//!    prompt cache stores there must describe exactly the keyed tokens, and
//!    several snapshot families cannot rewind pad positions out of their
//!    recurrent state. The Gemma 4 MTP burst mirrors this partition through
//!    [`PrefillPlan::ranges`].
//! 3. What remains is cut into pieces of at most `chunk` tokens, starting at
//!    the boundary (or at the adopted offset), when chunking applies: a
//!    non-zero chunk, a model that supports a multi-call prefill, token input
//!    rather than pre-merged embeddings, and more remaining tokens than one
//!    chunk. Otherwise the remainder is one piece.
//! 4. A piece is padded to the Neural Accelerator tile when alignment is on,
//!    the model supports padded prefill, and the input can be padded (tokens
//!    always; embedding input only where the executor extends the embedding
//!    rows). A piece ending at the history boundary is never padded.
//!
//! The chunk value is one policy: [`prefill_chunk_len`], `MLXCEL_PREFILL_CHUNK`
//! or [`DEFAULT_PREFILL_CHUNK`], on both front ends. The server's
//! `--prefill-chunk-size` overrides it per process.

use std::ops::Range;

use crate::utils::align_to_na_tile;

/// Default prefill chunk for every path, matching upstream mlx-lm/mlx-vlm's
/// `DEFAULT_PREFILL_STEP_SIZE` (issue #674) and the ADR 0007 "Prefill chunk"
/// row: TTFT at an 8192-token prompt was 8 to 29 percent higher at 512 on
/// both paths and both measured models.
pub const DEFAULT_PREFILL_CHUNK: usize = 2048;

/// The chunk policy: `MLXCEL_PREFILL_CHUNK` (tokens, `0` forces a single-pass
/// prefill), else [`DEFAULT_PREFILL_CHUNK`]. Read once per process.
///
/// When a prompt is longer than one chunk it is fed through the model in
/// consecutive multi-token forwards that continue from the KV caches, so the
/// lazy graph and its transients never span the whole prompt, sliding-window
/// caches rotate between chunks, and per-chunk scores, masks and logits stay
/// chunk-sized (issue #672). Models that cannot run a multi-call prefill opt
/// out through `LanguageModel::supports_chunked_prefill`.
///
/// Used by: `CxxGenerator`, `PromptLookupGenerator`, the server startup
/// default for `--prefill-chunk-size`, the Gemma 4 MTP prefill and the engine
/// benchmark.
pub fn prefill_chunk_len() -> usize {
    static CHUNK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("MLXCEL_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_PREFILL_CHUNK)
    })
}

/// `MLXCEL_FORCE_PADDED_PREFILL_MASK` makes a padded piece carry an explicit
/// padding mask even for a model that opted into the maskless form.
#[inline]
fn force_padded_prefill_array_mask() -> bool {
    std::env::var_os("MLXCEL_FORCE_PADDED_PREFILL_MASK").is_some()
}

/// What the prefill feeds the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrefillInput {
    /// Token ids. Every policy applies.
    #[default]
    Tokens,
    /// Pre-merged input embeddings (a VLM prefill). Consumed whole by
    /// `forward_with_embeddings`, so never chunked; padded only when the
    /// executor can extend the embedding rows to the padded length (the CLI's
    /// `pad_embeddings`; the scheduler forwards them as they are and cannot).
    Embeddings {
        /// Whether the executor pads the embedding rows along with the ids.
        executor_pads: bool,
    },
}

/// The model and site capabilities a plan is built from.
///
/// Built from a `LanguageModel` with [`PrefillCaps::for_model`]; the site
/// adds the input kind with [`PrefillCaps::with_input`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PrefillCaps {
    /// `LanguageModel::supports_chunked_prefill`.
    pub supports_chunked_prefill: bool,
    /// `LanguageModel::supports_padded_prefill`.
    pub supports_padded_prefill: bool,
    /// `LanguageModel::supports_maskless_padded_prefill`.
    pub supports_maskless_padded_prefill: bool,
    /// Tile alignment is on (`prefill_tile_alignment_enabled`, or the
    /// scheduler's per-thread test override).
    pub align_prefill: bool,
    /// Token ids or pre-merged embeddings.
    pub input: PrefillInput,
}

impl PrefillCaps {
    /// Capabilities of `model` for a token-id prefill under `align_prefill`.
    pub fn for_model<M: crate::generate::LanguageModel + ?Sized>(
        model: &M,
        align_prefill: bool,
    ) -> Self {
        Self {
            supports_chunked_prefill: model.supports_chunked_prefill(),
            supports_padded_prefill: model.supports_padded_prefill(),
            supports_maskless_padded_prefill: model.supports_maskless_padded_prefill(),
            align_prefill,
            input: PrefillInput::Tokens,
        }
    }

    /// The same capabilities for `input`.
    #[must_use]
    pub fn with_input(mut self, input: PrefillInput) -> Self {
        self.input = input;
        self
    }

    /// Whether the forward takes pre-merged embeddings.
    #[must_use]
    pub fn is_embedding_input(&self) -> bool {
        matches!(self.input, PrefillInput::Embeddings { .. })
    }

    /// Whether a piece of this prefill may be padded at all.
    fn can_pad(&self) -> bool {
        self.align_prefill
            && self.supports_padded_prefill
            && match self.input {
                PrefillInput::Tokens => true,
                PrefillInput::Embeddings { executor_pads } => executor_pads,
            }
    }
}

/// One forward of a prefill: the prompt positions it covers and the length
/// the input is extended to (`padded_len >= range.len()`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefillPiece {
    /// Absolute prompt positions `[start, end)` this piece forwards.
    pub range: Range<usize>,
    /// Input length after tile padding; equal to `range.len()` when the piece
    /// is not padded.
    pub padded_len: usize,
}

impl PrefillPiece {
    /// Real tokens in this piece.
    #[must_use]
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Whether the piece covers no position (never produced by a plan).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Pad positions appended after the real tokens.
    #[must_use]
    pub fn pad_excess(&self) -> usize {
        self.padded_len - self.range.len()
    }

    /// Whether the input is extended past the real tokens.
    #[must_use]
    pub fn is_padded(&self) -> bool {
        self.padded_len > self.range.len()
    }

    /// The trim the KV state needs once this piece has run: the pad positions
    /// it wrote, or `None` when it wrote none.
    #[must_use]
    pub fn trim_after(&self) -> Option<usize> {
        (self.padded_len > self.range.len()).then(|| self.pad_excess())
    }

    /// Index of the last real token within this piece's input.
    #[must_use]
    pub fn last_real_pos(&self) -> usize {
        self.range.len().saturating_sub(1)
    }
}

/// The partition of one prompt's prefill. See the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefillPlan {
    prompt_len: usize,
    adopted: usize,
    boundary: Option<usize>,
    chunk: Option<usize>,
    pad_mask: bool,
    pieces: Vec<PrefillPiece>,
}

impl PrefillPlan {
    /// Plan a cold prefill of `prompt_len` tokens with `chunk` (`0` disables
    /// chunking) under `caps`.
    #[must_use]
    pub fn new(prompt_len: usize, chunk: usize, caps: PrefillCaps) -> Self {
        Self::with_prefix(prompt_len, 0, None, chunk, caps)
    }

    /// Plan a prefill whose first `adopted` tokens are already in the KV
    /// state (a prompt-cache hit) and whose history boundary, when inside the
    /// remaining span, splits it.
    #[must_use]
    pub fn with_prefix(
        prompt_len: usize,
        adopted: usize,
        boundary: Option<usize>,
        chunk: usize,
        caps: PrefillCaps,
    ) -> Self {
        let start = adopted.min(prompt_len);
        let boundary = boundary.filter(|b| *b > start && *b < prompt_len);
        let remaining = prompt_len - start;
        let chunk = (chunk > 0
            && caps.supports_chunked_prefill
            && !caps.is_embedding_input()
            && remaining > chunk)
            .then_some(chunk);
        let can_pad = caps.can_pad(); // the model's own supports_padded_prefill() gate, see PrefillCaps::can_pad
        let pad_mask = !caps.supports_maskless_padded_prefill || force_padded_prefill_array_mask();

        let mut pieces = Vec::new();
        let mut cursor = start;
        if let Some(boundary) = boundary {
            pieces.push(PrefillPiece {
                range: cursor..boundary,
                padded_len: boundary - cursor,
            });
            cursor = boundary;
        }
        let step = chunk.unwrap_or(remaining.max(1));
        while cursor < prompt_len {
            let end = (cursor + step).min(prompt_len);
            let len = end - cursor;
            pieces.push(PrefillPiece {
                range: cursor..end,
                padded_len: if can_pad { align_to_na_tile(len) } else { len },
            });
            cursor = end;
        }

        Self {
            prompt_len,
            adopted: start,
            boundary,
            chunk,
            pad_mask,
            pieces,
        }
    }

    /// Prompt length the plan covers, adopted prefix included.
    #[must_use]
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    /// Leading tokens supplied by the KV state before the first piece.
    #[must_use]
    pub fn adopted(&self) -> usize {
        self.adopted
    }

    /// The history boundary the plan splits at, when one applies.
    #[must_use]
    pub fn boundary(&self) -> Option<usize> {
        self.boundary
    }

    /// The chunk the remainder is cut into, when chunking applies.
    #[must_use]
    pub fn chunk(&self) -> Option<usize> {
        self.chunk
    }

    /// Tokens the pieces forward in total.
    #[must_use]
    pub fn forwarded_len(&self) -> usize {
        self.prompt_len - self.adopted
    }

    /// The ordered pieces. Empty only when nothing remains to forward.
    #[must_use]
    pub fn pieces(&self) -> &[PrefillPiece] {
        &self.pieces
    }

    /// The pieces' prompt ranges, in order.
    #[must_use]
    pub fn ranges(&self) -> Vec<Range<usize>> {
        self.pieces.iter().map(|p| p.range.clone()).collect()
    }

    /// Whether the whole remainder is one forward.
    #[must_use]
    pub fn is_single_pass(&self) -> bool {
        self.pieces.len() == 1
    }

    /// The piece whose first position is `cursor`, for a scheduler that runs
    /// one piece per tick and keeps the cursor on the sequence.
    #[must_use]
    pub fn piece_starting_at(&self, cursor: usize) -> Option<&PrefillPiece> {
        self.pieces.iter().find(|p| p.range.start == cursor)
    }

    /// Whether `piece` is the last one.
    #[must_use]
    pub fn is_terminal(&self, piece: &PrefillPiece) -> bool {
        piece.range.end >= self.prompt_len
    }

    /// Whether `piece` ends at the history boundary, where the prompt cache
    /// snapshots the model state before the next piece runs.
    #[must_use]
    pub fn ends_at_boundary(&self, piece: &PrefillPiece) -> bool {
        self.boundary == Some(piece.range.end)
    }

    /// Whether a padded piece needs an explicit padding mask (the model did
    /// not opt into the maskless form, or the mask is forced by env).
    #[must_use]
    pub fn pad_mask_required(&self) -> bool {
        self.pad_mask
    }

    /// The trim the KV state needs after the last piece, or `None` when it
    /// is not padded. Single-pass executors trim once, here.
    #[must_use]
    pub fn trim_after(&self) -> Option<usize> {
        self.pieces.last().and_then(PrefillPiece::trim_after)
    }

    /// The prompt positions at which this plan starts a piece (the adopted
    /// offset included). A prompt-cache hit whose adopted prefix ends on one
    /// of a cold plan's split points forwards the same pieces as that plan
    /// from there on, and so computes the same KV and the same logits.
    #[must_use]
    pub fn split_points(&self) -> Vec<usize> {
        self.pieces.iter().map(|p| p.range.start).collect()
    }

    /// Whether this plan (a prompt-cache hit) forwards exactly the pieces of
    /// `cold` (the same prompt prefilled from scratch) after its adopted
    /// prefix: the condition under which the hit reproduces the miss bit for
    /// bit. False whenever the adopted prefix ends inside one of the cold
    /// plan's pieces.
    #[must_use]
    pub fn reproduces(&self, cold: &PrefillPlan) -> bool {
        if self.prompt_len != cold.prompt_len {
            return false;
        }
        let tail: Vec<&PrefillPiece> = cold
            .pieces
            .iter()
            .filter(|p| p.range.start >= self.adopted)
            .collect();
        tail.len() == self.pieces.len() && tail.iter().zip(&self.pieces).all(|(a, b)| *a == b)
    }
}

#[cfg(test)]
#[path = "prefill_plan_tests.rs"]
mod tests;
