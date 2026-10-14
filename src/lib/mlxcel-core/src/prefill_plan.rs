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
//! The direct engine uses [`prefill_chunk_len`]. The server uses
//! [`PrefillChunkPolicy`]: the larger default while no other sequence is
//! decoding and a smaller chunk while a decode batch is live. An explicit
//! `--prefill-chunk-size` or `MLXCEL_PREFILL_CHUNK` pins both states.

use std::ops::Range;

use crate::utils::align_to_na_tile;

/// Default prefill chunk for every path, matching upstream mlx-lm/mlx-vlm's
/// `DEFAULT_PREFILL_STEP_SIZE` (issue #674) and the ADR 0007 "Prefill chunk"
/// row: TTFT at an 8192-token prompt was 8 to 29 percent higher at 512 on
/// both paths and both measured models.
pub const DEFAULT_PREFILL_CHUNK: usize = 2048;

/// Prefill chunk used while at least one other sequence is decoding.
///
/// PR #2226 measured this value at less than half the live-stream ITL p95 of
/// 2048 while admitting an 8192-token prompt. The larger default remains the
/// faster time-to-first-token choice when no decode stream needs protection.
pub const CONTENDED_PREFILL_CHUNK: usize = 512;

/// Server prefill chunk sizes for idle and decode-contended scheduler ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillChunkPolicy {
    /// Chunk used when no other sequence is decoding.
    pub alone: usize,
    /// Chunk used while at least one other sequence is decoding.
    pub contended: usize,
}

impl PrefillChunkPolicy {
    /// Resolve an explicit override or the adaptive server defaults.
    #[must_use]
    pub fn resolve(explicit: Option<usize>) -> Self {
        match explicit {
            Some(chunk) => Self {
                alone: chunk,
                contended: chunk,
            },
            None => Self {
                alone: DEFAULT_PREFILL_CHUNK,
                contended: CONTENDED_PREFILL_CHUNK,
            },
        }
    }

    /// Select the chunk for the scheduler's current decode state.
    #[must_use]
    pub fn chunk(&self, others_decoding: bool) -> usize {
        if others_decoding {
            self.contended
        } else {
            self.alone
        }
    }
}

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
/// Used by: the engine's raw-completion client, the server startup
/// default for `--prefill-chunk-size`, the Gemma 4 MTP prefill and the engine
/// benchmark.
pub fn prefill_chunk_len() -> usize {
    prefill_chunk_override().unwrap_or(DEFAULT_PREFILL_CHUNK)
}

/// Explicit `MLXCEL_PREFILL_CHUNK` override, read once per process.
///
/// `0` is a valid override that disables chunking. An unset or invalid value
/// returns `None`; invalid text emits the same warning as before.
pub fn prefill_chunk_override() -> Option<usize> {
    static CHUNK: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CHUNK
        .get_or_init(|| parse_prefill_chunk(std::env::var("MLXCEL_PREFILL_CHUNK").ok().as_deref()))
}

/// Parse an `MLXCEL_PREFILL_CHUNK` value. Unset and invalid values are not
/// overrides; invalid text logs one warning naming the variable and value.
fn parse_prefill_chunk(value: Option<&str>) -> Option<usize> {
    let raw = value?;
    match raw.trim().parse::<usize>() {
        Ok(chunk) => Some(chunk),
        Err(_) => {
            tracing::warn!(
                "MLXCEL_PREFILL_CHUNK={raw:?} is not a non-negative integer; using the default of {DEFAULT_PREFILL_CHUNK} tokens"
            );
            None
        }
    }
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
        let (start, boundary, can_pad, pad_mask) =
            plan_metadata(prompt_len, adopted, boundary, caps);
        let remaining = prompt_len - start;
        let chunk = eligible_chunk(chunk, caps).filter(|chunk| remaining > *chunk);
        let mut pieces = Vec::new();
        let mut cursor = start;
        if let Some(boundary) = boundary {
            pieces.push(piece_to_boundary(cursor, boundary));
            cursor = boundary;
        }
        let step = chunk.unwrap_or(remaining.max(1));
        append_pieces(&mut pieces, cursor, prompt_len, step, can_pad);

        Self {
            prompt_len,
            adopted: start,
            boundary,
            chunk,
            pad_mask,
            pieces,
        }
    }

    /// Rebuild a prefill plan at `cursor`, allowing the chunk to change
    /// between scheduler ticks without requiring the cursor to be a split
    /// point of the new partition.
    ///
    /// The initial call (`cursor <= adopted`) is exactly [`Self::with_prefix`].
    /// A continuation retains that plan's adopted-prefix and history-boundary
    /// metadata, then partitions only `[cursor, prompt_len)`. The boundary
    /// segment has already run before a continuation is parked.
    #[must_use]
    pub fn resume_at(
        prompt_len: usize,
        adopted: usize,
        boundary: Option<usize>,
        cursor: usize,
        chunk: usize,
        caps: PrefillCaps,
    ) -> Self {
        let (adopted, boundary, can_pad, pad_mask) =
            plan_metadata(prompt_len, adopted, boundary, caps);
        if cursor <= adopted {
            return Self::with_prefix(prompt_len, adopted, boundary, chunk, caps);
        }

        debug_assert!(cursor <= prompt_len);
        debug_assert!(boundary.is_none_or(|boundary| cursor >= boundary));
        let cursor = cursor.min(prompt_len);
        let chunk = eligible_chunk(chunk, caps);
        let mut pieces = Vec::new();
        let step = chunk.unwrap_or((prompt_len - cursor).max(1));
        append_pieces(&mut pieces, cursor, prompt_len, step, can_pad);
        Self {
            prompt_len,
            adopted,
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

    /// Compute only the piece that would start at the cursor in
    /// Self::resume_at.
    ///
    /// This is the allocation-free form for capacity and reservation checks
    /// that need the next piece's padding but do not execute the full suffix.
    /// It follows the same initial-plan, continuation, boundary and padding
    /// rules as Self::resume_at.
    #[must_use]
    pub fn next_piece_at(
        prompt_len: usize,
        adopted: usize,
        boundary: Option<usize>,
        cursor: usize,
        chunk: usize,
        caps: PrefillCaps,
    ) -> Option<PrefillPiece> {
        let (adopted, boundary, can_pad) = plan_geometry(prompt_len, adopted, boundary, caps);
        if cursor < adopted || cursor >= prompt_len {
            return None;
        }
        if cursor == adopted {
            if let Some(boundary) = boundary {
                return Some(piece_to_boundary(cursor, boundary));
            }
            let remaining = prompt_len - cursor;
            let step = eligible_chunk(chunk, caps)
                .filter(|chunk| remaining > *chunk)
                .unwrap_or(remaining.max(1));
            return piece_from_cursor(cursor, prompt_len, step, can_pad);
        }

        debug_assert!(boundary.is_none_or(|boundary| cursor >= boundary));
        let remaining = prompt_len - cursor;
        let step = eligible_chunk(chunk, caps).unwrap_or(remaining.max(1));
        piece_from_cursor(cursor, prompt_len, step, can_pad)
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

fn plan_metadata(
    prompt_len: usize,
    adopted: usize,
    boundary: Option<usize>,
    caps: PrefillCaps,
) -> (usize, Option<usize>, bool, bool) {
    let (adopted, boundary, can_pad) = plan_geometry(prompt_len, adopted, boundary, caps);
    let pad_mask = !caps.supports_maskless_padded_prefill || force_padded_prefill_array_mask();
    (adopted, boundary, can_pad, pad_mask)
}

fn plan_geometry(
    prompt_len: usize,
    adopted: usize,
    boundary: Option<usize>,
    caps: PrefillCaps,
) -> (usize, Option<usize>, bool) {
    let adopted = adopted.min(prompt_len);
    // Embedding input is consumed whole and never split, so a history
    // boundary does not apply to it: honoring one would cut the embedding
    // rows into two forwards and advance a cursor the executor does not keep.
    let boundary = boundary.filter(|boundary| {
        !caps.is_embedding_input() && *boundary > adopted && *boundary < prompt_len
    });
    let can_pad = caps.can_pad();
    (adopted, boundary, can_pad)
}

fn eligible_chunk(chunk: usize, caps: PrefillCaps) -> Option<usize> {
    (chunk > 0 && caps.supports_chunked_prefill && !caps.is_embedding_input()).then_some(chunk)
}

fn piece_to_boundary(cursor: usize, boundary: usize) -> PrefillPiece {
    PrefillPiece {
        range: cursor..boundary,
        padded_len: boundary - cursor,
    }
}

fn piece_from_cursor(
    cursor: usize,
    prompt_len: usize,
    step: usize,
    can_pad: bool,
) -> Option<PrefillPiece> {
    if cursor >= prompt_len {
        return None;
    }
    let end = cursor.saturating_add(step).min(prompt_len);
    let len = end - cursor;
    Some(PrefillPiece {
        range: cursor..end,
        padded_len: if can_pad { align_to_na_tile(len) } else { len },
    })
}

fn append_pieces(
    pieces: &mut Vec<PrefillPiece>,
    mut cursor: usize,
    prompt_len: usize,
    step: usize,
    can_pad: bool,
) {
    while let Some(piece) = piece_from_cursor(cursor, prompt_len, step, can_pad) {
        cursor = piece.range.end;
        pieces.push(piece);
    }
}

#[cfg(test)]
#[path = "prefill_plan_tests.rs"]
mod tests;
