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

//! How many positions the sequence currently being prefilled will span.
//!
//! A prompt longer than one prefill chunk reaches the model as several
//! consecutive `forward` calls that continue from the same KV cache. Each of
//! them sees only its own `(cache_offset, chunk_len)`, and for every RoPE
//! variant whose frequency table is fixed at load time that is all it needs.
//! It is not enough for a variant that *picks* a table from how long the whole
//! prompt is.
//!
//! Phi-3 / Phi-4 LongRoPE is the case that forced this module. HuggingFace
//! transformers picks the `short_factor` table while the whole prompt fits in
//! `original_max_position_embeddings` and the `long_factor` table above it, and
//! it never sees the question per chunk because it prefills the entire prompt
//! in one pass. Deciding per chunk instead writes keys rotated with two
//! different tables into one cache: a 5136-token prompt at the CLI default
//! 2048-token chunk takes the short table for chunks 1 and 2 (their last
//! position is 4095, still inside the trained 4096 context) and the long table
//! for chunk 3, and the generated text degenerates into repetition.
//!
//! So the driver that splits a prefill announces the total length of the
//! sequence it is feeding, and a model that needs it reads the announcement
//! instead of its own chunk geometry.
//!
//! # Scope
//!
//! The announcement is thread-local and lives exactly as long as the returned
//! [`PrefillSpan`] guard. That matters for the server, where the scheduler
//! interleaves decode batches for other sequences between two chunks of one
//! prompt: a guard held across a tick boundary would hand another sequence's
//! decode step this prompt's length. Hold it around the chunk's forward call
//! and nothing else. Model forwards run on the thread that drives them
//! (`KVCache` is deliberately neither `Send` nor `Sync`), so a thread-local is
//! the right width.
//!
//! Every single-pass prefill can skip the announcement: there `cache_offset +
//! seq_len` already equals the sequence length, which is what the reader falls
//! back to.
//!
//! Used by: the CLI/bench chunked prefill in [`crate::generate`], the server
//! batch scheduler's chunked prefill, and Phi-3 / Phi-4 LongRoPE table
//! selection.

use std::cell::Cell;

thread_local! {
    /// Total positions of the sequence whose prefill is running on this thread.
    static SPAN: Cell<Option<i32>> = const { Cell::new(None) };
}

/// Live announcement of the current prefill's total position span.
///
/// Restores whatever was announced before it on drop, so nesting an inner
/// prefill inside an outer one cannot leave the outer one reading the inner
/// value.
#[derive(Debug)]
pub struct PrefillSpan {
    previous: Option<i32>,
}

impl Drop for PrefillSpan {
    fn drop(&mut self) {
        SPAN.with(|span| span.set(self.previous));
    }
}

/// Announce that the forward calls made while the returned guard is alive are
/// feeding a sequence `total_positions` tokens long.
///
/// `total_positions` counts the whole sequence, including any prefix already
/// resident in the KV cache from a prefix-cache hit or an earlier turn, because
/// that prefix occupies positions too.
///
/// A non-positive length announces nothing: it can only come from a caller that
/// has no prompt to feed, and letting it through would claim a sequence shorter
/// than the chunk being fed.
#[must_use = "the announcement ends as soon as the guard is dropped"]
pub fn announce(total_positions: i32) -> PrefillSpan {
    let previous = SPAN.with(|span| span.replace((total_positions > 0).then_some(total_positions)));
    PrefillSpan { previous }
}

/// The announced span, or `None` when no chunked prefill is in flight on this
/// thread.
///
/// A reader should take the larger of this and its own `cache_offset +
/// seq_len` rather than trusting it outright. The two agree on every correct
/// announcement, and taking the maximum means a driver that under-announces
/// still cannot make a model treat a long sequence as a short one.
#[must_use]
pub fn current() -> Option<i32> {
    SPAN.with(Cell::get)
}

#[cfg(test)]
#[path = "prefill_span_tests.rs"]
mod tests;
