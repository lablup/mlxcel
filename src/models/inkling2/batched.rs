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

//! Native batched forward for Inkling 2 (issue #3377).
//!
//! Every row keeps its own per-visit `InklingLayerCache`. One forward runs
//! over `[B, L]`: the projections, convolutions, residual mixes, loop
//! messages and experts see the whole batch, and only the attention core
//! (relative mask, sliding window, sink, log scaling) runs per row against
//! that row's KV cache, as Inkling 1's batched decode does (#2673).
//!
//! A convolution stacks the rows' f32 histories, left-padded with zero rows
//! to the longest one, and gives each row back exactly the history its
//! single-row forward would have kept. The padding never enters a
//! convolution: every history holds at least `kernel - 1` rows, and only the
//! newest `kernel - 1` are read.
//!
//! A forward of more than one token runs inside
//! [`super::conv::WideHistoryScope`], so the server's padded prefill cohort
//! (right-padded rows, trimmed per row afterwards through `trim_state`) can
//! remove any amount of padding exactly.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::inkling_layer::InklingLayerCache;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::Inkling2Model;
use super::conv::{WideHistoryScope, history_margin};

/// Run `f` on the `slot` convolution state of every row at once.
///
/// One row hands its own state to `f` untouched, so the B = 1 graph (and the
/// fused kernel's gate) is exactly the single-sequence one. Several rows are
/// stacked to an f32 `[B, R_max, C]` history; afterwards row `i` keeps the
/// newest `min(kernel - 1 + margin, R_i + L)` rows of the stacked next
/// history, which is the history its own forward keeps. `x` is the
/// convolution input `[B, L, C]`.
pub(crate) fn with_stacked_state<R>(
    rows: &mut [&mut InklingLayerCache],
    slot: usize,
    kernel: i32,
    x: &MlxArray,
    f: impl FnOnce(&mut Option<UniquePtr<MlxArray>>) -> R,
) -> R {
    if let [row] = rows {
        return f(&mut row.conv[slot]);
    }
    let shape = mlxcel_core::array_shape(x);
    let (length, channels) = (shape[1], shape[2]);
    let width = kernel - 1;
    let histories: Vec<_> = rows
        .iter()
        .map(
            |row| match row.conv[slot].as_ref().and_then(|value| value.as_ref()) {
                Some(history) => mlxcel_core::copy(history),
                None => mlxcel_core::zeros(&[1, width, channels], dtype::FLOAT32),
            },
        )
        .collect();
    let lens: Vec<i32> = histories
        .iter()
        .map(|history| mlxcel_core::array_shape(history)[1])
        .collect();
    let longest = lens.iter().copied().max().unwrap_or(width);
    let padded: Vec<_> = histories
        .iter()
        .zip(&lens)
        .map(|(history, &held)| {
            if held == longest {
                mlxcel_core::copy(history)
            } else {
                let pad = mlxcel_core::zeros(&[1, longest - held, channels], dtype::FLOAT32);
                mlxcel_core::concatenate(&pad, history, 1)
            }
        })
        .collect();
    let refs: Vec<&MlxArray> = padded.iter().map(|history| &**history).collect();
    let mut state = Some(mlxcel_core::concatenate_many(&refs, 0));
    let out = f(&mut state);
    let next = state.expect("a convolution always leaves its next history");
    let kept = mlxcel_core::array_shape(&next)[1];
    let limit = width + history_margin(length);
    for (i, row) in rows.iter_mut().enumerate() {
        let keep = limit.min(lens[i] + length);
        let i = i as i32;
        row.conv[slot] = Some(mlxcel_core::contiguous(
            &mlxcel_core::slice(&next, &[i, kept - keep, 0], &[i + 1, kept, channels]),
            false,
        ));
    }
    out
}

impl Inkling2Model {
    /// Logits `[B, L, vocab]` of one forward over every row of `input`, each
    /// against the model-owned state of its sequence id. A row without state
    /// yet (a cold prefill) gets fresh caches first.
    pub(super) fn forward_rows(&self, input: &MlxArray, ids: &[SequenceId]) -> UniquePtr<MlxArray> {
        for &id in ids {
            if !self.sequence_state.has_sequence_state(id) {
                self.sequence_state
                    .prepare_sequence_state(id, self.make_internal_caches());
            }
        }
        let length = mlxcel_core::array_shape(input)[1];
        let _wide = (length > 1).then(WideHistoryScope::enter);
        self.sequence_state
            .with_batched_sequence_states(ids, |states| {
                let mut rows: Vec<&mut [InklingLayerCache]> =
                    states.iter_mut().map(Vec::as_mut_slice).collect();
                let embeddings = self.normalized_input_embeddings(input);
                let (hidden, _) = self.hidden_rows(&embeddings, &mut rows, &[]);
                self.project(&hidden)
            })
            .unwrap_or_else(|error| panic!("Inkling 2 batched forward: {error}"))
    }
}
