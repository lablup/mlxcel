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

//! Per-row prefill for the batched Gemma 4 MTP adapter on the row-wise
//! geometries (issue #2190).
//!
//! The per-batch-row verify only matches classic decode if each row's prompt
//! KV is the KV classic serving builds: one row at a time, over the classic
//! chunk partition and history boundary ([`mtp_prefill_ranges`]), with the
//! first token taken from a last-row LM-head projection. A `[B, L]` prefill
//! changes the matmul row count, and a left-padded ragged prefill puts each
//! short row in a shifted RoPE frame no standalone run has. So every row is
//! prefilled on its own cache and the caches are stacked with each row's
//! history first and any shortfall as a zero tail
//! ([`crate::models::gemma4::stack_prefilled_rows`]). A short row then looks
//! exactly like a row whose valid end lags the shared offset after a
//! divergent accept, which the verify and finalize already handle.
use mlxcel_core::drafter::DrafterError;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::sample_token_optimized;
use mlxcel_core::speculative::mtp::target::MtpBatchedVerifyOutput;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{Gemma4MtpBatchedTargetAdapter, Gemma4MtpTargetAdapter, mtp_prefill_ranges};
use crate::models::gemma4::{Gemma4SpeculativeSinks, stack_prefilled_rows};

/// One prefilled row before stacking.
struct PrefilledRow {
    caches: Vec<crate::models::gemma4::Cache>,
    bonus: i32,
    hidden: UniquePtr<MlxArray>,
    shared_kv: Vec<UniquePtr<MlxArray>>,
}

impl<'a> Gemma4MtpBatchedTargetAdapter<'a> {
    /// Prefill every row on its own cache over the classic partition, stack
    /// the caches into the adapter's `[B, ...]` cache and seed the drafter.
    pub(super) fn prefill_and_seed_rows(
        &self,
        prompt_tokens_per_row: &[Vec<i32>],
        sampler: &SamplingConfig,
    ) -> Result<(Vec<i32>, MtpBatchedVerifyOutput), DrafterError> {
        if prompt_tokens_per_row.len() != self.batch_size {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Gemma4 batched MTP target: expected {} rows, got {}",
                    self.batch_size,
                    prompt_tokens_per_row.len()
                ),
            });
        }
        let mut rows = Vec::with_capacity(self.batch_size);
        for (r, prompt) in prompt_tokens_per_row.iter().enumerate() {
            if prompt.is_empty() {
                return Err(DrafterError::DraftFailed {
                    reason: format!("Gemma4 batched MTP target: prompt row {r} must be non-empty"),
                });
            }
            let boundary = self.prefill_boundaries.get(r).copied().flatten();
            rows.push(self.prefill_row(prompt, boundary, sampler));
        }

        let lengths: Vec<usize> = prompt_tokens_per_row.iter().map(Vec::len).collect();
        let bonuses: Vec<i32> = rows.iter().map(|row| row.bonus).collect();
        let hidden = concat_axis0(rows.iter().map(|row| &row.hidden));
        let slab_count = rows[0].shared_kv.len();
        if rows.iter().any(|row| row.shared_kv.len() != slab_count) {
            return Err(DrafterError::DraftFailed {
                reason: "Gemma4 batched MTP target: rows captured different shared K/V slabs"
                    .to_string(),
            });
        }
        let shared_kv: Vec<UniquePtr<MlxArray>> = (0..slab_count)
            .map(|slot| right_pad_and_stack(rows.iter().map(|row| &row.shared_kv[slot])))
            .collect();
        let caches = stack_prefilled_rows(rows.into_iter().map(|row| row.caches).collect())
            .map_err(|reason| DrafterError::DraftFailed {
                reason: format!(
                    "Gemma4 batched MTP target: cannot batch these prompts exactly ({reason}); \
                     falling back to per-row service"
                ),
            })?;
        *self.caches.borrow_mut() = caches;
        self.enable_rotating_cache_buffer();

        // Every row's history starts at slot 0, so there is no left padding;
        // a short row's valid length lags the shared offset instead.
        let left_padding = vec![0; self.batch_size];
        *self.positions.borrow_mut() = lengths.clone();
        *self.left_padding.borrow_mut() = left_padding.clone();
        let next_hidden = self.wrapper.speculative_draft_hidden(&hidden);
        let seed = self.build_seed_metadata(next_hidden, shared_kv, &lengths, &left_padding);
        Ok((bonuses, seed))
    }

    /// The B = 1 linear adapter's row-wise prefill, against a fresh cache.
    fn prefill_row(
        &self,
        prompt: &[i32],
        boundary: Option<usize>,
        sampler: &SamplingConfig,
    ) -> PrefilledRow {
        let mut caches = self.wrapper.make_speculative_caches();
        let mut sinks = Gemma4SpeculativeSinks::with_hidden_and_shared_kv();
        let ranges = mtp_prefill_ranges(0, prompt.len(), self.prefill_chunk_size, boundary);
        let mut chunks = ranges.into_iter().map(|range| &prompt[range]).peekable();
        let mut last = None;
        while let Some(chunk) = chunks.next() {
            let input = mlxcel_core::from_slice_i32(chunk, &[1, chunk.len() as i32]);
            let capture = chunks.peek().is_none().then_some(&mut sinks);
            let logits =
                self.wrapper
                    .prefill_mtp_chunk_explicit_cache(&input, &mut caches, capture);
            mlxcel_core::eval(&logits);
            last = Some(logits);
        }
        let logits = last.expect("a non-empty prompt has at least one prefill range");
        let (token, _) = sample_token_optimized(&logits, sampler, &[]);
        mlxcel_core::eval(&token);
        let bonus = mlxcel_core::item_i32(&token);

        let hidden_full = sinks
            .hidden_sink
            .and_then(|sink| sink.into_iter().next_back())
            .expect("the final prefill range captures the last hidden state");
        let hidden = Gemma4MtpTargetAdapter::last_position_hidden(&hidden_full);
        let shared_kv_map = sinks
            .shared_kv_sink
            .expect("with_hidden_and_shared_kv installs the shared K/V sink");
        let mut shared_kv = Vec::with_capacity(4);
        for kind in ["full_attention", "sliding_attention"] {
            if let Some((keys, values)) = shared_kv_map.get(kind) {
                shared_kv.push(mlxcel_core::copy(keys));
                shared_kv.push(mlxcel_core::copy(values));
            }
        }
        PrefilledRow {
            caches,
            bonus,
            hidden,
            shared_kv,
        }
    }
}

fn concat_axis0<'b>(arrays: impl Iterator<Item = &'b UniquePtr<MlxArray>>) -> UniquePtr<MlxArray> {
    let arrays: Vec<&MlxArray> = arrays.map(|a| a.as_ref().expect("non-null")).collect();
    mlxcel_core::concatenate_many(&arrays, 0)
}

/// Stack `[1, H, S_r, D]` slabs on axis 0 after padding each to the longest
/// `S` with a zero tail, the layout the drafter's `kv_valid_len` mask expects
/// when `left_padding` is zero.
fn right_pad_and_stack<'b>(
    slabs: impl Iterator<Item = &'b UniquePtr<MlxArray>> + Clone,
) -> UniquePtr<MlxArray> {
    let width = slabs
        .clone()
        .map(|s| mlxcel_core::array_shape(s)[2])
        .max()
        .unwrap_or(0);
    let padded: Vec<UniquePtr<MlxArray>> = slabs
        .map(|slab| {
            let shape = mlxcel_core::array_shape(slab);
            if shape[2] == width {
                return mlxcel_core::copy(slab);
            }
            let zeros = mlxcel_core::zeros(
                &[shape[0], shape[1], width - shape[2], shape[3]],
                mlxcel_core::array_dtype(slab),
            );
            mlxcel_core::concatenate(slab, &zeros, 2)
        })
        .collect();
    concat_axis0(padded.iter())
}

#[cfg(test)]
#[path = "gemma4_mtp_target_rows_tests.rs"]
mod tests;
