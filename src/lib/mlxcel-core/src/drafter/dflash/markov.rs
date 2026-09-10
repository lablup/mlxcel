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

//! [`VanillaMarkovHead`]: the DSpark low-rank token-transition head
//! (issue #1339).
//!
//! A DSpark drafter runs one non-causal DFlash forward over the block and
//! gets `gamma` independent logit rows out of it. On their own those rows
//! know nothing about each other: position `i + 1` was scored without seeing
//! what position `i` will be. The Markov head turns them into a sequential
//! chain. For each step it adds a learned transition bias
//! `markov_w2(markov_w1[prev])` to the step's base logits, where `prev` is
//! the token the chain just emitted (the anchor for step 0), and the argmax
//! of the sum becomes the proposal and the next `prev`.
//!
//! `markov_w1` is a `[vocab, rank]` embedding table and `markov_w2` a
//! `rank -> vocab` linear layer stored `[vocab, rank]`; the published
//! checkpoints use rank 256, so the bias costs one gather and one
//! `[1, 256] x [256, vocab]` matmul per step.
//!
//! The chain is built as one lazy MLX graph and materialized once per round:
//! every step's argmax stays on the device and is fed straight into the next
//! step's gather, so a round costs one host synchronization rather than
//! `gamma`.

use crate::ffi::{self, MlxArray};
use crate::layers::{UnifiedEmbedding, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;

/// DSpark vanilla Markov head: `bias(prev) = markov_w2(markov_w1[prev])`.
pub struct VanillaMarkovHead {
    /// `[vocab, rank]` token-to-rank embedding table.
    w1: UnifiedEmbedding,
    /// `rank -> vocab` projection, stored `[vocab, rank]`.
    w2: UnifiedLinear,
    /// Rank of the factorization, from the checkpoint config.
    rank: usize,
}

impl VanillaMarkovHead {
    /// Load `{prefix}.markov_w1` and `{prefix}.markov_w2` (`prefix` is
    /// `markov_head` on the published checkpoints). Quantized siblings
    /// (`.scales` / `.biases`) load through the same unified loaders the
    /// backbone uses.
    ///
    /// Both factors are measured against the config's `vocab` and `rank`
    /// before either is built, because neither shape is checked anywhere
    /// downstream: `markov_w1` is gathered at a token id once per chain step
    /// and MLX range-checks no positive gather index, so a table with fewer
    /// rows than the vocabulary the chain draws from reads past its own
    /// buffer into the logits; a `markov_w2` of the wrong width throws inside
    /// MLX instead, which crosses the cxx bridge as a process abort rather
    /// than a load error.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        vocab: usize,
        rank: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let w1_key = format!("{prefix}.markov_w1");
        let w2_key = format!("{prefix}.markov_w2");
        validate_markov_factor(weights, &w1_key, vocab, rank)?;
        validate_markov_factor(weights, &w2_key, vocab, rank)?;
        let w1 = UnifiedEmbedding::from_weights(weights, &w1_key, group_size, bits)?;
        let w2 = UnifiedLinear::from_weights(weights, &w2_key, group_size, bits)?;
        Ok(Self { w1, w2, rank })
    }

    /// Rank of the transition factorization.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Transition bias for one previous token: `[1, 1] -> [1, 1, vocab]`.
    ///
    /// `prev` may be a host-built id array or a device-side argmax result;
    /// either way the gather runs on the device.
    pub fn transition_logits(&self, prev: &MlxArray) -> UniquePtr<MlxArray> {
        let embedded = self.w1.forward(prev);
        self.w2.forward(&embedded)
    }

    /// Chain `base_logits` (`[1, gamma, vocab]`) into `gamma` proposals,
    /// starting from `anchor`, as a `[1, gamma]` int32 device array.
    ///
    /// Step `i` proposes `argmax(base_logits[:, i] + bias(prev_i))` with
    /// `prev_0 = anchor` and `prev_{i + 1}` the step-`i` proposal. Every
    /// position of `base_logits` is consumed, including position 0, which a
    /// plain DFlash draft would drop as the bonus scaffold: in the DSpark
    /// layout position 0 already sits past the anchor.
    pub fn sample_block_array(&self, base_logits: &MlxArray, anchor: i32) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(base_logits);
        debug_assert_eq!(
            shape.len(),
            3,
            "Markov head expects [1, gamma, vocab] base logits, got {shape:?}"
        );
        // The chain is sequential and seeded from ONE anchor, so it is B = 1
        // by construction; the per-step slice below would silently read row 0
        // and drop the rest. The batched DFlash drafter path refuses a DSpark
        // drafter for the same reason.
        debug_assert_eq!(
            shape[0], 1,
            "Markov head is B = 1 only, got batch {}",
            shape[0]
        );
        let gamma = shape[1];
        let vocab = shape[2];

        let mut prev = ffi::from_slice_i32(&[anchor], &[1, 1]);
        let mut steps: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(gamma as usize);
        for step in 0..gamma {
            let bias = self.transition_logits(&prev);
            let step_logits = ffi::slice(base_logits, &[0, step, 0], &[1, step + 1, vocab]);
            let combined = ffi::add(&step_logits, &bias);
            // `[1, 1, vocab] -> [1, 1]`; cast so the gather index and the
            // concatenated proposal row share one integer dtype.
            let chosen = ffi::astype(&ffi::argmax_last_axis(&combined), crate::dtype::INT32);
            prev = ffi::copy(&chosen);
            steps.push(chosen);
        }
        let refs: Vec<&MlxArray> = steps
            .iter()
            .map(|s| s.as_ref().expect("markov step array"))
            .collect();
        crate::ops::concatenate_many(&refs, 1)
    }

    /// Host-side variant of [`Self::sample_block_array`]: the `gamma`
    /// proposal ids, materialized with one device-to-host copy.
    pub fn sample_block(&self, base_logits: &MlxArray, anchor: i32) -> Vec<i32> {
        let gamma = ffi::array_shape(base_logits)[1] as usize;
        let proposals = self.sample_block_array(base_logits, anchor);
        super::materialize_argmax_i32_vec(&proposals, gamma)
    }
}

/// Check one `[vocab, rank]` Markov factor against the config that sizes it,
/// before the loader turns it into a layer.
///
/// A quantized factor bit-packs `rank` along its last axis only, so its row
/// count reads the same either way and its stored width is a function of the
/// bit depth rather than the rank; the rank check is therefore skipped for a
/// packed table and the row check is not.
///
/// Used by: [`VanillaMarkovHead::from_weights`].
fn validate_markov_factor(
    weights: &WeightMap,
    key: &str,
    vocab: usize,
    rank: usize,
) -> Result<(), String> {
    let name = format!("{key}.weight");
    let tensor = weights
        .get(&name)
        .ok_or_else(|| format!("Weight not found: {name}"))?;
    let shape = ffi::array_shape(tensor);
    let [rows, cols] = shape.as_slice() else {
        return Err(format!(
            "{name} must be a 2-D [vocab, rank] table, got shape {shape:?}"
        ));
    };
    if usize::try_from(*rows).ok() != Some(vocab) {
        return Err(format!(
            "{name} has {rows} rows but the drafter config declares vocab_size {vocab}; the \
             Markov chain gathers this table at a token id and MLX range-checks no positive \
             gather index"
        ));
    }
    if !weights.contains_key(&format!("{key}.scales")) && usize::try_from(*cols).ok() != Some(rank)
    {
        return Err(format!(
            "{name} is [{rows}, {cols}] but the drafter config declares markov_rank {rank}"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "markov_tests.rs"]
mod tests;
