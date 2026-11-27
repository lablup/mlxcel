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
//! A drafter may score a reduced draft vocabulary (issue #3261, upstream
//! `VanillaMarkov(vocab_size, rank, draft_vocab_size)`). `markov_w1` is then
//! still indexed by target ids (`[vocab, rank]`, `prev` is always a target
//! id), `markov_w2` projects to the draft vocabulary (`[draft_vocab, rank]`),
//! and each chosen draft id `t` leaves the chain as the target id
//! `t + d2t[t]` ([`DraftToTarget`]), which is also the next step's `prev`.
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
    /// `[vocab, rank]` token-to-rank embedding table, indexed by target ids.
    w1: UnifiedEmbedding,
    /// `rank -> draft_vocab` projection, stored `[draft_vocab, rank]`.
    w2: UnifiedLinear,
    /// Rank of the factorization, from the checkpoint config.
    rank: usize,
    /// Draft-id to target-id offsets of a reduced draft vocabulary; `None`
    /// when a draft id is already a target id.
    d2t: Option<DraftToTarget>,
}

/// The `d2t` table of a reduced-vocabulary DSpark drafter (issue #3261): the
/// target id of draft id `t` is `t + d2t[t]`.
///
/// Built only through [`DraftToTarget::from_offsets`], which is what makes
/// the gather in [`Self::to_target_ids`] safe: every mapped id is checked
/// against the target vocabulary once, at load, because MLX range-checks no
/// positive gather index and the mapped id indexes the target's embedding
/// table and `markov_w1` on the very next step.
pub struct DraftToTarget {
    /// `[draft_vocab]` int32 offsets on the device.
    offsets: UniquePtr<MlxArray>,
    /// Number of draft ids the table covers.
    draft_vocab: usize,
}

impl DraftToTarget {
    /// Validate `offsets` (one per draft id) against the two vocabularies and
    /// move them to the device.
    ///
    /// Returns `Ok(None)` when every offset is zero: the mapping is then the
    /// identity, and the chain skips the gather. One pass over the table.
    pub fn from_offsets(
        offsets: &[i32],
        draft_vocab: usize,
        vocab: usize,
    ) -> Result<Option<Self>, String> {
        if offsets.len() != draft_vocab {
            return Err(format!(
                "d2t has {} entries but the drafter config declares draft_vocab_size \
                 {draft_vocab}; the table needs one offset per draft id",
                offsets.len()
            ));
        }
        let mut identity = true;
        for (draft_id, &offset) in offsets.iter().enumerate() {
            identity &= offset == 0;
            // i64 so neither a negative offset nor a sum past `i32::MAX` wraps
            // into the valid range.
            let target = draft_id as i64 + i64::from(offset);
            if target < 0 || target >= vocab as i64 {
                return Err(format!(
                    "d2t maps draft id {draft_id} to target id {target} (offset {offset}), \
                     outside the target vocabulary [0, {vocab}); the mapped id indexes the \
                     target embedding table and MLX range-checks no positive gather index"
                ));
            }
        }
        if identity {
            return Ok(None);
        }
        let len = i32::try_from(draft_vocab)
            .map_err(|_| format!("draft_vocab_size {draft_vocab} does not fit an i32 shape"))?;
        Ok(Some(Self {
            offsets: ffi::from_slice_i32(offsets, &[len]),
            draft_vocab,
        }))
    }

    /// Read the checkpoint's `d2t` tensor and build the table.
    ///
    /// The tensor may be stored in any integer dtype (upstream casts it to
    /// int32 in `sanitize`); a floating-point `d2t` is refused rather than
    /// truncated.
    pub fn from_tensor(
        tensor: &MlxArray,
        draft_vocab: usize,
        vocab: usize,
    ) -> Result<Option<Self>, String> {
        let shape = ffi::array_shape(tensor);
        if shape.len() != 1 {
            return Err(format!(
                "d2t must be a 1-D [draft_vocab] table, got shape {shape:?}"
            ));
        }
        let dtype = ffi::array_dtype(tensor);
        let integer = matches!(
            dtype,
            crate::dtype::UINT8
                | crate::dtype::UINT16
                | crate::dtype::UINT32
                | crate::dtype::UINT64
                | crate::dtype::INT8
                | crate::dtype::INT16
                | crate::dtype::INT32
                | crate::dtype::INT64
        );
        if !integer {
            return Err(format!(
                "d2t must hold integer offsets, got a tensor of dtype code {dtype}"
            ));
        }
        let as_i32 = ffi::astype(tensor, crate::dtype::INT32);
        ffi::eval(&as_i32);
        let offsets = super::materialize_argmax_i32_vec(&as_i32, shape[0].max(0) as usize);
        Self::from_offsets(&offsets, draft_vocab, vocab)
    }

    /// Number of draft ids the table covers.
    pub fn draft_vocab(&self) -> usize {
        self.draft_vocab
    }

    /// Map draft ids (any shape, int32, each in `[0, draft_vocab)`) to
    /// target ids on the device: `t + d2t[t]`.
    pub fn to_target_ids(&self, draft_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let offsets = ffi::take(&self.offsets, draft_ids, 0);
        ffi::add(draft_ids, &offsets)
    }
}

impl VanillaMarkovHead {
    /// Load `{prefix}.markov_w1` and `{prefix}.markov_w2` (`prefix` is
    /// `markov_head` on the published checkpoints). Quantized siblings
    /// (`.scales` / `.biases`) load through the same unified loaders the
    /// backbone uses.
    ///
    /// [`Self::from_weights_with_draft_vocab`] for a head that scores the
    /// whole target vocabulary: both factors have `vocab` rows and a draft id
    /// is a target id.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        vocab: usize,
        rank: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Self::from_weights_with_draft_vocab(
            weights, prefix, vocab, vocab, rank, None, group_size, bits,
        )
    }

    /// Load `{prefix}.markov_w1` (`[vocab, rank]`) and `{prefix}.markov_w2`
    /// (`[draft_vocab, rank]`).
    ///
    /// Both factors are measured against the config before either is built,
    /// because neither shape is checked anywhere downstream: `markov_w1` is
    /// gathered at a target token id once per chain step and MLX range-checks
    /// no positive gather index, so a table with fewer rows than `vocab`
    /// reads past its own buffer into the logits; a `markov_w2` whose row
    /// count is not `draft_vocab` throws inside MLX when its output is added
    /// to the draft logits, which crosses the cxx bridge as a process abort
    /// rather than a load error.
    ///
    /// `draft_vocab` is the vocabulary the draft head scores
    /// (`DFlashConfig::draft_vocab`), equal to `vocab` unless the checkpoint
    /// declares a reduced one; `d2t` is the matching id table, already
    /// validated ([`DraftToTarget`]).
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights_with_draft_vocab(
        weights: &WeightMap,
        prefix: &str,
        vocab: usize,
        draft_vocab: usize,
        rank: usize,
        d2t: Option<DraftToTarget>,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let w1_key = format!("{prefix}.markov_w1");
        let w2_key = format!("{prefix}.markov_w2");
        validate_markov_factor(weights, &w1_key, vocab, rank)?;
        validate_markov_factor(weights, &w2_key, draft_vocab, rank)?;
        if let Some(table) = &d2t
            && table.draft_vocab() != draft_vocab
        {
            return Err(format!(
                "d2t covers {} draft ids but {w2_key} scores {draft_vocab}",
                table.draft_vocab()
            ));
        }
        let w1 = UnifiedEmbedding::from_weights(weights, &w1_key, group_size, bits)?;
        let w2 = UnifiedLinear::from_weights(weights, &w2_key, group_size, bits)?;
        Ok(Self { w1, w2, rank, d2t })
    }

    /// Whether chosen draft ids are remapped through a `d2t` table.
    pub fn remaps_draft_ids(&self) -> bool {
        self.d2t.is_some()
    }

    /// Rank of the transition factorization.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Transition bias for one previous token: `[1, 1] -> [1, 1, draft_vocab]`.
    ///
    /// `prev` is a TARGET id (the anchor, or a proposal already mapped
    /// through `d2t`), as a host-built id array or a device-side result;
    /// either way the gather runs on the device.
    pub fn transition_logits(&self, prev: &MlxArray) -> UniquePtr<MlxArray> {
        let embedded = self.w1.forward(prev);
        self.w2.forward(&embedded)
    }

    /// Chain `base_logits` (`[1, gamma, draft_vocab]`) into `gamma`
    /// proposals, starting from `anchor`, as a `[1, gamma]` int32 device
    /// array of TARGET ids.
    ///
    /// Step `i` chooses the draft id `argmax(base_logits[:, i] +
    /// bias(prev_i))`, maps it to a target id (`t + d2t[t]`, the identity
    /// without a table) and proposes that, with `prev_0 = anchor` and
    /// `prev_{i + 1}` the step-`i` proposal. Every position of `base_logits`
    /// is consumed, including position 0, which a plain DFlash draft would
    /// drop as the bonus scaffold: in the DSpark layout position 0 already
    /// sits past the anchor. `gamma == 0` yields an empty `[1, 0]` array.
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
        if gamma <= 0 {
            return ffi::zeros(&[1, 0], crate::dtype::INT32);
        }

        let mut prev = ffi::from_slice_i32(&[anchor], &[1, 1]);
        let mut steps: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(gamma as usize);
        for step in 0..gamma {
            let bias = self.transition_logits(&prev);
            let step_logits = ffi::slice(base_logits, &[0, step, 0], &[1, step + 1, vocab]);
            let combined = ffi::add(&step_logits, &bias);
            // `[1, 1, vocab] -> [1, 1]`; cast so the gather index and the
            // concatenated proposal row share one integer dtype.
            let chosen = ffi::astype(&ffi::argmax_last_axis(&combined), crate::dtype::INT32);
            // The draft head's id space ends here: everything downstream
            // (the next step's `markov_w1` gather, the verify input, the
            // walk against the target argmax) reads target ids.
            let chosen = match &self.d2t {
                Some(table) => table.to_target_ids(&chosen),
                None => chosen,
            };
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
/// count reads the same either way while its stored width is `rank * bits /
/// 32`. The row check therefore runs unchanged for a packed table, and the
/// rank check runs against the bit depths that could produce the stored width
/// rather than against the rank directly.
///
/// Used by: [`VanillaMarkovHead::from_weights_with_draft_vocab`].
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
    if weights.contains_key(&format!("{key}.scales")) {
        // Quantized: MLX packs the last axis u32-wise, so the stored width is
        // `rank * bits / 32` (`packed_in * 32 == bits * in_features`, the
        // invariant `infer_quantization_bits` in `layers.rs` solves). The
        // width is still checkable, just not directly: accept it when some
        // supported bit depth explains it. Testing against the depths rather
        // than against the caller's declared `bits` is deliberate, because a
        // per-tensor override may store this factor at a different depth from
        // the rest of the checkpoint, and a false load failure on a valid
        // checkpoint is worse than the narrower bound. Skipping the check
        // entirely is what is not acceptable: a `markov_w2` whose width does
        // not match `markov_w1`'s output reaches `quantized_matmul`, which
        // throws inside MLX and crosses the cxx bridge as a process abort
        // rather than a load error, and every mlx-community conversion of
        // these drafters is quantized.
        let packed_bits = i64::from(*cols) * 32;
        let explained = crate::layers::SUPPORTED_AFFINE_BITS
            .iter()
            .any(|&b| packed_bits == i64::from(b) * rank as i64);
        if !explained {
            return Err(format!(
                "{name} is a quantized [{rows}, {cols}] table, and no supported bit depth makes \
                 {cols} packed columns describe the markov_rank {rank} the drafter config \
                 declares"
            ));
        }
    } else if usize::try_from(*cols).ok() != Some(rank) {
        return Err(format!(
            "{name} is [{rows}, {cols}] but the drafter config declares markov_rank {rank}"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "markov_tests.rs"]
mod tests;
