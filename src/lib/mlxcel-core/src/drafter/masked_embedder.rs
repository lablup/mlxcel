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

//! Centroid-routed sparse softmax LM head used by Gemma 4 **E2B / E4B**
//! assistant drafters. Port of upstream
//! `references/mlx-vlm/mlx_vlm/speculative/drafters/gemma4_assistant/masked_embedder.py`.
//!
//! ## What it does
//!
//! Rather than computing the full dense `hidden @ embed.T` over the 262144-token
//! Gemma 4 vocabulary (expensive for a 4-layer drafter with `hidden_size=256`),
//! the drafter learns a `centroids` linear layer that scores `num_centroids`
//! (default 2048) token clusters and a `token_ordering` buffer that maps each
//! cluster id to a contiguous block of canonical token ids. At inference, the
//! top-K (default 32) clusters' tokens (~4096 of 262144) are materialised and
//! scored densely; the rest of the vocab is filled with a sentinel
//! `min(selected) - 1` so non-selected positions lose any argmax / sampling
//! competition.
//!
//! For the dense-LM-head variants (26B-A4B, 31B) `MaskedEmbedder` is **not**
//! used — those tie weights into the regular dense LM head. Only the E2B / E4B
//! family routes through this module.
//!
//! ## Forward algorithm
//!
//! Given `hidden_states: [B, L, hidden_size]` and the tied LM-head weight
//! `lm_head_weight: [vocab_size, hidden_size]`:
//!
//! ```text
//! 1. centroid_logits = centroids(hidden_states)               # [B, L, num_centroids]
//! 2. topk_idx        = argpartition(... , kth=num_centroids-top_k, -1)[..., -top_k:]
//!                                                              # [B, L, top_k]
//! 3. ordering        = token_ordering.reshape(num_centroids, vsc)
//! 4. selected_canon  = take(ordering, topk_idx, axis=0)        # [B, L, top_k, vsc]
//! 5. flat_idx        = selected_canon.reshape(-1)
//! 6. selected_emb    = take(lm_head_weight, flat_idx, 0)
//!                       .reshape(B, L, top_k*vsc, hidden)
//! 7. selected_logits = squeeze(hidden[..,None,:] @ selected_emb.T, -2)
//!                                                              # [B, L, top_k*vsc]
//! 8. mask_value      = min(selected_logits) - 1
//! 9. out             = full([B, L, vocab_size], mask_value, hidden.dtype)
//! 10. scatter_idx    = selected_canon.reshape(B, L, -1)
//! 11. return put_along_axis(out, scatter_idx, selected_logits, axis=-1)
//! ```
//!
//! `vsc = vocab_size / num_centroids` is the number of tokens per centroid
//! (`262144 / 2048 = 128` for the canonical Gemma 4 E2B/E4B drafter).
//!
//! ## Apple Silicon precision
//!
//! Centroid weights, `selected_emb` rows, and the sparse softmax dispatch all
//! preserve the model's native bf16/f16. The only host-side scalar is the
//! mask value extracted via `item_f32` on the `min` reduction; it is then
//! immediately cast back to `hidden_states.dtype` by `full_f32`. There is no
//! silent f32 promotion in the matmul or scatter paths.

use crate::ffi;
use crate::ffi::MlxArray;
use crate::layers::Linear;
use crate::weights::WeightMap;
use cxx::UniquePtr;

/// Canonical defaults for Gemma 4 E2B / E4B drafters
/// (`use_ordered_embeddings=True`, `num_centroids=2048`, `top_k=32`,
/// matching `Gemma4AssistantConfig`).
pub const DEFAULT_NUM_CENTROIDS: usize = 2048;
pub const DEFAULT_TOP_K: usize = 32;

/// Weight key under which the centroid token ordering is stored in the
/// assistant drafter checkpoint. Exposed as a constant so the
/// [`sanitize_token_ordering`] hook and the `Gemma4AssistantDraftModel`
/// (sub-3) can refer to the same name without drift.
pub const TOKEN_ORDERING_KEY: &str = "masked_embedding.token_ordering";

/// Centroid-routed sparse softmax LM head.
///
/// Mirrors HF's `Gemma4AssistantMaskedEmbedder` and the MLX port in
/// `masked_embedder.py`. The forward signature takes the tied LM-head weight
/// from the caller (i.e. the drafter's `embed_tokens.weight`) because Gemma 4
/// ties the LM head to the input embedding — the drafter rides on that tied
/// weight for the centroid head.
pub struct MaskedEmbedder {
    /// `[hidden_size, num_centroids]` linear layer that scores centroid
    /// logits per hidden position. Built from `masked_embedding.centroids.*`
    /// weights via [`Linear::from_weights`].
    pub centroids: Linear,

    /// `[vocab_size]` int32 buffer. Reshaped to
    /// `[num_centroids, vocab_size_per_centroid]` at forward time:
    /// row `c` holds the canonical token ids assigned to centroid `c`.
    ///
    /// Stored as a single 1-D tensor (not pre-reshaped) so the storage
    /// matches the upstream checkpoint layout byte-for-byte; the reshape is
    /// free at runtime.
    pub token_ordering: UniquePtr<MlxArray>,

    /// Embedding dimensionality. Cached from the config so forward does not
    /// need to re-derive it from `centroids.weight.shape()`.
    pub hidden_size: usize,

    /// Drafter vocabulary size. Must equal `num_centroids * vocab_size_per_centroid`.
    pub vocab_size: usize,

    /// Number of centroid clusters (default `DEFAULT_NUM_CENTROIDS = 2048`).
    pub num_centroids: usize,

    /// Number of centroids selected per step (default `DEFAULT_TOP_K = 32`).
    /// Must satisfy `0 < top_k <= num_centroids`.
    pub top_k: usize,

    /// `vocab_size / num_centroids`. Each selected centroid contributes this
    /// many candidate token ids to the sparse logit set.
    pub vocab_size_per_centroid: usize,
}

// Manual Debug impl because `Linear` and `MlxArray` are opaque FFI-backed
// types that do not derive Debug. We render only the scalar configuration
// metadata callers reliably want in log lines and assertion failure
// messages; the tensor bodies themselves are GPU-resident and not safe to
// read on the dispatch thread.
impl std::fmt::Debug for MaskedEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaskedEmbedder")
            .field("hidden_size", &self.hidden_size)
            .field("vocab_size", &self.vocab_size)
            .field("num_centroids", &self.num_centroids)
            .field("top_k", &self.top_k)
            .field("vocab_size_per_centroid", &self.vocab_size_per_centroid)
            .finish()
    }
}

/// Errors that can occur while constructing or running [`MaskedEmbedder`].
#[derive(Debug, thiserror::Error)]
pub enum MaskedEmbedderError {
    /// `vocab_size` is not exactly divisible by `num_centroids`. This is a
    /// load-time configuration error, not a runtime one — the entire
    /// algorithm assumes a uniform cluster size.
    #[error(
        "vocab_size {vocab_size} is not divisible by num_centroids {num_centroids} \
         (required for uniform centroid clustering)"
    )]
    VocabNotDivisible {
        vocab_size: usize,
        num_centroids: usize,
    },

    /// `top_k` is zero or exceeds `num_centroids`.
    #[error("top_k {top_k} out of range; must satisfy 0 < top_k <= num_centroids={num_centroids}")]
    TopKOutOfRange { top_k: usize, num_centroids: usize },

    /// A required weight is missing from the checkpoint.
    #[error("required weight {key:?} not found in checkpoint")]
    MissingWeight { key: String },
}

impl MaskedEmbedder {
    /// Construct from raw weights and config. Validates invariants:
    ///
    /// - `0 < top_k <= num_centroids`
    /// - `vocab_size % num_centroids == 0`
    ///
    /// `token_ordering` is taken as-is — if the checkpoint stores it as
    /// i64 / f32 it must be cast to i32 ahead of time (see
    /// [`sanitize_token_ordering`]).
    pub fn new(
        centroids: Linear,
        token_ordering: UniquePtr<MlxArray>,
        hidden_size: usize,
        vocab_size: usize,
        num_centroids: usize,
        top_k: usize,
    ) -> Result<Self, MaskedEmbedderError> {
        if top_k == 0 || top_k > num_centroids {
            return Err(MaskedEmbedderError::TopKOutOfRange {
                top_k,
                num_centroids,
            });
        }
        if !vocab_size.is_multiple_of(num_centroids) {
            return Err(MaskedEmbedderError::VocabNotDivisible {
                vocab_size,
                num_centroids,
            });
        }
        let vocab_size_per_centroid = vocab_size / num_centroids;
        Ok(Self {
            centroids,
            token_ordering,
            hidden_size,
            vocab_size,
            num_centroids,
            top_k,
            vocab_size_per_centroid,
        })
    }

    /// Load all `MaskedEmbedder` weights under `prefix` from a `WeightMap`.
    ///
    /// Expects:
    /// - `{prefix}.centroids.weight` — `[num_centroids, hidden_size]`
    /// - `{prefix}.token_ordering`   — `[vocab_size]` (int32 after sanitize)
    ///
    /// The `prefix` is typically `"masked_embedding"` in the Gemma 4
    /// assistant drafter checkpoint.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        vocab_size: usize,
        num_centroids: usize,
        top_k: usize,
    ) -> Result<Self, MaskedEmbedderError> {
        let centroids_prefix = format!("{prefix}.centroids");
        let centroids = Linear::from_weights(weights, &centroids_prefix).map_err(|_| {
            MaskedEmbedderError::MissingWeight {
                key: format!("{centroids_prefix}.weight"),
            }
        })?;
        let ordering_key = format!("{prefix}.token_ordering");
        let token_ordering = weights.get(&ordering_key).map(|w| ffi::copy(w)).ok_or(
            MaskedEmbedderError::MissingWeight {
                key: ordering_key.clone(),
            },
        )?;
        Self::new(
            centroids,
            token_ordering,
            hidden_size,
            vocab_size,
            num_centroids,
            top_k,
        )
    }

    /// Compute sparse logits over the full vocabulary.
    ///
    /// - `hidden_states`: `[B, L, hidden_size]` in the model's native dtype
    ///   (bf16 or f16 on Apple Silicon).
    /// - `lm_head_weight`: `[vocab_size, hidden_size]`. Tied to the drafter's
    ///   `embed_tokens.weight` by the caller (see `Gemma4AssistantDraftModel` in sub-3).
    ///
    /// Returns `[B, L, vocab_size]` with the same dtype as `hidden_states`.
    /// Non-selected vocab positions are filled with `min(selected) - 1` so
    /// they cannot win any argmax / sampling competition.
    pub fn forward(
        &self,
        hidden_states: &MlxArray,
        lm_head_weight: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(hidden_states);
        // Defensive shape handling: support both [B, L, H] and [L, H] inputs.
        // The 2-D case is what tests use to keep fixtures tiny; the 3-D case
        // is what the real drafter forward produces.
        let (batch, seq_len) = match shape.len() {
            3 => (shape[0], shape[1]),
            2 => (1, shape[0]),
            _ => panic!(
                "MaskedEmbedder::forward expected hidden of rank 2 or 3, got shape {:?}",
                shape
            ),
        };
        // Normalise to [B, L, H] for the rest of the pipeline so the
        // reshape arithmetic stays uniform.
        let hidden = if shape.len() == 2 {
            ffi::reshape(hidden_states, &[batch, seq_len, self.hidden_size as i32])
        } else {
            ffi::copy(hidden_states)
        };
        let dtype = ffi::array_dtype(&hidden);

        let top_k = self.top_k as i32;
        let vsc = self.vocab_size_per_centroid as i32;
        let num_centroids = self.num_centroids as i32;

        // Step 1: centroid_logits = hidden @ centroids.weight.T → [B, L, C].
        let centroid_logits = self.centroids.forward(&hidden);

        // Step 2: top-K centroid indices.
        //
        // argpartition guarantees that index position `kth` contains the
        // kth-largest element and positions > kth are ≥ kth. We want the
        // top_k largest, so kth = num_centroids - top_k and we slice the
        // last `top_k` positions. This matches the Python upstream's
        // `kth=-top_k` (NumPy-style negative kth on a length-C axis).
        let kth = num_centroids - top_k;
        let part_idx = ffi::argpartition(&centroid_logits, kth, -1);
        // Slice [..., kth..C] from the partitioned indices → [B, L, top_k].
        let part_shape = ffi::array_shape(&part_idx);
        let mut starts = vec![0i32; part_shape.len()];
        let mut stops = part_shape.clone();
        let last = part_shape.len() - 1;
        starts[last] = kth;
        stops[last] = num_centroids;
        let topk_idx = ffi::slice(&part_idx, &starts, &stops);

        // Step 3-4: gather centroid token-id blocks.
        //
        // ordering: [C, vsc]; topk_idx: [B, L, top_k].
        // take(ordering, topk_idx, axis=0) produces a tensor whose shape is
        // topk_idx.shape ++ ordering.shape[1:] = [B, L, top_k, vsc].
        let ordering = ffi::reshape(&self.token_ordering, &[num_centroids, vsc]);
        let selected_canonical = ffi::take(&ordering, &topk_idx, 0);

        // Step 5-6: gather corresponding lm_head rows.
        //
        // selected_emb_flat: [B*L*top_k*vsc, hidden]
        // → reshape to [B, L, top_k*vsc, hidden].
        let flat_idx = ffi::reshape(&selected_canonical, &[-1]);
        let selected_emb_flat = ffi::take(lm_head_weight, &flat_idx, 0);
        let selected_emb = ffi::reshape(
            &selected_emb_flat,
            &[batch, seq_len, top_k * vsc, self.hidden_size as i32],
        );

        // Step 7: selected_logits = hidden[...,None,:] @ selected_emb.T
        //                          .squeeze(-2)
        // hidden_3d: [B, L, H] → hidden_4d: [B, L, 1, H]
        let hidden_4d = ffi::expand_dims(&hidden, -2);
        let emb_t = ffi::swap_axes(&selected_emb, -1, -2); // [B,L,H,K*vsc]
        let logits_4d = ffi::matmul(&hidden_4d, &emb_t); // [B,L,1,K*vsc]
        let selected_logits = ffi::squeeze_axis(&logits_4d, -2); // [B,L,K*vsc]

        // Step 8: mask_value = float(selected_logits.min().item()) - 1.0.
        //
        // Matches the upstream Python exactly: a single CPU sync to read a
        // scalar, then broadcast back to the device. We deliberately do not
        // try to stay GPU-resident here because (a) it adds a tensor
        // subtract over a constant of 1.0 with no measurable saving, and
        // (b) `mx.full` is the path upstream uses to lay out the scattered
        // tensor, so any other approach would diverge from upstream
        // numerics.
        let min_arr = ffi::min_all(&selected_logits);
        ffi::eval(&min_arr);
        let mask_value = ffi::item_f32(&min_arr) - 1.0;

        // Step 9: out = full([B, L, vocab_size], mask_value, hidden.dtype).
        //
        // full_f32 casts the f32 scalar to the requested dtype, so bf16/f16
        // inputs produce a bf16/f16 scratch tensor and the scatter that
        // follows stays in the model's native precision (no f32 promotion).
        let out = ffi::full_f32(&[batch, seq_len, self.vocab_size as i32], mask_value, dtype);

        // Step 10-11: scatter selected logits into the masked tensor.
        //
        // scatter_idx: [B, L, top_k*vsc] (same shape as selected_logits).
        let scatter_idx = ffi::reshape(&selected_canonical, &[batch, seq_len, top_k * vsc]);
        ffi::put_along_axis(&out, &scatter_idx, &selected_logits, -1)
    }
}

/// Sanitize the centroid `token_ordering` entry inside a `WeightMap`.
///
/// Mirrors the upstream `Gemma4AssistantDraftModel.sanitize` behaviour
/// (`if k == "masked_embedding.token_ordering": v = v.astype(mx.int32)`).
/// HuggingFace checkpoints store the ordering as int64; mlxcel uses int32
/// throughout for indexing efficiency and to match the indexing dtype the
/// MLX `take` / `put_along_axis` ops expect cheaply.
///
/// No-op when `token_ordering` is absent from the map (e.g. for the
/// 26B-A4B / 31B drafters that tie weights into the dense LM head and do
/// not carry a centroid table). No-op when the entry already has dtype i32.
///
/// `prefix` is the dotted weight prefix under which the `MaskedEmbedder`
/// lives — `"masked_embedding"` for the Gemma 4 assistant drafter.
pub fn sanitize_token_ordering(weights: &mut WeightMap, prefix: &str) {
    let key = format!("{prefix}.token_ordering");
    if let Some(arr) = weights.get(&key) {
        if ffi::array_dtype(arr) != crate::dtype::INT32 {
            let cast = ffi::astype(arr, crate::dtype::INT32);
            weights.insert(key, cast);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;
    use crate::ffi::{
        argmax_last_axis, array_dtype, array_shape, eval, from_slice_f32, from_slice_i32,
        from_slice_i64, item_i32,
    };
    use crate::layers::Linear;

    /// Build a minimal `MaskedEmbedder` fixture: `num_centroids=4`,
    /// `vocab_size=16` (so `vocab_size_per_centroid = 4`), `top_k=2`,
    /// `hidden_size=2`. Centroid weight rows distinguish centroids by their
    /// first hidden dimension; token_ordering maps cluster c to ids
    /// `[c*4 .. (c+1)*4]`.
    fn small_fixture() -> MaskedEmbedder {
        let num_centroids: usize = 4;
        let vocab_size: usize = 16;
        let top_k: usize = 2;
        let hidden_size: usize = 2;

        // Centroids weight shape is [num_centroids, hidden_size] = [4, 2].
        // Row c = [10*c + 1, 0.0]: this puts centroid c's logit at
        // 10*c*h[0] for an input with h[1] = 0, so centroids are perfectly
        // ordered by their index when h[0] > 0.
        let cw = ffi::from_slice_f32(&[1.0, 0.0, 11.0, 0.0, 21.0, 0.0, 31.0, 0.0], &[4, 2]);
        let centroids = Linear::new(cw, None);

        // token_ordering[c*4 + k] = c*4 + k → contiguous block per centroid.
        let ordering_data: Vec<i32> = (0..(num_centroids * 4) as i32).collect();
        let token_ordering = from_slice_i32(&ordering_data, &[vocab_size as i32]);

        MaskedEmbedder::new(
            centroids,
            token_ordering,
            hidden_size,
            vocab_size,
            num_centroids,
            top_k,
        )
        .expect("fixture must satisfy invariants")
    }

    // ----- construction invariants ----------------------------------------

    #[test]
    fn new_rejects_top_k_zero() {
        let cw = ffi::from_slice_f32(&[0.0; 2], &[1, 2]);
        let centroids = Linear::new(cw, None);
        let ordering = from_slice_i32(&[0, 1, 2, 3], &[4]);
        let err = MaskedEmbedder::new(centroids, ordering, 2, 4, 1, 0).expect_err("must reject");
        assert!(matches!(err, MaskedEmbedderError::TopKOutOfRange { .. }));
    }

    #[test]
    fn new_rejects_top_k_too_large() {
        let cw = ffi::from_slice_f32(&[0.0; 4], &[2, 2]);
        let centroids = Linear::new(cw, None);
        let ordering = from_slice_i32(&[0, 1, 2, 3], &[4]);
        let err = MaskedEmbedder::new(centroids, ordering, 2, 4, 2, 3).expect_err("must reject");
        assert!(matches!(err, MaskedEmbedderError::TopKOutOfRange { .. }));
    }

    #[test]
    fn new_rejects_vocab_not_divisible() {
        let cw = ffi::from_slice_f32(&[0.0; 4], &[2, 2]);
        let centroids = Linear::new(cw, None);
        let ordering = from_slice_i32(&[0, 1, 2, 3, 4], &[5]);
        let err = MaskedEmbedder::new(centroids, ordering, 2, 5, 2, 1).expect_err("must reject");
        assert!(matches!(err, MaskedEmbedderError::VocabNotDivisible { .. }));
    }

    #[test]
    fn new_caches_vocab_size_per_centroid() {
        let mod_ = small_fixture();
        assert_eq!(mod_.vocab_size_per_centroid, 4);
        assert_eq!(mod_.num_centroids, 4);
        assert_eq!(mod_.top_k, 2);
        assert_eq!(mod_.vocab_size, 16);
    }

    // ----- forward shape and value semantics -----------------------------

    #[test]
    fn forward_returns_correct_output_shape() {
        let mod_ = small_fixture();
        // hidden_states: [B=1, L=1, H=2]
        let hidden = from_slice_f32(&[1.0, 0.0], &[1, 1, 2]);
        // lm_head_weight: [vocab=16, H=2]. Use a simple per-row signature
        // so the gather result is easy to reason about.
        let mut lm: Vec<f32> = Vec::with_capacity(32);
        for i in 0..16 {
            lm.push(i as f32);
            lm.push(0.0);
        }
        let lm_head = from_slice_f32(&lm, &[16, 2]);

        let out = mod_.forward(&hidden, &lm_head);
        eval(&out);
        assert_eq!(array_shape(&out), vec![1, 1, 16]);
        // dtype preserved from input
        assert_eq!(array_dtype(&out), dtype::FLOAT32);
    }

    #[test]
    fn forward_preserves_dtype_for_f16_hidden() {
        // Apple Silicon precision invariant: bf16/f16 hidden must NOT
        // silently promote to f32. We test f16 here as a stand-in for the
        // bf16 path because the bf16 from_bytes helper requires raw
        // bf16-encoded bytes; the dtype-preservation logic is identical
        // for both half-precision formats (full_f32 dispatches on dtype).
        let mod_ = small_fixture();
        // Build f16 hidden by casting from f32.
        let hidden_f32 = from_slice_f32(&[1.0, 0.0], &[1, 1, 2]);
        let hidden_f16 = ffi::astype(&hidden_f32, dtype::FLOAT16);
        let mut lm: Vec<f32> = Vec::with_capacity(32);
        for i in 0..16 {
            lm.push(i as f32);
            lm.push(0.0);
        }
        let lm_head_f32 = from_slice_f32(&lm, &[16, 2]);
        let lm_head_f16 = ffi::astype(&lm_head_f32, dtype::FLOAT16);

        let out = mod_.forward(&hidden_f16, &lm_head_f16);
        eval(&out);
        assert_eq!(
            array_dtype(&out),
            dtype::FLOAT16,
            "MaskedEmbedder must not promote f16 hidden to f32"
        );
    }

    #[test]
    fn mask_value_strictly_less_than_min_selected() {
        // Acceptance criterion: for every non-selected vocab index, logit
        // must be strictly less than the minimum selected logit.
        let mod_ = small_fixture();
        // hidden = [1, 0] → centroid_logits = [1, 11, 21, 31] (per the
        // fixture). top_k=2 → top centroids = [2, 3].
        let hidden = from_slice_f32(&[1.0, 0.0], &[1, 1, 2]);
        // lm_head_weight[i] = [i, 0] → gather rows i ∈ {8,9,10,11,12,13,14,15}.
        // selected_logits = i for those rows.
        let mut lm: Vec<f32> = Vec::with_capacity(32);
        for i in 0..16 {
            lm.push(i as f32);
            lm.push(0.0);
        }
        let lm_head = from_slice_f32(&lm, &[16, 2]);

        let out = mod_.forward(&hidden, &lm_head);
        eval(&out);
        let bytes = ffi::array_to_raw_bytes(&out);
        let mut logits = [0f32; 16];
        for (i, chunk) in bytes.chunks_exact(4).enumerate().take(16) {
            logits[i] = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        let min_selected = logits[8..16].iter().cloned().fold(f32::INFINITY, f32::min);
        for (i, &v) in logits.iter().enumerate() {
            if i < 8 {
                assert!(
                    v < min_selected,
                    "non-selected logit at idx {} = {} must be < min selected {}",
                    i,
                    v,
                    min_selected
                );
            } else {
                assert_eq!(
                    v, i as f32,
                    "selected logit at idx {} must equal lm_head row dot hidden",
                    i
                );
            }
        }
    }

    #[test]
    fn fixture_top_k_indices_match_hand_computed_reference() {
        // Acceptance criterion: with num_centroids=4, vocab_size=16, top_k=2,
        // verify the gathered indices match the hand-computed reference.
        // hidden = [1, 0], cw = [[1,0],[11,0],[21,0],[31,0]] ⇒ centroid
        // logits = [1, 11, 21, 31]. argpartition kth=2 → top-2 = {2, 3}.
        // ordering = [[0..3],[4..7],[8..11],[12..15]] ⇒ selected canonical =
        // ids 8..16.
        let mod_ = small_fixture();
        let hidden = from_slice_f32(&[1.0, 0.0], &[1, 1, 2]);
        // Construct lm_head where row i = [i+1000, 0] so we can spot the
        // exact row indices in the gathered scatter.
        let mut lm: Vec<f32> = Vec::with_capacity(32);
        for i in 0..16 {
            lm.push(1000.0 + i as f32);
            lm.push(0.0);
        }
        let lm_head = from_slice_f32(&lm, &[16, 2]);

        let out = mod_.forward(&hidden, &lm_head);
        eval(&out);
        let bytes = ffi::array_to_raw_bytes(&out);
        let mut logits = [0f32; 16];
        for (i, chunk) in bytes.chunks_exact(4).enumerate().take(16) {
            logits[i] = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for (offset, &v) in logits.iter().enumerate().skip(8).take(8) {
            assert_eq!(
                v,
                1000.0 + offset as f32,
                "selected logit at canonical idx {} must equal row {} dot hidden",
                offset,
                offset
            );
        }
    }

    #[test]
    fn degenerate_dense_equivalence_when_top_k_equals_num_centroids() {
        // Acceptance criterion (strictest correctness check): when
        // num_centroids == vocab_size and top_k == num_centroids, the output
        // must equal hidden @ embed.T exactly (modulo any precision
        // tolerance — f32 here makes this an exact check).
        //
        // Set up: vocab_size=8, num_centroids=8 → vsc=1. top_k=8 means
        // all centroids are selected, every vocab position is scattered, so
        // the mask_value never wins and the result is the dense matmul.
        let vocab_size: usize = 8;
        let num_centroids: usize = 8;
        let hidden_size: usize = 3;
        let top_k: usize = 8;

        // Centroids: arbitrary scoring, irrelevant when top_k=num_centroids
        // because all positions end up selected anyway. Use small numbers
        // to keep the matmul exact.
        let cw: Vec<f32> = (0..(num_centroids * hidden_size) as i32)
            .map(|i| (i % 5) as f32 * 0.25)
            .collect();
        let cw_arr = from_slice_f32(&cw, &[num_centroids as i32, hidden_size as i32]);
        let centroids = Linear::new(cw_arr, None);

        // Identity ordering: token_ordering[i] = i.
        let ordering_data: Vec<i32> = (0..vocab_size as i32).collect();
        let token_ordering = from_slice_i32(&ordering_data, &[vocab_size as i32]);

        let mod_ = MaskedEmbedder::new(
            centroids,
            token_ordering,
            hidden_size,
            vocab_size,
            num_centroids,
            top_k,
        )
        .expect("invariants must hold");

        let hidden_data: Vec<f32> = vec![0.5, -0.25, 1.0];
        let hidden = from_slice_f32(&hidden_data, &[1, 1, hidden_size as i32]);

        let emb_data: Vec<f32> = (0..(vocab_size * hidden_size) as i32)
            .map(|i| (i as f32 - 12.0) * 0.125)
            .collect();
        let lm_head = from_slice_f32(&emb_data, &[vocab_size as i32, hidden_size as i32]);

        // MaskedEmbedder forward.
        let masked = mod_.forward(&hidden, &lm_head);
        eval(&masked);

        // Reference: hidden @ lm_head.T directly.
        let lm_t = ffi::transpose(&lm_head);
        let dense = ffi::matmul(&hidden, &lm_t);
        eval(&dense);

        let masked_bytes = ffi::array_to_raw_bytes(&masked);
        let dense_bytes = ffi::array_to_raw_bytes(&dense);
        assert_eq!(
            masked_bytes.len(),
            dense_bytes.len(),
            "shape mismatch between MaskedEmbedder and dense reference"
        );
        for (i, (mb, db)) in masked_bytes
            .chunks_exact(4)
            .zip(dense_bytes.chunks_exact(4))
            .enumerate()
            .take(vocab_size)
        {
            let m = f32::from_ne_bytes([mb[0], mb[1], mb[2], mb[3]]);
            let d = f32::from_ne_bytes([db[0], db[1], db[2], db[3]]);
            assert!(
                (m - d).abs() <= 1e-5,
                "logit {}: masked {} != dense {}",
                i,
                m,
                d
            );
        }
    }

    #[test]
    fn argmax_never_picks_masked_token_with_argmax_sampling() {
        // Acceptance criterion: temp=0 argmax must never pick a masked-out
        // token. With our fixture, hidden = [1, 0] selects centroids {2, 3}
        // → canonical ids 8..16. argmax over the masked output must land
        // in 8..=15.
        let mod_ = small_fixture();
        let hidden = from_slice_f32(&[1.0, 0.0], &[1, 1, 2]);
        let mut lm: Vec<f32> = Vec::with_capacity(32);
        for i in 0..16 {
            // Decreasing row magnitudes so that even after the mask kicks
            // in for ids < 8, the argmax stays inside 8..16.
            lm.push(20.0 - i as f32);
            lm.push(0.0);
        }
        let lm_head = from_slice_f32(&lm, &[16, 2]);

        let out = mod_.forward(&hidden, &lm_head);
        eval(&out);

        let am = argmax_last_axis(&out);
        eval(&am);
        let picked = item_i32(&am);
        assert!(
            (8..16).contains(&picked),
            "argmax picked masked-out token id {}; expected one of 8..16",
            picked
        );
    }

    #[test]
    fn forward_handles_2d_hidden_input() {
        // The Python upstream accepts hidden_states with arbitrary leading
        // dimensions and only relies on the last two being [..., L, H].
        // We support [L, H] too so unit tests can keep tensor sizes minimal.
        let mod_ = small_fixture();
        let hidden = from_slice_f32(&[1.0, 0.0], &[1, 2]);
        let mut lm: Vec<f32> = Vec::with_capacity(32);
        for i in 0..16 {
            lm.push(i as f32);
            lm.push(0.0);
        }
        let lm_head = from_slice_f32(&lm, &[16, 2]);
        let out = mod_.forward(&hidden, &lm_head);
        eval(&out);
        assert_eq!(array_shape(&out), vec![1, 1, 16]);
    }

    // ----- sanitize_token_ordering ---------------------------------------

    #[test]
    fn sanitize_casts_i64_token_ordering_to_i32() {
        // HF checkpoints store the ordering as int64. mlxcel uses int32.
        // Verify the sanitize hook performs the cast.
        let mut weights: WeightMap = WeightMap::new();
        let data_i64: Vec<i64> = (0..16).collect();
        let arr_i64 = from_slice_i64(&data_i64, &[16]);
        assert_eq!(array_dtype(&arr_i64), dtype::INT64);
        weights.insert("masked_embedding.token_ordering".to_string(), arr_i64);

        sanitize_token_ordering(&mut weights, "masked_embedding");

        let after = weights
            .get("masked_embedding.token_ordering")
            .expect("entry preserved");
        assert_eq!(array_dtype(after), dtype::INT32);
        assert_eq!(array_shape(after), vec![16]);
    }

    #[test]
    fn sanitize_is_noop_when_already_i32() {
        let mut weights: WeightMap = WeightMap::new();
        let data: Vec<i32> = (0..8).collect();
        let arr = from_slice_i32(&data, &[8]);
        weights.insert("masked_embedding.token_ordering".to_string(), arr);

        sanitize_token_ordering(&mut weights, "masked_embedding");

        let after = weights
            .get("masked_embedding.token_ordering")
            .expect("entry preserved");
        assert_eq!(array_dtype(after), dtype::INT32);
    }

    #[test]
    fn sanitize_is_noop_when_key_missing() {
        // Dense-LM-head drafters (26B-A4B, 31B) carry no centroid table —
        // the hook must not insert anything in their absence.
        let mut weights: WeightMap = WeightMap::new();
        weights.insert("something.else".to_string(), from_slice_i32(&[0, 1], &[2]));

        sanitize_token_ordering(&mut weights, "masked_embedding");

        assert!(!weights.contains_key("masked_embedding.token_ordering"));
        assert!(weights.contains_key("something.else"));
    }

    // ----- default constants pinning -------------------------------------

    #[test]
    fn default_constants_match_canonical_gemma4_e2b_e4b_drafter() {
        // Pins's canonical defaults (per Gemma4AssistantConfig:
        // num_centroids=2048, centroid_intermediate_top_k=32). Changing
        // these in code without updating the README / config is a
        // regression source — this test fences that.
        assert_eq!(DEFAULT_NUM_CENTROIDS, 2048);
        assert_eq!(DEFAULT_TOP_K, 32);
        assert_eq!(TOKEN_ORDERING_KEY, "masked_embedding.token_ordering");
    }
}
