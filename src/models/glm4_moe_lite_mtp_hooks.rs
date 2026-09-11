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

//! MTP target hooks for GLM4 MoE Lite (issue #1326).
//!
//! Everything the `glm4_moe_lite_mtp_target` adapter needs from the model
//! that the classic forward does not provide:
//!
//! - a **verify forward** over the `[bonus] ++ draft` block (`[1, bs]`)
//!   that runs the *absorbed* MLA path classic decode runs at `M = 1`,
//!   rather than the un-absorbed prefill path [`MlaAttention::forward`]
//!   takes for any `l > 1`;
//! - the **model-owned per-sequence cache slot** the adapter routes through,
//!   so the adapter stays a stateless view the tick-cooperative slice path
//!   can rebuild every tick;
//! - trim-based **rollback** and the cache offset the round loop rebinds
//!   the drafter to;
//! - the **exactness probe** behind the shared
//!   [`crate::models::speculative_exactness::mtp_exactness_gate`].
//!
//! ## Why the verify attention is per position
//!
//! The temperature-0 contract is that an `M = bs` verify block emits the
//! same logits as `bs` single-token decode steps. Two things can break it:
//! which quantized-matmul kernel MLX dispatches at `M = bs` versus `M = 1`
//! (measured by the probe, and bought back by the gate's `qmv_wide` retry
//! where that suffices), and the attention itself, whose SDPA kernel and
//! score reduction differ between a one-row and a multi-row query. The
//! projections are batched here because that is where the verify saves
//! time; the attention is run one query row at a time against the key
//! prefix that row may see, so every SDPA and every `q_pe @ k_pe^T` call
//! has exactly the shape the single-token decode step issues. That is the
//! same split Qwen 3.5's `target_verify` attention makes. Shape alone is not
//! enough: a one-row slice of the batched `[1, H, bs, d]` query keeps the
//! parent's strides, and MLX's SDPA and matmul pick their kernel on
//! contiguity, so each row view is materialized before it is used. Left as
//! a view, the verify rows drifted from the chain in a data-dependent way
//! (first at position 4 on one real prompt and 14 on another) and the drift
//! compounded through the cache; the opt-in real-checkpoint gate in
//! `glm4_moe_lite_mtp_target_tests.rs` is what caught it.
//!
//! The MoE is the other width-sensitive piece, and it is deliberately left
//! batched. `SwitchGLU::forward` switches to the gather-sort path once
//! `n_tokens * top_k >= 64`, so at the default `block_size = 2` (8 rows with
//! `num_experts_per_tok = 4`) the verify block takes the same reduction as
//! the decode chain, while a wide `--draft-block-size` crosses the threshold
//! and takes a different one. Undoing that per row would give back the
//! verify's whole saving, so the probe is left to measure it: a width that
//! crosses the threshold and diverges declines the pairing rather than
//! emitting a divergent stream.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::KVCache;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{Glm4MoeLiteModel, MlaAttention, TransformerBlock};
use crate::models::speculative_exactness::{
    BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate,
};

/// Prompt length of one exactness-probe draw (matches the Qwen 3.5 and
/// Gemma 4 probes).
const PROBE_PROMPT_LEN: usize = 8;
/// Independent synthetic inputs per probe; any single divergence declines.
const PROBE_DRAWS: usize = 3;
/// Consecutive verify blocks walked per draw. One block per draw missed the
/// per-row query-view divergence this family's verify path once had: it was
/// data-dependent and surfaced only after several `M = bs` blocks had fed the
/// cache (position 4 on one real prompt, 14 on another), so each draw walks
/// this many blocks against the chain and compares every row.
const PROBE_BLOCKS_PER_DRAW: usize = 4;

impl MlaAttention {
    /// Verify-block attention: absorbed MLA at `M = bs`, one query row per
    /// SDPA call.
    ///
    /// The projections (`q_a`/`q_b`, `kv_a`, `embed_q`, `unembed_out`,
    /// `o_proj`) run once over the whole block. RoPE, the cache append and
    /// the latent slicing are the ones [`Self::forward`] performs at
    /// `l == 1`; causality comes from slicing the cached latent to the
    /// `offset + i + 1` entries row `i` may attend, so no additive causal
    /// mask is built and the per-row `pe_scores` is passed to SDPA exactly
    /// as the decode step passes it.
    ///
    /// Used by: [`TransformerBlock::forward_verify`].
    pub fn forward_verify(&self, x: &MlxArray, cache: &mut KVCache) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        debug_assert_eq!(b, 1, "verify forward is B = 1 only");

        let q = if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_a_out = q_a.forward(x);
            let q_a_normed = q_a_ln.forward(&q_a_out);
            q_b.forward(&q_a_normed)
        } else if let Some(q_proj) = &self.q_proj {
            q_proj.forward(x)
        } else {
            return mlxcel_core::zeros(
                &[b, l, self.num_heads * self.v_head_dim],
                mlxcel_core::dtype::FLOAT16,
            );
        };

        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.q_head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let q_nope = slice_axis(&q, -1, 0, self.qk_nope_head_dim);
        let q_pe = slice_axis(&q, -1, self.qk_nope_head_dim, -1);

        let compressed_kv = self.kv_a_proj_with_mqa.forward(x);
        let kv_latent = slice_axis(&compressed_kv, -1, 0, self.kv_lora_rank);
        let k_pe = slice_axis(&compressed_kv, -1, self.kv_lora_rank, -1);
        let kv_latent = self.kv_a_layernorm.forward(&kv_latent);
        let k_pe = mlxcel_core::reshape(&k_pe, &[b, l, 1, self.qk_rope_head_dim]);
        let k_pe = mlxcel_core::transpose_axes(&k_pe, &[0, 2, 1, 3]);

        let offset = cache.offset;
        let q_pe = mlxcel_core::fast_rope(
            &q_pe,
            self.qk_rope_head_dim,
            true,
            self.rope_base,
            1.0,
            offset,
        );
        let k_pe = mlxcel_core::fast_rope(
            &k_pe,
            self.qk_rope_head_dim,
            true,
            self.rope_base,
            1.0,
            offset,
        );

        let kv_latent = mlxcel_core::expand_dims(&kv_latent, 1);
        let (kv_latent, k_pe) = cache.update_and_fetch(kv_latent, k_pe);

        let scale_scalar = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(&q_pe));
        let q_pe_scaled = mlxcel_core::multiply(&q_pe, &scale_scalar);
        let q_projected = self.embed_q.forward(&q_nope);

        let mut rows: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(l as usize);
        for i in 0..l {
            let visible = offset + i + 1;
            // Row views are materialized: a one-row slice of an `l > 1`
            // tensor keeps the parent's strides, and MLX's SDPA and matmul
            // select their kernel on contiguity, so an unmaterialized view
            // would put this row on a different reduction than the
            // single-token decode step (the block-vs-chain divergence the
            // real-checkpoint gate below caught, data-dependent and
            // compounding through the cache).
            let q_pe_i = mlxcel_core::contiguous(
                &mlxcel_core::slice(
                    &q_pe_scaled,
                    &[0, 0, i, 0],
                    &[b, self.num_heads, i + 1, self.qk_rope_head_dim],
                ),
                false,
            );
            let k_pe_i = mlxcel_core::slice(
                &k_pe,
                &[0, 0, 0, 0],
                &[b, 1, visible, self.qk_rope_head_dim],
            );
            let k_pe_t = mlxcel_core::transpose_axes(&k_pe_i, &[0, 1, 3, 2]);
            let pe_scores = mlxcel_core::matmul(&q_pe_i, &k_pe_t);

            let q_i = mlxcel_core::contiguous(
                &mlxcel_core::slice(
                    &q_projected,
                    &[0, 0, i, 0],
                    &[b, self.num_heads, i + 1, self.kv_lora_rank],
                ),
                false,
            );
            let kv_i = mlxcel_core::slice(
                &kv_latent,
                &[0, 0, 0, 0],
                &[b, 1, visible, self.kv_lora_rank],
            );
            let pe_mask_ptr = &*pe_scores as *const MlxArray;
            // SAFETY: `pe_scores` outlives the call; the pointer is read
            // synchronously by `attention_from_ptr`.
            let out_i = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_i,
                    &kv_i,
                    &kv_i,
                    self.scale,
                    pe_mask_ptr,
                    0.0,
                    0,
                )
            };
            rows.push(out_i);
        }
        let refs: Vec<&MlxArray> = rows
            .iter()
            .map(|r| r.as_ref().expect("attention row"))
            .collect();
        let output = if refs.len() == 1 {
            mlxcel_core::copy(refs[0])
        } else {
            mlxcel_core::concatenate_many(&refs, 2)
        };
        let output = self.unembed_out.forward(&output);

        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[b, l, self.num_heads * self.v_head_dim]);
        self.o_proj.forward(&output)
    }
}

impl TransformerBlock {
    /// [`Self::forward`] with the attention on the verify path.
    ///
    /// Used by: [`Glm4MoeLiteModel::forward_verify`].
    pub fn forward_verify(&self, x: &MlxArray, cache: &mut KVCache) -> UniquePtr<MlxArray> {
        let normed = self.input_layernorm.forward(x);
        let attn_out = self.self_attn.forward_verify(&normed, cache);
        let h = mlxcel_core::add(x, &attn_out);
        let normed = self.post_attention_layernorm.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

impl Glm4MoeLiteModel {
    /// Verify forward over a `[1, bs]` block: full logits and the
    /// pre-final-norm hidden of every row, on the absorbed attention path.
    ///
    /// Used by: [`Self::forward_verify_for_sequence`],
    /// [`Self::probe_block_chain_exactness`].
    pub(crate) fn forward_verify(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let mut h = self.embed_tokens.forward(input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_verify(&h, &mut caches[i]);
        }
        let normed = self.norm.forward(&h);
        (self.lm_head.forward(&normed), h)
    }

    /// Apply the final RMSNorm to a captured pre-norm hidden block.
    ///
    /// The drafter consumes the target's post-final-norm hidden by default
    /// (the DeepSeek-V3 MTP convention and the Qwen 3.5 adapter's choice);
    /// the hidden the forwards above return is pre-norm.
    ///
    /// Used by: `Glm4MoeLiteMtpTargetAdapter`.
    pub(crate) fn apply_final_norm(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.norm.forward(hidden)
    }

    /// Install a fresh cache set in the sequence's MTP slot.
    ///
    /// `prefill_and_seed` starts from here rather than appending to whatever
    /// a stale slot holds; a `None` sequence id resets the internal fallback.
    ///
    /// Used by: `Glm4MoeLiteMtpTargetAdapter::prefill_and_seed`.
    pub(crate) fn reset_mtp_sequence_state(&self, seq_id: Option<SequenceId>) {
        match seq_id {
            Some(id) => self
                .mtp_sequence_state
                .replace_sequence_state(id, self.make_configured_caches()),
            None => self
                .mtp_sequence_state
                .replace_internal(self.make_configured_caches()),
        }
    }

    /// Sequence-routed [`Self::forward_with_hidden`] on the MTP slot.
    ///
    /// Used by: `Glm4MoeLiteMtpTargetAdapter::prefill_and_seed`.
    pub(crate) fn forward_prefill_with_last_hidden_for_sequence(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.mtp_sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_configured_caches(),
            |caches| self.forward_with_hidden(input_ids, caches),
        )
    }

    /// Sequence-routed [`Self::forward_verify`] on the MTP slot.
    ///
    /// Used by: `Glm4MoeLiteMtpTargetAdapter::verify_forward`.
    pub(crate) fn forward_verify_for_sequence(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.mtp_sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_configured_caches(),
            |caches| self.forward_verify(input_ids, caches),
        )
    }

    /// Drop the last `rejected` entries of every layer cache in the
    /// sequence's MTP slot. Returns the entries actually trimmed on layer 0.
    ///
    /// Used by: `Glm4MoeLiteMtpTargetAdapter::verify_finalize`.
    pub(crate) fn rollback_speculative_cache_for_sequence(
        &self,
        seq_id: Option<SequenceId>,
        rejected: i32,
    ) -> i32 {
        if rejected <= 0 {
            return 0;
        }
        self.mtp_sequence_state
            .with_sequence_state(seq_id, |caches| {
                let mut trimmed = 0;
                for (i, cache) in caches.iter_mut().enumerate() {
                    let n = cache.trim(rejected);
                    if i == 0 {
                        trimmed = n;
                    }
                }
                trimmed
            })
    }

    /// Absolute offset of the sequence's layer-0 MTP cache (what upstream
    /// reads as `prompt_cache[0].offset` when rebinding the drafter).
    ///
    /// Used by: `Glm4MoeLiteMtpTargetAdapter::{prefill_and_seed, verify_finalize}`.
    pub(crate) fn speculative_cache_offset_for_sequence(&self, seq_id: Option<SequenceId>) -> i32 {
        self.mtp_sequence_state
            .with_sequence_state(seq_id, |caches| {
                caches.first().map(|c| c.offset).unwrap_or(0)
            })
    }

    /// Whether MTP may engage on this checkpoint at `block_size`.
    ///
    /// The whole eligibility condition behind one call, so the server gate
    /// and the offline CLI gate cannot drift: there is no static kernel
    /// prerequisite for this family (plain MLA over `KVCache`, no Metal-only
    /// recurrence kernel), so the answer is the measured
    /// [`Self::probe_block_chain_exactness`], memoized per (model, block
    /// width) through
    /// [`mtp_exactness_gate`](crate::models::speculative_exactness::mtp_exactness_gate),
    /// which also owns the decline logging, the `qmv_wide` retry and the
    /// `MLXCEL_MTP_ALLOW_INEXACT` override.
    ///
    /// Used by: `speculative_burst::mtp_capable_target`, the offline
    /// `run_offline_mtp` gate.
    pub fn mtp_exactness_allows(&self, block_size: usize) -> bool {
        let key = ProbeKey {
            block_size: block_size as u32,
            hidden_size: self.hidden_size() as u32,
            num_hidden_layers: self.layers.len() as u32,
        };
        mtp_exactness_gate(key, || self.probe_block_chain_exactness(block_size))
    }

    /// Hidden width, read off the final norm so it needs no config handle.
    pub(crate) fn hidden_size(&self) -> usize {
        mlxcel_core::array_shape(&self.norm.weight)
            .last()
            .copied()
            .unwrap_or(0) as usize
    }

    /// Vocabulary width, read off a one-token probe through the head.
    pub(crate) fn vocab_size(&self) -> usize {
        let ids = mlxcel_core::from_slice_i32(&[0], &[1, 1]);
        let h = self.embed_tokens.forward(&ids);
        let logits = self.lm_head.forward(&h);
        mlxcel_core::array_shape(&logits)
            .last()
            .copied()
            .unwrap_or(0) as usize
    }

    /// Measure, on this loaded checkpoint, whether a `T = block_size`
    /// verify block produces byte-identical logits to `block_size`
    /// consecutive single-token decode steps.
    ///
    /// The chain arm is the classic [`Self::forward`] (prefill, then one
    /// token at a time, absorbed decode); the block arm is the same prefill
    /// followed by one [`Self::forward_verify`] over the block. Both run on
    /// throwaway caches and `&self`, so no per-sequence server state is
    /// touched. Repeated over [`PROBE_DRAWS`] synthetic inputs, and any
    /// single divergence is the answer, for the reasons
    /// [`crate::models::speculative_exactness`] gives.
    ///
    /// Used by: [`Self::mtp_exactness_allows`].
    pub fn probe_block_chain_exactness(&self, block_size: usize) -> BlockChainExactness {
        if block_size < 2 {
            return BlockChainExactness::NotRun("block width below 2 drafts nothing");
        }
        let vocab = self.vocab_size();
        if vocab < 2 {
            return BlockChainExactness::NotRun("degenerate vocabulary");
        }
        for draw in 0..PROBE_DRAWS {
            let verdict = self.probe_one_draw(block_size, vocab, draw);
            if !verdict.is_equal() {
                return verdict;
            }
        }
        BlockChainExactness::Equal
    }

    /// One (prompt, block) draw of [`Self::probe_block_chain_exactness`].
    fn probe_one_draw(&self, block_size: usize, vocab: usize, draw: usize) -> BlockChainExactness {
        let salt = draw * 977 + 1;
        let wrap =
            |i: usize, stride: usize, offset: usize| ((i * stride + offset + salt) % vocab) as i32;
        let prompt: Vec<i32> = (0..PROBE_PROMPT_LEN).map(|i| wrap(i, 7, 1)).collect();
        let walk: Vec<i32> = (0..block_size * PROBE_BLOCKS_PER_DRAW)
            .map(|i| wrap(i, 13, 3))
            .collect();

        let as_input =
            |tokens: &[i32]| mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let position_bytes = |logits: &MlxArray, index: i32| -> Vec<u8> {
            let shape = mlxcel_core::array_shape(logits);
            let row = mlxcel_core::slice(logits, &[0, index, 0], &[shape[0], index + 1, shape[2]]);
            mlxcel_core::eval(&row);
            mlxcel_core::array_to_raw_bytes(&row)
        };

        let mut chain_caches = self.make_configured_caches();
        let _ = self.forward(&as_input(&prompt), &mut chain_caches, None);
        let mut chain_positions: Vec<Vec<u8>> = Vec::with_capacity(walk.len());
        for token in &walk {
            let logits = self.forward(&as_input(&[*token]), &mut chain_caches, None);
            chain_positions.push(position_bytes(&logits, 0));
        }

        // Teacher-forced on the same tokens, so the block arm's cache is fed
        // by its own `M = bs` rows exactly as an engaged session's would be.
        // Both arms build through `make_configured_caches` for the same
        // reason: a quantized KV mode is part of the arithmetic the probe is
        // deciding about, so probing on FP16 would clear a path the engaged
        // session does not run (issue #1326).
        let mut block_caches = self.make_configured_caches();
        let _ = self.forward(&as_input(&prompt), &mut block_caches, None);
        let mut block_positions: Vec<Vec<u8>> = Vec::with_capacity(walk.len());
        for block in walk.chunks(block_size) {
            let (logits, _hidden) = self.forward_verify(&as_input(block), &mut block_caches);
            for i in 0..block.len() {
                block_positions.push(position_bytes(&logits, i as i32));
            }
        }

        compare_block_against_chain(&block_positions, &chain_positions)
    }
}
