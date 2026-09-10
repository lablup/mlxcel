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

//! LFM2 / LFM2.5 as a DSpark speculative-decoding target (issue #1339).
//!
//! The LFM2 DSpark drafter is fed five of the target's residual streams and
//! its proposals are verified by one causal target forward over the block,
//! so the target needs three things the generation forward does not have:
//!
//! - a verify forward that returns full-block logits and the post-layer
//!   hidden states at the drafter's `target_layer_ids`, captured before
//!   `embedding_norm` ([`Lfm2Model::forward_speculative`]);
//! - a rollback to the committed prefix after a partial accept. Attention
//!   layers trim their KV cache; a short-conv layer's state is the last
//!   `L_cache - 1` rows of its gated input, which cannot be trimmed, so it
//!   is recomputed from a snapshot the verify forward took
//!   ([`Lfm2Model::rollback_speculative_cache`]);
//! - the block-versus-chain exactness probe that decides whether a verify
//!   block of `bs` rows produces the same logits as `bs` single-token decode
//!   steps on this host ([`Lfm2Model::dflash_exactness_allows`]). The
//!   temperature-0 contract rests on that, and the MoE block in particular
//!   takes the fused single-row kernel only at `M = 1`.
//!
//! A child of `lfm2.rs` so it can reach the layer internals; the server
//! reaches it through `mlxcel_core::drafter::dflash::SpeculativeTarget`.

use mlxcel_core::concatenate_many;
use mlxcel_core::drafter::dflash::SpeculativeTarget;
use mlxcel_core::utils::{create_causal_mask, slice_axis};
use mlxcel_core::{MlxArray, UniquePtr};

use super::{Lfm2LayerCache, Lfm2Model};
use crate::models::speculative_exactness::{
    BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate,
};

/// Independent synthetic inputs the exactness probe compares before it
/// reports equality; see `models::qwen3_5::PROBE_DRAWS` for why one is not
/// enough.
const PROBE_DRAWS: usize = 3;

/// Prompt length of each probe draw.
const PROBE_PROMPT_LEN: usize = 8;

/// What one short-conv layer recorded during a verify forward so its state
/// can be rebuilt at any prefix of the block.
///
/// After the block, the conv state holds the last `L_cache - 1` rows of the
/// padded conv input. Truncating the block at `n` committed rows means the
/// state should hold the last `L_cache - 1` rows of
/// `concat(prev_state_or_zeros, bx_block[:, :n])` instead, which is exactly
/// what these two tensors reconstruct.
pub struct ConvRollbackSnapshot {
    /// Index of the layer in the decoder stack.
    pub layer_idx: usize,
    /// The `[1, L_cache - 1, hidden]` conv state before the block, or `None`
    /// when the block was the first forward on fresh caches.
    pub prev_state: Option<UniquePtr<MlxArray>>,
    /// The block's gated conv input `Bx = B * x`, `[1, bs, hidden]`.
    pub bx_block: UniquePtr<MlxArray>,
}

/// Verify-pass output for the speculative path.
///
/// `hidden_states` is ordered to match the `capture_layer_ids` argument so
/// the drafter can concatenate it along the feature axis; `conv_states` is
/// ordered by layer index over short-conv layers only.
///
/// Used by: the DFlash round loop, [`Lfm2Model::rollback_speculative_cache`].
pub struct VerifyOutput {
    /// `[1, bs, vocab]` logits for every verify row.
    pub logits: UniquePtr<MlxArray>,
    /// One `[1, bs, hidden]` post-layer residual stream per requested layer.
    pub hidden_states: Vec<UniquePtr<MlxArray>>,
    /// One snapshot per short-conv layer, in layer order.
    pub conv_states: Vec<ConvRollbackSnapshot>,
}

impl Lfm2Model {
    /// The mixed attention / short-conv cache vector a speculative burst
    /// runs on. Independent of the scheduler's per-sequence slots.
    ///
    /// Used by: the server DFlash burst (`DFlashTargetModel::make_dflash_caches`).
    pub fn make_speculative_caches(&self) -> Vec<Lfm2LayerCache> {
        self.make_caches()
    }

    /// One forward over `input_ids` (`[1, L]`) that also captures the
    /// post-layer residual streams at `capture_layer_ids` (before
    /// `embedding_norm`) and the short-conv rollback snapshots.
    ///
    /// The arithmetic is the generation forward's: the same causal mask
    /// anchored on the first attention layer's offset, the same
    /// left-padded conv, the same MoE routing. That is the exactness
    /// premise, and the probe below measures whether the kernels honour
    /// it at this block width.
    ///
    /// Used by: `SpeculativeTarget::verify_forward_with_capture_layers`,
    /// [`Self::probe_block_chain_exactness`].
    pub fn forward_speculative(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Lfm2LayerCache],
        capture_layer_ids: &[usize],
    ) -> VerifyOutput {
        let h0 = self.embed_tokens.forward(input_ids);
        let shape = mlxcel_core::array_shape(&h0);
        let seq_len = shape[1];

        let attn_offset = caches
            .iter()
            .find(|c| matches!(c, Lfm2LayerCache::Attention(_)))
            .map(|c| c.offset())
            .unwrap_or(0);
        let attn_mask = if seq_len > 1 {
            Some(create_causal_mask(seq_len, attn_offset))
        } else {
            None
        };

        let mut hidden_slots: Vec<Option<UniquePtr<MlxArray>>> =
            (0..capture_layer_ids.len()).map(|_| None).collect();
        let mut conv_states: Vec<ConvRollbackSnapshot> = Vec::new();

        let mut h = h0;
        for (i, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            let mask = if layer.is_attention() {
                attn_mask.as_deref()
            } else {
                None
            };
            h = layer.forward_with_capture(&h, cache, mask, None, Some((i, &mut conv_states)));
            for (slot, &want) in capture_layer_ids.iter().enumerate() {
                if want == i {
                    hidden_slots[slot] = Some(mlxcel_core::copy(&h));
                }
            }
        }

        // Keep the vector length-aligned with the request; the drafter's
        // compatibility gate rejects out-of-range ids before this runs.
        let hidden_dtype = mlxcel_core::array_dtype(&h);
        let hidden_states: Vec<UniquePtr<MlxArray>> = hidden_slots
            .into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    mlxcel_core::zeros(
                        &[shape[0], seq_len, self.config.hidden_size as i32],
                        hidden_dtype,
                    )
                })
            })
            .collect();

        let h = self.embedding_norm.forward(&h);
        let logits = self.embed_tokens.as_linear(&h);
        VerifyOutput {
            logits,
            hidden_states,
            conv_states,
        }
    }

    /// Rewind `caches` to the committed prefix after a partial accept of a
    /// `block_size`-row verify block: `n = accepted + 1` rows stay.
    ///
    /// Attention layers trim `block_size - n` positions off their KV cache
    /// (`offset` moves back, the buffers are overwritten by the next
    /// append). Short-conv layers rebuild their state from the snapshot:
    /// the last `L_cache - 1` rows of `concat(prev_state_or_zeros,
    /// bx_block[:, :n])`, which is the state a forward over exactly the
    /// committed rows would have left.
    ///
    /// A full accept never calls this; every cache then stays as the
    /// verify forward left it.
    ///
    /// Used by: `SpeculativeTarget::rollback_partial`.
    pub fn rollback_speculative_cache(
        &self,
        caches: &mut [Lfm2LayerCache],
        conv_states: &[ConvRollbackSnapshot],
        accepted: i32,
        block_size: i32,
    ) {
        let n = accepted + 1;
        let trim = block_size - n;
        if trim <= 0 {
            return;
        }
        let n_keep = self.config.conv_l_cache as i32 - 1;
        let mut snapshots = conv_states.iter();
        for (layer_idx, cache) in caches.iter_mut().enumerate() {
            match cache {
                Lfm2LayerCache::Attention(kv) => {
                    kv.trim(trim);
                }
                Lfm2LayerCache::Conv(state) => {
                    let Some(snap) = snapshots.next() else {
                        continue;
                    };
                    debug_assert_eq!(
                        snap.layer_idx, layer_idx,
                        "conv rollback snapshot must come from the same layer"
                    );
                    let committed = slice_axis(&snap.bx_block, 1, 0, n);
                    let padded = match &snap.prev_state {
                        Some(prev) => mlxcel_core::concatenate(prev, &committed, 1),
                        None => {
                            let bx_shape = mlxcel_core::array_shape(&snap.bx_block);
                            let zeros = mlxcel_core::zeros(
                                &[bx_shape[0], n_keep, bx_shape[2]],
                                mlxcel_core::array_dtype(&snap.bx_block),
                            );
                            mlxcel_core::concatenate(&zeros, &committed, 1)
                        }
                    };
                    let plen = n_keep + n;
                    let tail = slice_axis(&padded, 1, plen - n_keep, plen);
                    *state = Some(mlxcel_core::contiguous(&tail, false));
                }
            }
        }
    }

    /// Whether the DSpark burst may engage at `block_size` verify rows on
    /// this host: the block-versus-chain probe, run at most once per
    /// (model, width) per process through
    /// [`mtp_exactness_gate`], which also owns the decline log line, the
    /// `qmv_wide` retry and the `MLXCEL_MTP_ALLOW_INEXACT` override.
    ///
    /// Used by: the server DFlash burst gate.
    pub fn dflash_exactness_allows(&self, block_size: usize) -> bool {
        let key = ProbeKey {
            block_size: block_size as u32,
            hidden_size: self.config.hidden_size as u32,
            num_hidden_layers: self.config.num_hidden_layers as u32,
        };
        mtp_exactness_gate(key, || self.probe_block_chain_exactness(block_size))
    }

    /// Measure whether a `block_size`-row verify block produces logits
    /// byte-identical to `block_size` single-token decode steps from the
    /// same state, over [`PROBE_DRAWS`] synthetic inputs. Both arms run on
    /// fresh caches; no per-sequence state is touched.
    ///
    /// The single-token arm is the shape classic decode runs, so it is the
    /// contract's reference; on a MoE checkpoint it also takes the fused
    /// single-row expert kernel that the block arm cannot.
    pub fn probe_block_chain_exactness(&self, block_size: usize) -> BlockChainExactness {
        if block_size < 2 {
            return BlockChainExactness::NotRun("block width below 2 drafts nothing");
        }
        let vocab = self.config.vocab_size;
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

    fn probe_one_draw(&self, block_size: usize, vocab: usize, draw: usize) -> BlockChainExactness {
        let salt = draw * 977 + 1;
        let wrap =
            |i: usize, stride: usize, offset: usize| ((i * stride + offset + salt) % vocab) as i32;
        let prompt: Vec<i32> = (0..PROBE_PROMPT_LEN).map(|i| wrap(i, 7, 1)).collect();
        let block: Vec<i32> = (0..block_size).map(|i| wrap(i, 13, 3)).collect();

        let as_input =
            |tokens: &[i32]| mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let position_bytes = |logits: &MlxArray, index: i32| -> Vec<u8> {
            let shape = mlxcel_core::array_shape(logits);
            let row = mlxcel_core::slice(logits, &[0, index, 0], &[shape[0], index + 1, shape[2]]);
            mlxcel_core::array_to_raw_bytes(&row)
        };

        let mut chain_caches = self.make_caches();
        let _ = self.forward_speculative(&as_input(&prompt), &mut chain_caches, &[]);
        let mut chain_positions: Vec<Vec<u8>> = Vec::with_capacity(block_size);
        for token in &block {
            let out = self.forward_speculative(&as_input(&[*token]), &mut chain_caches, &[]);
            chain_positions.push(position_bytes(&out.logits, 0));
        }

        let mut block_caches = self.make_caches();
        let _ = self.forward_speculative(&as_input(&prompt), &mut block_caches, &[]);
        let out = self.forward_speculative(&as_input(&block), &mut block_caches, &[]);
        let block_positions: Vec<Vec<u8>> = (0..block_size)
            .map(|i| position_bytes(&out.logits, i as i32))
            .collect();

        compare_block_against_chain(&block_positions, &chain_positions)
    }
}

/// DFlash target adapter for LFM2 / LFM2.5 text models. The LFM2-VL
/// wrapper delegates to the same methods through its `text_model`.
impl SpeculativeTarget for Lfm2Model {
    type Cache = Lfm2LayerCache;
    type VerifyOut = VerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        // The drafter owns the capture list (`DFlashConfig::target_layer_ids`);
        // the round loop passes it through `verify_forward_with_capture_layers`.
        &[]
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.forward_speculative(verify_input, caches, &[])
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        self.forward_speculative(verify_input, caches, capture_layer_ids)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.rollback_speculative_cache(caches, &verify_out.conv_states, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        debug_assert!(
            !verify_out.hidden_states.is_empty(),
            "DSpark verify output must carry at least one captured hidden layer"
        );
        let slabs: Vec<&MlxArray> = verify_out
            .hidden_states
            .iter()
            .map(|slab| slab.as_ref().expect("captured hidden must be non-null"))
            .collect();
        concatenate_many(&slabs, -1)
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        verify_out
            .logits
            .as_ref()
            .expect("DSpark verify output must carry logits")
    }
}

/// Test seams for `lfm2_tests` (the captured slab must be the residual
/// stream before `embedding_norm`, which only the private norm and tied
/// head can demonstrate).
#[cfg(test)]
impl Lfm2Model {
    pub(crate) fn embedding_norm_for_test(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        self.embedding_norm.forward(h)
    }

    pub(crate) fn tied_projection_for_test(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_tokens.as_linear(h)
    }
}
