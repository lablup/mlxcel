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

//! DFlash [`SpeculativeTarget`] for the Laguna family (#1351).
//!
//! The verify forward is [`LagunaModel::forward_with_capture`] with the
//! drafter's `target_layer_ids`; rollback trims every layer cache by the
//! rejected tail. Sliding layers run a [`RotatingKVCache`] with
//! `block_size` positions of speculative slack
//! (`enable_speculative_buffer`), so a verify block append followed by its
//! rollback stays inside the temporal region and never wraps the ring.
//! [`LagunaWrapper`] forwards the same hooks so one object can serve both as
//! the `LanguageModel` the drafter binds to and as the target the round loop
//! drives.
//!
//! [`RotatingKVCache`]: mlxcel_core::layers::RotatingKVCache

use crate::models::laguna::{LagunaModel, LagunaWrapper};
use crate::models::laguna_layers::LagunaCache;
use crate::models::speculative_exactness::{
    BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate,
};
use mlxcel_core::drafter::dflash::SpeculativeTarget;
use mlxcel_core::{MlxArray, UniquePtr};

/// Independent synthetic inputs the exactness probe runs before it reports
/// equality; see `models::qwen3_5::PROBE_DRAWS` for why one is not enough.
const PROBE_DRAWS: usize = 3;
/// Prompt length of each probe draw.
const PROBE_PROMPT_LEN: usize = 8;

/// Output of one Laguna verify forward.
pub struct LagunaVerifyOutput {
    /// `[1, block_size, vocab]` logits.
    pub logits: UniquePtr<MlxArray>,
    /// Post-block residual stream of every captured layer, in
    /// `capture_layer_ids` order, each `[1, block_size, hidden]`.
    pub hidden_states: Vec<UniquePtr<MlxArray>>,
}

impl LagunaModel {
    /// Fresh per-layer caches for a speculative run. Rotating caches take
    /// `block_size` positions of rollback slack so a partial accept can trim
    /// the verify tail without the window having wrapped over it.
    pub fn make_speculative_caches(&self, block_size: usize) -> Vec<LagunaCache> {
        let mut caches = self.make_caches();
        self.enable_speculative_buffers(&mut caches, block_size);
        caches
    }

    /// Arm the rotating caches in `caches` with `block_size` rows of
    /// speculative slack. Dense caches are untouched.
    pub fn enable_speculative_buffers(&self, caches: &mut [LagunaCache], block_size: usize) {
        // The Muse Glimmer slack rule: `block_size` alone makes the buffered
        // cache compact on every round once the window is full, and eight
        // blocks of slack amortize that.
        let slack = crate::models::muse_glimmer::speculative_buffer_size(block_size);
        for (layer_idx, cache) in caches.iter_mut().enumerate() {
            if let LagunaCache::Rotating(rotating) = cache
                && let Err(reason) = rotating.enable_speculative_buffer(slack)
            {
                tracing::warn!(
                    layer_idx,
                    slack,
                    "Laguna sliding cache could not take a speculative buffer: {reason}; a \
                     verify block that crosses the window boundary will not be rewindable on \
                     this layer"
                );
            }
        }
    }

    /// Whether a `block_size`-row verify block is byte-identical to
    /// `block_size` single-token decode steps on this host, memoized per
    /// (model, width) through [`mtp_exactness_gate`], which also owns the
    /// decline log line, the `qmv_wide` retry and the
    /// `MLXCEL_MTP_ALLOW_INEXACT` override.
    ///
    /// The same measured gate LFM2 and Muse Glimmer run. On a GB10 with the
    /// NVFP4 checkpoint the `M >= 2` and `M = 1` quantized kernels round
    /// differently at bf16 logit ties (one or two positions per 128 tokens
    /// measured, every one at a chain-arm top-2 margin of 0.0 or 0.125), so
    /// the probe declines there and the operator opts in with the override.
    ///
    /// Used by: the server DFlash burst gate and the offline
    /// `mlxcel generate --draft-kind dflash` arm.
    pub fn dflash_exactness_allows(&self, block_size: usize) -> bool {
        let key = ProbeKey {
            block_size: block_size as u32,
            hidden_size: self.hidden_size() as u32,
            num_hidden_layers: self.layers.len() as u32,
        };
        mtp_exactness_gate(key, || self.probe_block_chain_exactness(block_size))
    }

    fn hidden_size(&self) -> usize {
        mlxcel_core::array_shape(self.embed_tokens.weight())[1] as usize
    }

    fn vocab_size(&self) -> usize {
        mlxcel_core::array_shape(self.embed_tokens.weight())[0] as usize
    }

    /// Measure whether a `block_size`-row verify block produces logits
    /// byte-identical to `block_size` single-token decode steps from the
    /// same state, over [`PROBE_DRAWS`] synthetic inputs. Both arms run on
    /// fresh caches; the block arm's rotating caches carry the same
    /// speculative slack the burst arms them with.
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

        // Chain arm: the classic decode shape (plain caches, one token per
        // forward).
        let mut chain_caches = self.make_caches();
        let _ = self.forward_with_caches(&as_input(&prompt), &mut chain_caches);
        let mut chain_positions: Vec<Vec<u8>> = Vec::with_capacity(block_size);
        for token in &block {
            let logits = self.forward_with_caches(&as_input(&[*token]), &mut chain_caches);
            chain_positions.push(position_bytes(&logits, 0));
        }

        // Block arm: the speculative shape (buffered rotating caches, one
        // `block_size`-row verify forward).
        let mut block_caches = self.make_speculative_caches(block_size);
        let _ = self.forward_speculative(&as_input(&prompt), &mut block_caches, &[]);
        let out = self.forward_speculative(&as_input(&block), &mut block_caches, &[]);
        let block_positions: Vec<Vec<u8>> = (0..block_size)
            .map(|i| position_bytes(&out.logits, i as i32))
            .collect();

        compare_block_against_chain(&block_positions, &chain_positions)
    }

    /// Verify forward with hidden capture. `capture_layer_ids` are 0-based
    /// target layers whose post-block residual stream is returned.
    pub fn forward_speculative(
        &self,
        input_ids: &MlxArray,
        caches: &mut [LagunaCache],
        capture_layer_ids: &[usize],
    ) -> LagunaVerifyOutput {
        let (logits, hidden_states) =
            self.forward_with_capture(input_ids, caches, capture_layer_ids);
        LagunaVerifyOutput {
            logits,
            hidden_states,
        }
    }

    /// Rewind every layer cache to the accepted prefix: the verify block
    /// appended `block_size` positions, `accepted + 1` of them stay. Returns
    /// the number of positions trimmed.
    pub fn rollback_speculative_cache(
        &self,
        caches: &mut [LagunaCache],
        accepted: i32,
        block_size: i32,
    ) -> i32 {
        let trim = block_size - (accepted + 1);
        if trim <= 0 {
            return 0;
        }
        for (layer_idx, cache) in caches.iter_mut().enumerate() {
            let trimmed = cache.trim(trim);
            debug_assert_eq!(trimmed, trim, "every Laguna cache must trim the full tail");
            if trimmed != trim {
                // A short rewind leaves the layer holding rejected rows, so
                // the continuation is no longer the committed prefix; name it
                // rather than let the stream drift silently in release builds.
                tracing::error!(
                    layer_idx,
                    requested = trim,
                    trimmed,
                    "Laguna speculative rollback trimmed fewer positions than the rejected \
                     tail; the sliding cache was armed with too little speculative slack"
                );
            }
        }
        trim
    }
}

impl SpeculativeTarget for LagunaModel {
    type Cache = LagunaCache;
    type VerifyOut = LagunaVerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        // The drafter owns the list (`LagunaDFlashConfig::target_layer_ids`);
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
        _verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.rollback_speculative_cache(caches, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        assert!(
            !verify_out.hidden_states.is_empty(),
            "Laguna DFlash verify output must carry at least one captured layer"
        );
        let refs: Vec<&MlxArray> = verify_out
            .hidden_states
            .iter()
            .map(|slab| {
                slab.as_ref()
                    .expect("captured hidden state must be non-null")
            })
            .collect();
        mlxcel_core::concatenate_many(&refs, -1)
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        &verify_out.logits
    }
}

impl SpeculativeTarget for LagunaWrapper {
    type Cache = LagunaCache;
    type VerifyOut = LagunaVerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        self.model.capture_layer_ids()
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.model.verify_forward(verify_input, caches)
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        self.model
            .verify_forward_with_capture_layers(verify_input, caches, capture_layer_ids)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.model
            .rollback_partial(caches, verify_out, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        self.model.concat_hidden_for_drafter(verify_out)
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        self.model.verify_logits(verify_out)
    }
}
