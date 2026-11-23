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

use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

use super::InklingModel;
use crate::models::speculative_exactness::{
    BlockChainExactness, ProbeKey, compare_block_against_chain, mtp_exactness_gate,
};

/// Prompt length of one exactness-probe draw (matches the Qwen 3.5, Gemma 4
/// and GLM-4.7-Flash probes).
const PROBE_PROMPT_LEN: usize = 8;
/// Independent synthetic inputs per probe; any single divergence declines.
const PROBE_DRAWS: usize = 3;
/// Consecutive verify blocks walked per draw, so the block arm's KV and
/// four-convolution states are fed by its own `M = bs` rows before the later
/// blocks are compared, as in an engaged session.
const PROBE_BLOCKS_PER_DRAW: usize = 2;

impl InklingModel {
    pub(crate) fn forward_prefill_with_hidden_for_sequence(
        &self,
        input: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                let embeddings = self.embed_tokens.forward(input);
                let hidden_pre = self.pre_norm_hidden_embeddings_with_caches(&embeddings, caches);
                let hidden_post = self.norm.forward(&hidden_pre);
                (self.project_hidden(&hidden_post), hidden_pre)
            },
        )
    }

    /// Forward a B=1 verify block after capturing every layer's exact KV and
    /// four-convolution pre-forward state.
    pub(crate) fn forward_speculative_for_sequence(
        &self,
        input: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        Vec<UniquePtr<MlxArray>>,
        Vec<i32>,
    ) {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                let mut snapshot_tensors = Vec::new();
                let mut snapshot_scalars = Vec::with_capacity(caches.len() * 2);
                for cache in caches.iter() {
                    cache.capture_flat(&mut snapshot_tensors, &mut snapshot_scalars);
                }
                let embeddings = self.embed_tokens.forward(input);
                let hidden_pre = self.pre_norm_hidden_embeddings_with_caches(&embeddings, caches);
                let hidden_post = self.norm.forward(&hidden_pre);
                let logits = self.project_hidden(&hidden_post);
                (logits, hidden_pre, snapshot_tensors, snapshot_scalars)
            },
        )
    }

    /// Restore the pre-verify state and replay the accepted prefix including
    /// its bonus token; tail trimming is insufficient for recurrent conv state.
    pub(crate) fn restore_and_replay_speculative_for_sequence(
        &self,
        replay_input: &MlxArray,
        seq_id: Option<SequenceId>,
        snapshot_tensors: Vec<UniquePtr<MlxArray>>,
        snapshot_scalars: &[i32],
    ) -> Result<(), String> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                if snapshot_scalars.len() != caches.len() * 2 {
                    return Err(format!(
                        "Inkling snapshot has {} scalars for {} layers",
                        snapshot_scalars.len(),
                        caches.len()
                    ));
                }
                let mut tensors = snapshot_tensors.into_iter();
                let mut scalar_index = 0;
                for cache in caches.iter_mut() {
                    cache.restore_flat(&mut tensors, snapshot_scalars, &mut scalar_index)?;
                }
                if tensors.next().is_some() {
                    return Err("Inkling snapshot contains trailing tensors".into());
                }
                let embeddings = self.embed_tokens.forward(replay_input);
                let _ = self.pre_norm_hidden_embeddings_with_caches(&embeddings, caches);
                Ok(())
            },
        )
    }

    pub(crate) fn speculative_cache_offset_for_sequence(&self, seq_id: Option<SequenceId>) -> i32 {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| caches.first().map_or(0, |cache| cache.kv.offset),
        )
    }

    /// Whether native MTP may engage on this checkpoint at `block_size`.
    ///
    /// Inkling has no Metal-only kernel prerequisite, so the condition is the
    /// measured [`Self::probe_block_chain_exactness`], memoized per (model,
    /// block width) through
    /// [`mtp_exactness_gate`](crate::models::speculative_exactness::mtp_exactness_gate),
    /// which also owns the decline logging, the `qmv_wide` retry and the
    /// `MLXCEL_MTP_ALLOW_INEXACT` override. Before this gate the family was
    /// admitted unconditionally, and on an M3 Ultra (Apple GPU generation 15)
    /// `mlx-community/Inkling-Small-mlx-4bit` with `--draft-block-size 2` or
    /// `3` emitted a 128-token greedy stream that differed from classic decode
    /// under the default `qmv_wide` dispatch, while `MLXCEL_QMV_WIDE=0` made
    /// both widths byte-identical (issue #1549).
    ///
    /// Used by: `speculative_burst::mtp_capable_target`, the offline
    /// `run_offline_mtp` gate.
    pub fn mtp_exactness_allows(&self, block_size: usize) -> bool {
        let text = &self.config.text_config;
        let key = ProbeKey {
            block_size: block_size as u32,
            hidden_size: text.hidden_size as u32,
            num_hidden_layers: text.num_hidden_layers as u32,
        };
        mtp_exactness_gate(key, || self.probe_block_chain_exactness(block_size))
    }

    /// Measure, on this loaded checkpoint, whether `T = block_size` verify
    /// blocks produce byte-identical logits to the same tokens decoded one at
    /// a time.
    ///
    /// The chain arm is classic decode's [`Self::forward_with_caches`] at
    /// `T = 1`; the block arm runs the same function over each block, which
    /// is the computation [`Self::forward_speculative_for_sequence`] performs
    /// for a verify round. Both arms use fresh internal caches and `&self`,
    /// so no per-sequence server state is touched.
    ///
    /// Used by: [`Self::mtp_exactness_allows`].
    pub fn probe_block_chain_exactness(&self, block_size: usize) -> BlockChainExactness {
        if block_size < 2 {
            return BlockChainExactness::NotRun("block width below 2 drafts nothing");
        }
        let text = &self.config.text_config;
        let vocab = text.unpadded_vocab_size.unwrap_or(text.vocab_size);
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

    /// One (prompt, walk) draw of [`Self::probe_block_chain_exactness`].
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

        let mut chain_caches = self.make_internal_caches();
        let _ = self.forward_with_caches(&as_input(&prompt), &mut chain_caches);
        let mut chain_positions = Vec::with_capacity(walk.len());
        for token in &walk {
            let logits = self.forward_with_caches(&as_input(&[*token]), &mut chain_caches);
            chain_positions.push(position_bytes(&logits, 0));
        }

        let mut block_caches = self.make_internal_caches();
        let _ = self.forward_with_caches(&as_input(&prompt), &mut block_caches);
        let mut block_positions = Vec::with_capacity(walk.len());
        for block in walk.chunks(block_size) {
            let logits = self.forward_with_caches(&as_input(block), &mut block_caches);
            for i in 0..block.len() {
                block_positions.push(position_bytes(&logits, i as i32));
            }
        }

        compare_block_against_chain(&block_positions, &chain_positions)
    }
}
