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
use crate::models::speculative_exactness_diag as diag;
use crate::models::speculative_exactness_draws as draws;

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

    /// Forward a B=1 verify block. A block the row-invariant scope covers
    /// records its own rewind data (see
    /// [`Self::keep_verified_prefix_for_sequence`]) and returns an empty
    /// snapshot; any other block first captures every layer's exact KV and
    /// four-convolution pre-forward state for
    /// [`Self::restore_and_replay_speculative_for_sequence`].
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
                let mut snapshot_scalars = Vec::new();
                if !self.verify_block_rewinds_in_place(input) {
                    for cache in caches.iter() {
                        cache.capture_flat(&mut snapshot_tensors, &mut snapshot_scalars);
                    }
                }
                let _verify = self.verify_block_scope();
                let embeddings = self.embed_tokens.forward(input);
                let hidden_pre = self.pre_norm_hidden_embeddings_with_caches(&embeddings, caches);
                let hidden_post = self.norm.forward(&hidden_pre);
                let logits = self.project_hidden(&hidden_post);
                (logits, hidden_pre, snapshot_tensors, snapshot_scalars)
            },
        )
    }

    /// Whether a verify forward over `input [1, L]` records the convolution
    /// windows [`Self::keep_verified_prefix_for_sequence`] rewinds from, so
    /// no pre-verify snapshot is needed.
    fn verify_block_rewinds_in_place(&self, input: &MlxArray) -> bool {
        let shape = mlxcel_core::array_shape(input);
        self.row_kernel_enabled()
            && shape.len() == 2
            && shape[0] == 1
            && (2..=mlxcel_core::row_invariant::MAX_ROWS).contains(&shape[1])
    }

    /// After a verify block of `appended` rows, keep its first `kept` rows in
    /// place: every layer trims its KV tail and takes its convolution states
    /// from the block's own input windows. A row-invariant verify block
    /// equals the single-token chain, so the result is the chain's state
    /// after those rows; the replay of the accepted prefix it replaces ran
    /// outside the verify scope and was a second target forward per round
    /// (issue #3280). Refused, with nothing changed, unless every layer
    /// recorded its windows.
    pub(crate) fn keep_verified_prefix_for_sequence(
        &self,
        seq_id: Option<SequenceId>,
        appended: usize,
        kept: usize,
    ) -> Result<(), String> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                if !caches.iter().all(|cache| cache.holds_block_tails()) {
                    return Err("an Inkling layer recorded no verify-block windows".into());
                }
                for cache in caches.iter_mut() {
                    cache.keep_verified_prefix(appended as i32, kept as i32)?;
                }
                Ok(())
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
    /// Since issue #3409 the model's dense products, short convolutions,
    /// sliding attention and experts are row-invariant inside a verify block
    /// (`mlxcel_core::row_invariant`), and the probe passes on the bf16
    /// `inkling-0.6b` checkpoint at widths 2, 4 and 8 on an M3 Ultra, also
    /// after prompts past the sliding window. The probe adds a long draw past
    /// that window. `MLXCEL_INKLING_ROW_INVARIANT=0` restores the old
    /// products, under which the probe declines. On `Inkling-Small-mlx-4bit`
    /// the probe passed at widths 2, 3, 4 and 8 but declined at 10, the MTP
    /// default, from layer 0 on: past 8 queries MLX's SDPA leaves its
    /// per-query kernel, so a verify block now attends in groups of at most 8
    /// rows (issue #3280).
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
        let lengths =
            draws::prompt_lengths(PROBE_PROMPT_LEN, PROBE_DRAWS, self.probe_long_context());
        self.probe_block_chain_exactness_at(block_size, &lengths)
    }

    /// Prompt length of the probe's long draw: past the sliding window, where
    /// a verify block's rows and the decode steps stop seeing the same keys
    /// unless the model accounts for it (issue #3409). `None` without a
    /// sliding layer.
    fn probe_long_context(&self) -> Option<usize> {
        let text = &self.config.text_config;
        (0..text.num_hidden_layers)
            .any(|index| text.layer_is_sliding(index))
            .then(|| draws::past_window(text.sliding_window_size))
    }

    /// [`Self::probe_block_chain_exactness`] over draws of the given prompt
    /// lengths, one draw per entry.
    ///
    /// Used by: [`Self::probe_block_chain_exactness`] and the exactness survey test.
    pub fn probe_block_chain_exactness_at(
        &self,
        block_size: usize,
        prompt_lengths: &[usize],
    ) -> BlockChainExactness {
        if block_size < 2 {
            return BlockChainExactness::NotRun("block width below 2 drafts nothing");
        }
        let text = &self.config.text_config;
        let vocab = text.unpadded_vocab_size.unwrap_or(text.vocab_size);
        if vocab < 2 {
            return BlockChainExactness::NotRun("degenerate vocabulary");
        }
        draws::run_draws(prompt_lengths, |draw, prompt_len| {
            self.probe_one_draw(block_size, vocab, draw, prompt_len)
        })
    }

    /// One (prompt, walk) draw of [`Self::probe_block_chain_exactness`]. A
    /// diverging draw is run again under [`diag`] capture, and the verdict
    /// then names the first layer whose output differs.
    fn probe_one_draw(
        &self,
        block_size: usize,
        vocab: usize,
        draw: usize,
        prompt_len: usize,
    ) -> BlockChainExactness {
        let verdict = self
            .probe_arms(block_size, vocab, draw, prompt_len, false)
            .0;
        if verdict.is_equal() || matches!(verdict, BlockChainExactness::NotRun(_)) {
            return verdict;
        }
        match self.probe_arms(block_size, vocab, draw, prompt_len, true) {
            (_, Some(location)) => BlockChainExactness::Localized {
                verdict: Box::new(verdict),
                location,
            },
            _ => verdict,
        }
    }

    /// Both arms of one draw. With `trace`, also where the first block
    /// parts from its chain steps, when it does.
    fn probe_arms(
        &self,
        block_size: usize,
        vocab: usize,
        draw: usize,
        prompt_len: usize,
        trace: bool,
    ) -> (BlockChainExactness, Option<String>) {
        let prompt = draws::prompt_tokens(draw, prompt_len, vocab);
        let walk = draws::block_tokens(draw, block_size * PROBE_BLOCKS_PER_DRAW, vocab);
        let traced = |run: &mut dyn FnMut() -> UniquePtr<MlxArray>| {
            if trace {
                diag::with_capture(run)
            } else {
                (run(), Vec::new())
            }
        };

        let mut chain_caches = self.make_internal_caches();
        let _ = self.forward_with_caches(&draws::as_input(&prompt), &mut chain_caches);
        let mut chain_positions = Vec::with_capacity(walk.len());
        let mut chain_trace = Vec::with_capacity(walk.len());
        for token in &walk {
            let (logits, stages) = traced(&mut || {
                self.forward_with_caches(&draws::as_input(&[*token]), &mut chain_caches)
            });
            chain_positions.push(draws::position_bytes(&logits, 0));
            chain_trace.push(stages);
        }

        let mut block_caches = self.make_internal_caches();
        let _ = self.forward_with_caches(&draws::as_input(&prompt), &mut block_caches);
        let mut block_positions = Vec::with_capacity(walk.len());
        let mut location = None;
        for (index, block) in walk.chunks(block_size).enumerate() {
            let (logits, stages) = traced(&mut || {
                // The scope the verify forward enters.
                let _verify = self.verify_block_scope();
                self.forward_with_caches(&draws::as_input(block), &mut block_caches)
            });
            for i in 0..block.len() {
                block_positions.push(draws::position_bytes(&logits, i as i32));
            }
            if trace && location.is_none() {
                let start = index * block_size;
                location =
                    diag::first_divergence(&stages, &chain_trace[start..start + block.len()]);
            }
        }

        (
            compare_block_against_chain(&block_positions, &chain_positions),
            location,
        )
    }
}
