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

//! MTP target adapter for Qwen3.8-Flash-Next (`qwen4_exp`), issue #2880.
//!
//! Wires [`Qwen4ExpModel`] to the
//! [`mlxcel_core::speculative::mtp::target::MtpTarget`] trait, the role
//! [`crate::models::qwen3_5_mtp_target`] plays for Qwen 3.5. The family
//! loads as `LoadedModel::Qwen4ExpVLM` only, so the one adapter takes the
//! wrapper's `text_model`; multimodal requests never reach it (the burst's
//! multimodal gate declines them to classic decode).
//!
//! ## Cache ownership
//!
//! `MtpTarget` is `&self`-only with no cache parameter. Every call routes
//! through the model's sequence-routed hooks (`*_for_sequence`), so the
//! adapter is a stateless view the tick-cooperative slice path can rebuild
//! every scheduler tick. The verify forward's rollback capture stays on the
//! model's per-sequence rollback log, the same log the engine's prompt-lookup
//! rewind pops; [`VerifyCaptured`] carries only the streams and the block
//! length.
//!
//! ## Hidden-state semantics
//!
//! The native head consumes the **pre-mixer** residual streams,
//! `[1, L, hc_count * hidden]`, captured before the target's
//! `hyper_connection_mixer` (upstream's `capture_layer_ids == []` hidden and
//! `speculative_draft_hidden`). This differs from the Qwen 3.5 adapter,
//! which hands out the post-final-norm `[1, L, hidden]` state; the drafter
//! refuses that shape by name.
//!
//! ## Scope
//!
//! B = 1, text only. The batched and tree methods keep their refusing
//! defaults or refuse by name: a GatedDeltaNet recurrence and a PLE
//! convolution have no attention mask, so a flattened tree would fold state
//! through siblings.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::{LogprobsConfig, TokenLogprobData};
use mlxcel_core::speculative::mtp::target::{
    MtpTarget, MtpVerifyOutput, TreeVerifyUnsupported, VerifyCaptured, VerifyForwardOutput,
};
use mlxcel_core::speculative::mtp::tree::DraftTree;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gemma4_mtp_target::Gemma4MtpTargetAdapter;
use crate::models::qwen4_exp::Qwen4ExpModel;

pub struct Qwen4ExpMtpTargetAdapter<'a> {
    model: &'a Qwen4ExpModel,
    seq_id: Option<SequenceId>,
    /// Adopted prompt-cache prefix length (issue #518): only the suffix is
    /// forwarded, and the seed advertises no full prompt hidden, so the
    /// drafter builds its history from accepted tokens only.
    prefill_start_offset: usize,
    /// Chunk the prompt prefill runs in. The server sets its configured
    /// `prefill_chunk_size`, which the classic decode it must match byte for
    /// byte also uses; the offline CLI keeps the process default.
    prefill_chunk_size: usize,
}

impl<'a> Qwen4ExpMtpTargetAdapter<'a> {
    /// Route every call through the per-sequence slot at `seq_id` (`None` is
    /// the model's internal fallback slot, the offline CLI shape).
    pub fn new(model: &'a Qwen4ExpModel, seq_id: Option<SequenceId>) -> Self {
        Self {
            model,
            seq_id,
            prefill_start_offset: 0,
            prefill_chunk_size: mlxcel_core::generate::prefill_chunk_len(),
        }
    }

    #[must_use]
    pub fn with_prefill_start_offset(mut self, prefill_start_offset: usize) -> Self {
        self.prefill_start_offset = prefill_start_offset;
        self
    }

    /// Run the prompt prefill in `chunk_size` pieces (`0` is one call).
    #[must_use]
    pub fn with_prefill_chunk_size(mut self, chunk_size: usize) -> Self {
        self.prefill_chunk_size = chunk_size;
        self
    }

    fn kv_offset(&self) -> usize {
        self.model
            .speculative_cache_offset_for_sequence(self.seq_id)
            .max(0) as usize
    }
}

/// `VerifyCaptured` layout: `tensors[0]` the pre-mixer streams
/// `[1, block, hc * hidden]`, `scalars[0]` the block length.
fn capture(streams: UniquePtr<MlxArray>, block_len: i32) -> VerifyCaptured {
    VerifyCaptured {
        tensors: vec![streams],
        scalars: vec![block_len],
    }
}

fn uncapture(captured: VerifyCaptured) -> (UniquePtr<MlxArray>, i32) {
    let VerifyCaptured { tensors, scalars } = captured;
    let streams = tensors
        .into_iter()
        .next()
        .expect("VerifyCaptured carries the streams at index 0");
    let block_len = *scalars
        .first()
        .expect("VerifyCaptured carries the block length");
    (streams, block_len)
}

impl<'a> MtpTarget for Qwen4ExpMtpTargetAdapter<'a> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &LogprobsConfig,
    ) -> (i32, MtpVerifyOutput, Option<TokenLogprobData>) {
        let offset = self.prefill_start_offset.min(prompt_tokens.len());
        let forward_tokens = &prompt_tokens[offset..];
        let prompt_arr =
            mlxcel_core::from_slice_i32(forward_tokens, &[1, forward_tokens.len() as i32]);
        let (logits, streams) = self.model.forward_prefill_with_streams_for_sequence(
            &prompt_arr,
            self.seq_id,
            self.prefill_chunk_size,
        );

        let logits_shape = mlxcel_core::array_shape(&logits);
        let last_pos = logits_shape[1] - 1;
        let last_logits = mlxcel_core::slice(
            &logits,
            &[0, last_pos, 0],
            &[logits_shape[0], last_pos + 1, logits_shape[2]],
        );
        let (token_arr, adjusted_logits) =
            mlxcel_core::sampling::sample_token_optimized(&last_logits, sampler, token_history);
        mlxcel_core::eval(&token_arr);
        let first_bonus = mlxcel_core::item_i32(&token_arr);
        let first_bonus_lp =
            mlxcel_core::sampling::compute_logprobs(&adjusted_logits, first_bonus, logprobs_config);

        let next_hidden = Gemma4MtpTargetAdapter::last_position_hidden(&streams);
        let verify_hidden_full = (offset == 0).then_some(streams);
        let kv_offset = self.kv_offset();
        let seed = MtpVerifyOutput {
            next_hidden,
            next_shared_kv: Vec::new(),
            kv_offset,
            bonus_position: kv_offset.saturating_sub(1),
            verify_hidden_full,
        };
        (first_bonus, seed, first_bonus_lp)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        let input_ids = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        self.model.get_embed_tokens(&input_ids)
    }

    fn verify_forward_tree(
        &self,
        _tree: &DraftTree,
        _sampler: &SamplingConfig,
        _logprobs_config: &LogprobsConfig,
    ) -> Result<VerifyForwardOutput, TreeVerifyUnsupported> {
        Err(TreeVerifyUnsupported {
            reason: "Qwen4-Exp is a GatedDeltaNet hybrid with a PLE convolution; neither has an \
                     attention mask, so a flattened tree would fold state through siblings",
        })
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &LogprobsConfig,
    ) -> VerifyForwardOutput {
        let verify_arr = mlxcel_core::from_slice_i32(verify_input, &[1, verify_input.len() as i32]);
        let (logits, streams) = self
            .model
            .forward_speculative_for_sequence(&verify_arr, self.seq_id);
        // The output-suppression bias goes before the argmax and the logprob
        // extraction (issue #350); an empty map is the raw logits.
        let logits = if sampler.token_bias.is_empty() {
            logits
        } else {
            mlxcel_core::sampling::apply_token_bias(&logits, &sampler.token_bias)
        };
        let target_tokens = Gemma4MtpTargetAdapter::argmax_per_position(&logits);
        let target_logprobs =
            Gemma4MtpTargetAdapter::per_position_logprobs(&logits, &target_tokens, logprobs_config);
        VerifyForwardOutput {
            target_tokens,
            target_logprobs,
            captured: capture(streams, verify_input.len() as i32),
        }
    }

    fn verify_finalize_tree(
        &self,
        _path: &[usize],
        _block_size: usize,
        _captured: VerifyCaptured,
    ) -> Result<MtpVerifyOutput, TreeVerifyUnsupported> {
        Err(TreeVerifyUnsupported {
            reason: "Qwen4-Exp rolls its recurrent and convolution state back by replaying the \
                     accepted prefix, and replaying a scattered path is not implemented",
        })
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        let (streams, block_len) = uncapture(captured);
        let next_hidden = Gemma4MtpTargetAdapter::hidden_at_position(&streams, accepted);
        // Keep the bonus plus the accepted drafts; upstream skips the call on
        // a full accept, where the capture is simply committed.
        let keep = (accepted as i32 + 1).min(block_len);
        if keep < block_len {
            if let Err(err) = self
                .model
                .rollback_last_speculative_for_sequence(self.seq_id, keep)
            {
                tracing::error!(
                    error = %err,
                    accepted,
                    block_size,
                    "Qwen4-Exp MTP rollback failed; the sequence state may be ahead of its tokens"
                );
            }
        } else {
            self.model.commit_last_speculative_for_sequence(self.seq_id);
        }
        let kv_offset = self.kv_offset();
        MtpVerifyOutput {
            next_hidden,
            next_shared_kv: Vec::new(),
            kv_offset,
            bonus_position: kv_offset.saturating_sub(1),
            verify_hidden_full: Some(streams),
        }
    }

    fn num_layers(&self) -> usize {
        mlxcel_core::generate::LanguageModel::num_layers(self.model)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        mlxcel_core::generate::LanguageModel::eos_token_ids(self.model)
    }
}
