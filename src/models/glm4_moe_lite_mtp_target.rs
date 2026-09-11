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

//! MTP target adapter for GLM4 MoE Lite (GLM-4.7-Flash), issue #1326.
//!
//! Wires [`Glm4MoeLiteModel`] to the
//! [`mlxcel_core::speculative::mtp::target::MtpTarget`] trait, the same role
//! [`crate::models::qwen3_5_mtp_target`] plays for Qwen 3.5.
//!
//! ## Cache ownership
//!
//! `MtpTarget`'s methods are `&self` with no cache parameter. The classic
//! GLM path keeps its caches in the scheduler's `CachePool` (the model is a
//! `DenseKvCache` family), and the server's default MTP path is the
//! tick-cooperative slice that rebuilds this adapter every tick, so the
//! adapter cannot own the caches either. The model therefore carries a
//! separate model-owned per-sequence slot used only by MTP
//! (`Glm4MoeLiteModel::mtp_sequence_state`), and every call here routes
//! through the model's `*_for_sequence` hooks. `seq_id = None` selects the
//! internal fallback slot (the offline CLI).
//!
//! ## Adopted prompt-cache prefix
//!
//! An adopted prefix lives in the pool caches, not in the MTP slot, so it
//! cannot be reused here. The scheduler and the burst decline MTP for this
//! family when `prefill_start_offset > 0` (`mtp_adopted_prefix_reusable`);
//! should an adapter still be built with an offset, `prefill_and_seed`
//! forwards the whole prompt on a fresh slot, which is correct and merely
//! forfeits the reuse.
//!
//! ## Hidden-state tap
//!
//! The drafter consumes the target's post-final-norm hidden by default (the
//! DeepSeek-V3 MTP convention: the nextn block normalizes it again through
//! `hnorm`). The forwards capture the pre-norm residual, so every hidden
//! handed out goes through `apply_final_norm` first.
//! `MLXCEL_GLM_MTP_HIDDEN_TAP=pre` hands out the pre-norm residual instead,
//! the switch the real-checkpoint acceptance measurement is made with.
//!
//! ## Scope
//!
//! B = 1 only. The batched and tree `MtpTarget` methods keep their erroring
//! trait defaults, so a B > 1 window declines to classic decode.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::{LogprobsConfig, TokenLogprobData};
use mlxcel_core::speculative::mtp::target::{
    MtpTarget, MtpVerifyOutput, VerifyCaptured, VerifyForwardOutput,
};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::gemma4_mtp_target::Gemma4MtpTargetAdapter;
use crate::models::glm4_moe_lite::Glm4MoeLiteModel;

/// Which target hidden the drafter is fed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenTap {
    /// The output of `model.norm` (default).
    PostFinalNorm,
    /// The residual stream before `model.norm`.
    PreFinalNorm,
}

impl HiddenTap {
    /// `MLXCEL_GLM_MTP_HIDDEN_TAP`: `pre` selects the pre-norm residual,
    /// anything else (or unset) the post-norm default. Read once per process.
    pub fn from_env() -> Self {
        static TAP: std::sync::OnceLock<HiddenTap> = std::sync::OnceLock::new();
        *TAP.get_or_init(|| {
            match std::env::var("MLXCEL_GLM_MTP_HIDDEN_TAP")
                .map(|v| v.trim().to_ascii_lowercase())
                .as_deref()
            {
                Ok("pre") | Ok("pre_norm") | Ok("pre-norm") => HiddenTap::PreFinalNorm,
                _ => HiddenTap::PostFinalNorm,
            }
        })
    }
}

/// MTP target adapter binding a [`Glm4MoeLiteModel`] to a per-sequence MTP
/// cache slot. Holds no tensor state and does not own the model.
pub struct Glm4MoeLiteMtpTargetAdapter<'a> {
    model: &'a Glm4MoeLiteModel,
    seq_id: Option<SequenceId>,
    prefill_start_offset: usize,
    tap: HiddenTap,
}

impl<'a> Glm4MoeLiteMtpTargetAdapter<'a> {
    /// Construct an adapter routing every call through the MTP slot at
    /// `seq_id` (`None` = the internal fallback slot, the offline CLI shape).
    pub fn new(model: &'a Glm4MoeLiteModel, seq_id: Option<SequenceId>) -> Self {
        Self {
            model,
            seq_id,
            prefill_start_offset: 0,
            tap: HiddenTap::from_env(),
        }
    }

    /// Record the adopted prompt-cache prefix length. This family cannot
    /// reuse it (see the module docs); the dispatch sites decline before
    /// building the adapter, and the field only keeps the constructor shape
    /// the other families share.
    #[must_use]
    pub fn with_prefill_start_offset(mut self, prefill_start_offset: usize) -> Self {
        self.prefill_start_offset = prefill_start_offset;
        self
    }

    /// Override the hidden tap (tests and the acceptance measurement).
    #[must_use]
    pub fn with_hidden_tap(mut self, tap: HiddenTap) -> Self {
        self.tap = tap;
        self
    }

    /// The hidden handed to the drafter for a captured pre-norm block.
    fn tap_hidden(&self, hidden_pre: &MlxArray) -> UniquePtr<MlxArray> {
        match self.tap {
            HiddenTap::PostFinalNorm => self.model.apply_final_norm(hidden_pre),
            HiddenTap::PreFinalNorm => mlxcel_core::copy(hidden_pre),
        }
    }
}

impl<'a> MtpTarget for Glm4MoeLiteMtpTargetAdapter<'a> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &LogprobsConfig,
    ) -> (i32, MtpVerifyOutput, Option<TokenLogprobData>) {
        // Always a fresh slot: the adopted prefix, if any, is in the pool
        // caches and not reachable from here, so the whole prompt is
        // forwarded (correct; the reuse is what is forfeited).
        if self.prefill_start_offset > 0 {
            tracing::debug!(
                prefill_start_offset = self.prefill_start_offset,
                "glm4_moe_lite MTP: adopted prompt-cache prefix is not reusable by the MTP \
                 slot; forwarding the whole prompt on a fresh cache set"
            );
        }
        self.model.reset_mtp_sequence_state(self.seq_id);
        let prompt_arr =
            mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);

        // Classic batched-causal prefill, byte-identical to the non-MTP path.
        let (logits, hidden_pre) = self
            .model
            .forward_prefill_with_last_hidden_for_sequence(&prompt_arr, self.seq_id);

        // First bonus from the last-position logits with the penalty history
        // and the sampler's token bias, as the classic first token is drawn.
        let logits_shape = mlxcel_core::array_shape(&logits);
        let last_pos = logits_shape[1] - 1;
        let vocab = logits_shape[2];
        let last_logits = mlxcel_core::slice(
            &logits,
            &[0, last_pos, 0],
            &[logits_shape[0], last_pos + 1, vocab],
        );
        let (token_arr, adjusted_logits) =
            mlxcel_core::sampling::sample_token_optimized(&last_logits, sampler, token_history);
        mlxcel_core::eval(&token_arr);
        let first_bonus = mlxcel_core::item_i32(&token_arr);
        let first_bonus_lp =
            mlxcel_core::sampling::compute_logprobs(&adjusted_logits, first_bonus, logprobs_config);

        let hidden_tap = self.tap_hidden(&hidden_pre);
        let next_hidden = Gemma4MtpTargetAdapter::last_position_hidden(&hidden_tap);

        let kv_offset = self
            .model
            .speculative_cache_offset_for_sequence(self.seq_id)
            .max(0) as usize;
        let seed = MtpVerifyOutput {
            next_hidden,
            next_shared_kv: Vec::new(),
            kv_offset,
            bonus_position: kv_offset.saturating_sub(1),
            // The whole prompt was forwarded, so the full block always feeds
            // the stateful drafter's prompt prefill.
            verify_hidden_full: Some(hidden_tap),
        };
        (first_bonus, seed, first_bonus_lp)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        let input_ids = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        <Glm4MoeLiteModel as mlxcel_core::generate::LanguageModel>::embed_tokens(
            self.model, &input_ids,
        )
        .expect("Glm4MoeLiteModel exposes its embed_tokens table")
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &LogprobsConfig,
    ) -> VerifyForwardOutput {
        let verify_arr = mlxcel_core::from_slice_i32(verify_input, &[1, verify_input.len() as i32]);
        let (logits, hidden_pre) = self
            .model
            .forward_verify_for_sequence(&verify_arr, self.seq_id);

        // Output-suppression bias before the per-position argmax, as the
        // other adapters do; an empty map short-circuits to the raw logits.
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
            captured: VerifyCaptured {
                tensors: vec![self.tap_hidden(&hidden_pre)],
                scalars: Vec::new(),
            },
        }
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        let mut tensors = captured.tensors.into_iter();
        let hidden = tensors
            .next()
            .expect("VerifyCaptured must carry the verify hidden at index 0");

        let next_hidden = Gemma4MtpTargetAdapter::hidden_at_position(&hidden, accepted);

        // Trim every layer cache back to the accepted prefix: the block
        // appended `block_size` entries and `accepted + 1` of them stay.
        if accepted + 1 < block_size {
            let rejected = (block_size - (accepted + 1)) as i32;
            let _ = self
                .model
                .rollback_speculative_cache_for_sequence(self.seq_id, rejected);
        }

        let kv_offset = self
            .model
            .speculative_cache_offset_for_sequence(self.seq_id)
            .max(0) as usize;

        MtpVerifyOutput {
            next_hidden,
            next_shared_kv: Vec::new(),
            kv_offset,
            bonus_position: kv_offset.saturating_sub(1),
            verify_hidden_full: Some(hidden),
        }
    }

    fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        <Glm4MoeLiteModel as mlxcel_core::generate::LanguageModel>::eos_token_ids(self.model)
    }
}

#[cfg(test)]
#[path = "glm4_moe_lite_mtp_target_tests.rs"]
mod tests;
