// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! B=1 MTP target adapter for Inkling's KV + short-convolution decoder.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::sampling::{LogprobsConfig, TokenLogprobData};
use mlxcel_core::speculative::mtp::target::{
    MtpTarget, MtpVerifyOutput, TreeVerifyUnsupported, VerifyCaptured, VerifyForwardOutput,
};
use mlxcel_core::{MlxArray, UniquePtr};

use super::gemma4_mtp_target::Gemma4MtpTargetAdapter;
use super::inkling::InklingModel;

pub struct InklingMtpTargetAdapter<'a> {
    model: &'a InklingModel,
    seq_id: Option<SequenceId>,
    prefill_start_offset: usize,
}

impl<'a> InklingMtpTargetAdapter<'a> {
    pub fn new(model: &'a InklingModel, seq_id: Option<SequenceId>) -> Self {
        Self {
            model,
            seq_id,
            prefill_start_offset: 0,
        }
    }

    #[must_use]
    pub fn with_prefill_start_offset(mut self, offset: usize) -> Self {
        self.prefill_start_offset = offset;
        self
    }

    fn capture(
        hidden: UniquePtr<MlxArray>,
        verify_input: UniquePtr<MlxArray>,
        snapshot_tensors: Vec<UniquePtr<MlxArray>>,
        scalars: Vec<i32>,
    ) -> VerifyCaptured {
        let mut tensors = Vec::with_capacity(snapshot_tensors.len() + 2);
        tensors.push(hidden);
        tensors.push(verify_input);
        tensors.extend(snapshot_tensors);
        VerifyCaptured { tensors, scalars }
    }
}

impl MtpTarget for InklingMtpTargetAdapter<'_> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &LogprobsConfig,
    ) -> (i32, MtpVerifyOutput, Option<TokenLogprobData>) {
        let offset = self.prefill_start_offset.min(prompt_tokens.len());
        let suffix = &prompt_tokens[offset..];
        let input = mlxcel_core::from_slice_i32(suffix, &[1, suffix.len() as i32]);
        let (logits, hidden_pre) = self
            .model
            .forward_prefill_with_hidden_for_sequence(&input, self.seq_id);
        let shape = mlxcel_core::array_shape(&logits);
        let last_logits = mlxcel_core::slice(
            &logits,
            &[0, shape[1] - 1, 0],
            &[shape[0], shape[1], shape[2]],
        );
        let (token, adjusted_logits) =
            mlxcel_core::sampling::sample_token_optimized(&last_logits, sampler, token_history);
        mlxcel_core::eval(&token);
        let first_bonus = mlxcel_core::item_i32(&token);
        let logprob =
            mlxcel_core::sampling::compute_logprobs(&adjusted_logits, first_bonus, logprobs_config);
        let next_hidden = Gemma4MtpTargetAdapter::last_position_hidden(&hidden_pre);
        let verify_hidden_full = (offset == 0).then_some(hidden_pre);
        let kv_offset = self
            .model
            .speculative_cache_offset_for_sequence(self.seq_id)
            .max(0) as usize;
        (
            first_bonus,
            MtpVerifyOutput {
                next_hidden,
                next_shared_kv: Vec::new(),
                kv_offset,
                bonus_position: kv_offset.saturating_sub(1),
                verify_hidden_full,
            },
            logprob,
        )
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        let input = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        self.model
            .embed_tokens(&input)
            .expect("Inkling exposes its token embedding")
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &LogprobsConfig,
    ) -> VerifyForwardOutput {
        let input = mlxcel_core::from_slice_i32(verify_input, &[1, verify_input.len() as i32]);
        let (logits, hidden_pre, snapshot_tensors, snapshot_scalars) = self
            .model
            .forward_speculative_for_sequence(&input, self.seq_id);
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
            captured: Self::capture(hidden_pre, input, snapshot_tensors, snapshot_scalars),
        }
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        assert!(
            accepted < block_size,
            "accepted prefix exceeds Inkling verify block"
        );
        let VerifyCaptured { tensors, scalars } = captured;
        let mut tensors = tensors.into_iter();
        let hidden_pre = tensors
            .next()
            .expect("Inkling capture is missing hidden state");
        let verify_input = tensors
            .next()
            .expect("Inkling capture is missing verify input");
        let snapshot_tensors = tensors.collect();
        let shape = mlxcel_core::array_shape(&verify_input);
        assert!(
            accepted < shape[1] as usize,
            "accepted prefix exceeds verify input"
        );
        let replay = mlxcel_core::slice(&verify_input, &[0, 0], &[shape[0], accepted as i32 + 1]);
        self.model
            .restore_and_replay_speculative_for_sequence(
                &replay,
                self.seq_id,
                snapshot_tensors,
                &scalars,
            )
            .expect("Inkling MTP snapshot must restore exactly");
        let next_hidden = Gemma4MtpTargetAdapter::hidden_at_position(&hidden_pre, accepted);
        let kv_offset = self
            .model
            .speculative_cache_offset_for_sequence(self.seq_id)
            .max(0) as usize;
        MtpVerifyOutput {
            next_hidden,
            next_shared_kv: Vec::new(),
            kv_offset,
            bonus_position: kv_offset.saturating_sub(1),
            verify_hidden_full: Some(hidden_pre),
        }
    }

    fn verify_forward_tree(
        &self,
        _tree: &mlxcel_core::speculative::mtp::tree::DraftTree,
        _sampler: &SamplingConfig,
        _logprobs_config: &LogprobsConfig,
    ) -> Result<VerifyForwardOutput, TreeVerifyUnsupported> {
        Err(TreeVerifyUnsupported {
            reason: "Inkling short-convolution state is linear and has no tree-aware fork",
        })
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.model.eos_token_ids()
    }
}

/// MTP target adapter for an Inkling HMLP VLM checkpoint.
///
/// The server admits this adapter only for text-only requests. Image-bearing
/// requests remain on the classic VLM prefill path, where
/// [`crate::vision::InklingVlModel::prepare_input_embeddings`] builds HMLP
/// features and the wrapper's prepared-embedding `LanguageModel` entry points
/// feed them to the decoder without applying input normalization twice. Once
/// prefill is complete, speculative decode is entirely a text-backbone
/// operation, so every MTP hook delegates to `vlm.text` through the same
/// adapter used by a standalone Inkling target.
pub struct InklingVLMtpTargetAdapter<'a> {
    inner: InklingMtpTargetAdapter<'a>,
}

impl<'a> InklingVLMtpTargetAdapter<'a> {
    pub fn new(vlm: &'a crate::vision::InklingVlModel, seq_id: Option<SequenceId>) -> Self {
        Self {
            inner: InklingMtpTargetAdapter::new(&vlm.text, seq_id),
        }
    }

    #[must_use]
    pub fn with_prefill_start_offset(mut self, offset: usize) -> Self {
        self.inner = self.inner.with_prefill_start_offset(offset);
        self
    }
}

impl MtpTarget for InklingVLMtpTargetAdapter<'_> {
    fn prefill_and_seed(
        &self,
        prompt_tokens: &[i32],
        sampler: &SamplingConfig,
        token_history: &[i32],
        logprobs_config: &LogprobsConfig,
    ) -> (i32, MtpVerifyOutput, Option<TokenLogprobData>) {
        self.inner
            .prefill_and_seed(prompt_tokens, sampler, token_history, logprobs_config)
    }

    fn embed_token(&self, token_id: i32) -> UniquePtr<MlxArray> {
        self.inner.embed_token(token_id)
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        sampler: &SamplingConfig,
        logprobs_config: &LogprobsConfig,
    ) -> VerifyForwardOutput {
        self.inner
            .verify_forward(verify_input, sampler, logprobs_config)
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        block_size: usize,
        captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        self.inner.verify_finalize(accepted, block_size, captured)
    }

    fn verify_forward_tree(
        &self,
        tree: &mlxcel_core::speculative::mtp::tree::DraftTree,
        sampler: &SamplingConfig,
        logprobs_config: &LogprobsConfig,
    ) -> Result<VerifyForwardOutput, TreeVerifyUnsupported> {
        self.inner
            .verify_forward_tree(tree, sampler, logprobs_config)
    }

    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.inner.eos_token_ids()
    }
}
