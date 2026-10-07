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

//! The direct `Engine` arm of the parity harness (#2172): one sequence
//! driven through [`Engine::open`], [`Engine::prefill`], [`Engine::step`] and
//! [`Engine::close`] with no scheduler in between, the way Phase 5 will drive
//! the CLI. It uses the same prefill plan, sampler, finish step and merged
//! EOS set the server's B=1 row uses, so a divergence between this arm and
//! the server's dense B=1 arm is scheduler policy, not engine execution.

use anyhow::{Context, Result, anyhow};
use mlxcel_core::decode_finish::FinishHooks;
use mlxcel_core::engine::{Engine, PrefillStep, SequenceSpec, StepBatch, StepRow, StepRowHooks};
use mlxcel_core::generate::{LanguageModel, SamplingConfig, prefill_tile_alignment_enabled};
use mlxcel_core::generation_policy::{
    initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
use mlxcel_core::prefill_plan::{PrefillCaps, PrefillPlan};
use mlxcel_core::sampling::LogprobsConfig;
use mlxcel_core::sampling_row_step::{LogitMask, RowSampler};

use crate::LoadedModel;

/// Hooks for a bare sequence: no stop strings, no generation or context
/// bound, no structured-output constraint and no thinking budget, which is
/// what a parity request carries.
#[derive(Debug, Clone, Copy, Default)]
struct BareHooks;

impl FinishHooks for BareHooks {
    fn stop_text(&mut self, _token: i32, _generated_len: usize) -> bool {
        false
    }

    fn bound_stopped(&self) -> bool {
        false
    }

    fn context_bound_due(&self, _generated_len: usize) -> bool {
        false
    }
}

impl StepRowHooks for BareHooks {
    fn logit_mask(&mut self) -> Option<&mut dyn LogitMask> {
        None
    }

    fn override_token(&mut self, sampled: i32) -> i32 {
        sampled
    }

    fn consume_sampled(&mut self, _sampled: i32) -> Result<bool, String> {
        Ok(false)
    }
}

/// An [`Engine`] over the loaded model, decoding one request at a time.
pub struct DirectEngine {
    engine: Engine<LoadedModel>,
    prefill_chunk: usize,
}

impl DirectEngine {
    /// Wrap `model`; `prefill_chunk` is the piece size of the prefill plan
    /// (the server's `--prefill-chunk-size`, `0` for one forward).
    pub fn new(model: LoadedModel, prefill_chunk: usize) -> Self {
        Self {
            engine: Engine::with_capacity(model, 1),
            prefill_chunk,
        }
    }

    /// Hand the model back.
    pub fn into_model(self) -> LoadedModel {
        self.engine.into_parts().0
    }

    /// Prefill `prompt_tokens` and decode up to `max_tokens` under
    /// `sampling`, returning the generated tokens.
    pub fn run(
        &mut self,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
    ) -> Result<Vec<i32>> {
        let eos = merged_eos_token_ids(
            self.engine.model().eos_token_ids(),
            &sampling.stop_token_ids,
        );
        let mut history = initial_token_history(prompt_tokens, sampling.needs_token_history());
        let mut generated: Vec<i32> = Vec::new();
        let mut sampler = RowSampler::new(sampling);
        let logprobs = LogprobsConfig::default();
        let id = self
            .engine
            .open(SequenceSpec::default())
            .map_err(|err| anyhow!("open: {err}"))?;
        let result = self.decode(
            id,
            prompt_tokens,
            max_tokens,
            sampling,
            &eos,
            &mut history,
            &mut generated,
            &mut sampler,
            &logprobs,
        );
        self.engine.close(id);
        result.map(|()| generated)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &mut self,
        id: mlxcel_core::cache::SequenceId,
        prompt_tokens: &[i32],
        max_tokens: usize,
        sampling: &SamplingConfig,
        eos: &[i32],
        history: &mut Vec<i32>,
        generated: &mut Vec<i32>,
        sampler: &mut RowSampler,
        logprobs: &LogprobsConfig,
    ) -> Result<()> {
        // The same partition the scheduler builds for a cold prompt.
        let caps = PrefillCaps::for_model(self.engine.model(), prefill_tile_alignment_enabled());
        let plan = PrefillPlan::new(prompt_tokens.len(), self.prefill_chunk, caps);
        let mut last_logits = None;
        for piece in plan.pieces() {
            let tokens = &prompt_tokens[piece.range.clone()];
            // A fresh sequence's KV state holds exactly the earlier pieces.
            let (input, pad_mask) =
                mlxcel_core::engine::piece_input(&plan, piece, tokens, piece.range.start as i32);
            let outcome = self
                .engine
                .prefill(&PrefillStep {
                    seq_id: id,
                    input: &input,
                    embeddings: None,
                    mask: pad_mask.as_deref(),
                    last_pos: piece.last_real_pos(),
                    trim_excess: piece.trim_after().unwrap_or(0) as i32,
                    eval: !plan.is_terminal(piece),
                })
                .map_err(|err| anyhow!("prefill: {err}"))?;
            outcome.eval.map_err(|msg| anyhow!("prefill eval: {msg}"))?;
            outcome.trim.map_err(|msg| anyhow!("prefill trim: {msg}"))?;
            last_logits = Some(outcome.logits);
        }
        let logits = last_logits.context("the prefill plan had no pieces")?;

        seed_rng_if_needed(sampling);
        let first = {
            let mut row = StepRow {
                seq_id: id,
                sampler: &mut *sampler,
                sampling,
                token_history: &mut *history,
                generated: &mut *generated,
                eos,
                max_tokens,
                logprobs,
                needs_mask: false,
                needs_override: false,
                hooks: BareHooks,
            };
            self.engine.complete_prefill(&logits, &mut row)
        };
        if let Some(error) = first.error {
            return Err(anyhow!("first token: {}", error.message()));
        }
        if first.finish.is_some() {
            return Ok(());
        }

        loop {
            let last = *generated.last().context("decode without a token")?;
            let input = mlxcel_core::from_slice_i32(&[last], &[1, 1]);
            let batch = StepBatch {
                seq_ids: std::slice::from_ref(&id),
                input: &input,
            };
            let out = {
                let row = StepRow {
                    seq_id: id,
                    sampler: &mut *sampler,
                    sampling,
                    token_history: &mut *history,
                    generated: &mut *generated,
                    eos,
                    max_tokens,
                    logprobs,
                    needs_mask: false,
                    needs_override: false,
                    hooks: BareHooks,
                };
                self.engine
                    .step(&batch, &mut [row])
                    .map_err(|err| anyhow!("step: {err}"))?
            };
            let outcome = out
                .rows
                .into_iter()
                .next()
                .context("step returned no row")?;
            if let Some(error) = outcome.error {
                return Err(anyhow!("step: {}", error.message()));
            }
            if outcome.finish.is_some() {
                return Ok(());
            }
        }
    }
}
