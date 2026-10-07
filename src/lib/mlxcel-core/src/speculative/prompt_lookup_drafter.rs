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

//! Prompt lookup as a [`Drafter`] (#2176): the n-gram lookup of
//! [`super::prompt_lookup`] behind the trait the server's MTP and DFlash
//! bursts drive, so one engine round loop
//! ([`crate::engine::DirectEngine::generate_with_drafter`]) serves it and
//! the server can offer it through the same trait. It needs no checkpoint
//! and no hidden states: its context is the prompt plus every emitted token,
//! which the round loop reports through [`Drafter::accept_verified_tokens`].
//!
//! The adaptive governor and the shadow probe of the original loop live here
//! too, so the policy that decides how much to propose travels with the
//! drafter rather than with the loop.

use super::prompt_lookup::{
    DraftGovernor, NgramIndex, PromptLookupConfig, PromptLookupStats, SHADOW_CONFIRM, ShadowProbe,
};
use crate::MlxArray;
use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::weights::WeightMap;

/// The prompt-lookup drafter. One instance serves one completion; a second
/// run through the same instance starts from the prompt the loop reports to
/// [`Drafter::prefill_from_target_hidden`].
#[derive(Debug, Clone)]
pub struct PromptLookupDrafter {
    config: PromptLookupConfig,
    shadow_config: PromptLookupConfig,
    /// Prompt followed by every emitted token. The lookup searches this.
    context: Vec<i32>,
    index: NgramIndex,
    governor: DraftGovernor,
    /// A proposal looked up while a `DraftPolicy::Gated` governor is paused,
    /// waiting for decoding to show whether it comes true.
    shadow: Option<ShadowProbe>,
    /// Whether the last `draft_block` returned nothing because the governor
    /// paused proposals (or the budget left room for none), rather than
    /// because the lookup found no match.
    last_paused: bool,
    stats: PromptLookupStats,
}

impl PromptLookupDrafter {
    /// A drafter for `config` (validated by the caller through
    /// [`PromptLookupConfig::validate`]).
    #[must_use]
    pub fn new(config: PromptLookupConfig) -> Self {
        Self {
            config,
            shadow_config: PromptLookupConfig {
                max_draft: SHADOW_CONFIRM,
                ..config
            },
            context: Vec::new(),
            index: NgramIndex::new(&config),
            governor: DraftGovernor::new(&config),
            shadow: None,
            last_paused: false,
            stats: PromptLookupStats::default(),
        }
    }

    /// The configuration this drafter proposes under.
    #[must_use]
    pub fn config(&self) -> &PromptLookupConfig {
        &self.config
    }

    /// Acceptance accounting for the current (or most recent) run.
    #[must_use]
    pub fn stats(&self) -> PromptLookupStats {
        self.stats
    }

    fn start(&mut self, prompt_tokens: &[i32], first_token: i32) {
        self.context.clear();
        self.context.extend_from_slice(prompt_tokens);
        self.context.push(first_token);
        self.index = NgramIndex::new(&self.config);
        self.index.extend(&self.context);
        self.governor = DraftGovernor::new(&self.config);
        self.shadow = None;
        self.last_paused = false;
        self.stats = PromptLookupStats::default();
    }
}

impl Drafter for PromptLookupDrafter {
    fn bind(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }

    fn drafts_from_tokens_only(&self) -> bool {
        true
    }

    fn prefill_from_target_hidden(
        &mut self,
        prompt_tokens: &[i32],
        _hidden: &MlxArray,
        first_bonus: i32,
        _sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.start(prompt_tokens, first_bonus);
        Ok(())
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        debug_assert_eq!(
            self.context.last().copied(),
            Some(last_bonus),
            "the loop reports every emitted token before it asks for a proposal"
        );
        if let Some(confirmed) = self
            .shadow
            .as_ref()
            .and_then(|probe| probe.settle(&self.context))
        {
            self.shadow = None;
            if confirmed {
                self.governor.shadow_confirmed();
                self.stats.shadow_confirmations += 1;
            }
        }
        let budget = self.governor.budget().min(block_size);
        self.stats.rounds += 1;
        if budget == 0 {
            // Paused: look up anyway and let the next tokens decoding emits
            // say whether proposing would have paid, at the cost of a
            // host-side hash probe instead of a verify forward.
            self.last_paused = true;
            self.stats.paused_rounds += 1;
            if self.shadow.is_none() && self.governor.probes_while_paused() {
                self.index.extend(&self.context);
                let proposal = self.index.find(&self.context, &self.shadow_config);
                if proposal.len() >= SHADOW_CONFIRM {
                    self.shadow = Some(ShadowProbe {
                        start: self.context.len(),
                        proposal,
                    });
                }
            }
            return Ok(Vec::new());
        }
        self.last_paused = false;
        self.index.extend(&self.context);
        let mut draft = self.index.find(&self.context, &self.config);
        draft.truncate(budget);
        if !draft.is_empty() {
            self.stats.drafted_rounds += 1;
            self.stats.proposed_draft_tokens += draft.len();
        }
        Ok(draft)
    }

    fn accept_verified_tokens(
        &mut self,
        _verify_hidden: &MlxArray,
        draft_tokens: &[i32],
        accepted: usize,
        new_tokens: &[i32],
        _sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.context.extend_from_slice(new_tokens);
        if !draft_tokens.is_empty() {
            self.governor.record(accepted);
            self.stats.accepted_draft_tokens += accepted;
        }
        Ok(())
    }

    fn reset(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.context.clear();
        self.shadow = None;
        self.last_paused = false;
        Ok(())
    }

    fn configured_block_size(&self) -> Option<usize> {
        Some(self.config.max_draft)
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::PromptLookup
    }
}
