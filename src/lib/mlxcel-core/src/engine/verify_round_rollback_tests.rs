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

//! The state a verify round leaves behind in the all-reject, all-accept and
//! mixed regimes, on the stubs of [`super::speculative_distribution_tests`].

use super::direct_decode::DecodeState;
use super::speculative_distribution_tests::{
    ContextFreeModel, GREEDY_TARGET, Proposals, REJECTING_DRAFT, ScriptedDrafter, stochastic_config,
};
use super::*;
use crate::cache::SequenceId;
use crate::ffi;
use crate::generate::{LanguageModel, SamplingConfig};

/// Prefills `prompt` and returns the sequence with the first token pending.
fn open_sequence<M: LanguageModel>(engine: &mut Engine<M>, prompt: &[i32]) -> SequenceId {
    let id = engine.open(SequenceSpec::default()).unwrap();
    let input = ffi::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let logits = engine.verify(id, &input).unwrap();
    ffi::eval(&logits);
    engine.commit_appends(id, prompt.len() as i32);
    id
}

/// After every unfinished round the state holds the prompt plus every emitted
/// token except the pending current one, on every layer.
fn assert_state_is_exact<M: LanguageModel>(
    engine: &Engine<M>,
    id: SequenceId,
    prompt_len: usize,
    generated: usize,
    context: &str,
) {
    let want = (prompt_len + generated - 1) as i32;
    let set = engine.pool().get(id).unwrap();
    assert_eq!(set.current_offset, want, "{context}: pool offset");
    for (layer, cache) in set.caches.iter().enumerate() {
        assert_eq!(cache.seq_len(), want, "{context}: cache layer {layer}");
    }
}

fn run_rounds<M: LanguageModel>(
    model: M,
    drafter: &mut ScriptedDrafter,
    block: usize,
    config: &SamplingConfig,
    context: &str,
) -> usize {
    const PROMPT: [i32; 1] = [2];
    const N: usize = 40;
    let mut engine = Engine::with_capacity(model, 1);
    let id = open_sequence(&mut engine, &PROMPT);
    let mut state = DecodeState::new(id, &PROMPT, config, Vec::new(), N);
    state.generated.push(2);
    let mut accepted = 0;
    loop {
        let current = *state.generated.last().unwrap();
        let budget = block.min(N.saturating_sub(state.generated.len()).saturating_sub(1));
        if budget == 0 {
            return accepted;
        }
        let draft = drafter.propose(current, budget);
        let round = engine
            .verify_round(&mut state.row(), current, &draft, |_| true)
            .unwrap();
        assert!(round.error().is_none(), "{context}");
        accepted += round.accepted;
        assert_state_is_exact(&engine, id, PROMPT.len(), state.generated.len(), context);
    }
}

/// The rewind arithmetic in the all-reject and all-accept regimes.
#[test]
fn kv_cache_length_is_exact_in_both_acceptance_regimes() {
    let greedy = SamplingConfig::greedy();
    for block in [1usize, 3, 5] {
        let mut rejecting = ScriptedDrafter(Proposals::Orbit(REJECTING_DRAFT));
        let accepted = run_rounds(GREEDY_TARGET, &mut rejecting, block, &greedy, "all-reject");
        assert_eq!(accepted, 0);
        let mut agreeing = ScriptedDrafter(Proposals::Orbit(GREEDY_TARGET));
        let accepted = run_rounds(GREEDY_TARGET, &mut agreeing, block, &greedy, "all-accept");
        assert!(accepted > 0);
    }
}

/// Mixed sampled accept/reject rounds keep the same arithmetic, on every layer.
#[test]
fn kv_cache_stays_consistent_across_mixed_accept_reject_rounds() {
    let config = stochastic_config();
    let mut drafter = ScriptedDrafter::sampled(0xC0DE_CAFE_F00D_BEEF);
    let mut accepted = 0;
    for block in [1usize, 3, 5] {
        for _ in 0..8 {
            accepted += run_rounds(&ContextFreeModel, &mut drafter, block, &config, "mixed");
        }
    }
    assert!(accepted > 0, "the mixed regime must accept proposals");
}
