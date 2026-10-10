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

//! The shared verify round (#2255): what it commits and unwinds on an
//! accepted, a rejected and a failed block, and the eligibility predicate's
//! model check.

use super::direct_decode::DecodeState;
use super::*;
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::speculative::prompt_lookup::PromptLookupConfig;
use crate::speculative::prompt_lookup_drafter::PromptLookupDrafter;
use crate::utils::array_to_vec_f32;
use crate::{MlxArray, UniquePtr, ffi};

const VOCAB: usize = 8;

/// Greedy continuation of `t` is `(t + 1) % 7`; `short_logits` returns one
/// position for a multi-token input, `padded_prefill` is what it reports.
struct Cycle {
    short_logits: bool,
    padded_prefill: bool,
}

impl LanguageModel for Cycle {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(input_ids);
        for cache in caches.iter_mut() {
            let k = ffi::zeros(&[1, 1, shape[1], 4], crate::dtype::FLOAT32);
            let v = ffi::zeros(&[1, 1, shape[1], 4], crate::dtype::FLOAT32);
            cache.update(k, v);
        }
        ffi::eval(input_ids);
        let tokens: Vec<i32> = array_to_vec_f32(&ffi::astype(input_ids, crate::dtype::FLOAT32))
            .into_iter()
            .map(|t| t as i32)
            .collect();
        let mut logits = vec![0.0f32; tokens.len() * VOCAB];
        for (i, tok) in tokens.into_iter().enumerate() {
            logits[i * VOCAB + (tok + 1).rem_euclid(7) as usize] = 10.0;
        }
        if self.short_logits && shape[1] > 1 {
            return ffi::from_slice_f32(&logits[..VOCAB], &[1, 1, VOCAB as i32]);
        }
        ffi::from_slice_f32(&logits, &[shape[0], shape[1], VOCAB as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![7]
    }

    fn supports_padded_prefill(&self) -> bool {
        self.padded_prefill
    }
}

/// A sequence holding `prompt` plus `current` as its last emitted token
/// (not yet in the state), ready for a round.
fn sequence(engine: &mut Engine<&Cycle>, prompt: &[i32]) -> SequenceId {
    let id = engine.open(SequenceSpec::default()).unwrap();
    let input = ffi::from_slice_i32(prompt, &[1, prompt.len() as i32]);
    let logits = engine.verify(id, &input).unwrap();
    ffi::eval(&logits);
    engine.commit_appends(id, prompt.len() as i32);
    id
}

fn kv_len(engine: &Engine<&Cycle>, id: SequenceId) -> (i32, i32) {
    let set = engine.pool().get(id).unwrap();
    (set.current_offset, set.caches[0].seq_len())
}

/// An accepted prefix and the target's own token are emitted; the state
/// keeps the current token and the accepted proposals, as the equivalent
/// one-token steps would.
#[test]
fn a_round_commits_the_accepted_prefix_and_unwinds_the_rest() {
    let model = Cycle {
        short_logits: false,
        padded_prefill: true,
    };
    let mut engine = Engine::with_capacity(&model, 2);
    let id = sequence(&mut engine, &[0, 1]);
    let greedy = SamplingConfig::greedy();
    let mut state = DecodeState::new(id, &[0, 1], &greedy, vec![7], 16);
    state.generated.push(2);
    // Target after 2 is 3 (accepted), after 3 is 4 (rejects 6).
    let round = engine
        .verify_round(&mut state.row(), 2, &[3, 6, 0], |_| true)
        .unwrap();
    assert_eq!(round.accepted, 1);
    assert!(round.error().is_none() && !round.ended());
    assert_eq!(state.generated, vec![2, 3, 4]);
    // Prompt (2) + current (2) + accepted (3) are in the state; 4 is the
    // next round's current token.
    assert_eq!(kv_len(&engine, id), (4, 4));
}

/// A failed position unwinds every append of the round and reports the
/// error on the last outcome (I7): the state is the pre-round state.
#[test]
fn a_failed_round_unwinds_every_append() {
    let model = Cycle {
        short_logits: true,
        padded_prefill: true,
    };
    let mut engine = Engine::with_capacity(&model, 2);
    let id = sequence(&mut engine, &[0]);
    let before = kv_len(&engine, id);
    let greedy = SamplingConfig::greedy();
    let mut state = DecodeState::new(id, &[0], &greedy, vec![7], 16);
    state.generated.push(1);
    let round = engine
        .verify_round(&mut state.row(), 1, &[2, 3], |_| true)
        .unwrap();
    assert!(matches!(round.error(), Some(RowError::Eval(_))));
    assert!(round.ended());
    assert_eq!(kv_len(&engine, id), before, "every append unwound");
}

/// The eligibility predicate refuses a model whose state a trim cannot roll
/// back, after the sampler, drafter and trim checks.
#[test]
fn the_predicate_refuses_a_model_a_trim_cannot_roll_back() {
    let drafter = PromptLookupDrafter::new(PromptLookupConfig::default());
    let greedy = SamplingConfig::greedy();
    let model = Cycle {
        short_logits: false,
        padded_prefill: false,
    };
    let mut engine = Engine::with_capacity(&model, 1);
    let id = engine.open(SequenceSpec::default()).unwrap();
    assert!(matches!(
        engine.verify_rounds_unsupported(id, &greedy, &drafter),
        Some(SpeculativeRunError::ModelUnsupported(_))
    ));
    let mirostat = SamplingConfig {
        mirostat: 2,
        ..SamplingConfig::with_temperature(0.8)
    };
    assert_eq!(
        engine.verify_rounds_unsupported(id, &mirostat, &drafter),
        Some(SpeculativeRunError::SamplerFeedbackState)
    );
    let fine = Cycle {
        short_logits: false,
        padded_prefill: true,
    };
    let mut engine = Engine::with_capacity(&fine, 1);
    let id = engine.open(SequenceSpec::default()).unwrap();
    assert_eq!(
        engine.verify_rounds_unsupported(id, &greedy, &drafter),
        None
    );
}
