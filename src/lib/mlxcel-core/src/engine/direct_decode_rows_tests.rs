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

//! The raw-completion client's per-row lookahead pipeline against its
//! synchronous loop (#2229): penalty, DRY, mirostat and adaptive-p requests
//! emit the synchronous stream token for token, leave the same sequence
//! state at every kind of finish, and are routed to the pipeline only when
//! the fused pipeline turns them away.

use std::cell::RefCell;

use super::direct_decode::Teardown;
use super::direct_decode_tests::{CountModel, OwnedModel, Trace, host_tokens, trace};
use super::*;
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::sampling::LogprobsConfig;
use crate::sampling_row_step::RowSampler;
use crate::{ffi, from_slice_i32};

const NOISY_VOCAB: i32 = 16;

/// A dense-cache model whose next-token logits are a fixed hash of the input
/// token, spread over `[0, 3)` with small gaps, so a history penalty, DRY or
/// a stale history changes the draw.
#[derive(Default)]
struct NoisyModel {
    forwards: RefCell<usize>,
}

impl LanguageModel for NoisyModel {
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
        *self.forwards.borrow_mut() += 1;
        let tokens = host_tokens(input_ids);
        let mut logits = Vec::with_capacity(tokens.len() * NOISY_VOCAB as usize);
        for tok in tokens {
            for v in 0..NOISY_VOCAB {
                let h = (tok * 131 + v * 71 + 17).rem_euclid(23);
                logits.push(h as f32 * 3.0 / 23.0);
            }
        }
        ffi::from_slice_f32(&logits, &[shape[0], shape[1], NOISY_VOCAB])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![NOISY_VOCAB - 1]
    }
}

fn seeded(seed: u64) -> SamplingConfig {
    SamplingConfig {
        temperature: 0.8,
        seed: Some(seed),
        ..SamplingConfig::default()
    }
}

/// Every sampler family the fused pipeline turns away, greedy where the
/// stage allows it and seeded otherwise.
fn row_configs() -> Vec<(&'static str, SamplingConfig)> {
    vec![
        (
            "greedy repetition penalty",
            SamplingConfig {
                repetition_penalty: 1.3,
                ..SamplingConfig::greedy()
            },
        ),
        (
            "seeded frequency and presence penalties",
            SamplingConfig {
                frequency_penalty: 0.4,
                presence_penalty: 0.3,
                ..seeded(42)
            },
        ),
        (
            "greedy DRY",
            SamplingConfig {
                dry_multiplier: 0.8,
                dry_allowed_length: 1,
                ..SamplingConfig::greedy()
            },
        ),
        (
            "seeded DRY with a windowed repetition penalty",
            SamplingConfig {
                dry_multiplier: 0.8,
                repetition_penalty: 1.15,
                penalty_last_n: 8,
                ..seeded(7)
            },
        ),
        (
            "seeded mirostat v2",
            SamplingConfig {
                temperature: 1.0,
                mirostat: 2,
                mirostat_tau: 0.5,
                mirostat_eta: 1.0,
                seed: Some(5),
                ..SamplingConfig::default()
            },
        ),
        (
            "seeded adaptive-p",
            SamplingConfig {
                temperature: 1.0,
                adaptive_target: 0.4,
                adaptive_decay: 0.9,
                seed: Some(9),
                ..SamplingConfig::default()
            },
        ),
    ]
}

/// The teardown the client's `decode` picks for the per-row pipeline on a
/// fresh sequence, `None` when it does not take it.
fn row_teardown<M: LanguageModel>(
    client: &mut DirectEngine<M>,
    sampling: &SamplingConfig,
) -> Option<Teardown> {
    let id = client.open_sequence().unwrap();
    let teardown = client.decode_row_lookahead(id, sampling);
    client.close_sequence(id);
    teardown
}

/// Pipelined and synchronous runs of one request on fresh clients over
/// `make()`'s model, with each run's forward count.
fn both<M: LanguageModel>(
    make: impl Fn() -> M,
    forwards: impl Fn(&M) -> usize,
    prompt: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    deliveries: Option<usize>,
) -> ((Trace, usize), (Trace, usize)) {
    let run = |force_sync: bool| {
        let model = make();
        let mut client = DirectEngine::new(&model, 0).with_force_sync(force_sync);
        let expected = (!force_sync).then_some(Teardown::Unwind);
        assert_eq!(row_teardown(&mut client, sampling), expected);
        let trace = trace(&mut client, prompt, max_tokens, sampling, deliveries);
        (trace, forwards(&model))
    };
    (run(false), run(true))
}

fn count_both(
    prompt: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    deliveries: Option<usize>,
) -> ((Trace, usize), (Trace, usize)) {
    both(
        CountModel::default,
        CountModel::forwards,
        prompt,
        max_tokens,
        sampling,
        deliveries,
    )
}

/// The path assertion: a penalty request decodes on the per-row pipeline
/// (one speculative forward past the EOS, unwound), not on the synchronous
/// loop, with the synchronous stream and state.
#[test]
fn a_penalty_request_takes_the_per_row_pipeline() {
    let penalty = SamplingConfig {
        repetition_penalty: 1.3,
        ..SamplingConfig::greedy()
    };
    let ((piped, piped_fwd), (sync, sync_fwd)) = count_both(&[2, 3], 16, &penalty, None);
    assert_eq!(piped, sync);
    assert_eq!(sync.tokens, vec![4, 5, 6]);
    assert_eq!((sync.kv_len, sync.offset), (Some(5), 5));
    assert_eq!(sync_fwd, 4);
    assert_eq!(piped_fwd, 5, "the forward fed by the EOS was submitted");
}

/// Every per-row sampler family: an EOS finish, a budget finish (no forward
/// after the token that spends it) and callback stops match the synchronous
/// loop exactly on the counting model.
#[test]
fn eos_length_and_callback_finishes_match_the_synchronous_loop() {
    for (name, sampling) in row_configs() {
        let ((piped, piped_fwd), (sync, sync_fwd)) = count_both(&[2, 3], 16, &sampling, None);
        assert_eq!(piped, sync, "{name}: EOS");
        assert!(sync.tokens.len() < 16, "{name}: finished on the EOS");
        assert_eq!(piped_fwd, sync_fwd + 1, "{name}: one unwound forward");

        // The EOS is banned so the budget ends every run (adaptive-p
        // otherwise picks it early).
        let mut no_eos = sampling.clone();
        no_eos
            .token_bias
            .insert(super::direct_decode_tests::EOS, f32::NEG_INFINITY);
        for max_tokens in [1, 2, 3] {
            let ((piped, piped_fwd), (sync, sync_fwd)) =
                count_both(&[0], max_tokens, &no_eos, None);
            assert_eq!(piped, sync, "{name}: max_tokens {max_tokens}");
            assert_eq!(sync.tokens.len(), max_tokens, "{name}: spent the budget");
            assert_eq!(piped_fwd, sync_fwd, "{name}: max_tokens {max_tokens}");
        }

        for deliveries in [1, 2] {
            let ((piped, _), (sync, _)) = count_both(&[0], 16, &sampling, Some(deliveries));
            assert_eq!(piped, sync, "{name}: stop after {deliveries}");
            assert_eq!(sync.delivered.len(), deliveries, "{name}");
        }
    }
}

/// On a model whose draws the penalties actually move, every per-row sampler
/// family emits the synchronous stream token for token over long runs.
#[test]
fn penalized_streams_match_the_synchronous_loop_token_for_token() {
    for (name, sampling) in row_configs() {
        for prompt in [&[1, 4][..], &[9][..], &[3, 3, 12][..]] {
            let ((piped, _), (sync, _)) = both(
                NoisyModel::default,
                |m: &NoisyModel| *m.forwards.borrow(),
                prompt,
                48,
                &sampling,
                None,
            );
            assert_eq!(piped, sync, "{name}: prompt {prompt:?}");
        }
    }
    // The noisy model is sensitive: the penalty changes the greedy stream.
    let greedy = SamplingConfig::greedy();
    let penalty = &row_configs()[0].1;
    let model = NoisyModel::default();
    let mut client = DirectEngine::new(&model, 0).with_force_sync(true);
    let plain = client.run(&[1, 4], 48, &greedy).unwrap();
    let penalized = client.run(&[1, 4], 48, penalty).unwrap();
    assert_ne!(plain, penalized);
}

/// The per-row pipeline is the fallback of the fused one only: a fused
/// request keeps the fused pipeline, and `MLXCEL_FORCE_SYNC` keeps every
/// request synchronous.
#[test]
fn the_per_row_pipeline_takes_only_what_the_fused_one_turns_away() {
    let model = CountModel::default();
    let mut client = DirectEngine::new(&model, 0).with_force_sync(false);
    assert_eq!(row_teardown(&mut client, &SamplingConfig::greedy()), None);
    assert_eq!(row_teardown(&mut client, &seeded(1)), None);
    for (name, sampling) in row_configs() {
        assert_eq!(
            row_teardown(&mut client, &sampling),
            Some(Teardown::Unwind),
            "{name}"
        );
    }
    let mut forced = DirectEngine::new(&model, 0).with_force_sync(true);
    for (name, sampling) in row_configs() {
        assert_eq!(row_teardown(&mut forced, &sampling), None, "{name}");
    }
}

/// Model-owned families follow the fused pipeline's teardown rules: an exact
/// unwind when the model rewinds its own state, a discarded append released
/// with the sequence when it cannot.
#[test]
fn model_owned_families_pipeline_penalty_requests_with_the_same_teardowns() {
    let penalty = SamplingConfig {
        repetition_penalty: 1.3,
        ..SamplingConfig::greedy()
    };
    let run = |rewinds: bool, force_sync: bool| {
        let model = OwnedModel::new(rewinds);
        let mut client = DirectEngine::new(&model, 0).with_force_sync(force_sync);
        let teardown = row_teardown(&mut client, &penalty);
        let tokens = client.run(&[2, 3], 16, &penalty).unwrap();
        let released = model.released.borrow().last().copied();
        (teardown, tokens, released, model.forwards.get())
    };
    let (teardown, sync_tokens, sync_len, sync_fwd) = run(true, true);
    assert_eq!(teardown, None);
    assert_eq!(sync_tokens, vec![4, 5, 6]);

    let (teardown, tokens, len, fwd) = run(true, false);
    assert_eq!(teardown, Some(Teardown::Unwind));
    assert_eq!((tokens, len), (sync_tokens.clone(), sync_len));
    assert_eq!(fwd, sync_fwd + 1);

    let (teardown, tokens, len, fwd) = run(false, false);
    assert_eq!(teardown, Some(Teardown::Discard));
    assert_eq!(tokens, sync_tokens);
    assert_eq!(fwd, sync_fwd + 1);
    assert_eq!(len, sync_len.map(|l| l + 1));
}

/// `draw_row` confirms the draw at once, so it refuses a row that needs its
/// host token first: a logit mask, a token override or a logprobs payload.
#[test]
fn draw_row_rejects_rows_that_need_the_host_token() {
    let sampling = SamplingConfig::greedy();
    let mut engine = Engine::with_capacity(CountModel::default(), 1);
    let id = engine.open(SequenceSpec::default()).unwrap();
    let logits = engine
        .submit_forward(&StepBatch {
            seq_ids: &[id],
            input: &from_slice_i32(&[3], &[1, 1]),
        })
        .unwrap();
    let enabled = LogprobsConfig {
        enabled: true,
        ..LogprobsConfig::default()
    };
    let disabled = LogprobsConfig::default();
    for (mask, over, logprobs) in [
        (true, false, &disabled),
        (false, true, &disabled),
        (false, false, &enabled),
        (false, false, &disabled),
    ] {
        let mut sampler = RowSampler::new(&sampling);
        let (mut history, mut generated) = (vec![3], vec![3]);
        let mut row = StepRow {
            seq_id: id,
            sampler: &mut sampler,
            sampling: &sampling,
            token_history: &mut history,
            generated: &mut generated,
            eos: &[super::direct_decode_tests::EOS],
            max_tokens: 8,
            logprobs,
            needs_mask: mask,
            needs_override: over,
            hooks: BareHooks,
        };
        let drawn = engine.draw_row(&logits, &mut row);
        if mask || over || logprobs.enabled {
            assert!(matches!(drawn, Err(EngineError::Batch(_))));
        } else {
            assert_eq!(ffi::item_i32(&drawn.unwrap()), 4);
        }
    }
    engine.close(id);
}
