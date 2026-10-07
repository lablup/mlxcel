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

//! The per-row sampling step (#2169): the CLI decode loops and the server's
//! per-row step emit the same token stream for the same logits, the CLI carries
//! feedback state across steps, and the hooks behave. Fused eligibility is
//! covered in `sampling_row_step_fused_tests.rs`.

use super::{LogitMask, RowSampler};
use crate::ffi;
use crate::ffi::MlxArray;
use crate::generate::{LanguageModel, SamplingConfig};
use crate::generation_policy::{initial_token_history, seed_rng_if_needed};
use crate::sampling::{
    AdaptivePState, SamplerState, TokenBiasMap, sample_token_optimized,
    sample_token_optimized_with_state,
};
use crate::session::MlxInferenceSession;
use crate::test_support::induction::{INDUCTION_MODELS, INDUCTION_VOCAB, lcg_tokens};
use cxx::UniquePtr;

const MAX_TOKENS: usize = 64;

/// The four public `MlxInferenceSession` entry points (the CLI surface over the
/// engine client), each on a fresh session; the embeddings variants run without
/// embeddings.
fn cli_streams<M: LanguageModel>(
    model: &M,
    prompt: &[i32],
    sampling: &SamplingConfig,
) -> [(&'static str, Vec<i32>); 4] {
    [
        (
            "generate",
            MlxInferenceSession::new(1).generate(model, prompt, MAX_TOKENS, sampling),
        ),
        (
            "generate_streaming_with_embeddings",
            MlxInferenceSession::new(1).generate_streaming_with_embeddings(
                model,
                prompt,
                None,
                None,
                MAX_TOKENS,
                sampling,
                |_| true,
            ),
        ),
        (
            "generate_with_stats_and_embeddings",
            MlxInferenceSession::new(1)
                .generate_with_stats_and_embeddings(model, prompt, None, None, MAX_TOKENS, sampling)
                .0,
        ),
        (
            "generate_with_stats",
            MlxInferenceSession::new(1)
                .generate_with_stats(model, prompt, MAX_TOKENS, sampling)
                .0,
        ),
    ]
}

/// The server's per-row step, call for call: the row's seed applied right
/// before its first draw (`finish_prefill`), then per token one forward, one
/// [`RowSampler::draw`], the host read, [`RowSampler::resolve`] with no
/// override, and the history push.
fn server_row_stream<M: LanguageModel>(
    model: &M,
    prompt: &[i32],
    sampling: &SamplingConfig,
) -> Vec<i32> {
    let mut caches = model.make_caches();
    let mut sampler = RowSampler::new(sampling);
    let needs_history = sampling.needs_token_history();
    let mut history = initial_token_history(prompt, needs_history);
    let mut input = prompt.to_vec();
    let mut emitted = Vec::with_capacity(MAX_TOKENS);
    seed_rng_if_needed(sampling);
    while emitted.len() < MAX_TOKENS {
        let ids = ffi::from_slice_i32(&input, &[1, input.len() as i32]);
        let logits = model.forward(&ids, &mut caches, None);
        let draw = sampler
            .draw(&logits, sampling, &history, None, false)
            .expect("an unmasked draw cannot fail");
        ffi::eval(&draw.token);
        let (_, token) = sampler.resolve(ffi::item_i32(&draw.token), |t| t);
        emitted.push(token);
        if needs_history {
            history.push(token);
        }
        input = vec![token];
    }
    emitted
}

/// Plain sequential decoding through the chain functions themselves, with one
/// persistent state when `stateful` and a fresh one per step otherwise (what
/// the CLI did for feedback samplers before #2169).
fn chain_reference<M: LanguageModel>(
    model: &M,
    prompt: &[i32],
    sampling: &SamplingConfig,
    stateful: bool,
) -> Vec<i32> {
    let mut caches = model.make_caches();
    let mut state: Option<SamplerState> = None;
    let mut input = prompt.to_vec();
    let mut emitted = Vec::with_capacity(MAX_TOKENS);
    seed_rng_if_needed(sampling);
    while emitted.len() < MAX_TOKENS {
        let ids = ffi::from_slice_i32(&input, &[1, input.len() as i32]);
        let logits = model.forward(&ids, &mut caches, None);
        let (token, _) = if stateful {
            sample_token_optimized_with_state(&logits, sampling, &[], &mut state)
        } else {
            sample_token_optimized(&logits, sampling, &[])
        };
        let token = ffi::item_i32(&token);
        if let Some(state) = state.as_mut() {
            state.accept_token(token);
        }
        emitted.push(token);
        input = vec![token];
    }
    emitted
}

fn seeded(seed: u64) -> SamplingConfig {
    SamplingConfig {
        temperature: 0.8,
        seed: Some(seed),
        ..SamplingConfig::default()
    }
}

fn mirostat_v2() -> SamplingConfig {
    SamplingConfig {
        temperature: 1.0,
        mirostat: 2,
        mirostat_tau: 0.5,
        mirostat_eta: 1.0,
        seed: Some(5),
        ..SamplingConfig::default()
    }
}

fn identity_configs() -> Vec<(&'static str, SamplingConfig)> {
    let mut bias = TokenBiasMap::new();
    bias.insert(2, -1.5);
    bias.insert(3, 0.7);
    bias.insert(5, f32::NEG_INFINITY);
    vec![
        ("greedy", SamplingConfig::greedy()),
        (
            "seeded penalties, full history",
            SamplingConfig {
                repetition_penalty: 1.15,
                frequency_penalty: 0.3,
                presence_penalty: 0.2,
                ..seeded(42)
            },
        ),
        (
            "seeded penalties, windowed",
            SamplingConfig {
                repetition_penalty: 1.15,
                frequency_penalty: 0.3,
                presence_penalty: 0.2,
                penalty_last_n: 8,
                ..seeded(43)
            },
        ),
        (
            "seeded DRY",
            SamplingConfig {
                dry_multiplier: 0.8,
                ..seeded(7)
            },
        ),
        (
            "seeded logit_bias",
            SamplingConfig {
                token_bias: bias,
                ..seeded(11)
            },
        ),
        ("seeded mirostat v2", mirostat_v2()),
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

#[test]
fn cli_loops_and_server_row_step_emit_identical_streams() {
    for (name, sampling) in identity_configs() {
        for model in &INDUCTION_MODELS {
            for prompt_seed in 1..=3 {
                let prompt = lcg_tokens(prompt_seed, 32, INDUCTION_VOCAB as u64);
                let server = server_row_stream(model, &prompt, &sampling);
                for (entry, cli) in cli_streams(model, &prompt, &sampling) {
                    assert_eq!(
                        cli, server,
                        "{name}: {entry} vs the server row step, earliest={}, prompt {prompt_seed}",
                        model.earliest
                    );
                }
            }
        }
    }
}

/// Before #2169 the CLI created `SamplerState` only for history-reading
/// samplers, so mirostat's `mu` restarted at `2 * tau` every token. With a small
/// `tau` that pins every draw to the argmax; carried state lets `mu` grow until
/// the runner-up token is admitted.
#[test]
fn cli_loops_carry_feedback_state_across_steps() {
    let sampling = mirostat_v2();
    assert!(sampling.needs_sampler_feedback_state());
    assert!(!sampling.needs_token_history());
    let mut carried_differs = false;
    for model in &INDUCTION_MODELS {
        for prompt_seed in 1..=3 {
            let prompt = lcg_tokens(prompt_seed, 32, INDUCTION_VOCAB as u64);
            let carried = chain_reference(model, &prompt, &sampling, true);
            let restarted = chain_reference(model, &prompt, &sampling, false);
            carried_differs |= carried != restarted;
            for (entry, cli) in cli_streams(model, &prompt, &sampling) {
                assert_eq!(
                    cli, carried,
                    "{entry} dropped the mirostat state, earliest={}, prompt {prompt_seed}",
                    model.earliest
                );
            }
        }
    }
    assert!(
        carried_differs,
        "carrying mu never changed a token; the test cannot tell the paths apart"
    );
}

#[test]
fn state_is_created_for_history_and_feedback_samplers_only() {
    let cases = [
        ("greedy", SamplingConfig::greedy(), false, false),
        ("plain stochastic", seeded(1), false, false),
        (
            "repetition penalty",
            SamplingConfig {
                repetition_penalty: 1.1,
                ..seeded(1)
            },
            true,
            true,
        ),
        (
            "DRY",
            SamplingConfig {
                dry_multiplier: 0.8,
                ..seeded(1)
            },
            true,
            true,
        ),
        ("mirostat", mirostat_v2(), true, false),
        (
            "adaptive-p",
            SamplingConfig {
                adaptive_target: 0.4,
                ..seeded(1)
            },
            true,
            false,
        ),
    ];
    for (name, config, has_state, host_read) in cases {
        let sampler = RowSampler::new(&config);
        assert_eq!(sampler.state().is_some(), has_state, "{name}: state");
        assert_eq!(RowSampler::needs_state(&config), has_state, "{name}: rule");
        assert_eq!(
            sampler.needs_host_token_before_sample(),
            host_read,
            "{name}: host read before the next draw"
        );
    }
}

fn adaptive_sampler_with_pending(token: i32) -> RowSampler {
    let mut adaptive = AdaptivePState::new(0.4, 0.9);
    adaptive.pending = Some((token, 0.25));
    let mut state = SamplerState::default();
    state.adaptive = Some(adaptive);
    RowSampler {
        state: Some(state),
        needs_history: false,
    }
}

fn adaptive_ema(sampler: &RowSampler) -> (f32, f32) {
    let adaptive = sampler
        .state()
        .and_then(|state| state.adaptive.as_ref())
        .expect("adaptive state");
    (adaptive.weighted_sum, adaptive.total_weight)
}

#[test]
fn resolve_returns_sampled_and_final_and_confirms_only_the_final_token() {
    let mut overridden = adaptive_sampler_with_pending(3);
    let before = adaptive_ema(&overridden);
    // A thinking budget of 0 forces the end-of-think id on the first step.
    assert_eq!(overridden.resolve(3, |_| 9), (3, 9));
    assert_eq!(
        adaptive_ema(&overridden),
        before,
        "an override leaves the EMA untouched"
    );

    let mut kept = adaptive_sampler_with_pending(3);
    assert_eq!(kept.resolve(3, |t| t), (3, 3));
    assert_ne!(
        adaptive_ema(&kept),
        before,
        "the drawn token folds into the EMA"
    );

    let mut drawn = adaptive_sampler_with_pending(3);
    drawn.accept_drawn();
    assert_eq!(
        adaptive_ema(&drawn),
        adaptive_ema(&kept),
        "confirming at draw time equals resolving with no override"
    );
}

/// Bans one token, or refuses the row.
struct TestMask {
    banned: Option<i32>,
}

impl LogitMask for TestMask {
    fn apply(
        &mut self,
        logits: UniquePtr<MlxArray>,
        vocab_size: usize,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let Some(banned) = self.banned else {
            return Err("no token is allowed".to_string());
        };
        let mut row = vec![0.0f32; vocab_size];
        row[banned as usize] = f32::NEG_INFINITY;
        let mask = ffi::from_slice_f32(&row, &[1, 1, vocab_size as i32]);
        Ok(ffi::add(&logits, &mask))
    }
}

#[test]
fn the_mask_runs_before_the_chain_and_its_error_aborts_the_draw() {
    let logits = ffi::from_slice_f32(&[0.0, 5.0, 4.0, 1.0], &[1, 1, 4]);
    let greedy = SamplingConfig::greedy();
    let mut sampler = RowSampler::new(&greedy);

    let draw = sampler
        .draw(&logits, &greedy, &[], None, false)
        .expect("unmasked");
    assert_eq!(ffi::item_i32(&draw.token), 1);

    let mut ban_argmax = TestMask { banned: Some(1) };
    let draw = sampler
        .draw(&logits, &greedy, &[], Some(&mut ban_argmax), true)
        .expect("masked");
    assert_eq!(ffi::item_i32(&draw.token), 2);
    assert!(
        draw.distribution.is_some(),
        "the distribution was requested"
    );

    let mut refuse = TestMask { banned: None };
    let err = sampler
        .draw(&logits, &greedy, &[], Some(&mut refuse), false)
        .err()
        .expect("a refusing mask aborts the draw");
    assert_eq!(err, "no token is allowed");
}
